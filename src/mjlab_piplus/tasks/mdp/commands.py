"""Goal-pose command for the PiPlus go-to-pose task.

Samples a 2D target position (relative to the robot's current base position,
at a random distance/bearing) plus a random target yaw. The command exposed to
the policy is the goal expressed in the robot base frame:

    [dx_b, dy_b, yaw_err]

where ``(dx_b, dy_b)`` is the goal position in the base frame and ``yaw_err`` is
the wrapped yaw difference to the goal orientation. This 3-vector mirrors with
signs ``[1, -1, -1]`` under a left-right reflection, matching the velocity
command layout used by ``symmetry.py`` (so the actor obs layout is unchanged).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, wrap_to_pi

from mjlab_piplus.tasks.mdp.push_gui import PushGui

if TYPE_CHECKING:
  import viser
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


# Ground hits further than this from the camera are treated as misses: a click
# near the horizon is almost never a deliberate goal hundreds of metres away.
_MAX_CLICK_DISTANCE = 50.0


def _ray_ground_hit(
  origin, direction, z_plane: float = 0.0, max_distance: float = _MAX_CLICK_DISTANCE
) -> np.ndarray | None:
  """Where a click ray meets the ground plane, or None if it usefully doesn't.

  viser's frontend builds this ray in the browser from the live camera -- it
  already accounts for camera pose and field of view -- and sends it in world
  coordinates. mjlab adds its scene nodes at top level with MuJoCo poses
  unchanged and no up-direction override, so viser's world frame is MuJoCo's and
  the hit is directly a world xy.

  Returns None when there is no sensible hit: no ray at all, a ray parallel to
  the ground, the ground behind the camera, or a near-horizon click whose
  intersection runs off past ``max_distance``. That last one matters: clicking
  a few pixels above the horizon yields a mathematically valid hit hundreds of
  metres away, which would fling the goal off the map.
  """
  if origin is None or direction is None:
    return None
  origin = np.asarray(origin, dtype=float)
  direction = np.asarray(direction, dtype=float)
  norm = float(np.linalg.norm(direction))
  if norm < 1e-9:
    return None
  direction = direction / norm  # viser does not promise a unit vector.
  if abs(direction[2]) < 1e-6:
    return None  # Looking along the horizon.
  t = (z_plane - origin[2]) / direction[2]
  if t <= 0.0:
    return None  # Ground is behind the camera.
  hit = origin + t * direction
  if float(np.linalg.norm(hit[:2] - origin[:2])) > max_distance:
    return None
  return hit


def _yaw_to_wxyz(yaw: float) -> tuple[float, float, float, float]:
  return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def _wxyz_to_yaw(wxyz) -> float:
  w, x, y, z = (float(v) for v in wxyz)
  return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class GoalPoseCommand(CommandTerm):
  cfg: GoalPoseCommandCfg

  def __init__(self, cfg: GoalPoseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]

    # Goal expressed in the world frame.
    self.goal_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.goal_yaw_w = torch.zeros(self.num_envs, device=self.device)

    # Goal expressed in the robot base frame: [dx_b, dy_b, yaw_err].
    self.goal_command_b = torch.zeros(self.num_envs, 3, device=self.device)

    self.is_standing_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    # The goal is sampled as an offset on resample and anchored on the robot in
    # _update_command. At an episode reset the command manager runs *before*
    # sim.forward(), so the robot pose visible during _resample_command is still
    # the pre-reset one; anchoring there would place the goal relative to
    # wherever the robot used to be. _update_command runs after the forward.
    self._pending_place = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._pending_offset = torch.zeros(self.num_envs, 2, device=self.device)

    # Viewer control. joystick_gate is set by MultiCommand to "is goal_pose the
    # selected type"; see the JoystickTerm protocol in multi_command.py.
    self.joystick_gate: Callable[[], bool] | None = None
    self._gui_get_env_idx: Callable[[], int] | None = None
    self._manual_goal: tuple[float, float] | None = None
    self._manual_yaw: float = 0.0
    # True after the placing click, while a direction is still expected.
    self._awaiting_direction = False

    self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """Network-friendly goal pose: ``[dx_b, dy_b, cos(yaw_err), sin(yaw_err)]``.

    ``yaw_err`` is a wrapping angle, discontinuous at +/-pi and awkward to learn
    against, so the orientation is handed to the policy as a unit vector
    instead. Rewards read ``goal_command_b`` / ``yaw_err`` off this term rather
    than slicing here.
    """
    yaw_err = self.goal_command_b[:, 2]
    return torch.stack(
      [
        self.goal_command_b[:, 0],
        self.goal_command_b[:, 1],
        torch.cos(yaw_err),
        torch.sin(yaw_err),
      ],
      dim=1,
    )

  @property
  def yaw_err(self) -> torch.Tensor:
    """Wrapped yaw error to the goal orientation, ``(num_envs,)``."""
    return self.goal_command_b[:, 2]

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_pos"] += (
      torch.norm(self.goal_command_b[:, :2], dim=1) / max_command_step
    )
    self.metrics["error_yaw"] += torch.abs(self.goal_command_b[:, 2]) / max_command_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)

    # Only the offset from the robot is decided here, in the robot's heading
    # frame; _update_command rotates and anchors it once the pose is current.
    radius = torch.empty(n, device=self.device).uniform_(*self.cfg.radius_range)
    bearing = torch.empty(n, device=self.device).uniform_(*self.cfg.bearing_range)
    self._pending_offset[env_ids] = torch.stack(
      [radius * torch.cos(bearing), radius * torch.sin(bearing)], dim=1
    )
    self.goal_yaw_w[env_ids] = torch.empty(n, device=self.device).uniform_(
      *self.cfg.yaw_range
    )
    self.is_standing_env[env_ids] = (
      torch.empty(n, device=self.device).uniform_(0.0, 1.0)
      <= self.cfg.rel_standing_envs
    )
    self._pending_place[env_ids] = True

  def _update_command(self) -> None:
    # Anchor any goal sampled since the last update on the current robot pose.
    pending = self._pending_place.nonzero().flatten()
    if len(pending) > 0:
      offset = self._pending_offset[pending]
      heading = self.robot.data.heading_w[pending]
      cos_p, sin_p = torch.cos(heading), torch.sin(heading)
      offset_w = torch.stack(
        [
          cos_p * offset[:, 0] - sin_p * offset[:, 1],
          sin_p * offset[:, 0] + cos_p * offset[:, 1],
        ],
        dim=1,
      )
      self.goal_pos_w[pending] = self.robot.data.root_link_pos_w[pending, :2] + offset_w
      # Standing envs: goal == current pose so the robot learns to hold still.
      stand_ids = pending[self.is_standing_env[pending]]
      if len(stand_ids) > 0:
        self.goal_pos_w[stand_ids] = self.robot.data.root_link_pos_w[stand_ids, :2]
        self.goal_yaw_w[stand_ids] = self.robot.data.heading_w[stand_ids]
      self._pending_place[pending] = False

    base_pos_w = self.robot.data.root_link_pos_w[:, :2]
    heading_w = self.robot.data.heading_w

    delta_w = self.goal_pos_w - base_pos_w  # [N, 2]
    cos_h = torch.cos(heading_w)
    sin_h = torch.sin(heading_w)
    # Rotate world-frame delta into the base frame (rotation by -heading).
    self.goal_command_b[:, 0] = cos_h * delta_w[:, 0] + sin_h * delta_w[:, 1]
    self.goal_command_b[:, 1] = -sin_h * delta_w[:, 0] + cos_h * delta_w[:, 1]
    self.goal_command_b[:, 2] = wrap_to_pi(self.goal_yaw_w - heading_w)

  # Viewer control: click the ground to place the goal, click again to aim it.
  #
  # No scene objects are created for this. An earlier version raised a large
  # translucent plate to drag on, which is the only way viser can stream pointer
  # motion (there is no hover event -- the protocol carries just "click" and
  # "rect-select"). But a scene-sized transparent surface breaks the renderer's
  # transparency sorting: the grid and the debug arrows drop out for as long as
  # it is up. Not worth it for a live preview.

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    del name, on_change, request_action
    self._gui_get_env_idx = get_env_idx

    @server.scene.on_click()
    def _(event) -> None:
      # Ignore scene clicks unless this command is the one being driven, so
      # clicking around while another command runs cannot move a hidden goal.
      if not self._joystick_active():
        return
      hit = _ray_ground_hit(
        getattr(event, "ray_origin", None), getattr(event, "ray_direction", None)
      )
      if hit is None:
        return
      if self._awaiting_direction:
        self._aim_at(float(hit[0]), float(hit[1]))
        self._awaiting_direction = False
      else:
        self._manual_goal = (float(hit[0]), float(hit[1]))
        self._awaiting_direction = True

  def _aim_at(self, x: float, y: float) -> None:
    """Point the goal from itself toward ``(x, y)``, ignoring degenerate aims."""
    if self._manual_goal is None:
      return
    dx = x - self._manual_goal[0]
    dy = y - self._manual_goal[1]
    if math.hypot(dx, dy) > 1e-3:
      self._manual_yaw = math.atan2(dy, dx)

  @property
  def awaiting_direction(self) -> bool:
    """True between the placing click and the aiming click."""
    return self._awaiting_direction

  def _joystick_active(self) -> bool:
    # Only meaningful under a MultiCommand: the command-type selector is the
    # on/off switch, as it is for the velocity and kick controls.
    return self.joystick_gate is not None and self.joystick_gate()

  def apply_joystick(self) -> None:
    """Override the sampled goal with the one placed in the viewer."""
    if not self._joystick_active():
      self._awaiting_direction = False
      return
    if self._manual_goal is None:
      return
    assert self._gui_get_env_idx is not None
    idx = self._gui_get_env_idx()

    goal = torch.tensor(self._manual_goal, device=self.device)
    yaw = torch.tensor(self._manual_yaw, device=self.device)
    self.goal_pos_w[idx] = goal
    self.goal_yaw_w[idx] = yaw
    # _update_command already derived goal_command_b from the sampled goal, so
    # redo it here or the observation would trail the click by a step.
    delta = goal - self.robot.data.root_link_pos_w[idx, :2]
    heading = self.robot.data.heading_w[idx]
    cos_h, sin_h = torch.cos(heading), torch.sin(heading)
    self.goal_command_b[idx, 0] = cos_h * delta[0] + sin_h * delta[1]
    self.goal_command_b[idx, 1] = -sin_h * delta[0] + cos_h * delta[1]
    self.goal_command_b[idx, 2] = wrap_to_pi(yaw - heading)

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    goal_pos_ws = self.goal_pos_w.cpu().numpy()
    goal_yaw_ws = self.goal_yaw_w.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()

    arrow_len = self.cfg.viz.arrow_length
    z = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      gx, gy = goal_pos_ws[batch]
      gyaw = goal_yaw_ws[batch]
      goal_xyz = np.array([gx, gy, z])

      # Line from robot to goal (yellow).
      robot_xyz = np.array([base_pos_w[0], base_pos_w[1], z])
      visualizer.add_arrow(robot_xyz, goal_xyz, color=(0.9, 0.9, 0.1, 0.5), width=0.01)

      # Goal orientation arrow (red).
      heading_to = goal_xyz + np.array(
        [arrow_len * math.cos(gyaw), arrow_len * math.sin(gyaw), 0.0]
      )
      visualizer.add_arrow(goal_xyz, heading_to, color=(0.9, 0.1, 0.1, 0.9), width=0.02)


@dataclass(kw_only=True)
class GoalPoseCommandCfg(CommandTermCfg):
  entity_name: str
  radius_range: tuple[float, float] = (0.5, 3.0)
  """Distance range (meters) from the robot's current base position at which the
  goal position is sampled."""
  bearing_range: tuple[float, float] = (-math.pi, math.pi)
  """Direction range (radians) of the goal position, relative to the robot's
  heading at placement. The default is a full circle, where rotating by the
  heading changes nothing about the distribution."""
  yaw_range: tuple[float, float] = (-math.pi, math.pi)
  """Goal yaw sampling range (radians)."""
  rel_standing_envs: float = 0.0
  """Fraction of environments whose goal is set to the current pose, so the
  robot learns to hold still. Leave at 0 under a ``MultiCommand``, where
  standing is its own command type."""

  @dataclass
  class VizCfg:
    z_offset: float = 0.05
    arrow_length: float = 0.3

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> GoalPoseCommand:
    return GoalPoseCommand(self, env)


class UniformVelocityCommandWithStop(CommandTerm):
  """Uniform velocity command with an explicit stand/walk-in-place flag.

  This is a standalone copy of mjlab's ``UniformVelocityCommand`` (no
  inheritance) extended with a 4th command element, ``stand_flag``:

      [vx, vy, wz, stand_flag]

  ``stand_flag`` decouples "should the feet keep stepping" from "is the
  commanded velocity zero". Standing envs (zero velocity) are split into two
  behaviors:

  - ``stand_flag == 1``: stand still (freeze the gait, plant the feet).
  - ``stand_flag == 0``: walk in place (keep marching despite zero velocity).

  Moving envs always get ``stand_flag == 0``. The flag is exposed to the policy
  as part of the command and is consumed by the ``phase`` observation to decide
  whether to zero the gait phase. The mirror sign for the flag is ``+1`` (a
  left-right reflection leaves it unchanged) — see ``symmetry.py``.
  """

  cfg: UniformVelocityCommandWithStopCfg

  def __init__(
    self, cfg: UniformVelocityCommandWithStopCfg, env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)
    self.is_walk_in_place_env = torch.zeros_like(self.is_heading_env)
    self.is_world_env = torch.zeros_like(self.is_heading_env)
    self.is_forward_env = torch.zeros_like(self.is_heading_env)

    # 1.0 = stand still (no stepping), 0.0 = keep stepping.
    self.stand_flag = torch.zeros(self.num_envs, 1, device=self.device)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # When set by an owning composite command (see MultiCommand.create_gui),
    # this decides whether the joystick drives the command, replacing the
    # term's own "Enable" checkbox: the command-type selector is the switch.
    self.joystick_gate: Callable[[], bool] | None = None

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None
    self._stand_checkbox: viser.GuiCheckboxHandle | None = None
    # Only built when this term runs standalone; under a composite command the
    # owner builds the push controls, the same rule the joystick switch follows.
    self._push_gui = PushGui()

  @property
  def command(self) -> torch.Tensor:
    if not self.cfg.expose_stand_flag:
      return self.vel_command_b
    return torch.cat([self.vel_command_b, self.stand_flag], dim=1)

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    # Stand-still and walk-in-place are sampled as two independent probabilities.
    # Both zero the velocity command; they differ only in the gait: stand-still
    # freezes it (stand_flag=1), walk-in-place keeps stepping (stand_flag=0). On
    # the rare overlap (an env drawn as both) stand-still wins.
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
    self.is_walk_in_place_env[env_ids] = (
      r.uniform_(0.0, 1.0) <= self.cfg.rel_walk_in_place_envs
    ) & ~self.is_standing_env[env_ids]
    self.stand_flag[env_ids, 0] = self.is_standing_env[env_ids].float()

    # Randomly assign world-frame envs.
    self.is_world_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_world_envs
    # Copy sampled velocities as world-frame reference for world envs.
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    # Forward-only envs: positive lin_vel_x, zero lateral and angular.
    self.is_forward_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
    fwd_ids = env_ids[self.is_forward_env[env_ids]]
    if len(fwd_ids) > 0:
      self.vel_command_b[fwd_ids, 0] = (
        self.vel_command_b[fwd_ids, 0].abs().clamp(min=0.3)
      )
      self.vel_command_b[fwd_ids, 1] = 0.0
      self.vel_command_b[fwd_ids, 2] = 0.0

    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    if len(init_vel_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    # World-frame envs: rotate world-frame linear vel into body frame.
    if self.is_world_env.any():
      w_ids = self.is_world_env.nonzero(as_tuple=False).flatten()
      heading = self.robot.data.heading_w[w_ids]
      cos_h = torch.cos(heading)
      sin_h = torch.sin(heading)
      vx_w = self.vel_command_w[w_ids, 0]
      vy_w = self.vel_command_w[w_ids, 1]
      self.vel_command_b[w_ids, 0] = cos_h * vx_w + sin_h * vy_w
      self.vel_command_b[w_ids, 1] = -sin_h * vx_w + cos_h * vy_w

    # Zero the velocity for both stand-still and walk-in-place envs.
    zero_vel_env_ids = (
      (self.is_standing_env | self.is_walk_in_place_env)
      .nonzero(as_tuple=False)
      .flatten()
    )
    self.vel_command_b[zero_vel_env_ids, :] = 0.0
    self.vel_command_w[zero_vel_env_ids, :] = 0.0

  # GUI.

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon

    ranges = self.cfg.ranges

    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      # Standalone this term owns its on/off switch. Under a composite command
      # the command-type selector already decides when this command is live, so
      # a second switch here would just be a way to disagree with it.
      if self.joystick_gate is None:
        self._joystick_enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max_val,
          step=0.1,
          min=0.0,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      # Walk-in-place vs. stand-still toggle for the joystick env. Omitted when
      # an owner gates this term: standing is a command type there, not a flag.
      if self.joystick_gate is None:
        self._stand_checkbox = server.gui.add_checkbox(
          "Stand still", initial_value=False
        )

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    if self.joystick_gate is None:
      self._push_gui.create_gui(server, get_env_idx)

    # Store GUI state for apply_joystick().
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def _joystick_active(self) -> bool:
    if self.joystick_gate is not None:
      return self.joystick_gate()
    return self._joystick_enabled is not None and self._joystick_enabled.value

  def apply_joystick(self) -> None:
    """Write the viewer's sliders into the command for the viewed env.

    Must run after ``_update_command``, which zeroes the velocity of standing
    and walk-in-place envs. Called from ``compute`` when this term runs itself,
    and by ``MultiCommand`` when it owns this term (it drives the sub-terms
    through ``_update_command`` directly and never calls their ``compute``).
    """
    if not self._joystick_sliders or not self._joystick_active():
      return
    assert self._joystick_get_env_idx is not None
    idx = self._joystick_get_env_idx()
    for i, s in enumerate(self._joystick_sliders):
      self.vel_command_b[idx, i] = s.value
    if self._stand_checkbox is not None:
      self.stand_flag[idx, 0] = 1.0 if self._stand_checkbox.value else 0.0

  def compute(self, dt: float) -> None:
    super().compute(dt)
    self.apply_joystick()
    if self.joystick_gate is None:
      self._push_gui.apply(self._env)

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # Helper to transform local to world coordinates.
      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      # Command linear velocity arrow (blue).
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Command angular velocity arrow (green).
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual linear velocity arrow (cyan).
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green).
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class UniformVelocityCommandWithStopCfg(CommandTermCfg):
  entity_name: str
  expose_stand_flag: bool = True
  """Whether ``command`` carries the stand flag as a 4th element.

  True (default) gives the standalone velocity task its ``[vx, vy, wz,
  stand_flag]`` layout. Set False when this term is a sub-command of a
  ``MultiCommand``, where standing is its own command type: the command is then
  the bare 3-dim twist and a zero-velocity command means walk-in-place."""
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  """Fraction of environments commanded to stand still: zero velocity with the
  gait frozen (stand_flag=1)."""
  rel_walk_in_place_envs: float = 0.5
  """Fraction of environments commanded to walk in place: zero velocity but the
  gait keeps stepping (stand_flag=0). Sampled independently of
  ``rel_standing_envs``; on the rare overlap, stand-still takes precedence."""
  rel_heading_envs: float = 1.0
  rel_world_envs: float = 0.0
  """Fraction of environments that use world-frame velocity commands.
  World-frame envs sample linear velocity in world frame and rotate to body
  frame each step, so the command direction stays fixed in the world."""
  rel_forward_envs: float = 0.0
  """Fraction of environments that receive forward-only commands (positive
  lin_vel_x, zero lin_vel_y and ang_vel_z). Increases training coverage for
  straight-line walking, which is important for stair climbing."""
  init_velocity_prob: float = 0.0

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommandWithStop:
    return UniformVelocityCommandWithStop(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
