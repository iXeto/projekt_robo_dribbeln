"""Kick command: send the ball off in a commanded direction at a commanded speed.

The command owns the ball. While it is driving an env it places the ball near
the robot; while it is not, it parks the ball far away so the other command
types never trip over it. That makes the ball's presence in the scene entirely
this command's business.

What the policy sees is split in two, because the two halves are different
kinds of thing. The *command* is which way to kick and how hard:

    [cos(dir_err), sin(dir_err), strength]

``dir_err`` is the commanded kick direction relative to the robot's heading,
encoded as a unit vector because it wraps. ``MultiCommand`` zeroes the whole
block on envs the kick command is not driving.

Where the ball is and how fast it moves is *sensed*, not commanded, so it is a
separate observation term (:class:`ball_state`) with its own, much longer
sensing delay -- the ball is seen by a camera, not by an encoder. See that
class for the delay and the zeroing.

The ball lifecycle
------------------
The ball lives in a repeating cycle, driven by contact rather than by a timer:

    place -> approach -> strike -> post-kick window -> place -> ...

A strike is detected from the *ball's* speed, not from a foot/ball contact
sensor: a fast strike can touch and separate again inside a single control
step's physics substeps, which a sensor sampled once per step silently misses.
The strike opens a short window (``window_s``) over which the kick is scored
densely, and when it closes the ball is re-placed near the robot with a fresh
target direction and strength -- *without* ending the episode.

That cycle is the whole point of the design, and the fix for the reward flaw it
replaces. The previous version had two disjoint phases: a dense approach reward
until the ball moved, then a single one-off payout and nothing at all for the
rest of the command period (plus a "stand still" request whose tight posture
reward is hard to satisfy while recovering from a kick, and the ever-present
-200 for falling). Kicking therefore *ended* the robot's earning opportunity
while carrying all the fall risk, so the best policy was to walk up to the ball
and dawdle next to it. With the cycle, a kick is the only way to reach the
richest part of the reward, and finishing one immediately hands the robot
another ball to kick -- delay is now strictly worse than kicking.

Reward design
-------------
Adapted from bit-bots/AMP_mjlab's ``kick_env_cfg.py`` (branch
``amp-soccer-kick``), with the state that lives on ``env`` there kept on this
command instead -- it already owns the ball, so it also owns the cycle:

* ``kick_approach``   -- speed toward the ball along a cubed cosine alignment,
                         measured from a point offset toward the foot that
                         should strike. Signed, so backing off is penalized and
                         oscillation nets zero.
* ``kick_face_ball``  -- keep the ball ahead of the torso while approaching.
* ``kick_idle``       -- flat penalty for standing still with an untouched ball
                         still out of reach.
* ``kick_foot_alignment`` -- penalty for a foot's inside (medial) edge not
                         facing the ball when close: discourages toe-pokes.
* ``kick_contact``    -- one-shot bootstrap payout on the strike, halved into a
                         penalty if the wrong foot did it. Meant to be dropped
                         by curriculum once kicking is discovered.
* ``kick_impact``     -- dense over the window: contact style x speed match x
                         signed aim. This is the payload of a good kick.
* ``kick_direction``  -- dense over the window: signed aim on its own, so aim
                         has a gradient even when style or foot were wrong.

Two deliberate departures from the template. It has no commanded kick speed and
so rewards raw (tanh-saturated) power; this task does command a speed, so
``kick_impact`` scores the ball's peak speed against ``strength`` instead. And
its cross-foot-touch penalty is not carried over: it needs two extra per-foot
ball contact sensors, and the window here closes 1 s after the *first* touch,
which leaves far less room for both feet to fumble at the ball.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.utils.buffers import DelayBuffer
from mjlab.utils.lab_api.math import quat_apply, wrap_to_pi

from mjlab_piplus.objects import BALL_RADIUS
from mjlab_piplus.tasks.mdp.multi_command import resolve_command_term

if TYPE_CHECKING:
  import viser
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

__all__ = [
  "KickCommand",
  "KickCommandCfg",
  "ball_state",
  "kick_approach",
  "kick_ball_speed",
  "kick_contact",
  "kick_direction",
  "kick_face_ball",
  "kick_foot_alignment",
  "kick_idle",
  "kick_impact",
  "kick_phase",
]


def select_kick_foot_is_left(
  ball_pos_b: torch.Tensor,
  kick_dir_b: torch.Tensor,
  angle_threshold: float,
) -> torch.Tensor:
  """Which foot *should* strike, from the ball position and target direction.

  Both inputs are in the robot's base frame (x forward, y left). Within
  ``angle_threshold`` radians of straight ahead, take the foot on the ball's own
  side -- a roughly forward kick is played with whichever foot is nearer.
  Outside that band the target angle decides on its own: a kick angled to the
  left is played with the right foot swinging across the body, one angled right
  with the left foot. A hard rule, not a blend, so every consumer (approach
  offset, foot-correctness at contact) agrees on the answer at any instant.
  """
  angle = torch.atan2(kick_dir_b[:, 1], kick_dir_b[:, 0])
  in_band = angle.abs() <= angle_threshold
  return torch.where(in_band, ball_pos_b[:, 1] >= 0.0, angle < 0.0)


def foot_medial_alignment(
  foot_pos_xy: torch.Tensor,
  foot_quat: torch.Tensor,
  ball_pos_xy: torch.Tensor,
  medial_sign: torch.Tensor,
) -> torch.Tensor:
  """Cosine in [-1, 1] between a foot's medial axis and the foot->ball direction.

  +1 means the inside of the foot points straight at the ball (the side-foot
  kick), 0 the front or back of the foot, -1 the outside edge.

  ``medial_sign`` says whether the body's local +Y is the inside direction for
  that foot. In piplus.xml both ankle-roll links share local +Y = world +Y at
  zero joint angles, but the left foot sits on the +Y side and the right on the
  -Y side, so +Y is medial for the right foot (+1) and lateral for the left
  (-1). Shapes broadcast over a trailing foot axis.
  """
  to_ball = ball_pos_xy - foot_pos_xy
  direction = to_ball / (torch.norm(to_ball, dim=-1, keepdim=True) + 1e-6)
  local_y = torch.zeros_like(foot_quat[..., :3])
  local_y[..., 1] = medial_sign
  medial_xy = quat_apply(foot_quat, local_y)[..., :2]
  medial_xy = medial_xy / (torch.norm(medial_xy, dim=-1, keepdim=True) + 1e-6)
  return torch.sum(medial_xy * direction, dim=-1)


class KickCommand(CommandTerm):
  cfg: KickCommandCfg

  def __init__(self, cfg: KickCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.ball: Entity = env.scene[cfg.ball_entity_name]

    self._foot_ids = [self.robot.find_bodies(n)[0][0] for n in cfg.foot_body_names]
    self._foot_is_left = torch.tensor(
      [n.startswith("l_") for n in cfg.foot_body_names], device=self.device
    )
    self._medial_signs = torch.tensor(cfg.medial_signs, device=self.device)
    self._window_steps = max(1, round(cfg.window_s / env.step_dt))
    self._select_angle = math.radians(cfg.foot_select_angle_deg)

    # Commanded kick: a world-frame heading and a target ball speed. Both are
    # redrawn on every ball placement, not just on command resample.
    self.kick_dir_w = torch.zeros(self.num_envs, device=self.device)
    self.strength = torch.zeros(self.num_envs, device=self.device)

    # Ball lifecycle state. ``touched`` is the phase flag (approach vs. window),
    # ``window_left`` counts the post-kick scoring window down in steps, and
    # ``style``/``foot_correct`` are frozen at the moment of the strike.
    self.touched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.window_left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.new_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.style = torch.zeros(self.num_envs, device=self.device)
    self.foot_correct = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    # Fastest the ball got since the strike. Scoring the instantaneous velocity
    # would chase a ball that is already decelerating under rolling friction;
    # the peak is what the kick actually produced.
    self.peak_speed = torch.zeros(self.num_envs, device=self.device)
    self.peak_vel_w = torch.zeros(self.num_envs, 2, device=self.device)

    # Ball state in the robot base frame, refreshed every step.
    self.ball_pos_b = torch.zeros(self.num_envs, 2, device=self.device)
    self.ball_vel_b = torch.zeros(self.num_envs, 2, device=self.device)
    self.ball_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.dir_err = torch.zeros(self.num_envs, device=self.device)

    # Approach geometry, refreshed every step: which foot should strike, the
    # world-frame unit vector from the offset approach point to the ball, and
    # the ball's bearing from that same point.
    self.desired_foot_is_left = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.approach_dir_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.approach_angle = torch.zeros(self.num_envs, device=self.device)
    self.foot_dist = torch.zeros(self.num_envs, len(self._foot_ids), device=self.device)
    self.foot_align = torch.zeros(
      self.num_envs, len(self._foot_ids), device=self.device
    )

    # Set each step by MultiCommand; None means "this term drives every env".
    self.active_env_mask: torch.Tensor | None = None
    self._was_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # write_root_state_to_sim does not show up in the entity's derived state
    # until the next forward, so remember what we wrote and use that for this
    # step's observation. Without it a freshly placed ball would be observed at
    # its parked position for one step -- a 100 m spike into the policy.
    self._wrote = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._wrote_pos = torch.zeros(self.num_envs, 2, device=self.device)

    # Placement is sampled whenever a ball life ends but applied in
    # _update_command. At an episode reset the command manager runs *before*
    # sim.forward(), so the robot pose visible during _resample_command is still
    # the pre-reset one; placing there would put the ball relative to wherever
    # the robot used to be. _update_command runs after the forward, so it sees
    # the real pose -- and the same deferral gives the post-kick respawn a
    # current pose to anchor on too.
    self._pending_place = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._pending_forward = torch.zeros(self.num_envs, device=self.device)
    self._pending_side = torch.zeros(self.num_envs, device=self.device)

    # Viewer control. joystick_gate is set by MultiCommand to "is the kick type
    # selected"; see the JoystickTerm protocol in multi_command.py.
    self.joystick_gate: Callable[[], bool] | None = None
    self._gui_direction: viser.GuiSliderHandle | None = None
    self._gui_strength: viser.GuiSliderHandle | None = None
    self._gui_get_env_idx: Callable[[], int] | None = None

    # Kicks completed per episode. The per-kick quality numbers are logged from
    # ``_score_finished_kicks`` instead: averaging them over an episode that
    # contains a variable number of kicks says very little.
    self.metrics["kicks"] = torch.zeros(self.num_envs, device=self.device)

  # Properties.

  @property
  def command(self) -> torch.Tensor:
    return torch.cat(
      [
        torch.cos(self.dir_err).unsqueeze(1),
        torch.sin(self.dir_err).unsqueeze(1),
        self.strength.unsqueeze(1),
      ],
      dim=1,
    )

  @property
  def ball_state_b(self) -> torch.Tensor:
    """Ball position and velocity in the base frame, ``(num_envs, 4)``.

    Not part of ``command``: it is a sensed quantity, observed through
    :class:`ball_state` with its own delay.
    """
    return torch.cat([self.ball_pos_b, self.ball_vel_b], dim=1)

  @property
  def target_vel_w(self) -> torch.Tensor:
    """Commanded ball velocity in world xy, ``(num_envs, 2)``."""
    return self.strength.unsqueeze(1) * torch.stack(
      [torch.cos(self.kick_dir_w), torch.sin(self.kick_dir_w)], dim=1
    )

  # The GoalPoseTerm contract, with the ball as the "goal": lets the shared
  # goal-bearing rewards be pointed at this command too.

  @property
  def goal_pos_w(self) -> torch.Tensor:
    return self.ball_pos_w

  @property
  def goal_command_b(self) -> torch.Tensor:
    return torch.cat([self.ball_pos_b, self.dir_err.unsqueeze(1)], dim=1)

  @property
  def yaw_err(self) -> torch.Tensor:
    return self.dir_err

  # Command lifecycle.

  def _active(self) -> torch.Tensor:
    if self.active_env_mask is None:
      return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    return self.active_env_mask

  def _write_ball(self, env_ids: torch.Tensor, pos_xy: torch.Tensor) -> None:
    """Teleport the ball to ``pos_xy`` at rest."""
    if len(env_ids) == 0:
      return
    state = torch.zeros(len(env_ids), 13, device=self.device)
    state[:, :2] = pos_xy
    state[:, 2] = self.cfg.ball_radius
    state[:, 3] = 1.0  # Identity quaternion; velocities stay zero.
    self.ball.write_root_state_to_sim(state, env_ids)
    self._wrote[env_ids] = True
    self._wrote_pos[env_ids] = pos_xy

  def _begin_ball_life(self, env_ids: torch.Tensor) -> None:
    """Queue a placement and clear everything scored against the old ball."""
    if len(env_ids) == 0:
      return
    n = len(env_ids)
    # Polar around the robot's heading: a radius and a bearing inside the
    # placement cone. Sampling a forward distance and a sideways offset instead
    # would leave the widest bearing an implicit function of both, and bound the
    # ball's actual distance only at the cone's centre line.
    distance = torch.empty(n, device=self.device).uniform_(*self.cfg.distance_range)
    cone = math.radians(self.cfg.placement_cone_deg)
    bearing = torch.empty(n, device=self.device).uniform_(-cone, cone)

    self._pending_forward[env_ids] = distance * torch.cos(bearing)
    self._pending_side[env_ids] = distance * torch.sin(bearing)
    self._pending_place[env_ids] = True

    self.touched[env_ids] = False
    self.window_left[env_ids] = 0
    self.style[env_ids] = 0.0
    self.foot_correct[env_ids] = False
    self.peak_speed[env_ids] = 0.0
    self.peak_vel_w[env_ids] = 0.0

  def _apply_pending_placement(self) -> None:
    """Place any queued ball, now that the robot pose is current.

    The commanded direction and strength are drawn here rather than at sample
    time for the same reason: the direction is a cone around the robot's heading
    *at placement*, which only reads correctly after the forward.
    """
    env_ids = self._pending_place.nonzero().flatten()
    if len(env_ids) == 0:
      return
    n = len(env_ids)
    base_xy = self.robot.data.root_link_pos_w[env_ids, :2]
    yaw = self.robot.data.heading_w[env_ids]
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    # Local -Y is the robot's right (the convention foot_medial_alignment uses),
    # so right_w = R(yaw) @ (0, -1) = (sin, -cos).
    forward = torch.stack([cos_y, sin_y], dim=1)
    right = torch.stack([sin_y, -cos_y], dim=1)
    offset = forward * self._pending_forward[env_ids].unsqueeze(
      1
    ) + right * self._pending_side[env_ids].unsqueeze(1)
    self._write_ball(env_ids, base_xy + offset)

    cone = math.radians(self.cfg.direction_cone_deg)
    self.kick_dir_w[env_ids] = wrap_to_pi(
      yaw + torch.empty(n, device=self.device).uniform_(-cone, cone)
    )
    self.strength[env_ids] = torch.empty(n, device=self.device).uniform_(
      *self.cfg.strength_range
    )
    self._pending_place[env_ids] = False

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._begin_ball_life(env_ids)

  def _update_command(self) -> None:
    active = self._active()
    self._apply_pending_placement()

    # Park the ball wherever this command is not driving, so it cannot get in
    # the way of whatever command took over. Normally only on the transition --
    # a parked ball rests where it is and needs no further writes -- but also
    # for freshly reset envs, because scene.reset() restores every entity to its
    # default pose and so undoes the parking.
    just_reset = self._env.episode_length_buf <= 1
    newly_inactive = (~active & (self._was_active | just_reset)).nonzero().flatten()
    if len(newly_inactive) > 0:
      park = torch.tensor(self.cfg.park_position, device=self.device)
      park_xy = self._env.scene.env_origins[newly_inactive, :2] + park
      self._write_ball(newly_inactive, park_xy)
    self._was_active = active.clone()

    # Ball state, preferring what we just wrote where the sim has not caught up.
    pos_w = self.ball.data.root_link_pos_w[:, :2]
    vel_w = self.ball.data.root_link_lin_vel_w[:, :2]
    if bool(self._wrote.any()):
      fresh = self._wrote.unsqueeze(1)
      pos_w = torch.where(fresh, self._wrote_pos, pos_w)
      vel_w = torch.where(fresh, torch.zeros_like(vel_w), vel_w)
      self._wrote[:] = False
    self.ball_pos_w = pos_w

    # Into the base frame.
    base_xy = self.robot.data.root_link_pos_w[:, :2]
    heading = self.robot.data.heading_w
    cos_h, sin_h = torch.cos(heading), torch.sin(heading)
    delta = pos_w - base_xy
    self.ball_pos_b = torch.stack(
      [
        cos_h * delta[:, 0] + sin_h * delta[:, 1],
        -sin_h * delta[:, 0] + cos_h * delta[:, 1],
      ],
      dim=1,
    )
    self.ball_vel_b = torch.stack(
      [
        cos_h * vel_w[:, 0] + sin_h * vel_w[:, 1],
        -sin_h * vel_w[:, 0] + cos_h * vel_w[:, 1],
      ],
      dim=1,
    )
    self.dir_err = wrap_to_pi(self.kick_dir_w - heading)

    self._update_approach_geometry(cos_h, sin_h)
    self._update_cycle(active, vel_w)

  def _update_approach_geometry(self, cos_h: torch.Tensor, sin_h: torch.Tensor) -> None:
    """Foot-to-ball geometry and the offset approach target, refreshed per step."""
    foot_pos = self.robot.data.body_link_pos_w[:, self._foot_ids, :2]
    foot_quat = self.robot.data.body_link_quat_w[:, self._foot_ids]
    ball_xy = self.ball_pos_w.unsqueeze(1)
    self.foot_dist = torch.norm(ball_xy - foot_pos, dim=-1)
    self.foot_align = foot_medial_alignment(
      foot_pos, foot_quat, ball_xy, self._medial_signs
    )

    kick_dir_b = torch.stack([torch.cos(self.dir_err), torch.sin(self.dir_err)], dim=1)
    self.desired_foot_is_left = select_kick_foot_is_left(
      self.ball_pos_b, kick_dir_b, self._select_angle
    )

    # Aim the approach at a point offset sideways toward the foot that should
    # strike, rather than at the body centre: arriving with the torso on the
    # ball leaves neither foot in a position to swing.
    offset_y = torch.where(
      self.desired_foot_is_left,
      torch.full_like(self.ball_pos_b[:, 1], self.cfg.approach_foot_offset),
      torch.full_like(self.ball_pos_b[:, 1], -self.cfg.approach_foot_offset),
    )
    rel = self.ball_pos_b - torch.stack([torch.zeros_like(offset_y), offset_y], dim=1)
    self.approach_angle = torch.atan2(rel[:, 1], rel[:, 0])
    dir_b = rel / (torch.norm(rel, dim=1, keepdim=True) + 1e-6)
    self.approach_dir_w = torch.stack(
      [
        cos_h * dir_b[:, 0] - sin_h * dir_b[:, 1],
        sin_h * dir_b[:, 0] + cos_h * dir_b[:, 1],
      ],
      dim=1,
    )

  def _update_cycle(self, active: torch.Tensor, vel_w: torch.Tensor) -> None:
    """Detect the strike, run the scoring window, and start the next ball life."""
    speed = torch.norm(vel_w, dim=1)

    self.new_contact = active & ~self.touched & (speed > self.cfg.contact_speed)
    if bool(self.new_contact.any()):
      # Score the strike with whichever foot is actually nearest the ball: a
      # contact sensor can miss the touch entirely (see the module docstring),
      # and the near foot is what swung.
      closest = torch.argmin(self.foot_dist, dim=1)
      align = torch.gather(self.foot_align, 1, closest.unsqueeze(1)).squeeze(1)
      # Squared, so only a near-medial contact earns close to full credit and a
      # glancing one is heavily discounted.
      style = torch.clamp(align, 0.0, 1.0) ** 2
      correct = self._foot_is_left[closest] == self.desired_foot_is_left

      self.style = torch.where(self.new_contact, style, self.style)
      self.foot_correct = torch.where(self.new_contact, correct, self.foot_correct)
      self.touched |= self.new_contact
      self.window_left = torch.where(
        self.new_contact,
        torch.full_like(self.window_left, self._window_steps),
        self.window_left,
      )
      self.peak_speed = torch.where(
        self.new_contact, torch.zeros_like(self.peak_speed), self.peak_speed
      )
      self.metrics["kicks"] += self.new_contact.float()

    in_window = active & (self.window_left > 0)
    better = in_window & (speed > self.peak_speed)
    self.peak_speed = torch.where(better, speed, self.peak_speed)
    self.peak_vel_w = torch.where(better.unsqueeze(1), vel_w, self.peak_vel_w)

    expiring = in_window & (self.window_left == 1)
    self.window_left = torch.where(in_window, self.window_left - 1, self.window_left)
    if bool(expiring.any()):
      self._score_finished_kicks(expiring)
      self._begin_ball_life(expiring.nonzero().flatten())

  def _score_finished_kicks(self, done: torch.Tensor) -> None:
    """Log what the kicks that just finished actually achieved."""
    ball_dir = torch.atan2(self.peak_vel_w[:, 1], self.peak_vel_w[:, 0])
    dir_err = wrap_to_pi(ball_dir - self.kick_dir_w).abs()
    n = done.sum().clamp(min=1)
    log = self._env.extras.setdefault("log", {})
    log["Metrics/kick_peak_speed"] = ((self.peak_speed * done).sum() / n).item()
    log["Metrics/kick_speed_error"] = (
      ((self.peak_speed - self.strength).abs() * done).sum() / n
    ).item()
    log["Metrics/kick_aim_error"] = ((dir_err * done).sum() / n).item()
    log["Metrics/kick_style"] = ((self.style * done).sum() / n).item()
    log["Metrics/kick_right_foot"] = ((self.foot_correct & done).sum() / n).item()

  def _update_metrics(self) -> None:
    # Everything worth reporting is either a per-episode count (accumulated in
    # ``_update_cycle``) or a per-kick value (logged as each window closes).
    pass

  # Viewer control.

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    del on_change, request_action
    lo_str, hi_str = self.cfg.strength_range
    with server.gui.add_folder(name.capitalize()):
      self._gui_direction = server.gui.add_slider(
        "direction [rad]", min=-math.pi, max=math.pi, step=0.05, initial_value=0.0
      )
      self._gui_strength = server.gui.add_slider(
        "strength [m/s]",
        min=lo_str,
        max=hi_str,
        step=0.1,
        initial_value=0.5 * (lo_str + hi_str),
      )
    self._gui_get_env_idx = get_env_idx

  def _joystick_active(self) -> bool:
    # Only meaningful under a MultiCommand, which is the only place this term is
    # used: the command-type selector is the on/off switch.
    return self.joystick_gate is not None and self.joystick_gate()

  def apply_joystick(self) -> None:
    """Drive the commanded kick from the viewer sliders for the viewed env."""
    if self._gui_direction is None or self._gui_strength is None:
      return
    if not self._joystick_active():
      return
    assert self._gui_get_env_idx is not None
    idx = self._gui_get_env_idx()
    self.kick_dir_w[idx] = self._gui_direction.value
    self.strength[idx] = self._gui_strength.value
    # dir_err was derived in _update_command from the previous direction; redo
    # it so this step's observation already reflects the slider.
    self.dir_err[idx] = wrap_to_pi(
      self.kick_dir_w[idx] - self.robot.data.heading_w[idx]
    )

  # Visualization.

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    import numpy as np

    ball_xy = self.ball_pos_w.cpu().numpy()
    dirs = self.kick_dir_w.cpu().numpy()
    strengths = self.strength.cpu().numpy()
    in_window = (self.window_left > 0).cpu().numpy()
    for batch in env_indices:
      # Arrow from the ball along the commanded kick, scaled by strength. Green
      # while the post-kick window is being scored, orange while approaching.
      length = 0.15 * float(strengths[batch])
      start = np.array([ball_xy[batch, 0], ball_xy[batch, 1], BALL_RADIUS])
      end = start + np.array(
        [length * math.cos(dirs[batch]), length * math.sin(dirs[batch]), 0.0]
      )
      color = (0.2, 0.9, 0.3, 0.9) if in_window[batch] else (1.0, 0.4, 0.0, 0.9)
      visualizer.add_arrow(start, end, color=color, width=0.02)


@dataclass(kw_only=True)
class KickCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  ball_entity_name: str = "ball"
  ball_radius: float = BALL_RADIUS

  foot_body_names: tuple[str, ...] = ("r_ankle_roll_link", "l_ankle_roll_link")
  """Bodies that may strike the ball. Names starting with ``l_`` are taken as
  the left foot."""
  medial_signs: tuple[float, ...] = (1.0, -1.0)
  """Per foot, the local Y sign that points to the inside of the foot; see
  :func:`foot_medial_alignment`. Order matches ``foot_body_names``."""

  distance_range: tuple[float, float] = (0.3, 1.2)
  """How far from the robot the ball is placed, in metres. A true radius: the
  ball is this far away whatever bearing it is placed at."""
  placement_cone_deg: float = 60.0
  """Half-width of the fan the ball is placed in, either side of the robot's
  heading. Straight ahead is the middle of the range, not an edge case, and the
  ball is never placed beside or behind the robot -- it has no way to know a
  ball is there beyond what the command block tells it. Widen once the approach
  is reliable to make the robot turn and find the ball."""
  direction_cone_deg: float = 80.0
  """Commanded kick heading is drawn within this cone of the robot's heading at
  placement -- a direction it has some chance of hitting from where it stands.
  Widen towards 180 to demand turning the ball around."""

  strength_range: tuple[float, float] = (1.0, 5.0)
  """Commanded ball speed in m/s."""

  park_position: tuple[float, float] = (100.0, 100.0)
  """Where the ball is sent, relative to the env origin, while this command is
  not driving the env -- far enough that it cannot interact with anything."""

  contact_speed: float = 0.5
  """Ball speed (m/s) that counts as a strike and opens the scoring window.
  High enough that brushing past the ball while walking does not start a kick."""

  window_s: float = 1.0
  """Seconds the post-kick scoring window stays open. The kick rewards are paid
  densely across it, then the ball is re-placed and the cycle restarts. Long
  enough to be worth more than the approach, short enough that the robot cannot
  catch up with the ball and keep topping its speed up."""

  foot_select_angle_deg: float = 20.0
  """Half-width of the "straight ahead" band in
  :func:`select_kick_foot_is_left`."""
  approach_foot_offset: float = 0.2
  """Metres to offset the approach target sideways toward the striking foot."""

  resampling_time_range: tuple[float, float] = (8.0, 16.0)

  @dataclass
  class VizCfg:
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> KickCommand:
    return KickCommand(self, env)


##
# Shared helpers.
##


def _kick_term(
  env: ManagerBasedRlEnv, command_name: str, sub_command_name: str | None
) -> KickCommand:
  term, _ = resolve_command_term(env, command_name, sub_command_name)
  assert isinstance(term, KickCommand)
  return term


def _direction_factor(
  vel_w: torch.Tensor, kick_dir_w: torch.Tensor, sigma: float
) -> torch.Tensor:
  """Signed aim quality in [-1, 1] against the commanded direction.

  +1 dead on, 0 at roughly the tolerance boundary, -1 straight backwards. Signed
  rather than a plain Gaussian so a hard kick in the wrong direction is an
  active mistake, not merely a missed reward -- otherwise "kick hard, ignore
  aim" is a perfectly good strategy. ``sigma`` is meant to start generous and be
  narrowed by curriculum.
  """
  ball_dir = torch.atan2(vel_w[:, 1], vel_w[:, 0])
  err = wrap_to_pi(ball_dir - kick_dir_w)
  return 2.0 * torch.exp(-0.5 * (err / sigma) ** 2) - 1.0


def _speed_score(
  speed: torch.Tensor, strength: torch.Tensor, rel_std: float
) -> torch.Tensor:
  """How close the ball's peak speed came to the commanded strength, in [0, 1].

  The tolerance is relative to the command, so a 5 m/s kick is not held to the
  same absolute precision as a 1 m/s one.
  """
  err = (speed - strength) / (rel_std * strength.clamp(min=1e-3))
  return torch.exp(-0.5 * err**2)


##
# Observation.
##


class ball_state:
  """Delayed ball observation: ``[ball_x, ball_y, ball_vx, ball_vy]``, base frame.

  Deliberately not part of the kick *command*. The ball is seen, not commanded,
  and it is seen by a camera: the delay here is ``min_lag``-``max_lag`` control
  steps, an order of magnitude longer than the one-step lag the proprioceptive
  terms carry. That is its own sensing channel, so it gets its own term rather
  than riding along on a command block whose other entries the policy is told
  instantly.

  The observation is zeroed on every env the kick command is not driving --
  *after* the delay as well as before it. Both matter, and for different
  reasons. Masking on the way in keeps the parked ball (100 m away, see
  ``KickCommandCfg.park_position``) out of the history, so switching *to* kick
  cannot serve a 100 m spike out of the buffer; masking on the way out makes
  the zeroing immediate, so switching *away* from kick does not leave a stale
  ball in the observation for the length of the lag while the one-hot already
  says otherwise.

  mjlab's own ``delay_min_lag`` / ``delay_max_lag`` are applied by the
  observation manager *after* the term, which would delay the mask along with
  the data -- hence the buffer here. It is mjlab's ``DelayBuffer`` all the
  same, with the same per-env, resampled-every-step lag, and the observation
  manager still resets it per episode through the ``reset`` hook below.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv) -> None:
    self._buffer = DelayBuffer(
      min_lag=int(cfg.params["min_lag"]),
      max_lag=int(cfg.params["max_lag"]),
      batch_size=env.num_envs,
      device=env.device,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sub_command_name: str | None = None,
    min_lag: int = 0,
    max_lag: int = 0,
  ) -> torch.Tensor:
    del min_lag, max_lag  # Read once in __init__; the buffer owns them.
    term, active = resolve_command_term(env, command_name, sub_command_name)
    assert isinstance(term, KickCommand)
    mask = active.unsqueeze(1).float()
    self._buffer.append(term.ball_state_b * mask)
    return self._buffer.compute() * mask

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._buffer.reset(batch_ids=env_ids)


##
# Metrics.
##


def kick_ball_speed(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
) -> torch.Tensor:
  """Ball speed in m/s (world xy). Reads 0 while the ball sits parked."""
  term = _kick_term(env, command_name, sub_command_name)
  return torch.norm(term.ball.data.root_link_lin_vel_w[:, :2], dim=1)


def kick_phase(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
) -> torch.Tensor:
  """Where in the ball cycle an env is.

  0 = the kick command is not driving this env, 1 = approaching the ball,
  2 = inside the post-kick scoring window.
  """
  term, active = resolve_command_term(env, command_name, sub_command_name)
  assert isinstance(term, KickCommand)
  ones = torch.ones(env.num_envs, device=env.device)
  phase = torch.where(term.window_left > 0, 2.0 * ones, ones)
  return torch.where(active, phase, torch.zeros_like(phase))


##
# Rewards: approach.
##


def kick_approach(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  max_speed: float = 1.0,
  alignment_power: float = 3.0,
) -> torch.Tensor:
  """Reward moving toward the ball, before it has been struck.

  ``speed * sign(cos) * |cos|**alignment_power``, where ``cos`` is between the
  base's world velocity and the direction to the ball from a point offset toward
  the striking foot. Cubing narrows the reward around a well-aimed approach: 30
  degrees off keeps ~0.65 of the credit instead of a bare cosine's 0.87, and 60
  degrees off keeps ~0.13. The sign is carried through, so retreating is
  penalized and walking out and back nets zero -- there is nothing to farm by
  oscillating, unlike a distance-delta reward.

  No distance term: scaling by distance would vanish exactly where the robot
  needs the guidance most. ``max_speed`` caps the credited speed so charging the
  ball flat out is not better than arriving under control.
  """
  term = _kick_term(env, command_name, sub_command_name)
  vel = term.robot.data.root_link_lin_vel_w[:, :2]
  speed = torch.norm(vel, dim=1)
  cos_align = torch.clamp(
    torch.sum(vel * term.approach_dir_w, dim=1) / (speed + 1e-6), -1.0, 1.0
  )
  signed = torch.sign(cos_align) * cos_align.abs() ** alignment_power
  reward = speed.clamp(max=max_speed) * signed
  return torch.where(term.touched, torch.zeros_like(reward), reward)


def kick_face_ball(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
) -> torch.Tensor:
  """Reward keeping the ball ahead of the torso while approaching it.

  ``exp(-|bearing|)`` from the same offset approach point ``kick_approach``
  uses: 1 with the ball dead ahead, decaying as it swings to the side or behind.
  Arriving side-on or backing into the ball leaves no way to swing at it.
  """
  term = _kick_term(env, command_name, sub_command_name)
  reward = torch.exp(-term.approach_angle.abs())
  return torch.where(term.touched, torch.zeros_like(reward), reward)


def kick_idle(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  vel_threshold: float = 0.1,
  near_dist: float = 0.35,
) -> torch.Tensor:
  """Flat penalty for standing still with an out-of-reach ball still untouched.

  Flat rather than proportional to how slow, so it cannot be paid off cheaply by
  creeping along just above the threshold. This is what removes the do-nothing
  optimum: without it, refusing to engage costs nothing at all, since
  ``kick_approach`` is signed and reads zero when standing.

  Only applies beyond ``near_dist``. Planting the support foot to swing at the
  ball drops the base speed to almost nothing, and that is the very motion the
  task wants -- penalizing it would be self-defeating.
  """
  term = _kick_term(env, command_name, sub_command_name)
  speed = torch.norm(term.robot.data.root_link_lin_vel_w[:, :2], dim=1)
  dist = torch.norm(term.ball_pos_b, dim=1)
  return ((speed < vel_threshold) & ~term.touched & (dist > near_dist)).float()


def kick_foot_alignment(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  close_dist: float = 0.9,
) -> torch.Tensor:
  """Penalize a nearby foot's inside edge not facing the ball.

  ``1 - cos(angle)`` per foot within ``close_dist``, summed: 0 when the inside
  of the foot points at the ball, up to 2 when it points away. Targets the
  toe-poke -- striking with the front of the foot, which is neither accurate nor
  the motion a side-foot pass wants. Applied to both feet, since either may end
  up being the one that swings, and only while the ball is still untouched.
  """
  term = _kick_term(env, command_name, sub_command_name)
  misalign = 1.0 - torch.clamp(term.foot_align, -1.0, 1.0)
  penalty = (misalign * (term.foot_dist < close_dist).float()).sum(dim=1)
  return torch.where(term.touched, torch.zeros_like(penalty), penalty)


##
# Rewards: the kick itself.
##


def kick_contact(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  wrong_foot_penalty: float = 0.5,
) -> torch.Tensor:
  """One-shot payout on the strike, gated on striking with the right foot.

  Pure bootstrap: it rewards making contact at all, which is what the policy has
  to stumble into before any of the quality terms can say anything useful. Drop
  it by curriculum once kicking is established -- left on, a flat payout per
  contact makes a stream of weak taps competitive with real kicks, because the
  ball is re-placed after every one.

  ``-wrong_foot_penalty`` instead when the other foot got there first: which
  foot should play the ball follows from the ball's position and the target
  direction (see :func:`select_kick_foot_is_left`).

  Divided by the timestep, because the reward manager scales every term by dt:
  the payout then lands as its face value, whatever the control rate.
  """
  term = _kick_term(env, command_name, sub_command_name)
  payout = torch.where(
    term.foot_correct,
    torch.ones_like(term.style),
    torch.full_like(term.style, -wrong_foot_penalty),
  )
  return torch.where(term.new_contact, payout, torch.zeros_like(payout)) / env.step_dt


def kick_impact(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  dir_sigma: float = 1.05,
  speed_rel_std: float = 0.35,
  min_ball_speed: float = 0.2,
  style_floor: float = 0.25,
) -> torch.Tensor:
  """Score the kick, densely, for as long as the post-kick window is open.

  ``style * speed_score * direction_factor``, all measured on the *peak* ball
  velocity since the strike:

  * ``style``: how medial the contact was, frozen at the strike, lifted off zero
    by ``style_floor``. The template multiplies by the raw value, which zeroes
    the payload for a toe-poke -- but a toe-poke that sends the ball at the
    commanded speed and direction is exactly the discovery the policy has to
    make first, and it is already penalized by ``kick_foot_alignment``. The
    floor keeps a clean side-foot contact worth ``1/style_floor`` times as much
    without making the alternative worthless.
  * ``speed_score``: how close the peak came to the commanded strength.
  * ``direction_factor``: signed aim, so power in the wrong direction is a
    mistake rather than a smaller reward.

  Paid every step of the window rather than once, so the kick -- not the walk up
  to it -- is where the reward lives. That is the flaw this design set out to
  fix: a single payout at the moment of contact is easily worth less than the
  approach that preceded it, which makes never kicking the better policy.

  The window is short and the ball is re-placed when it closes, which is what
  stops the obvious abuse of a dense ball-velocity reward: the robot cannot
  dribble the ball along, topping its speed back up, because the ball is gone
  a second after the first touch.

  Gated on the correct foot: a kick with the wrong one already takes
  :func:`kick_contact`'s penalty and should not collect the payload on top.
  """
  term = _kick_term(env, command_name, sub_command_name)
  active = (term.window_left > 0) & term.foot_correct
  active &= term.peak_speed > min_ball_speed
  style = style_floor + (1.0 - style_floor) * term.style
  score = (
    style
    * _speed_score(term.peak_speed, term.strength, speed_rel_std)
    * _direction_factor(term.peak_vel_w, term.kick_dir_w, dir_sigma)
  )
  return torch.where(active, score, torch.zeros_like(score))


def kick_direction(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  dir_sigma: float = 1.05,
  min_ball_speed: float = 0.2,
) -> torch.Tensor:
  """Signed aim of the kick, densely over the window.

  The same factor :func:`kick_impact` folds in, on its own and ungated by style
  or foot correctness, so aim keeps a gradient even on a kick that was scrappy
  in every other respect. Shares ``kick_impact``'s parameter name so a single
  curriculum can narrow both tolerances together -- letting them drift apart
  would score the same kick two different ways.
  """
  term = _kick_term(env, command_name, sub_command_name)
  active = (term.window_left > 0) & (term.peak_speed > min_ball_speed)
  factor = _direction_factor(term.peak_vel_w, term.kick_dir_w, dir_sigma)
  return torch.where(active, factor, torch.zeros_like(factor))
