"""Ball terms for the Nova dribble task.

All ball quantities are planar (x, y) and expressed in the robot's yaw frame:
x points forward, y points left, and the torso's pitch and roll are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import math

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from mjlab_piplus.tasks.mdp.commands import (
  UniformVelocityCommandWithStop,
  UniformVelocityCommandWithStopCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
_BALL = SceneEntityCfg("ball")


def _ball_in_robot_frame(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = _ROBOT,
  ball_cfg: SceneEntityCfg = _BALL,
) -> torch.Tensor:
  """Ball xy relative to the robot base, in the robot's yaw frame.

  Returns shape (num_envs, 2): column 0 is forward, column 1 is left.
  """
  robot = env.scene[robot_cfg.name]
  ball = env.scene[ball_cfg.name]
  # Vector from robot to ball in world coordinates. Height is dropped:
  # only where the ball lies on the ground matters.
  delta_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
  delta_w[:, 2] = 0.0
  # Rotate that vector back by the robot's heading, so the result no
  # longer depends on which way the robot faces in the world.
  heading = yaw_quat(robot.data.root_link_quat_w)
  return quat_apply_inverse(heading, delta_w)[:, :2]


def ball_state(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = _ROBOT,
  ball_cfg: SceneEntityCfg = _BALL,
) -> torch.Tensor:
  """Ball observation ``[x, y, vx, vy]`` in the robot's yaw frame.

  Returns shape (num_envs, 4). Position and velocity are one term so that an
  observation delay lags both by the same number of steps, like one camera
  frame would.
  """
  robot = env.scene[robot_cfg.name]
  ball = env.scene[ball_cfg.name]
  pos_b = _ball_in_robot_frame(env, robot_cfg, ball_cfg)
  # The ball's own velocity over the ground (not relative to the robot),
  # rotated into the robot's heading the same way as the position.
  # clone(): the line below writes into the tensor, and it must not write
  # into the simulator's own velocity buffer.
  vel_w = ball.data.root_link_lin_vel_w.clone()
  vel_w[:, 2] = 0.0
  heading = yaw_quat(robot.data.root_link_quat_w)
  vel_b = quat_apply_inverse(heading, vel_w)[:, :2]
  return torch.cat([pos_b, vel_b], dim=1)



def ball_near_feet(
  env: ManagerBasedRlEnv,
  std: float,
  target_offset: float = 0.25,
  robot_cfg: SceneEntityCfg = _ROBOT,
  ball_cfg: SceneEntityCfg = _BALL,
) -> torch.Tensor:
  """Reward for the ball lying ``target_offset`` m straight ahead of the robot.

  Returns shape (num_envs,), in (0, 1]: 1 when the ball is exactly on the
  target point, falling towards 0 with distance. ``std`` sets how fast it
  falls: a large std still pays from far away, a small one only up close.
  """
  pos_b = _ball_in_robot_frame(env, robot_cfg, ball_cfg)
  # Target point: target_offset m forward (x), centred (y = 0).
  target = torch.tensor([target_offset, 0.0], device=env.device)
  # Squared distance between where the ball is and where it should be.
  dist_sq = torch.sum(torch.square(pos_b - target), dim=1)
  return torch.exp(-dist_sq / std**2)

class BallChaseCommand(UniformVelocityCommandWithStop):
  """Walk behind the ball, then push it along goal_dir_w."""

  cfg: BallChaseCommandCfg

  def __init__(self, cfg: BallChaseCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    # World-frame unit vector the ball should travel along (z stays 0).
    self.goal_dir_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.goal_dir_w[:, 0] = 1.0

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    yaw = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
    self.goal_dir_w[env_ids, 0] = torch.cos(yaw)
    self.goal_dir_w[env_ids, 1] = torch.sin(yaw)

  def _update_command(self) -> None:
    ball_b = _ball_in_robot_frame(self._env)
    heading = yaw_quat(self.robot.data.root_link_quat_w)
    dir_b = quat_apply_inverse(heading, self.goal_dir_w)[:, :2]
    approach_b = ball_b - self.cfg.target_offset * dir_b #ein Punkt 0.25 m hinter dem Ball, auf der Linie Ziel–Ball. Steht der Roboter dort, liegen Ball und Ziel vor ihm.
    dist = torch.norm(approach_b, dim=1, keepdim=True)
    close = (dist < self.cfg.close_radius).float()
    vel = self.cfg.gain * approach_b + close * self.cfg.dribble_speed * dir_b
    yaw_err = torch.atan2(ball_b[:, 1], ball_b[:, 0])
    vel = vel * torch.cos(yaw_err).clamp(min=0.0).unsqueeze(1)
    r = self.cfg.ranges
    self.vel_command_b[:, 0] = vel[:, 0].clamp(*r.lin_vel_x)
    self.vel_command_b[:, 1] = vel[:, 1].clamp(*r.lin_vel_y)
    self.vel_command_b[:, 2] = (self.cfg.yaw_gain * yaw_err).clamp(*r.ang_vel_z)
    self.vel_command_b[self.is_standing_env] = 0.0


@dataclass(kw_only=True)
class BallChaseCommandCfg(UniformVelocityCommandWithStopCfg):
  gain: float = 1.0
  yaw_gain: float = 1.0
  target_offset: float = 0.25
  dribble_speed: float = 0.3
  close_radius: float = 0.15

  def build(self, env: ManagerBasedRlEnv) -> BallChaseCommand:
    return BallChaseCommand(self, env)


def ball_velocity_to_goal(
  env: ManagerBasedRlEnv,
  max_speed: float = 0.5,
  command_name: str = "twist",
  ball_cfg: SceneEntityCfg = _BALL,
) -> torch.Tensor:
  """Ball speed along the command's goal direction, scaled to [0, 1]."""
  term = env.command_manager.get_term(command_name)
  ball = env.scene[ball_cfg.name]
  vel_xy = ball.data.root_link_lin_vel_w[:, :2]
  speed = torch.sum(vel_xy * term.goal_dir_w[:, :2], dim=1)
  return speed.clamp(0.0, max_speed) / max_speed # Ab max_speed wird nicht mehr belohnt. Wegschießen lohnt sich also nicht.