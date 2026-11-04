"""Ball terms for the Nova dribble task.

All ball quantities are planar (x, y) and expressed in the robot's yaw frame:
x points forward, y points left, and the torso's pitch and roll are ignored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

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