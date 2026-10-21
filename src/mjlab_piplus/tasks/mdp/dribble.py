"""Ball terms for the mini dribble task.

Everything is planar and uses the robot's yaw-only frame, so torso pitch and
roll do not leak into where the policy thinks the ball is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from mjlab_piplus.tasks.mdp.multi_command import resolve_goal_term

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
_BALL = SceneEntityCfg("ball")


def _to_xyz(xy: torch.Tensor) -> torch.Tensor:
  return torch.cat([xy, torch.zeros_like(xy[:, :1])], dim=1)


def _ball_offset_b(
  env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Ball position relative to the robot base, yaw frame, ``(num_envs, 2)``."""
  robot = env.scene[robot_cfg.name]
  ball = env.scene[ball_cfg.name]
  delta_w = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  q = yaw_quat(robot.data.root_link_quat_w)
  return quat_apply_inverse(q, _to_xyz(delta_w))[:, :2]


##
# Observations.
##


def ball_pos_b(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = _ROBOT,
  ball_cfg: SceneEntityCfg = _BALL,
) -> torch.Tensor:
  """Ball xy relative to the base, in the robot's yaw frame."""
  return _ball_offset_b(env, robot_cfg, ball_cfg)


def ball_vel_b(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = _ROBOT,
  ball_cfg: SceneEntityCfg = _BALL,
) -> torch.Tensor:
  """Ball xy velocity (world, not relative) rotated into the robot's yaw frame."""
  robot = env.scene[robot_cfg.name]
  ball = env.scene[ball_cfg.name]
  q = yaw_quat(robot.data.root_link_quat_w)
  vel_w = _to_xyz(ball.data.root_link_lin_vel_w[:, :2])
  return quat_apply_inverse(q, vel_w)[:, :2]


def goal_pos_b(
  env: ManagerBasedRlEnv, command_name: str, sub_command_name: str | None = None
) -> torch.Tensor:
  """Goal xy relative to the base, in the robot's yaw frame.

  Only the position: where the ball has to end up. The goal command's yaw is
  meaningless for a ball, so it is left out of the observation.
  """
  term, _ = resolve_goal_term(env, command_name, sub_command_name)
  return term.goal_command_b[:, :2]


##
# Rewards.
##


def ball_in_front(
  env: ManagerBasedRlEnv,
  std: float,
  target_offset: float = 0.25,
  robot_cfg: SceneEntityCfg = _ROBOT,
  ball_cfg: SceneEntityCfg = _BALL,
) -> torch.Tensor:
  """Keep the ball at a point ``target_offset`` m ahead of the base."""
  offset_b = _ball_offset_b(env, robot_cfg, ball_cfg)
  target = torch.tensor([target_offset, 0.0], device=env.device)
  err = torch.sum(torch.square(offset_b - target), dim=1)
  return torch.exp(-err / std**2)


def ball_goal_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_speed: float,
  sub_command_name: str | None = None,
  ball_cfg: SceneEntityCfg = _BALL,
) -> torch.Tensor:
  """Ball speed towards the goal, in m/s, capped at ``max_speed``.

  Negative when the ball rolls away from the goal. Uncapped below, so pushing
  the ball the wrong way costs as much as it would have paid the right way.
  """
  term, _ = resolve_goal_term(env, command_name, sub_command_name)
  ball = env.scene[ball_cfg.name]
  to_goal = term.goal_pos_w - ball.data.root_link_pos_w[:, :2]
  direction = to_goal / torch.norm(to_goal, dim=1, keepdim=True).clamp_min(1e-6)
  speed = torch.sum(ball.data.root_link_lin_vel_w[:, :2] * direction, dim=1)
  return torch.clamp(speed, max=max_speed)


def ball_at_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  sub_command_name: str | None = None,
  ball_cfg: SceneEntityCfg = _BALL,
) -> torch.Tensor:
  """``exp(-d^2 / std^2)`` of the ball's distance to the goal."""
  term, _ = resolve_goal_term(env, command_name, sub_command_name)
  ball = env.scene[ball_cfg.name]
  d_sq = torch.sum(
    torch.square(term.goal_pos_w - ball.data.root_link_pos_w[:, :2]), dim=1
  )
  return torch.exp(-d_sq / std**2)


##
# Terminations.
##


def ball_too_far(
  env: ManagerBasedRlEnv,
  max_distance: float,
  robot_cfg: SceneEntityCfg = _ROBOT,
  ball_cfg: SceneEntityCfg = _BALL,
) -> torch.Tensor:
  """End the episode once the ball is lost."""
  offset_b = _ball_offset_b(env, robot_cfg, ball_cfg)
  return torch.norm(offset_b, dim=1) > max_distance
