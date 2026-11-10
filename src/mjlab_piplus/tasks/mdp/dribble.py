from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab_piplus.tasks.mdp.kick import KickCommand
from mjlab_piplus.tasks.mdp.multi_command import resolve_command_term

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

__all__ = ["ball_lost", "dribble_ball_proximity", "dribble_ball_velocity", "dribble_robot_velocity"]

def _kick(
  env: ManagerBasedRlEnv, command_name: str, sub_command_name: str | None
) -> KickCommand:
  term, _ = resolve_command_term(env, command_name, sub_command_name)
  assert isinstance(term, KickCommand)
  return term

def dribble_ball_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  std: float = 0.25,
) -> torch.Tensor:
  """``exp(-|v_ball - v_target|^2 / std^2)``: the ball rolls as commanded."""
  term = _kick(env, command_name, sub_command_name)
  ball_vel = term.ball.data.root_link_lin_vel_w[:, :2]
  err = torch.sum(torch.square(ball_vel - term.target_vel_w), dim=1)
  return torch.exp(-err / std**2)

def dribble_robot_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  std: float = 0.25,
) -> torch.Tensor:
  """``exp(-|v_ball - v_target|^2 / std^2)``: the ball rolls as commanded."""
  term = _kick(env, command_name, sub_command_name)
  ball_vel = term.robot.data.root_link_lin_vel_w[:, :2]
  err = torch.sum(torch.square(ball_vel - term.target_vel_w), dim=1)
  return torch.exp(-err / std**2)

def dribble_ball_proximity(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  std: float = 0.3,
) -> torch.Tensor:
  """``exp(-d^2 / std^2)``, d = ball to the nearest foot."""
  term = _kick(env, command_name, sub_command_name)
  nearest = term.foot_dist.min(dim=1).values
  return torch.exp(-torch.square(nearest) / std**2)

def ball_lost(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  max_dist: float = 1.2,
) -> torch.Tensor:
  """End the episode once the ball is more than ``max_dist`` from both feet."""
  term = _kick(env, command_name, sub_command_name)
  nearest = term.foot_dist.min(dim=1).values
  return (env.episode_length_buf > 5) & (nearest > max_dist)

# def dribble_ball_still(
#   env: ManagerBasedRlEnv,
#   command_name: str,
#   sub_command_name: str | None = None,
#   vel_threshold: float = 0.1,
#   std: float = 0.25,
# ) -> torch.Tensor:
#   """``exp(-|v_ball - v_target|^2 / std^2)``: the ball rolls as commanded."""
#   term = _kick(env, command_name, sub_command_name)
#   speed = torch.norm(term.robot.data.root_link_lin_vel_w[:, :2], dim=1)
#   return (speed < vel_threshold).float()
  