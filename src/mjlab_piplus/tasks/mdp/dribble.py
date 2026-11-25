from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab_piplus.tasks.mdp.kick import KickCommand
from mjlab_piplus.tasks.mdp.multi_command import resolve_command_term

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

__all__ = ["ball_lost", "dribble_ball_proximity", "dribble_ball_velocity", "dribble_robot_velocity", "dribble_idle"]

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
  path_command_name: str | None = None,
  goal_radius: float = 0.5,
) -> torch.Tensor:
  """``exp(-|v_ball - v_target|^2 / std^2)``: the robot rolls as commanded."""
  term = _kick(env, command_name, sub_command_name)
  robot_vel = term.robot.data.root_link_lin_vel_w[:, :2]
  target = term.strength.unsqueeze(1) * term.approach_dir_w # soll zu ball laufen NEU
  err = torch.sum(torch.square(robot_vel - target), dim=1) 
  reward = torch.exp(-err / std**2)

  if path_command_name is not None:
      path = env.command_manager.get_term(path_command_name)
      ball_xy = term.ball.data.root_link_pos_w[:, :2]
      ball_goal_dist = torch.norm(path.goal_pos_w - ball_xy, dim=1)
      reward = reward * (ball_goal_dist > goal_radius).float()

  return reward

def dribble_ball_proximity(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  std: float = 0.15,
) -> torch.Tensor:
  """``exp(-d^2 / std^2)``, d = ball to the nearest foot."""
  term = _kick(env, command_name, sub_command_name)
  #nearest = term.foot_dist.min(dim=1).values
  nearest = torch.norm(term.ball_pos_b - term.ball_pos_b.new_tensor([0.25, 0.0]), dim=1) # damit der Ball nicht zwischen den füßen ist, soll der ca. 25 cm davor sein -> somit sollte er dribbeln
  return torch.exp(-torch.square(nearest) / std**2)

def ball_lost(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  max_dist: float = 1.3,
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

def dribble_idle( #aus kick_idle
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  vel_threshold: float = 0.1,
  near_dist: float = 0.2,
  path_command_name: str | None = None,
  goal_radius: float = 0.5,
) -> torch.Tensor:
  """ bestraft wenn der Ball still liegt und er nicht zum Ball geht.
  Aus, sobald der Ball näher als ``goal_radius`` am Pfad-Ziel liegt.
  """
  term = _kick(env, command_name, sub_command_name)
  ball_speed = torch.norm(term.ball.data.root_link_lin_vel_w[:, :2], dim=1)
  robo_vel = term.robot.data.root_link_lin_vel_w[:, :2]
  dist = torch.sum(robo_vel * term.approach_dir_w, dim=1)
  idle = (ball_speed < vel_threshold) & (dist < near_dist)

  if path_command_name is not None:
    path = env.command_manager.get_term(path_command_name)
    ball_xy = term.ball.data.root_link_pos_w[:, :2]
    ball_goal_dist = torch.norm(path.goal_pos_w - ball_xy, dim=1)
    idle = idle & (ball_goal_dist > goal_radius)

  return idle.float()
  