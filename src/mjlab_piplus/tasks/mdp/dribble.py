from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import math

from mjlab_piplus.tasks.mdp.kick import KickCommand
from mjlab_piplus.tasks.mdp.multi_command import resolve_command_term

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

__all__ = ["ball_lost", "dribble_ball_proximity", "dribble_ball_velocity", "dribble_robot_velocity", "dribble_to_goal_vel", "dribble_to_goal_pos", "dribble_idle", "dribble_goal_pos_b"]

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
  """``exp(-|v_ball - v_target|^2 / std^2)``: the robot rolls as commanded."""
  term = _kick(env, command_name, sub_command_name)
  ball_vel = term.robot.data.root_link_lin_vel_w[:, :2]
  target = term.strength.unsqueeze(1) * term.approach_dir_w # soll zu ball laufen NEU
  err = torch.sum(torch.square(ball_vel - target), dim=1) 
  return torch.exp(-err / std**2)

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
) -> torch.Tensor:
  """ belohnt Robot fürs zugehen auf den ball
  """
  term = _kick(env, command_name, sub_command_name)
  ball_speed = torch.norm(term.ball.data.root_link_lin_vel_w[:, :2], dim=1)
  robo_vel = term.robot.data.root_link_lin_vel_w[:, :2]
  dist = torch.sum(robo_vel * term.approach_dir_w, dim=1)
  return ((ball_speed < vel_threshold) & (dist > near_dist)).float()

#Neuer
class dribble_to_goal_pos: # generiert random ziele für den Ball, die der Roboter erreichen soll, ist eine Klasse, damit die Ziele persistent sind und nicht bei jedem Aufruf neu generiert werden
    goals: torch.Tensor | None = None
    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.env = env
        dribble_to_goal_pos.goals = torch.zeros((env.num_envs, 2), device=env.device)
        
        self.reset(torch.arange(env.num_envs, device=env.device))

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        if len(env_ids) == 0:
            return
            
        robot_pos = self.env.scene["robot"].data.root_link_pos_w[env_ids, :2]
        
        distanz = torch.empty(len(env_ids), device=self.env.device).uniform_(3.0, 6.0)
        
        winkel = torch.empty(len(env_ids), device=self.env.device).uniform_(-math.pi, math.pi)
        
        offset_x = distanz * torch.cos(winkel)
        offset_y = distanz * torch.sin(winkel)
        
        dribble_to_goal_pos.goals[env_ids, 0] = robot_pos[:, 0] + offset_x
        dribble_to_goal_pos.goals[env_ids, 1] = robot_pos[:, 1] + offset_y

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        sub_command_name: str | None = None,
        std: float = 2.0,
    ) -> torch.Tensor:
        term = _kick(env, command_name, sub_command_name)
        
        ball_pos = term.ball.data.root_link_pos_w[:, :2]
        
        delta = dribble_to_goal_pos.goals - ball_pos
        dist = torch.norm(delta, dim=1)
        
        return torch.exp(-torch.square(dist) / std**2)

#Neuer
def dribble_to_goal_vel( # Neuer Velocity Reward, der den Ball in Richtung des Ziels beschleunigt, weil der alte Reward sich mit der Zielpos. widersprechen würde
    env: ManagerBasedRlEnv,
    command_name: str,
    sub_command_name: str | None = None,
    max_speed: float = 3.0
  ) -> torch.Tensor:
    term = _kick(env, command_name, sub_command_name)

    ball_vel = term.ball.data.root_link_lin_vel_w[:, :2]
    ball_pos = term.ball.data.root_link_pos_w[:, :2]
    goals = dribble_to_goal_pos.goals
    if goals is None:
        return torch.zeros(env.num_envs, device=env.device)

    delta = goals - ball_pos
    dist = torch.norm(delta, dim=1)
    direction = delta / (dist.unsqueeze(1) + 1e-6)

    skalarprodukt =torch.sum(ball_vel * direction, dim=1)
    vel_reward = torch.clamp(skalarprodukt, min=0.0, max=max_speed) / max_speed

    return vel_reward

def dribble_goal_pos_b(env: ManagerBasedRlEnv) -> torch.Tensor:

    if dribble_to_goal_pos.goals is None:
        return torch.zeros((env.num_envs, 2), device=env.device)

    robot = env.scene["robot"]
    bas_pos_w = robot.data.root_link_pos_w[:, :2]
    heading_w = robot.data.heading_w

    delta_w = dribble_to_goal_pos.goals - bas_pos_w
    cos_h = torch.cos(heading_w)
    sin_h = torch.sin(heading_w)

    goal_x_b = cos_h * delta_w[:, 0] + sin_h * delta_w[:, 1]
    goal_y_b = -sin_h * delta_w[:, 0] + cos_h * delta_w[:, 1]
    
    return torch.stack([goal_x_b, goal_y_b], dim=1)