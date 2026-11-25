from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab_piplus.tasks.mdp.kick import KickCommand
from mjlab_piplus.tasks.mdp.multi_command import resolve_command_term

#NEW

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab_piplus.tasks.mdp.path_following.path_command import PathCommand
from mjlab_piplus.tasks.mdp.path_following.path_gen_astar import (OBSTACLE_KIND_CYLINDER,)
from mjlab_piplus.tasks.mdp.path_following.utils import heading_from_quat

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

#NEW

__all__ = ["ball_lost", "dribble_ball_proximity", "dribble_ball_velocity", "dribble_robot_velocity", "dribble_idle", "dribble_path_progress","dribble_close_to_path","dribble_goal_reached","dribble_obstacle_penalty", "nearest_obstacle_obs_b",]

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


#NEW

def _path(env: ManagerBasedRlEnv, command_name: str = "path") -> PathCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, PathCommand)
  return term


def _ball_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["ball"].data.root_link_pos_w[:, :2]


class dribble_path_progress:

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.prev_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.last_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._was_reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      self._was_reset[:] = True
    else:
      self._was_reset[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str = "path",
    max_speed: float = 0.8,
  ) -> torch.Tensor:
    cmd = _path(env, command_name) 
    ball_xy = _ball_xy(env)

    dists = torch.norm(cmd.paths[..., :2] - ball_xy.unsqueeze(1), dim=-1)
    closest_idx = torch.argmin(dists, dim=-1)

    is_new = (cmd.command_counter != self.last_counter) | self._was_reset
    self.prev_idx[is_new] = closest_idx[is_new]
    self.last_counter = cmd.command_counter.clone()
    self._was_reset[:] = False

    progress_idx = closest_idx - self.prev_idx
    speed = progress_idx.float() * cmd.cfg.point_spacing / env.step_dt
    self.prev_idx = closest_idx.clone()

    return torch.clamp(speed, max=max_speed)


def dribble_close_to_path(
  env: ManagerBasedRlEnv,
  command_name: str = "path",
  std: float = 0.35,
) -> torch.Tensor:
  """Belohnt, wenn der BALL nah an der A*-Pfadlinie bleibt."""
  cmd = _path(env, command_name)
  ball_xy = _ball_xy(env)
  dists = torch.norm(cmd.paths[..., :2] - ball_xy.unsqueeze(1), dim=-1)
  min_dist = torch.min(dists, dim=-1).values
  return torch.exp(-torch.square(min_dist) / std**2)


def dribble_goal_reached(
  env: ManagerBasedRlEnv,
  command_name: str = "path",
  pos_std: float = 0.25,
) -> torch.Tensor:
  """Belohnt, wenn der BALL den letzten Wegpunkt (Ziel) erreicht."""
  cmd = _path(env, command_name)
  ball_xy = _ball_xy(env)
  dist = torch.norm(cmd.goal_pos_w - ball_xy, dim=-1)
  return torch.exp(-torch.square(dist) / pos_std**2)


def _obstacle_sdf_gpu(
  pts_xy: torch.Tensor, obstacles: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Vektorisierte GPU-SDF zu allen aktiven Hindernissen (B, M, 7).

  Gibt (min_dist (B,), nearest_vec_w (B, 2)) zurück.
  """
  kind = obstacles[..., 0]  # (B, M)
  ox = obstacles[..., 1]
  oy = obstacles[..., 2]
  yaw = obstacles[..., 3]
  a = obstacles[..., 4]
  b = obstacles[..., 5]

  dx = pts_xy[:, 0:1] - ox  # (B, M)
  dy = pts_xy[:, 1:2] - oy  # (B, M)

  # Zylinder-Abstand
  dist_cyl = torch.hypot(dx, dy) - a

  # Rotierter-Quader-Abstand (identisch zu path_gen_astar._obstacle_distance)
  cos_y = torch.cos(yaw)
  sin_y = torch.sin(yaw)
  local_x = torch.abs(cos_y * dx + sin_y * dy) - a
  local_y = torch.abs(-sin_y * dx + cos_y * dy) - b
  outside = torch.hypot(
    torch.clamp(local_x, min=0.0), torch.clamp(local_y, min=0.0)
  )
  inside = torch.clamp(torch.maximum(local_x, local_y), max=0.0)
  dist_box = outside + inside

  is_cyl = kind == OBSTACLE_KIND_CYLINDER
  is_valid = kind >= 0.0
  dist = torch.where(is_cyl, dist_cyl, dist_box)
  dist = torch.where(is_valid, dist, torch.full_like(dist, float("inf")))

  min_dist, min_idx = torch.min(dist, dim=1)  # (B,)
  batch_idx = torch.arange(pts_xy.shape[0], device=pts_xy.device)
  nearest_ox = ox[batch_idx, min_idx]
  nearest_oy = oy[batch_idx, min_idx]
  vec_to_obs_w = torch.stack([nearest_ox - pts_xy[:, 0], nearest_oy - pts_xy[:, 1]], dim=1)
  return min_dist, vec_to_obs_w


def dribble_obstacle_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "path",
  robot_margin: float = 0.22,
  ball_margin: float = 0.11,
) -> torch.Tensor:
  """Bestraft (positiver Cost -> mit negativem Gewicht nutzen), sobald Roboter
  oder Ball ein virtuelles Hindernis berühren oder dessen Sicherheitsradius
  unterschreiten."""
  cmd = _path(env, command_name)
  robot_xy = env.scene["robot"].data.root_link_pos_w[:, :2]
  ball_xy = _ball_xy(env)

  robot_dist, _ = _obstacle_sdf_gpu(robot_xy, cmd.obstacles)
  ball_dist, _ = _obstacle_sdf_gpu(ball_xy, cmd.obstacles)

  robot_pen = torch.clamp(robot_margin - robot_dist, min=0.0)
  ball_pen = torch.clamp(ball_margin - ball_dist, min=0.0)
  # Berührung + Eindringtiefe bestrafen
  touched = ((robot_dist < robot_margin) | (ball_dist < ball_margin)).float()
  return touched + 5.0 * (robot_pen + ball_pen)


def nearest_obstacle_obs_b(
  env: ManagerBasedRlEnv,
  command_name: str = "path",
  max_range: float = 2.5,
) -> torch.Tensor:
  """Observation [dx_b, dy_b, surface_dist] zum nächsten Hindernis im Base-Frame."""
  cmd = _path(env, command_name)
  robot = env.scene["robot"]
  robot_xy = robot.data.root_link_pos_w[:, :2]
  heading = heading_from_quat(robot.data.root_link_quat_w)

  min_dist, vec_w = _obstacle_sdf_gpu(robot_xy, cmd.obstacles)
  # Falls gar kein Hindernis existiert (inf), auf max_range setzen
  no_obs = torch.isinf(min_dist)
  vec_w = torch.where(no_obs.unsqueeze(1), torch.full_like(vec_w, max_range), vec_w)
  surface_dist = torch.clamp(min_dist, min=-1.0, max=max_range)

  cos_h = torch.cos(heading)
  sin_h = torch.sin(heading)
  dx_b = torch.clamp(cos_h * vec_w[:, 0] + sin_h * vec_w[:, 1], -max_range, max_range)
  dy_b = torch.clamp(-sin_h * vec_w[:, 0] + cos_h * vec_w[:, 1], -max_range, max_range)
  return torch.stack([dx_b, dy_b, surface_dist], dim=1)