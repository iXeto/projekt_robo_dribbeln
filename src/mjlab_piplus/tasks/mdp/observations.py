from mjlab_piplus.tasks import chris_dribble_rl_cfg
import torch
from mjlab.envs import ManagerBasedRlEnv

from mjlab_piplus.tasks.mdp.multi_command import stand_mask


def phase(env: ManagerBasedRlEnv, frequency: float, command_name: str) -> torch.Tensor:
  period = 1 / frequency
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  phase = torch.zeros(env.num_envs, 2, device=env.device)
  phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
  phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
  # Freeze the gait (plant feet, zero phase) when the command asks the robot to
  # stand still. This decouples "walk in place" (zero velocity, keep stepping)
  # from "stand still" -- under a MultiCommand the latter is its own command
  # type, otherwise it is the velocity command's stand flag.
  standing = stand_mask(env, command_name)
  phase = torch.where(standing.unsqueeze(1), torch.zeros_like(phase), phase)
  return phase


def goal_pose_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Network-friendly goal-pose observation.

  The goal command term stores ``[dx_b, dy_b, yaw_err]`` (goal in the base
  frame). ``yaw_err`` is a wrapping angle, which is discontinuous at +/-pi and
  hard to learn an equivariant policy for. Here we encode the orientation as
  ``(cos(yaw_err), sin(yaw_err))`` instead, giving a continuous 4-dim obs
  ``[dx_b, dy_b, cos(yaw_err), sin(yaw_err)]``. The command tensor itself stays
  3-dim, so rewards/metrics/gating are unaffected.
  """
  command = env.command_manager.get_command(command_name)
  out = torch.empty(env.num_envs, 4, device=env.device)
  out[:, :2] = command[:, :2]
  out[:, 2] = torch.cos(command[:, 2])
  out[:, 3] = torch.sin(command[:, 2])
  return out


def zeros(env: ManagerBasedRlEnv, shape) -> torch.Tensor:
  return torch.zeros((env.num_envs, *shape), device=env.device)

def relative_ball_position(env: ManagerBasedRlEnv, ball_entity) -> torch.Tensor:
  
    self.ball: Entity = env.scene[cfg.ball_entity_name]
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
  return 