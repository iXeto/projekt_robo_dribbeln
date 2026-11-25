from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch
from mjlab.tasks.velocity.mdp.curriculums import VelocityStage

from mjlab_piplus.tasks.mdp.commands import (
  GoalPoseCommandCfg,
  UniformVelocityCommandWithStopCfg,
)
from mjlab_piplus.tasks.mdp.curriculums import SpawnConditionStage
from mjlab_piplus.tasks.mdp.kick import KickCommandCfg
from mjlab_piplus.tasks.mdp.multi_command import MultiCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "EventCurriculumStage",
  "event_curriculum",
  "GoalRadiusStage",
  "commands_goal_radius",
  "RewardWeightStage",
  "reward_weight",
  "reward_param_linear",
  "commands_vel_multi",
  "commands_spawning_conditions",
]


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Ramp a reward term's weight over training steps.

  Later stages take precedence once ``env.common_step_counter`` passes their
  ``step`` threshold. Mirrors the ``reward_weight`` curriculum shipped with some
  mjlab builds, which this project's mjlab version does not include.
  """
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])

def commands_spawning_conditions(
  env: ManagerBasedRlEnv,
  command_name: str,
  placement_cone_degrees: list[SpawnConditionStage],
) -> dict[str, torch.Tensor]:
  
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(KickCommandCfg, command_term.cfg)
  for stage in placement_cone_degrees:
    if env.common_step_counter >= stage["step"]:
      cfg.placement_cone_deg = stage["placement_cone_deg"]
  return {
    "placement_cone_deg": torch.tensor(cfg.placement_cone_deg),
  }

def reward_param_linear(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_names: list[str],
  param_name: str,
  start_step: int,
  end_step: int,
  start_value: float,
  end_value: float,
) -> torch.Tensor:
  """Ramp one param of one or more reward terms linearly over training steps.

  Holds ``start_value`` until ``start_step``, interpolates to ``end_value`` by
  ``end_step``, then holds. Several terms share one schedule because a tolerance
  is usually split across terms that have to agree on it -- the kick's aim
  tolerance lives on both ``kick_impact`` and ``kick_direction``, and letting
  them drift apart would score the same kick two different ways.

  Tightening a tolerance faster than the policy can follow is a known way to
  break a signed reward: once the term goes reliably negative, the behavior it
  was meant to shape stops being worth attempting at all. Give it a long runway.
  """
  del env_ids  # Unused.
  step = env.common_step_counter
  if step <= start_step:
    value = start_value
  elif step >= end_step:
    value = end_value
  else:
    frac = (step - start_step) / max(end_step - start_step, 1)
    value = start_value + frac * (end_value - start_value)
  for name in reward_names:
    env.reward_manager.get_term_cfg(name).params[param_name] = value
  return torch.tensor(value)

class SpawnConditionStage(TypedDict):
  step: int
  placement_cone_degree: float

class GoalRadiusStage(TypedDict):
  step: int
  radius_range: tuple[float, float]


def commands_goal_radius(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  radius_stages: list[GoalRadiusStage],
) -> dict[str, torch.Tensor]:
  """Ramp the goal-pose sampling radius over training steps.

  Later stages take precedence once ``env.common_step_counter`` reaches their
  ``step`` threshold.
  """
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(GoalPoseCommandCfg, command_term.cfg)
  for stage in radius_stages:
    if env.common_step_counter >= stage["step"]:
      cfg.radius_range = stage["radius_range"]
  return {
    "radius_min": torch.tensor(cfg.radius_range[0]),
    "radius_max": torch.tensor(cfg.radius_range[1]),
  }


class _EventCurriculumStageOptional(TypedDict, total=False):
  params: dict[str, Any]


class EventCurriculumStage(_EventCurriculumStageOptional):
  step: int


def event_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  stages: list[EventCurriculumStage],
) -> dict[str, torch.Tensor]:
  """Update an event term's params based on training steps.

  Each stage specifies a ``step`` threshold and a ``params`` dict. When
  ``env.common_step_counter`` reaches a stage's ``step``, the params are
  applied. Later stages take precedence. Params are deep-merged: nested
  dicts are updated recursively rather than replaced wholesale.
  """
  del env_ids
  event_term_cfg = env.event_manager.get_term_cfg(event_name)
  for stage in stages:
    if env.common_step_counter >= stage["step"]:
      if "params" in stage:
        _deep_update(event_term_cfg.params, stage["params"])
  result: dict[str, torch.Tensor] = {}
  for k, v in event_term_cfg.params.items():
    if isinstance(v, (int, float)):
      result[k] = torch.tensor(v)
    elif isinstance(v, tuple) and len(v) == 2:
      result[k + "_min"] = torch.tensor(v[0])
      result[k + "_max"] = torch.tensor(v[1])
    elif isinstance(v, torch.Tensor):
      result[k] = v
  return result


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
  for k, v in source.items():
    if k in target and isinstance(target[k], dict) and isinstance(v, dict):
      _deep_update(target[k], v)
    else:
      target[k] = v


def commands_vel_multi(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  sub_command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  """Ramp the velocity ranges of a ``MultiCommand`` sub-command over training.

  Same behavior as mjlab's ``commands_vel``, but the velocity command lives
  inside a composite command term, so it cannot be reached by name through the
  command manager.
  """
  del env_ids  # Unused.
  multi = env.command_manager.get_term(command_name)
  assert isinstance(multi, MultiCommand), (
    f"Command '{command_name}' is not a MultiCommand."
  )
  cfg = cast(UniformVelocityCommandWithStopCfg, multi.terms[sub_command_name].cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }
