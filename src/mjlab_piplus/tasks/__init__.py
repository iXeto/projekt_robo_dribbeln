from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner

from .chris_dribble_env_cfg import chris_dribble_env_cfg
from .chris_dribble_rl_cfg import get_chris_dribble_rl_cfg
from .dribble_mini_env_cfg import dribble_mini_env_cfg
from .dribble_mini_rl_cfg import dribble_mini_rl_cfg
from .piplus_multi_env_cfg import piplus_multi_env_cfg
from .piplus_multi_rl_cfg import get_piplus_multi_rl_cfg
from .piplus_path_env_cfg import piplus_path_env_cfg
from .piplus_path_rl_cfg import get_piplus_path_rl_cfg
from .piplus_standup_imitation_env_cfg import piplus_standup_imitation_env_cfg
from .piplus_standup_imitation_rl_cfg import get_piplus_standup_imitation_rl_cfg
from .piplus_velocity_env_cfg import piplus_velocity_env_cfg
from .piplus_velocity_rl_cfg import get_piplus_velocity_rl_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-PiPlus",
  env_cfg=piplus_velocity_env_cfg(),
  play_env_cfg=piplus_velocity_env_cfg(play=True),
  rl_cfg=get_piplus_velocity_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Path-PiPlus",
  env_cfg=piplus_path_env_cfg(),
  play_env_cfg=piplus_path_env_cfg(play=True),
  rl_cfg=get_piplus_path_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
# One policy driven by a velocity, path, or stand command, switching between
# them during the episode.
register_mjlab_task(
  task_id="Mjlab-Multi-PiPlus",
  env_cfg=piplus_multi_env_cfg(),
  play_env_cfg=piplus_multi_env_cfg(play=True),
  rl_cfg=get_piplus_multi_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
# Validation task for the getup imitation bridge: no commands, no shaping, no
# penalties -- the imitation reward is the entire objective, so the return is a
# direct readout of how well the student tracks the teacher.
register_mjlab_task(
  task_id="Mjlab-StandupImitation-PiPlus",
  env_cfg=piplus_standup_imitation_env_cfg(),
  play_env_cfg=piplus_standup_imitation_env_cfg(play=True),
  rl_cfg=get_piplus_standup_imitation_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
# Smoke test for dribbling: the velocity task plus a ball to keep in front of
# the feet and move at the commanded velocity.
register_mjlab_task(
  task_id="Mjlab-DribbleMini-PiPlus",
  env_cfg=dribble_mini_env_cfg(),
  play_env_cfg=dribble_mini_env_cfg(play=True),
  rl_cfg=dribble_mini_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
# Dribble task built step by step from a copy of the velocity task.
register_mjlab_task(
  task_id="Mjlab-ChrisDribble-PiPlus",
  env_cfg=chris_dribble_env_cfg(),
  play_env_cfg=chris_dribble_env_cfg(play=True),
  rl_cfg=get_chris_dribble_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
