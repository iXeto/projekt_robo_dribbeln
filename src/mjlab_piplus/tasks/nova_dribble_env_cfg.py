"""Nova dribble task, starting point: the PiPlus velocity task unchanged.

Add the ball, observations, rewards and terminations here by editing ``cfg``
before it is returned.

Note: the RL cfg uses ``PiPlusSymmetry`` (12-DoF velocity layout). If the actor
obs or the DoF count changes here, update the symmetry class as well.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from .piplus_velocity_env_cfg import piplus_velocity_env_cfg


def nova_dribble_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = piplus_velocity_env_cfg(play=play)
  return cfg
