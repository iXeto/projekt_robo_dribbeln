from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from mjlab_piplus.tasks.mdp import PiPlusMultiSymmetry
from mjlab_piplus.tasks.chris_dribble_env_cfg import (
  NUM_FOOTSTEP_OBSERVATIONS,
  NUM_PATH_OBSERVATIONS,
)


@dataclass
class ChrisDribblePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  # Left-right symmetry for the multi task. Unlike velocity/path this env
  # actuates the arms (20 DoF) and its command is the MultiCommand composite
  # [one_hot(6) | velocity(3) | path(2N+2) | footstep(8M) | goal_pose(4) |
  # kick(7)], so the mirror is built from that block list -- see symmetry.py.
  # The sub-command sizes are taken from the env cfg's own constants so the two
  # cannot drift apart.
  symmetry_cfg: dict[str, Any] | None = field(
    default_factory=lambda: {
      "data_augmentation_func": PiPlusMultiSymmetry(
        num_path_observations=NUM_PATH_OBSERVATIONS,
        num_footstep_observations=NUM_FOOTSTEP_OBSERVATIONS,
      ),
      "use_data_augmentation": True,
      "use_mirror_loss": True,
      "mirror_loss_coeff": 0.1,
    }
  )


def get_chris_dribble_rl_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=ChrisDribblePpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    wandb_project="mjlab_piplus",
    experiment_name="chris_dribble",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=6_400,
  )
