from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from mjlab_piplus.tasks.mdp import PiPlusSymmetry


@dataclass
class PiPlusPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  symmetry_cfg: dict[str, Any] | None = field(
    default_factory=lambda: {
      "data_augmentation_func": PiPlusSymmetry(),
      # NOTE: defaults to False in rsl_rl. Without this, symmetry is enforced
      # only by the soft mirror loss; enabling it appends mirrored samples to
      # every mini-batch so the policy gradient itself drives a symmetric gait.
      "use_data_augmentation": False,
      "use_mirror_loss": False,
      "mirror_loss_coeff": 0.5,
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
    algorithm=PiPlusPpoAlgorithmCfg(
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
    max_iterations=3_200,
  )
