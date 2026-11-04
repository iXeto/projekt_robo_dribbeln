from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from mjlab_piplus.tasks.mdp import PiPlusSymmetry
from mjlab_piplus.tasks.mdp.symmetry import (
  JOINT_ORDER_12,
  Mirror,
  ball_state_block,
  prefix_blocks,
  velocity_command_block,
)

class NovaSymmetry(PiPlusSymmetry):
  """Velocity-task mirror plus the 4 ball columns [x, y, vx, vy].

  The layout is built into the class instead of being passed as an argument:
  the train CLI (tyro) rebuilds this object from its defaults, which would
  silently drop an obs_mirror argument and fall back to the 48-dim layout.
  """

  @staticmethod
  def _default_mirror() -> Mirror:
    return Mirror(
      [
        *prefix_blocks(JOINT_ORDER_12),
        velocity_command_block(stand_flag=True),
        ball_state_block(),
      ]
    )

@dataclass
class PiPlusPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  symmetry_cfg: dict[str, Any] | None = field(
    default_factory=lambda: {
      "data_augmentation_func": NovaSymmetry(),
      "use_data_augmentation": False,
      "use_mirror_loss": False,
      "mirror_loss_coeff": 0.5,
    }
  )


def get_nova_dribble_rl_cfg() -> RslRlOnPolicyRunnerCfg:
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
    # Own folder under logs/ and own run name in W&B.
    experiment_name="nova_dribble",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=900, #statt 3_200
  )
