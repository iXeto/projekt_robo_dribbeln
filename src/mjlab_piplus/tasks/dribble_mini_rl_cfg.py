from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

# Smoke test length: enough to see whether the reward moves and to render a
# checkpoint, not enough for a good policy.
MAX_ITERATIONS = 300


def dribble_mini_rl_cfg() -> RslRlOnPolicyRunnerCfg:
  # Same PPO setup as the PiPlus velocity task, without its symmetry
  # augmentation: the mirror there does not know about the ball observations.
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
    algorithm=RslRlPpoAlgorithmCfg(
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
    # TensorBoard so the test runs without a wandb login.
    logger="tensorboard",
    upload_model=False,
    experiment_name="piplus_dribble_mini",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=MAX_ITERATIONS,
  )
