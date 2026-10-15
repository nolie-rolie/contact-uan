"""RL config for Unitree G1 UAN rollout training."""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_uan_rollout_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name="mjlab.rl.uan_actor:UANActor",
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=False,
      history_length=3,
      num_blocks=2,
      num_actuators=5,
      use_bias=True,
      final_layer_init_uniform=0.01,
      final_layer_bias_zero=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.3,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 512, 512, 512),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.0,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.95,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="uan_rollout",
    run_name="",
    logger="wandb",
    wandb_project="mjlab-uan-rollout",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=100_000,
    clip_actions=None,
  )
