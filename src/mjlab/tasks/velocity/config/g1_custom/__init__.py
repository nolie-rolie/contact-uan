from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.utils.g1_variants import has_uan_checkpoint_env
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_custom_env_cfg,
  unitree_g1_custom_uan_env_cfg,
  unitree_g1_custom_uan_eval_env_cfg,
)
from .rl_cfg import unitree_g1_custom_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Custom-Unitree-G1",
  env_cfg=unitree_g1_custom_env_cfg(),
  play_env_cfg=unitree_g1_custom_env_cfg(play=True),
  rl_cfg=unitree_g1_custom_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


if has_uan_checkpoint_env():
  register_mjlab_task(
    task_id="Mjlab-Velocity-Custom-UAN-Unitree-G1",
    env_cfg=unitree_g1_custom_uan_env_cfg(),
    play_env_cfg=unitree_g1_custom_uan_eval_env_cfg(),
    rl_cfg=unitree_g1_custom_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
