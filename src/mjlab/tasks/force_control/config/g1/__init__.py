from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.utils.g1_variants import has_uan_checkpoint_env

from .env_cfgs import (
  make_force_control_env_cfg,
  make_force_control_eval_env_cfg,
  make_force_control_uan_env_cfg,
  make_force_control_uan_eval_env_cfg,
)
from .rl_cfg import unitree_g1_force_control_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-ForceControl-Unitree-G1",
  env_cfg=make_force_control_env_cfg(),
  play_env_cfg=make_force_control_eval_env_cfg(),
  rl_cfg=unitree_g1_force_control_ppo_runner_cfg(),
)

if has_uan_checkpoint_env():
  register_mjlab_task(
    task_id="Mjlab-ForceControl-UAN-Unitree-G1",
    env_cfg=make_force_control_uan_env_cfg(),
    play_env_cfg=make_force_control_uan_eval_env_cfg(),
    rl_cfg=unitree_g1_force_control_ppo_runner_cfg(),
  )
