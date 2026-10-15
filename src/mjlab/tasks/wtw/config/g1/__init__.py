from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.wtw.rl import WtwOnPolicyRunner

from .env_cfgs import (
  make_wtw_env_cfg,
  make_wtw_eval_env_cfg,
)
from .rl_cfg import unitree_g1_wtw_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-WTW-Unitree-G1",
  env_cfg=make_wtw_env_cfg(),
  play_env_cfg=make_wtw_eval_env_cfg(),
  rl_cfg=unitree_g1_wtw_ppo_runner_cfg(),
  runner_cls=WtwOnPolicyRunner,
)
