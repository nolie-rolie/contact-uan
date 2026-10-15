"""Register fixed-objective Unitree G1 JumpingHeight tasks."""

from mjlab.tasks.jumping_track.config.g1.rl_cfg import (
  unitree_g1_jumping_track_ppo_runner_cfg,
)
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.utils.g1_variants import has_uan_checkpoint_env

from .env_cfgs import (
  unitree_g1_jumping_height_baseline_env_cfg,
  unitree_g1_jumping_height_uan_env_cfg,
)

register_mjlab_task(
  task_id="Mjlab-JumpingHeight-Baseline-Unitree-G1",
  env_cfg=unitree_g1_jumping_height_baseline_env_cfg(),
  play_env_cfg=unitree_g1_jumping_height_baseline_env_cfg(play=True),
  rl_cfg=unitree_g1_jumping_track_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

if has_uan_checkpoint_env():
  register_mjlab_task(
    task_id="Mjlab-JumpingHeight-UAN-Unitree-G1",
    env_cfg=unitree_g1_jumping_height_uan_env_cfg(),
    play_env_cfg=unitree_g1_jumping_height_uan_env_cfg(play=True),
    rl_cfg=unitree_g1_jumping_track_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
  )
