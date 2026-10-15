"""Register Unitree G1 jumping-track pretraining tasks."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.utils.g1_variants import has_uan_checkpoint_env

from .env_cfgs import (
  unitree_g1_jumping_track_baseline_env_cfg,
  unitree_g1_jumping_track_uan_env_cfg,
)
from .rl_cfg import unitree_g1_jumping_track_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-JumpingTrack-Baseline-Unitree-G1",
  env_cfg=unitree_g1_jumping_track_baseline_env_cfg(),
  play_env_cfg=unitree_g1_jumping_track_baseline_env_cfg(play=True),
  rl_cfg=unitree_g1_jumping_track_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

if has_uan_checkpoint_env():
  register_mjlab_task(
    task_id="Mjlab-JumpingTrack-UAN-Unitree-G1",
    env_cfg=unitree_g1_jumping_track_uan_env_cfg(),
    play_env_cfg=unitree_g1_jumping_track_uan_env_cfg(play=True),
    rl_cfg=unitree_g1_jumping_track_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
  )
