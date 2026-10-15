"""Baseline and frozen-UAN Unitree G1 jumping-track configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.jumping_track.jumping_track_env_cfg import (
  apply_jumping_track_observations,
)
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.utils.g1_variants import (
  remove_robot_specific_training_dr,
  strip_variant_training_events_for_play,
  swap_in_torque_speed_clipped_action,
  swap_in_uan_torque_speed_clipped_action,
)


def _unitree_g1_jumping_track_base_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Share the reference observations and torque-speed action between variants."""
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  apply_jumping_track_observations(cfg)
  if not play:
    cfg.scene.num_envs = 16_384
    cfg.episode_length_s = 20.0
  swap_in_torque_speed_clipped_action(cfg)
  return cfg


def unitree_g1_jumping_track_baseline_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the torque-speed-clipped jumping-track baseline."""
  cfg = _unitree_g1_jumping_track_base_cfg(play=play)
  if play:
    strip_variant_training_events_for_play(cfg)
  return cfg


def unitree_g1_jumping_track_uan_env_cfg(
  play: bool = False,
  uan_checkpoint_path: str | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create jumping-track with a frozen UAN actuator model."""
  cfg = _unitree_g1_jumping_track_base_cfg(play=play)
  swap_in_uan_torque_speed_clipped_action(cfg, checkpoint_path=uan_checkpoint_path)
  remove_robot_specific_training_dr(cfg)
  if play:
    strip_variant_training_events_for_play(cfg)
  return cfg
