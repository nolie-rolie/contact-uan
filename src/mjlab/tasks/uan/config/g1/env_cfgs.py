"""Unitree G1 UAN rollout environment configs."""

from __future__ import annotations

from mjlab.tasks.uan.uan_rollout_env_cfg import (
  UANRolloutEnvCfg,
  make_uan_rollout_env_cfg,
)


def unitree_g1_uan_rollout_env_cfg(play: bool = False) -> UANRolloutEnvCfg:
  return make_uan_rollout_env_cfg(num_envs=16 if play else 4096)
