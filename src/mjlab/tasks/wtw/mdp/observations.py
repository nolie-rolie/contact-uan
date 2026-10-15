"""Custom observation functions for WTW environment.

These output joint-level observations in Isaac Lab order by applying the
MuJoCo -> IsaacLab permutation. This is necessary because the frozen WTW
policy was trained with Isaac Lab joint ordering.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from mjlab.envs.mdp import observations as mdp_obs
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ._joint_order import MUJOCO_TO_ISAACLAB

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_MJ_TO_IL: torch.Tensor | None = None
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_command_tensor(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  if command is None:
    raise TypeError(f"Command {command_name!r} is not active.")
  return cast(torch.Tensor, command)


def _get_mj_to_il(device: torch.device) -> torch.Tensor:
  global _MJ_TO_IL
  if _MJ_TO_IL is None or _MJ_TO_IL.device != device:
    _MJ_TO_IL = torch.tensor(MUJOCO_TO_ISAACLAB, dtype=torch.long, device=device)
  return _MJ_TO_IL


def joint_pos_rel_il(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Joint positions relative to default, in Isaac Lab order."""
  raw = mdp_obs.joint_pos_rel(env, asset_cfg=asset_cfg)
  return raw[:, _get_mj_to_il(raw.device)]


def joint_vel_rel_il(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Joint velocities relative to default, in Isaac Lab order."""
  raw = mdp_obs.joint_vel_rel(env, asset_cfg=asset_cfg)
  return raw[:, _get_mj_to_il(raw.device)]


def last_action_il(
  env: ManagerBasedRlEnv,
  action_name: str | None = None,
) -> torch.Tensor:
  """Last raw action in Isaac Lab order."""
  raw = mdp_obs.last_action(env, action_name=action_name)
  return raw[:, _get_mj_to_il(raw.device)]


def root_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Root-link height as a one-column observation."""
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


def gait_phase(
  env: ManagerBasedRlEnv,
  command_name: str,
  freq_range: tuple[float, float],
) -> torch.Tensor:
  """Gait phase: sin(2*pi*freq*t), zeroed when standing.

  Args:
      env: The environment.
      command_name: Name of velocity command (for motion mask).
      freq_range: (min_hz, max_hz) mapping from [-1,1] frequency command.

  Returns:
      (num_envs, 1) gait phase tensor.
  """
  freq = torch.full((env.num_envs,), freq_range[0], device=env.device)
  if "gait_frequency" in env.command_manager.active_terms:
    freq_cmd = _get_command_tensor(env, "gait_frequency")[:, 0]
    freq = (freq_cmd + 1) / 2 * (freq_range[1] - freq_range[0]) + freq_range[0]

  t = env.episode_length_buf * env.step_dt
  phase = torch.sin(2 * math.pi * freq * t)

  vel_cmd = _get_command_tensor(env, command_name)
  motion_mask = vel_cmd.norm(dim=1) > 1e-6
  phase = phase * motion_mask.float()

  return phase.unsqueeze(1)
