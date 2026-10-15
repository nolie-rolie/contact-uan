"""Observation terms for UAN replay rollout tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from .actions import ReplayResidualJointPositionAction
from .replay_command import ReplayWindowCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def replay_target_pos_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  action_name: str,
) -> torch.Tensor:
  command = cast(ReplayWindowCommand, env.command_manager.get_term(command_name))
  action = cast(
    ReplayResidualJointPositionAction,
    env.action_manager.get_term(action_name),
  )
  joint_pos = command.robot.data.joint_pos[:, action.target_ids]
  return command.current_target - joint_pos


def recorded_target(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(ReplayWindowCommand, env.command_manager.get_term(command_name))
  return command.current_target


def recorded_joint_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(ReplayWindowCommand, env.command_manager.get_term(command_name))
  return command.state_raw[command.current_action_state_row, 38:67]


def root_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Root-link height as a one-column observation."""
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


def body_pos_b(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Selected body positions in the root-link frame."""
  asset = env.scene[asset_cfg.name]
  body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
  rel_pos_w = body_pos_w - asset.data.root_link_pos_w[:, None, :]
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, rel_pos_w.shape[1], -1
  )
  body_pos_b = quat_apply_inverse(
    root_quat_w.reshape(-1, 4),
    rel_pos_w.reshape(-1, 3),
  )
  return body_pos_b.reshape(env.num_envs, -1)


def prime_actor_replay_history_buffers(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str = "replay",
  group_name: str = "actor",
  target_pos_error_term: str = "target_pos_error",
  joint_vel_term: str = "joint_vel",
  joint_pos_rel_term: str = "joint_pos_rel",
  joint_vel_scale: float | torch.Tensor | None = None,
) -> None:
  """Prime actor history buffers from recorded replay rows after reset.

  MJLab clears observation history buffers inside ``observation_manager.reset()``.
  ``ReplayWindowCommand.reset()`` runs immediately after that and before the first
  post-reset ``compute(update_history=True)``, so this helper fills the two UAN
  actor history buffers with the replay frames that should precede the first
  live observation append. ``joint_pos_rel`` is primed when that optional third
  actor term is present.
  """
  command = cast(ReplayWindowCommand, env.command_manager.get_term(command_name))
  history_buffers = env.observation_manager._group_obs_term_history_buffer
  if group_name not in history_buffers:
    raise RuntimeError(f"Observation group '{group_name}' has no history buffers.")
  group_buffers = history_buffers[group_name]
  required_terms = (target_pos_error_term, joint_vel_term)
  missing = [name for name in required_terms if name not in group_buffers]
  if missing:
    raise RuntimeError(
      f"Cannot prime UAN actor history; missing history buffer(s): {missing}."
    )

  target_buffer = group_buffers[target_pos_error_term]
  vel_buffer = group_buffers[joint_vel_term]
  optional_joint_pos_rel_buffer = group_buffers.get(joint_pos_rel_term)
  history_lengths = {
    target_pos_error_term: int(target_buffer.max_length),
    joint_vel_term: int(vel_buffer.max_length),
  }
  if optional_joint_pos_rel_buffer is not None:
    history_lengths[joint_pos_rel_term] = int(optional_joint_pos_rel_buffer.max_length)
  if len(set(history_lengths.values())) != 1:
    raise RuntimeError(
      f"UAN actor history lengths differ across terms: {history_lengths}."
    )

  history_length = int(target_buffer.max_length)
  if history_length < 1:
    raise RuntimeError("UAN actor history length must be at least 1.")
  if history_length == 1:
    return

  cursor = command.cursor[env_ids]
  first_k = cursor - history_length + 1
  prior_ks = (
    first_k[:, None]
    + torch.arange(history_length - 1, device=env.device, dtype=torch.long)[None, :]
  )
  prior_state_rows = command.action_state_row_ids(prior_ks)
  if torch.any(prior_state_rows < 0):
    raise RuntimeError(
      "Cannot prime UAN actor history: replay cursor does not have enough "
      f"previous rows for history_length={history_length}."
    )

  prior_targets = command.action_raw[prior_ks]
  prior_joint_pos = command.state_raw[prior_state_rows, 9:38]
  prior_joint_vel = command.state_raw[prior_state_rows, 38:67]
  prior_target_error = prior_targets - prior_joint_pos
  default_joint_pos = command.robot.data.default_joint_pos
  if default_joint_pos.shape[-1] != prior_joint_pos.shape[-1]:
    raise RuntimeError(
      "Replay default joint position width does not match replay joint positions: "
      f"{default_joint_pos.shape[-1]} vs {prior_joint_pos.shape[-1]}."
    )
  prior_joint_pos_rel = prior_joint_pos - default_joint_pos[env_ids][:, None, :]

  if joint_vel_scale is None:
    joint_vel_scale = cast(
      float | torch.Tensor | None,
      env.observation_manager.get_term_cfg(group_name, joint_vel_term).scale,
    )
  if joint_vel_scale is None:
    prior_joint_vel_scaled = prior_joint_vel
  else:
    prior_joint_vel_scaled = prior_joint_vel * joint_vel_scale
  _prime_circular_buffer_for_next_append(target_buffer, env_ids, prior_target_error)
  _prime_circular_buffer_for_next_append(vel_buffer, env_ids, prior_joint_vel_scaled)
  if optional_joint_pos_rel_buffer is not None:
    _prime_circular_buffer_for_next_append(
      optional_joint_pos_rel_buffer,
      env_ids,
      prior_joint_pos_rel,
    )


def _prime_circular_buffer_for_next_append(
  circular_buffer,
  env_ids: torch.Tensor,
  prior_values: torch.Tensor,
) -> None:
  """Fill selected rows so the next append completes the desired history.

  ``prior_values`` has shape ``[num_envs_to_reset, history_length - 1, obs_dim]``.
  The next observation-manager compute appends the current frame into the global
  circular-buffer write slot. We therefore place these prior frames in the
  chronological slots that will precede that next write, without changing the
  global pointer and disturbing non-reset env rows.
  """
  history_length = int(circular_buffer.max_length)
  if prior_values.shape[1] != max(history_length - 1, 0):
    raise RuntimeError(
      f"Expected {history_length - 1} prior frames, got {prior_values.shape[1]}."
    )

  obs_shape = tuple(prior_values.shape[2:])
  if circular_buffer._buffer is None:
    circular_buffer._buffer = torch.zeros(
      (history_length, circular_buffer.batch_size, *obs_shape),
      dtype=prior_values.dtype,
      device=circular_buffer.device,
    )
  elif tuple(circular_buffer._buffer.shape[2:]) != obs_shape:
    raise RuntimeError(
      "Existing observation history buffer has incompatible shape: "
      f"{tuple(circular_buffer._buffer.shape[2:])} vs {obs_shape}."
    )

  next_write_idx = (int(circular_buffer._pointer) + 1) % history_length
  chronological_start_after_append = (next_write_idx + 1) % history_length

  if history_length > 1:
    for hist_idx in range(history_length - 1):
      slot = (chronological_start_after_append + hist_idx) % history_length
      circular_buffer._buffer[slot, env_ids] = prior_values[:, hist_idx]
    circular_buffer._buffer[next_write_idx, env_ids] = prior_values[:, -1]

  circular_buffer._num_pushes[env_ids] = max(history_length - 1, 0)
