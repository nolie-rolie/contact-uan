"""Utilities for replay-window UAN rollout tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  build_g1_joint_to_actuator_map,
)
from mjlab.utils.lab_api.math import quat_apply

NUM_JOINTS = 29
STATE_DIM = 96
ACTION_DIM = 29
EXTENSION_DIM = 15
TERMINATION_DIM = 1

UAN_BLOCK_TARGET_POS_ERROR = "target_pos_error"
UAN_BLOCK_JOINT_VEL = "joint_vel"
UAN_OBS_BLOCK_NAMES = (UAN_BLOCK_TARGET_POS_ERROR, UAN_BLOCK_JOINT_VEL)


def rollout_uan_obs_block_names() -> tuple[str, ...]:
  return UAN_OBS_BLOCK_NAMES


def resolve_uan_obs_block_names(metadata: Mapping[str, Any]) -> tuple[str, ...]:
  """Validate the paper observation layout when loading a checkpoint."""
  names = tuple(metadata.get("uan_obs_block_names", UAN_OBS_BLOCK_NAMES))
  if names != UAN_OBS_BLOCK_NAMES or metadata.get("num_blocks", 2) != 2:
    raise ValueError(
      "Contact-UAN requires target-position-error and joint-velocity observations."
    )
  if metadata.get("uan_history_length", 3) != 3:
    raise ValueError("Contact-UAN requires three observation history frames.")
  return names


def resolve_g1_uan_actuator_sharing(
  *,
  joint_names: Sequence[str],
  num_actuators: int = 5,
  joint_to_actuator_map: Sequence[int] | None = None,
) -> tuple[list[int], str, tuple[str, ...]]:
  """Validate the five actuator-family groups used by Contact-UAN."""
  mapping, grouping, group_names = build_g1_joint_to_actuator_map(
    joint_names, num_actuators
  )
  if joint_to_actuator_map is not None and list(joint_to_actuator_map) != mapping:
    raise ValueError("Contact-UAN requires the canonical five actuator-family groups.")
  return mapping, grouping, group_names


def find_state_action_csvs(data_dir: Path) -> list[Path]:
  paths = list(data_dir.glob("**/state_action_data_*.csv"))

  def natural_key(path: Path) -> tuple[int, str]:
    suffix = path.stem.rsplit("_", maxsplit=1)[-1]
    return (int(suffix), path.stem) if suffix.isdigit() else (0, path.stem)

  return sorted(paths, key=natural_key)


def load_replay_array(data_dir: str, max_files: int | None = None) -> np.ndarray:
  if not data_dir.strip():
    raise ValueError(
      "Replay data directory is required. Set --env.commands.replay.data-dir "
      "to your processed replay directory."
    )
  path = Path(data_dir).expanduser().resolve()
  npy_path = path / "all_data.npy"
  if npy_path.exists():
    raw = np.load(npy_path, mmap_mode="r")
    print(
      f"[UANReplay] Memory-mapped {npy_path}: {raw.shape}, "
      f"{raw.nbytes / 1e9:.1f} GB virtual"
    )
    return raw

  csv_paths = find_state_action_csvs(path)
  if max_files is not None:
    csv_paths = csv_paths[:max_files]
  if not csv_paths:
    raise FileNotFoundError(f"No all_data.npy or state_action_data_*.csv in {path}")
  arrays = [pd.read_csv(p, header=None).values.astype(np.float32) for p in csv_paths]
  raw = np.concatenate(arrays, axis=0)
  print(f"[UANReplay] Loaded {len(csv_paths)} CSV files from {path}: {raw.shape}")
  return raw


def split_replay_array(
  raw: np.ndarray,
  *,
  state_dim: int = STATE_DIM,
  action_dim: int = ACTION_DIM,
  extension_dim: int = EXTENSION_DIM,
  termination_dim: int = TERMINATION_DIM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  state, action, ext, contact, term = split_replay_array_with_contact(
    raw,
    state_dim=state_dim,
    action_dim=action_dim,
    extension_dim=extension_dim,
    termination_dim=termination_dim,
  )
  del contact
  return state, action, ext, term


def split_replay_array_with_contact(
  raw: np.ndarray,
  *,
  state_dim: int = STATE_DIM,
  action_dim: int = ACTION_DIM,
  extension_dim: int = EXTENSION_DIM,
  termination_dim: int = TERMINATION_DIM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  total_cols = int(raw.shape[1])
  contact_dim = total_cols - state_dim - action_dim - extension_dim - termination_dim
  if contact_dim < 0:
    raise ValueError(
      f"Replay row width {total_cols} is too small for state={state_dim}, "
      f"action={action_dim}, extension={extension_dim}, termination={termination_dim}."
    )

  col = 0
  state = np.array(raw[:, col : col + state_dim], dtype=np.float32, copy=True)
  col += state_dim
  action = np.array(raw[:, col : col + action_dim], dtype=np.float32, copy=True)
  col += action_dim
  ext = np.array(raw[:, col : col + extension_dim], dtype=np.float32, copy=True)
  col += extension_dim
  contact = np.array(raw[:, col : col + contact_dim], dtype=np.float32, copy=True)
  col += contact_dim
  term = np.array(
    raw[:, col : col + termination_dim], dtype=np.float32, copy=True
  ).reshape(-1)
  return state, action, ext, contact, term


def build_valid_starts(
  term: np.ndarray,
  *,
  history_horizon: int,
  rollout_horizon: int,
  contact: np.ndarray | None = None,
  require_contact_at_reset: bool = False,
  contact_threshold: float = 0.8,
) -> np.ndarray:
  """Return pre-action transition starts with history and rollout inside an episode.

  Release replay datasets use a fixed convention: row ``i`` contains the state
  before action ``i`` is applied, and the rollout target is row ``i + 1``.
  """
  terminated = np.asarray(term > 0.5, dtype=np.int64)
  target_row_offset = 1
  reset_row_offset = 0
  window_size = history_horizon + rollout_horizon + target_row_offset
  n_valid = len(terminated) - window_size + 1
  if n_valid <= 0:
    raise ValueError(
      f"Replay data is too short for history={history_horizon} and "
      f"rollout_horizon={rollout_horizon}."
    )
  cumsum = np.concatenate([[0], np.cumsum(terminated)])
  term_count = cumsum[window_size : n_valid + window_size] - cumsum[:n_valid]
  window_starts = np.where(term_count == 0)[0].astype(np.int64)
  if require_contact_at_reset:
    if contact is None:
      raise ValueError("require_contact_at_reset=True requires replay contact columns.")
    contact = np.asarray(contact, dtype=np.float32)
    if contact.ndim != 2 or contact.shape[1] < 2:
      raise ValueError(
        "require_contact_at_reset=True expects contact columns "
        "[left_foot_contact, right_foot_contact, ...]."
      )
    reset_rows = window_starts + history_horizon + reset_row_offset
    contact_ok = (contact[reset_rows, 0:2] >= float(contact_threshold)).any(axis=1)
    window_starts = window_starts[contact_ok]
  if window_starts.size == 0:
    suffix = " with reset contact" if require_contact_at_reset else ""
    raise ValueError(f"No valid replay windows found{suffix}.")
  return window_starts + history_horizon


def env_ids_tensor(env, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
  if env_ids is None or isinstance(env_ids, slice):
    return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
  return env_ids.to(env.device, dtype=torch.long)


def recover_quat_from_projected_gravity(
  projected_gravity: torch.Tensor,
) -> torch.Tensor:
  """Recover a yaw-free world quaternion from recorded projected gravity."""
  g_b = projected_gravity / (projected_gravity.norm(dim=-1, keepdim=True) + 1.0e-8)
  target = torch.tensor([0.0, 0.0, -1.0], device=g_b.device).expand_as(g_b)
  dot = (g_b * target).sum(dim=-1, keepdim=True)
  cross = torch.cross(g_b, target, dim=-1)

  anti_mask = dot.squeeze(-1) < -0.999
  if anti_mask.any():
    perp = torch.zeros_like(g_b)
    perp[:, 0] = 1.0
    bad = g_b[:, 0].abs() > 0.9
    perp[bad, 0] = 0.0
    perp[bad, 1] = 1.0
    perp = perp - g_b * (perp * g_b).sum(dim=-1, keepdim=True)
    perp = perp / (perp.norm(dim=-1, keepdim=True) + 1.0e-8)
    cross[anti_mask] = perp[anti_mask]
    dot[anti_mask] = 0.0

  quat = torch.cat([1.0 + dot, cross], dim=-1)
  return quat / (quat.norm(dim=-1, keepdim=True) + 1.0e-8)


def write_recorded_state(
  env,
  robot,
  *,
  state_raw: torch.Tensor,
  ext_raw: torch.Tensor,
  row_ids: torch.Tensor,
  env_ids: torch.Tensor,
  joint_ids: torch.Tensor,
  default_base_height: float,
  init_height_mode: str = "lowest_foot",
  foot_body_ids: list[int] | None = None,
) -> None:
  state = state_raw[row_ids]
  lin_vel_b = state[:, 0:3]
  ang_vel_b = state[:, 3:6]
  projected_gravity = state[:, 6:9]
  joint_pos = state[:, 9:38]
  joint_vel = state[:, 38:67]

  quat_wxyz = recover_quat_from_projected_gravity(projected_gravity)
  lin_vel_w = quat_apply(quat_wxyz, lin_vel_b)
  ang_vel_w = quat_apply(quat_wxyz, ang_vel_b)

  root_pose = torch.zeros(env_ids.numel(), 7, device=env.device)
  if ext_raw.numel() > 0 and ext_raw.shape[1] > 4:
    root_pose[:, 2] = ext_raw[row_ids, 4]
  else:
    root_pose[:, 2] = default_base_height
  root_pose[:, 3:7] = quat_wxyz
  root_vel = torch.cat([lin_vel_w, ang_vel_w], dim=-1)

  robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
  robot.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)
  robot.write_joint_state_to_sim(
    joint_pos,
    joint_vel,
    joint_ids=joint_ids,
    env_ids=env_ids,
  )

  if init_height_mode == "root_height":
    return
  if init_height_mode != "lowest_foot":
    raise ValueError(
      "Unknown UAN replay init_height_mode "
      f"{init_height_mode!r}; expected 'root_height' or 'lowest_foot'."
    )
  if foot_body_ids is None or len(foot_body_ids) != 2:
    raise ValueError(
      "init_height_mode='lowest_foot' requires exactly two foot body ids."
    )
  if ext_raw.numel() == 0 or ext_raw.shape[1] < 2:
    raise ValueError(
      "init_height_mode='lowest_foot' requires recorded foot heights in ext[0:2]."
    )

  env.scene.write_data_to_sim()
  env.sim.forward()

  foot_ids = torch.as_tensor(foot_body_ids, dtype=torch.long, device=env.device)
  recorded_foot_z = ext_raw[row_ids, 0:2]
  simulated_foot_z = robot.data.body_link_pos_w[env_ids[:, None], foot_ids, 2]
  lowest_recorded_idx = torch.argmin(recorded_foot_z, dim=1, keepdim=True)
  target_foot_z = torch.gather(recorded_foot_z, 1, lowest_recorded_idx).squeeze(1)
  current_foot_z = torch.gather(simulated_foot_z, 1, lowest_recorded_idx).squeeze(1)

  adjusted_root_pose = torch.cat(
    [robot.data.root_link_pos_w[env_ids], robot.data.root_link_quat_w[env_ids]],
    dim=-1,
  )
  adjusted_root_pose[:, 2] += target_foot_z - current_foot_z
  robot.write_root_link_pose_to_sim(adjusted_root_pose, env_ids=env_ids)
  robot.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)
