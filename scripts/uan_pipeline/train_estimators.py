"""Train the six independent Contact-UAN state estimators.

The default command trains the paper configuration: three frames of proprioception,
a base-velocity model, a contact classifier, and one stance-height and orientation
model per foot. Architectures are fixed to the paper setup. Use --target_group to
retrain an individual component or a left/right stance pair.

Usage (from repo root):
    uv run python scripts/uan_pipeline/train_estimators.py \
        --data estimator_training_data.pt \
        --output_dir logs/estimators/contact_uan
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import wandb

if not __package__:
  sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.uan_pipeline.estimator_config import (
  COMPONENT_HIDDEN_DIMS,
  COMPONENT_SUBDIRS,
  COMPONENT_TARGETS,
  HISTORY_LENGTH,
  PROPRIO_ACTION_REPRESENTATION,
  PROPRIO_DIM,
  PROPRIO_TERMS,
)

PAIR_COMPONENTS = {
  "stance_foot_height": ("left_stance_foot_height", "right_stance_foot_height"),
  "stance_foot_projected_gravity": (
    "left_stance_foot_projected_gravity",
    "right_stance_foot_projected_gravity",
  ),
}
BUNDLE_MANIFEST = "paper_estimators_meta.json"


def _wandb_summary() -> Any:
  if wandb.run is None:
    raise RuntimeError("W&B summary requested before wandb.init().")
  return cast(Any, wandb.run.summary)


def _repo_root() -> str:
  return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _repo_relative(path: str) -> str | None:
  repo_root = _repo_root()
  path_abs = os.path.abspath(path)
  try:
    if os.path.commonpath([repo_root, path_abs]) != repo_root:
      return None
  except ValueError:
    return None
  return os.path.relpath(path_abs, repo_root)


class EstimatorWithNorm(nn.Module):
  def __init__(self, net, in_mean, in_std, out_mean, out_std):
    super().__init__()
    self.net = net
    self.register_buffer("in_mean", in_mean)
    self.register_buffer("in_std", in_std)
    self.register_buffer("out_mean", out_mean)
    self.register_buffer("out_std", out_std)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    in_mean = cast(torch.Tensor, self.in_mean)
    in_std = cast(torch.Tensor, self.in_std)
    out_mean = cast(torch.Tensor, self.out_mean)
    out_std = cast(torch.Tensor, self.out_std)
    x_norm = (x - in_mean) / in_std
    y_norm = self.net(x_norm)
    return y_norm * out_std + out_mean


def _save_jit(model, input_mean, input_std, target_mean, target_std, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  wrapped = EstimatorWithNorm(
    model.cpu(), input_mean, input_std, target_mean, target_std
  )
  wrapped.eval()
  dummy = torch.randn(1, input_mean.shape[0])
  jit_model = cast(torch.jit.ScriptModule, torch.jit.trace(wrapped, dummy))
  jit_path = os.path.join(output_dir, "state_estimator.jit")
  jit_model.save(jit_path)
  return jit_path


def parse_args(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(description="Train state estimator MLP.")
  parser.add_argument(
    "--data",
    type=str,
    nargs="+",
    required=True,
    help="One or more estimator_training_data.pt metadata files.",
  )
  parser.add_argument("--epochs", type=int, default=200)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--batch_size", type=int, default=4096)
  parser.add_argument("--val_fraction", type=float, default=0.1)
  parser.add_argument("--device", type=str, default="cuda:0")
  parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory to save estimator JIT and metadata.",
  )
  parser.add_argument(
    "--wandb_project",
    type=str,
    default="g1-state-estimator",
    help="W&B project name. Disabled if not set.",
  )
  parser.add_argument("--run_name", type=str, default=None, help="W&B run name.")
  parser.add_argument(
    "--target_group",
    choices=("paper", *COMPONENT_TARGETS, *PAIR_COMPONENTS),
    default="paper",
    help=(
      "'paper' trains all six independent estimators into their standard bundle "
      "subdirectories. A component writes directly to --output_dir; a stance pair "
      "writes left/right component subdirectories."
    ),
  )
  parser.add_argument(
    "--drop_failed_episodes",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
      "Drop every sample from episodes that ended in a non-timeout termination. "
      "Requires datasets collected with terminated.npy and episode_ids.npy."
    ),
  )
  return parser.parse_args(argv)


def build_mlp(input_dim, output_dim, hidden_dims, activation=nn.ELU):
  layers = []
  prev = input_dim
  for dim in hidden_dims:
    layers.append(nn.Linear(prev, dim))
    layers.append(activation())
    prev = dim
  layers.append(nn.Linear(prev, output_dim))
  return nn.Sequential(*layers)


BASE_STATE_TARGET_NAMES = [
  "lin_vel_x",
  "lin_vel_y",
  "lin_vel_z",
  "ang_vel_x",
  "ang_vel_y",
  "ang_vel_z",
  "root_height",
]
FOOT_HEIGHT_TARGET_NAMES = ["left_foot_height", "right_foot_height"]
STATE_TARGET_NAMES = BASE_STATE_TARGET_NAMES + FOOT_HEIGHT_TARGET_NAMES
FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
FOOT_HEIGHT_TARGET = "body_link_world_z"
PROJECTED_GRAVITY_TARGET_NAMES = ["proj_grav_x", "proj_grav_y", "proj_grav_z"]
JOINT_VEL_TARGET_NAMES = [f"joint_vel_{i:02d}" for i in range(29)]
FOOT_CONTACT_TARGET_NAMES = ["left_foot_contact", "right_foot_contact"]
FOOT_LOG_LOAD_TARGET_NAMES = ["left_foot_log_load", "right_foot_log_load"]
FOOT_CONTACT_LOAD_TARGET_NAMES = FOOT_CONTACT_TARGET_NAMES + FOOT_LOG_LOAD_TARGET_NAMES
STANCE_FOOT_PROJECTED_GRAVITY_TARGET_NAMES = [
  "left_foot_proj_grav_x",
  "left_foot_proj_grav_y",
  "left_foot_proj_grav_z",
  "right_foot_proj_grav_x",
  "right_foot_proj_grav_y",
  "right_foot_proj_grav_z",
]
LEFT_STANCE_FOOT_PROJECTED_GRAVITY_TARGET_NAMES = (
  STANCE_FOOT_PROJECTED_GRAVITY_TARGET_NAMES[:3]
)
RIGHT_STANCE_FOOT_PROJECTED_GRAVITY_TARGET_NAMES = (
  STANCE_FOOT_PROJECTED_GRAVITY_TARGET_NAMES[3:6]
)
STANCE_FOOT_PROJECTED_GRAVITY_TARGET = "body_link_projected_gravity"
TARGET_GROUP_NAMES = COMPONENT_TARGETS
EXPECTED_PROPRIO_TERMS = list(PROPRIO_TERMS)
EXPECTED_PROPRIO_ACTION_REPRESENTATION = PROPRIO_ACTION_REPRESENTATION


def _infer_state_target_layout(target_width: int) -> list[str]:
  if target_width == len(BASE_STATE_TARGET_NAMES):
    return list(BASE_STATE_TARGET_NAMES)
  if target_width == len(STATE_TARGET_NAMES):
    return list(STATE_TARGET_NAMES)
  contact_state_names = STATE_TARGET_NAMES + FOOT_CONTACT_LOAD_TARGET_NAMES
  if target_width == len(contact_state_names):
    return list(contact_state_names)
  denoise_base_names = (
    BASE_STATE_TARGET_NAMES + PROJECTED_GRAVITY_TARGET_NAMES + JOINT_VEL_TARGET_NAMES
  )
  if target_width == len(denoise_base_names):
    return list(denoise_base_names)
  denoise_state_names = (
    STATE_TARGET_NAMES + PROJECTED_GRAVITY_TARGET_NAMES + JOINT_VEL_TARGET_NAMES
  )
  if target_width == len(denoise_state_names):
    return list(denoise_state_names)
  denoise_contact_state_names = (
    STATE_TARGET_NAMES
    + PROJECTED_GRAVITY_TARGET_NAMES
    + JOINT_VEL_TARGET_NAMES
    + FOOT_CONTACT_LOAD_TARGET_NAMES
  )
  if target_width == len(denoise_contact_state_names):
    return list(denoise_contact_state_names)
  for base in (
    BASE_STATE_TARGET_NAMES,
    STATE_TARGET_NAMES,
    STATE_TARGET_NAMES + FOOT_CONTACT_LOAD_TARGET_NAMES,
    STATE_TARGET_NAMES + PROJECTED_GRAVITY_TARGET_NAMES + JOINT_VEL_TARGET_NAMES,
    denoise_contact_state_names,
  ):
    names = list(base) + STANCE_FOOT_PROJECTED_GRAVITY_TARGET_NAMES
    if target_width == len(names):
      return names
  raise ValueError(
    f"Cannot infer target_layout for target width {target_width}. "
    "Recollect with a collector that writes target_layout metadata."
  )


def _has_foot_height_targets(target_layout: list[str]) -> bool:
  return any(name in target_layout for name in FOOT_HEIGHT_TARGET_NAMES)


def _has_projected_gravity_targets(target_layout: list[str]) -> bool:
  return all(name in target_layout for name in PROJECTED_GRAVITY_TARGET_NAMES)


def _has_joint_vel_targets(target_layout: list[str]) -> bool:
  return all(name in target_layout for name in JOINT_VEL_TARGET_NAMES)


def _has_foot_contact_targets(target_layout: list[str]) -> bool:
  return all(name in target_layout for name in FOOT_CONTACT_TARGET_NAMES)


def _has_foot_load_targets(target_layout: list[str]) -> bool:
  return all(name in target_layout for name in FOOT_LOG_LOAD_TARGET_NAMES)


def _has_stance_foot_projected_gravity_targets(target_layout: list[str]) -> bool:
  return any(
    all(name in target_layout for name in names)
    for names in (
      LEFT_STANCE_FOOT_PROJECTED_GRAVITY_TARGET_NAMES,
      RIGHT_STANCE_FOOT_PROJECTED_GRAVITY_TARGET_NAMES,
    )
  )


def _target_indices(target_layout: list[str], target_names: list[str]) -> list[int]:
  return [target_layout.index(name) for name in target_names if name in target_layout]


def _projected_gravity_vector_groups(
  target_layout: list[str],
) -> list[tuple[str, list[int]]]:
  groups: list[tuple[str, list[int]]] = []
  candidates = {
    "projected_gravity": PROJECTED_GRAVITY_TARGET_NAMES,
    "left_stance_foot_projected_gravity": list(
      LEFT_STANCE_FOOT_PROJECTED_GRAVITY_TARGET_NAMES
    ),
    "right_stance_foot_projected_gravity": list(
      RIGHT_STANCE_FOOT_PROJECTED_GRAVITY_TARGET_NAMES
    ),
  }
  for name, target_names in candidates.items():
    if all(target_name in target_layout for target_name in target_names):
      groups.append(
        (name, [target_layout.index(target_name) for target_name in target_names])
      )
  return groups


class CombinedArrayView:
  """Virtual concatenation of multiple 2D arrays without materializing one big copy."""

  def __init__(self, arrays):
    if not arrays:
      raise ValueError("CombinedArrayView requires at least one array")
    self.arrays = arrays
    self.lengths = np.array([len(arr) for arr in arrays], dtype=np.int64)
    self.offsets = np.concatenate(([0], np.cumsum(self.lengths)))
    self.total_rows = int(self.offsets[-1])
    self.n_cols = int(arrays[0].shape[1])

  def __len__(self):
    return self.total_rows

  @property
  def shape(self):
    return (self.total_rows, self.n_cols)

  def take(self, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    out = np.empty((len(indices), self.n_cols), dtype=np.float32)
    array_ids = np.searchsorted(self.offsets[1:], indices, side="right")
    for array_id in np.unique(array_ids):
      mask = array_ids == array_id
      local_idx = indices[mask] - self.offsets[array_id]
      out[mask] = np.asarray(self.arrays[array_id][local_idx], dtype=np.float32)
    return out

  def iter_blocks(self, block_size: int):
    for arr in self.arrays:
      for start in range(0, len(arr), block_size):
        yield np.asarray(arr[start : start + block_size], dtype=np.float64)


class IndexedArrayView:
  """View that applies a row index filter without materializing a large copy."""

  def __init__(self, array, indices: np.ndarray):
    self.array = array
    self.indices = np.asarray(indices, dtype=np.int64)
    self.n_cols = int(array.shape[1])

  def __len__(self):
    return int(len(self.indices))

  @property
  def shape(self):
    return (len(self), self.n_cols)

  def __getitem__(self, item):
    return self.array[self.indices[item]]


class ColumnSubsetArrayView:
  """View that selects columns from a 2D array/view without copying all rows."""

  def __init__(self, array, columns: list[int]):
    self.array = array
    self.columns = np.asarray(columns, dtype=np.int64)
    self.n_cols = int(len(columns))

  def __len__(self):
    return len(self.array)

  @property
  def shape(self):
    return (len(self), self.n_cols)

  def __getitem__(self, item):
    rows = np.asarray(self.array[item])
    if rows.ndim == 1:
      return rows[self.columns]
    return rows[:, self.columns]

  def take(self, indices: np.ndarray) -> np.ndarray:
    if hasattr(self.array, "take") and not isinstance(self.array, np.ndarray):
      rows = self.array.take(indices)
    else:
      rows = np.asarray(self.array[indices])
    return np.asarray(rows[:, self.columns], dtype=np.float32)


def _target_group_names(target_group: str, target_layout: list[str]) -> list[str]:
  if target_group not in TARGET_GROUP_NAMES:
    raise ValueError(f"Unsupported target_group={target_group!r}")
  names = list(TARGET_GROUP_NAMES[target_group])
  missing = [name for name in names if name not in target_layout]
  if missing:
    raise ValueError(
      f"target_group={target_group!r} requires missing targets {missing}; "
      f"dataset target_layout={target_layout}"
    )
  return names


def _select_target_group(
  ds: dict,
  target_group: str,
) -> dict:
  source_layout = list(ds["target_layout"])
  ds = _apply_target_group_row_filter(ds, target_group, source_layout)
  selected_names = _target_group_names(target_group, source_layout)
  indices = [source_layout.index(name) for name in selected_names]
  targets = ColumnSubsetArrayView(ds["targets"], indices)

  selected = dict(ds)
  selected["targets"] = targets
  selected["target_group"] = target_group
  selected["source_target_layout"] = source_layout
  selected["target_layout"] = selected_names
  selected["target_indices"] = indices
  selected["estimate_foot_heights"] = _has_foot_height_targets(selected_names)
  selected["estimate_projected_gravity"] = _has_projected_gravity_targets(
    selected_names
  )
  selected["estimate_joint_vel"] = _has_joint_vel_targets(selected_names)
  selected["estimate_stance_foot_projected_gravity"] = (
    _has_stance_foot_projected_gravity_targets(selected_names)
  )
  return selected


def _apply_target_group_row_filter(
  ds: dict,
  target_group: str,
  source_layout: list[str],
) -> dict:
  """Apply target-group-specific row filters before selecting target columns."""
  left_stance_groups = (
    "left_stance_foot_height",
    "left_stance_foot_projected_gravity",
  )
  right_stance_groups = (
    "right_stance_foot_height",
    "right_stance_foot_projected_gravity",
  )
  if target_group not in (*left_stance_groups, *right_stance_groups):
    return ds

  contact_name = (
    "left_foot_contact" if target_group in left_stance_groups else "right_foot_contact"
  )
  if contact_name not in source_layout:
    raise ValueError(
      f"target_group={target_group!r} requires {contact_name!r} in "
      f"dataset target_layout={source_layout}. Recollect with "
      "the default estimator data collector."
    )

  contact_idx = source_layout.index(contact_name)
  keep_chunks = []
  chunk_size = 1_000_000
  row_count = len(ds["targets"])
  for start in range(0, row_count, chunk_size):
    end = min(start + chunk_size, row_count)
    target_rows = np.asarray(ds["targets"][start:end], dtype=np.float32)
    local_keep = np.nonzero(target_rows[:, contact_idx] >= 0.5)[0]
    if local_keep.size:
      keep_chunks.append((local_keep + start).astype(np.int64))
  keep_indices = (
    np.concatenate(keep_chunks, axis=0) if keep_chunks else np.zeros(0, dtype=np.int64)
  )
  if keep_indices.size == 0:
    raise ValueError(
      f"target_group={target_group!r} found no stance rows using {contact_name}."
    )

  filtered = dict(ds)
  filtered["proprio"] = IndexedArrayView(ds["proprio"], keep_indices)
  filtered["targets"] = IndexedArrayView(ds["targets"], keep_indices)
  for key in ("env_ids", "episode_ids", "terminated", "truncated", "dones"):
    if ds.get(key) is not None:
      filtered[key] = np.asarray(ds[key])[keep_indices]
  base_filter = ds.get("filter")
  filtered["filter"] = {
    "type": "stance_contact_rows",
    "target_group": target_group,
    "contact_name": contact_name,
    "contact_threshold": 0.5,
    "samples_before": int(len(ds["proprio"])),
    "samples_after": int(keep_indices.size),
    "samples_removed": int(len(ds["proprio"]) - keep_indices.size),
    "kept_fraction": float(keep_indices.size / max(len(ds["proprio"]), 1)),
    "parent_filter": base_filter,
  }
  return filtered


def _validate_proprio_contract(data: dict, data_path: str) -> None:
  proprio_terms = data.get("proprio_terms")
  action_repr = data.get("proprio_action_representation")
  if proprio_terms is None or tuple(proprio_terms) != PROPRIO_TERMS:
    raise ValueError(
      f"{data_path} uses unsupported proprio_terms={proprio_terms!r}. "
      "This estimator trainer now requires datasets collected with previous "
      "q_des. Recollect estimator data with the updated collector."
    )
  if action_repr != EXPECTED_PROPRIO_ACTION_REPRESENTATION:
    raise ValueError(
      f"{data_path} uses unsupported proprio_action_representation="
      f"{action_repr!r}. This estimator trainer now requires previous q_des "
      "as the estimator action block."
    )
  for field, expected in (
    ("history_length", HISTORY_LENGTH),
    ("proprio_dim", PROPRIO_DIM),
  ):
    if data.get(field) != expected:
      raise ValueError(
        f"{data_path} requires {field}={expected} for the paper estimators; "
        f"received {data.get(field)!r}. Recollect using the default configuration."
      )
  if data.get("joint_order", "mujoco") != "mujoco":
    raise ValueError(f"{data_path} requires MuJoCo joint ordering.")


def _load_dataset(data_path: str) -> dict:
  """Load one estimator dataset."""
  data = torch.load(data_path, map_location="cpu", weights_only=False)
  _validate_proprio_contract(data, data_path)
  data_dir = os.path.dirname(os.path.abspath(data_path))

  if "proprio_history" in data:
    proprio_np = (
      data["proprio_history"].numpy().reshape(data["proprio_history"].shape[0], -1)
    )
    targets_np = data["targets"].numpy()
    env_ids_src = data.get("env_ids")
    episode_ids_src = data.get("episode_ids")
    terminated_src = data.get("terminated")
    truncated_src = data.get("truncated")
    dones_src = data.get("dones")
  elif "proprio_file" in data:
    proprio_np = np.load(os.path.join(data_dir, data["proprio_file"]), mmap_mode="r")
    targets_np = np.load(os.path.join(data_dir, data["targets_file"]), mmap_mode="r")
    env_ids_path = os.path.join(data_dir, data.get("env_ids_file", "env_ids.npy"))
    env_ids_src = (
      np.load(env_ids_path, mmap_mode="r").astype(np.int32)
      if os.path.exists(env_ids_path)
      else data.get("env_ids")
    )
    episode_ids_path = os.path.join(
      data_dir, data.get("episode_ids_file", "episode_ids.npy")
    )
    episode_ids_src = (
      np.load(episode_ids_path, mmap_mode="r").astype(np.int32)
      if os.path.exists(episode_ids_path)
      else data.get("episode_ids")
    )
    terminated_path = os.path.join(
      data_dir, data.get("terminated_file", "terminated.npy")
    )
    terminated_src = (
      np.load(terminated_path, mmap_mode="r")
      if os.path.exists(terminated_path)
      else data.get("terminated")
    )
    truncated_path = os.path.join(data_dir, data.get("truncated_file", "truncated.npy"))
    truncated_src = (
      np.load(truncated_path, mmap_mode="r")
      if os.path.exists(truncated_path)
      else data.get("truncated")
    )
    dones_path = os.path.join(data_dir, data.get("dones_file", "dones.npy"))
    dones_src = (
      np.load(dones_path, mmap_mode="r")
      if os.path.exists(dones_path)
      else data.get("dones")
    )
  else:
    raise ValueError(f"Unrecognized data format in {data_path}")

  if isinstance(env_ids_src, torch.Tensor):
    env_ids_np = env_ids_src.numpy().astype(np.int32)
  elif env_ids_src is not None:
    env_ids_np = np.asarray(env_ids_src, dtype=np.int32)
  else:
    env_ids_np = None
  if isinstance(episode_ids_src, torch.Tensor):
    episode_ids_np = episode_ids_src.numpy().astype(np.int32)
  elif episode_ids_src is not None:
    episode_ids_np = np.asarray(episode_ids_src, dtype=np.int32)
  else:
    episode_ids_np = None

  def _optional_bool_array(src):
    if isinstance(src, torch.Tensor):
      return src.numpy().astype(bool)
    if src is not None:
      return np.asarray(src, dtype=bool)
    return None

  terminated_np = _optional_bool_array(terminated_src)
  truncated_np = _optional_bool_array(truncated_src)
  dones_np = _optional_bool_array(dones_src)

  H = int(data["history_length"])
  proprio_dim = int(data["proprio_dim"])
  proprio_terms = data.get(
    "proprio_terms",
    EXPECTED_PROPRIO_TERMS,
  )
  proprio_action_representation = data.get("proprio_action_representation")
  proprio_scales = data.get("proprio_scales", [0.25, 1.0, 1.0, 0.05, 1.0])
  noise_ranges = data.get(
    "noise_ranges",
    {"ang_vel": 0.2, "proj_grav": 0.05, "joint_pos": 0.01, "joint_vel": 1.5},
  )
  target_layout = data.get("target_layout")
  if target_layout is None:
    target_layout = _infer_state_target_layout(targets_np.shape[1])
  else:
    target_layout = list(target_layout)
  if len(target_layout) != targets_np.shape[1]:
    raise ValueError(
      f"{data_path} has target_layout length {len(target_layout)} but "
      f"targets width {targets_np.shape[1]}"
    )
  collection = dict(data.get("collection") or {})
  if not collection:
    collection = {
      "task": data.get("task"),
      "checkpoint_dir": data.get("checkpoint_dir"),
      "checkpoint_dir_repo_relative": data.get("checkpoint_dir_repo_relative"),
      "requested_iterations": data.get("requested_iterations"),
      "valid_iterations": data.get("valid_iterations"),
      "checkpoint_paths": data.get("checkpoint_paths"),
      "checkpoint_paths_repo_relative": data.get("checkpoint_paths_repo_relative"),
      "terrain": data.get("terrain"),
    }
    collection = {k: v for k, v in collection.items() if v is not None}

  return {
    "path": os.path.abspath(data_path),
    "meta": data,
    "proprio": proprio_np,
    "targets": targets_np,
    "env_ids": env_ids_np,
    "episode_ids": episode_ids_np,
    "terminated": terminated_np,
    "truncated": truncated_np,
    "dones": dones_np,
    "history_length": H,
    "proprio_dim": proprio_dim,
    "proprio_terms": list(proprio_terms),
    "proprio_action_representation": proprio_action_representation,
    "proprio_scales": list(proprio_scales),
    "noise_ranges": noise_ranges,
    "target_layout": list(target_layout),
    "estimate_foot_heights": bool(
      data.get("estimate_foot_heights", _has_foot_height_targets(list(target_layout)))
    ),
    "estimate_projected_gravity": bool(
      data.get(
        "estimate_projected_gravity",
        _has_projected_gravity_targets(list(target_layout)),
      )
    ),
    "estimate_joint_vel": bool(
      data.get("estimate_joint_vel", _has_joint_vel_targets(list(target_layout)))
    ),
    "estimate_foot_contact": bool(
      data.get(
        "estimate_foot_contact",
        _has_foot_contact_targets(list(target_layout))
        and _has_foot_load_targets(list(target_layout)),
      )
    ),
    "estimate_stance_foot_projected_gravity": bool(
      data.get(
        "estimate_stance_foot_orientation",
        data.get(
          "estimate_stance_foot_projected_gravity",
          _has_stance_foot_projected_gravity_targets(list(target_layout)),
        ),
      )
    ),
    "foot_contact_load_sensor_name": data.get("foot_contact_load_sensor_name"),
    "foot_contact_load_threshold_n": data.get("foot_contact_load_threshold_n"),
    "foot_contact_load_normalization": data.get("foot_contact_load_normalization"),
    "foot_body_names": data.get(
      "foot_body_names",
      FOOT_BODY_NAMES if _has_foot_height_targets(list(target_layout)) else [],
    ),
    "foot_height_target": data.get(
      "foot_height_target",
      FOOT_HEIGHT_TARGET if _has_foot_height_targets(list(target_layout)) else None,
    ),
    "stance_foot_orientation_target": data.get("stance_foot_orientation_target"),
    "stance_foot_orientation_target_names": data.get(
      "stance_foot_orientation_target_names",
      (
        STANCE_FOOT_PROJECTED_GRAVITY_TARGET_NAMES
        if _has_stance_foot_projected_gravity_targets(list(target_layout))
        else []
      ),
    ),
    "num_envs": data.get("num_envs"),
    "task": data.get("task"),
    "checkpoint_dir": data.get("checkpoint_dir"),
    "checkpoint_dir_repo_relative": data.get("checkpoint_dir_repo_relative"),
    "requested_iterations": data.get("requested_iterations"),
    "valid_iterations": data.get("valid_iterations"),
    "checkpoint_paths": data.get("checkpoint_paths"),
    "checkpoint_paths_repo_relative": data.get("checkpoint_paths_repo_relative"),
    "terrain_override": data.get("terrain_override"),
    "terrain_type": data.get("terrain_type"),
    "terrain": data.get("terrain"),
    "termination": data.get("termination"),
    "filter": None,
    "collection": collection,
  }


def _validate_data_paths(data_paths: list[str]) -> None:
  missing = [path for path in data_paths if not os.path.isfile(path)]
  if not missing:
    return

  lines = [
    "Estimator training data file not found.",
    "Pass --data pointing at the estimator_training_data.pt file produced by "
    "scripts/uan_pipeline/collect_estimator_data.py.",
    "Missing path(s):",
  ]
  lines.extend(f"  - {path}" for path in missing)
  raise FileNotFoundError("\n".join(lines))


def _drop_failed_episodes(ds: dict) -> dict:
  env_ids = ds.get("env_ids")
  episode_ids = ds.get("episode_ids")
  terminated = ds.get("terminated")
  if env_ids is None or episode_ids is None or terminated is None:
    raise ValueError(
      "--drop_failed_episodes requires env_ids, episode_ids, and terminated "
      f"metadata in {ds['path']}. Recollect with the updated collector."
    )

  env_ids_np = np.asarray(env_ids, dtype=np.int64)
  episode_ids_np = np.asarray(episode_ids, dtype=np.int64)
  terminated_np = np.asarray(terminated, dtype=bool)
  before = len(env_ids_np)
  max_episode_id = int(episode_ids_np.max()) if before else 0
  episode_stride = max_episode_id + 1
  episode_keys = env_ids_np * episode_stride + episode_ids_np
  failed_keys = np.unique(episode_keys[terminated_np])
  keep_mask = ~np.isin(episode_keys, failed_keys)
  keep_indices = np.nonzero(keep_mask)[0].astype(np.int64)

  filtered = dict(ds)
  filtered["proprio"] = IndexedArrayView(ds["proprio"], keep_indices)
  filtered["targets"] = IndexedArrayView(ds["targets"], keep_indices)
  filtered["env_ids"] = np.asarray(ds["env_ids"])[keep_indices]
  filtered["episode_ids"] = np.asarray(ds["episode_ids"])[keep_indices]
  if ds.get("terminated") is not None:
    filtered["terminated"] = np.asarray(ds["terminated"])[keep_indices]
  if ds.get("truncated") is not None:
    filtered["truncated"] = np.asarray(ds["truncated"])[keep_indices]
  if ds.get("dones") is not None:
    filtered["dones"] = np.asarray(ds["dones"])[keep_indices]
  filtered["filter"] = {
    "type": "drop_failed_episodes",
    "failed_episode_count": int(len(failed_keys)),
    "samples_before": int(before),
    "samples_after": int(len(keep_indices)),
    "samples_removed": int(before - len(keep_indices)),
    "removed_fraction": float((before - len(keep_indices)) / before) if before else 0.0,
  }
  return filtered


def _dataset_record(ds: dict) -> dict:
  data_path = os.path.abspath(ds["path"])
  dataset_dir = os.path.dirname(data_path)
  checkpoint_paths = ds.get("checkpoint_paths") or []
  return {
    "dataset_name": os.path.basename(dataset_dir),
    "dataset_dir": dataset_dir,
    "dataset_dir_repo_relative": _repo_relative(dataset_dir),
    "metadata_path": data_path,
    "metadata_path_repo_relative": _repo_relative(data_path),
    "sample_count": int(len(ds["proprio"])),
    "num_envs": int(ds["num_envs"]) if ds["num_envs"] is not None else None,
    "history_length": int(ds["history_length"]),
    "proprio_dim": int(ds["proprio_dim"]),
    "proprio_terms": list(ds["proprio_terms"]),
    "proprio_action_representation": ds["proprio_action_representation"],
    "proprio_scales": list(ds["proprio_scales"]),
    "target_layout": list(ds["target_layout"]),
    "target_group": ds.get("target_group"),
    "source_target_layout": list(ds.get("source_target_layout", ds["target_layout"])),
    "target_indices": list(ds.get("target_indices", range(len(ds["target_layout"])))),
    "estimate_foot_heights": bool(ds.get("estimate_foot_heights", False)),
    "estimate_projected_gravity": bool(ds.get("estimate_projected_gravity", False)),
    "estimate_joint_vel": bool(ds.get("estimate_joint_vel", False)),
    "estimate_foot_contact": bool(ds.get("estimate_foot_contact", False)),
    "estimate_stance_foot_projected_gravity": bool(
      ds.get("estimate_stance_foot_projected_gravity", False)
    ),
    "foot_contact_load_sensor_name": ds.get("foot_contact_load_sensor_name"),
    "foot_contact_load_threshold_n": ds.get("foot_contact_load_threshold_n"),
    "foot_contact_load_normalization": ds.get("foot_contact_load_normalization"),
    "foot_body_names": ds.get("foot_body_names") or [],
    "foot_height_target": ds.get("foot_height_target"),
    "task": ds.get("task"),
    "checkpoint_dir": ds.get("checkpoint_dir"),
    "checkpoint_dir_repo_relative": ds.get("checkpoint_dir_repo_relative"),
    "requested_iterations": ds.get("requested_iterations"),
    "valid_iterations": ds.get("valid_iterations"),
    "checkpoint_paths": checkpoint_paths,
    "checkpoint_paths_repo_relative": ds.get("checkpoint_paths_repo_relative")
    or [_repo_relative(path) or path for path in checkpoint_paths],
    "terrain_override": ds.get("terrain_override"),
    "terrain_type": ds.get("terrain_type"),
    "terrain": ds.get("terrain"),
    "termination": ds.get("termination"),
    "filter": ds.get("filter"),
    "collection": ds.get("collection") or {},
  }


def _unique_present(values) -> list:
  out = []
  for value in values:
    if value is None:
      continue
    if value not in out:
      out.append(value)
  return out


def _summarize_collection_records(dataset_records: list[dict]) -> dict:
  checkpoint_paths = []
  checkpoint_paths_repo_relative = []
  for record in dataset_records:
    checkpoint_paths.extend(record.get("checkpoint_paths") or [])
    checkpoint_paths_repo_relative.extend(
      record.get("checkpoint_paths_repo_relative") or []
    )

  return {
    "tasks": _unique_present(record.get("task") for record in dataset_records),
    "terrain_overrides": _unique_present(
      record.get("terrain_override") for record in dataset_records
    ),
    "terrain_types": _unique_present(
      record.get("terrain_type") for record in dataset_records
    ),
    "checkpoint_dirs": _unique_present(
      record.get("checkpoint_dir") for record in dataset_records
    ),
    "checkpoint_dirs_repo_relative": _unique_present(
      record.get("checkpoint_dir_repo_relative") for record in dataset_records
    ),
    "checkpoint_paths": _unique_present(checkpoint_paths),
    "checkpoint_paths_repo_relative": _unique_present(checkpoint_paths_repo_relative),
    "datasets": [
      record.get("collection")
      or {
        "task": record.get("task"),
        "terrain": record.get("terrain"),
        "checkpoint_paths": record.get("checkpoint_paths"),
      }
      for record in dataset_records
    ],
  }


def _ensure_compatible_datasets(datasets: list[dict]):
  ref = datasets[0]
  fields = [
    "history_length",
    "proprio_dim",
    "proprio_terms",
    "proprio_action_representation",
    "proprio_scales",
    "target_layout",
  ]
  for ds in datasets[1:]:
    for field in fields:
      if ds[field] != ref[field]:
        raise ValueError(
          f"Incompatible datasets: field {field!r} differs between "
          f"{ref['path']} and {ds['path']}"
        )
    if ds["proprio"].shape[1] != ref["proprio"].shape[1]:
      raise ValueError(f"Incompatible proprio width for {ds['path']}")
    if ds["targets"].shape[1] != ref["targets"].shape[1]:
      raise ValueError(f"Incompatible target width for {ds['path']}")


def _build_combined_env_ids(
  datasets: list[dict],
) -> tuple[np.ndarray | None, int | None]:
  has_env_ids = [ds["env_ids"] is not None for ds in datasets]
  if any(has_env_ids) and not all(has_env_ids):
    raise ValueError("Either all datasets must provide env_ids, or none of them can.")
  if not any(has_env_ids):
    return None, None

  offset = 0
  env_chunks = []
  for ds in datasets:
    env_ids = np.asarray(ds["env_ids"], dtype=np.int32)
    num_envs = ds["num_envs"]
    if num_envs is None:
      num_envs = int(env_ids.max()) + 1
    env_chunks.append(env_ids + offset)
    offset += int(num_envs)
  return np.concatenate(env_chunks, axis=0), offset


def _chunked_mean_std(arr_view, stat_chunk: int):
  """Compute mean/std over axis 0 for CombinedArrayView or numpy arrays."""
  d = arr_view.shape[1]
  s = np.zeros(d, dtype=np.float64)
  sq = np.zeros(d, dtype=np.float64)
  n = 0

  if isinstance(arr_view, CombinedArrayView):
    blocks = arr_view.iter_blocks(stat_chunk)
  else:

    def _single_array_blocks():
      for start in range(0, len(arr_view), stat_chunk):
        yield np.asarray(arr_view[start : start + stat_chunk], dtype=np.float64)

    blocks = _single_array_blocks()

  for block in blocks:
    s += block.sum(axis=0)
    sq += (block**2).sum(axis=0)
    n += len(block)

  mean = (s / n).astype(np.float32)
  std = np.sqrt(sq / n - (s / n) ** 2).astype(np.float32) + 1e-6
  return torch.from_numpy(mean), torch.from_numpy(std)


def _split_indices(
  datasets: list[dict], val_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
  """Build the same holdout for preflight and component training."""
  env_ids, total_num_envs = _build_combined_env_ids(datasets)
  rng = np.random.RandomState(42)
  if env_ids is not None:
    num_envs = total_num_envs or int(env_ids.max()) + 1
    n_val_envs = max(1, int(num_envs * val_fraction))
    val_envs = rng.permutation(num_envs)[:n_val_envs]
    val_mask = np.isin(env_ids, val_envs)
    train_idx = np.nonzero(~val_mask)[0]
    val_idx = np.nonzero(val_mask)[0]
  else:
    n_samples = sum(len(ds["proprio"]) for ds in datasets)
    indices = rng.permutation(n_samples)
    n_val = int(n_samples * val_fraction)
    val_idx, train_idx = indices[:n_val], indices[n_val:]
  if not len(train_idx) or not len(val_idx):
    group = datasets[0].get("target_group", "paper")
    raise ValueError(
      f"{group}: training and validation splits must both contain samples after "
      "episode and stance filtering. Collect more data across environments or "
      "adjust --val_fraction."
    )
  return train_idx, val_idx


def _validate_dataset_arrays(ds: dict) -> None:
  """Reject malformed data before any component begins training."""
  proprio, targets = ds["proprio"], ds["targets"]
  n_samples = len(proprio)
  if (
    len(proprio.shape) != 2
    or proprio.shape[1] != HISTORY_LENGTH * PROPRIO_DIM
    or len(targets.shape) != 2
    or len(targets) != n_samples
    or n_samples == 0
  ):
    raise ValueError(
      f"{ds['path']}: expected nonempty proprio shape (N, "
      f"{HISTORY_LENGTH * PROPRIO_DIM}) and targets shape (N, D); "
      f"received {proprio.shape} and {targets.shape}."
    )
  if len(set(ds["target_layout"])) != len(ds["target_layout"]):
    raise ValueError(f"{ds['path']}: duplicate names in target_layout.")
  for key in ("env_ids", "episode_ids", "terminated", "truncated", "dones"):
    values = ds.get(key)
    if values is not None and np.asarray(values).shape != (n_samples,):
      raise ValueError(f"{ds['path']}: {key} must have shape ({n_samples},).")
  scales = np.asarray(ds["proprio_scales"], dtype=np.float64)
  if scales.shape != (len(PROPRIO_TERMS),) or not np.all(
    np.isfinite(scales) & (scales > 0)
  ):
    raise ValueError(f"{ds['path']}: invalid proprio_scales.")
  contact_indices = _target_indices(ds["target_layout"], FOOT_CONTACT_TARGET_NAMES)
  for start in range(0, n_samples, 500_000):
    x = np.asarray(proprio[start : start + 500_000])
    y = np.asarray(targets[start : start + 500_000])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
      raise ValueError(f"{ds['path']}: estimator data contains non-finite values.")
    if contact_indices and not np.isin(y[:, contact_indices], [0, 1]).all():
      raise ValueError(f"{ds['path']}: foot contact targets must be binary labels.")


def _training_groups(target_group: str) -> tuple[str, ...]:
  if target_group == "paper":
    return tuple(COMPONENT_TARGETS)
  if target_group in PAIR_COMPONENTS:
    return PAIR_COMPONENTS[target_group]
  if target_group in COMPONENT_TARGETS:
    return (target_group,)
  raise ValueError(f"Unsupported target_group={target_group!r}")


def _preflight_training(args: argparse.Namespace) -> dict[str, list[dict]]:
  """Load, filter, and validate every requested estimator before training any."""
  if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
    raise ValueError("epochs, batch_size, and lr must be positive.")
  if not 0 < args.val_fraction < 1:
    raise ValueError("val_fraction must be strictly between 0 and 1.")
  _validate_data_paths(args.data)
  groups = _training_groups(args.target_group)
  datasets = [_load_dataset(path) for path in args.data]
  for ds in datasets:
    _validate_dataset_arrays(ds)
    for group in groups:
      _target_group_names(group, ds["target_layout"])
  _ensure_compatible_datasets(datasets)
  if args.drop_failed_episodes:
    datasets = [_drop_failed_episodes(ds) for ds in datasets]
  grouped = {}
  for group in groups:
    selected = [_select_target_group(ds, group) for ds in datasets]
    _ensure_compatible_datasets(selected)
    _split_indices(selected, args.val_fraction)
    grouped[group] = selected
  return grouped


def _write_manifest(path: str, manifest: dict) -> None:
  temporary = path + ".tmp"
  with open(temporary, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
  os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  grouped = _preflight_training(args)
  output_dir = os.path.abspath(args.output_dir)
  parent_dir = os.path.dirname(output_dir)
  os.makedirs(parent_dir, exist_ok=True)
  outputs = {}
  # Keep an existing bundle intact if any component fails to train or export.
  with tempfile.TemporaryDirectory(
    prefix=".estimator-training-", dir=parent_dir
  ) as tmp:
    for group, datasets in grouped.items():
      if args.target_group == "paper":
        subdir = COMPONENT_SUBDIRS[group]
      elif args.target_group in PAIR_COMPONENTS:
        subdir = group
      else:
        subdir = "."
      component_args = argparse.Namespace(**vars(args))
      component_args.target_group = group
      component_args.output_dir = os.path.join(tmp, subdir)
      if args.run_name and len(grouped) > 1:
        component_args.run_name = f"{args.run_name}_{group}"
      print(f"[Train] {group} -> {os.path.join(output_dir, subdir)}")
      _train_estimator(component_args, datasets)
      outputs[group] = {
        "state_estimator_jit": os.path.join(subdir, "state_estimator.jit"),
        "estimator_meta": os.path.join(subdir, "estimator_meta.json"),
      }
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(tmp, output_dir, dirs_exist_ok=True)
  if len(grouped) > 1:
    ref = next(iter(grouped.values()))[0]
    manifest = {
      "version": 1,
      "mode": args.target_group,
      "groups": list(grouped),
      "history_length": HISTORY_LENGTH,
      "proprio_dim": PROPRIO_DIM,
      "proprio_terms": list(PROPRIO_TERMS),
      "proprio_action_representation": PROPRIO_ACTION_REPRESENTATION,
      "proprio_scales": ref["proprio_scales"],
      "joint_order": "mujoco",
      "outputs": outputs,
    }
    filename = (
      BUNDLE_MANIFEST
      if args.target_group == "paper"
      else f"{args.target_group}_estimators_meta.json"
    )
    _write_manifest(os.path.join(output_dir, filename), manifest)
  print(f"[Saved] {output_dir}")


def _target_group_units_note(target_group: str) -> str:
  if target_group == "base_lin":
    return "base linear velocity in body frame"
  if target_group == "foot_contact":
    return "left/right loaded-foot contact logits; targets are binary labels"
  if target_group.endswith("_height"):
    return "stance-filtered ankle-roll link world height in meters"
  return "stance-filtered projected gravity vector in the ankle-roll link frame"


def _train_estimator(args: argparse.Namespace, datasets: list[dict]) -> None:
  device = torch.device(args.device)
  hidden_dims = list(COMPONENT_HIDDEN_DIMS[args.target_group])
  ref = datasets[0]

  H = ref["history_length"]
  proprio_dim = ref["proprio_dim"]
  TARGET_NAMES = ref["target_layout"]
  TARGET_GROUP = ref.get("target_group", args.target_group)
  LOSS_TYPE = "bce_logits" if TARGET_GROUP == "foot_contact" else "mse"
  SOURCE_TARGET_LAYOUT = list(ref.get("source_target_layout", TARGET_NAMES))
  TARGET_INDICES = list(ref.get("target_indices", range(len(TARGET_NAMES))))

  proprio_np = CombinedArrayView([ds["proprio"] for ds in datasets])
  targets_np = CombinedArrayView([ds["targets"] for ds in datasets])
  N = len(proprio_np)
  input_dim = H * proprio_dim
  output_dim = targets_np.shape[1]

  print(
    f"[Data] {N:,} samples across {len(datasets)} dataset(s), "
    f"H={H}, proprio_dim={proprio_dim}"
  )
  for ds in datasets:
    print(f"  - {ds['path']} ({len(ds['proprio']):,} samples)")
  print(f"[Model] {input_dim} → {hidden_dims} → {output_dim}")
  print(f"[Targets] group={TARGET_GROUP} layout={TARGET_NAMES}")

  # --- Compute normalization stats from numpy (chunked to limit RAM) ---
  STAT_CHUNK = 500_000

  input_mean, input_std = _chunked_mean_std(proprio_np, STAT_CHUNK)
  target_mean, target_std = _chunked_mean_std(targets_np, STAT_CHUNK)

  print(f"[Stats] Target means: {target_mean.tolist()}")
  print(f"[Stats] Target stds:  {target_std.tolist()}")
  if LOSS_TYPE == "bce_logits":
    target_mean = torch.zeros_like(target_mean)
    target_std = torch.ones_like(target_std)
    print(f"[Stats] {TARGET_GROUP} uses raw binary labels with BCEWithLogitsLoss")

  train_idx, val_idx = _split_indices(datasets, args.val_fraction)
  train_size = len(train_idx)
  val_size = len(val_idx)
  print(f"[Split] train={train_size:,}, val={val_size:,}")

  # Normalization stats on device for batch-level normalization
  input_mean_d = input_mean.to(device)
  input_std_d = input_std.to(device)
  target_mean_d = target_mean.to(device)
  target_std_d = target_std.to(device)

  def _load_batch(indices):
    """Load a batch from numpy memmaps, normalize, move to GPU."""
    x = torch.from_numpy(proprio_np.take(indices)).to(device)
    y = torch.from_numpy(targets_np.take(indices)).to(device)
    x_norm = (x - input_mean_d) / input_std_d
    if LOSS_TYPE == "bce_logits":
      y_norm = y
    else:
      y_norm = (y - target_mean_d) / target_std_d
    return x_norm, y_norm

  # --- Build model ---
  model = build_mlp(input_dim, output_dim, hidden_dims).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  loss_fn = nn.BCEWithLogitsLoss() if LOSS_TYPE == "bce_logits" else nn.MSELoss()

  def _init_binary_counts() -> dict[str, torch.Tensor]:
    return {
      "tp": torch.zeros((), device=device),
      "tn": torch.zeros((), device=device),
      "fp": torch.zeros((), device=device),
      "fn": torch.zeros((), device=device),
      "pos": torch.zeros((), device=device),
      "total": torch.zeros((), device=device),
    }

  def _accumulate_binary_counts(
    counts: dict[str, torch.Tensor],
    prob: torch.Tensor,
    target: torch.Tensor,
  ) -> None:
    pred_pos = prob >= 0.5
    true_pos = target >= 0.5
    counts["tp"] += (pred_pos & true_pos).sum()
    counts["tn"] += ((~pred_pos) & (~true_pos)).sum()
    counts["fp"] += (pred_pos & (~true_pos)).sum()
    counts["fn"] += ((~pred_pos) & true_pos).sum()
    counts["pos"] += true_pos.sum()
    counts["total"] += torch.as_tensor(true_pos.numel(), device=device)

  def _binary_metrics(counts: dict[str, torch.Tensor]) -> dict[str, float]:
    tp = counts["tp"].item()
    tn = counts["tn"].item()
    fp = counts["fp"].item()
    fn = counts["fn"].item()
    total = max(counts["total"].item(), 1.0)
    return {
      "accuracy": float((tp + tn) / total),
      "precision": float(tp / max(tp + fp, 1.0)),
      "recall": float(tp / max(tp + fn, 1.0)),
      "fpr": float(fp / max(fp + tn, 1.0)),
      "fnr": float(fn / max(fn + tp, 1.0)),
      "positive_fraction": float(counts["pos"].item() / total),
    }

  def _decode_pred(pred: torch.Tensor) -> torch.Tensor:
    if LOSS_TYPE == "bce_logits":
      return torch.sigmoid(pred)
    return pred * target_std_d + target_mean_d

  def _decode_true(y: torch.Tensor) -> torch.Tensor:
    if LOSS_TYPE == "bce_logits":
      return y
    return y * target_std_d + target_mean_d

  n_params = sum(p.numel() for p in model.parameters())
  print(f"[Model] {n_params:,} parameters")
  print(f"[Train] epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}")

  # --- W&B ---
  proprio_terms = ref["proprio_terms"]
  proprio_action_representation = ref["proprio_action_representation"]
  proprio_scales = ref["proprio_scales"]
  noise_ranges = ref["noise_ranges"]
  estimate_foot_heights = _has_foot_height_targets(TARGET_NAMES)
  estimate_projected_gravity = _has_projected_gravity_targets(TARGET_NAMES)
  estimate_joint_vel = _has_joint_vel_targets(TARGET_NAMES)
  estimate_stance_foot_projected_gravity = _has_stance_foot_projected_gravity_targets(
    TARGET_NAMES
  )
  projected_gravity_indices = _target_indices(
    TARGET_NAMES, PROJECTED_GRAVITY_TARGET_NAMES
  )
  projected_gravity_vector_groups = _projected_gravity_vector_groups(TARGET_NAMES)
  joint_vel_indices = _target_indices(TARGET_NAMES, JOINT_VEL_TARGET_NAMES)
  foot_body_names = ref.get("foot_body_names") or (
    FOOT_BODY_NAMES
    if estimate_foot_heights or estimate_stance_foot_projected_gravity
    else []
  )
  foot_height_target = ref.get("foot_height_target") or (
    FOOT_HEIGHT_TARGET if estimate_foot_heights else None
  )
  dataset_records = [_dataset_record(ds) for ds in datasets]
  collection_metadata = _summarize_collection_records(dataset_records)
  terrain_overrides = collection_metadata["terrain_overrides"]
  terrain_types = collection_metadata["terrain_types"]
  terrain_override = terrain_overrides[0] if len(terrain_overrides) == 1 else "mixed"
  terrain_type = terrain_types[0] if len(terrain_types) == 1 else "mixed"

  use_wandb = bool(args.wandb_project)
  if use_wandb:
    wandb.init(
      project=args.wandb_project,
      name=args.run_name,
      config={
        # Architecture
        "hidden_dims": hidden_dims,
        "activation": "ELU",
        "input_dim": input_dim,
        "output_dim": output_dim,
        "n_params": n_params,
        # Training
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "val_fraction": args.val_fraction,
        "grad_clip_norm": 1.0,
        "optimizer": "Adam",
        # Data
        "data_paths": [ds["path"] for ds in datasets],
        "data_paths_repo_relative": [
          record["metadata_path_repo_relative"] or record["metadata_path"]
          for record in dataset_records
        ],
        "dataset_dirs": [
          record["dataset_dir_repo_relative"] or record["dataset_dir"]
          for record in dataset_records
        ],
        "dataset_count": len(datasets),
        "n_samples": N,
        "datasets": dataset_records,
        "collection": collection_metadata,
        "terrain_override": terrain_override,
        "terrain_type": terrain_type,
        "train_size": train_size,
        "val_size": val_size,
        # Proprio config
        "history_length": H,
        "proprio_dim": proprio_dim,
        "proprio_terms": proprio_terms,
        "proprio_action_representation": proprio_action_representation,
        "proprio_scales": proprio_scales,
        "noise_ranges": noise_ranges,
        # Targets
        "target_layout": TARGET_NAMES,
        "target_group": TARGET_GROUP,
        "loss_type": LOSS_TYPE,
        "source_target_layout": SOURCE_TARGET_LAYOUT,
        "target_indices": TARGET_INDICES,
        "estimate_foot_heights": estimate_foot_heights,
        "estimate_projected_gravity": estimate_projected_gravity,
        "estimate_joint_vel": estimate_joint_vel,
        "estimate_stance_foot_projected_gravity": (
          estimate_stance_foot_projected_gravity
        ),
        "foot_body_names": foot_body_names,
        "foot_height_target": foot_height_target,
        "stance_foot_orientation_target": (
          STANCE_FOOT_PROJECTED_GRAVITY_TARGET
          if estimate_stance_foot_projected_gravity
          else None
        ),
        "target_means": target_mean.tolist(),
        "target_stds": target_std.tolist(),
      },
    )

  best_val_loss = float("inf")
  best_epoch = 0
  best_state = None

  B = args.batch_size

  for epoch in range(args.epochs):
    # --- Train ---
    model.train()
    train_loss = 0.0
    train_sq_err_sum = torch.zeros(output_dim, device=device)
    train_binary_counts = _init_binary_counts() if LOSS_TYPE == "bce_logits" else None
    train_sample_count = 0
    n_batches = 0

    # Shuffle train indices each epoch
    shuf = np.random.permutation(train_size)

    for start in range(0, train_size, B):
      batch_pos = shuf[start : start + B]
      batch_indices = train_idx[batch_pos]
      x_batch, y_batch = _load_batch(batch_indices)

      pred = model(x_batch)
      loss = loss_fn(pred, y_batch)

      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()

      train_loss += loss.item()
      n_batches += 1

      # Accumulate train RMSE (detached, denormalized)
      with torch.no_grad():
        pred_raw = _decode_pred(pred)
        true_raw = _decode_true(y_batch)
        train_sq_err_sum += ((pred_raw - true_raw) ** 2).sum(dim=0)
        if train_binary_counts is not None:
          _accumulate_binary_counts(train_binary_counts, pred_raw, true_raw)
        train_sample_count += pred.shape[0]

    train_loss /= n_batches
    train_rmse = (train_sq_err_sum / train_sample_count).sqrt()

    # --- Validate (in chunks to limit GPU memory) ---
    model.eval()
    val_sq_err_sum = torch.zeros(output_dim, device=device)
    val_binary_counts = _init_binary_counts() if LOSS_TYPE == "bce_logits" else None
    val_loss_sum = 0.0
    val_batches = 0
    with torch.no_grad():
      for vs in range(0, val_size, B):
        ve = min(vs + B, val_size)
        batch_indices = val_idx[vs:ve]
        x_val_b, y_val_b = _load_batch(batch_indices)
        val_pred_b = model(x_val_b)
        val_loss_sum += loss_fn(val_pred_b, y_val_b).item()
        val_batches += 1
        # Denormalize for per-component RMSE
        pred_raw = _decode_pred(val_pred_b)
        true_raw = _decode_true(y_val_b)
        val_sq_err_sum += ((pred_raw - true_raw) ** 2).sum(dim=0)
        if val_binary_counts is not None:
          _accumulate_binary_counts(val_binary_counts, pred_raw, true_raw)

    val_loss = val_loss_sum / val_batches
    per_comp_rmse = (val_sq_err_sum / val_size).sqrt()
    train_binary_metrics = (
      _binary_metrics(train_binary_counts) if train_binary_counts is not None else {}
    )
    val_binary_metrics = (
      _binary_metrics(val_binary_counts) if val_binary_counts is not None else {}
    )

    # --- Log ---
    if use_wandb:
      log_dict = {
        "train/loss": train_loss,
        "val/loss": val_loss,
        "epoch": epoch + 1,
      }
      for i, name in enumerate(TARGET_NAMES):
        log_dict[f"train/rmse_{name}"] = train_rmse[i].item()
        log_dict[f"val/rmse_{name}"] = per_comp_rmse[i].item()
        if "height" in name:
          log_dict[f"train/rmse_mm_{name}"] = train_rmse[i].item() * 1000.0
          log_dict[f"val/rmse_mm_{name}"] = per_comp_rmse[i].item() * 1000.0
      log_dict["train/rmse_mean"] = train_rmse.mean().item()
      log_dict["val/rmse_mean"] = per_comp_rmse.mean().item()
      if projected_gravity_indices:
        idx = projected_gravity_indices
        log_dict["train/rmse_projected_gravity_mean"] = train_rmse[idx].mean().item()
        log_dict["val/rmse_projected_gravity_mean"] = per_comp_rmse[idx].mean().item()
      if joint_vel_indices:
        idx = joint_vel_indices
        log_dict["train/rmse_joint_vel_mean"] = train_rmse[idx].mean().item()
        log_dict["val/rmse_joint_vel_mean"] = per_comp_rmse[idx].mean().item()
      for name, value in train_binary_metrics.items():
        log_dict[f"train/{name}"] = value
      for name, value in val_binary_metrics.items():
        log_dict[f"val/{name}"] = value
      wandb.log(log_dict)

    if (epoch + 1) % 10 == 0 or epoch == 0:
      val_rmse_str = ", ".join(
        f"{TARGET_NAMES[i]}={per_comp_rmse[i]:.4f}" for i in range(output_dim)
      )
      train_rmse_str = ", ".join(
        f"{TARGET_NAMES[i]}={train_rmse[i]:.4f}" for i in range(output_dim)
      )
      print(
        f"  [Epoch {epoch + 1:3d}] "
        f"train={train_loss:.6f}, val={val_loss:.6f}\n"
        f"    train RMSE: {train_rmse_str}\n"
        f"    val   RMSE: {val_rmse_str}"
      )
      if LOSS_TYPE == "bce_logits":
        print(
          "    val binary: "
          f"acc={val_binary_metrics['accuracy']:.4f}, "
          f"precision={val_binary_metrics['precision']:.4f}, "
          f"recall={val_binary_metrics['recall']:.4f}, "
          f"fpr={val_binary_metrics['fpr']:.4f}, "
          f"fnr={val_binary_metrics['fnr']:.4f}, "
          f"positive_fraction={val_binary_metrics['positive_fraction']:.4f}"
        )

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_epoch = epoch + 1
      best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
      os.makedirs(args.output_dir, exist_ok=True)
      torch.save(
        {
          "model_state_dict": best_state,
          "input_mean": input_mean,
          "input_std": input_std,
          "target_mean": target_mean,
          "target_std": target_std,
          "hidden_dims": hidden_dims,
          "input_dim": input_dim,
          "output_dim": output_dim,
          "target_dim": output_dim,
          "terrain_override": terrain_override,
          "terrain_type": terrain_type,
          "collection": collection_metadata,
          "target_layout": TARGET_NAMES,
          "target_group": TARGET_GROUP,
          "loss_type": LOSS_TYPE,
          "source_target_layout": SOURCE_TARGET_LAYOUT,
          "target_indices": TARGET_INDICES,
          "estimate_foot_heights": estimate_foot_heights,
          "estimate_projected_gravity": estimate_projected_gravity,
          "estimate_joint_vel": estimate_joint_vel,
          "estimate_stance_foot_projected_gravity": (
            estimate_stance_foot_projected_gravity
          ),
          "estimate_foot_contact": (
            _has_foot_contact_targets(TARGET_NAMES)
            and _has_foot_load_targets(TARGET_NAMES)
          ),
          "foot_body_names": foot_body_names,
          "foot_height_target": foot_height_target,
          "stance_foot_orientation_target": (
            STANCE_FOOT_PROJECTED_GRAVITY_TARGET
            if estimate_stance_foot_projected_gravity
            else None
          ),
        },
        os.path.join(args.output_dir, "state_estimator.pt"),
      )
      _save_jit(model, input_mean, input_std, target_mean, target_std, args.output_dir)
      model.to(device)

  # --- Restore best model ---
  if best_state is None:
    raise RuntimeError("Training did not produce a finite validation loss.")
  model.load_state_dict(best_state)
  model.eval()
  print(f"\n[Best] epoch {best_epoch}, val_loss={best_val_loss:.6f}")

  # --- Final per-component RMSE ---
  val_sq_err_sum = torch.zeros(output_dim, device=device)
  final_binary_counts = _init_binary_counts() if LOSS_TYPE == "bce_logits" else None
  val_abs_err_chunks = []
  with torch.no_grad():
    for vs in range(0, val_size, B):
      ve = min(vs + B, val_size)
      batch_indices = val_idx[vs:ve]
      x_val_b, y_val_b = _load_batch(batch_indices)
      val_pred_b = model(x_val_b)
      pred_raw = _decode_pred(val_pred_b)
      true_raw = _decode_true(y_val_b)
      abs_err = (pred_raw - true_raw).abs()
      val_sq_err_sum += (abs_err**2).sum(dim=0)
      val_abs_err_chunks.append(abs_err.detach().cpu())
      if final_binary_counts is not None:
        _accumulate_binary_counts(final_binary_counts, pred_raw, true_raw)
  per_comp_rmse = (val_sq_err_sum / val_size).sqrt()
  val_abs_err = torch.cat(val_abs_err_chunks, dim=0)
  per_comp_p95 = torch.quantile(val_abs_err, 0.95, dim=0)
  per_comp_max = val_abs_err.max(dim=0).values
  final_binary_metrics = (
    _binary_metrics(final_binary_counts) if final_binary_counts is not None else {}
  )
  angular_error_deg: dict[str, float] = {}
  if projected_gravity_vector_groups:
    angle_sum = torch.zeros(len(projected_gravity_vector_groups), device=device)
    angle_count = 0
    with torch.no_grad():
      for vs in range(0, val_size, B):
        ve = min(vs + B, val_size)
        batch_indices = val_idx[vs:ve]
        x_val_b, y_val_b = _load_batch(batch_indices)
        val_pred_b = model(x_val_b)
        pred_raw = _decode_pred(val_pred_b)
        true_raw = _decode_true(y_val_b)
        for group_idx, (_group_name, idx) in enumerate(projected_gravity_vector_groups):
          pred_vec = pred_raw[:, idx]
          true_vec = true_raw[:, idx]
          pred_vec = pred_vec / pred_vec.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
          true_vec = true_vec / true_vec.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
          dot = (pred_vec * true_vec).sum(dim=-1).clamp(-1.0, 1.0)
          angle_sum[group_idx] += torch.acos(dot).sum()
        angle_count += ve - vs
    for group_idx, (group_name, _idx) in enumerate(projected_gravity_vector_groups):
      angular_error_deg[group_name] = float(
        (angle_sum[group_idx] / max(angle_count, 1) * 180.0 / torch.pi).item()
      )

  print("\n[Final RMSE on validation set]")
  for i, name in enumerate(TARGET_NAMES):
    print(
      f"  {name:>12s}: rmse={per_comp_rmse[i]:.6f}, "
      f"p95={per_comp_p95[i]:.6f}, max={per_comp_max[i]:.6f}"
    )
  if projected_gravity_indices:
    print(
      f"  projected_gravity_mean: {per_comp_rmse[projected_gravity_indices].mean():.6f}"
    )
  if joint_vel_indices:
    print(f"  joint_vel_mean: {per_comp_rmse[joint_vel_indices].mean():.6f}")
  for group_name, angle_deg in angular_error_deg.items():
    print(f"  angular_error_deg_{group_name}: {angle_deg:.4f}")
  if final_binary_metrics:
    print("\n[Final binary metrics on validation set]")
    for name, value in final_binary_metrics.items():
      print(f"  {name:>18s}: {value:.6f}")

  if use_wandb:
    wandb_summary = _wandb_summary()
    for i, name in enumerate(TARGET_NAMES):
      wandb_summary[f"best/rmse_{name}"] = per_comp_rmse[i].item()
      if "height" in name:
        wandb_summary[f"best/rmse_mm_{name}"] = per_comp_rmse[i].item() * 1000.0
    wandb_summary["best/rmse_mean"] = per_comp_rmse.mean().item()
    if projected_gravity_indices:
      wandb_summary["best/rmse_projected_gravity_mean"] = (
        per_comp_rmse[projected_gravity_indices].mean().item()
      )
    if joint_vel_indices:
      wandb_summary["best/rmse_joint_vel_mean"] = (
        per_comp_rmse[joint_vel_indices].mean().item()
      )
    for group_name, angle_deg in angular_error_deg.items():
      wandb_summary[f"best/angular_error_deg_{group_name}"] = angle_deg
    for name, value in final_binary_metrics.items():
      wandb_summary[f"best/{name}"] = value
    wandb_summary["best/epoch"] = best_epoch
    wandb_summary["best/val_loss"] = best_val_loss

  # --- Export final JIT (best already saved during training) ---
  jit_path = os.path.join(args.output_dir, "state_estimator.jit")
  print(f"\n[Saved] {jit_path}")

  # Also save raw model checkpoint
  ckpt_path = os.path.join(args.output_dir, "state_estimator.pt")
  torch.save(
    {
      "model_state_dict": best_state,
      "input_mean": input_mean,
      "input_std": input_std,
      "target_mean": target_mean,
      "target_std": target_std,
      "hidden_dims": hidden_dims,
      "input_dim": input_dim,
      "output_dim": output_dim,
      "target_dim": output_dim,
      "target_layout": TARGET_NAMES,
      "target_group": TARGET_GROUP,
      "loss_type": LOSS_TYPE,
      "source_target_layout": SOURCE_TARGET_LAYOUT,
      "target_indices": TARGET_INDICES,
      "estimate_foot_heights": estimate_foot_heights,
      "estimate_projected_gravity": estimate_projected_gravity,
      "estimate_joint_vel": estimate_joint_vel,
      "estimate_stance_foot_projected_gravity": (
        estimate_stance_foot_projected_gravity
      ),
      "estimate_foot_contact": (
        _has_foot_contact_targets(TARGET_NAMES) and _has_foot_load_targets(TARGET_NAMES)
      ),
      "foot_body_names": foot_body_names,
      "foot_height_target": foot_height_target,
      "stance_foot_orientation_target": (
        STANCE_FOOT_PROJECTED_GRAVITY_TARGET
        if estimate_stance_foot_projected_gravity
        else None
      ),
      "per_component_rmse": per_comp_rmse,
      "per_component_p95_abs_error": per_comp_p95,
      "per_component_max_abs_error": per_comp_max,
      "angular_error_deg": angular_error_deg,
    },
    ckpt_path,
  )
  print(f"[Saved] {ckpt_path}")

  # Save metadata
  meta = {
    # Architecture
    "hidden_dims": hidden_dims,
    "activation": "ELU",
    "input_dim": input_dim,
    "output_dim": output_dim,
    "target_dim": output_dim,
    # Proprio config (needed at inference time)
    "joint_order": "mujoco",
    "proprio_dim": proprio_dim,
    "history_length": H,
    "target_layout": TARGET_NAMES,
    "target_group": TARGET_GROUP,
    "loss_type": LOSS_TYPE,
    "source_target_layout": SOURCE_TARGET_LAYOUT,
    "target_indices": TARGET_INDICES,
    "target_units_note": _target_group_units_note(TARGET_GROUP),
    "estimate_foot_heights": estimate_foot_heights,
    "estimate_projected_gravity": estimate_projected_gravity,
    "estimate_joint_vel": estimate_joint_vel,
    "estimate_stance_foot_projected_gravity": estimate_stance_foot_projected_gravity,
    "estimate_foot_contact": (
      _has_foot_contact_targets(TARGET_NAMES) and _has_foot_load_targets(TARGET_NAMES)
    ),
    "foot_body_names": foot_body_names,
    "foot_height_target": foot_height_target,
    "stance_foot_orientation_target": (
      STANCE_FOOT_PROJECTED_GRAVITY_TARGET
      if estimate_stance_foot_projected_gravity
      else None
    ),
    "proprio_terms": proprio_terms,
    "proprio_action_representation": proprio_action_representation,
    "proprio_scales": proprio_scales,
    "noise_ranges": noise_ranges,
    "terrain_override": terrain_override,
    "terrain_type": terrain_type,
    "collection": collection_metadata,
    # Training config
    "training": {
      "data_paths": [ds["path"] for ds in datasets],
      "data_paths_repo_relative": [
        record["metadata_path_repo_relative"] or record["metadata_path"]
        for record in dataset_records
      ],
      "dataset_dirs": [
        record["dataset_dir_repo_relative"] or record["dataset_dir"]
        for record in dataset_records
      ],
      "dataset_count": len(datasets),
      "n_samples": N,
      "datasets": dataset_records,
      "n_samples_per_dataset": {ds["path"]: int(len(ds["proprio"])) for ds in datasets},
      "epochs": args.epochs,
      "lr": args.lr,
      "batch_size": args.batch_size,
      "val_fraction": args.val_fraction,
      "best_epoch": best_epoch,
      "best_val_loss": float(best_val_loss),
    },
    # Results
    "per_component_rmse": {
      TARGET_NAMES[i]: float(per_comp_rmse[i]) for i in range(output_dim)
    },
    "per_component_p95_abs_error": {
      TARGET_NAMES[i]: float(per_comp_p95[i]) for i in range(output_dim)
    },
    "per_component_max_abs_error": {
      TARGET_NAMES[i]: float(per_comp_max[i]) for i in range(output_dim)
    },
    "angular_error_deg": angular_error_deg,
    "binary_metrics": final_binary_metrics,
  }
  meta_path = os.path.join(args.output_dir, "estimator_meta.json")
  with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
  print(f"[Saved] {meta_path}")

  if use_wandb:
    wandb.finish()


def export_jit_from_pt(pt_path: str, output_dir: str | None = None) -> str:
  """Export state_estimator.jit from a saved state_estimator.pt checkpoint.

  Args:
    pt_path: Path to state_estimator.pt saved by this script.
    output_dir: Directory to save the JIT (defaults to same dir as pt_path).

  Returns:
    Path to the saved JIT file.
  """
  ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
  hidden_dims = ckpt["hidden_dims"]
  input_dim = ckpt["input_dim"]
  output_dim = ckpt["output_dim"]

  model = build_mlp(input_dim, output_dim, hidden_dims)
  model.load_state_dict(ckpt["model_state_dict"])
  model.eval()

  if output_dir is None:
    output_dir = os.path.dirname(os.path.abspath(pt_path))

  return _save_jit(
    model,
    ckpt["input_mean"],
    ckpt["input_std"],
    ckpt["target_mean"],
    ckpt["target_std"],
    output_dir,
  )


if __name__ == "__main__":
  import sys

  if "--export_from_pt" in sys.argv:
    idx = sys.argv.index("--export_from_pt")
    pt_path = sys.argv[idx + 1]
    out_dir = None
    if "--output_dir" in sys.argv:
      out_dir = sys.argv[sys.argv.index("--output_dir") + 1]
    jit_path = export_jit_from_pt(pt_path, out_dir)
    print(f"[Saved] {jit_path}")
  else:
    main()
