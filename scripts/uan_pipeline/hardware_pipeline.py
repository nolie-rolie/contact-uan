#!/usr/bin/env python3
"""Manifest-driven hardware data pipeline.

This CLI anchors new hardware datasets under:

    assets/hardware/data/{task}/{policy}/{collection_datetime}/

Hardware collection, trimming, post-processing, packaging, and same-policy
combination workflows use the canonical layout above.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from postprocess_real_data import (
  _ACTION_DIM,
  _STATE_DIM,
  _TERMINATION_DIM,
  write_processed_files,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_ROOT = REPO_ROOT / "assets"
HARDWARE_ROOT = ASSETS_ROOT / "hardware"
HARDWARE_DATA_ROOT = HARDWARE_ROOT / "data"
DEFAULT_CONTROL_DT = 0.02
MANIFEST_SCHEMA_VERSION = 1

G1_HARDWARE_JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]

G1_ISAACLAB_JOINT_NAMES = [
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "waist_yaw_joint",
  "left_hip_roll_joint",
  "right_hip_roll_joint",
  "waist_roll_joint",
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "waist_pitch_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_shoulder_roll_joint",
  "right_shoulder_roll_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_elbow_joint",
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
]


@dataclass(frozen=True)
class RolloutSummary:
  relpath: str
  csv_prefix: str
  n_rows: int
  n_cols: int


def now_iso() -> str:
  return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_slug() -> str:
  return datetime.now().strftime("%Y%m%d_%H%M%S")


def natural_key(path: Path) -> tuple[int, str]:
  name = path.stem
  m = re.search(r"_(\d+)$", name)
  if m:
    return (int(m.group(1)), name)
  return (0, name)


def detect_remote_spec(spec: str) -> bool:
  return ":" in spec and not Path(spec).exists()


def basename_from_spec(spec: str) -> str:
  raw = spec.split(":", 1)[1] if detect_remote_spec(spec) else spec
  return Path(raw.rstrip("/")).name


def canonical_dataset_root(task: str, policy: str, collection_datetime: str) -> Path:
  return HARDWARE_DATA_ROOT / task / policy / collection_datetime


def canonical_merged_root(
  task: str, policy: str, merge_datetime: str | None = None
) -> Path:
  return (
    HARDWARE_DATA_ROOT / task / policy / "merged" / (merge_datetime or timestamp_slug())
  )


def default_dataset_root(args: argparse.Namespace) -> Path:
  if getattr(args, "dataset_root", None):
    return Path(args.dataset_root).resolve()
  return canonical_dataset_root(
    args.task, args.policy, args.collection_datetime
  ).resolve()


def ensure_dataset_dirs(dataset_root: Path) -> dict[str, Path]:
  paths = {
    "raw": dataset_root / "raw",
    "inspection": dataset_root / "inspection",
    "trimmed": dataset_root / "trimmed",
    "processed": dataset_root / "processed",
    "artifacts": dataset_root / "artifacts",
  }
  dataset_root.mkdir(parents=True, exist_ok=True)
  for path in paths.values():
    path.mkdir(parents=True, exist_ok=True)
  return paths


def manifest_path(dataset_root: Path) -> Path:
  return dataset_root / "dataset_manifest.json"


def trim_manifest_path(dataset_root: Path) -> Path:
  return dataset_root / "artifacts" / "trim_manifest.json"


def load_json(path: Path, default=None):
  if not path.exists():
    return default
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def write_json(path: Path, payload: dict):
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")


def init_manifest(
  dataset_root: Path,
  *,
  task: str,
  policy: str,
  collection_datetime: str | None,
  dataset_type: str,
  parents: list[str] | None = None,
) -> dict:
  manifest = {
    "schema_version": MANIFEST_SCHEMA_VERSION,
    "layout_version": 1,
    "dataset_type": dataset_type,
    "dataset_root": str(dataset_root),
    "task": task,
    "policy": policy,
    "collection_datetime": collection_datetime,
    "created_at": now_iso(),
    "updated_at": now_iso(),
    "paths": {
      "raw": "raw",
      "inspection": "inspection",
      "trimmed": "trimmed",
      "processed": "processed",
      "artifacts": "artifacts",
    },
    "status": {
      "synced": False,
      "inspected": False,
      "trimmed": False,
      "postprocessed": False,
      "packaged": False,
      "rendered": False,
      "combined": dataset_type == "merged",
    },
    "data_schema": None,
    "estimator": None,
    "runs": {},
    "parents": parents or [],
    "notes": [],
  }
  return manifest


def load_or_init_manifest(
  dataset_root: Path,
  *,
  task: str,
  policy: str,
  collection_datetime: str | None,
  dataset_type: str = "collection",
  parents: list[str] | None = None,
) -> dict:
  path = manifest_path(dataset_root)
  manifest = load_json(path)
  if manifest is None:
    manifest = init_manifest(
      dataset_root,
      task=task,
      policy=policy,
      collection_datetime=collection_datetime,
      dataset_type=dataset_type,
      parents=parents,
    )
  manifest["dataset_root"] = str(dataset_root)
  manifest["task"] = task
  manifest["policy"] = policy
  if collection_datetime is not None:
    manifest["collection_datetime"] = collection_datetime
  manifest["updated_at"] = now_iso()
  return manifest


def save_manifest(dataset_root: Path, manifest: dict):
  manifest["updated_at"] = now_iso()
  write_json(manifest_path(dataset_root), manifest)


def write_dataset_readme(dataset_root: Path, manifest: dict):
  readme_path = dataset_root / "README.md"
  lines = [
    f"# {manifest['task']} / {manifest['policy']}",
    "",
    f"- Dataset root: `{dataset_root}`",
    f"- Type: `{manifest['dataset_type']}`",
    f"- Collection datetime: `{manifest.get('collection_datetime')}`",
    f"- Status: `{', '.join(k for k, v in manifest['status'].items() if v) or 'none'}`",
    f"- Runs tracked: `{len(manifest.get('runs', {}))}`",
  ]
  if manifest.get("estimator"):
    lines.append(f"- Estimator: `{manifest['estimator'].get('path')}`")
  if manifest.get("parents"):
    lines.append(f"- Parents: `{len(manifest['parents'])}`")
  if manifest.get("data_schema"):
    schema = manifest["data_schema"]
    lines.append(
      "- Schema: "
      f"state={schema['state_dim']}, action={schema['action_dim']}, "
      f"extension={schema['extension_dim']}, contact={schema['contact_dim']}"
    )
  lines.extend(
    [
      "",
      "Authoritative files:",
      "- `dataset_manifest.json` for dataset-level provenance and stage status",
      "- `artifacts/trim_manifest.json` for trim suggestions and approved intervals",
      "- `processed/state_action_data_*.csv` for canonical training data",
    ]
  )
  readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def infer_csv_prefix(data_dir: Path) -> str:
  if list(data_dir.glob("state_action_data_*.csv")):
    return "state_action_data"
  if list(data_dir.glob("data_*.csv")):
    return "data"
  raise FileNotFoundError(
    f"No rollout CSVs found in {data_dir}. Expected data_*.csv or state_action_data_*.csv."
  )


def list_csv_shards(data_dir: Path) -> list[Path]:
  prefix = infer_csv_prefix(data_dir)
  return sorted(data_dir.glob(f"{prefix}_*.csv"), key=natural_key)


def find_rollout_dirs(base_dir: Path) -> list[Path]:
  if not base_dir.exists():
    return []
  rollout_dirs = []
  for dirpath, _, filenames in os.walk(base_dir):
    has_csv = any(
      name.startswith("data_")
      and name.endswith(".csv")
      or name.startswith("state_action_data_")
      and name.endswith(".csv")
      for name in filenames
    )
    if has_csv:
      rollout_dirs.append(Path(dirpath))
  rollout_dirs.sort()
  return rollout_dirs


def rollout_relpath(dataset_root: Path, rollout_dir: Path) -> str:
  return rollout_dir.relative_to(dataset_root).as_posix()


def load_rollout_array(
  data_dir: Path, total_cols: int | None = None
) -> tuple[np.ndarray, str]:
  npy_path = data_dir / "all_data.npy"
  if npy_path.exists():
    arr = np.load(npy_path, mmap_mode="r")
    if total_cols is not None and arr.shape[1] != total_cols:
      raise ValueError(f"{npy_path}: expected {total_cols} columns, got {arr.shape[1]}")
    return np.asarray(arr, dtype=np.float32), "all_data"

  prefix = infer_csv_prefix(data_dir)
  shards = list_csv_shards(data_dir)
  chunks = []
  for shard in shards:
    rows = []
    with open(shard, "r", newline="", encoding="utf-8") as f:
      reader = csv.reader(f)
      for row in reader:
        if row:
          rows.append([float(v) for v in row])
    if rows:
      arr = np.array(rows, dtype=np.float32)
      if total_cols is not None and arr.shape[1] != total_cols:
        raise ValueError(f"{shard}: expected {total_cols} columns, got {arr.shape[1]}")
      chunks.append(arr)
  if not chunks:
    raise ValueError(f"No non-empty CSV rows found in {data_dir}")
  return np.concatenate(chunks, axis=0), prefix


def remove_existing_shards(output_dir: Path, prefix: str):
  for path in output_dir.glob(f"{prefix}_*.csv"):
    path.unlink()


def reset_processed_stage(processed_dir: Path):
  if not processed_dir.exists():
    processed_dir.mkdir(parents=True, exist_ok=True)
    return
  for child in processed_dir.iterdir():
    if child.is_dir():
      shutil.rmtree(child)
    else:
      child.unlink()


def read_row_count(csv_path: Path) -> int:
  with open(csv_path, "r", encoding="utf-8") as f:
    return sum(1 for _ in f)


def summarize_rollout_dir(data_dir: Path) -> RolloutSummary:
  prefix = infer_csv_prefix(data_dir)
  shards = list_csv_shards(data_dir)
  if not shards:
    raise FileNotFoundError(f"No shards found in {data_dir}")
  sample = np.loadtxt(shards[0], delimiter=",", max_rows=1, dtype=np.float32)
  if sample.ndim == 0:
    sample = np.expand_dims(sample, 0)
  n_cols = int(sample.shape[-1])
  n_rows = sum(read_row_count(path) for path in shards)
  return RolloutSummary(
    relpath=data_dir.as_posix(),
    csv_prefix=prefix,
    n_rows=n_rows,
    n_cols=n_cols,
  )


def detect_schema(
  data: np.ndarray, extension_dim: int | None, contact_dim: int | None
) -> dict:
  total_cols = int(data.shape[1])
  remaining = total_cols - _STATE_DIM - _ACTION_DIM - _TERMINATION_DIM
  if remaining < 0:
    raise ValueError(f"Row width {total_cols} is too small for the expected layout")
  if extension_dim is None:
    extension_dim = 15 if contact_dim is None else remaining - int(contact_dim)
  if contact_dim is None:
    contact_dim = remaining - int(extension_dim)
  if extension_dim < 0 or contact_dim < 0:
    raise ValueError(
      f"Invalid schema inference for total_cols={total_cols}, "
      f"extension_dim={extension_dim}, contact_dim={contact_dim}"
    )
  if (
    _STATE_DIM + _ACTION_DIM + extension_dim + contact_dim + _TERMINATION_DIM
    != total_cols
  ):
    raise ValueError(
      "Schema does not match total columns: "
      f"state={_STATE_DIM}, action={_ACTION_DIM}, extension={extension_dim}, "
      f"contact={contact_dim}, total={total_cols}"
    )
  return {
    "state_dim": _STATE_DIM,
    "action_dim": _ACTION_DIM,
    "extension_dim": int(extension_dim),
    "contact_dim": int(contact_dim),
    "termination_dim": _TERMINATION_DIM,
    "total_cols": total_cols,
  }


def compute_signals(data: np.ndarray) -> dict[str, np.ndarray]:
  actions = data[:, _STATE_DIM : _STATE_DIM + _ACTION_DIM]
  ang_vel = data[:, 3:6]
  gravity = data[:, 6:9]
  joint_vel = data[:, 38:67]

  action_delta = np.zeros_like(actions)
  if len(actions) > 1:
    action_delta[1:] = actions[1:] - actions[:-1]

  return {
    "gravity_z": gravity[:, 2],
    "ang_vel_norm": np.linalg.norm(ang_vel, axis=1),
    "joint_vel_norm": np.linalg.norm(joint_vel, axis=1),
    "joint_vel_max_abs": np.max(np.abs(joint_vel), axis=1),
    "action_norm": np.linalg.norm(actions, axis=1),
    "action_delta_norm": np.linalg.norm(action_delta, axis=1),
  }


def find_low_delta_spans(
  values: np.ndarray,
  *,
  threshold: float,
  min_run_length: int,
) -> list[dict]:
  """Find spans where a 1D signal stays below a threshold for a while."""
  mask = np.asarray(values) <= threshold
  spans: list[dict] = []
  start = None
  for idx, is_low in enumerate(mask):
    if is_low and start is None:
      start = idx
    elif not is_low and start is not None:
      run_length = idx - start
      if run_length >= min_run_length:
        spans.append(
          {
            "start_row": int(start),
            "end_row": int(idx - 1),
            "length": int(run_length),
            "threshold": float(threshold),
          }
        )
      start = None
  if start is not None:
    run_length = len(mask) - start
    if run_length >= min_run_length:
      spans.append(
        {
          "start_row": int(start),
          "end_row": int(len(mask) - 1),
          "length": int(run_length),
          "threshold": float(threshold),
        }
      )
  return spans


def suggest_trim_interval(data: np.ndarray) -> dict:
  sig = compute_signals(data)
  motion_mask = (sig["joint_vel_norm"] > 0.25) | (sig["action_delta_norm"] > 0.02)
  if motion_mask.any():
    start = int(np.argmax(motion_mask))
  else:
    start = 0

  stable_mask = (
    (sig["gravity_z"] < -0.75)
    & (sig["ang_vel_norm"] < 4.5)
    & (sig["joint_vel_max_abs"] < 18.0)
  )
  stable_after_start = np.where(stable_mask & (np.arange(len(data)) >= start))[0]
  end = int(stable_after_start[-1]) if len(stable_after_start) else len(data) - 1
  if end < start:
    end = len(data) - 1

  reason = "auto-generated from motion/orientation heuristics"
  frozen_tail_spans = find_low_delta_spans(
    sig["action_delta_norm"], threshold=1e-4, min_run_length=25
  )
  if frozen_tail_spans:
    last_span = frozen_tail_spans[-1]
    if last_span["end_row"] >= len(data) - 1 and last_span["start_row"] > start:
      end = min(end, int(last_span["start_row"] - 1))
      reason += "; trimmed trailing near-frozen action span"

  return {
    "name": "main",
    "start_row": start,
    "end_row": end,
    "reason": reason,
  }


def inspection_summary(data: np.ndarray, control_dt: float) -> dict:
  sig = compute_signals(data)
  suggestion = suggest_trim_interval(data)
  duration_s = len(data) * control_dt
  keep_rows = suggestion["end_row"] - suggestion["start_row"] + 1
  exact_repeat_spans = find_low_delta_spans(
    sig["action_delta_norm"], threshold=0.0, min_run_length=2
  )
  near_frozen_action_spans = find_low_delta_spans(
    sig["action_delta_norm"], threshold=1e-4, min_run_length=10
  )
  return {
    "n_rows": int(len(data)),
    "duration_s": float(duration_s),
    "gravity_z_min": float(sig["gravity_z"].min()),
    "gravity_z_max": float(sig["gravity_z"].max()),
    "ang_vel_norm_max": float(sig["ang_vel_norm"].max()),
    "joint_vel_norm_max": float(sig["joint_vel_norm"].max()),
    "joint_vel_max_abs_max": float(sig["joint_vel_max_abs"].max()),
    "action_norm_max": float(sig["action_norm"].max()),
    "action_delta_norm_min": float(sig["action_delta_norm"].min()),
    "action_delta_norm_max": float(sig["action_delta_norm"].max()),
    "suggested_keep_ratio": float(keep_rows / max(len(data), 1)),
    "exact_action_repeat_span_count": int(len(exact_repeat_spans)),
    "near_frozen_action_span_count": int(len(near_frozen_action_spans)),
    "near_frozen_action_spans": near_frozen_action_spans,
    "suggested_interval": suggestion,
  }


def write_inspection_plot(
  data: np.ndarray, output_png: Path, control_dt: float, title: str
):
  sig = compute_signals(data)
  t = np.arange(len(data)) * control_dt
  summary = inspection_summary(data, control_dt)
  interval = summary["suggested_interval"]

  fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
  plot_items = [
    ("gravity_z", sig["gravity_z"]),
    ("ang_vel_norm", sig["ang_vel_norm"]),
    ("joint_vel_max_abs", sig["joint_vel_max_abs"]),
    ("action_norm", sig["action_norm"]),
    ("action_delta_norm", sig["action_delta_norm"]),
  ]
  for ax, (label, values) in zip(axes, plot_items, strict=True):
    ax.plot(t, values, linewidth=1.0)
    ax.set_ylabel(label)
    ax.axvline(
      interval["start_row"] * control_dt, color="green", linestyle="--", linewidth=1.0
    )
    ax.axvline(
      interval["end_row"] * control_dt, color="red", linestyle="--", linewidth=1.0
    )
    ax.grid(alpha=0.3)
  axes[-1].set_xlabel("Time (s)")
  fig.suptitle(title)
  fig.tight_layout()
  output_png.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_png, dpi=140)
  plt.close(fig)


def build_trim_manifest(
  dataset_root: Path, stage: str, suggestions: dict[str, dict]
) -> dict:
  existing = load_json(trim_manifest_path(dataset_root), default=None)
  manifest = existing or {
    "dataset_root": str(dataset_root),
    "stage": stage,
    "generated_at": now_iso(),
    "runs": {},
  }
  manifest["stage"] = stage
  manifest["generated_at"] = now_iso()
  runs = cast(dict[str, dict], manifest["runs"])

  for relpath, suggestion in suggestions.items():
    run_entry = runs.get(relpath, {})
    approved = run_entry.get("approved_intervals")
    if not approved:
      approved = [suggestion["suggested_interval"]]
    runs[relpath] = {
      "source_relpath": suggestion["source_relpath"],
      "summary_relpath": suggestion["summary_relpath"],
      "plot_relpath": suggestion["plot_relpath"],
      "suggested_intervals": [suggestion["suggested_interval"]],
      "approved_intervals": approved,
    }
  return manifest


def write_segment_meta(path: Path, payload: dict):
  with open(path / "meta.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")


def copy_source_tree(source_spec: str, dest_dir: Path, overwrite: bool):
  if dest_dir.exists():
    if not overwrite:
      raise FileExistsError(
        f"{dest_dir} already exists. Use --overwrite to replace it."
      )
    shutil.rmtree(dest_dir)
  dest_dir.parent.mkdir(parents=True, exist_ok=True)
  if detect_remote_spec(source_spec):
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
      ["rsync", "-a", f"{source_spec.rstrip('/')}/", f"{dest_dir}/"],
      check=True,
    )
  else:
    shutil.copytree(source_spec, dest_dir)


def consolidate_state_action_dir(data_dir: Path, overwrite: bool = True) -> Path:
  rollout_dirs = find_rollout_dirs(data_dir)
  if not rollout_dirs:
    raise FileNotFoundError(f"No state_action_data_*.csv shards found in {data_dir}")
  output_path = data_dir / "all_data.npy"
  if output_path.exists():
    if overwrite:
      output_path.unlink()
    else:
      raise FileExistsError(
        f"{output_path} already exists. Use --overwrite to replace it."
      )

  sample = np.loadtxt(
    list_csv_shards(rollout_dirs[0])[0],
    delimiter=",",
    max_rows=1,
    dtype=np.float32,
  )
  if sample.ndim == 0:
    sample = np.expand_dims(sample, 0)
  total_cols = int(sample.shape[-1])
  total_rows = sum(
    read_row_count(path)
    for rollout_dir in rollout_dirs
    for path in list_csv_shards(rollout_dir)
  )
  arr = np.lib.format.open_memmap(
    output_path, mode="w+", dtype=np.float32, shape=(total_rows, total_cols)
  )

  offset = 0
  for rollout_dir in rollout_dirs:
    rollout_arrays = []
    for path in list_csv_shards(rollout_dir):
      data = np.loadtxt(path, delimiter=",", dtype=np.float32)
      if data.ndim == 1:
        data = data[None, :]
      rollout_arrays.append(data)
    rollout_data = np.concatenate(rollout_arrays, axis=0)
    # Mark rollout boundaries in the packaged artifact so downstream
    # windowing never crosses between separate deployments/segments.
    rollout_data[-1, -1] = 1.0
    arr[offset : offset + len(rollout_data)] = rollout_data
    offset += len(rollout_data)
  arr.flush()
  return output_path


def ensure_same_policy_compatibility(manifests: list[dict]):
  base = manifests[0]
  fields = ("task", "policy")
  for field in fields:
    if any(m[field] != base[field] for m in manifests[1:]):
      raise ValueError(f"Cannot combine datasets with different {field}s")
  base_schema = base.get("data_schema")
  if not base_schema:
    raise ValueError("Input dataset is missing data_schema in manifest")
  for manifest in manifests[1:]:
    if manifest.get("data_schema") != base_schema:
      raise ValueError("Cannot combine datasets with different data_schema")


def render_rollout_dir(
  rollout_dir: Path,
  output_path: Path,
  *,
  width: int,
  height: int,
  stride: int,
  control_dt: float,
  start_row: int | None = None,
  end_row: int | None = None,
):
  import imageio
  import mujoco
  from imageio.core.format import Format

  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

  os.environ.setdefault("MUJOCO_GL", "egl")

  data_arr, _ = load_rollout_array(rollout_dir)
  n_rows = len(data_arr)
  start = 0 if start_row is None else int(start_row)
  end = n_rows - 1 if end_row is None else int(end_row)
  if start < 0 or end < start or end >= n_rows:
    raise ValueError(
      f"Invalid render window for {rollout_dir}: start_row={start_row}, "
      f"end_row={end_row}, n_rows={n_rows}"
    )
  data_arr = data_arr[start : end + 1]
  joint_pos = data_arr[:, 9:38]
  projected_gravity = data_arr[:, 6:9]
  ext_col_start = _STATE_DIM + _ACTION_DIM
  if data_arr.shape[1] >= ext_col_start + 5:
    base_height = data_arr[:, ext_col_start + 4]
    base_height = np.where(
      np.isfinite(base_height) & (base_height > 0.2),
      base_height,
      0.78,
    )
  else:
    base_height = np.full(len(data_arr), 0.78, dtype=np.float32)

  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  data = mujoco.MjData(model)
  free_joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
  )
  if free_joint_id < 0:
    raise ValueError(f"floating_base_joint not found in {G1_XML.name}")
  free_qpos_adr = model.jnt_qposadr[free_joint_id]

  option = mujoco.MjvOption()
  # Show visual meshes, hide collision geoms.
  option.geomgroup[2] = 1
  option.geomgroup[3] = 0
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.distance = 2.6
  camera.azimuth = 135.0
  camera.elevation = -18.0
  camera.lookat[:] = np.array([0.0, 0.0, 0.75], dtype=np.float64)

  joint_qpos = {}
  # Canonical hardware rollout state stores joint positions in MuJoCo order.
  for joint_name in G1_HARDWARE_JOINT_NAMES:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
      raise ValueError(f"Joint {joint_name} not found in {G1_XML.name}")
    joint_qpos[joint_name] = model.jnt_qposadr[jid]

  fps = max(1, round(1.0 / (control_dt * max(stride, 1))))
  output_path.parent.mkdir(parents=True, exist_ok=True)

  world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)

  def quat_from_projected_gravity(gravity_body: np.ndarray) -> np.ndarray:
    """Recover roll/pitch from projected gravity with yaw fixed to zero."""
    a = np.asarray(gravity_body, dtype=np.float64)
    norm = np.linalg.norm(a)
    if norm < 1e-8:
      return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    a = a / norm
    dot = float(np.clip(np.dot(a, world_gravity), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
      return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if dot < -1.0 + 1e-8:
      # 180 deg ambiguity: choose a fixed sideways axis.
      axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
      return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float64)

    axis = np.cross(a, world_gravity)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
      return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / axis_norm
    angle = np.arccos(dot)
    half = 0.5 * angle
    sin_half = np.sin(half)
    quat = np.array(
      [np.cos(half), axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half],
      dtype=np.float64,
    )
    return quat / np.linalg.norm(quat)

  with (
    imageio.get_writer(output_path, fps=fps) as writer,
    mujoco.Renderer(model, height=height, width=width) as renderer,
  ):
    for row, gravity, z in zip(
      joint_pos[::stride],
      projected_gravity[::stride],
      base_height[::stride],
      strict=False,
    ):
      # Keep x/y fixed, but match the recorded base height and roll/pitch.
      data.qpos[free_qpos_adr : free_qpos_adr + 3] = np.array(
        [0.0, 0.0, float(z)], dtype=np.float64
      )
      data.qpos[free_qpos_adr + 3 : free_qpos_adr + 7] = quat_from_projected_gravity(
        gravity
      )
      for idx, joint_name in enumerate(G1_HARDWARE_JOINT_NAMES):
        data.qpos[joint_qpos[joint_name]] = float(row[idx])
      mujoco.mj_forward(model, data)
      renderer.update_scene(data, camera=camera, scene_option=option)
      frame = renderer.render()
      cast(Format.Writer, writer).append_data(frame)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Canonical hardware data pipeline")
  sub = parser.add_subparsers(dest="command", required=True)

  sync = sub.add_parser(
    "sync", help="Copy raw rollout directories into a canonical dataset root"
  )
  sync.add_argument("--dataset-root", type=str, default=None)
  sync.add_argument("--task", type=str, default="wtw")
  sync.add_argument("--policy", type=str, required=True)
  sync.add_argument("--collection-datetime", type=str, required=True)
  sync.add_argument(
    "--source",
    action="append",
    required=True,
    help="Local path or host:/path rollout dir",
  )
  sync.add_argument(
    "--overwrite", action="store_true", help="Replace an existing raw run directory"
  )

  inspect = sub.add_parser(
    "inspect", help="Compute summary stats and plots for rollout dirs"
  )
  inspect.add_argument("--dataset-root", type=str, required=True)
  inspect.add_argument(
    "--stage", choices=("raw", "trimmed", "processed"), default="raw"
  )
  inspect.add_argument("--control-dt", type=float, default=DEFAULT_CONTROL_DT)

  trim = sub.add_parser(
    "trim", help="Materialize approved intervals from artifacts/trim_manifest.json"
  )
  trim.add_argument("--dataset-root", type=str, required=True)
  trim.add_argument("--rows-per-file", type=int, default=10_000)
  trim.add_argument("--overwrite", action="store_true")
  trim.add_argument(
    "--use-suggested",
    action="store_true",
    help="Use suggested intervals when no approved intervals exist",
  )

  post = sub.add_parser(
    "postprocess",
    help="Prepare canonical shards without state estimation",
    description=(
      "Prepare canonical shards without changing recorded state. "
      "Use contact-root-relabel --estimator-root to apply state estimation."
    ),
  )
  post.add_argument("--dataset-root", type=str, required=True)
  post.add_argument("--input-stage", choices=("trimmed", "raw"), default=None)
  post.add_argument("--extension-dim", type=int, default=None)
  post.add_argument("--contact-dim", type=int, default=None)
  post.add_argument("--rows-per-file", type=int, default=10_000)
  post.add_argument(
    "--overwrite",
    action="store_true",
    help="Replace an existing processed/ stage. Omit this for relabeling into new sibling roots.",
  )

  package = sub.add_parser(
    "package", help="Create all_data.npy from processed canonical shards"
  )
  package.add_argument("--dataset-root", type=str, required=True)
  package.add_argument("--overwrite", action="store_true")

  contact_relabel = sub.add_parser(
    "contact-root-relabel",
    help="Estimate state with the six Contact-UAN networks and stance-foot geometry",
  )
  contact_relabel.add_argument("--source-root", type=str, required=True)
  contact_relabel.add_argument("--output-root", type=str, required=True)
  contact_relabel.add_argument(
    "--estimator-root",
    type=str,
    default=None,
    help="Root containing the six networks written by train_estimators.py.",
  )
  contact_relabel.add_argument("--base-lin-estimator-dir", type=str, default=None)
  contact_relabel.add_argument("--foot-contact-estimator-dir", type=str, default=None)
  contact_relabel.add_argument(
    "--left-stance-foot-height-estimator-dir", type=str, default=None
  )
  contact_relabel.add_argument(
    "--right-stance-foot-height-estimator-dir", type=str, default=None
  )
  contact_relabel.add_argument(
    "--left-stance-foot-projected-gravity-estimator-dir", type=str, default=None
  )
  contact_relabel.add_argument(
    "--right-stance-foot-projected-gravity-estimator-dir", type=str, default=None
  )
  contact_relabel.add_argument(
    "--input-stage", choices=("trimmed", "raw"), default="trimmed"
  )
  contact_relabel.add_argument("--task", type=str, default="Mjlab-WTW-Unitree-G1")
  contact_relabel.add_argument("--device", type=str, default="cuda:0")
  contact_relabel.add_argument("--extension-dim", type=int, default=15)
  contact_relabel.add_argument("--contact-dim", type=int, default=2)
  contact_relabel.add_argument("--rows-per-file", type=int, default=10_000)
  contact_relabel.add_argument("--solver-batch-size", type=int, default=128)
  contact_relabel.add_argument("--contact-threshold", type=float, default=0.5)
  contact_relabel.add_argument("--geometry-contact-threshold", type=float, default=0.8)
  contact_relabel.add_argument("--default-base-height", type=float, default=0.75)
  contact_relabel.add_argument("--overwrite", action="store_true")

  combine = sub.add_parser(
    "combine", help="Combine compatible same-policy datasets into a merged dataset root"
  )
  combine.add_argument(
    "--input", action="append", required=True, help="Input dataset root"
  )
  combine.add_argument("--output-root", type=str, default=None)
  combine.add_argument("--merge-datetime", type=str, default=None)
  combine.add_argument("--rows-per-file", type=int, default=10_000)
  combine.add_argument("--consolidate", action="store_true")

  render = sub.add_parser(
    "render", help="Render fixed-base MP4s for raw or trimmed rollouts"
  )
  render.add_argument("--dataset-root", type=str, required=True)
  render.add_argument("--stage", choices=("raw", "trimmed"), default="trimmed")
  render.add_argument("--output-dir", type=str, default=None)
  render.add_argument("--control-dt", type=float, default=DEFAULT_CONTROL_DT)
  render.add_argument("--stride", type=int, default=2)
  render.add_argument("--width", type=int, default=640)
  render.add_argument("--height", type=int, default=480)
  render.add_argument("--start-row", type=int, default=None)
  render.add_argument("--end-row", type=int, default=None)

  report = sub.add_parser(
    "report", help="Write a dataset summary report under artifacts/"
  )
  report.add_argument("--dataset-root", type=str, required=True)

  return parser


def cmd_sync(args: argparse.Namespace):
  dataset_root = default_dataset_root(args)
  ensure_dataset_dirs(dataset_root)
  manifest = load_or_init_manifest(
    dataset_root,
    task=args.task,
    policy=args.policy,
    collection_datetime=args.collection_datetime,
  )

  for source in args.source:
    run_name = basename_from_spec(source)
    dest_dir = dataset_root / "raw" / run_name
    print(f"[sync] {source} -> {dest_dir}")
    copy_source_tree(source, dest_dir, overwrite=args.overwrite)
    summary = summarize_rollout_dir(dest_dir)
    manifest["runs"][run_name] = {
      "source_spec": source,
      "raw_relpath": f"raw/{run_name}",
      "csv_prefix": summary.csv_prefix,
      "n_rows": summary.n_rows,
      "n_cols": summary.n_cols,
      "migration_status": "migrated",
    }

  manifest["status"]["synced"] = True
  save_manifest(dataset_root, manifest)
  write_dataset_readme(dataset_root, manifest)
  print(f"[sync] Updated {manifest_path(dataset_root)}")


def cmd_inspect(args: argparse.Namespace):
  dataset_root = Path(args.dataset_root).resolve()
  manifest = load_json(manifest_path(dataset_root))
  if manifest is None:
    raise FileNotFoundError(
      f"No dataset manifest found at {manifest_path(dataset_root)}"
    )

  stage_root = dataset_root / args.stage
  rollout_dirs = find_rollout_dirs(stage_root)
  if not rollout_dirs:
    raise FileNotFoundError(f"No rollout dirs found under {stage_root}")

  suggestions = {}
  for rollout_dir in rollout_dirs:
    rel = rollout_dir.relative_to(stage_root).as_posix()
    data, _ = load_rollout_array(rollout_dir)
    summary = inspection_summary(data, args.control_dt)

    out_dir = dataset_root / "inspection" / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "summary.png"
    summary_path = out_dir / "summary.json"

    write_inspection_plot(data, plot_path, args.control_dt, title=f"{args.stage}/{rel}")
    write_json(summary_path, summary)

    suggestions[rel] = {
      "source_relpath": rollout_relpath(dataset_root, rollout_dir),
      "summary_relpath": summary_path.relative_to(dataset_root).as_posix(),
      "plot_relpath": plot_path.relative_to(dataset_root).as_posix(),
      "suggested_interval": summary["suggested_interval"],
    }
    print(f"[inspect] {rollout_dir} -> {summary_path}")

  trim_doc = build_trim_manifest(dataset_root, args.stage, suggestions)
  write_json(trim_manifest_path(dataset_root), trim_doc)

  manifest["status"]["inspected"] = True
  save_manifest(dataset_root, manifest)
  write_dataset_readme(dataset_root, manifest)
  print(f"[inspect] Trim manifest written to {trim_manifest_path(dataset_root)}")


def cmd_trim(args: argparse.Namespace):
  dataset_root = Path(args.dataset_root).resolve()
  manifest = load_json(manifest_path(dataset_root))
  trim_doc = load_json(trim_manifest_path(dataset_root))
  if manifest is None or trim_doc is None:
    raise FileNotFoundError(
      "Dataset manifest and trim manifest are required before trimming"
    )

  outputs_written = 0
  for relpath, run_info in trim_doc.get("runs", {}).items():
    intervals = run_info.get("approved_intervals")
    if not intervals and args.use_suggested:
      intervals = run_info.get("suggested_intervals", [])
    if not intervals:
      continue

    source_dir = dataset_root / run_info["source_relpath"]
    data, prefix = load_rollout_array(source_dir)
    for interval in intervals:
      start = int(interval["start_row"])
      end = int(interval["end_row"])
      name = interval.get("name", "segment")
      if start < 0 or end >= len(data) or end < start:
        raise ValueError(f"Invalid trim interval for {relpath}: {interval}")

      dest_dir = dataset_root / "trimmed" / relpath / name
      if dest_dir.exists():
        if not args.overwrite:
          raise FileExistsError(f"{dest_dir} exists. Use --overwrite to replace it.")
        shutil.rmtree(dest_dir)
      dest_dir.mkdir(parents=True, exist_ok=True)

      trimmed = data[start : end + 1]
      write_processed_files(
        trimmed, str(dest_dir), args.rows_per_file, output_prefix=prefix
      )
      write_segment_meta(
        dest_dir,
        {
          "source_relpath": run_info["source_relpath"],
          "interval": interval,
          "n_rows": int(len(trimmed)),
          "csv_prefix": prefix,
        },
      )
      outputs_written += 1
      print(f"[trim] {run_info['source_relpath']}[{start}:{end}] -> {dest_dir}")

  manifest["status"]["trimmed"] = outputs_written > 0
  save_manifest(dataset_root, manifest)
  write_dataset_readme(dataset_root, manifest)


def cmd_postprocess(args: argparse.Namespace):
  dataset_root = Path(args.dataset_root).resolve()
  manifest = load_json(manifest_path(dataset_root))
  if manifest is None:
    raise FileNotFoundError(
      f"No dataset manifest found at {manifest_path(dataset_root)}"
    )

  input_stage = args.input_stage
  if input_stage is None:
    input_stage = "trimmed" if find_rollout_dirs(dataset_root / "trimmed") else "raw"
  stage_root = dataset_root / input_stage
  rollout_dirs = find_rollout_dirs(stage_root)
  if not rollout_dirs:
    raise FileNotFoundError(f"No rollout dirs found under {stage_root}")

  source_segments = []
  schema = None
  processed_dir = dataset_root / "processed"
  if processed_dir.exists() and any(processed_dir.iterdir()) and not args.overwrite:
    raise FileExistsError(
      f"{processed_dir} is not empty. Refusing to overwrite existing processed data. "
      "Use a new sibling dataset root for relabeling, or pass --overwrite explicitly."
    )
  reset_processed_stage(processed_dir)
  total_processed_rows = 0
  for rollout_dir in rollout_dirs:
    raw, _ = load_rollout_array(rollout_dir)
    rollout_meta = load_json(rollout_dir / "meta.json", default={}) or {}
    schema = schema or detect_schema(raw, args.extension_dim, args.contact_dim)
    input_action_representation = rollout_meta.get(
      "action_representation", "absolute_joint_target"
    )
    input_joint_order = rollout_meta.get("joint_order", "mujoco")
    if input_joint_order != "mujoco":
      raise ValueError(
        f"{rollout_dir} has joint_order={input_joint_order!r}; "
        "canonical hardware datasets must be saved in MuJoCo order"
      )
    if input_action_representation != "absolute_joint_target":
      raise ValueError(
        f"{rollout_dir} has action_representation={input_action_representation!r}; "
        "canonical hardware datasets must save absolute joint targets"
      )
    chunk = raw
    rel_output_dir = rollout_dir.relative_to(stage_root)
    dest_dir = processed_dir / rel_output_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    write_processed_files(
      chunk,
      str(dest_dir),
      args.rows_per_file,
      output_prefix="state_action_data",
    )
    write_json(
      dest_dir / "meta.json",
      {
        "source_relpath": rollout_relpath(dataset_root, rollout_dir),
        "source_stage": input_stage,
        "n_rows": int(len(chunk)),
        "schema": schema,
        "lin_vel_estimated": False,
        "ang_vel_estimated": False,
        "root_height_estimated": False,
        "foot_heights_estimated": False,
        "projected_gravity_estimated": False,
        "joint_velocity_estimated": False,
        "estimator_target_layout": [],
        "input_mujoco_order": True,
        "output_mujoco_order": True,
        "input_action_representation": "absolute_joint_target",
        "output_action_representation": "absolute_joint_target",
      },
    )
    total_processed_rows += int(len(chunk))
    source_segments.append(
      {
        "source_relpath": rollout_relpath(dataset_root, rollout_dir),
        "processed_relpath": rel_output_dir.as_posix(),
        "n_rows": int(len(chunk)),
        "input_joint_order": "mujoco",
        "input_action_representation": "absolute_joint_target",
      }
    )
    print(f"[postprocess] {rollout_dir} -> {len(chunk):,} rows")

  processed_meta = {
    "source_stage": input_stage,
    "estimator_dir": None,
    "n_rows": total_processed_rows,
    "source_segments": source_segments,
    "schema": schema,
    "lin_vel_estimated": False,
    "ang_vel_estimated": False,
    "root_height_estimated": False,
    "foot_heights_estimated": False,
    "projected_gravity_estimated": False,
    "joint_velocity_estimated": False,
    "estimator_target_layout": [],
    "preserve_rollout_boundaries": True,
    "input_mujoco_order": True,
    "output_mujoco_order": True,
    "input_action_representation": "absolute_joint_target",
    "output_action_representation": "absolute_joint_target",
  }
  write_json(processed_dir / "meta.json", processed_meta)

  manifest["data_schema"] = schema
  manifest["estimator"] = None
  manifest["processed_sources"] = source_segments
  manifest["status"]["postprocessed"] = True
  manifest["status"]["packaged"] = False
  artifacts = manifest.get("artifacts", {})
  for key in ("all_data_npy", "baseline_mse_npy", "baseline_mse_meta_npz"):
    artifacts.pop(key, None)
  manifest["artifacts"] = artifacts
  save_manifest(dataset_root, manifest)
  write_dataset_readme(dataset_root, manifest)


def cmd_package(args: argparse.Namespace):
  dataset_root = Path(args.dataset_root).resolve()
  manifest = load_json(manifest_path(dataset_root))
  if manifest is None:
    raise FileNotFoundError(
      f"No dataset manifest found at {manifest_path(dataset_root)}"
    )
  processed_dir = dataset_root / "processed"
  output = consolidate_state_action_dir(processed_dir, overwrite=args.overwrite)
  manifest["status"]["packaged"] = True
  manifest.setdefault("artifacts", {})
  manifest["artifacts"] = manifest.get("artifacts", {})
  manifest["artifacts"]["all_data_npy"] = str(output.relative_to(dataset_root))
  save_manifest(dataset_root, manifest)
  write_dataset_readme(dataset_root, manifest)
  print(f"[package] Wrote {output}")


def cmd_contact_root_relabel(args: argparse.Namespace):
  script = Path(__file__).with_name("relabel_hardware_stance_geometry_mjlab.py")
  cmd = [
    sys.executable,
    str(script),
    "--source-root",
    args.source_root,
    "--output-root",
    args.output_root,
    "--input-stage",
    args.input_stage,
    "--task",
    args.task,
    "--device",
    args.device,
    "--extension-dim",
    str(args.extension_dim),
    "--contact-dim",
    str(args.contact_dim),
    "--rows-per-file",
    str(args.rows_per_file),
    "--solver-batch-size",
    str(args.solver_batch_size),
    "--contact-threshold",
    str(args.contact_threshold),
    "--geometry-contact-threshold",
    str(args.geometry_contact_threshold),
    "--default-base-height",
    str(args.default_base_height),
  ]
  if args.estimator_root:
    cmd.extend(["--estimator-root", args.estimator_root])
  if args.base_lin_estimator_dir:
    cmd.extend(["--base-lin-estimator-dir", args.base_lin_estimator_dir])
  if args.foot_contact_estimator_dir:
    cmd.extend(["--foot-contact-estimator-dir", args.foot_contact_estimator_dir])
  if args.left_stance_foot_height_estimator_dir:
    cmd.extend(
      [
        "--left-stance-foot-height-estimator-dir",
        args.left_stance_foot_height_estimator_dir,
      ]
    )
  if args.right_stance_foot_height_estimator_dir:
    cmd.extend(
      [
        "--right-stance-foot-height-estimator-dir",
        args.right_stance_foot_height_estimator_dir,
      ]
    )
  if args.left_stance_foot_projected_gravity_estimator_dir:
    cmd.extend(
      [
        "--left-stance-foot-projected-gravity-estimator-dir",
        args.left_stance_foot_projected_gravity_estimator_dir,
      ]
    )
  if args.right_stance_foot_projected_gravity_estimator_dir:
    cmd.extend(
      [
        "--right-stance-foot-projected-gravity-estimator-dir",
        args.right_stance_foot_projected_gravity_estimator_dir,
      ]
    )
  if args.overwrite:
    cmd.append("--overwrite")
  subprocess.check_call(cmd)


def cmd_combine(args: argparse.Namespace):
  input_roots = [Path(path).resolve() for path in args.input]
  manifests = []
  for root in input_roots:
    manifest = load_json(manifest_path(root))
    if manifest is None:
      raise FileNotFoundError(f"No manifest found at {manifest_path(root)}")
    manifests.append(manifest)
  ensure_same_policy_compatibility(manifests)
  base = manifests[0]
  output_root = (
    Path(args.output_root).resolve()
    if args.output_root
    else canonical_merged_root(
      base["task"], base["policy"], args.merge_datetime
    ).resolve()
  )
  ensure_dataset_dirs(output_root)
  out_manifest = load_or_init_manifest(
    output_root,
    task=base["task"],
    policy=base["policy"],
    collection_datetime=args.merge_datetime or timestamp_slug(),
    dataset_type="merged",
    parents=[str(root) for root in input_roots],
  )

  chunks = []
  parents = []
  for root, manifest in zip(input_roots, manifests, strict=True):
    processed_dir = root / "processed"
    data, _ = load_rollout_array(
      processed_dir, total_cols=manifest["data_schema"]["total_cols"]
    )
    chunks.append(data)
    parents.append(
      {
        "dataset_root": str(root),
        "n_rows": int(len(data)),
        "manifest_path": str(manifest_path(root)),
      }
    )
    print(f"[combine] {root} -> {len(data):,} rows")

  combined = np.concatenate(chunks, axis=0)
  processed_dir = output_root / "processed"
  remove_existing_shards(processed_dir, "state_action_data")
  write_processed_files(
    combined,
    str(processed_dir),
    args.rows_per_file,
    output_prefix="state_action_data",
  )
  processed_meta = {
    "combined_from": parents,
    "n_rows": int(len(combined)),
    "schema": base["data_schema"],
  }
  write_json(processed_dir / "meta.json", processed_meta)

  out_manifest["parents"] = [str(root) for root in input_roots]
  out_manifest["source_datasets"] = parents
  out_manifest["data_schema"] = base["data_schema"]
  out_manifest["status"]["combined"] = True
  out_manifest["status"]["postprocessed"] = True
  if args.consolidate:
    output = consolidate_state_action_dir(processed_dir, overwrite=True)
    out_manifest["status"]["packaged"] = True
    out_manifest.setdefault("artifacts", {})
    out_manifest["artifacts"]["all_data_npy"] = str(output.relative_to(output_root))
  save_manifest(output_root, out_manifest)
  write_dataset_readme(output_root, out_manifest)


def cmd_render(args: argparse.Namespace):
  dataset_root = Path(args.dataset_root).resolve()
  manifest = load_json(manifest_path(dataset_root))
  if manifest is None:
    raise FileNotFoundError(
      f"No dataset manifest found at {manifest_path(dataset_root)}"
    )

  stage_root = dataset_root / args.stage
  rollout_dirs = find_rollout_dirs(stage_root)
  if not rollout_dirs:
    raise FileNotFoundError(f"No rollout dirs found under {stage_root}")

  output_dir = (
    Path(args.output_dir).resolve()
    if args.output_dir
    else dataset_root / "artifacts" / "renders"
  )
  for rollout_dir in rollout_dirs:
    rel = rollout_dir.relative_to(stage_root)
    suffix = ""
    if args.start_row is not None or args.end_row is not None:
      start = 0 if args.start_row is None else int(args.start_row)
      end = "end" if args.end_row is None else str(int(args.end_row))
      suffix = f"_rows_{start}_{end}"
    output_path = output_dir / rel.parent / f"{rel.name}{suffix}.mp4"
    render_rollout_dir(
      rollout_dir,
      output_path,
      width=args.width,
      height=args.height,
      stride=args.stride,
      control_dt=args.control_dt,
      start_row=args.start_row,
      end_row=args.end_row,
    )
    print(f"[render] {rollout_dir} -> {output_path}")

  manifest["status"]["rendered"] = True
  save_manifest(dataset_root, manifest)
  write_dataset_readme(dataset_root, manifest)


def cmd_report(args: argparse.Namespace):
  dataset_root = Path(args.dataset_root).resolve()
  manifest = load_json(manifest_path(dataset_root))
  if manifest is None:
    raise FileNotFoundError(
      f"No dataset manifest found at {manifest_path(dataset_root)}"
    )

  report_payload = {
    "dataset_root": str(dataset_root),
    "task": manifest["task"],
    "policy": manifest["policy"],
    "dataset_type": manifest["dataset_type"],
    "status": manifest["status"],
    "run_count": len(manifest.get("runs", {})),
    "processed_sources": manifest.get("processed_sources", []),
    "data_schema": manifest.get("data_schema"),
    "parents": manifest.get("parents", []),
    "estimator": manifest.get("estimator"),
  }
  report_path = dataset_root / "artifacts" / "report.json"
  write_json(report_path, report_payload)
  print(json.dumps(report_payload, indent=2))
  print(f"[report] Wrote {report_path}")


def main():
  parser = build_parser()
  args = parser.parse_args()

  if args.command == "sync":
    cmd_sync(args)
  elif args.command == "inspect":
    cmd_inspect(args)
  elif args.command == "trim":
    cmd_trim(args)
  elif args.command == "postprocess":
    cmd_postprocess(args)
  elif args.command == "package":
    cmd_package(args)
  elif args.command == "contact-root-relabel":
    cmd_contact_root_relabel(args)
  elif args.command == "combine":
    cmd_combine(args)
  elif args.command == "render":
    cmd_render(args)
  elif args.command == "report":
    cmd_report(args)
  else:
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
  main()
