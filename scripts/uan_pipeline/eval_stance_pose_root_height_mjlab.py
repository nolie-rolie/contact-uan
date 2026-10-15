#!/usr/bin/env python3
"""Evaluate stance-foot estimators and root-height reconstruction on replay."""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import mjlab.tasks  # noqa: F401,E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab.tasks.uan.mdp.utils import write_recorded_state  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

from collect_estimator_data import read_stance_foot_orientation_targets  # noqa: E402
from postprocess_real_data import (  # noqa: E402
  load_estimator_bundle,
  predict_estimator_targets,
)
from stance_root_height import (  # noqa: E402
  RootHeightSolveCfg,
  compute_contact_geometry_root_z,
)


STATE_DIM = 96
ACTION_DIM = 29
EXTENSION_DIM = 15
CONTACT_DIM = 2
TERMINATION_DIM = 1
EXPECTED_COLS = STATE_DIM + ACTION_DIM + EXTENSION_DIM + CONTACT_DIM + TERMINATION_DIM
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
CONTACT_REQUIRED = ("left_foot_contact", "right_foot_contact")
LEFT_STANCE_REQUIRED = ("left_foot_height",)
RIGHT_STANCE_REQUIRED = ("right_foot_height",)
LEFT_STANCE_PROJECTED_GRAVITY_REQUIRED = (
  "left_foot_proj_grav_x",
  "left_foot_proj_grav_y",
  "left_foot_proj_grav_z",
)
RIGHT_STANCE_PROJECTED_GRAVITY_REQUIRED = (
  "right_foot_proj_grav_x",
  "right_foot_proj_grav_y",
  "right_foot_proj_grav_z",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--dataset",
    required=True,
    help="Labeled replay directory containing all_data.npy and optional metadata.json.",
  )
  parser.add_argument("--foot-contact-estimator-dir", required=True)
  parser.add_argument("--left-stance-foot-height-estimator-dir", required=True)
  parser.add_argument("--right-stance-foot-height-estimator-dir", required=True)
  parser.add_argument(
    "--left-stance-foot-projected-gravity-estimator-dir",
    required=True,
  )
  parser.add_argument(
    "--right-stance-foot-projected-gravity-estimator-dir",
    required=True,
  )
  parser.add_argument("--task", default="Mjlab-WTW-Unitree-G1")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--contact-threshold", type=float, default=0.5)
  parser.add_argument("--default-base-height", type=float, default=0.75)
  parser.add_argument("--output-dir", default=None)
  parser.add_argument(
    "--save-per-row",
    action="store_true",
    help="Write per_row_metrics.npz with row-level diagnostics.",
  )
  return parser.parse_args()


def _load_json(path: Path) -> dict:
  if not path.exists():
    return {}
  with path.open("r", encoding="utf-8") as f:
    data = json.load(f)
  if not isinstance(data, dict):
    raise ValueError(f"{path} must contain a JSON object.")
  return data


def _load_replay(dataset: Path) -> tuple[np.ndarray, dict]:
  data_path = dataset / "all_data.npy"
  if not data_path.exists():
    raise FileNotFoundError(f"Replay rows not found: {data_path}")
  raw = np.load(data_path).astype(np.float32, copy=False)
  if raw.ndim != 2:
    raise ValueError(f"{data_path} must be a 2D array, got shape={raw.shape}.")
  if raw.shape[1] != EXPECTED_COLS:
    raise ValueError(
      f"{data_path} must use state_96 | action_29 | extension_15 | "
      f"contact_2 | termination_1 ({EXPECTED_COLS} columns), got {raw.shape[1]}."
    )
  metadata = _load_json(dataset / "metadata.json")
  if int(metadata.get("extension_dim", EXTENSION_DIM)) != EXTENSION_DIM:
    raise ValueError(f"{dataset} must have extension_dim={EXTENSION_DIM}.")
  if int(metadata.get("contact_dim", CONTACT_DIM)) != CONTACT_DIM:
    raise ValueError(f"{dataset} must have contact_dim={CONTACT_DIM}.")
  return raw, metadata


def _target_indices(meta: dict, names: tuple[str, ...]) -> list[int]:
  layout = list(meta.get("target_layout", []))
  missing = [name for name in names if name not in layout]
  if missing:
    raise ValueError(
      f"{meta.get('target_group')} estimator is missing targets {missing}; "
      f"layout={layout}"
    )
  return [layout.index(name) for name in names]


def _load_bundle(estimator_dir: str, expected_group: str | None = None) -> dict:
  estimator, meta = load_estimator_bundle(estimator_dir)
  if expected_group is not None and meta.get("target_group") != expected_group:
    raise ValueError(
      f"{estimator_dir} has target_group={meta.get('target_group')!r}; "
      f"expected {expected_group!r}."
    )
  return {
    "dir": str(Path(estimator_dir).resolve()),
    "estimator": estimator,
    "meta": meta,
  }


def _normalize_vectors(values: np.ndarray) -> np.ndarray:
  norm = np.linalg.norm(values, axis=-1, keepdims=True)
  return values / np.clip(norm, 1.0e-8, None)


def _angle_error_deg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
  pred_n = _normalize_vectors(pred)
  target_n = _normalize_vectors(target)
  dot = np.sum(pred_n * target_n, axis=-1)
  return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))).astype(np.float32)


def _error_stats(error: np.ndarray, mask: np.ndarray | None = None) -> dict:
  values = np.asarray(error, dtype=np.float64).reshape(-1)
  finite = np.isfinite(values)
  if mask is not None:
    finite &= np.asarray(mask, dtype=bool).reshape(-1)
  values = values[finite]
  if values.size == 0:
    return {
      "count": 0,
      "mae": float("nan"),
      "rmse": float("nan"),
      "bias": float("nan"),
      "median": float("nan"),
      "p90": float("nan"),
      "p95": float("nan"),
    }
  abs_values = np.abs(values)
  return {
    "count": int(values.size),
    "mae": float(abs_values.mean()),
    "rmse": float(np.sqrt(np.square(values).mean())),
    "bias": float(values.mean()),
    "median": float(np.median(abs_values)),
    "p90": float(np.percentile(abs_values, 90)),
    "p95": float(np.percentile(abs_values, 95)),
  }


def _contact_stats(
  contact_prob: np.ndarray,
  true_contact: np.ndarray,
  *,
  threshold: float,
  mask: np.ndarray,
) -> dict:
  pred = contact_prob >= threshold
  target = true_contact >= 0.5
  valid = np.asarray(mask, dtype=bool)
  pred = pred[valid]
  target = target[valid]
  if pred.size == 0:
    return {
      "count": 0,
      "accuracy": float("nan"),
      "precision": float("nan"),
      "recall": float("nan"),
      "f1": float("nan"),
      "pred_contact_rate": float("nan"),
      "true_contact_rate": float("nan"),
    }
  tp = int(np.logical_and(pred, target).sum())
  fp = int(np.logical_and(pred, ~target).sum())
  fn = int(np.logical_and(~pred, target).sum())
  precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
  recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
  f1 = (
    2.0 * precision * recall / (precision + recall)
    if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
    else float("nan")
  )
  return {
    "count": int(pred.size),
    "accuracy": float((pred == target).mean()),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "pred_contact_rate": float(pred.mean()),
    "true_contact_rate": float(target.mean()),
  }


def root_height_methods_from_candidates(
  fallback_root_z: np.ndarray,
  left_root_z: np.ndarray,
  right_root_z: np.ndarray,
  contact_prob: np.ndarray,
  *,
  contact_threshold: float,
) -> dict[str, np.ndarray]:
  """Compute the three root-height selector outputs from root candidates."""
  fallback = np.asarray(fallback_root_z, dtype=np.float32)
  root_candidates = np.stack(
    [
      np.asarray(left_root_z, dtype=np.float32),
      np.asarray(right_root_z, dtype=np.float32),
    ],
    axis=1,
  )
  contact = np.asarray(contact_prob, dtype=np.float32)

  max_contact_idx = np.argmax(contact, axis=1)
  max_contact_prob = contact[np.arange(contact.shape[0]), max_contact_idx]
  max_contact_root = root_candidates[
    np.arange(root_candidates.shape[0]), max_contact_idx
  ]
  max_contact_root = np.where(
    max_contact_prob >= contact_threshold,
    max_contact_root,
    fallback,
  ).astype(np.float32)

  weights = np.where(
    contact >= contact_threshold,
    np.clip(contact, 0.0, None),
    0.0,
  ).astype(np.float32)
  weight_sum = weights.sum(axis=1)
  weighted = (root_candidates * weights).sum(axis=1) / np.clip(
    weight_sum,
    1.0e-6,
    None,
  )
  blend = np.where(weight_sum > 0.0, weighted, fallback).astype(np.float32)

  return {
    "lowest_foot": fallback.astype(np.float32),
    "max_contact": max_contact_root,
    "blend": blend,
  }


def _build_eval_env(
  *,
  task: str,
  num_envs: int,
  device: str,
) -> tuple[ManagerBasedRlEnv, torch.Tensor, list[int]]:
  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs = int(num_envs)
  terrain = getattr(env_cfg.scene, "terrain", None)
  if terrain is not None and hasattr(terrain, "num_envs"):
    terrain.num_envs = int(num_envs)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  robot = env.scene["robot"]
  joint_ids, _ = robot.find_joints((".*",))
  foot_body_ids, foot_body_names = robot.find_bodies(FOOT_BODY_NAMES)
  if len(foot_body_ids) != 2:
    raise RuntimeError(
      f"Expected foot bodies {FOOT_BODY_NAMES}, got {foot_body_names}."
    )
  return (
    env,
    torch.as_tensor(joint_ids, dtype=torch.long, device=env.device),
    foot_body_ids,
  )


def _ground_truth_foot_projected_gravity(
  env: ManagerBasedRlEnv,
  *,
  raw: np.ndarray,
  joint_ids: torch.Tensor,
  foot_body_ids: list[int],
  batch_size: int,
  default_base_height: float,
) -> np.ndarray:
  out = np.zeros((raw.shape[0], 2, 3), dtype=np.float32)
  robot = env.scene["robot"]
  device = env.device
  for start in range(0, raw.shape[0], batch_size):
    end = min(start + batch_size, raw.shape[0])
    n_rows = end - start
    state_rows = torch.as_tensor(
      raw[start:end, :STATE_DIM], dtype=torch.float32, device=device
    )
    ext_rows = torch.as_tensor(
      raw[start:end, STATE_DIM + ACTION_DIM : STATE_DIM + ACTION_DIM + EXTENSION_DIM],
      dtype=torch.float32,
      device=device,
    )
    row_ids = torch.arange(n_rows, dtype=torch.long, device=device)
    env_ids = torch.arange(n_rows, dtype=torch.long, device=device)
    write_recorded_state(
      env,
      robot,
      state_raw=state_rows,
      ext_raw=ext_rows,
      row_ids=row_ids,
      env_ids=env_ids,
      joint_ids=joint_ids,
      default_base_height=default_base_height,
      init_height_mode="root_height",
      foot_body_ids=None,
    )
    env.scene.write_data_to_sim()
    env.sim.forward()
    target = read_stance_foot_orientation_targets(robot.data, foot_body_ids)
    out[start:end] = target[:n_rows].reshape(n_rows, 2, 3).detach().cpu().numpy()
  return out


def _root_height_methods_from_replay(
  env: ManagerBasedRlEnv,
  *,
  raw: np.ndarray,
  predicted_foot_heights: np.ndarray,
  contact_prob: np.ndarray,
  joint_ids: torch.Tensor,
  foot_body_ids: list[int],
  batch_size: int,
  cfg: RootHeightSolveCfg,
) -> dict[str, np.ndarray]:
  lowest = np.zeros(raw.shape[0], dtype=np.float32)
  max_contact = np.zeros(raw.shape[0], dtype=np.float32)
  blend = np.zeros(raw.shape[0], dtype=np.float32)
  device = env.device
  ext_start = STATE_DIM + ACTION_DIM
  for start in range(0, raw.shape[0], batch_size):
    end = min(start + batch_size, raw.shape[0])
    state_rows = torch.as_tensor(
      raw[start:end, :STATE_DIM], dtype=torch.float32, device=device
    )
    ext_rows = torch.as_tensor(
      raw[start:end, ext_start : ext_start + EXTENSION_DIM],
      dtype=torch.float32,
      device=device,
    )
    foot_heights = torch.as_tensor(
      predicted_foot_heights[start:end],
      dtype=torch.float32,
      device=device,
    )
    contact = torch.as_tensor(
      contact_prob[start:end], dtype=torch.float32, device=device
    )
    root_z, diag = compute_contact_geometry_root_z(
      env,
      state_rows=state_rows,
      ext_rows=ext_rows,
      predicted_foot_heights=foot_heights,
      contact_prob=contact,
      joint_ids=joint_ids,
      foot_body_ids=foot_body_ids,
      cfg=cfg,
    )
    methods = root_height_methods_from_candidates(
      diag["fallback_root_z"].detach().cpu().numpy(),
      diag["left_root_z"].detach().cpu().numpy(),
      diag["right_root_z"].detach().cpu().numpy(),
      contact_prob[start:end],
      contact_threshold=cfg.contact_threshold,
    )
    lowest[start:end] = methods["lowest_foot"]
    max_contact[start:end] = methods["max_contact"]
    blend[start:end] = root_z.detach().cpu().numpy().astype(np.float32)
  return {
    "lowest_foot": lowest,
    "max_contact": max_contact,
    "blend": blend,
  }


def _write_root_height_csv(path: Path, root_metrics: dict[str, dict]) -> None:
  fieldnames = ["method", "count", "mae", "rmse", "bias", "median", "p90", "p95"]
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for method, stats in root_metrics.items():
      writer.writerow({"method": method, **{k: stats[k] for k in fieldnames[1:]}})


def _default_output_dir(dataset: Path) -> Path:
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  return REPO_ROOT / "logs/eval/stance_pose_root_height" / f"{dataset.name}_{stamp}"


def main() -> None:
  args = parse_args()
  configure_torch_backends()
  dataset = Path(args.dataset).expanduser().resolve()
  output_dir = (
    Path(args.output_dir).expanduser().resolve()
    if args.output_dir is not None
    else _default_output_dir(dataset)
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  raw, metadata = _load_replay(dataset)
  ext_start = STATE_DIM + ACTION_DIM
  contact_start = ext_start + EXTENSION_DIM
  true_foot_heights = raw[:, ext_start : ext_start + 2]
  true_root_z = raw[:, ext_start + 4]
  true_contact = raw[:, contact_start : contact_start + CONTACT_DIM]

  contact_bundle = _load_bundle(args.foot_contact_estimator_dir, "foot_contact")
  left_height_bundle = _load_bundle(
    args.left_stance_foot_height_estimator_dir,
    "left_stance_foot_height",
  )
  right_height_bundle = _load_bundle(
    args.right_stance_foot_height_estimator_dir,
    "right_stance_foot_height",
  )
  left_orientation_bundle = _load_bundle(
    args.left_stance_foot_projected_gravity_estimator_dir,
    "left_stance_foot_projected_gravity",
  )
  right_orientation_bundle = _load_bundle(
    args.right_stance_foot_projected_gravity_estimator_dir,
    "right_stance_foot_projected_gravity",
  )

  contact_indices = _target_indices(contact_bundle["meta"], CONTACT_REQUIRED)
  left_height_idx = _target_indices(left_height_bundle["meta"], LEFT_STANCE_REQUIRED)[0]
  right_height_idx = _target_indices(
    right_height_bundle["meta"], RIGHT_STANCE_REQUIRED
  )[0]
  left_orientation_indices = _target_indices(
    left_orientation_bundle["meta"],
    LEFT_STANCE_PROJECTED_GRAVITY_REQUIRED,
  )
  right_orientation_indices = _target_indices(
    right_orientation_bundle["meta"],
    RIGHT_STANCE_PROJECTED_GRAVITY_REQUIRED,
  )

  contact_pred, contact_valid = predict_estimator_targets(
    raw,
    contact_bundle["estimator"],
    contact_bundle["meta"],
    STATE_DIM,
  )
  left_height_pred, left_height_valid = predict_estimator_targets(
    raw,
    left_height_bundle["estimator"],
    left_height_bundle["meta"],
    STATE_DIM,
  )
  right_height_pred, right_height_valid = predict_estimator_targets(
    raw,
    right_height_bundle["estimator"],
    right_height_bundle["meta"],
    STATE_DIM,
  )
  left_orientation_pred, left_orientation_valid = predict_estimator_targets(
    raw,
    left_orientation_bundle["estimator"],
    left_orientation_bundle["meta"],
    STATE_DIM,
  )
  right_orientation_pred, right_orientation_valid = predict_estimator_targets(
    raw,
    right_orientation_bundle["estimator"],
    right_orientation_bundle["meta"],
    STATE_DIM,
  )

  contact_prob = np.clip(contact_pred[:, contact_indices], 0.0, 1.0).astype(np.float32)
  predicted_foot_heights = np.stack(
    [
      left_height_pred[:, left_height_idx],
      right_height_pred[:, right_height_idx],
    ],
    axis=1,
  ).astype(np.float32)
  predicted_projected_gravity = np.zeros((raw.shape[0], 2, 3), dtype=np.float32)
  predicted_projected_gravity[:, 0] = left_orientation_pred[:, left_orientation_indices]
  predicted_projected_gravity[:, 1] = right_orientation_pred[
    :, right_orientation_indices
  ]
  predicted_projected_gravity = _normalize_vectors(predicted_projected_gravity)

  root_valid = contact_valid & left_height_valid & right_height_valid
  pose_valid = root_valid & left_orientation_valid & right_orientation_valid
  finite_root = np.isfinite(true_root_z)
  finite_height = np.isfinite(true_foot_heights).all(axis=1)
  root_valid &= finite_root & finite_height
  pose_valid &= finite_height

  env, joint_ids, foot_body_ids = _build_eval_env(
    task=args.task,
    num_envs=int(args.batch_size),
    device=args.device,
  )
  try:
    true_projected_gravity = _ground_truth_foot_projected_gravity(
      env,
      raw=raw,
      joint_ids=joint_ids,
      foot_body_ids=foot_body_ids,
      batch_size=int(args.batch_size),
      default_base_height=float(args.default_base_height),
    )
    root_methods = _root_height_methods_from_replay(
      env,
      raw=raw,
      predicted_foot_heights=predicted_foot_heights,
      contact_prob=contact_prob,
      joint_ids=joint_ids,
      foot_body_ids=foot_body_ids,
      batch_size=int(args.batch_size),
      cfg=RootHeightSolveCfg(
        contact_threshold=float(args.contact_threshold),
        default_base_height=float(args.default_base_height),
      ),
    )
  finally:
    env.close()

  height_error = predicted_foot_heights - true_foot_heights
  angle_error = _angle_error_deg(predicted_projected_gravity, true_projected_gravity)
  true_contact_mask = true_contact >= 0.5

  root_metrics = {
    method: _error_stats(root_z - true_root_z, root_valid)
    for method, root_z in root_methods.items()
  }
  foot_names = ("left", "right")
  foot_height_metrics = {}
  orientation_metrics = {}
  contact_metrics = {
    "all": _contact_stats(
      contact_prob.reshape(-1),
      true_contact.reshape(-1),
      threshold=float(args.contact_threshold),
      mask=np.repeat(root_valid, 2),
    )
  }
  for foot_idx, foot_name in enumerate(foot_names):
    foot_mask = pose_valid
    stance_mask = pose_valid & true_contact_mask[:, foot_idx]
    foot_height_metrics[f"{foot_name}_all"] = _error_stats(
      height_error[:, foot_idx],
      foot_mask,
    )
    foot_height_metrics[f"{foot_name}_true_contact"] = _error_stats(
      height_error[:, foot_idx],
      stance_mask,
    )
    orientation_metrics[f"{foot_name}_all"] = _error_stats(
      angle_error[:, foot_idx],
      foot_mask,
    )
    orientation_metrics[f"{foot_name}_true_contact"] = _error_stats(
      angle_error[:, foot_idx],
      stance_mask,
    )
    contact_metrics[foot_name] = _contact_stats(
      contact_prob[:, foot_idx],
      true_contact[:, foot_idx],
      threshold=float(args.contact_threshold),
      mask=root_valid,
    )

  summary = {
    "dataset": str(dataset),
    "output_dir": str(output_dir),
    "num_rows": int(raw.shape[0]),
    "num_root_valid_rows": int(root_valid.sum()),
    "num_pose_valid_rows": int(pose_valid.sum()),
    "contact_threshold": float(args.contact_threshold),
    "metadata": {
      "step_dt": metadata.get("step_dt"),
      "actual_seconds": metadata.get("actual_seconds"),
      "num_envs": metadata.get("num_envs"),
    },
    "estimator_dirs": {
      "foot_contact": contact_bundle["dir"],
      "left_stance_foot_height": left_height_bundle["dir"],
      "right_stance_foot_height": right_height_bundle["dir"],
      "left_stance_foot_projected_gravity": left_orientation_bundle["dir"],
      "right_stance_foot_projected_gravity": right_orientation_bundle["dir"],
    },
    "root_height": root_metrics,
    "foot_height": foot_height_metrics,
    "projected_gravity_angle_deg": orientation_metrics,
    "contact": contact_metrics,
  }

  summary_path = output_dir / "summary.json"
  with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
    f.write("\n")
  _write_root_height_csv(output_dir / "per_method_root_height.csv", root_metrics)

  if args.save_per_row:
    np.savez_compressed(
      output_dir / "per_row_metrics.npz",
      true_root_z=true_root_z,
      root_valid=root_valid,
      pose_valid=pose_valid,
      true_foot_heights=true_foot_heights,
      predicted_foot_heights=predicted_foot_heights,
      foot_height_error=height_error,
      true_projected_gravity=true_projected_gravity,
      predicted_projected_gravity=predicted_projected_gravity,
      projected_gravity_angle_error_deg=angle_error,
      true_contact=true_contact,
      contact_prob=contact_prob,
      lowest_foot_root_z=root_methods["lowest_foot"],
      max_contact_root_z=root_methods["max_contact"],
      blend_root_z=root_methods["blend"],
    )

  print(f"[Eval] rows={raw.shape[0]:,}, root_valid={int(root_valid.sum()):,}")
  for method, stats in root_metrics.items():
    print(
      f"[Root] {method}: rmse={stats['rmse']:.5f} "
      f"mae={stats['mae']:.5f} count={stats['count']:,}"
    )
  print(f"[Eval] Wrote {summary_path}")


if __name__ == "__main__":
  main()
