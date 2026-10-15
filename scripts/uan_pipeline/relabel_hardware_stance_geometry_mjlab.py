#!/usr/bin/env python3
"""Relabel hardware data with contact-gated stance-foot root height."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from estimator_bundle import load_paper_estimator_bundle
from estimator_config import COMPONENT_TARGETS
from hardware_pipeline import (
  _ACTION_DIM,
  _STATE_DIM,
  detect_schema,
  ensure_dataset_dirs,
  find_rollout_dirs,
  init_manifest,
  load_json,
  load_rollout_array,
  manifest_path,
  reset_processed_stage,
  rollout_relpath,
  save_manifest,
  write_dataset_readme,
  write_json,
)
from postprocess_real_data import (
  predict_estimator_targets,
  write_processed_files,
)
from stance_orientation import (
  contact_geometry_root_projected_gravity,
  normalize_np,
)
from stance_root_height import (
  RootHeightSolveCfg,
  compute_contact_geometry_root_z,
)

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--source-root", required=True, help="Existing hardware dataset root."
  )
  parser.add_argument("--output-root", required=True, help="New sibling dataset root.")
  parser.add_argument(
    "--estimator-root",
    default=None,
    help="Root containing the six separate paper estimators.",
  )
  parser.add_argument("--base-lin-estimator-dir", default=None)
  parser.add_argument("--foot-contact-estimator-dir", default=None)
  parser.add_argument("--left-stance-foot-height-estimator-dir", default=None)
  parser.add_argument("--right-stance-foot-height-estimator-dir", default=None)
  parser.add_argument(
    "--left-stance-foot-projected-gravity-estimator-dir", default=None
  )
  parser.add_argument(
    "--right-stance-foot-projected-gravity-estimator-dir", default=None
  )
  parser.add_argument("--input-stage", choices=("trimmed", "raw"), default="trimmed")
  parser.add_argument("--task", default="Mjlab-WTW-Unitree-G1")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--extension-dim", type=int, default=15)
  parser.add_argument("--contact-dim", type=int, default=2)
  parser.add_argument("--rows-per-file", type=int, default=10_000)
  parser.add_argument("--solver-batch-size", type=int, default=128)
  parser.add_argument("--contact-threshold", type=float, default=0.5)
  parser.add_argument(
    "--geometry-contact-threshold",
    type=float,
    default=0.8,
    help=(
      "Contact probability threshold for stance foot-height selection and "
      "orientation reconstruction."
    ),
  )
  parser.add_argument("--default-base-height", type=float, default=0.75)
  parser.add_argument("--overwrite", action="store_true")
  return parser.parse_args()


def _ensure_output_root(source_root: Path, output_root: Path, overwrite: bool) -> None:
  if output_root.exists():
    if not overwrite and any(output_root.iterdir()):
      raise FileExistsError(
        f"{output_root} already exists and is not empty. Choose a new "
        "sibling root or pass --overwrite."
      )
    if overwrite:
      shutil.rmtree(output_root)
  output_root.mkdir(parents=True, exist_ok=True)
  for name in ("raw", "inspection", "trimmed", "artifacts"):
    src = source_root / name
    dst = output_root / name
    if src.exists():
      shutil.copytree(src, dst)
  for name in ("README.md",):
    src = source_root / name
    if src.exists():
      shutil.copy2(src, output_root / name)


def _build_solver_env(
  args: argparse.Namespace, num_envs: int
) -> tuple[ManagerBasedRlEnv, torch.Tensor, list[int]]:
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = num_envs
  terrain_cfg = env_cfg.scene.terrain
  if terrain_cfg is not None and hasattr(terrain_cfg, "num_envs"):
    terrain_cfg.num_envs = num_envs
  env_cfg.sim.contact_sensor_maxmatch = max(
    getattr(env_cfg.sim, "contact_sensor_maxmatch", 64), 64
  )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
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


def _solve_contact_geometry_batch(
  env: ManagerBasedRlEnv,
  *,
  state_rows_np: np.ndarray,
  ext_rows_np: np.ndarray,
  predicted_foot_heights_np: np.ndarray,
  contact_prob_np: np.ndarray,
  joint_ids: torch.Tensor,
  foot_body_ids: list[int],
  cfg: RootHeightSolveCfg,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
  state_rows = torch.as_tensor(state_rows_np, dtype=torch.float32, device=env.device)
  ext_rows = torch.as_tensor(ext_rows_np, dtype=torch.float32, device=env.device)
  foot_heights = torch.as_tensor(
    predicted_foot_heights_np, dtype=torch.float32, device=env.device
  )
  contact_prob = torch.as_tensor(
    contact_prob_np, dtype=torch.float32, device=env.device
  )
  root_z, diag = compute_contact_geometry_root_z(
    env,
    state_rows=state_rows,
    ext_rows=ext_rows,
    predicted_foot_heights=foot_heights,
    contact_prob=contact_prob,
    joint_ids=joint_ids,
    foot_body_ids=foot_body_ids,
    cfg=cfg,
  )
  root_z_np = root_z.detach().cpu().numpy().astype(np.float32)
  fallback = diag["fallback_root_z"].detach().cpu().numpy().astype(np.float32)
  result = {
    "prior_root_z": fallback,
    "solved_root_z": root_z_np,
    "root_z_delta": root_z_np - fallback,
    "solver_score": np.zeros_like(root_z_np, dtype=np.float32),
    "used_stance_geometry": diag["used_stance_geometry"]
    .detach()
    .cpu()
    .numpy()
    .astype(bool),
    "active_contact_count": diag["active_contact_count"]
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32),
    "double_stance_disagreement": (
      diag["double_stance_disagreement"].detach().cpu().numpy().astype(np.float32)
    ),
  }
  return root_z_np, result


def _summarize_bool(values: np.ndarray) -> dict:
  values = np.asarray(values, dtype=bool)
  return {
    "count": int(values.size),
    "true_count": int(values.sum()),
    "true_fraction": float(values.mean()) if values.size else 0.0,
  }


def _summarize_float(values: np.ndarray) -> dict:
  values = np.asarray(values, dtype=np.float64).reshape(-1)
  values = values[np.isfinite(values)]
  return {
    "mean": float(values.mean()) if values.size else 0.0,
    "median": float(np.median(values)) if values.size else 0.0,
    "p95": float(np.percentile(values, 95)) if values.size else 0.0,
    "min": float(values.min()) if values.size else 0.0,
    "max": float(values.max()) if values.size else 0.0,
  }


def mark_rollout_end(data: np.ndarray) -> None:
  """Mark the last row of one independently collected replay rollout."""
  if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
    raise ValueError(f"Expected a non-empty 2D rollout array, got {data.shape}.")
  data[-1, -1] = 1.0


def main() -> None:
  args = parse_args()
  configure_torch_backends()
  os.environ.setdefault("MUJOCO_GL", "egl")

  bundles = load_paper_estimator_bundle(
    args.estimator_root,
    {group: getattr(args, f"{group}_estimator_dir") for group in COMPONENT_TARGETS},
  )
  base_bundle = bundles["base_lin"]
  contact_bundle = bundles["foot_contact"]
  left_stance_bundle = bundles["left_stance_foot_height"]
  right_stance_bundle = bundles["right_stance_foot_height"]
  left_orientation_bundle = bundles["left_stance_foot_projected_gravity"]
  right_orientation_bundle = bundles["right_stance_foot_projected_gravity"]
  estimator_dirs = {
    f"{group}_estimator_dir": bundle["dir"] for group, bundle in bundles.items()
  }

  source_root = Path(args.source_root).resolve()
  output_root = Path(args.output_root).resolve()
  if (
    source_root == output_root
    or source_root in output_root.parents
    or output_root in source_root.parents
  ):
    raise ValueError(
      "Source and output roots must be separate, non-nested directories."
    )
  stage_root = source_root / args.input_stage
  rollout_dirs = find_rollout_dirs(stage_root)
  if not rollout_dirs:
    raise FileNotFoundError(f"No rollout dirs found under {stage_root}")
  for rollout_dir in rollout_dirs:
    rollout_meta = load_json(rollout_dir / "meta.json", default={}) or {}
    if rollout_meta.get("joint_order", "mujoco") != "mujoco":
      raise ValueError(f"{rollout_dir}: relabeling requires MuJoCo joint order.")
    if rollout_meta.get("action_representation", "absolute_joint_target") != (
      "absolute_joint_target"
    ):
      raise ValueError(f"{rollout_dir}: relabeling requires absolute joint targets.")
  _ensure_output_root(source_root, output_root, args.overwrite)
  ensure_dataset_dirs(output_root)

  source_manifest = load_json(manifest_path(source_root), default={}) or {}
  manifest = init_manifest(
    output_root,
    task=source_manifest.get("task", "wtw"),
    policy=source_manifest.get("policy", "g1_wtw"),
    collection_datetime=source_manifest.get("collection_datetime"),
    dataset_type="collection",
    parents=[str(source_root)],
  )
  manifest.update(
    {
      k: v
      for k, v in source_manifest.items()
      if k not in ("dataset_root", "paths", "status", "estimator")
    }
  )
  manifest["dataset_root"] = str(output_root)
  manifest["parents"] = [str(source_root)]

  solver_envs = args.solver_batch_size
  env, joint_ids, foot_body_ids = _build_solver_env(args, solver_envs)
  solve_cfg = RootHeightSolveCfg(
    contact_threshold=args.contact_threshold,
    default_base_height=args.default_base_height,
  )

  processed_dir = output_root / "processed"
  reset_processed_stage(processed_dir)

  all_chunks = []
  diag_chunks: dict[str, list[np.ndarray]] = {
    "contact_prob": [],
    "prior_root_z": [],
    "solved_root_z": [],
    "root_z_delta": [],
    "solver_score": [],
    "used_stance_geometry": [],
    "active_contact_count": [],
    "double_stance_disagreement": [],
    "root_projected_gravity": [],
    "used_stance_orientation": [],
    "active_orientation_contact_count": [],
    "double_stance_orientation_disagreement_rad": [],
    "valid_prediction": [],
  }
  source_segments = []
  schema = None
  total_rows = 0

  for rollout_dir in rollout_dirs:
    raw, _ = load_rollout_array(rollout_dir)
    schema = schema or detect_schema(raw, args.extension_dim, args.contact_dim)
    extension_dim = schema["extension_dim"]
    if extension_dim < 5:
      raise ValueError(
        f"Hybrid root-height relabeling requires extension_dim >= 5, "
        f"got {extension_dim} for {rollout_dir}."
      )
    action_col_start = _STATE_DIM
    ext_col_start = action_col_start + _ACTION_DIM

    base_pred, base_valid = predict_estimator_targets(
      raw,
      base_bundle["estimator"],
      base_bundle["meta"],
      action_col_start,
    )
    contact_pred, contact_valid = predict_estimator_targets(
      raw,
      contact_bundle["estimator"],
      contact_bundle["meta"],
      action_col_start,
    )
    left_pred, left_valid = predict_estimator_targets(
      raw,
      left_stance_bundle["estimator"],
      left_stance_bundle["meta"],
      action_col_start,
    )
    right_pred, right_valid = predict_estimator_targets(
      raw,
      right_stance_bundle["estimator"],
      right_stance_bundle["meta"],
      action_col_start,
    )
    left_orientation_pred, left_orientation_valid = predict_estimator_targets(
      raw,
      left_orientation_bundle["estimator"],
      left_orientation_bundle["meta"],
      action_col_start,
    )
    right_orientation_pred, right_orientation_valid = predict_estimator_targets(
      raw,
      right_orientation_bundle["estimator"],
      right_orientation_bundle["meta"],
      action_col_start,
    )
    valid = (
      base_valid
      & contact_valid
      & left_valid
      & right_valid
      & left_orientation_valid
      & right_orientation_valid
    )

    processed = raw.copy()
    processed[valid, 0:3] = base_pred[valid]
    source_foot_heights = raw[:, ext_col_start : ext_col_start + 2].astype(
      np.float32,
      copy=True,
    )
    foot_heights = source_foot_heights.copy()

    contact_prob = np.clip(contact_pred, 0.0, 1.0).astype(np.float32)
    contact_col_start = ext_col_start + extension_dim
    if schema["contact_dim"] >= 2:
      processed[valid, contact_col_start : contact_col_start + 2] = contact_prob[valid]
    else:
      raise ValueError(
        f"Root-height relabeling requires at least 2 contact columns, "
        f"got contact_dim={schema['contact_dim']}."
      )

    stance_foot_heights = np.concatenate([left_pred, right_pred], axis=1)
    confident = contact_prob >= float(args.geometry_contact_threshold)
    foot_heights[valid & confident[:, 0], 0] = stance_foot_heights[
      valid & confident[:, 0], 0
    ]
    foot_heights[valid & confident[:, 1], 1] = stance_foot_heights[
      valid & confident[:, 1], 1
    ]
    stance_foot_projected_gravity = normalize_np(
      np.stack([left_orientation_pred, right_orientation_pred], axis=1)
    )

    processed[valid, ext_col_start + 0] = foot_heights[valid, 0]
    processed[valid, ext_col_start + 1] = foot_heights[valid, 1]

    prior_root_z = np.full(raw.shape[0], np.nan, dtype=np.float32)
    solved_root_z = np.full(raw.shape[0], np.nan, dtype=np.float32)
    root_z_delta = np.full(raw.shape[0], np.nan, dtype=np.float32)
    solver_score = np.zeros(raw.shape[0], dtype=np.float32)
    used_stance_geometry = np.zeros(raw.shape[0], dtype=bool)
    active_contact_count = np.zeros(raw.shape[0], dtype=np.float32)
    double_stance_disagreement = np.full(raw.shape[0], np.nan, dtype=np.float32)
    root_projected_gravity = normalize_np(
      processed[:, 6:9].astype(np.float32, copy=True)
    )
    used_stance_orientation = np.zeros(raw.shape[0], dtype=bool)
    active_orientation_contact_count = np.zeros(raw.shape[0], dtype=np.float32)
    double_stance_orientation_disagreement_rad = np.full(
      raw.shape[0],
      np.nan,
      dtype=np.float32,
    )

    valid_idx = np.nonzero(valid)[0]
    for start in range(0, len(valid_idx), args.solver_batch_size):
      batch_idx = valid_idx[start : start + args.solver_batch_size]
      root_z_b, diag = _solve_contact_geometry_batch(
        env=env,
        state_rows_np=processed[batch_idx, :_STATE_DIM],
        ext_rows_np=processed[
          batch_idx,
          ext_col_start : ext_col_start + extension_dim,
        ],
        predicted_foot_heights_np=foot_heights[batch_idx],
        contact_prob_np=contact_prob[batch_idx],
        joint_ids=joint_ids,
        foot_body_ids=foot_body_ids,
        cfg=solve_cfg,
      )
      processed[batch_idx, ext_col_start + 4] = root_z_b
      prior_root_z[batch_idx] = diag["prior_root_z"]
      solved_root_z[batch_idx] = diag["solved_root_z"]
      root_z_delta[batch_idx] = diag["root_z_delta"]
      solver_score[batch_idx] = diag["solver_score"]
      used_stance_geometry[batch_idx] = diag.get(
        "used_stance_geometry", np.zeros(len(batch_idx), dtype=bool)
      )
      active_contact_count[batch_idx] = diag.get(
        "active_contact_count", np.zeros(len(batch_idx), dtype=np.float32)
      )
      double_stance_disagreement[batch_idx] = diag.get(
        "double_stance_disagreement",
        np.full(len(batch_idx), np.nan, dtype=np.float32),
      )
      root_pg_b, pg_diag = contact_geometry_root_projected_gravity(
        env,
        state_rows_np=processed[batch_idx, :_STATE_DIM],
        ext_rows_np=processed[
          batch_idx,
          ext_col_start : ext_col_start + extension_dim,
        ],
        foot_projected_gravity_np=stance_foot_projected_gravity[batch_idx],
        contact_prob_np=contact_prob[batch_idx],
        joint_ids=joint_ids,
        foot_body_ids=foot_body_ids,
        contact_threshold=float(args.geometry_contact_threshold),
        default_base_height=float(args.default_base_height),
      )
      root_projected_gravity[batch_idx] = root_pg_b
      used_stance_orientation[batch_idx] = pg_diag["used_stance_orientation"]
      active_orientation_contact_count[batch_idx] = pg_diag[
        "active_orientation_contact_count"
      ]
      double_stance_orientation_disagreement_rad[batch_idx] = pg_diag[
        "double_stance_orientation_disagreement_rad"
      ]

    processed[valid, 6:9] = root_projected_gravity[valid]

    # The trimmed hardware clips are independent replay episodes even when their
    # source termination column is unset. Mark each relabeled clip boundary so
    # rollout windows cannot span two separate deployments in all_data.npy.
    mark_rollout_end(processed)

    rel_output_dir = rollout_dir.relative_to(stage_root)
    dest_dir = processed_dir / rel_output_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    write_processed_files(
      processed,
      str(dest_dir),
      args.rows_per_file,
      output_prefix="state_action_data",
    )
    write_json(
      dest_dir / "meta.json",
      {
        "source_relpath": rollout_relpath(source_root, rollout_dir),
        "source_stage": args.input_stage,
        "n_rows": int(len(processed)),
        "schema": schema,
        "root_height_relabel_mode": "hybrid_foot_height_stance_geometry",
        **estimator_dirs,
      },
    )
    all_chunks.append(processed)
    for key, value in (
      ("contact_prob", contact_prob),
      ("prior_root_z", prior_root_z),
      ("solved_root_z", solved_root_z),
      ("root_z_delta", root_z_delta),
      ("solver_score", solver_score),
      ("used_stance_geometry", used_stance_geometry),
      ("active_contact_count", active_contact_count),
      ("double_stance_disagreement", double_stance_disagreement),
      ("root_projected_gravity", root_projected_gravity),
      ("used_stance_orientation", used_stance_orientation),
      ("active_orientation_contact_count", active_orientation_contact_count),
      (
        "double_stance_orientation_disagreement_rad",
        double_stance_orientation_disagreement_rad,
      ),
      ("valid_prediction", valid),
    ):
      diag_chunks[key].append(np.asarray(value))
    total_rows += int(len(processed))
    source_segments.append(
      {
        "source_relpath": rollout_relpath(source_root, rollout_dir),
        "processed_relpath": rel_output_dir.as_posix(),
        "n_rows": int(len(processed)),
      }
    )
    print(
      f"[relabel] {rollout_dir} -> {len(processed):,} rows, "
      f"stance_geometry={used_stance_geometry[valid].mean() if valid.any() else 0.0:.2%}"
    )

  all_data = np.concatenate(all_chunks, axis=0).astype(np.float32, copy=False)
  np.save(processed_dir / "all_data.npy", all_data)
  diag_np = {key: np.concatenate(values, axis=0) for key, values in diag_chunks.items()}
  np.savez_compressed(
    processed_dir / "stance_root_height_diagnostics.npz",
    allow_pickle=False,
    **diag_np,
  )

  processed_meta = {
    "source_stage": args.input_stage,
    "n_rows": total_rows,
    "source_segments": source_segments,
    "schema": schema,
    "root_height_relabel_mode": "hybrid_foot_height_stance_geometry",
    **estimator_dirs,
    "solver": solve_cfg.__dict__,
    "stats": {
      "valid_prediction": _summarize_bool(diag_np["valid_prediction"]),
      "used_stance_geometry": _summarize_bool(diag_np["used_stance_geometry"]),
      "root_z_delta": _summarize_float(
        diag_np["root_z_delta"][diag_np["valid_prediction"]]
      ),
      "double_stance_disagreement": _summarize_float(
        diag_np["double_stance_disagreement"][
          diag_np["valid_prediction"] & diag_np["used_stance_geometry"]
        ]
      ),
      "used_stance_orientation": _summarize_bool(diag_np["used_stance_orientation"]),
      "double_stance_orientation_disagreement_rad": _summarize_float(
        diag_np["double_stance_orientation_disagreement_rad"][
          diag_np["valid_prediction"] & diag_np["used_stance_orientation"]
        ]
      ),
      "solver_score": _summarize_float(
        diag_np["solver_score"][diag_np["valid_prediction"]]
      ),
      "contact_prob": _summarize_float(diag_np["contact_prob"]),
    },
    "preserve_rollout_boundaries": True,
    "input_mujoco_order": True,
    "output_mujoco_order": True,
    "input_action_representation": "absolute_joint_target",
    "output_action_representation": "absolute_joint_target",
  }
  write_json(processed_dir / "meta.json", processed_meta)
  write_json(processed_dir / "stance_root_height_meta.json", processed_meta)

  manifest["data_schema"] = schema
  manifest["estimator"] = {
    "mode": "hybrid_foot_height_stance_geometry",
    **estimator_dirs,
  }
  manifest["processed_sources"] = source_segments
  manifest["status"] = manifest.get("status", {})
  manifest["status"]["postprocessed"] = True
  manifest["status"]["packaged"] = True
  manifest["artifacts"] = manifest.get("artifacts", {})
  manifest["artifacts"]["all_data_npy"] = "processed/all_data.npy"
  manifest["artifacts"]["stance_root_height_diagnostics"] = (
    "processed/stance_root_height_diagnostics.npz"
  )
  save_manifest(output_root, manifest)
  write_dataset_readme(output_root, manifest)

  print(f"[saved] {processed_dir / 'all_data.npy'} shape={all_data.shape}")
  print(json.dumps(processed_meta["stats"], indent=2))


if __name__ == "__main__":
  main()
