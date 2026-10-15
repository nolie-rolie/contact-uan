"""Prepare hardware CSV shards without changing recorded state.

State estimation uses the six-network Contact-UAN pipeline:
    uv run python scripts/uan_pipeline/hardware_pipeline.py contact-root-relabel \\
        --source-root "<hardware_root>" --output-root "<labeled_hardware_root>" \\
        --estimator-root "<estimator_root>"

This module also provides inference helpers used by relabeling and evaluation.
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants matching DataCollector
# ---------------------------------------------------------------------------
_STATE_DIM = 96
_ACTION_DIM = 29
_DEFAULT_EXTENSION_DIM = 15
_DEFAULT_CONTACT_DIM = 2
_TERMINATION_DIM = 1

# Proprio term layout within state_96 used as estimator input:
# ang_vel: state[3:6], gravity: state[6:9],
# joint_pos: state[9:38], joint_vel: state[38:67], prev_q_des: previous action block
_PROPRIO_SLICE_ANG_VEL = slice(3, 6)
_PROPRIO_SLICE_GRAVITY = slice(6, 9)
_PROPRIO_SLICE_JOINT_POS = slice(9, 38)
_PROPRIO_SLICE_JOINT_VEL = slice(38, 67)
_EXPECTED_PROPRIO_TERMS = [
  "ang_vel_b",
  "proj_grav",
  "joint_pos_abs",
  "joint_vel",
  "prev_q_des",
]
_EXPECTED_PROPRIO_ACTION_REPRESENTATION = "prev_q_des_absolute_joint_target"


def parse_args(argv: list[str] | None = None):
  p = argparse.ArgumentParser(
    description=(
      "Prepare hardware CSV shards without state estimation. For estimation, "
      "use hardware_pipeline.py contact-root-relabel --estimator-root."
    )
  )
  p.add_argument(
    "--data_dir", required=True, help="Directory of raw CSV files from DataCollector"
  )
  p.add_argument(
    "--output_dir", required=True, help="Output directory for processed CSV files"
  )
  p.add_argument("--extension_dim", type=int, default=_DEFAULT_EXTENSION_DIM)
  p.add_argument("--contact_dim", type=int, default=_DEFAULT_CONTACT_DIM)
  p.add_argument(
    "--overwrite",
    action="store_true",
    help="Replace existing files in output_dir. Omit this when relabeling into new sibling roots.",
  )
  p.add_argument(
    "--rows_per_file", type=int, default=10_000, help="Rows per output CSV file"
  )
  p.add_argument(
    "--output_prefix",
    type=str,
    default="data",
    help="CSV shard prefix to write, e.g. 'data' or 'state_action_data'",
  )
  return p.parse_args(argv)


def ensure_output_dir_safe(output_dir: str, overwrite: bool) -> None:
  if not os.path.exists(output_dir):
    return
  if overwrite:
    return
  if any(os.scandir(output_dir)):
    raise FileExistsError(
      f"{output_dir!r} is not empty. Refusing to overwrite existing processed data. "
      "Choose a new sibling output directory or pass --overwrite explicitly."
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_raw_csv_files(data_dir: str, total_cols: int) -> np.ndarray:
  """Load all data_*.csv files in sorted order."""
  csv_files = sorted(glob.glob(os.path.join(data_dir, "data_*.csv")))
  if not csv_files:
    # Fall back to state_action_data_*.csv naming
    csv_files = sorted(glob.glob(os.path.join(data_dir, "state_action_data_*.csv")))
  if not csv_files:
    raise FileNotFoundError(
      f"No data CSV files found in {data_dir!r}. "
      "Expected data_*.csv or state_action_data_*.csv."
    )

  chunks = []
  for fpath in csv_files:
    rows = []
    with open(fpath, newline="") as f:
      reader = csv.reader(f)
      for row in reader:
        if row:
          rows.append([float(v) for v in row])
    if rows:
      arr = np.array(rows, dtype=np.float32)
      if total_cols > 0 and arr.shape[1] != total_cols:
        raise ValueError(f"{fpath}: expected {total_cols} columns, got {arr.shape[1]}")
      chunks.append(arr)

  raw = np.concatenate(chunks, axis=0)
  print(f"Loaded {len(raw):,} rows from {len(csv_files)} files")
  return raw


def load_rollout_meta(data_dir: str) -> dict | None:
  meta_path = os.path.join(data_dir, "meta.json")
  if not os.path.exists(meta_path):
    return None
  with open(meta_path, "r", encoding="utf-8") as f:
    return json.load(f)


def validate_estimator_meta(meta: dict, meta_path: str) -> None:
  proprio_terms = meta.get("proprio_terms")
  action_repr = meta.get("proprio_action_representation")
  if proprio_terms != _EXPECTED_PROPRIO_TERMS:
    raise ValueError(
      f"{meta_path} uses unsupported proprio_terms={proprio_terms!r}. "
      "This hardware postprocess path now requires estimators collected "
      "with previous q_des. Recollect estimator data and retrain."
    )
  if action_repr != _EXPECTED_PROPRIO_ACTION_REPRESENTATION:
    raise ValueError(
      f"{meta_path} uses unsupported proprio_action_representation="
      f"{action_repr!r}. This hardware postprocess path now requires "
      "previous q_des as the estimator action block."
    )


def load_estimator_bundle(estimator_dir: str):
  est_jit_path = os.path.join(estimator_dir, "state_estimator.jit")
  est_meta_path = os.path.join(estimator_dir, "estimator_meta.json")
  if not os.path.exists(est_jit_path):
    raise FileNotFoundError(f"Estimator not found: {est_jit_path}")
  if not os.path.exists(est_meta_path):
    raise FileNotFoundError(f"Estimator meta not found: {est_meta_path}")

  estimator = torch.jit.load(est_jit_path, map_location="cpu")
  estimator.eval()
  with open(est_meta_path, encoding="utf-8") as f:
    meta = json.load(f)
  validate_estimator_meta(meta, est_meta_path)
  return estimator, meta


# ---------------------------------------------------------------------------
# Estimator inference
# ---------------------------------------------------------------------------


def build_proprio_history_matrix(
  raw: np.ndarray,
  meta: dict,
  action_col_start: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Build flattened proprio histories matching estimator collection.

  Returns:
      (history, valid_mask), where ``history`` is ``(N, H * proprio_dim)``
      in chronological order and ``valid_mask`` is false for the first
      ``H - 1`` cold-start rows.
  """
  H = int(meta["history_length"])
  proprio_dim = int(meta["proprio_dim"])
  proprio_scales = meta.get("proprio_scales", [0.25, 1.0, 1.0, 0.05, 1.0])
  N = raw.shape[0]
  out = np.zeros((N, H * proprio_dim), dtype=np.float32)
  valid = np.zeros(N, dtype=bool)

  q_des = raw[:, action_col_start : action_col_start + _ACTION_DIM]
  history = np.zeros((H, proprio_dim), dtype=np.float32)
  for i in range(N):
    row = raw[i]
    state = row[:_STATE_DIM]
    ang_vel = state[_PROPRIO_SLICE_ANG_VEL] * proprio_scales[0]
    gravity = state[_PROPRIO_SLICE_GRAVITY] * proprio_scales[1]
    joint_pos = state[_PROPRIO_SLICE_JOINT_POS] * proprio_scales[2]
    joint_vel = state[_PROPRIO_SLICE_JOINT_VEL] * proprio_scales[3]
    prev_q_des = q_des[i - 1] if i > 0 else np.zeros(_ACTION_DIM, dtype=np.float32)
    prev_q_des = prev_q_des * proprio_scales[4]
    frame = np.concatenate([ang_vel, gravity, joint_pos, joint_vel, prev_q_des])

    history[:-1] = history[1:]
    history[-1] = frame
    out[i] = history.reshape(-1)
    valid[i] = i >= H - 1

  return out, valid


def predict_estimator_targets(
  raw: np.ndarray,
  estimator,
  meta: dict,
  action_col_start: int,
  *,
  batch_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
  """Run an estimator JIT and return raw predictions plus valid mask."""
  history, valid = build_proprio_history_matrix(raw, meta, action_col_start)
  output_dim = int(meta.get("output_dim", len(meta.get("target_layout", []))))
  predictions = np.zeros((raw.shape[0], output_dim), dtype=np.float32)
  loss_type = meta.get("loss_type", "mse")
  with torch.no_grad():
    for start in range(0, raw.shape[0], batch_size):
      end = min(start + batch_size, raw.shape[0])
      y = estimator(torch.from_numpy(history[start:end]))
      if loss_type == "bce_logits":
        y = torch.sigmoid(y)
      predictions[start:end] = y.cpu().numpy()
  return predictions, valid


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_processed_files(
  data: np.ndarray,
  output_dir: str,
  rows_per_file: int,
  output_prefix: str = "data",
):
  os.makedirs(output_dir, exist_ok=True)
  N = data.shape[0]
  n_files = (N + rows_per_file - 1) // rows_per_file
  for file_idx in range(n_files):
    start = file_idx * rows_per_file
    end = min(start + rows_per_file, N)
    fname = os.path.join(output_dir, f"{output_prefix}_{file_idx:05d}.csv")
    with open(fname, "w", newline="") as f:
      writer = csv.writer(f)
      for row in data[start:end]:
        writer.writerow(row.tolist())
  print(f"Wrote {N:,} rows across {n_files} files to {output_dir!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
  args = parse_args()
  input_meta = load_rollout_meta(args.data_dir)
  if input_meta is not None:
    joint_order = input_meta.get("joint_order", "mujoco")
    action_representation = input_meta.get(
      "action_representation", "absolute_joint_target"
    )
    if joint_order != "mujoco":
      raise ValueError(
        f"{args.data_dir} has joint_order={joint_order!r}; "
        "canonical hardware datasets must be saved in MuJoCo order"
      )
    if action_representation != "absolute_joint_target":
      raise ValueError(
        f"{args.data_dir} has action_representation={action_representation!r}; "
        "canonical hardware datasets must save absolute joint targets"
      )

  total_cols = (
    _STATE_DIM + _ACTION_DIM + args.extension_dim + args.contact_dim + _TERMINATION_DIM
  )
  # Load data
  print(f"Loading data from {args.data_dir!r} ...")
  raw = load_raw_csv_files(args.data_dir, total_cols=total_cols)

  # Write output
  ensure_output_dir_safe(args.output_dir, args.overwrite)
  write_processed_files(
    raw,
    args.output_dir,
    args.rows_per_file,
    output_prefix=args.output_prefix,
  )

  # Write meta
  out_meta = {
    "source_data_dir": os.path.abspath(args.data_dir),
    "estimator_dir": None,
    "n_rows": len(raw),
    "total_cols": total_cols,
    "state_dim": _STATE_DIM,
    "action_dim": _ACTION_DIM,
    "extension_dim": args.extension_dim,
    "contact_dim": args.contact_dim,
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
    "output_prefix": args.output_prefix,
  }
  with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
    json.dump(out_meta, f, indent=2)
  print(f"Done. Meta written to {args.output_dir}/meta.json")


if __name__ == "__main__":
  main()
