"""CSV data collector for real-world hardware data collection.

Produces CSV files matching the format of ``collect_g1_mixed_data.py``:

  [state_96D | action_29D | extension_15D | contact_ND | termination_1D]

State layout (96D):
  [0:3]   base_lin_vel (m/s, body frame) — zeros on hardware (filled by postprocessor)
  [3:6]   base_ang_vel (rad/s, body frame)
  [6:9]   projected_gravity (body frame)
  [9:38]  joint_pos (rad, absolute, MuJoCo order)
  [38:67] joint_vel (rad/s, MuJoCo order)
  [67:96] reconstructed PD torque + tau_ff (Nm, MuJoCo order)

Action layout (29D):
  Absolute joint position targets ``q_des`` in radians, stored in MuJoCo order.
  These are the final clipped commands sent to the low-level controller.

Extension layout (15D):
  [0:2]   foot heights L/R (m, world frame z)
  [2:4]   foot z-velocity L/R (m/s)
  [4]     base height (m)
  [5:11]  foot contact loads — zeros (no F/T sensor)
  [11]    ee_added_mass (kg) — 0.0 for WTW/velocity, configured for force control
  [12:15] applied_force_yaw (N, body frame) — zeros for WTW/velocity, active for force control

Contact layout (ND):
  Binary contact flags derived from foot heights (< threshold).

Termination (1D):
  Always 0.0 on hardware (no automatic resets).
"""

import csv
import json
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import traceback
from typing import Any, cast

import numpy as np

from deploy.utils.g1_kinematics import G1KinematicsHelper

_STATE_DIM = 96
_ACTION_DIM = 29
_EXTENSION_DIM = 15
_CONTACT_DIM = 2  # left foot, right foot
_TERMINATION_DIM = 1
_FOOT_CONTACT_HEIGHT_THRESHOLD = 0.05  # meters


class DataCollector:
  """Collects real-world deployment data into CSV files.

  Args:
      output_dir: Directory to write CSV files into.
      robot_configuration: Robot config with joint remapping utilities.
      kinematics: Optional ``G1KinematicsHelper`` instance for FK-based foot heights.
      control_dt: Control timestep in seconds (used for foot z-vel estimation).
      n_rows_per_file: Rows per output CSV file.
      collection_mode: ``"wtw"`` or ``"force_control"`` — controls how the
          extension's added-mass and applied-force fields are populated.
      ee_mass: End-effector added mass (kg) for force-control mode.
      joint_order: Joint ordering used for the state/action rows written to disk.
          Defaults to ``"mujoco"`` so downstream MJLab tooling can consume
          hardware datasets without extra remapping.
      action_representation: Representation used for the 29D action block.
          Defaults to ``"absolute_joint_target"`` so datasets record the
          actual joint position targets sent to hardware.
  """

  def __init__(
    self,
    output_dir: str,
    robot_configuration: Any,
    kinematics: G1KinematicsHelper | None,
    control_dt: float = 0.02,
    n_rows_per_file: int = 10_000,
    collection_mode: str = "wtw",
    ee_mass: float = 0.0,
    joint_order: str = "mujoco",
    action_representation: str = "absolute_joint_target",
  ):
    self.output_dir = output_dir
    self.robot_configuration = robot_configuration
    self.kinematics = kinematics
    self.control_dt = control_dt
    self.n_rows_per_file = n_rows_per_file
    self.collection_mode = collection_mode
    self.ee_mass = ee_mass
    self.joint_order = joint_order
    self.action_representation = action_representation

    os.makedirs(output_dir, exist_ok=True)

    self._file_idx = 0
    self._row_count = 0
    self._rows: list[list[float]] = []
    self._lock = threading.Lock()
    self._start_time = time.time()
    self._write_queue = mp.Queue()
    self._error_queue = mp.Queue()
    self._writer_error: Exception | None = None
    self._writer_process = mp.Process(
      target=_writer_process_main,
      args=(self.output_dir, self._write_queue, self._error_queue),
      name="data-collector-writer",
      daemon=True,
    )
    self._writer_process.start()

    # Metadata
    self._meta: dict[str, Any] = {
      "collection_mode": collection_mode,
      "state_dim": _STATE_DIM,
      "action_dim": _ACTION_DIM,
      "extension_dim": _EXTENSION_DIM,
      "contact_dim": _CONTACT_DIM,
      "termination_dim": _TERMINATION_DIM,
      "control_dt": control_dt,
      "ee_mass": ee_mass,
      "joint_order": joint_order,
      "action_representation": action_representation,
      "torque_representation": "reconstructed_pd_plus_tau_ff",
      "start_time": self._start_time,
    }

    print(
      f"[DataCollector] Writing to {output_dir!r}, "
      f"{n_rows_per_file} rows/file, mode={collection_mode!r}"
    )

  # -----------------------------------------------------------------------
  # Public API
  # -----------------------------------------------------------------------

  def write_step(
    self,
    state_96: np.ndarray,
    action_29: np.ndarray,
    extension_15: np.ndarray,
    contact_n: np.ndarray,
    termination: np.ndarray,
  ) -> None:
    """Append one row to the in-memory buffer, queueing full shards for async write."""
    self._raise_if_writer_failed()
    row = np.concatenate([state_96, action_29, extension_15, contact_n, termination])
    batch_to_flush: tuple[int, list[list[float]]] | None = None
    with self._lock:
      self._rows.append(row.tolist())
      self._row_count += 1
      if len(self._rows) >= self.n_rows_per_file:
        batch_to_flush = self._swap_full_buffer_locked()
    if batch_to_flush is not None:
      self._write_queue.put(batch_to_flush)

  def compute_extension(
    self,
    joint_pos_hw: np.ndarray,
    base_height: float,
    force_xyz: np.ndarray | None = None,
  ) -> np.ndarray:
    """Build the 15D extension vector.

    Args:
        joint_pos_hw: Joint positions in hardware (mujoco) order, shape (29,).
        base_height: Base height from LCM ``p[2]`` (metres).
        force_xyz: Applied end-effector force (N) for force-control mode.
            If None and mode is ``"force_control"``, zeros are used.

    Returns:
        extension_15 (np.ndarray, shape (15,))
    """
    fk = cast(G1KinematicsHelper, self.kinematics).compute(
      joint_pos_hw, base_height, dt=self.control_dt
    )
    ext = np.zeros(_EXTENSION_DIM)
    ext[0:2] = fk["foot_heights"]  # L, R foot z
    ext[2:4] = fk["foot_z_vel"]  # L, R foot z-velocity
    ext[4] = fk["base_height"]
    # ext[5:11] = zeros (foot contact loads)
    ext[11] = self.ee_mass

    if self.collection_mode == "force_control" and force_xyz is not None:
      ext[12:15] = np.asarray(force_xyz, dtype=float)[:3]
    # else ext[12:15] = zeros

    return ext

  def compute_placeholder_extension(
    self,
    base_height: float,
    force_xyz: np.ndarray | None = None,
  ) -> np.ndarray:
    """Build a lightweight extension vector without online FK.

    This is intended for live hardware collection. Foot heights,
    foot z-velocities, and contact-load slots are left as zeros so a
    later post-processing pass can reconstruct them off-robot from the
    recorded joint trajectories.
    """
    ext = np.zeros(_EXTENSION_DIM)
    ext[4] = float(base_height)
    ext[11] = self.ee_mass
    if self.collection_mode == "force_control" and force_xyz is not None:
      ext[12:15] = np.asarray(force_xyz, dtype=float)[:3]
    return ext

  def save_meta(self) -> None:
    """Flush remaining rows, wait for async writes, and write meta.json."""
    batch_to_flush: tuple[int, list[list[float]]] | None = None
    with self._lock:
      if self._rows:
        batch_to_flush = self._swap_full_buffer_locked()
    if batch_to_flush is not None:
      self._write_queue.put(batch_to_flush)

    self._write_queue.put(None)
    self._writer_process.join()
    self._raise_if_writer_failed()

    persisted_files = sorted(
      fname
      for fname in os.listdir(self.output_dir)
      if fname.startswith("data_") and fname.endswith(".csv")
    )
    persisted_rows = sum(
      _count_csv_rows(os.path.join(self.output_dir, fname)) for fname in persisted_files
    )
    persisted_file_count = len(persisted_files)

    meta_path = os.path.join(self.output_dir, "meta.json")
    self._meta["end_time"] = time.time()
    self._meta["total_rows"] = persisted_rows
    self._meta["n_files"] = persisted_file_count
    self._meta["expected_total_rows"] = self._row_count
    self._meta["expected_n_files"] = self._file_idx
    self._meta["persisted_total_rows"] = persisted_rows
    self._meta["persisted_n_files"] = persisted_file_count
    with open(meta_path, "w") as f:
      json.dump(self._meta, f, indent=2)
    if persisted_rows != self._row_count or persisted_file_count != self._file_idx:
      print(
        "[DataCollector] WARNING: persisted CSV shards do not match in-memory counts. "
        f"expected_rows={self._row_count}, persisted_rows={persisted_rows}, "
        f"expected_files={self._file_idx}, persisted_files={persisted_file_count}"
      )
    print(
      f"[DataCollector] Saved {persisted_rows} rows across {persisted_file_count} files. "
      f"Meta: {meta_path}"
    )

  # -----------------------------------------------------------------------
  # Internal helpers
  # -----------------------------------------------------------------------

  def _swap_full_buffer_locked(self) -> tuple[int, list[list[float]]]:
    """Detach the current row buffer and reserve the next shard index.

    Caller must hold ``self._lock``.
    """
    file_idx = self._file_idx
    rows = self._rows
    self._file_idx += 1
    self._rows = []
    return file_idx, rows

  def _raise_if_writer_failed(self) -> None:
    if self._writer_error is None:
      try:
        error_message = self._error_queue.get_nowait()
      except queue.Empty:
        error_message = None
      if error_message is not None:
        self._writer_error = RuntimeError(error_message)
    if self._writer_error is not None:
      raise RuntimeError(
        "DataCollector background writer failed"
      ) from self._writer_error


def _write_rows(output_dir: str, file_idx: int, rows: list[list[float]]) -> None:
  fname = os.path.join(output_dir, f"data_{file_idx:05d}.csv")
  with open(fname, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)


def _count_csv_rows(path: str) -> int:
  with open(path, "r", newline="") as f:
    return sum(1 for _ in f)


def _writer_process_main(
  output_dir: str,
  write_queue: mp.Queue,
  error_queue: mp.Queue,
) -> None:
  # Let the main deploy process handle Ctrl+C and drive graceful shutdown.
  # The parent always flushes pending rows and sends a ``None`` sentinel in
  # ``save_meta()``, so the writer should wait for that instead of bailing
  # out mid-collection when SIGINT is broadcast to the whole process group.
  signal.signal(signal.SIGINT, signal.SIG_IGN)

  while True:
    item = write_queue.get()
    try:
      if item is None:
        return
      file_idx, rows = item
      _write_rows(output_dir, file_idx, rows)
    except Exception:
      error_queue.put(traceback.format_exc())
      return
