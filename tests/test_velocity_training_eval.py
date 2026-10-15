from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import torch


def _load_eval_module():
  module_path = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "uan_pipeline"
    / "eval_velocity_training_env_mjlab.py"
  )
  spec = importlib.util.spec_from_file_location(
    "eval_velocity_training_env_mjlab",
    module_path,
  )
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


eval_training = _load_eval_module()


def test_dynamic_saturation_fraction_uses_shared_threshold():
  torques = torch.tensor([[9.49, 4.74], [9.50, 4.75]])
  joint_vel = torch.tensor([[0.0, 5.0], [0.0, 5.0]])
  effort = torch.full((2, 2), 10.0)
  velocity_limit = torch.full((2, 2), 10.0)

  result = eval_training.dynamic_saturation_fraction(
    torques,
    joint_vel,
    effort,
    velocity_limit,
    threshold=0.95,
  )

  assert torch.allclose(result, torch.tensor([0.0, 1.0]))


def test_dynamic_saturation_fraction_uses_velocity_limit():
  torques = torch.tensor([[6.0], [6.0]])
  joint_vel = torch.tensor([[0.0], [5.0]])
  effort = torch.full((2, 1), 10.0)
  velocity_limit = torch.full((2, 1), 10.0)

  result = eval_training.dynamic_saturation_fraction(
    torques,
    joint_vel,
    effort,
    velocity_limit,
    threshold=0.95,
  )

  assert torch.allclose(result, torch.tensor([0.0, 1.0]))


def test_torque_metric_accumulation_stops_for_failed_envs():
  total = torch.zeros(2, dtype=torch.float64)
  count = torch.zeros(2, dtype=torch.long)

  step_1 = eval_training.sum_squared_torques(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
  eval_training._accumulate_alive_mean(
    total,
    count,
    step_1,
    torch.tensor([True, True]),
  )

  step_2 = eval_training.sum_squared_torques(torch.tensor([[2.0, 2.0], [5.0, 5.0]]))
  eval_training._accumulate_alive_mean(
    total,
    count,
    step_2,
    torch.tensor([True, False]),
  )

  assert torch.allclose(total, torch.tensor([13.0, 25.0], dtype=torch.float64))
  assert torch.equal(count, torch.tensor([2, 1]))


def test_checkpoint_manifest_generation_writes_checkpoint_rows(tmp_path):
  policies = {}
  labels = [
    "baseline",
    "custom_g1",
    "uan_custom_g1",
  ]
  for label in labels:
    run_dir = tmp_path / f"run_{label}"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "model_29999.pt"
    checkpoint.write_bytes(b"checkpoint")
    policies[label] = str(checkpoint)

  manifest = tmp_path / "checkpoints.csv"
  eval_training.write_checkpoint_manifest(policies, manifest)

  with manifest.open(newline="") as f:
    rows = list(csv.DictReader(f))

  assert len(rows) == 3
  assert rows[0]["label"] == "baseline"
  assert rows[0]["run"] == "run_baseline"
  assert rows[0]["checkpoint"].endswith("run_baseline/model_29999.pt")
  assert set(rows[0]) == {"label", "run", "checkpoint"}
