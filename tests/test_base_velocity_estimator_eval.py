"""Regression coverage for the base-velocity rollout diagnostic."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def eval_module(monkeypatch):
  script_dir = Path(__file__).resolve().parents[1] / "scripts" / "uan_pipeline"
  monkeypatch.syspath_prepend(str(script_dir))
  return importlib.import_module("eval_estimator_mjlab")


def test_base_lin_compares_three_body_frame_velocity_targets(eval_module):
  robot_data = SimpleNamespace(
    root_link_quat_w=torch.tensor(
      [[1.0, 0.0, 0.0, 0.0], [2.0**-0.5, 0.0, 0.0, 2.0**-0.5]]
    ),
    root_link_lin_vel_w=torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
  )
  targets = eval_module.read_base_linear_velocity_targets(robot_data)
  torch.testing.assert_close(targets, torch.tensor([[1.0, 2.0, 3.0], [2.0, -1.0, 3.0]]))
  estimator = torch.nn.Linear(279, 3)
  predictions = estimator(torch.zeros(2, 279))
  error_sum = ((predictions - targets) ** 2).sum(dim=0)
  assert error_sum.shape == (3,)
  assert bool(torch.isfinite(error_sum).all())


def test_rollout_eval_rejects_combined_estimator_before_simulation(eval_module):
  with pytest.raises(ValueError, match="expected 'base_lin'"):
    eval_module.validate_estimator_meta(
      {"target_group": "base_lin_foot_height"}, "estimator_meta.json"
    )


def test_documented_eval_command_defaults_to_plane(eval_module):
  args = eval_module.parse_args(
    [
      "--checkpoint_dir",
      "policy_run",
      "--iteration",
      "30000",
      "--estimator_dir",
      "estimators/base_lin",
    ]
  )
  assert args.terrain == "plane"
