from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def bundle_module():
  script_dir = Path(__file__).resolve().parents[1] / "scripts/uan_pipeline"
  sys.path.insert(0, str(script_dir))
  try:
    yield importlib.import_module("estimator_bundle")
  finally:
    sys.path.remove(str(script_dir))


@pytest.fixture(scope="module")
def original_bundle(tmp_path_factory, bundle_module):
  root = tmp_path_factory.mktemp("paper_estimators")
  input_dim = bundle_module.HISTORY_LENGTH * bundle_module.PROPRIO_DIM
  for group, targets in bundle_module.COMPONENT_TARGETS.items():
    directory = root / bundle_module.COMPONENT_SUBDIRS[group]
    directory.mkdir(parents=True)
    hidden = bundle_module.COMPONENT_HIDDEN_DIMS[group]
    widths = (input_dim, *hidden, len(targets))
    layers = []
    for i in range(len(widths) - 1):
      layers.append(torch.nn.Linear(widths[i], widths[i + 1]))
      if i < len(widths) - 2:
        layers.append(torch.nn.ELU())
    model = cast(
      torch.jit.ScriptModule,
      torch.jit.trace(torch.nn.Sequential(*layers), torch.zeros(1, input_dim)),
    )
    model.save(str(directory / "state_estimator.jit"))
    row_filter = None
    if "stance_foot" in group:
      row_filter = {
        "type": "stance_contact_rows",
        "target_group": group,
        "contact_name": f"{group.split('_', 1)[0]}_foot_contact",
        "contact_threshold": 0.5,
      }
    meta = {
      "target_group": group,
      "target_layout": list(targets),
      "output_dim": len(targets),
      "input_dim": input_dim,
      "history_length": bundle_module.HISTORY_LENGTH,
      "proprio_dim": bundle_module.PROPRIO_DIM,
      "proprio_terms": list(bundle_module.PROPRIO_TERMS),
      "proprio_action_representation": bundle_module.PROPRIO_ACTION_REPRESENTATION,
      "proprio_scales": [0.25, 1.0, 1.0, 0.05, 1.0],
      "hidden_dims": list(hidden),
      "activation": "ELU",
      "loss_type": "bce_logits" if group == "foot_contact" else "mse",
      "training": {"datasets": [{"filter": row_filter}]},
    }
    (directory / "estimator_meta.json").write_text(json.dumps(meta))
  return root


@pytest.fixture
def bundle_root(tmp_path, original_bundle):
  root = tmp_path / "estimators"
  shutil.copytree(original_bundle, root)
  return root


def test_loads_six_existing_paper_exports_without_new_manifest(
  bundle_module, original_bundle
):
  loaded = bundle_module.load_paper_estimator_bundle(original_bundle)
  assert tuple(loaded) == tuple(bundle_module.COMPONENT_TARGETS)
  assert loaded["base_lin"]["estimator"](torch.zeros(4, 279)).shape == (4, 3)


def test_component_override_and_explicit_six_paths(bundle_module, original_bundle):
  paths = bundle_module.resolve_estimator_dirs(original_bundle)
  explicit = bundle_module.load_paper_estimator_bundle(overrides=paths)
  assert len(explicit) == 6
  override = original_bundle / "retrained_left_height"
  paths = bundle_module.resolve_estimator_dirs(
    original_bundle, {"left_stance_foot_height": override}
  )
  assert paths["left_stance_foot_height"] == override
  with pytest.raises(ValueError, match="all six"):
    bundle_module.resolve_estimator_dirs(None, {"base_lin": paths["base_lin"]})


def test_missing_orientation_estimator_is_rejected(bundle_module, bundle_root):
  group = "right_stance_foot_projected_gravity"
  path = bundle_root / bundle_module.COMPONENT_SUBDIRS[group] / "state_estimator.jit"
  path.unlink()
  with pytest.raises(FileNotFoundError, match=group):
    bundle_module.load_paper_estimator_bundle(bundle_root)


@pytest.mark.parametrize(
  ("key", "value"),
  [
    ("target_group", "all"),
    ("target_layout", ["lin_vel_y", "lin_vel_x", "lin_vel_z"]),
    ("history_length", 10),
    ("output_dim", 4),
    ("input_dim", 930),
    ("proprio_dim", 90),
    ("proprio_action_representation", "action"),
    ("proprio_terms", ["ang_vel_b", "proj_grav", "joint_pos", "joint_vel", "action"]),
    ("joint_order", "isaac"),
    ("hidden_dims", [128, 128]),
    ("proprio_scales", [1, 1, 1, 1, 1]),
  ],
)
def test_incompatible_component_metadata_is_rejected(
  bundle_module, bundle_root, key, value
):
  path = bundle_root / "base_lin/estimator_meta.json"
  meta = json.loads(path.read_text())
  meta[key] = value
  path.write_text(json.dumps(meta))
  with pytest.raises(ValueError, match=key):
    bundle_module.load_paper_estimator_bundle(bundle_root)


def test_foot_estimator_requires_matching_stance_training(bundle_module, bundle_root):
  group = "left_stance_foot_height"
  path = bundle_root / bundle_module.COMPONENT_SUBDIRS[group] / "estimator_meta.json"
  meta = json.loads(path.read_text())
  meta["training"]["datasets"][0]["filter"]["contact_name"] = "right_foot_contact"
  path.write_text(json.dumps(meta))
  with pytest.raises(ValueError, match="stance_contact_rows"):
    bundle_module.load_paper_estimator_bundle(bundle_root)


def test_model_architecture_is_checked_against_metadata(bundle_module, bundle_root):
  model = cast(
    torch.jit.ScriptModule,
    torch.jit.trace(torch.nn.Linear(279, 3), torch.zeros(1, 279)),
  )
  model.save(str(bundle_root / "base_lin/state_estimator.jit"))
  with pytest.raises(ValueError, match="model layer shapes"):
    bundle_module.load_paper_estimator_bundle(bundle_root)


def test_nonfinite_model_predictions_are_rejected(bundle_module, bundle_root):
  path = bundle_root / "base_lin/state_estimator.jit"
  model = torch.jit.load(str(path))
  with torch.no_grad():
    next(model.parameters()).fill_(float("nan"))
  model.save(str(path))
  with pytest.raises(ValueError, match="non-finite predictions"):
    bundle_module.load_paper_estimator_bundle(bundle_root)


def test_model_forward_shape_is_checked(bundle_module, bundle_root):
  path = bundle_root / "base_lin/state_estimator.jit"
  model = torch.jit.load(str(path))
  flattened = cast(
    torch.jit.ScriptModule,
    torch.jit.trace(
      torch.nn.Sequential(model, torch.nn.Flatten(0)), torch.zeros(2, 279)
    ),
  )
  flattened.save(str(path))
  with pytest.raises(ValueError, match="model returned"):
    bundle_module.load_paper_estimator_bundle(bundle_root)


def test_relabel_validates_bundle_before_overwrite_or_environment(
  bundle_module, bundle_root, tmp_path, monkeypatch
):
  relabel = importlib.import_module("relabel_hardware_stance_geometry_mjlab")
  output = tmp_path / "output"
  output.mkdir()
  sentinel = output / "keep.txt"
  sentinel.write_text("existing dataset")
  (bundle_root / "base_lin/state_estimator.jit").unlink()
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "relabel",
      "--source-root",
      str(tmp_path / "source"),
      "--output-root",
      str(output),
      "--estimator-root",
      str(bundle_root),
      "--overwrite",
    ],
  )

  def unexpected_environment(*args):
    pytest.fail("Environment created before bundle validation.")

  monkeypatch.setattr(relabel, "_build_solver_env", unexpected_environment)
  with pytest.raises(FileNotFoundError, match="base_lin"):
    relabel.main()
  assert sentinel.read_text() == "existing dataset"


def test_relabel_uses_all_six_components_with_contact_gated_geometry(
  bundle_module, original_bundle, tmp_path, monkeypatch
):
  relabel = importlib.import_module("relabel_hardware_stance_geometry_mjlab")
  source = tmp_path / "source"
  rollout = source / "trimmed/run/segment"
  rollout.mkdir(parents=True)
  output = tmp_path / "relabeled"
  raw = np.zeros((4, 143), dtype=np.float32)
  raw[:, 125:127] = [0.05, 0.06]
  predictions = {
    "base_lin": [0.1, 0.2, 0.3],
    "foot_contact": [0.9, 0.2],
    "left_stance_foot_height": [-0.01],
    "right_stance_foot_height": [-0.02],
    "left_stance_foot_projected_gravity": [0.1, 0, -0.99],
    "right_stance_foot_projected_gravity": [-0.1, 0, -0.99],
  }
  used_groups = []

  def predict(raw, estimator, meta, action_col_start):
    group = meta["target_group"]
    used_groups.append(group)
    return np.tile(predictions[group], (len(raw), 1)), np.array(
      [False, False, True, True]
    )

  def solve(**kwargs):
    np.testing.assert_allclose(kwargs["predicted_foot_heights_np"], [[-0.01, 0.06]] * 2)
    n = len(kwargs["state_rows_np"])
    zeros = np.zeros(n, dtype=np.float32)
    root_z = np.full(n, 0.77, dtype=np.float32)
    return root_z, {
      "prior_root_z": root_z,
      "solved_root_z": root_z,
      "root_z_delta": zeros,
      "solver_score": zeros,
      "used_stance_geometry": np.ones(n, dtype=bool),
      "active_contact_count": np.ones(n),
      "double_stance_disagreement": zeros,
    }

  def orientation(env, **kwargs):
    n = len(kwargs["state_rows_np"])
    return np.tile([0.1, 0, -0.99], (n, 1)), {
      "used_stance_orientation": np.ones(n, dtype=bool),
      "active_orientation_contact_count": np.ones(n),
      "double_stance_orientation_disagreement_rad": np.zeros(n),
    }

  monkeypatch.setattr(relabel, "find_rollout_dirs", lambda _: [rollout])
  monkeypatch.setattr(relabel, "load_rollout_array", lambda _: (raw.copy(), []))
  monkeypatch.setattr(relabel, "predict_estimator_targets", predict)
  monkeypatch.setattr(
    relabel, "_build_solver_env", lambda *_: (None, torch.arange(29), [0, 1])
  )
  monkeypatch.setattr(relabel, "_solve_contact_geometry_batch", solve)
  monkeypatch.setattr(relabel, "contact_geometry_root_projected_gravity", orientation)
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "relabel",
      "--source-root",
      str(source),
      "--output-root",
      str(output),
      "--estimator-root",
      str(original_bundle),
    ],
  )
  relabel.main()
  result = np.load(output / "processed/all_data.npy")
  assert set(used_groups) == set(bundle_module.COMPONENT_TARGETS)
  np.testing.assert_allclose(result[:2, :-1], raw[:2, :-1])
  np.testing.assert_allclose(result[2:, :3], [[0.1, 0.2, 0.3]] * 2)
  np.testing.assert_allclose(result[2:, 6:9], [[0.1, 0, -0.99]] * 2)
  np.testing.assert_allclose(result[2:, 125:127], [[-0.01, 0.06]] * 2)
  np.testing.assert_allclose(result[2:, 129], 0.77)
  assert result[-1, -1] == 1.0
