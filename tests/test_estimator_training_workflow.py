from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.uan_pipeline import train_estimators as trainer
from scripts.uan_pipeline.estimator_bundle import load_paper_estimator_bundle
from scripts.uan_pipeline.estimator_config import (
  COMPONENT_HIDDEN_DIMS,
  COMPONENT_SUBDIRS,
  COMPONENT_TARGETS,
  HISTORY_LENGTH,
  PROPRIO_ACTION_REPRESENTATION,
  PROPRIO_DIM,
  PROPRIO_TERMS,
)


@pytest.fixture
def training_data(tmp_path: Path) -> Path:
  rows = 64
  rng = np.random.default_rng(12)
  layout = list(
    dict.fromkeys(name for names in COMPONENT_TARGETS.values() for name in names)
  )
  targets = rng.normal(scale=0.02, size=(rows, len(layout))).astype(np.float32)
  targets[:, layout.index("left_foot_contact")] = (np.arange(rows) // 4) % 3 != 0
  targets[:, layout.index("right_foot_contact")] = (np.arange(rows) // 4) % 3 != 1
  path = tmp_path / "estimator_training_data.pt"
  torch.save(
    {
      "proprio_history": torch.from_numpy(
        rng.normal(size=(rows, HISTORY_LENGTH, PROPRIO_DIM)).astype(np.float32)
      ),
      "targets": torch.from_numpy(targets),
      "history_length": HISTORY_LENGTH,
      "proprio_dim": PROPRIO_DIM,
      "proprio_terms": list(PROPRIO_TERMS),
      "proprio_action_representation": PROPRIO_ACTION_REPRESENTATION,
      "target_layout": layout,
      "env_ids": np.arange(rows) % 4,
      "episode_ids": np.zeros(rows, dtype=np.int32),
      "terminated": np.zeros(rows, dtype=bool),
      "num_envs": 4,
    },
    path,
  )
  return path


def _args(data: Path, output: Path, *extra: str) -> list[str]:
  return [
    "--data",
    str(data),
    "--output_dir",
    str(output),
    "--device",
    "cpu",
    "--epochs",
    "1",
    "--batch_size",
    "128",
    "--wandb_project",
    "",
    *extra,
  ]


def test_cli_defaults_to_six_paper_models_and_rejects_unsupported_setups(
  training_data: Path,
  tmp_path: Path,
) -> None:
  args = _args(training_data, tmp_path / "models")
  assert trainer.parse_args(args).target_group == "paper"
  assert trainer.parse_args(args).drop_failed_episodes
  for removed in ("all", "separate", "base_vel", "joint_vel", "height"):
    with pytest.raises(SystemExit):
      trainer.parse_args([*args, "--target_group", removed])
  with pytest.raises(SystemExit):
    trainer.parse_args([*args, "--hidden_dims", "64,32"])


@pytest.mark.parametrize("invalid", ["history", "missing_orientation", "no_stance"])
def test_paper_preflight_checks_every_component_before_training(
  training_data: Path,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  invalid: str,
) -> None:
  data = torch.load(training_data, weights_only=False)
  if invalid == "history":
    data["history_length"] = 10
  elif invalid == "missing_orientation":
    data["target_layout"][-1] = "unused_target"
  else:
    column = data["target_layout"].index("right_foot_contact")
    data["targets"][:, column] = 0
  torch.save(data, training_data)
  calls = []
  monkeypatch.setattr(trainer, "_train_estimator", lambda *args: calls.append(args))
  with pytest.raises(ValueError):
    trainer.main(_args(training_data, tmp_path / "models"))
  assert not calls
  assert not (tmp_path / "models").exists()


def test_failed_training_preserves_existing_bundle(
  training_data: Path,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  output = tmp_path / "models"
  output.mkdir()
  manifest = output / trainer.BUNDLE_MANIFEST
  manifest.write_text("old bundle")
  old_model = output / "base_lin/state_estimator.jit"
  old_model.parent.mkdir()
  old_model.write_bytes(b"old model")

  def fail_second_component(args, datasets):
    if args.target_group == "foot_contact":
      raise RuntimeError("training failed")
    destination = Path(args.output_dir)
    destination.mkdir(parents=True)
    (destination / "state_estimator.jit").write_bytes(b"new model")

  monkeypatch.setattr(trainer, "_train_estimator", fail_second_component)
  with pytest.raises(RuntimeError, match="training failed"):
    trainer.main(_args(training_data, output))
  assert manifest.read_text() == "old bundle"
  assert old_model.read_bytes() == b"old model"


def test_cpu_training_exports_six_models_and_allows_one_component_retraining(
  training_data: Path,
  tmp_path: Path,
) -> None:
  output = tmp_path / "models"
  old_threads = torch.get_num_threads()
  torch.set_num_threads(1)
  try:
    trainer.main(_args(training_data, output))
    bundle = load_paper_estimator_bundle(output)
    assert set(bundle) == set(COMPONENT_TARGETS)
    manifest = json.loads((output / trainer.BUNDLE_MANIFEST).read_text())
    assert manifest["mode"] == "paper"
    assert set(manifest["outputs"]) == set(COMPONENT_TARGETS)
    original_models = {}
    for component, targets in COMPONENT_TARGETS.items():
      component_dir = output / COMPONENT_SUBDIRS[component]
      meta = json.loads((component_dir / "estimator_meta.json").read_text())
      assert meta["target_layout"] == list(targets)
      assert meta["hidden_dims"] == list(COMPONENT_HIDDEN_DIMS[component])
      assert meta["history_length"] == 3
      assert meta["input_dim"] == 279
      if "stance" in component:
        row_filter = meta["training"]["datasets"][0]["filter"]
        assert row_filter["type"] == "stance_contact_rows"
        assert row_filter["target_group"] == component
        assert row_filter["samples_after"] < row_filter["samples_before"]
      model_path = component_dir / "state_estimator.jit"
      model = torch.jit.load(str(model_path))
      assert model(torch.zeros(2, 279)).shape == (2, len(targets))
      original_models[component] = model_path.read_bytes()
    component = "right_stance_foot_height"
    trainer.main(
      _args(
        training_data,
        output / COMPONENT_SUBDIRS[component],
        "--target_group",
        component,
      )
    )
    load_paper_estimator_bundle(output)
    for other, original_bytes in original_models.items():
      if other != component:
        path = output / COMPONENT_SUBDIRS[other] / "state_estimator.jit"
        assert path.read_bytes() == original_bytes
  finally:
    torch.set_num_threads(old_threads)
