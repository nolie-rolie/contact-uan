"""Public hardware preparation and six-estimator relabeling interfaces."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def pipeline_modules(monkeypatch):
  pipeline_dir = Path(__file__).resolve().parents[1] / "scripts" / "uan_pipeline"
  monkeypatch.syspath_prepend(str(pipeline_dir))
  return (
    importlib.import_module("hardware_pipeline"),
    importlib.import_module("postprocess_real_data"),
  )


@pytest.mark.parametrize("flag", ["--estimator-dir", "--separate-estimators-dir"])
def test_preparation_rejects_legacy_estimator_workflows(pipeline_modules, flag):
  pipeline, _ = pipeline_modules
  with pytest.raises(SystemExit) as error:
    pipeline.build_parser().parse_args(
      ["postprocess", "--dataset-root", "dataset", flag, "estimators"]
    )
  assert error.value.code == 2


@pytest.mark.parametrize("flag", ["--estimator_dir", "--separate-estimators-dir"])
def test_standalone_preparation_rejects_legacy_estimators(pipeline_modules, flag):
  _, postprocess = pipeline_modules
  with pytest.raises(SystemExit) as error:
    postprocess.parse_args(
      ["--data_dir", "input", "--output_dir", "output", flag, "estimators"]
    )
  assert error.value.code == 2


def test_relabel_wrapper_forwards_bundle_and_component_override(
  pipeline_modules,
  monkeypatch,
):
  pipeline, _ = pipeline_modules
  args = pipeline.build_parser().parse_args(
    [
      "contact-root-relabel",
      "--source-root",
      "input",
      "--output-root",
      "output",
      "--estimator-root",
      "estimators",
      "--left-stance-foot-height-estimator-dir",
      "retrained_left_height",
      "--geometry-contact-threshold",
      "0.9",
    ]
  )
  calls = []
  monkeypatch.setattr(pipeline.subprocess, "check_call", calls.append)
  pipeline.cmd_contact_root_relabel(args)
  assert len(calls) == 1
  options = dict(zip(calls[0][2::2], calls[0][3::2], strict=True))
  assert options["--estimator-root"] == "estimators"
  assert options["--left-stance-foot-height-estimator-dir"] == "retrained_left_height"
  assert options["--geometry-contact-threshold"] == "0.9"
  assert "--foot-contact-estimator-dir" not in options
  assert "--root-height-mode" not in options


@pytest.mark.parametrize(
  "flag,value",
  [
    ("--base-lin-foot-height-estimator-dir", "combined"),
    ("--root-height-mode", "contact_geometry"),
    ("--base-ang-vel-estimator-dir", "angular_velocity"),
    ("--projected-gravity-estimator-dir", "gravity"),
    ("--joint-vel-estimator-dir", "joint_velocity"),
  ],
)
def test_relabel_wrapper_rejects_unsupported_setups(pipeline_modules, flag, value):
  pipeline, _ = pipeline_modules
  with pytest.raises(SystemExit) as error:
    pipeline.build_parser().parse_args(
      [
        "contact-root-relabel",
        "--source-root",
        "input",
        "--output-root",
        "output",
        "--estimator-root",
        "estimators",
        flag,
        value,
      ]
    )
  assert error.value.code == 2


def test_preparation_preserves_recorded_values_and_rollout_boundaries(
  pipeline_modules,
  tmp_path,
):
  pipeline, _ = pipeline_modules
  pipeline.ensure_dataset_dirs(tmp_path)
  manifest = pipeline.init_manifest(
    tmp_path,
    task="wtw",
    policy="test",
    collection_datetime=None,
    dataset_type="collection",
  )
  pipeline.save_manifest(tmp_path, manifest)
  recorded = {}
  for run, offset in (("run_a", 0), ("run_b", 1000)):
    values = np.arange(2 * 143, dtype=np.float32).reshape(2, 143) + offset
    recorded[run] = values
    pipeline.write_processed_files(values, str(tmp_path / "trimmed" / run), 1)
  args = pipeline.build_parser().parse_args(
    ["postprocess", "--dataset-root", str(tmp_path)]
  )
  pipeline.cmd_postprocess(args)

  for run, values in recorded.items():
    output, _ = pipeline.load_rollout_array(tmp_path / "processed" / run)
    np.testing.assert_array_equal(output, values)
  processed_meta = pipeline.load_json(tmp_path / "processed" / "meta.json")
  assert processed_meta["n_rows"] == 4
  assert processed_meta["estimator_dir"] is None
  assert processed_meta["estimator_target_layout"] == []
  assert processed_meta["lin_vel_estimated"] is False
  assert processed_meta["preserve_rollout_boundaries"] is True
  assert len(processed_meta["source_segments"]) == 2
  manifest = pipeline.load_json(pipeline.manifest_path(tmp_path))
  assert manifest["estimator"] is None
  assert manifest["status"]["postprocessed"] is True
