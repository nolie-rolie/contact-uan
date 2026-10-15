from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from scripts.uan_pipeline import stance_orientation


def _load_relabel_module():
  script_dir = Path(__file__).resolve().parents[1] / "scripts/uan_pipeline"
  spec = importlib.util.spec_from_file_location(
    "relabel_hardware_stance_geometry_mjlab",
    script_dir / "relabel_hardware_stance_geometry_mjlab.py",
  )
  assert spec is not None
  assert spec.loader is not None
  sys.path.insert(0, str(script_dir))
  try:
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
  finally:
    sys.path.remove(str(script_dir))
  return module


def test_mark_rollout_end_preserves_existing_markers_and_marks_last_row() -> None:
  module = _load_relabel_module()
  data = np.zeros((4, 143), dtype=np.float32)
  data[1, -1] = 1.0

  module.mark_rollout_end(data)

  np.testing.assert_array_equal(data[:, -1], [0.0, 1.0, 0.0, 1.0])


@pytest.mark.parametrize("shape", [(0, 143), (4, 0), (143,)])
def test_mark_rollout_end_rejects_empty_or_non_matrix_data(
  shape: tuple[int, ...],
) -> None:
  module = _load_relabel_module()

  with pytest.raises(ValueError, match="non-empty 2D rollout array"):
    module.mark_rollout_end(np.zeros(shape, dtype=np.float32))


def test_stance_orientation_reconstruction_blends_contacts_and_preserves_fallback(
  monkeypatch,
) -> None:
  state = np.zeros((3, 96), dtype=np.float32)
  state[:, 6:9] = [0.0, -0.6, -0.8]
  extension = np.zeros((3, 15), dtype=np.float32)
  foot_rotations = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [2.0**-0.5, 2.0**-0.5, 0.0, 0.0]]
  ).repeat(3, 1, 1)
  scene = MagicMock()
  scene.__getitem__.return_value = SimpleNamespace(
    data=SimpleNamespace(body_link_quat_w=foot_rotations)
  )
  env = SimpleNamespace(device="cpu", scene=scene, sim=MagicMock())
  writes = []
  monkeypatch.setattr(
    stance_orientation,
    "write_recorded_state",
    lambda *args, **kwargs: writes.append(kwargs),
  )

  gravity, diagnostics = stance_orientation.contact_geometry_root_projected_gravity(
    cast(Any, env),
    state_rows_np=state,
    ext_rows_np=extension,
    foot_projected_gravity_np=np.tile([0.0, 0.0, -1.0], (3, 2, 1)),
    contact_prob_np=np.array([[0.9, 0.1], [0.9, 0.9], [0.1, 0.2]]),
    joint_ids=torch.arange(29),
    foot_body_ids=[0, 1],
    contact_threshold=0.8,
    default_base_height=0.75,
  )

  np.testing.assert_allclose(
    gravity,
    [[0.0, 0.0, -1.0], [0.0, 2.0**-0.5, -(2.0**-0.5)], [0.0, -0.6, -0.8]],
    atol=1e-6,
  )
  np.testing.assert_array_equal(
    diagnostics["used_stance_orientation"], [True, True, False]
  )
  torch.testing.assert_close(
    writes[0]["state_raw"][:, 6:9], torch.tensor([[0.0, 0.0, -1.0]]).repeat(3, 1)
  )
  np.testing.assert_allclose(state[:, 6:9], [[0.0, -0.6, -0.8]] * 3)
