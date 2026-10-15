from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_eval_module():
  script_path = (
    Path(__file__).resolve().parents[1]
    / "scripts/uan_pipeline/eval_stance_pose_root_height_mjlab.py"
  )
  spec = importlib.util.spec_from_file_location(
    "eval_stance_pose_root_height_mjlab",
    script_path,
  )
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def test_root_height_methods_select_fallback_max_contact_and_blend() -> None:
  eval_module = _load_eval_module()
  fallback = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
  left_root = np.array([0.8, 0.7, 0.9, 0.2], dtype=np.float32)
  right_root = np.array([1.2, 1.3, 1.1, 2.0], dtype=np.float32)
  contact = np.array(
    [
      [0.1, 0.2],
      [0.9, 0.2],
      [0.1, 0.95],
      [0.8, 0.6],
    ],
    dtype=np.float32,
  )

  methods = eval_module.root_height_methods_from_candidates(
    fallback,
    left_root,
    right_root,
    contact,
    contact_threshold=0.5,
  )

  np.testing.assert_allclose(methods["lowest_foot"], fallback)
  np.testing.assert_allclose(
    methods["max_contact"],
    np.array([1.0, 0.7, 1.1, 0.2], dtype=np.float32),
  )
  np.testing.assert_allclose(
    methods["blend"],
    np.array(
      [
        1.0,
        0.7,
        1.1,
        (0.2 * 0.8 + 2.0 * 0.6) / (0.8 + 0.6),
      ],
      dtype=np.float32,
    ),
  )
