"""Load and validate the six separate Contact-UAN state estimators."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

if not __package__:
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.uan_pipeline.estimator_config import (  # noqa: E402
  COMPONENT_HIDDEN_DIMS,
  COMPONENT_SUBDIRS,
  COMPONENT_TARGETS,
  HISTORY_LENGTH,
  PROPRIO_ACTION_REPRESENTATION,
  PROPRIO_DIM,
  PROPRIO_TERMS,
)


def resolve_estimator_dirs(
  estimator_root: str | Path | None,
  overrides: dict[str, str | Path | None] | None = None,
) -> dict[str, Path]:
  """Resolve the paper bundle, allowing a retrained component to override its path."""
  overrides = overrides or {}
  unknown = set(overrides) - set(COMPONENT_TARGETS)
  if unknown:
    raise ValueError(f"Unknown estimator components: {sorted(unknown)}")
  root = Path(estimator_root).resolve() if estimator_root is not None else None
  paths = {}
  for group, subdir in COMPONENT_SUBDIRS.items():
    explicit = overrides.get(group)
    if explicit is not None:
      paths[group] = Path(explicit).resolve()
    elif root is not None:
      paths[group] = root / subdir
    else:
      raise ValueError(
        f"Missing {group} estimator. Provide --estimator-root or all six "
        "component estimator directories."
      )
  return paths


def _has_stance_filter(record: dict, group: str) -> bool:
  row_filter = record.get("filter")
  contact_name = f"{group.split('_', 1)[0]}_foot_contact"
  while isinstance(row_filter, dict):
    if (
      row_filter.get("type") == "stance_contact_rows"
      and row_filter.get("target_group") == group
      and row_filter.get("contact_name") == contact_name
      and row_filter.get("contact_threshold") == 0.5
    ):
      return True
    row_filter = row_filter.get("parent_filter")
  return False


def validate_component_meta(meta: dict, group: str, location: Path) -> None:
  """Check the inference contract and recorded training setup before loading a JIT."""
  expected = {
    "target_group": group,
    "target_layout": list(COMPONENT_TARGETS[group]),
    "output_dim": len(COMPONENT_TARGETS[group]),
    "input_dim": HISTORY_LENGTH * PROPRIO_DIM,
    "history_length": HISTORY_LENGTH,
    "proprio_dim": PROPRIO_DIM,
    "proprio_terms": list(PROPRIO_TERMS),
    "proprio_action_representation": PROPRIO_ACTION_REPRESENTATION,
    "hidden_dims": list(COMPONENT_HIDDEN_DIMS[group]),
    "activation": "ELU",
    "loss_type": "bce_logits" if group == "foot_contact" else "mse",
  }
  for key, value in expected.items():
    if meta.get(key) != value:
      raise ValueError(
        f"{location}: {key}={meta.get(key)!r}; expected {value!r} "
        f"for the {group} estimator."
      )
  if "target_dim" in meta and meta["target_dim"] != expected["output_dim"]:
    raise ValueError(f"{location}: target_dim does not match output_dim.")
  # The original collector/exporter used MuJoCo order without recording this key.
  # Retain those exports, while rejecting any explicitly different convention.
  if meta.get("joint_order", "mujoco") != "mujoco":
    raise ValueError(f"{location}: joint_order must be 'mujoco'.")
  scales = np.asarray(meta.get("proprio_scales"), dtype=np.float64)
  if scales.shape != (len(PROPRIO_TERMS),) or not np.all(
    np.isfinite(scales) & (scales > 0)
  ):
    raise ValueError(f"{location}: proprio_scales must contain five positive values.")
  if "stance_foot" in group:
    datasets = meta.get("training", {}).get("datasets", [])
    if not datasets or not all(_has_stance_filter(ds, group) for ds in datasets):
      raise ValueError(
        f"{location}: {group} requires stance_contact_rows training metadata "
        "for its foot (contact threshold 0.5) in every training dataset."
      )


def _validate_model(estimator, group: str, location: Path) -> None:
  input_dim = HISTORY_LENGTH * PROPRIO_DIM
  output_dim = len(COMPONENT_TARGETS[group])
  widths = (input_dim, *COMPONENT_HIDDEN_DIMS[group], output_dim)
  expected_weights = [(widths[i + 1], widths[i]) for i in range(len(widths) - 1)]
  actual_weights = [
    tuple(value.shape) for value in estimator.parameters() if value.ndim == 2
  ]
  if actual_weights != expected_weights:
    raise ValueError(
      f"{location}: model layer shapes {actual_weights} do not match "
      f"the {group} architecture {expected_weights}."
    )
  samples = torch.stack((torch.zeros(input_dim), torch.linspace(-1.0, 1.0, input_dim)))
  try:
    with torch.inference_mode():
      output = estimator(samples)
  except Exception as exc:
    raise ValueError(
      f"{location}: estimator failed a {input_dim}-input check."
    ) from exc
  if not isinstance(output, torch.Tensor) or output.shape != (2, output_dim):
    shape = getattr(output, "shape", type(output).__name__)
    raise ValueError(f"{location}: model returned {shape}; expected (2, {output_dim}).")
  if not bool(torch.isfinite(output).all()):
    raise ValueError(f"{location}: estimator returned non-finite predictions.")


def load_paper_estimator_bundle(
  estimator_root: str | Path | None = None,
  overrides: dict[str, str | Path | None] | None = None,
) -> dict[str, dict]:
  """Validate all six models on CPU, without creating environments or outputs."""
  paths = resolve_estimator_dirs(estimator_root, overrides)
  metadata = {}
  for group, directory in paths.items():
    meta_path = directory / "estimator_meta.json"
    model_path = directory / "state_estimator.jit"
    for required in (meta_path, model_path):
      if not required.is_file():
        raise FileNotFoundError(f"Missing {group} estimator file: {required}")
    with meta_path.open(encoding="utf-8") as stream:
      meta = json.load(stream)
    validate_component_meta(meta, group, meta_path)
    metadata[group] = meta
  reference_scales = metadata["base_lin"]["proprio_scales"]
  for group, meta in metadata.items():
    if meta["proprio_scales"] != reference_scales:
      raise ValueError(f"{paths[group]}: proprio_scales differ across estimators.")
  bundles = {}
  for group, directory in paths.items():
    model_path = directory / "state_estimator.jit"
    estimator = torch.jit.load(str(model_path), map_location="cpu").eval()
    _validate_model(estimator, group, model_path)
    bundles[group] = {
      "dir": str(directory),
      "estimator": estimator,
      "meta": metadata[group],
    }
  return bundles
