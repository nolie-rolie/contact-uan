"""Helpers for clipping deploy force commands to reachable XY displacements."""

from __future__ import annotations

import numpy as np


def clip_xy_force_to_reach(
  force_xyz: np.ndarray,
  stiffness: float | None,
  max_xy_reach_m: float | None,
) -> np.ndarray:
  """Clamp x/y force so the implied per-axis displacement stays reachable."""
  clipped = np.asarray(force_xyz, dtype=float).copy()
  if stiffness is None or max_xy_reach_m is None:
    return clipped
  if stiffness < 0.0:
    raise ValueError("stiffness must be >= 0 or None.")
  if max_xy_reach_m < 0.0:
    raise ValueError("max_xy_reach_m must be >= 0 or None.")

  max_xy_force = float(stiffness) * float(max_xy_reach_m)
  clipped[:2] = np.clip(clipped[:2], -max_xy_force, max_xy_force)
  return clipped
