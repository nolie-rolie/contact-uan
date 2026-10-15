"""Helpers for clipping force commands to reachable XY displacements."""

from __future__ import annotations

import torch


def clip_xy_force_to_reach(
  force_command_b: torch.Tensor,
  stiffness: torch.Tensor,
  max_xy_reach_m: float | None,
) -> torch.Tensor:
  """Clamp x/y force so the implied per-axis displacement stays reachable."""
  if max_xy_reach_m is None:
    return force_command_b
  if max_xy_reach_m < 0.0:
    raise ValueError("max_xy_reach_m must be >= 0 or None.")

  max_xy_force = (
    stiffness.to(
      device=force_command_b.device,
      dtype=force_command_b.dtype,
    ).reshape(force_command_b.shape[:-1] + (1,))
    * max_xy_reach_m
  )
  clipped_force = force_command_b.clone()
  clipped_force[..., :2] = torch.clamp(
    clipped_force[..., :2],
    min=-max_xy_force,
    max=max_xy_force,
  )
  return clipped_force
