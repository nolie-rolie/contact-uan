"""Kinematic phase helpers for reference-conditioned jumping."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.tracking.mdp import MotionCommand


def resolve_motion_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...]
) -> tuple[int, ...]:
  """Resolve body names into the motion command's configured body order."""
  name_to_index = {name: index for index, name in enumerate(command.cfg.body_names)}
  missing = [name for name in body_names if name not in name_to_index]
  if missing:
    raise ValueError(
      f"Motion command is missing bodies {tuple(missing)}; "
      f"available bodies are {tuple(command.cfg.body_names)}."
    )
  return tuple(name_to_index[name] for name in body_names)


def kinematic_airborne_masks(
  env: ManagerBasedRlEnv,
  command: MotionCommand,
  foot_body_indexes: tuple[int, ...],
  height_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return robot and reference airborne masks from foot-link heights."""
  robot_foot_airborne, reference_foot_airborne = kinematic_foot_airborne_masks(
    env,
    command,
    foot_body_indexes,
    height_threshold,
  )
  return robot_foot_airborne.all(dim=1), reference_foot_airborne.all(dim=1)


def kinematic_foot_airborne_masks(
  env: ManagerBasedRlEnv,
  command: MotionCommand,
  foot_body_indexes: tuple[int, ...],
  height_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return per-foot robot and reference airborne masks."""
  origin_z = env.scene.env_origins[:, 2].unsqueeze(1)
  robot_foot_z = command.robot_body_pos_w[:, foot_body_indexes, 2] - origin_z
  reference_foot_z = command.body_pos_w[:, foot_body_indexes, 2] - origin_z
  return robot_foot_z > height_threshold, reference_foot_z > height_threshold
