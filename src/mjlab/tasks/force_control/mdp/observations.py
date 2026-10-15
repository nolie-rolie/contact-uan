"""Observation functions for EE force control."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.force_control.mdp.commands import ForceFieldCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def force_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Commanded 3D force in body frame (num_envs, 3)."""
  cmd = cast(ForceFieldCommand, env.command_manager.get_term(command_name))
  return cmd.target_forces_b


def forcefield_force_applied(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Applied impedance force in body frame (num_envs, 3)."""
  cmd = cast(ForceFieldCommand, env.command_manager.get_term(command_name))
  return cmd.forcefield_forces_b
