"""Action terms for UAN replay rollout tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from .replay_command import ReplayWindowCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ReplayResidualJointPositionActionCfg(ActionTermCfg):
  """Residual joint-position action applied on top of replay targets."""

  actuator_names: tuple[str, ...] | list[str]
  command_name: str = "replay"
  residual_scale: float = 1.0
  residual_clip: float | None = None

  def build(self, env: ManagerBasedRlEnv) -> ReplayResidualJointPositionAction:
    return ReplayResidualJointPositionAction(self, env)


class ReplayResidualJointPositionAction(ActionTerm):
  cfg: ReplayResidualJointPositionActionCfg

  def __init__(self, cfg: ReplayResidualJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    target_ids, target_names = self._entity.find_joints_by_actuator_names(
      cfg.actuator_names
    )
    self._target_ids = torch.tensor(target_ids, dtype=torch.long, device=self.device)
    self._target_names = target_names
    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._processed_residual = torch.zeros_like(self._raw_actions)
    self._processed_residual_delta = torch.zeros_like(self._raw_actions)
    self._command_target = torch.zeros_like(self._raw_actions)

  @property
  def action_dim(self) -> int:
    return int(self._target_ids.numel())

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def target_ids(self) -> torch.Tensor:
    return self._target_ids

  @property
  def target_names(self) -> list[str]:
    return self._target_names

  @property
  def processed_residual(self) -> torch.Tensor:
    return self._processed_residual

  @property
  def processed_residual_delta(self) -> torch.Tensor:
    return self._processed_residual_delta

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    residual = actions * float(self.cfg.residual_scale)
    if self.cfg.residual_clip is not None:
      clip = float(self.cfg.residual_clip)
      residual = residual.clamp(-clip, clip)
    previous_residual = self._processed_residual.clone()
    command = self._replay_command()
    self._processed_residual_delta[:] = residual - previous_residual
    self._processed_residual[:] = residual
    self._command_target[:] = command.current_target + self._processed_residual

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      self._command_target - encoder_bias,
      joint_ids=self._target_ids,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._processed_residual[env_ids] = 0.0
    self._processed_residual_delta[env_ids] = 0.0
    self._command_target[env_ids] = 0.0

  def _replay_command(self) -> ReplayWindowCommand:
    command = self._env.command_manager.get_term(self.cfg.command_name)
    if command is None:
      raise RuntimeError(f"Replay command {self.cfg.command_name!r} not found.")
    return cast(ReplayWindowCommand, command)
