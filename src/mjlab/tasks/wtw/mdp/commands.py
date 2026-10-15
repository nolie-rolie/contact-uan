"""Custom command terms for WTW environment.

GaitFrequencyCommand: samples 1D gait frequency command in [-1, 1].
FootSwingHeightCommand: samples 1D foot swing height command in [-1, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class GaitFrequencyCommand(CommandTerm):
  """Samples gait frequency command uniformly in a configurable range."""

  cfg: GaitFrequencyCommandCfg

  def __init__(self, cfg: GaitFrequencyCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.frequency_cmd = torch.zeros(self.num_envs, 1, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.frequency_cmd

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.frequency_cmd[env_ids, 0] = r.uniform_(*self.cfg.frequency_range)

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    pass


@dataclass(kw_only=True)
class GaitFrequencyCommandCfg(CommandTermCfg):
  """Configuration for gait frequency command."""

  frequency_range: tuple[float, float] = (-1.0, 1.0)
  """Range for gait frequency command (uniform sampling)."""

  def build(self, env: ManagerBasedRlEnv) -> GaitFrequencyCommand:
    return GaitFrequencyCommand(self, env)


class FootSwingHeightCommand(CommandTerm):
  """Samples foot swing height command uniformly in a configurable range."""

  cfg: FootSwingHeightCommandCfg

  def __init__(self, cfg: FootSwingHeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.swing_height_cmd = torch.zeros(self.num_envs, 1, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.swing_height_cmd

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.swing_height_cmd[env_ids, 0] = r.uniform_(*self.cfg.swing_height_range)

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    pass


@dataclass(kw_only=True)
class FootSwingHeightCommandCfg(CommandTermCfg):
  """Configuration for foot swing height command."""

  swing_height_range: tuple[float, float] = (-1.0, 1.0)
  """Range for swing height command (uniform sampling)."""

  def build(self, env: ManagerBasedRlEnv) -> FootSwingHeightCommand:
    return FootSwingHeightCommand(self, env)
