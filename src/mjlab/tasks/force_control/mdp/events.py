"""Step event for applying force field forces to the robot."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.tasks.force_control.mdp.commands import ForceFieldCommand


class apply_forcefield_force:
  """Apply impedance forces from the ForceFieldCommand to a robot body each step.

  Uses ``write_external_wrench_to_sim`` which writes world-frame forces to
  MuJoCo's ``xfrc_applied``. Forces persist across substeps until overwritten.

  Use with ``mode="step"``.
  """

  def __init__(self, cfg: object, env: ManagerBasedRlEnv) -> None:
    cfg_t = cast("EventTermCfg", cfg)
    self._asset: Entity = env.scene[cfg_t.params["asset_cfg"].name]
    self._body_ids: list[int] = self._asset.find_bodies(cfg_t.params["body_name"])[0]
    self._command_name: str = cfg_t.params["command_name"]
    self._num_envs = env.num_envs
    self._device = env.device

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    body_name: str,
    command_name: str,
  ) -> None:
    del env_ids, asset_cfg, body_name, command_name  # Resolved in __init__.

    cmd = cast(
      "ForceFieldCommand",
      env.command_manager.get_term(self._command_name),
    )
    # (num_envs, 3) → (num_envs, 1, 3) for single body.
    forces = cmd.forcefield_forces_w.unsqueeze(1)
    torques = torch.zeros_like(forces)
    self._asset.write_external_wrench_to_sim(forces, torques, body_ids=self._body_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Zero forces for reset environments."""
    if env_ids is None:
      env_ids = slice(None)
    if isinstance(env_ids, slice):
      n = self._num_envs
    else:
      n = len(env_ids)
    if n == 0:
      return
    zeros = torch.zeros(n, 1, 3, device=self._device)
    self._asset.write_external_wrench_to_sim(
      zeros, zeros, env_ids=env_ids, body_ids=self._body_ids
    )
