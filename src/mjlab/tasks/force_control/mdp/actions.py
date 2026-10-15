"""Force field action term: apply impedance forces at every substep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ForceFieldActionCfg(ActionTermCfg):
  """Configuration for the force field action term."""

  entity_name: str = "robot"
  body_name: str = "right_wrist_yaw_link"
  command_name: str = "desired_force"

  def build(self, env: ManagerBasedRlEnv) -> ForceFieldAction:
    return ForceFieldAction(self, env)


class ForceFieldAction(ActionTerm):
  """Apply impedance force to an EE body every substep.

  This is a zero-dimensional action term (no policy outputs). It reads
  the force command state (setpoint, stiffness, damping, ramp factor)
  and recomputes the impedance force from the current EE position and
  velocity at every substep, then writes it to ``xfrc_applied``.

  This replaces the ``apply_forcefield_force`` step event, which only
  updated the force once per policy step (50Hz). Moving it here gives
  200Hz force updates (or whatever the physics rate is).
  """

  cfg: ForceFieldActionCfg

  def __init__(self, cfg: ForceFieldActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._body_ids = self._entity.find_bodies(cfg.body_name)[0]
    self._command_name = cfg.command_name
    self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    pass  # No policy outputs to process.

  def apply_actions(self) -> None:
    from mjlab.tasks.force_control.mdp.commands import ForceFieldCommand

    cmd = cast(
      ForceFieldCommand,
      self._env.command_manager.get_term(self._command_name),
    )
    data = self._entity.data

    # Recompute impedance force from current EE state.
    ee_pos = data.body_link_pos_w[:, cmd._ee_body_id]
    ee_vel = data.body_link_vel_w[:, cmd._ee_body_id, :3]
    pos_err = ee_pos - cmd._forcefield_centers_w

    impedance_force = cmd._stiffness * pos_err + cmd._damping * ee_vel
    factor = cmd._compute_ramp_factor()
    raw_force = -impedance_force * factor.unsqueeze(-1)

    if cmd.cfg.max_applied_force is not None:
      max_force = cmd.cfg.max_applied_force
    else:
      max_force = max(
        abs(cmd.cfg.force_range[0][2]),
        abs(cmd.cfg.force_range[1][2]),
        50.0,
      )
    clamped_force = raw_force.clamp(-max_force, max_force)

    # Write to xfrc_applied.
    forces = clamped_force.unsqueeze(1)  # (N, 1, 3)
    torques = torch.zeros_like(forces)
    self._entity.write_external_wrench_to_sim(forces, torques, body_ids=self._body_ids)

    # Update command's cached forces (for obs/rewards that read them).
    cmd._forcefield_forces_w = clamped_force
    # Express in yaw-rotated frame (not EE body frame).
    from mjlab.tasks.force_control.mdp.commands import _yaw_frame_origin

    root_quat = data.root_link_quat_w
    _, yaw_quat = _yaw_frame_origin(data.root_link_pos_w, root_quat)
    cmd._forcefield_forces_b = quat_apply(quat_inv(yaw_quat), clamped_force)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    if isinstance(env_ids, slice):
      n = self.num_envs
    else:
      n = len(env_ids)
    if n == 0:
      return
    zeros = torch.zeros(n, 1, 3, device=self.device)
    self._entity.write_external_wrench_to_sim(
      zeros, zeros, env_ids=env_ids, body_ids=self._body_ids
    )
