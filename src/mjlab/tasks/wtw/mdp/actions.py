"""UAN-augmented joint position action term.

The RL policy outputs 29D joint position targets (MuJoCo order).
This action term adds UAN residuals internally before applying to the sim.
The UAN operates in native MuJoCo joint order — no reindexing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from mjlab.envs.mdp.actions.actions import (
  JointPositionAction,
  JointPositionActionCfg,
)
from mjlab.rl.uan_actor import UANActor
from mjlab.tasks.uan.mdp.utils import resolve_uan_obs_block_names

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# -- UAN observation helpers (inlined, pure torch) -------------------------


def _compute_uan_obs_step(
  joint_pos: torch.Tensor,
  base_targets: torch.Tensor,
  joint_vel: torch.Tensor,
  vel_scale: float = 0.05,
) -> torch.Tensor:
  """Paper UAN inputs: target-position error followed by scaled velocity."""
  return torch.cat((base_targets - joint_pos, joint_vel * vel_scale), dim=-1)


def _build_uan_flat_obs(
  history_buffer: torch.Tensor,
  num_joints: int = 29,
  num_blocks: int = 2,
) -> torch.Tensor:
  """Flatten history buffer to time-major per-term layout.

  history_buffer: (N, H, num_blocks * num_joints)
  Returns: (N, num_blocks * H * num_joints)
  """
  N = history_buffer.shape[0]
  parts = []
  for block_idx in range(num_blocks):
    start = block_idx * num_joints
    block = history_buffer[:, :, start : start + num_joints]
    parts.append(block.reshape(N, -1))
  return torch.cat(parts, dim=-1)


def _resolve_uan_obs_block_names_from_checkpoint_meta(
  metadata: dict,
) -> tuple[str, ...]:
  return resolve_uan_obs_block_names(metadata)


# -- Action term -----------------------------------------------------------


@dataclass(kw_only=True)
class UANAugmentedJointPositionActionCfg(JointPositionActionCfg):
  """Configuration for UAN-augmented joint position control.

  The UAN is loaded from a ``.pt`` training checkpoint via
  ``UANActor.from_checkpoint()``.  History length and action clip
  are read from the checkpoint metadata automatically.
  """

  uan_checkpoint_path: str | None = None
  """Path to a single UAN training checkpoint (``.pt`` file)."""

  uan_action_clip: float | None = None
  """Clip UAN residual output to [-clip, clip].  Loaded from
  checkpoint metadata when ``None``."""

  uan_vel_scale: float = 0.05
  """Scale factor for joint velocity in UAN observation."""

  def build(self, env: ManagerBasedRlEnv) -> UANAugmentedJointPositionAction:
    return UANAugmentedJointPositionAction(self, env)


class UANAugmentedJointPositionAction(JointPositionAction):
  """Joint position action with frozen UAN residual correction.

  Everything operates in native MuJoCo joint order — no reindexing.

  Flow:
      1. process_actions(raw_29d_mj) -> scale+offset -> _processed_actions
      2. process_actions():
         a. Build UAN obs from sim state + targets (MJ order)
         b. UAN forward -> 29D residual (MJ order)
         c. corrected = target + residual, cache for substeps
      3. apply_actions() writes the cached target every physics substep
  """

  # Constructed only with this specialized action config.
  cfg: UANAugmentedJointPositionActionCfg  # pyright: ignore[reportIncompatibleVariableOverride]

  def __init__(
    self,
    cfg: UANAugmentedJointPositionActionCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg=cfg, env=env)
    checkpoint_path = cfg.uan_checkpoint_path
    if not checkpoint_path:
      raise ValueError("No UAN checkpoint provided. Pass uan_checkpoint_path.")

    self._uan = UANActor.from_checkpoint(checkpoint_path, device=self.device)
    self._uan.eval()
    ckpt_meta = torch.load(
      checkpoint_path,
      map_location=self.device,
      weights_only=False,
    )
    self._uan_history_length = ckpt_meta.get("uan_history_length", 3)
    self._uan_action_clip = cfg.uan_action_clip
    if self._uan_action_clip is None:
      self._uan_action_clip = ckpt_meta.get("uan_action_clip", 1.0)
    self._uan_obs_block_names = _resolve_uan_obs_block_names_from_checkpoint_meta(
      ckpt_meta
    )
    self._uan_num_blocks = len(self._uan_obs_block_names)
    meta_num_blocks = ckpt_meta.get("num_blocks")
    if meta_num_blocks is not None and int(meta_num_blocks) != self._uan_num_blocks:
      raise ValueError(
        "UAN checkpoint num_blocks does not match resolved block semantics: "
        f"metadata num_blocks={meta_num_blocks}, "
        f"resolved block names={self._uan_obs_block_names}."
      )
    print(
      f"[UAN] Loaded checkpoint: {checkpoint_path}; "
      f"history_length={self._uan_history_length}, "
      f"action_clip={self._uan_action_clip}, "
      f"num_blocks={self._uan_num_blocks}, "
      f"block_names={self._uan_obs_block_names}, "
      f"output_per_joint={self._uan.output_per_joint}"
    )

    num_joints = self._uan.num_joints

    # (num_envs, history_length, num_blocks * num_joints)
    self._uan_history = torch.zeros(
      self.num_envs,
      self._uan_history_length,
      self._uan_num_blocks * num_joints,
      device=self.device,
    )
    self._last_uan_action = torch.zeros(
      self.num_envs,
      num_joints,
      device=self.device,
    )
    self._last_policy_target = torch.zeros(
      self.num_envs,
      num_joints,
      device=self.device,
    )
    self._last_corrected_target = torch.zeros(
      self.num_envs,
      num_joints,
      device=self.device,
    )

  def process_actions(self, actions: torch.Tensor) -> None:
    """Process one policy action and update UAN state once.

    ``apply_actions`` is called at the physics substep rate, so UAN history and
    inference live here to stay aligned with the policy/UAN training rate.
    """
    super().process_actions(actions)
    base_target = self._processed_actions
    num_joints = self._uan.num_joints
    expected_obs_dim = self._uan_history.shape[-1]
    expected_num_blocks = expected_obs_dim // num_joints

    joint_pos = self._entity.data.joint_pos[:, self._target_ids]
    joint_vel = self._entity.data.joint_vel[:, self._target_ids]

    obs_step = _compute_uan_obs_step(
      joint_pos,
      base_target,
      joint_vel,
      vel_scale=self.cfg.uan_vel_scale,
    )
    if obs_step.shape[-1] != expected_obs_dim:
      raise RuntimeError(
        "UAN observation width mismatch during process_actions: "
        f"expected {expected_obs_dim} (blocks={expected_num_blocks}), "
        f"got {obs_step.shape[-1]}."
      )

    self._uan_history = torch.roll(self._uan_history, shifts=-1, dims=1)
    self._uan_history[:, -1] = obs_step

    flat_obs = _build_uan_flat_obs(
      self._uan_history,
      num_joints,
      self._uan_num_blocks,
    )

    with torch.no_grad():
      obs_td = TensorDict({"uan_obs": flat_obs}, batch_size=[flat_obs.shape[0]])
      uan_action = self._uan(obs_td, stochastic_output=False)
    if self._uan_action_clip is not None:
      uan_action = uan_action.clamp(-self._uan_action_clip, self._uan_action_clip)
    self._last_uan_action = uan_action.clone()
    self._last_policy_target = base_target.clone()

    corrected = base_target + uan_action
    self._last_corrected_target = corrected.clone()

  def apply_actions(self) -> None:
    """Write the cached UAN-corrected target during each physics substep."""
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      self._last_corrected_target - encoder_bias, joint_ids=self._target_ids
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self._uan_history[env_ids] = 0.0
    self._last_uan_action[env_ids] = 0.0
    self._last_policy_target[env_ids] = 0.0
    self._last_corrected_target[env_ids] = 0.0
