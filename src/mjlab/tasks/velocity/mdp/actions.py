"""Velocity-task-specific action terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.rl.uan_actor import UANActor
from mjlab.tasks.wtw.mdp.actions import (
  _compute_uan_obs_step,
  _resolve_uan_obs_block_names_from_checkpoint_meta,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _build_uan_flat_obs(
  history_buffer: torch.Tensor,
  num_joints: int,
  num_blocks: int,
) -> torch.Tensor:
  """Flatten history buffer to time-major per-term layout."""
  num_envs = history_buffer.shape[0]
  parts = []
  for block_idx in range(num_blocks):
    start = block_idx * num_joints
    block = history_buffer[:, :, start : start + num_joints]
    parts.append(block.reshape(num_envs, -1))
  return torch.cat(parts, dim=-1)


def _resolve_target_tensor(
  value: float | dict[str, float],
  *,
  target_names: list[str],
  num_envs: int,
  device: str,
  field_name: str,
) -> torch.Tensor:
  """Resolve a scalar-or-regex mapping to a per-target tensor."""
  if isinstance(value, (float, int)):
    resolved = torch.full(
      (num_envs, len(target_names)),
      float(value),
      dtype=torch.float,
      device=device,
    )
  elif isinstance(value, dict):
    resolved = torch.full(
      (num_envs, len(target_names)),
      float("nan"),
      dtype=torch.float,
      device=device,
    )
    index_list, _, value_list = resolve_matching_names_values(value, target_names)
    resolved[:, index_list] = torch.tensor(value_list, dtype=torch.float, device=device)
    if torch.isnan(resolved).any():
      missing = [
        name
        for idx, name in enumerate(target_names)
        if torch.isnan(resolved[0, idx]).item()
      ]
      raise ValueError(f"Missing {field_name} values for targets: {missing}.")
  else:
    raise TypeError(
      f"Unsupported {field_name} type: {type(value).__name__}. "
      "Expected float or dict[str, float]."
    )

  if torch.any(resolved <= 0.0):
    raise ValueError(f"{field_name} must be strictly positive for every target.")
  return resolved


def _clip_position_targets(
  *,
  position_target: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  kp: torch.Tensor,
  kd: torch.Tensor,
  saturation_effort: torch.Tensor,
  velocity_limit: torch.Tensor,
) -> torch.Tensor:
  """Clip targets to the paper's asymmetric torque-speed envelope.

  Both bounds constrain ``kp * (target - joint_pos) - kd * joint_vel``.
  Clipping must include the opposite bound when damping reverses the torque
  relative to position error. At or above no-load speed, motoring torque is
  zero while full braking torque remains available. A frozen UAN adds its
  simulator correction after this operation.
  """
  speed_fraction = joint_vel / velocity_limit
  min_torque = -saturation_effort * (1.0 + torch.clamp(speed_fraction, -1.0, 0.0))
  max_torque = saturation_effort * (1.0 - torch.clamp(speed_fraction, 0.0, 1.0))
  upper = joint_pos + (max_torque + kd * joint_vel) / kp
  lower = joint_pos + (min_torque + kd * joint_vel) / kp
  return torch.clamp(position_target, min=lower, max=upper)


@dataclass(kw_only=True)
class TorqueSpeedClippedJointPositionActionCfg(JointPositionActionCfg):
  """Joint position action with asymmetric torque-speed target clipping."""

  saturation_effort: float | dict[str, float]
  """Per-target torque at zero speed."""

  velocity_limit: float | dict[str, float]
  """Per-target no-load speed."""

  def build(self, env: ManagerBasedRlEnv) -> TorqueSpeedClippedJointPositionAction:
    return TorqueSpeedClippedJointPositionAction(self, env)


class TorqueSpeedClippedJointPositionAction(JointPositionAction):
  """Clip both target bounds using the paper's asymmetric torque-speed envelope."""

  # The constructor and config factory require this specialized config type.
  cfg: TorqueSpeedClippedJointPositionActionCfg  # pyright: ignore[reportIncompatibleVariableOverride]

  def __init__(
    self, cfg: TorqueSpeedClippedJointPositionActionCfg, env: ManagerBasedRlEnv
  ):
    super().__init__(cfg=cfg, env=env)
    self._saturation_effort = _resolve_target_tensor(
      cfg.saturation_effort,
      target_names=self._target_names,
      num_envs=self.num_envs,
      device=self.device,
      field_name="saturation_effort",
    )
    self._velocity_limit = _resolve_target_tensor(
      cfg.velocity_limit,
      target_names=self._target_names,
      num_envs=self.num_envs,
      device=self.device,
      field_name="velocity_limit",
    )
    self._global_ctrl_ids = self._resolve_global_ctrl_ids()
    self._last_clipped_target = torch.zeros_like(self._processed_actions)

  def _resolve_global_ctrl_ids(self) -> torch.Tensor:
    ctrl_ids_by_name: dict[str, int] = {}
    for actuator in self._entity.actuators:
      for target_name, ctrl_id in zip(
        actuator.target_names,
        actuator.global_ctrl_ids.tolist(),
        strict=True,
      ):
        if target_name in ctrl_ids_by_name:
          raise ValueError(f"Duplicate actuator target mapping for {target_name}.")
        ctrl_ids_by_name[target_name] = int(ctrl_id)

    missing = [name for name in self._target_names if name not in ctrl_ids_by_name]
    if missing:
      raise ValueError(f"Missing actuator ctrl ids for targets: {missing}.")

    return torch.tensor(
      [ctrl_ids_by_name[name] for name in self._target_names],
      dtype=torch.long,
      device=self.device,
    )

  def _current_pd_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
    kp = self._env.sim.model.actuator_gainprm[:, self._global_ctrl_ids, 0]
    kd = -self._env.sim.model.actuator_biasprm[:, self._global_ctrl_ids, 2]
    if torch.any(kp <= 0.0):
      raise RuntimeError(
        "Torque-speed clipping requires strictly positive actuator kp."
      )
    if torch.any(kd < 0.0):
      raise RuntimeError("Torque-speed clipping requires non-negative actuator kd.")
    return kp, kd

  def _clip_target(self, position_target: torch.Tensor) -> torch.Tensor:
    joint_pos = self._entity.data.joint_pos[:, self._target_ids]
    joint_vel = self._entity.data.joint_vel[:, self._target_ids]
    kp, kd = self._current_pd_gains()
    clipped = _clip_position_targets(
      position_target=position_target,
      joint_pos=joint_pos,
      joint_vel=joint_vel,
      kp=kp,
      kd=kd,
      saturation_effort=self._saturation_effort,
      velocity_limit=self._velocity_limit,
    )
    self._last_clipped_target[:] = clipped
    return clipped

  def process_actions(self, actions: torch.Tensor) -> None:
    """Clip once per policy step, matching the deployed controller."""
    super().process_actions(actions)
    self._clip_target(self._processed_actions)

  def apply_actions(self) -> None:
    """Hold the clipped target during the policy step's physics substeps."""
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      self._last_clipped_target - encoder_bias,
      joint_ids=self._target_ids,
    )


@dataclass(kw_only=True)
class UANAugmentedTorqueSpeedClippedJointPositionActionCfg(
  TorqueSpeedClippedJointPositionActionCfg
):
  """Torque-speed-clipped joint position action with a frozen UAN residual."""

  uan_checkpoint_path: str | None = None
  """Path to a single UAN training checkpoint (``.pt`` file)."""

  uan_action_clip: float | None = None
  """Optional clip on the UAN residual output."""

  uan_vel_scale: float = 0.05
  """Scale factor for joint velocity in the UAN observation."""

  def build(
    self, env: ManagerBasedRlEnv
  ) -> UANAugmentedTorqueSpeedClippedJointPositionAction:
    return UANAugmentedTorqueSpeedClippedJointPositionAction(self, env)


class UANAugmentedTorqueSpeedClippedJointPositionAction(
  TorqueSpeedClippedJointPositionAction
):
  """Clip the base policy target, then add the frozen UAN residual.

  The corrected target is cached without another clipping pass and applied
  during each physics substep.
  """

  # The constructor and config factory require this specialized config type.
  cfg: UANAugmentedTorqueSpeedClippedJointPositionActionCfg  # pyright: ignore[reportIncompatibleVariableOverride]

  def __init__(
    self,
    cfg: UANAugmentedTorqueSpeedClippedJointPositionActionCfg,
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
    num_joints = self._uan.num_joints
    print(
      f"[UAN] Loaded checkpoint: {checkpoint_path}; "
      f"history_length={self._uan_history_length}, "
      f"action_clip={self._uan_action_clip}, "
      f"num_blocks={self._uan_num_blocks}, "
      f"block_names={self._uan_obs_block_names}, "
      f"output_per_joint={self._uan.output_per_joint}"
    )

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
    self._last_policy_target_clipped = torch.zeros_like(self._processed_actions)
    self._last_corrected_target = torch.zeros_like(self._processed_actions)
    self._last_uan_obs_step = torch.zeros(
      self.num_envs,
      self._uan_num_blocks * num_joints,
      device=self.device,
    )

  def process_actions(self, actions: torch.Tensor) -> None:
    """Process one policy action and update UAN state once."""
    super().process_actions(actions)
    policy_target_clipped = self._last_clipped_target
    self._last_policy_target_clipped[:] = policy_target_clipped

    num_joints = self._uan.num_joints
    expected_obs_dim = self._uan_history.shape[-1]
    expected_num_blocks = expected_obs_dim // num_joints

    joint_pos = self._entity.data.joint_pos[:, self._target_ids]
    joint_vel = self._entity.data.joint_vel[:, self._target_ids]

    obs_step = _compute_uan_obs_step(
      joint_pos,
      policy_target_clipped,
      joint_vel,
      vel_scale=self.cfg.uan_vel_scale,
    )
    if obs_step.shape[-1] != expected_obs_dim:
      raise RuntimeError(
        "UAN observation width mismatch during process_actions: "
        f"expected {expected_obs_dim} (blocks={expected_num_blocks}), "
        f"got {obs_step.shape[-1]}."
      )
    self._last_uan_obs_step[:] = obs_step

    self._uan_history = torch.roll(self._uan_history, shifts=-1, dims=1)
    self._uan_history[:, -1] = obs_step

    flat_obs = _build_uan_flat_obs(
      self._uan_history,
      num_joints,
      self._uan_num_blocks,
    )

    with torch.no_grad():
      from tensordict import TensorDict

      obs_td = TensorDict({"uan_obs": flat_obs}, batch_size=[flat_obs.shape[0]])
      uan_action = self._uan(obs_td, stochastic_output=False)
    if self._uan_action_clip is not None:
      uan_action = uan_action.clamp(-self._uan_action_clip, self._uan_action_clip)
    self._last_uan_action[:] = uan_action

    corrected = policy_target_clipped + uan_action
    self._last_corrected_target[:] = corrected

  def apply_actions(self) -> None:
    """Write the cached UAN-corrected target during each physics substep."""
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      self._last_corrected_target - encoder_bias,
      joint_ids=self._target_ids,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self._uan_history[env_ids] = 0.0
    self._last_uan_action[env_ids] = 0.0
    self._last_policy_target_clipped[env_ids] = 0.0
    self._last_corrected_target[env_ids] = 0.0
    self._last_uan_obs_step[env_ids] = 0.0
