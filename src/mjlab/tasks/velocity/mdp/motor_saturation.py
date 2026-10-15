from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _ids_to_tensor(
  ids: list[int] | slice,
  *,
  count: int,
  device: str,
) -> torch.Tensor:
  all_ids = torch.arange(count, dtype=torch.long, device=device)
  if isinstance(ids, slice):
    return all_ids[ids]
  return torch.tensor(ids, dtype=torch.long, device=device)


def _names_from_ids(names: Sequence[str], ids: list[int] | slice) -> Sequence[str]:
  if isinstance(ids, slice):
    return names[ids]
  return [names[i] for i in ids]


def _resolve_named_tensor(
  value: float | dict[str, float],
  *,
  target_names: Sequence[str],
  num_envs: int,
  device: str,
  field_name: str,
) -> torch.Tensor:
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
    for target_index, target_name in enumerate(target_names):
      matched_key = None
      matched_value = None
      for pattern, candidate_value in value.items():
        if re.fullmatch(pattern, target_name):
          if matched_key is not None:
            raise ValueError(
              f"Multiple {field_name} values match target '{target_name}': "
              f"{matched_key!r} and {pattern!r}."
            )
          matched_key = pattern
          matched_value = float(candidate_value)
      if matched_value is not None:
        resolved[:, target_index] = matched_value
    if torch.isnan(resolved).any():
      missing = [
        name
        for index, name in enumerate(target_names)
        if torch.isnan(resolved[0, index]).item()
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


class MotorSaturationUsage:
  """Compute per-motor torque usage against static or dynamic torque limits."""

  def __init__(
    self,
    cfg: Any,
    env: ManagerBasedRlEnv,
    *,
    default_mode: str,
  ):
    params = cfg.params
    self._asset_cfg: SceneEntityCfg = params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self._action_name: str = params.get("action_name", "joint_pos")
    self._mode: str = params.get("mode", default_mode)
    self._eps = float(params.get("eps", 1.0e-6))

    if self._mode not in {"static", "dynamic"}:
      raise ValueError(f"Unsupported saturation mode: {self._mode!r}.")

    asset: Entity = env.scene[self._asset_cfg.name]
    self._joint_ids = _ids_to_tensor(
      self._asset_cfg.joint_ids,
      count=asset.num_joints,
      device=env.device,
    )
    self._joint_names = _names_from_ids(asset.joint_names, self._asset_cfg.joint_ids)

    action = env.action_manager.get_term(self._action_name)
    action_target_indices = self._resolve_action_target_indices(action)
    self._ctrl_ids = self._resolve_ctrl_ids(asset, action, action_target_indices, env)

    if self._mode == "dynamic":
      self._saturation_effort = self._resolve_limit_source(
        params=params,
        action=action,
        action_target_indices=action_target_indices,
        env=env,
        field_name="saturation_effort",
      )
      self._velocity_limit = self._resolve_limit_source(
        params=params,
        action=action,
        action_target_indices=action_target_indices,
        env=env,
        field_name="velocity_limit",
      )

  @property
  def joint_ids(self) -> torch.Tensor:
    return self._joint_ids

  @property
  def mode(self) -> str:
    return self._mode

  def compute(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene[self._asset_cfg.name]
    torques = torch.abs(asset.data.qfrc_actuator[:, self._joint_ids])
    limits = self.limits(env, asset)
    return torques / torch.clamp(limits, min=self._eps)

  def limits(self, env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    if self._mode == "static":
      return torch.abs(env.sim.model.actuator_forcerange[:, self._ctrl_ids, 1])

    joint_vel = asset.data.joint_vel[:, self._joint_ids]
    return self._saturation_effort * torch.clamp(
      1.0 - torch.abs(joint_vel / self._velocity_limit),
      min=0.0,
    )

  def _resolve_action_target_indices(self, action: Any) -> torch.Tensor:
    target_ids = action.target_ids
    if isinstance(target_ids, torch.Tensor):
      target_ids_list = [int(value) for value in target_ids.detach().cpu().tolist()]
    else:
      target_ids_list = [int(value) for value in target_ids]
    target_index_by_joint_id = {
      joint_id: index for index, joint_id in enumerate(target_ids_list)
    }

    joint_ids = [int(value) for value in self._joint_ids.detach().cpu().tolist()]
    missing = [
      joint_id for joint_id in joint_ids if joint_id not in target_index_by_joint_id
    ]
    if missing:
      raise ValueError(
        f"Action term '{self._action_name}' does not target joint ids {missing}."
      )

    return torch.tensor(
      [target_index_by_joint_id[joint_id] for joint_id in joint_ids],
      dtype=torch.long,
      device=self._joint_ids.device,
    )

  def _resolve_ctrl_ids(
    self,
    asset: Entity,
    action: Any,
    action_target_indices: torch.Tensor,
    env: ManagerBasedRlEnv,
  ) -> torch.Tensor:
    if hasattr(action, "_global_ctrl_ids"):
      return action._global_ctrl_ids[action_target_indices]

    ctrl_ids_by_name: dict[str, int] = {}
    for actuator in asset.actuators:
      for target_name, ctrl_id in zip(
        actuator.target_names,
        actuator.global_ctrl_ids.detach().cpu().tolist(),
        strict=True,
      ):
        if target_name in ctrl_ids_by_name:
          raise ValueError(f"Duplicate actuator target mapping for {target_name}.")
        ctrl_ids_by_name[target_name] = int(ctrl_id)

    target_index_list = [
      int(value) for value in action_target_indices.detach().cpu().tolist()
    ]
    target_names = [action.target_names[index] for index in target_index_list]
    missing = [name for name in target_names if name not in ctrl_ids_by_name]
    if missing:
      raise ValueError(f"Missing actuator ctrl ids for targets: {missing}.")

    return torch.tensor(
      [ctrl_ids_by_name[name] for name in target_names],
      dtype=torch.long,
      device=env.device,
    )

  def _resolve_limit_source(
    self,
    *,
    params: dict[str, Any],
    action: Any,
    action_target_indices: torch.Tensor,
    env: ManagerBasedRlEnv,
    field_name: str,
  ) -> torch.Tensor:
    cfg_value = params.get(field_name)
    if cfg_value is not None:
      return _resolve_named_tensor(
        cfg_value,
        target_names=self._joint_names,
        num_envs=env.num_envs,
        device=env.device,
        field_name=field_name,
      )

    attr_name = f"_{field_name}"
    if hasattr(action, attr_name):
      return getattr(action, attr_name)[:, action_target_indices]

    raise ValueError(
      f"Dynamic motor saturation requires '{field_name}' in the term params "
      f"or on action term '{self._action_name}'."
    )
