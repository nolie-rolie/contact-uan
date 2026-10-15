"""Reward functions for EE force control."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.force_control.mdp.commands import ForceFieldCommand
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def track_force_ee(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma: float = 10.0,
) -> torch.Tensor:
  """Exponential reward for tracking desired EE force.

  ``exp(-||target_f - applied_f||^2 / sigma^2)``
  """
  cmd = cast(ForceFieldCommand, env.command_manager.get_term(command_name))
  target = cmd.target_forces_w
  applied = cmd.forcefield_forces_w
  err_sq = torch.sum((target - applied) ** 2, dim=-1)
  return torch.exp(-err_sq / (sigma**2))


def track_force_ee_smoothness(
  env: ManagerBasedRlEnv,
  command_name: str,
  normalize_by_dt: bool = True,
) -> torch.Tensor:
  """Penalize rapid step-to-step changes in applied EE force."""
  cmd = cast(ForceFieldCommand, env.command_manager.get_term(command_name))
  applied = cmd.forcefield_forces_w

  attr = f"_force_smooth_prev_{command_name}"
  if not hasattr(env, attr):
    setattr(env, attr, applied.clone())
    return torch.zeros(env.num_envs, device=env.device)

  prev = getattr(env, attr)
  if prev.shape != applied.shape:
    setattr(env, attr, applied.clone())
    return torch.zeros(env.num_envs, device=env.device)

  delta = applied - prev
  setattr(env, attr, applied.clone())

  if normalize_by_dt:
    delta = delta / max(float(env.step_dt), 1e-6)

  penalty = torch.sum(delta**2, dim=-1)
  penalty[env.episode_length_buf <= 1] = 0.0
  return penalty


def track_lin_vel_xy_exp_zero(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward for zero linear velocity (standing still).

  Equivalent to ``track_linear_velocity`` with a zero velocity command.
  Penalizes all 3 axes (xy from command=0, z always toward zero).
  """
  asset = env.scene[asset_cfg.name]
  lin_vel_b = asset.data.root_link_lin_vel_b
  error = torch.sum(lin_vel_b**2, dim=1)
  return torch.exp(-error / (std**2))


def track_ang_vel_z_exp_zero(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward for zero angular velocity (standing still).

  Equivalent to ``track_angular_velocity`` with a zero velocity command.
  Penalizes all 3 axes (z from command=0, xy always toward zero).
  """
  asset = env.scene[asset_cfg.name]
  ang_vel_b = asset.data.root_link_ang_vel_b
  error = torch.sum(ang_vel_b**2, dim=1)
  return torch.exp(-error / (std**2))


def feet_slide_proportional(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize foot sliding: xy velocity weighted by contact force."""
  sensor = env.scene[sensor_name]
  asset = env.scene[asset_cfg.name]
  body_ids = cast(list[int], asset_cfg.body_ids)
  contact_force = sensor.data.force.norm(dim=-1)  # (num_envs, num_feet)
  foot_vel_xy = asset.data.body_link_lin_vel_w[:, body_ids, :2]
  slide = foot_vel_xy.norm(dim=-1)  # (num_envs, num_feet)
  # Weight by contact: only penalize sliding when in contact.
  return (slide * contact_force).sum(dim=-1)


def lin_vel_z_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize vertical base linear velocity (squared)."""
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b[:, 2].square()


def ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize roll/pitch angular velocity (sum of squares)."""
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b[:, :2].square().sum(dim=1)


def feet_parallel_to_ground(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  sensor_name: str | None = "feet_ground_contact",
) -> torch.Tensor:
  """Penalize feet not parallel to the ground, masked by contact.

  Measures deviation of each foot's local z-axis from world z.
  """
  asset = env.scene[asset_cfg.name]
  z_unit = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=torch.float32)
  total = torch.zeros(env.num_envs, device=env.device)

  body_ids = cast(list[int], asset_cfg.body_ids)
  in_contact = None
  if sensor_name is not None:
    sensor = env.scene[sensor_name]
    # found > 0 means contact detected; shape (num_envs, num_feet).
    in_contact = (sensor.data.found > 0).float()

  for i, body_id in enumerate(body_ids):
    foot_quat = asset.data.body_link_quat_w[:, body_id]
    foot_z = quat_apply(foot_quat, z_unit.unsqueeze(0).expand(env.num_envs, 3))
    err = 1.0 - torch.abs(foot_z[:, 2])
    if in_contact is not None:
      err = err * in_contact[:, i]
    total = total + err.square()

  return total


def feet_heading_aligned(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize heading misalignment between feet (dot product in xy)."""
  asset = env.scene[asset_cfg.name]
  x_unit = (
    torch.tensor([1.0, 0.0, 0.0], device=env.device, dtype=torch.float32)
    .unsqueeze(0)
    .expand(env.num_envs, 3)
  )

  body_ids = cast(list[int], asset_cfg.body_ids)
  left_id, right_id = body_ids[0], body_ids[1]
  left_fwd = quat_apply(asset.data.body_link_quat_w[:, left_id], x_unit)[:, :2]
  right_fwd = quat_apply(asset.data.body_link_quat_w[:, right_id], x_unit)[:, :2]
  left_fwd = left_fwd / left_fwd.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  right_fwd = right_fwd / right_fwd.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  dot = (left_fwd * right_fwd).sum(dim=-1)
  return (1.0 - dot).square()


class fixed_posture:
  """Penalize deviation from default pose with per-joint tolerance.

  Reward: ``exp(-mean(error² / std²))`` where error = joint_pos - default.
  Same as ``variable_posture`` but with a single fixed std (no speed-dependent
  switching, no velocity command dependency).

  Params:
    asset_cfg: Robot entity with ``joint_names`` set.
    std: dict mapping joint name patterns to std values.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    from mjlab.entity import Entity

    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.default_joint_pos = asset.data.default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    _, _, std_vals = resolve_matching_names_values(
      data=cfg.params["std"],
      list_of_strings=joint_names,
    )
    self.std = torch.tensor(std_vals, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    del std  # Resolved in __init__.
    asset = env.scene[asset_cfg.name]

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))


def selective_joint_deviation(
  env: ManagerBasedRlEnv,
  joint_names: list[str],
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize deviation from default pose for specific joints."""
  asset = env.scene[asset_cfg.name]
  ids, _ = asset.find_joints(joint_names)
  return (
    (asset.data.default_joint_pos[:, ids] - asset.data.joint_pos[:, ids])
    .abs()
    .sum(dim=-1)
  )
