"""Ported WTW reward functions for mjlab.

These are mjlab-compatible ports of the Isaac Lab WTW reward functions from
source/anystunt_gym/.../tracking/mdp/rewards.py and utils.py.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_yaw, quat_conjugate

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# -- Gait timing helpers --------------------------------------------------


def _get_command_tensor(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  if command is None:
    raise TypeError(f"Command {command_name!r} is not active.")
  return cast(torch.Tensor, command)


def _set_env_cache(env: ManagerBasedRlEnv, name: str, value: object) -> None:
  setattr(env, name, value)


def _get_env_cache(env: ManagerBasedRlEnv, name: str) -> object:
  return getattr(env, name)


def _get_frequency(
  env: ManagerBasedRlEnv,
  freq_range: tuple[float, float],
) -> torch.Tensor:
  """Map [-1,1] frequency command to actual Hz in freq_range."""
  freq = torch.full((env.num_envs,), freq_range[0], device=env.device)
  if "gait_frequency" in env.command_manager.active_terms:
    cmd = _get_command_tensor(env, "gait_frequency")[:, 0]
    freq = (cmd + 1) / 2 * (freq_range[1] - freq_range[0]) + freq_range[0]
  return freq


def _get_gait_timing(
  env: ManagerBasedRlEnv,
  command_name: str,
  freq_range: tuple[float, float],
  phase: float = 0.0,
) -> torch.Tensor:
  """Compute gait phase: sin(2*pi*freq*t + phase), zeroed when standing."""
  t = env.episode_length_buf * env.step_dt
  freq = _get_frequency(env, freq_range)
  gait_phase = torch.sin(2 * math.pi * freq * t + phase)

  vel_cmd = _get_command_tensor(env, command_name)
  motion_mask = vel_cmd.norm(dim=1) > 1e-6
  return gait_phase * motion_mask.float()


def _get_biped_desired_contact_status(
  env: ManagerBasedRlEnv,
  command_name: str,
  freq_range: tuple[float, float],
  overlap: float = 0.25,
) -> torch.Tensor:
  """Desired contact status for biped gait. Returns (N, 2) float."""
  timing = _get_gait_timing(env, command_name, freq_range)
  stance_A = (timing > -overlap).float()
  stance_B = (timing < overlap).float()
  desired = torch.zeros(env.num_envs, 2, device=env.device)
  desired[:, 0] = stance_A
  desired[:, 1] = stance_B
  return desired


# -- Reward functions ------------------------------------------------------


def power_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  joule_heating_scale: float = 0.03,
  k_regen: float = 0.0,
  gravity: float = 9.81,
) -> torch.Tensor:
  """Energy penalty: Joule heating + mechanical power + gravitational
  potential, normalized by commanded velocity."""
  asset: Entity = env.scene[asset_cfg.name]

  torques = asset.data.qfrc_actuator
  joint_vel = asset.data.joint_vel

  P_el = joule_heating_scale * torques.square().sum(dim=-1)

  mech_power = (torques * joint_vel).sum(dim=-1)
  P_mech = torch.where(mech_power > 0, mech_power, k_regen * mech_power)

  if not hasattr(env, "_robot_total_mass"):
    body_ids = asset.data.indexing.body_ids.cpu().numpy()
    _set_env_cache(
      env, "_robot_total_mass", float(env.sim.mj_model.body_mass[body_ids].sum())
    )
  robot_total_mass = cast(float, _get_env_cache(env, "_robot_total_mass"))
  root_vel_z = asset.data.root_link_lin_vel_w[:, 2]
  P_pot = robot_total_mass * gravity * root_vel_z

  P_total = P_el + P_mech + P_pot

  cmd_vel = _get_command_tensor(env, command_name)[:, :2]
  gamma_v = 1.0 / (cmd_vel.square().sum(dim=-1) + 1.0)

  return gamma_v * P_total


def biped_gait_phase(
  env: ManagerBasedRlEnv,
  command_name: str,
  freq_range: tuple[float, float],
  overlap: float = 0.25,
  sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Penalize deviation from desired gait contact pattern."""
  desired = _get_biped_desired_contact_status(env, command_name, freq_range, overlap)
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  contacts = (sensor.data.found > 0).float()
  contacts_match = (contacts == desired).float().sum(dim=1)
  return 2 - contacts_match


def biped_swing_height(
  env: ManagerBasedRlEnv,
  command_name: str,
  freq_range: tuple[float, float],
  overlap: float = 0.25,
  swing_height_range: tuple[float, float] = (0.05, 0.1),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  foot_radii: list[float] | None = None,
) -> torch.Tensor:
  """Penalize swing foot height tracking error."""
  if foot_radii is None:
    foot_radii = [0.05, 0.05]
  asset: Entity = env.scene[asset_cfg.name]

  foot_heights = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]

  timing = _get_gait_timing(env, command_name, freq_range)
  desired_contacts = _get_biped_desired_contact_status(
    env, command_name, freq_range, overlap
  )

  if "foot_swing_height" in env.command_manager.active_terms:
    cmd = _get_command_tensor(env, "foot_swing_height")[:, 0]
    swing_height = (cmd + 1) / 2 * (
      swing_height_range[1] - swing_height_range[0]
    ) + swing_height_range[0]
  else:
    swing_height = torch.full((env.num_envs,), swing_height_range[0], device=env.device)

  radii_t = torch.tensor(foot_radii, device=env.device).reshape(1, -1)
  foot_target_height = (
    swing_height.unsqueeze(1) * torch.abs((1 - desired_contacts) * timing.unsqueeze(1))
    + radii_t
  )

  tracking_errors = foot_heights - foot_target_height
  tracking_errors_clipped = tracking_errors.clamp(
    min=-swing_height_range[1], max=swing_height_range[1]
  )

  return (tracking_errors_clipped.square() * (1 - desired_contacts)).sum(dim=-1)


def biped_raibert_heuristic(
  env: ManagerBasedRlEnv,
  command_name: str,
  freq_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  stance_width: float = 0.21,
) -> torch.Tensor:
  """Penalize deviation from Raibert footstep placement heuristic."""
  asset: Entity = env.scene[asset_cfg.name]
  num_feet = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else 2

  foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  base_pos_w = asset.data.root_link_pos_w
  base_quat_w = asset.data.root_link_quat_w

  foot_pos_translated = foot_pos_w - base_pos_w.unsqueeze(1)
  quat_conj = quat_conjugate(base_quat_w)

  footsteps_in_body_frame = torch.zeros(env.num_envs, num_feet, 3, device=env.device)
  for i in range(num_feet):
    footsteps_in_body_frame[:, i, :] = quat_apply_yaw(
      quat_conj, foot_pos_translated[:, i, :]
    )
  actual_footsteps_xy = footsteps_in_body_frame[:, :, :2]

  vel_cmd = _get_command_tensor(env, command_name)
  x_vel_des = vel_cmd[:, 0:1]
  y_vel_des = vel_cmd[:, 1:2]
  yaw_vel_des = vel_cmd[:, 2:3]

  desired_ys_nom = (
    torch.tensor([stance_width / 2, -stance_width / 2], device=env.device)
    .unsqueeze(0)
    .expand(env.num_envs, -1)
  )
  desired_xs_nom = torch.zeros(env.num_envs, num_feet, device=env.device)

  timing = _get_gait_timing(env, command_name, freq_range, phase=math.pi / 2)
  phases = torch.stack([timing, -timing], dim=1)

  freq = _get_frequency(env, freq_range)
  step_time_half = (0.5 / freq).unsqueeze(1)

  vel_from_yaw_x = -yaw_vel_des * desired_ys_nom
  desired_xs_offset = phases * (x_vel_des + vel_from_yaw_x) * step_time_half
  desired_ys_offset = phases * y_vel_des * step_time_half

  desired_xs = desired_xs_nom + desired_xs_offset
  desired_ys = desired_ys_nom + desired_ys_offset

  desired_footsteps = torch.stack([desired_xs, desired_ys], dim=2)

  err = desired_footsteps - actual_footsteps_xy
  return err.square().sum(dim=(1, 2))


def biped_arm_swing(
  env: ManagerBasedRlEnv,
  command_name: str,
  freq_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  static_cfg: SceneEntityCfg | None = None,
  shoulder_amplitude: float = 0.2,
  elbow_amplitude: float = 0.08,
) -> torch.Tensor:
  """Penalize deviation from desired contralateral arm swing.

  asset_cfg.joint_ids must resolve to exactly 4 joints in order:
  [left_shoulder_pitch, right_shoulder_pitch, left_elbow, right_elbow].
  """
  asset: Entity = env.scene[asset_cfg.name]

  cos_phase = _get_gait_timing(env, command_name, freq_range, phase=math.pi / 2)
  vel_cmd = _get_command_tensor(env, command_name)
  lin_vel_x = vel_cmd[:, 0]

  arm_amp = (shoulder_amplitude * lin_vel_x.abs()).clamp(max=shoulder_amplitude)
  elbow_amp = (elbow_amplitude * lin_vel_x.abs()).clamp(max=elbow_amplitude)

  desired_offsets = torch.stack(
    [
      arm_amp * cos_phase,
      -arm_amp * cos_phase,
      elbow_amp * cos_phase,
      -elbow_amp * cos_phase,
    ],
    dim=-1,
  )

  default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  current_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  actual_offsets = current_pos - default_pos

  error = (actual_offsets - desired_offsets).abs().sum(dim=-1)

  if static_cfg is not None:
    static_default = asset.data.default_joint_pos[:, static_cfg.joint_ids]
    static_current = asset.data.joint_pos[:, static_cfg.joint_ids]
    error = error + (static_current - static_default).abs().sum(dim=-1)

  return error


def joint_deviation_from_nominal(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint deviation from default pose."""
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  current = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return (default - current).abs().sum(dim=-1)


def feet_slide_proportional(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Penalize foot sliding proportional to contact force magnitude.

  Reward = sum over feet of (foot_xy_speed * contact_force_norm).
  Matches Isaac Lab ``feet_slide_proportional``.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  asset: Entity = env.scene[asset_cfg.name]

  # Contact force magnitude per foot: [B, N]
  assert sensor.data.force is not None
  contact_forces = sensor.data.force.norm(dim=-1)

  # Foot XY velocity: [B, N, 2]
  body_vel_xy = asset.data.body_link_vel_w[:, asset_cfg.body_ids, :2]

  return torch.sum(body_vel_xy.norm(dim=-1) * contact_forces, dim=1)
