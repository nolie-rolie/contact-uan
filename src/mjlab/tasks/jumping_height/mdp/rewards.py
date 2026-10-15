from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.utils.lab_api.math import euler_xyz_from_quat

from .phase import resolve_motion_body_indexes

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def body_roll_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize the sum of squared world-frame roll over selected bodies."""
  asset = env.scene[asset_cfg.name]
  body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
  batch_size, num_bodies = body_quat_w.shape[:2]
  roll, _, _ = euler_xyz_from_quat(body_quat_w.reshape(-1, 4))
  return roll.square().reshape(batch_size, num_bodies).sum(dim=-1)


class jump_com_ballistic_rise:
  """Reward predicted ballistic COM rise once per reference-motion segment.

  A reset, command resampling, or clip wrap opens one reward opportunity when
  the reference is grounded. The robot must then establish two-foot ground
  contact before its first confirmed two-foot takeoff can consume the
  opportunity. Reference landing does not re-arm it, so an additional hop
  within the same motion segment earns nothing. Returning the event value as a
  rate cancels the reward manager's normal ``step_dt`` scaling, making the
  episodic contribution equal to ``weight * v_z**2 / (2 * gravity)``.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._command_name = str(cfg.params["command_name"])
    self._sensor_name = str(cfg.params["sensor_name"])
    self._airborne_height_threshold = float(
      cfg.params.get("airborne_height_threshold", 0.06)
    )
    self._contact_confirmation_steps = int(
      cfg.params.get("contact_confirmation_steps", 2)
    )
    if self._contact_confirmation_steps < 1:
      raise ValueError(
        "contact_confirmation_steps must be positive, got "
        f"{self._contact_confirmation_steps}."
      )
    self._gravity = float(cfg.params.get("gravity", 9.81))
    if self._gravity <= 0.0:
      raise ValueError(f"gravity must be positive, got {self._gravity}.")
    self._step_dt = float(env.step_dt)
    if self._step_dt <= 0.0:
      raise ValueError(f"step_dt must be positive, got {self._step_dt}.")

    command = cast(MotionCommand, env.command_manager.get_term(self._command_name))
    self._foot_body_indexes = resolve_motion_body_indexes(
      command, tuple(cfg.params["foot_body_names"])
    )
    reference_foot_z = command.motion.body_pos_w[:, self._foot_body_indexes, 2]
    self.reference_airborne_by_frame = (
      reference_foot_z > self._airborne_height_threshold
    ).all(dim=1)

    self.reference_airborne = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self.previous_time_steps = torch.full(
      (env.num_envs,),
      -1,
      device=env.device,
      dtype=torch.long,
    )

    self.opportunity_armed = torch.zeros_like(self.reference_airborne)
    self.opportunity_consumed = torch.zeros_like(self.reference_airborne)
    self.takeoff_ready = torch.zeros_like(self.reference_airborne)
    self.grounded_streak = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
    self.airborne_streak = torch.zeros_like(self.grounded_streak)
    self.candidate_pending = torch.zeros_like(self.reference_airborne)
    self.candidate_com_z_velocity = torch.zeros(env.num_envs, device=env.device)
    self.active = torch.zeros_like(self.reference_airborne)

    self.com_z_velocity = torch.zeros(env.num_envs, device=env.device)
    self.last_takeoff_com_z_velocity = torch.zeros_like(self.com_z_velocity)
    self.last_ballistic_rise = torch.zeros_like(self.com_z_velocity)
    self.bonus_event_count = torch.zeros_like(self.grounded_streak)

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del kwargs
    command = cast(MotionCommand, env.command_manager.get_term(self._command_name))
    self.reference_airborne = self.reference_airborne_by_frame[command.time_steps]
    first_sample = self.previous_time_steps < 0
    command_discontinuity = ~first_sample & (
      command.time_steps != self.previous_time_steps + 1
    )
    new_segment = first_sample | command_discontinuity
    if torch.any(new_segment):
      self.opportunity_armed[new_segment] = False
      self.opportunity_consumed[new_segment] = False
      self.takeoff_ready[new_segment] = False
      self.grounded_streak[new_segment] = 0
      self.airborne_streak[new_segment] = 0
      self.candidate_pending[new_segment] = False
      self.candidate_com_z_velocity[new_segment] = 0.0
      self.opportunity_armed[new_segment & ~self.reference_airborne] = True

    sensor: ContactSensor = env.scene[self._sensor_name]
    assert sensor.data.found is not None
    if sensor.data.found.shape[1] != len(self._foot_body_indexes):
      raise ValueError(
        f"Expected {len(self._foot_body_indexes)} foot contacts from "
        f"'{self._sensor_name}', got shape {tuple(sensor.data.found.shape)}."
      )
    foot_contact = sensor.data.found > 0
    both_grounded = foot_contact.all(dim=1)
    both_airborne = ~foot_contact.any(dim=1)

    robot_data = env.scene["robot"].data
    body_masses = robot_data.model.body_mass[:, robot_data.indexing.body_ids]
    body_com_z_velocity = robot_data.body_com_lin_vel_w[:, :, 2]
    self.com_z_velocity = (body_masses * body_com_z_velocity).sum(dim=1) / (
      body_masses.sum(dim=1)
    )

    available = self.opportunity_armed & ~self.opportunity_consumed
    establishing_contact = available & both_grounded & ~self.takeoff_ready
    self.grounded_streak = torch.where(
      establishing_contact,
      self.grounded_streak + 1,
      torch.where(
        self.takeoff_ready,
        self.grounded_streak,
        torch.zeros_like(self.grounded_streak),
      ),
    )
    self.takeoff_ready |= (
      self.grounded_streak >= self._contact_confirmation_steps
    ) & available

    first_shared_airborne = (
      available & self.takeoff_ready & both_airborne & (self.airborne_streak == 0)
    )
    self.candidate_pending[first_shared_airborne] = True
    self.candidate_com_z_velocity[first_shared_airborne] = self.com_z_velocity[
      first_shared_airborne
    ]
    self.airborne_streak = torch.where(
      both_airborne,
      self.airborne_streak + 1,
      torch.zeros_like(self.airborne_streak),
    )
    self.candidate_pending &= both_airborne

    self.active = (
      available
      & self.candidate_pending
      & (self.airborne_streak >= self._contact_confirmation_steps)
    )
    ballistic_rise = self.candidate_com_z_velocity.clamp(min=0.0).square() / (
      2.0 * self._gravity
    )
    reward = torch.zeros_like(self.com_z_velocity)
    reward[self.active] = ballistic_rise[self.active] / self._step_dt

    self.last_takeoff_com_z_velocity[self.active] = self.candidate_com_z_velocity[
      self.active
    ]
    self.last_ballistic_rise[self.active] = ballistic_rise[self.active]
    self.bonus_event_count += self.active.long()
    self.opportunity_consumed |= self.active
    self.takeoff_ready[self.active] = False
    self.candidate_pending[self.active] = False

    self.previous_time_steps.copy_(command.time_steps)
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.reference_airborne[env_ids] = False
    self.previous_time_steps[env_ids] = -1
    self.opportunity_armed[env_ids] = False
    self.opportunity_consumed[env_ids] = False
    self.takeoff_ready[env_ids] = False
    self.grounded_streak[env_ids] = 0
    self.airborne_streak[env_ids] = 0
    self.candidate_pending[env_ids] = False
    self.candidate_com_z_velocity[env_ids] = 0.0
    self.active[env_ids] = False
    self.com_z_velocity[env_ids] = 0.0
    self.last_takeoff_com_z_velocity[env_ids] = 0.0
    self.last_ballistic_rise[env_ids] = 0.0
    self.bonus_event_count[env_ids] = 0
