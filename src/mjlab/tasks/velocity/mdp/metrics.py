from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp.motor_saturation import MotorSaturationUsage
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

DEFAULT_HEEL_STRIKE_COMMAND_BINS: tuple[tuple[str, float, float], ...] = (
  ("vx_reverse", float("-inf"), -0.05),
  ("vx_near_zero", -0.05, 0.05),
  ("vx_0_1", 0.05, 1.0),
  ("vx_1_2", 1.0, 2.0),
  ("vx_2_3", 2.0, 3.0),
  ("vx_ge_3", 3.0, float("inf")),
)


def _log_metric(env: ManagerBasedRlEnv, key: str, value: torch.Tensor) -> None:
  env.extras.setdefault("log", {})[key] = value.detach()


def heel_strike_angle_by_command(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_name: str | None = "twist",
  command_bins: tuple[tuple[str, float, float], ...] = DEFAULT_HEEL_STRIKE_COMMAND_BINS,
) -> torch.Tensor:
  """Log touchdown foot pitch angle globally and bucketed by forward command."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_mask = first_contact.float()

  asset: Entity = env.scene[asset_cfg.name]
  body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
  local_x = torch.zeros(*body_quat_w.shape[:-1], 3, device=env.device)
  local_x[..., 0] = 1.0
  foot_forward_w = quat_apply(body_quat_w, local_x)
  pitch_sin = torch.abs(foot_forward_w[..., 2])
  angle_rad = torch.asin(torch.clamp(pitch_sin, max=1.0))

  strike_count = torch.sum(landing_mask)
  strike_angle_sum = torch.sum(angle_rad * landing_mask)
  strike_sin_sum = torch.sum(pitch_sin * landing_mask)
  safe_count = torch.clamp(strike_count, min=1.0)
  mean_angle_rad = strike_angle_sum / safe_count
  mean_pitch_sin = strike_sin_sum / safe_count

  _log_metric(env, "Metrics/heel_strike_angle_rad_mean", mean_angle_rad)
  _log_metric(env, "Metrics/heel_strike_count", strike_count)
  _log_metric(env, "Metrics/foot_strike_angle_mean", mean_pitch_sin)

  command = env.command_manager.get_command(command_name) if command_name else None
  if command is not None:
    command_x = command[:, 0].unsqueeze(-1)
    for bin_name, lower, upper in command_bins:
      in_bin = (command_x >= lower) & (command_x < upper)
      bin_mask = landing_mask * in_bin.float()
      bin_count = torch.sum(bin_mask)
      bin_mean = torch.sum(angle_rad * bin_mask) / torch.clamp(bin_count, min=1.0)
      _log_metric(env, f"Metrics/heel_strike_angle_rad/{bin_name}", bin_mean)
      _log_metric(env, f"Metrics/heel_strike_count/{bin_name}", bin_count)

  per_env_count = torch.sum(landing_mask, dim=-1)
  return torch.sum(angle_rad * landing_mask, dim=-1) / torch.clamp(
    per_env_count, min=1.0
  )


class motor_saturation_fraction:
  """Fraction of selected motors at or above a static or dynamic torque limit."""

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    self._usage = MotorSaturationUsage(cfg, env, default_mode="static")
    self._threshold = float(cfg.params.get("threshold", 0.95))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    **kwargs,
  ) -> torch.Tensor:
    del kwargs  # Parameters are resolved once during initialization.
    usage = self._usage.compute(env)
    return (usage >= self._threshold).float().mean(dim=-1)
