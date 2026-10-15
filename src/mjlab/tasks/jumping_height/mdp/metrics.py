"""Jumping task metrics."""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.tasks.tracking.mdp import MotionCommand

from .phase import (
  kinematic_airborne_masks,
  resolve_motion_body_indexes,
)


def reward_term_state(
  env: ManagerBasedRlEnv,
  reward_name: str,
  attribute: str,
) -> torch.Tensor:
  """Expose a tensor maintained by a stateful reward term as a metric."""
  reward = env.reward_manager.get_term_cfg(reward_name).func
  value = getattr(reward, attribute)
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"Reward '{reward_name}' attribute '{attribute}' must be a tensor.")
  return value.float()


class last_jump_peak_root_height:
  """Track root apex above baseline for each environment's last completed jump."""

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    self._command_name = str(cfg.params["command_name"])
    self._base_height = float(cfg.params.get("base_height", 0.79))
    self._airborne_height_threshold = float(
      cfg.params.get("airborne_height_threshold", 0.06)
    )
    command = env.command_manager.get_term(self._command_name)
    assert isinstance(command, MotionCommand)
    self._foot_body_indexes = resolve_motion_body_indexes(
      command, tuple(cfg.params["foot_body_names"])
    )
    self.current_peak = torch.zeros(env.num_envs, device=env.device)
    self.last_peak = torch.zeros(env.num_envs, device=env.device)
    self.was_airborne = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del kwargs
    command = env.command_manager.get_term(self._command_name)
    assert isinstance(command, MotionCommand)
    robot_airborne, _ = kinematic_airborne_masks(
      env,
      command,
      self._foot_body_indexes,
      self._airborne_height_threshold,
    )
    root_height = (
      env.scene["robot"].data.root_link_pose_w[:, 2]
      - env.scene.env_origins[:, 2]
      - self._base_height
    ).clamp(min=0.0)

    takeoff = robot_airborne & ~self.was_airborne
    landed = ~robot_airborne & self.was_airborne
    self.current_peak = torch.where(takeoff, root_height, self.current_peak)
    self.current_peak = torch.where(
      robot_airborne,
      torch.maximum(self.current_peak, root_height),
      self.current_peak,
    )
    self.last_peak = torch.where(landed, self.current_peak, self.last_peak)
    self.current_peak = torch.where(
      landed, torch.zeros_like(self.current_peak), self.current_peak
    )
    self.was_airborne = robot_airborne
    return self.last_peak

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.current_peak[env_ids] = 0.0
    self.last_peak[env_ids] = 0.0
    self.was_airborne[env_ids] = False
