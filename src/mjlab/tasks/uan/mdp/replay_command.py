"""Replay-window command state for UAN rollout training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from .utils import (
  ACTION_DIM,
  EXTENSION_DIM,
  STATE_DIM,
  TERMINATION_DIM,
  build_valid_starts,
  load_replay_array,
  split_replay_array_with_contact,
  write_recorded_state,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ReplayWindowCommandCfg(CommandTermCfg):
  """Configuration for random replay-window sampling."""

  data_dir: str
  max_files: int | None = None
  asset_name: str = "robot"

  state_dim: int = STATE_DIM
  action_dim: int = ACTION_DIM
  extension_dim: int = EXTENSION_DIM
  termination_dim: int = TERMINATION_DIM

  history_horizon: int = 3
  rollout_horizon: int = 8
  require_contact_at_reset: bool = True
  contact_threshold: float = 0.8

  default_base_height: float = 0.75
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> ReplayWindowCommand:
    return ReplayWindowCommand(self, env)


class ReplayWindowCommand(CommandTerm):
  """Owns replay rows, sampled windows, and per-env replay cursors."""

  cfg: ReplayWindowCommandCfg

  def __init__(self, cfg: ReplayWindowCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    raw = load_replay_array(cfg.data_dir, cfg.max_files)
    state, action, ext, contact, term = split_replay_array_with_contact(
      raw,
      state_dim=cfg.state_dim,
      action_dim=cfg.action_dim,
      extension_dim=cfg.extension_dim,
      termination_dim=cfg.termination_dim,
    )
    valid_starts = build_valid_starts(
      term,
      history_horizon=cfg.history_horizon,
      rollout_horizon=cfg.rollout_horizon,
      contact=contact,
      require_contact_at_reset=cfg.require_contact_at_reset,
      contact_threshold=cfg.contact_threshold,
    )

    self.robot = env.scene[cfg.asset_name]
    self.state_raw = torch.as_tensor(state, device=self.device)
    self.action_raw = torch.as_tensor(action, device=self.device)
    self.ext_raw = torch.as_tensor(ext, device=self.device)
    self.contact_raw = torch.as_tensor(contact, device=self.device)
    self.term_raw = torch.as_tensor(term, device=self.device)
    self.valid_starts = torch.as_tensor(
      valid_starts, dtype=torch.long, device=self.device
    )

    self.cursor = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.start_row = torch.zeros_like(self.cursor)
    self.rollout_step = torch.zeros_like(self.cursor)
    self._joint_pos_rmse_sum = torch.zeros(self.num_envs, device=self.device)
    self._joint_pos_rmse_count = torch.zeros(self.num_envs, device=self.device)
    self._joint_pos_sq_error_sum: torch.Tensor | None = None
    self._joint_pos_metric_names: tuple[str, ...] = ()
    self.metrics["joint_pos_rmse"] = torch.zeros(self.num_envs, device=self.device)

    self.rng = torch.Generator(device=self.device)
    self.rng.manual_seed(int(env.cfg.seed or 0))

    print(
      f"[UANReplay] data_dir={cfg.data_dir} rows={self.state_raw.shape[0]} "
      f"valid_windows={self.valid_starts.numel()} num_envs={self.num_envs} "
      f"rollout_horizon={cfg.rollout_horizon} "
      f"row_timing=pre_action "
      f"reset_row_offset={self.reset_row_offset} "
      f"target_row_offset={self.target_row_offset} "
      "init_height_mode=root_height "
      f"require_contact_at_reset={cfg.require_contact_at_reset} "
      f"contact_threshold={cfg.contact_threshold}"
    )

  @property
  def command(self) -> torch.Tensor:
    return self.current_target

  @property
  def current_target(self) -> torch.Tensor:
    return self.action_raw[self.cursor]

  @property
  def current_gt_joint_pos(self) -> torch.Tensor:
    return self.state_raw[self.current_target_row, 9:38]

  @property
  def current_gt_joint_vel(self) -> torch.Tensor:
    return self.state_raw[self.current_target_row, 38:67]

  @property
  def current_action_state_row(self) -> torch.Tensor:
    return self.action_state_row_ids(self.cursor)

  @property
  def current_target_row(self) -> torch.Tensor:
    return self.target_row_ids(self.cursor)

  @property
  def reset_row_offset(self) -> int:
    return 0

  @property
  def target_row_offset(self) -> int:
    return 1

  def action_state_row_ids(self, action_row_ids: torch.Tensor) -> torch.Tensor:
    return action_row_ids + self.reset_row_offset

  def target_row_ids(self, action_row_ids: torch.Tensor) -> torch.Tensor:
    return action_row_ids + self.target_row_offset

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    if not isinstance(env_ids, torch.Tensor):
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    from .observations import prime_actor_replay_history_buffers

    prime_actor_replay_history_buffers(
      self._env,
      env_ids=env_ids,
      command_name="replay",
    )
    self.metrics["joint_pos_rmse"][env_ids] = self._joint_pos_rmse_sum[
      env_ids
    ] / self._joint_pos_rmse_count[env_ids].clamp_min(1.0)
    if self._joint_pos_sq_error_sum is not None:
      per_joint_rmse = (
        self._joint_pos_sq_error_sum[env_ids]
        / self._joint_pos_rmse_count[env_ids, None].clamp_min(1.0)
      ).sqrt()
      for joint_idx, metric_name in enumerate(self._joint_pos_metric_names):
        self.metrics[metric_name][env_ids] = per_joint_rmse[:, joint_idx]
    extras = {}
    for metric_name, metric_value in self.metrics.items():
      extras[metric_name] = torch.mean(metric_value[env_ids]).item()
      metric_value[env_ids] = 0.0
    self._joint_pos_rmse_sum[env_ids] = 0.0
    self._joint_pos_rmse_count[env_ids] = 0.0
    if self._joint_pos_sq_error_sum is not None:
      self._joint_pos_sq_error_sum[env_ids] = 0.0
    self.command_counter[env_ids] = 0
    return extras

  def compute(self, dt: float) -> None:
    self._update_metrics()
    if dt > 0.0:
      active = self._env.episode_length_buf > 0
      if active.any():
        self.cursor[active] += 1
        self.rollout_step[active] = self._env.episode_length_buf[active]

  def reset_to_random_window(self, env_ids: torch.Tensor) -> torch.Tensor:
    sample_ids = torch.randint(
      low=0,
      high=self.valid_starts.numel(),
      size=(env_ids.numel(),),
      device=self.device,
      generator=self.rng,
    )
    starts = self.valid_starts[sample_ids]
    self.start_row[env_ids] = starts
    self.cursor[env_ids] = starts
    self.rollout_step[env_ids] = 0
    return starts

  def write_recorded_state(
    self, env_ids: torch.Tensor, row_ids: torch.Tensor, joint_ids: torch.Tensor
  ) -> None:
    write_recorded_state(
      self._env,
      self.robot,
      state_raw=self.state_raw,
      ext_raw=self.ext_raw,
      row_ids=row_ids,
      env_ids=env_ids,
      joint_ids=joint_ids,
      default_base_height=self.cfg.default_base_height,
      init_height_mode="root_height",
    )

  def record_joint_position_error(
    self, joint_error: torch.Tensor, joint_names: list[str] | tuple[str, ...]
  ) -> None:
    self._ensure_joint_position_metrics(joint_error.shape[-1], joint_names)
    self._joint_pos_rmse_sum += joint_error.square().mean(dim=-1).sqrt()
    assert self._joint_pos_sq_error_sum is not None
    self._joint_pos_sq_error_sum += joint_error.square()
    self._joint_pos_rmse_count += 1.0

  def _ensure_joint_position_metrics(
    self, num_joints: int, joint_names: list[str] | tuple[str, ...]
  ) -> None:
    if self._joint_pos_sq_error_sum is not None:
      return
    if len(joint_names) != num_joints:
      joint_names = tuple(f"joint_{idx:02d}" for idx in range(num_joints))
    metric_names = []
    for joint_idx, joint_name in enumerate(joint_names):
      safe_name = str(joint_name).replace("/", "_").replace(" ", "_")
      metric_name = f"joint_pos_rmse/{safe_name}"
      if metric_name in self.metrics:
        metric_name = f"joint_pos_rmse/{joint_idx:02d}_{safe_name}"
      self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)
      metric_names.append(metric_name)
    self._joint_pos_metric_names = tuple(metric_names)
    self._joint_pos_sq_error_sum = torch.zeros(
      self.num_envs, num_joints, device=self.device
    )

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    del env_ids

  def _update_command(self) -> None:
    pass
