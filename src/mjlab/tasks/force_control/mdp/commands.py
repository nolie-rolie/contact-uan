"""Force field command term for EE force control.

Applies impedance-controlled forces to a single end-effector body with smooth
ramping to avoid discontinuities at command resampling boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.force_control.mdp.force_reach_utils import (
  clip_xy_force_to_reach,
)
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_inv,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_DEFAULT_ROOT_HEIGHT = 0.793
_DEFAULT_RIGHT_EE_OFFSET_B = (0.1998, -0.1487, 0.0952)


def _extract_yaw(quat: torch.Tensor) -> torch.Tensor:
  """Extract yaw angle from wxyz quaternion."""
  w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
  siny = 2 * (w * z + x * y)
  cosy = 1 - 2 * (y * y + z * z)
  return torch.atan2(siny, cosy)


def _yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
  """Convert yaw angle to wxyz quaternion."""
  half = yaw * 0.5
  w = torch.cos(half)
  z = torch.sin(half)
  zeros = torch.zeros_like(w)
  return torch.stack([w, zeros, zeros, z], dim=-1)


def _yaw_frame_origin(root_pos: torch.Tensor, root_quat: torch.Tensor):
  """Compute yaw-rotated frame: xy from root, fixed z, yaw-only rotation."""
  yaw = _extract_yaw(root_quat)
  yaw_quat = _yaw_to_quat(yaw)
  origin = torch.zeros_like(root_pos)
  origin[:, 0] = root_pos[:, 0]
  origin[:, 1] = root_pos[:, 1]
  origin[:, 2] = _DEFAULT_ROOT_HEIGHT
  return origin, yaw_quat


@dataclass(kw_only=True)
class ForceFieldCommandCfg(CommandTermCfg):
  """Configuration for the force field command term."""

  entity_name: str = "robot"
  """Name of the robot entity in the scene."""

  body_name: str = "right_wrist_yaw_link"
  """Name of the end-effector body to apply forces to."""

  force_range: tuple[tuple[float, float, float], tuple[float, float, float]] = (
    (-30.0, -30.0, -50.0),
    (30.0, 30.0, 50.0),
  )
  """Force range ((x_min, y_min, z_min), (x_max, y_max, z_max)) in Newtons."""

  stiffness_range: tuple[float, float] = (100.0, 1000.0)
  """Impedance controller stiffness range (N/m)."""

  damping_proportional_to_stiffness: float = 0.1
  """When > 0, damping = stiffness * this value (overrides damping_range)."""

  damping_range: tuple[float, float] = (10.0, 20.0)
  """Damping range (Ns/m), used only when damping_proportional_to_stiffness <= 0."""

  sample_zero_force_probability: float = 0.2
  """Probability of sampling zero force for a given environment."""

  force_ramp_duration_range_s: tuple[float, float] = (2.0, 4.0)
  """Range for the force ramp-up/ramp-down duration in seconds."""

  force_hold_duration_s: float = 2.0
  """Duration to hold the peak force in seconds."""

  startup_zero_force_duration_s: float = 2.0
  """Duration to hold zero force at the start of each episode."""

  max_applied_force: float | None = None
  """Clamp applied impedance force to [-val, val] N. If None, defaults to
  max(abs(force_range)) for backwards compatibility."""

  max_xy_reach_m: float | None = None
  """Clamp sampled x/y force commands so per-axis displacement |F| / stiffness
  stays within this reach limit in meters."""

  default_ee_offset_b: tuple[float, float, float] | None = _DEFAULT_RIGHT_EE_OFFSET_B
  """Default EE offset in the yaw-rotated body frame (relative to
  yaw-frame origin at z=DEFAULT_ROOT_HEIGHT).  When set, the impedance
  setpoint is fixed to this constant (the robot's default pose) instead
  of being read from the live EE position at reset.  ``None`` restores
  the previous live-position behaviour."""

  def build(self, env: ManagerBasedRlEnv) -> ForceFieldCommand:
    return ForceFieldCommand(self, env)


class ForceFieldCommand(CommandTerm):
  """Impedance-controlled force field applied to a single EE body.

  Lifecycle:
    1. ``reset()`` — set body-frame setpoint from current EE position (fixed
       for the episode), zero all forces.
    2. ``_resample_command()`` — sample new peak forces, stiffness, damping,
       and ramp duration.
    3. ``_update_command()`` — update world-frame setpoint from body-frame,
       apply force ramp envelope, compute impedance force.
  """

  cfg: ForceFieldCommandCfg

  def __init__(self, cfg: ForceFieldCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._entity = env.scene[cfg.entity_name]
    self._ee_body_id = self._entity.find_bodies(cfg.body_name)[0][0]

    N = self.num_envs
    D = self.device

    # Force state.
    self._peak_target_forces_w = torch.zeros(N, 3, device=D)
    self._target_forces_w = torch.zeros(N, 3, device=D)
    self._target_forces_b = torch.zeros(N, 3, device=D)
    self._forcefield_forces_w = torch.zeros(N, 3, device=D)
    self._forcefield_forces_b = torch.zeros(N, 3, device=D)

    # Impedance setpoint (body-frame, fixed per episode).
    self._setpoint_b = torch.zeros(N, 3, device=D)
    self._forcefield_centers_w = torch.zeros(N, 3, device=D)

    # Impedance parameters.
    self._stiffness = torch.zeros(N, 1, device=D)
    self._damping = torch.zeros(N, 1, device=D)

    # Ramp state.
    self._steps_since_resample = torch.zeros(N, dtype=torch.long, device=D)
    self._ramp_duration_steps = torch.ones(N, dtype=torch.long, device=D)

    # Fixed default-pose EE offset (optional).
    if cfg.default_ee_offset_b is not None:
      self._default_ee_offset_b = torch.tensor(
        cfg.default_ee_offset_b,
        dtype=torch.float32,
        device=D,
      )
    else:
      self._default_ee_offset_b = None

    # The ``command`` property returns target forces in body frame.
    self._command = torch.zeros(N, 3, device=D)

    self.metrics: dict[str, torch.Tensor] = {
      "force_error": torch.zeros(N, device=D),
    }

  # -- Properties ----------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    return self._command

  @property
  def target_forces_w(self) -> torch.Tensor:
    return self._target_forces_w

  @property
  def target_forces_b(self) -> torch.Tensor:
    return self._target_forces_b

  @property
  def forcefield_forces_w(self) -> torch.Tensor:
    return self._forcefield_forces_w

  @property
  def forcefield_forces_b(self) -> torch.Tensor:
    return self._forcefield_forces_b

  # -- CommandTerm interface -----------------------------------------------

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    elif isinstance(env_ids, slice):
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[
        env_ids
      ]
    elif not isinstance(env_ids, torch.Tensor):
      env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
    else:
      env_ids = env_ids.to(device=self.device, dtype=torch.long)

    if len(env_ids) == 0:
      return {}

    metrics = {}
    for metric_name, metric_value in self.metrics.items():
      metrics[metric_name] = torch.mean(metric_value[env_ids]).item()
      metric_value[env_ids] = 0.0
    self.command_counter[env_ids] = 0

    data = self._entity.data
    root_pos = data.root_link_pos_w[env_ids]
    root_quat = data.root_link_quat_w[env_ids]

    # Fix setpoint in yaw-rotated frame (xy translate + yaw rotate, fixed height).
    origin, yaw_quat = _yaw_frame_origin(root_pos, root_quat)
    if self._default_ee_offset_b is not None:
      self._setpoint_b[env_ids] = self._default_ee_offset_b.expand(
        len(env_ids),
        -1,
      )
    else:
      ee_pos = data.body_link_pos_w[env_ids, self._ee_body_id]
      self._setpoint_b[env_ids] = quat_apply_inverse(
        yaw_quat,
        ee_pos - origin,
      )
    self._forcefield_centers_w[env_ids] = origin + quat_apply(
      yaw_quat,
      self._setpoint_b[env_ids],
    )
    self._forcefield_forces_w[env_ids] = 0.0
    self._forcefield_forces_b[env_ids] = 0.0
    self._target_forces_w[env_ids] = 0.0
    self._target_forces_b[env_ids] = 0.0
    self._peak_target_forces_w[env_ids] = 0.0
    self._stiffness[env_ids] = 0.0
    self._damping[env_ids] = 0.0
    self._steps_since_resample[env_ids] = 0
    self._ramp_duration_steps[env_ids] = 1
    self._command[env_ids] = 0.0

    if self.cfg.startup_zero_force_duration_s > 0.0:
      self.time_left[env_ids] = self.cfg.startup_zero_force_duration_s
    else:
      self.time_left[env_ids] = self.time_left[env_ids].uniform_(
        *self.cfg.resampling_time_range
      )
      self._resample_command(env_ids)
      self.command_counter[env_ids] = 1

    return metrics

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
      return

    # Sample target forces in yaw-rotated frame.
    lo = torch.tensor(self.cfg.force_range[0], device=self.device)
    hi = torch.tensor(self.cfg.force_range[1], device=self.device)
    forces_yaw = torch.rand(n, 3, device=self.device) * (hi - lo) + lo

    # Zero-force masking.
    zero_mask = (
      torch.rand(n, device=self.device) < self.cfg.sample_zero_force_probability
    )
    forces_yaw[zero_mask] = 0.0

    # Sample impedance parameters before clipping the commanded force.
    s_lo, s_hi = self.cfg.stiffness_range
    self._stiffness[env_ids] = sample_uniform(s_lo, s_hi, (n, 1), self.device)
    if self.cfg.damping_proportional_to_stiffness > 0:
      self._damping[env_ids] = (
        self._stiffness[env_ids] * self.cfg.damping_proportional_to_stiffness
      )
    else:
      d_lo, d_hi = self.cfg.damping_range
      self._damping[env_ids] = sample_uniform(d_lo, d_hi, (n, 1), self.device)

    forces_yaw = clip_xy_force_to_reach(
      forces_yaw,
      self._stiffness[env_ids],
      self.cfg.max_xy_reach_m,
    )
    self._target_forces_b[env_ids] = forces_yaw

    # Convert to world frame using yaw-only rotation.
    root_quat = self._entity.data.root_link_quat_w[env_ids]
    _, yaw_quat = _yaw_frame_origin(
      self._entity.data.root_link_pos_w[env_ids], root_quat
    )
    self._peak_target_forces_w[env_ids] = quat_apply(yaw_quat, forces_yaw)

    # Reset ramp state.
    self._steps_since_resample[env_ids] = 0
    ramp_lo, ramp_hi = self.cfg.force_ramp_duration_range_s
    self._ramp_duration_steps[env_ids] = (
      (sample_uniform(ramp_lo, ramp_hi, (n,), self.device) / self._env.step_dt)
      .long()
      .clamp(min=1)
    )

  def _update_command(self) -> None:
    data = self._entity.data
    root_pos = data.root_link_pos_w
    root_quat = data.root_link_quat_w

    # Update world-frame setpoint from yaw frame (xy translate + yaw rotate).
    origin, yaw_quat = _yaw_frame_origin(root_pos, root_quat)
    self._forcefield_centers_w = origin + quat_apply(yaw_quat, self._setpoint_b)

    # Apply force ramp envelope.
    self._apply_force_ramp()

    # Keep yaw-frame target in sync with ramped world-frame target.
    yaw_quat_inv = quat_inv(yaw_quat)
    self._target_forces_b = quat_apply(yaw_quat_inv, self._target_forces_w)
    self._command = self._target_forces_b

    # NOTE: Impedance force computation moved to ForceFieldAction
    # (runs every substep for higher-fidelity force updates).

  def _update_metrics(self) -> None:
    err = torch.linalg.norm(self._target_forces_w - self._forcefield_forces_w, dim=-1)
    self.metrics["force_error"] = err

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    for idx in env_indices:
      center = self._forcefield_centers_w[idx].cpu().numpy()
      visualizer.add_sphere(
        center=center,
        radius=0.03,
        color=(0.2, 0.2, 0.8, 0.8),
        label=f"ff_setpoint_{idx}",
      )

  # -- Internal helpers ----------------------------------------------------

  def _compute_ramp_factor(self) -> torch.Tensor:
    """Compute per-env ramp factor in [0, 1]."""
    dt = self._env.step_dt
    hold_steps = int(self.cfg.force_hold_duration_s / dt)
    ramp = self._ramp_duration_steps.float().clamp(min=1.0)
    step = self._steps_since_resample.float()

    phase_end_hold = ramp + hold_steps
    phase_end_down = phase_end_hold + ramp

    # Ramp up: 0 → 1.
    factor = (step / ramp).clamp(0.0, 1.0)
    # Hold: 1.
    factor = torch.where(
      (step >= ramp) & (step < phase_end_hold),
      torch.ones_like(factor),
      factor,
    )
    # Ramp down: 1 → 0.
    down = 1.0 - ((step - phase_end_hold) / ramp).clamp(0.0, 1.0)
    factor = torch.where(
      (step >= phase_end_hold) & (step < phase_end_down),
      down,
      factor,
    )
    # After ramp down: 0.
    factor = torch.where(step >= phase_end_down, torch.zeros_like(factor), factor)
    return factor

  def _apply_force_ramp(self) -> None:
    """Apply ramp envelope to target forces and increment step counter."""
    factor = self._compute_ramp_factor()
    self._target_forces_w = self._peak_target_forces_w * factor.unsqueeze(-1)
    self._steps_since_resample += 1
