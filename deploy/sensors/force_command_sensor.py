"""Force command sensor for force-control policies.

Reads a 3D end-effector force command (Fx, Fy, Fz) and applies a scale
factor matching the training normalisation (typically 1/max_force).

The force can be set programmatically via ``set_force_command()`` (for
scripted experiments) or driven by interface input:
  - real joystick force deploy: d-pad steps Fx/Fz in 1 N increments, A zeros
  - MuJoCo keyboard: arrows -> Fx/Fz, space -> zero
"""

import numpy as np

from deploy.sensors.base_sensor import BaseSensor
from deploy.utils.force_reach_utils import (
  clip_xy_force_to_reach,
)


class ForceCommandSensor(BaseSensor):
  """3-DOF end-effector force command sensor.

  Args:
      interface: Robot interface providing joystick / button state.
      scale: Normalisation factor applied to the raw force command
          before returning (e.g. ``1/30.0`` for a ±30 N range).
      max_force_xyz: Maximum force magnitude for each axis (N).
      max_xy_reach_m: Optional per-axis xy reach limit in meters.
      clip_stiffness: Stiffness used to convert reach limit into a
          maximum xy force. When unset, reach clipping is disabled.
  """

  dim = 3

  def __init__(
    self,
    interface,
    scale: float = 1.0 / 30.0,
    max_force_xyz: tuple[float, float, float] = (30.0, 30.0, 30.0),
    max_xy_reach_m: float | None = None,
    clip_stiffness: float | None = None,
  ):
    super().__init__(interface, scale)
    self.max_force_xyz = np.asarray(max_force_xyz, dtype=float)
    self.max_xy_reach_m = None if max_xy_reach_m is None else float(max_xy_reach_m)
    self._clip_stiffness = None if clip_stiffness is None else float(clip_stiffness)
    self._requested_force_xyz = np.zeros(3)
    self._force_xyz = np.zeros(3)
    self._programmatic = False

  # ------------------------------------------------------------------
  # Programmatic control
  # ------------------------------------------------------------------

  def set_force_command(self, force_xyz: np.ndarray) -> None:
    """Set force command directly (bypasses joystick)."""
    self._requested_force_xyz = np.asarray(force_xyz, dtype=float)
    self._programmatic = True

  def clear_programmatic(self) -> None:
    """Revert to joystick-driven force command."""
    self._programmatic = False

  def set_clip_stiffness(self, stiffness: float | None) -> None:
    """Update the stiffness used for reach-based XY clipping."""
    self._clip_stiffness = None if stiffness is None else float(stiffness)

  # ------------------------------------------------------------------
  # BaseSensor API
  # ------------------------------------------------------------------

  def get_data(self) -> np.ndarray:
    """Return normalised 3D force command [Fx, Fy, Fz]."""
    self._force_xyz = self._resolve_force_xyz()
    return self._force_xyz * self.scale

  def get_raw_force(self) -> np.ndarray:
    """Return the un-normalised force command in Newtons."""
    self._force_xyz = self._resolve_force_xyz()
    return self._force_xyz.copy()

  def _resolve_force_xyz(self) -> np.ndarray:
    """Return the clipped force command in Newtons."""
    if self._programmatic:
      requested_force = self._requested_force_xyz
    else:
      requested_force = self._read_from_joystick()
    return clip_xy_force_to_reach(
      requested_force,
      stiffness=self._clip_stiffness,
      max_xy_reach_m=self.max_xy_reach_m,
    )

  # ------------------------------------------------------------------
  # Joystick reading
  # ------------------------------------------------------------------

  def _read_from_joystick(self) -> np.ndarray:
    """Map normalized interface input to a 3D force command."""
    try:
      if hasattr(self.interface, "get_force_command_newtons"):
        force_newtons = np.asarray(
          self.interface.get_force_command_newtons(),
          dtype=float,
        )
        return np.clip(force_newtons, -self.max_force_xyz, self.max_force_xyz)
      if hasattr(self.interface, "get_force_command_input"):
        force_input = np.asarray(
          self.interface.get_force_command_input(),
          dtype=float,
        )
      else:
        vel = self.interface.get_joystick_commands()
        buttons = self.interface.get_buttons()
        r1 = buttons[0][1]
        l1 = buttons[1][1]
        force_input = np.array([vel[0], vel[1], r1 - l1], dtype=float)
    except Exception:
      return np.zeros(3)

    force_input = np.clip(force_input, -1.0, 1.0)
    return force_input * self.max_force_xyz
