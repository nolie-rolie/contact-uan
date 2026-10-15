import numpy as np

from deploy.sensors.base_sensor import BaseSensor


def map_normalized_velocity_commands_to_range(
  raw_commands,
  min_limits,
  max_limits,
) -> np.ndarray:
  """Map normalized joystick inputs in [-1, 1] onto deploy command ranges.

  For ranges that straddle zero, keep zero stick input mapped to zero command
  even when the positive and negative limits are asymmetric.
  """
  raw_commands = np.clip(np.asarray(raw_commands, dtype=np.float32), -1.0, 1.0)
  mapped = np.zeros(3, dtype=np.float32)

  for idx, raw in enumerate(raw_commands[:3]):
    lower = float(min_limits[idx])
    upper = float(max_limits[idx])
    if lower <= 0.0 <= upper:
      mapped[idx] = raw * (upper if raw >= 0.0 else abs(lower))
    else:
      mapped[idx] = 0.5 * (raw + 1.0) * (upper - lower) + lower

  return mapped


class JoystickVelocitySensor(BaseSensor):
  dim = 3

  def __init__(self, interface, scale=1.0):
    super().__init__(interface, scale=scale)

    self.velocity_cmd_min_limits = np.array(
      [
        self.interface.cfg["command_ranges"]["lin_vel_x"][0],
        self.interface.cfg["command_ranges"]["lin_vel_y"][0],
        self.interface.cfg["command_ranges"]["ang_vel_yaw"][0],
      ]
    )

    self.velocity_cmd_max_limits = np.array(
      [
        self.interface.cfg["command_ranges"]["lin_vel_x"][1],
        self.interface.cfg["command_ranges"]["lin_vel_y"][1],
        self.interface.cfg["command_ranges"]["ang_vel_yaw"][1],
      ]
    )

  def get_data(self):
    base_velocity_commands = map_normalized_velocity_commands_to_range(
      self.interface.get_joystick_commands(),
      self.velocity_cmd_min_limits,
      self.velocity_cmd_max_limits,
    )

    base_velocity_commands_scaled = np.array(
      [
        base_velocity_commands[0] * self.interface.cfg["obs_scales"]["lin_vel"],
        base_velocity_commands[1] * self.interface.cfg["obs_scales"]["lin_vel"],
        base_velocity_commands[2] * self.interface.cfg["obs_scales"]["ang_vel"],
      ],
      dtype=np.float32,
    )

    return base_velocity_commands_scaled * self.scale
