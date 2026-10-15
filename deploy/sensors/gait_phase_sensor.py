import math

import numpy as np

from deploy.sensors.base_sensor import BaseSensor


class GaitPhaseSensor(BaseSensor):
  """Gait phase sensor for deployment.

  Computes the sinusoidal gait phase signal used by WTW-style policies:
      gait_phase = sin(2 * pi * freq * t)

  where:
      - freq is mapped from the gait_frequency command [-1, 1] to
        [freq_range[0], freq_range[1]] (e.g. [1.0, 2.5] Hz)
      - t is the elapsed time since policy deployment start

  The phase is masked to 0 when the velocity command norm is near zero
  (robot is standing), matching training behavior.

  dim = 1
  """

  dim = 1

  def __init__(
    self,
    interface,
    scale=1.0,
    gait_freq_sensor=None,
    velocity_sensor=None,
    swing_height_sensor=None,
    freq_range=(1.0, 2.5),
    dt=0.02,
  ):
    super().__init__(interface, scale=scale)
    self.gait_freq_sensor = gait_freq_sensor
    self.velocity_sensor = velocity_sensor
    self.swing_height_sensor = swing_height_sensor
    self.freq_range = freq_range
    self.dt = dt
    self.step_count = 0

  def reset(self):
    self.step_count = 0

  def get_data(self):
    # Elapsed time
    t = self.step_count * self.dt
    self.step_count += 1

    # Get actual frequency from gait_frequency command
    # gait_freq_cmd is in [-1, 1], map to freq_range
    if self.gait_freq_sensor is not None:
      gait_freq_cmd = self.gait_freq_sensor.value  # raw value in [-1, 1]
    else:
      gait_freq_cmd = 0.0  # default mid-range

    freq = (gait_freq_cmd + 1.0) / 2.0 * (
      self.freq_range[1] - self.freq_range[0]
    ) + self.freq_range[0]

    # Sinusoidal gait phase
    phase = math.sin(2.0 * math.pi * freq * t)

    # Mask to 0 when velocity command is near zero (standing)
    if self.velocity_sensor is not None:
      vel_cmd = self.velocity_sensor.get_data()  # already scaled
      if np.linalg.norm(vel_cmd) < 1e-6:
        phase = 0.0

    # Print consolidated WTW command status
    swing_h = (
      self.swing_height_sensor.value if self.swing_height_sensor is not None else 0.0
    )
    print(
      f"\rWTW cmd: freq={gait_freq_cmd:+.1f} swing_h={swing_h:+.1f} phase={phase:+.2f}",
      end="",
    )

    return np.array([phase]) * self.scale
