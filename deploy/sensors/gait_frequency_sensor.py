import numpy as np

from deploy.sensors.base_sensor import BaseSensor


class GaitFrequencyCommandSensor(BaseSensor):
  """Gait frequency command sensor for deployment.

  Returns a scalar gait frequency command in [-1, 1].
  Reads from the interface's gait params (keyboard: [ / ] keys).
  Falls back to a fixed default_value if the interface doesn't support it.

  dim = 1
  """

  dim = 1

  def __init__(self, interface, scale=1.0, default_value=0.0):
    super().__init__(interface, scale=scale)
    self.default_value = default_value

  @property
  def value(self):
    """Current gait frequency command in [-1, 1]."""
    if hasattr(self.interface, "get_gait_params"):
      return self.interface.get_gait_params()[0]
    return self.default_value

  def get_data(self):
    return np.array([self.value]) * self.scale
