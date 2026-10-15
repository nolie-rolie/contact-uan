from deploy.sensors.base_sensor import BaseSensor


class RootAngVelSensor(BaseSensor):
  dim = 3

  def __init__(self, interface, scale=1.0):
    super().__init__(interface, scale)

  def get_data(self):
    return self.interface.get_root_ang_vel() * self.scale
