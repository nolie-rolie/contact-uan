from deploy.sensors.base_sensor import BaseSensor


class LastActionSensor(BaseSensor):
  dim = "nj"

  def __init__(self, interface, scale=1.0):
    super().__init__(interface, scale=1.0)

  def get_data(self, last_action):
    # The controller supplies the previous policy action.
    return last_action
