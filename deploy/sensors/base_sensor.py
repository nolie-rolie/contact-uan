class BaseSensor:
  def __init__(self, interface, scale=1.0):
    self.interface = interface
    self.scale = scale

  def get_data(self):
    raise NotImplementedError
