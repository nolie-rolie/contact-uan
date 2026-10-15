import numpy as np

from deploy.sensors.base_sensor import BaseSensor
from deploy.sensors.motion_clip_player import MotionClipPlayer


class MotionAnchorAngVelSensor(BaseSensor):
  """Reference torso-frame angular velocity."""

  dim = 3

  def __init__(self, interface, player: MotionClipPlayer):
    super().__init__(interface, scale=1.0)
    self._player = player

  def get_data(self) -> np.ndarray:
    return self._player.anchor_ang_vel_b
