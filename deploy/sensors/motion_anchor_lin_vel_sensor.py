import numpy as np

from deploy.sensors.base_sensor import BaseSensor
from deploy.sensors.motion_clip_player import MotionClipPlayer


class MotionAnchorLinVelSensor(BaseSensor):
  """Reference torso-frame linear velocity."""

  dim = 3

  def __init__(self, interface, player: MotionClipPlayer):
    super().__init__(interface, scale=1.0)
    self._player = player

  def get_data(self) -> np.ndarray:
    return self._player.anchor_lin_vel_b
