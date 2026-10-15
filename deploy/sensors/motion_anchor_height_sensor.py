import numpy as np

from deploy.sensors.base_sensor import BaseSensor
from deploy.sensors.motion_clip_player import MotionClipPlayer


class MotionAnchorHeightSensor(BaseSensor):
  """Reference torso height relative to the motion ground origin."""

  dim = 1

  def __init__(self, interface, player: MotionClipPlayer):
    super().__init__(interface, scale=1.0)
    self._player = player

  def get_data(self) -> np.ndarray:
    return np.array([self._player.anchor_height], dtype=np.float32)
