import numpy as np

from deploy.sensors.base_sensor import BaseSensor
from deploy.sensors.motion_clip_player import MotionClipPlayer


class MotionJointPosSensor(BaseSensor):
  """Reference joint positions from the motion clip."""

  dim = 29

  def __init__(self, interface, player: MotionClipPlayer):
    super().__init__(interface, scale=1.0)
    self._player = player

  def get_data(self) -> np.ndarray:
    return self._player.current_joint_pos
