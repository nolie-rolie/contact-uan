from deploy.sensors.base_sensor import BaseSensor


class JointPosSensor(BaseSensor):
  dim = "nj"

  def __init__(self, interface, scale=1.0, joint_order="isaaclab"):
    super().__init__(interface, scale)
    self.joint_order = joint_order

  def get_data(self):
    joint_pos_hw = self.interface.get_joint_pos("hw")
    joint_pos_policy = self.interface.robot_configuration.remap_joint_array(
      joint_pos_hw, from_order="hw", to_order=self.joint_order
    )
    joint_pos_policy = (
      joint_pos_policy
      - self.interface.robot_configuration.joint_default_positions(
        joint_order=self.joint_order
      )
    )

    return joint_pos_policy * self.scale
