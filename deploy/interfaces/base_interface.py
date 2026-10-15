class BaseInterface:
  def __init__(self, robot_configuration, dt):
    self.robot_configuration = robot_configuration

  ######################
  # Main API Functions #
  ######################

  def initialize(self, render=False):
    raise NotImplementedError()

  def get_joint_pos(self, joint_order):
    raise NotImplementedError()

  def get_joint_vel(self, joint_order):
    raise NotImplementedError()

  def get_joint_acc(self, joint_order):
    raise NotImplementedError()

  def get_joint_torque(self, joint_order):
    raise NotImplementedError()

  def get_gravity_vector(self):
    # get the gravity vector in the base frame
    raise NotImplementedError()

  def get_buttons(self):
    raise NotImplementedError()

  def get_force_command_input(self):
    raise NotImplementedError()

  def apply_command(self, kp, kd, q, dq, tau):
    raise NotImplementedError()
