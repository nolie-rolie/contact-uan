import time

import numpy as np

from .base_interface import BaseInterface


class DummyInterface(BaseInterface):
  def __init__(self, robot_configuration, cfg, task_name):
    self.robot_configuration = robot_configuration
    self.cfg = cfg

    self.init_time = time.time()

    self.key_state = [
      ["R1", 0],
      ["L1", 0],
      ["start", 0],
      ["select", 0],
      ["R2", 0],
      ["L2", 0],
      ["F1", 0],
      ["F2", 0],
      ["A", 0],
      ["B", 0],
      ["X", 0],
      ["Y", 0],
      ["up", 0],
      ["right", 0],
      ["down", 0],
      ["left", 0],
    ]

  ######################
  # Main API Functions #
  ######################

  def initialize(self, render=False):
    pass

  def get_joint_pos(self, joint_order):
    if joint_order == "isaaclab":
      raise ValueError(
        "This interface exposes hardware-order joints; remap them in the controller."
      )
    elif joint_order == "hw":
      return np.zeros(self.robot_configuration.num_joints_hw)
    else:
      raise ValueError("Invalid joint order")

  def get_joint_vel(self, joint_order):
    if joint_order == "isaaclab":
      raise ValueError(
        "This interface exposes hardware-order joints; remap them in the controller."
      )
    elif joint_order == "hw":
      return np.zeros(self.robot_configuration.num_joints_hw)
    else:
      raise ValueError("Invalid joint order")

  def get_joint_acc(self, joint_order):
    if joint_order == "isaaclab":
      raise ValueError(
        "This interface exposes hardware-order joints; remap them in the controller."
      )
    elif joint_order == "hw":
      return np.zeros(self.robot_configuration.num_joints_hw)
    else:
      raise ValueError("Invalid joint order")

  def get_joint_torque(self, joint_order):
    if joint_order == "isaaclab":
      raise ValueError(
        "This interface exposes hardware-order joints; remap them in the controller."
      )
    elif joint_order == "hw":
      return np.zeros(self.robot_configuration.num_joints_hw)
    else:
      raise ValueError("Invalid joint order")

  def get_gravity_vector(self):
    # get the gravity vector in the base frame
    return np.array([0, 0, -1])

  def get_buttons(self):
    # Emulate L1 press if it's been more than 1 second
    if time.time() - self.init_time > 1 and self.key_state[1][1] == 0:
      self.key_state[1][1] = 1
      print("[DummyInterface] Emulating L1 press!")

    # Emulate X press if it's been more than 7 seconds
    if time.time() - self.init_time > 7 and self.key_state[10][1] == 0:
      self.key_state[10][1] = 1
      print("[DummyInterface] Emulating X press!")

    return self.key_state

  def apply_command(self, kp, kd, q, dq, tau, torque_mode=False):
    # print(f"Applying command")
    # print(f"kp: {kp}")
    # print(f"kd: {kd}")
    # print(f"q: {q}")
    # print(f"dq: {dq}")
    # print(f"tau: {tau}")
    pass

  def get_obs_timestamp(self):
    return 0

  def get_pub_timestamp(self):
    return 0

  def get_write_timestamp(self):
    return 0

  def repeat_last_command(self):
    pass

  def get_joystick_commands(self):
    return np.zeros(3)

  def get_height_commands(self):
    return 0.95

  def get_frequency_command(self):
    return 1.2

  def check_comms_safety(self):
    return True
