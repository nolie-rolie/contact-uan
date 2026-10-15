import math
import select
import threading
import time
from typing import Any

import lcm
import numpy as np
from lcm_types.arm_action_lcmt import arm_action_lcmt
from lcm_types.body_control_data_lcmt import body_control_data_lcmt
from lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt
from lcm_types.rc_command_v2_lcmt import rc_command_v2_lcmt
from lcm_types.state_estimator_lcmt import state_estimator_lcmt

from deploy.interfaces.base_interface import BaseInterface
from deploy.utils.lcm_bridge_config import validate_lcm_pd_gains


def _apply_stick_deadzone(value: float, deadzone: float = 0.1) -> float:
  value = float(np.clip(value, -1.0, 1.0))
  if abs(value) <= deadzone:
    return 0.0
  return np.sign(value) * ((abs(value) - deadzone) / (1.0 - deadzone))


class G1LCMInterface(BaseInterface):
  def __init__(self, robot_configuration, cfg, task_name):
    self.robot_configuration = robot_configuration
    self.cfg = cfg
    self.task_name = task_name or ""
    self.is_force_control = "ForceControl" in self.task_name
    # Reject unsupported gains before creating transport or publishing any target.
    validate_lcm_pd_gains(
      robot_configuration.joint_stiffnesses("hw"),
      robot_configuration.joint_dampings("hw"),
      robot_configuration.joint_names("hw"),
    )
    self.lc = lcm.LCM()

    self.height_commands = 0.8
    self.frequency_commands = 1.1
    self.pitch_commands = 0.0
    self.velocity_commands = np.zeros((3,))
    self.left_stick = np.zeros(2)
    self.right_stick = np.zeros(2)

    # WTW gait parameters (adjustable via d-pad)
    self.gait_frequency_cmd = 0.0  # in [-1, 1]
    self.swing_height_cmd = 0.0  # in [-1, 1]
    self._prev_dpad = {"up": 0, "down": 0, "left": 0, "right": 0}
    self._dpad_last_trigger = {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}
    self.DPAD_COOLDOWN = 0.3  # seconds
    self.force_step_n = 1.0
    self.force_command_max_abs = np.asarray(
      self.cfg.get("force_command_max_abs", (30.0, 30.0, 30.0)),
      dtype=float,
    )
    self.manual_force_xyz_n = np.zeros(3, dtype=float)
    self._prev_force_zero_button = 0

    self.init_time = time.time()
    self.imu_time = time.time()
    self.counter = 0
    self.received_first_bodydata = False

  ######################
  # Main API Functions #
  ######################

  def initialize(self, render=False):
    self.initialize_state_variables()
    self.initialize_lcm()
    self.damping_mode_active = False
    self.zero_command()
    self.spin()

  def apply_command(self, kp, kd, q, dq, tau, torque_mode=False):
    msg: Any = pd_tau_targets_lcmt()
    msg.q_des = q
    msg.tau_ff = tau

    msg_arm = arm_action_lcmt()
    msg_arm.act = q[15:]

    # Continuously check if R2 is pressed
    if (
      self.key_state[4][1] == 0 or self.damping_mode_active
    ):  # If R2 was pressed or in damping mode
      pass
    else:
      print("R2 button pressed. Switching to damping mode.")
      self.damping_mode()
      return
    self.lc.publish("pd_plustau_targets", msg.encode())
    self.lc.publish("arm_action", msg_arm.encode())

  def repeat_last_command(self):
    # Continuously check if R2 is pressed
    if (
      self.key_state[4][1] == 0 or self.damping_mode_active
    ):  # If R2 was pressed or in damping mode
      pass
    else:
      print("R2 button pressed. Switching to damping mode.")
      self.damping_mode()
      return

  def get_joint_pos(self, joint_order):
    if joint_order == "hw":
      return self.joint_pos
    elif joint_order == "isaaclab":
      raise ValueError(
        "This interface exposes hardware-order joints; remap them in the controller."
      )
    else:
      raise ValueError("Invalid joint order argument")

  def get_joint_vel(self, joint_order):
    if joint_order == "hw":
      return self.joint_vel
    elif joint_order == "isaaclab":
      raise ValueError(
        "This interface exposes hardware-order joints; remap them in the controller."
      )
    else:
      raise ValueError("Invalid joint order argument")

  def get_joint_acc(self, joint_order):
    if joint_order == "hw":
      return np.zeros((29,))
    elif joint_order == "isaaclab":
      raise ValueError(
        "This interface exposes hardware-order joints; remap them in the controller."
      )
    else:
      raise ValueError("Invalid joint order argument")

  def get_joint_torque(self, joint_order):
    if joint_order == "hw":
      return np.zeros(
        29,
      )
    elif joint_order == "isaaclab":
      raise ValueError(
        "This interface exposes hardware-order joints; remap them in the controller."
      )
    else:
      raise ValueError("Invalid joint order argument")

  def get_joystick_commands(self):
    return self.velocity_commands

  def get_force_command_input(self):
    if self.is_force_control:
      with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.divide(
          self.manual_force_xyz_n,
          self.force_command_max_abs,
          out=np.zeros(3, dtype=float),
          where=self.force_command_max_abs > 0.0,
        )
      return np.clip(normalized, -1.0, 1.0)
    return np.array(
      [
        _apply_stick_deadzone(self.right_stick[1]),
        _apply_stick_deadzone(-self.right_stick[0]),
        _apply_stick_deadzone(self.left_stick[1]),
      ]
    )

  def get_force_command_newtons(self):
    if self.is_force_control:
      return self.manual_force_xyz_n.copy()
    return self.get_force_command_input() * self.force_command_max_abs

  def _print_manual_force_command(self):
    print(
      "[ForceCmd] "
      f"desired=({self.manual_force_xyz_n[0]:+.1f}, "
      f"{self.manual_force_xyz_n[1]:+.1f}, "
      f"{self.manual_force_xyz_n[2]:+.1f}) N"
    )

  def _update_force_command_from_buttons(
    self, curr_dpad: dict[str, int], zero_button: int
  ):
    now = time.time()
    changed = False

    if (
      curr_dpad["up"]
      and not self._prev_dpad["up"]
      and now - self._dpad_last_trigger["up"] > self.DPAD_COOLDOWN
    ):
      self.manual_force_xyz_n[0] = min(
        self.force_command_max_abs[0],
        self.manual_force_xyz_n[0] + self.force_step_n,
      )
      self._dpad_last_trigger["up"] = now
      changed = True
    if (
      curr_dpad["down"]
      and not self._prev_dpad["down"]
      and now - self._dpad_last_trigger["down"] > self.DPAD_COOLDOWN
    ):
      self.manual_force_xyz_n[0] = max(
        -self.force_command_max_abs[0],
        self.manual_force_xyz_n[0] - self.force_step_n,
      )
      self._dpad_last_trigger["down"] = now
      changed = True
    if (
      curr_dpad["right"]
      and not self._prev_dpad["right"]
      and now - self._dpad_last_trigger["right"] > self.DPAD_COOLDOWN
    ):
      self.manual_force_xyz_n[2] = min(
        self.force_command_max_abs[2],
        self.manual_force_xyz_n[2] + self.force_step_n,
      )
      self._dpad_last_trigger["right"] = now
      changed = True
    if (
      curr_dpad["left"]
      and not self._prev_dpad["left"]
      and now - self._dpad_last_trigger["left"] > self.DPAD_COOLDOWN
    ):
      self.manual_force_xyz_n[2] = max(
        -self.force_command_max_abs[2],
        self.manual_force_xyz_n[2] - self.force_step_n,
      )
      self._dpad_last_trigger["left"] = now
      changed = True
    if zero_button and not self._prev_force_zero_button:
      self.manual_force_xyz_n[:] = 0.0
      changed = True

    self._prev_dpad = curr_dpad
    self._prev_force_zero_button = int(bool(zero_button))
    if changed:
      self._print_manual_force_command()

  def get_gait_params(self):
    """Return [gait_frequency_cmd, swing_height_cmd] both in [-1, 1]."""
    return [self.gait_frequency_cmd, self.swing_height_cmd]

  def get_height_commands(self):
    return self.height_commands

  def get_pitch_commands(self):
    return self.pitch_commands

  def get_frequency_command(self):
    return self.frequency_commands

  def get_gravity_vector(self):
    # get the gravity vector in the base frame
    return self.gravity_vec

  def get_root_ang_vel(self):
    return self.omegaBody

  def get_root_lin_vel(self):
    return self.vBody

  def get_foot_contact_states(self):
    return self.contact_estimate

  def get_root_height(self):
    return self.p[2]

  def get_obs_timestamp(self):
    return 0

  def get_pub_timestamp(self):
    return 0

  def get_write_timestamp(self):
    return 0

  # ############
  # # Handlers #
  # ############
  def _bodydata_cb(self, channel, data):
    if not self.received_first_bodydata:
      self.received_first_bodydata = True
      print(f"First body data: {time.time() - self.init_time}")

    msg = body_control_data_lcmt.decode(data)
    self.joint_pos = np.array(msg.q)
    self.joint_vel = np.array(msg.qd)

  def _imu_cb(self, channel, data):
    msg = state_estimator_lcmt.decode(data)
    self.rpy = msg.rpy
    R = self.get_rotation_matrix_from_rpy(self.rpy)
    self.gravity_vec = np.dot(
      R.T, np.array([0, 0, -1])
    )  # Gravity vector in the world frame
    self.omegaBody = np.array(msg.omegaBody)
    self.vBody = np.array(msg.vBody)
    self.p = np.array(msg.p)
    self.contact_estimate = np.array(msg.contact_estimate)
    self.counter += 1
    if self.counter % 1000 == 0:
      print("frequency of imu state = ", 1000 / (time.time() - self.imu_time))
      self.counter = 0
      self.imu_time = time.time()

  def _rc_command_cb(self, channel, data):
    msg = rc_command_v2_lcmt.decode(data)
    self.left_stick = np.array(msg.left_stick, dtype=float)
    self.right_stick = np.array(msg.right_stick, dtype=float)
    self.velocity_commands = np.array([msg.left_stick[1], 0, -1 * msg.right_stick[0]])
    self.key_state = [
      ["R1", msg.R1],
      ["L1", msg.L1],
      ["start", msg.start],
      ["select", msg.select],
      ["R2", msg.R2],
      ["L2", msg.L2],
      ["F1", msg.F1],
      ["F2", msg.F2],
      ["A", msg.A],
      ["B", msg.B],
      ["X", msg.X],
      ["Y", msg.Y],
      ["up", msg.up],
      ["right", msg.right],
      ["down", msg.down],
      ["left", msg.left],
    ]

    self.height_commands = min(max(0.75 + 0.35 * msg.left_stick[1], 0.4), 0.9)
    self.pitch_commands = 0.0  # min(max(0.0 + 2.0 * msg.right_stick[0], -1.5), 2.0)
    # self.frequency_commands = 1.2 + 0.3 * msg.left_stick[1]

    curr_dpad = {"up": msg.up, "down": msg.down, "left": msg.left, "right": msg.right}
    if self.is_force_control:
      self._update_force_command_from_buttons(curr_dpad, msg.A)
    else:
      now = time.time()
      # D-pad gait parameter adjustment (rising-edge detection + debounce)
      # up/down: gait frequency, right/left: swing height
      if (
        curr_dpad["up"]
        and not self._prev_dpad["up"]
        and now - self._dpad_last_trigger["up"] > self.DPAD_COOLDOWN
      ):
        self.gait_frequency_cmd = min(1.0, self.gait_frequency_cmd + 0.2)
        self._dpad_last_trigger["up"] = now
        print(f"Gait freq cmd: {self.gait_frequency_cmd:+.1f}")
      if (
        curr_dpad["down"]
        and not self._prev_dpad["down"]
        and now - self._dpad_last_trigger["down"] > self.DPAD_COOLDOWN
      ):
        self.gait_frequency_cmd = max(-1.0, self.gait_frequency_cmd - 0.2)
        self._dpad_last_trigger["down"] = now
        print(f"Gait freq cmd: {self.gait_frequency_cmd:+.1f}")
      if (
        curr_dpad["right"]
        and not self._prev_dpad["right"]
        and now - self._dpad_last_trigger["right"] > self.DPAD_COOLDOWN
      ):
        self.swing_height_cmd = min(1.0, self.swing_height_cmd + 0.2)
        self._dpad_last_trigger["right"] = now
        print(f"Swing height cmd: {self.swing_height_cmd:+.1f}")
      if (
        curr_dpad["left"]
        and not self._prev_dpad["left"]
        and now - self._dpad_last_trigger["left"] > self.DPAD_COOLDOWN
      ):
        self.swing_height_cmd = max(-1.0, self.swing_height_cmd - 0.2)
        self._dpad_last_trigger["left"] = now
        print(f"Swing height cmd: {self.swing_height_cmd:+.1f}")
      self._prev_dpad = curr_dpad
      self._prev_force_zero_button = int(bool(msg.A))

  def initialize_state_variables(self):
    self.joint_pos = None
    self.joint_vel = np.zeros(self.robot_configuration.num_joints)
    self.joint_torque = np.zeros(self.robot_configuration.num_joints)
    self.rpy = None
    self.gravity_vec = None
    self.omegaBody = None
    self.vBody = np.zeros(3)
    self.p = np.array([0.0, 0.0, 0.75])
    self.contact_estimate = np.zeros(4)
    self.left_stick = np.zeros(2)
    self.right_stick = np.zeros(2)
    self.manual_force_xyz_n = np.zeros(3, dtype=float)
    self._prev_dpad = {"up": 0, "down": 0, "left": 0, "right": 0}
    self._prev_force_zero_button = 0
    self._dpad_last_trigger = {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}
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

  def get_buttons(self):
    return self.key_state

  def zero_command(self):
    msg: Any = pd_tau_targets_lcmt()
    msg.q_des = np.zeros((29,))
    msg.tau_ff = np.zeros((29,))
    self.lc.publish("pd_plustau_targets", msg.encode())

  def initialize_lcm(self):
    self.imu_subscription = self.lc.subscribe("state_estimator_data", self._imu_cb)
    self.bodydata_state_subscription = self.lc.subscribe(
      "body_control_data", self._bodydata_cb
    )
    self.rc_command_subscription = self.lc.subscribe(
      "rc_command_v2", self._rc_command_cb
    )
    print("initialized lcm")

  def damping_mode(self):
    """Switch all joints to damping mode."""
    print("Switching to damping mode.")
    self.damping_mode_active = True
    msg: Any = pd_tau_targets_lcmt()
    msg.q_des = np.zeros((29,))
    msg.tau_ff = np.zeros((29,))
    self.lc.publish("pd_plustau_targets", msg.encode())

  def get_rotation_matrix_from_rpy(self, rpy):
    """
    Get rotation matrix from the given quaternion.
    Args:
        q (np.array[float[4]]): quaternion [w,x,y,z]
    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    r, p, y = rpy
    R_x = np.array(
      [[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]]
    )

    R_y = np.array(
      [[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]]
    )

    R_z = np.array(
      [[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]]
    )

    rot = np.dot(R_z, np.dot(R_y, R_x))
    return rot

  ######################
  # Run lcm functions #
  ######################

  def poll(self, cb=None):
    try:
      while True:
        timeout = 0.01
        rfds, wfds, efds = select.select([self.lc.fileno()], [], [], timeout)
        # nrfds, nwfds, nefds = select.select([self.new_lcm.fileno()], [], [], timeout)
        if rfds:
          self.lc.handle()
          # print(f'Freq {1. / (time.time() - t)} Hz'); t = time.time()
        else:
          continue
    except KeyboardInterrupt:
      pass

  def spin(self):
    self.run_thread = threading.Thread(target=self.poll, daemon=False)
    self.run_thread.start()

  def shutdown(self):
    print("exiting lcm interface")
    self.run_thread.join()
    exit(0)
