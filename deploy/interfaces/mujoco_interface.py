import re
import time
from typing import Any, cast

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from deploy.interfaces.base_interface import BaseInterface
from deploy.utils.math_utils import (
  get_rotation_matrix_from_rpy,
  get_rpy_from_quaternion,
)


class MujocoInterface(BaseInterface):
  def __init__(self, robot_configuration, cfg, headless=False, task_name=None):
    self.robot_configuration = robot_configuration
    self.cfg = cfg

    self.init_time = time.time()
    self.step_counter = 0

    self.task_name = task_name

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

    self.joint_idxs = [i for i in range(19)]
    self.last_action = None
    self.base_lock_pos = None

    self.max_foot_contact_load_n = 0.0

    self.headless = headless

  def initialize(self, render=False, keyframe_name="home"):
    import mujoco
    import mujoco.viewer

    self.model = mujoco.MjModel.from_xml_path(
      self.robot_configuration._mujoco_xml_paths["default"]
    )
    self.jpos_idx = 7
    self.jvel_idx = 6

    self.data = mujoco.MjData(self.model)

    # Physics engine parameters
    self.model.opt.timestep = (
      0.002  # Simulation timestep (smaller = more accurate but slower)
    )
    self.model.opt.integrator = 1  # 0: semi-implicit Euler, 1: RK4
    self.model.opt.iterations = 10  # Maximum number of constraint solver iterations
    self.model.opt.solver = 0  # 0: PGS, 1: CG, 2: Newton
    self.model.opt.tolerance = 1e-10  # Solver convergence tolerance

    self.disable_self_collision()

    # Initialize state from keyframe if specified
    if keyframe_name is not None:
      self.load_keyframe(keyframe_name)

    # Set initial base pose from training config (overrides keyframe base)
    init_base_pos = self.cfg.get("control", {}).get("init_base_pos", None)
    init_base_rot = self.cfg.get("control", {}).get("init_base_rot", None)
    if init_base_pos is not None:
      self.data.qpos[0:3] = init_base_pos
    if init_base_rot is not None:
      self.data.qpos[3:7] = init_base_rot
    # Set initial joint positions from training config defaults
    default_joint_pos = self.cfg.get("control", {}).get("default_joint_positions", None)
    if default_joint_pos is not None:
      joint_names = [
        mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(self.model.njnt)
      ]
      for joint_pattern, value in default_joint_pos.items():
        for joint_name in joint_names:
          if joint_name is None:
            continue
          if re.match(joint_pattern, joint_name):
            jnt_id = mujoco.mj_name2id(
              self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if jnt_id >= 0:
              qpos_addr = self.model.jnt_qposadr[jnt_id]
              self.data.qpos[qpos_addr] = value
    mujoco.mj_forward(self.model, self.data)

    from deploy.joysticks.mujoco_joystick import MujocoJoystick

    self.joystick = MujocoJoystick(task_name=self.task_name)
    if not self.headless:
      self.viewer = mujoco.viewer.launch_passive(
        self.model, self.data, key_callback=self.joystick.key_callback
      )
      cast(Any, self.viewer)._hide_overlay = True
      cast(Any, self.viewer)._render_every_frame = True
    else:
      self.viewer = None

    # Initialize state values
    self.joint_pos = np.array(self.data.qpos[self.jpos_idx : self.jpos_idx + 19])
    self.joint_vel = np.array(self.data.qvel[self.jvel_idx : self.jvel_idx + 19])
    self.gravity_vector = self._calculate_gravity_vector()
    self.root_ang_vel = np.array(self.data.qvel[3:6])
    self.max_foot_contact_load_n = 0.0

    self.left_foot_geom_ids = {
      geom_id
      for geom_id in (
        mujoco.mj_name2id(
          self.model,
          mujoco.mjtObj.mjOBJ_GEOM,
          f"left_ankle_roll_{idx}",
        )
        for idx in [1, 2, 3, 4]
      )
      if geom_id >= 0
    }
    self.right_foot_geom_ids = {
      geom_id
      for geom_id in (
        mujoco.mj_name2id(
          self.model,
          mujoco.mjtObj.mjOBJ_GEOM,
          f"right_ankle_roll_{idx}",
        )
        for idx in [1, 2, 3, 4]
      )
      if geom_id >= 0
    }

  def disable_self_collision(self):
    # Disable self-collision
    print("[MujocoInterface] Disabling self-collision for robot geoms")
    # Known environment geom names
    env_geom_names = ["floor", "wall", "ground"]
    env_geom_ids = []

    for name in env_geom_names:
      geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
      if geom_id >= 0:
        env_geom_ids.append(geom_id)

    # All geoms except environment
    robot_geom_ids = [i for i in range(self.model.ngeom) if i not in env_geom_ids]

    for geom_id in robot_geom_ids:
      self.model.geom_contype[geom_id] = 0  # Robot collision group
      self.model.geom_conaffinity[geom_id] = 2  # Only collides with group 1 (ground)

  def get_joint_pos(self, joint_order):
    self.joint_pos = np.array(
      self.data.qpos[
        self.jpos_idx : self.jpos_idx + self.robot_configuration.num_joints
      ]
    )
    if joint_order == "isaaclab":
      raise ValueError(
        "This interface exposes hardware-order joints; remap them in the controller."
      )
    elif joint_order == "hw":
      hw_joint_pos = self.robot_configuration.remap_joint_array(
        self.joint_pos, from_order="mujoco", to_order="hw"
      )
      return hw_joint_pos
    else:
      raise ValueError("Invalid joint order")

  def get_joint_vel(self, joint_order):
    self.joint_vel = np.array(
      self.data.qvel[
        self.jvel_idx : self.jvel_idx + self.robot_configuration.num_joints
      ]
    )
    if joint_order == "isaaclab":
      raise ValueError(
        "This interface exposes hardware-order joints; remap them in the controller."
      )
    elif joint_order == "hw":
      hw_joint_vel = self.robot_configuration.remap_joint_array(
        self.joint_vel, from_order="mujoco", to_order="hw"
      )
      return hw_joint_vel
    else:
      raise ValueError("Invalid joint order")

  def get_joint_acc(self, joint_order):
    self.joint_acc = np.array(
      self.data.qacc[
        self.jvel_idx : self.jvel_idx + self.robot_configuration.num_joints
      ]
    )
    if joint_order == "isaaclab":
      raise ValueError(
        "This interface exposes hardware-order joints; remap them in the controller."
      )
    elif joint_order == "hw":
      hw_joint_acc = self.robot_configuration.remap_joint_array(
        self.joint_acc, from_order="mujoco", to_order="hw"
      )
      return hw_joint_acc
    else:
      raise ValueError("Invalid joint order")

  def get_joint_torque(self, joint_order):
    self.joint_torque = np.array(self.data.ctrl[: self.robot_configuration.num_joints])
    if joint_order == "isaaclab":
      raise ValueError(
        "This interface exposes hardware-order joints; remap them in the controller."
      )
    elif joint_order == "hw":
      hw_joint_torque = self.robot_configuration.remap_joint_array(
        self.joint_torque, from_order="mujoco", to_order="hw"
      )
      return hw_joint_torque
    else:
      raise ValueError("Invalid joint order")

  def get_gravity_vector(self):
    return self.gravity_vector

  def get_root_ang_vel(self):
    return self.root_ang_vel

  def get_root_lin_vel(self):
    return self.root_lin_vel

  def get_foot_contact_states(self):
    return self.foot_contact_states

  def get_max_foot_contact_load(self):
    return float(self.max_foot_contact_load_n)

  def get_root_height(self):
    return self.root_height

  def get_obs_timestamp(self):
    return 0

  def get_pub_timestamp(self):
    return 0

  def get_write_timestamp(self):
    return 0

  def get_buttons(self):
    abxy = self.joystick.get_abxy()
    r1 = self.joystick.get_r1()

    if 1 in abxy:
      print(abxy)

    # Update key state
    self.key_state[0][1] = r1
    self.key_state[8][1] = abxy[0]
    self.key_state[9][1] = abxy[1]
    self.key_state[10][1] = abxy[2]
    self.key_state[11][1] = abxy[3]

    # Emulate L1 press after 1 second
    if time.time() - self.init_time > 1 and self.key_state[1][1] == 0:
      self.key_state[1][1] = 1
      print("[MujocoInterface] Emulating L1 press!")

    # # # Emulate X press after 7 seconds
    # if time.time() - self.init_time > 3 and self.key_state[10][1] == 0:
    #     self.key_state[10][1] = 1
    #     # print("[MujocoInterface] Emulating X press!")

    return self.key_state

  def set_base_pose(self, pose, vel=None, adjust_camera=False):
    """
    Set the base pose using IMU data.

    Args:
        accel (np.ndarray): 3D acceleration vector from IMU
        gyro (np.ndarray): 3D angular velocity vector from IMU
    """
    # R = get_rotation_matrix_from_rpy(euler_angles)
    R = pose[:3, :3]

    if adjust_camera:
      # Transform camera to body frame
      camera_rot = get_rotation_matrix_from_rpy(np.array([0, 0.88, 0]))
      R = R @ camera_rot.T

    # Convert to quaternion [x, y, z, w]
    trace = np.trace(R)
    if trace > 0:
      S = np.sqrt(trace + 1.0) * 2
      w = 0.25 * S
      x = (R[2, 1] - R[1, 2]) / S
      y = (R[0, 2] - R[2, 0]) / S
      z = (R[1, 0] - R[0, 1]) / S
    else:
      if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
      elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
      else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    # Update base position and orientation
    # In MuJoCo's free joint, first 3 values are position, next 4 are quaternion [w x y z]
    self.data.qpos[0:3] = pose[:3, 3]  # + np.array([0, 0, 1.1])  # Set position
    self.data.qpos[3:7] = [w, x, y, z]  # Set orientation
    self.data.qvel[0:6] = 0  # gyro  # Angular velocity

    # Forward kinematics to update all positions
    mujoco.mj_forward(self.model, self.data)

  def lock_base(self):
    self.base_lock_pos = self.data.qpos[:7].copy()

  def unlock_base(self):
    self.base_lock_pos = None

  def apply_command(self, kp, kd, q, dq, tau, torque_mode=False):
    if torque_mode:
      q_meas = self.get_joint_pos("hw")
      dq_meas = self.get_joint_vel("hw")
      tau_comp = kp * (q - q_meas) + kd * (dq - dq_meas)
      # for j in range(20):
      #     q[j] = 0.0
      #     dq[j] = 0.0
      #     kp[j] = 0.0
      #     kd[j] = 0.0
      #     tau[j] = tau_comp[j] + tau[j]
      q[:] = 0.0
      dq[:] = 0.0
      kp[:] = 0.0
      kd[:] = 0.0
      tau[:] = tau_comp + tau

    step_max_foot_contact_load_n = 0.0
    right_contact = False
    left_contact = False

    for _ in range(int(self.robot_configuration.dt / self.model.opt.timestep)):
      joint_pos = self.get_joint_pos("hw")
      joint_vel = self.get_joint_vel("hw")

      # Apply PD control
      torques = kp * (q - joint_pos) - kd * joint_vel + tau
      torques = self.robot_configuration.remap_joint_array(
        torques, from_order="hw", to_order="mujoco"
      )
      self.data.ctrl = torques

      if self.base_lock_pos is not None:
        self.data.qpos[:7] = self.base_lock_pos
        self.data.qvel[:6] = 0

      mujoco.mj_step(self.model, self.data)

      (
        substep_left_contact,
        substep_right_contact,
        substep_max_foot_contact_load_n,
      ) = self._compute_foot_contact_summary()
      left_contact = left_contact or substep_left_contact
      right_contact = right_contact or substep_right_contact
      step_max_foot_contact_load_n = max(
        step_max_foot_contact_load_n,
        substep_max_foot_contact_load_n,
      )

    steps_per_frame = int(0.04 / self.robot_configuration.dt)
    if self.viewer is not None and self.step_counter % steps_per_frame == 0:
      self.viewer.sync()

    self.last_action = q

    # Update state values
    self.joint_pos = np.array(self.data.qpos[self.jpos_idx : self.jpos_idx + 19])
    self.joint_vel = np.array(self.data.qvel[self.jvel_idx : self.jvel_idx + 19])
    self.gravity_vector = self._calculate_gravity_vector()
    self.root_ang_vel = np.array(self.data.qvel[3:6])

    # privileged state information
    root_quat_wxyz = self.data.qpos[3:7]
    root_quat_xyzw = root_quat_wxyz[[1, 2, 3, 0]]
    r = R.from_quat(root_quat_xyzw)
    self.root_lin_vel = r.apply(np.array(self.data.qvel[0:3]), inverse=True)

    # Update the foot contact states
    # self.foot_contact_states = np.array([left_contact, right_contact])
    self.foot_contact_states = np.array([right_contact, left_contact])
    self.max_foot_contact_load_n = step_max_foot_contact_load_n

    # print(f"Left foot IDs: {left_foot_geom_ids}, Right foot IDs: {right_foot_geom_ids}")
    # print(f"Foot contact states: {self.foot_contact_states}")

    self.root_height = self.data.qpos[2:3]

    self.step_counter += 1

  def get_joystick_commands(self):
    return self.joystick.get_command()[0:3]

  def get_force_command_input(self):
    return np.asarray(self.joystick.get_force_command(), dtype=float)

  def get_gait_params(self):
    """Return [gait_frequency_cmd, swing_height_cmd] both in [-1, 1]."""
    return self.joystick.get_gait_params()

  def get_height_commands(self):
    return self.joystick.get_command()[3] + 0.75

  def get_pitch_commands(self):
    return 0.0

  def get_frequency_command(self):
    return 0.9

  def check_comms_safety(self):
    return True

  def damping_mode(self):
    print("============================")
    print("ENTER DAMPING MODE")
    print("============================")

  def shutdown(self):
    if self.viewer is not None:
      self.viewer.close()

  def _compute_foot_contact_summary(self):
    left_contact = False
    right_contact = False
    max_foot_contact_load_n = 0.0
    contact_force = np.zeros(6, dtype=np.float64)

    for i in range(self.data.ncon):
      contact = self.data.contact[i]
      involves_left = (
        contact.geom1 in self.left_foot_geom_ids
        or contact.geom2 in self.left_foot_geom_ids
      )
      involves_right = (
        contact.geom1 in self.right_foot_geom_ids
        or contact.geom2 in self.right_foot_geom_ids
      )
      if not (involves_left or involves_right):
        continue

      mujoco.mj_contactForce(self.model, self.data, i, contact_force)
      max_foot_contact_load_n = max(
        max_foot_contact_load_n,
        float(np.linalg.norm(contact_force[:3])),
      )
      left_contact = left_contact or involves_left
      right_contact = right_contact or involves_right

    return left_contact, right_contact, max_foot_contact_load_n

  def _calculate_gravity_vector(self):
    # Get orientation

    # torso_body_id = self.model.body("torso_link").id
    # quat = np.array(self.data.xquat[torso_body_id])
    # rpy = get_rpy_from_quaternion(quat)
    quat = np.array(self.data.qpos[3:7])
    rpy = get_rpy_from_quaternion(quat)

    #         pelvis_body_id = self.model.body("pelvis").id
    #         quat = np.array(self.data.xquat[pelvis_body_id])
    #         rpy = get_rpy_from_quaternion(quat)

    rotation_matrix = get_rotation_matrix_from_rpy(np.array(rpy))

    # Transform gravity vector from world to body frame
    gravity_world = np.array([0, 0, -1])
    gravity_body = rotation_matrix.T @ gravity_world

    return gravity_body

  def repeat_last_command(self):
    # if self.last_action is not None:
    #     self.apply_command(None, None, self.last_action, None, None)
    pass

  def load_keyframe(self, keyframe_name):
    """
    Load a named keyframe from the model.

    Args:
        keyframe_name (str): Name of the keyframe to load

    Returns:
        bool: True if keyframe was found and loaded, False otherwise
    """
    key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    if key_id >= 0:
      print(f"Loading keyframe '{keyframe_name}'")
      # Copy keyframe state into qpos
      self.data.qpos[:] = self.model.key_qpos[
        key_id * self.model.nq : (key_id + 1) * self.model.nq
      ]
      # Reset velocities to zero
      self.data.qvel[:] = 0
      # Forward kinematics to update all positions
      mujoco.mj_forward(self.model, self.data)
      return True
    else:
      print(f"Warning: Keyframe '{keyframe_name}' not found")
      return False

  def get_available_keyframes(self):
    """
    Get list of available keyframe names in the model.

    Returns:
        list: List of keyframe names
    """
    keyframes = []
    for i in range(self.model.nkey):
      name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i)
      if name:
        keyframes.append(name)
    return keyframes
