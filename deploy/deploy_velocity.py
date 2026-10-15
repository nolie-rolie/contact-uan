"""Deploy script for MJLab custom velocity policies.

Trains in MJLab (velocity task). Uses MuJoCo joint order.
Velocity actors may be either 96D or 99D per frame depending on whether
base_lin_vel is included. History and active observation terms come from
env.yaml.

Usage:
    uv run --locked python deploy/deploy_velocity.py \
        --policy /path/to/your/velocity-policy.jit [options]
"""

# ruff: noqa: E402
import argparse
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy.base_controller import BaseController
from deploy.sensors.base_sensor import BaseSensor
from deploy.sensors.gravity_sensor import GravitySensor
from deploy.sensors.joint_pos_sensor import JointPosSensor
from deploy.sensors.joint_vel_sensor import JointVelSensor
from deploy.sensors.joystick_velocity_sensor import (
  map_normalized_velocity_commands_to_range,
)
from deploy.sensors.last_action_sensor import LastActionSensor
from deploy.sensors.root_ang_vel_sensor import RootAngVelSensor
from deploy.sensors.root_lin_vel_sensor import RootLinVelSensor

INTERFACE_TYPE = "mujoco"
TASK_NAME = "MJLab-G1-Velocity"


def parse_args():
  p = argparse.ArgumentParser(description="Deploy MJLab velocity policy on G1")
  p.add_argument("--policy", required=True, help="Path to your exported .jit policy.")
  p.add_argument("--interface", default=INTERFACE_TYPE)
  p.add_argument("--collect", action="store_true", help="Enable CSV data collection")
  p.add_argument("--output_dir", default=None)
  p.add_argument("--render", action="store_true")
  return p.parse_args()


def _button_is_pressed(buttons, index: int, name: str) -> bool:
  if isinstance(buttons, dict):
    return bool(buttons.get(name, 0))
  try:
    entry = buttons[index]
  except (IndexError, TypeError):
    return False
  if isinstance(entry, (list, tuple, np.ndarray)):
    if len(entry) < 2:
      return False
    return bool(entry[1])
  return bool(entry)


class VelocityCommandState:
  """Compose velocity commands from joystick or D-pad speed mode."""

  MODE_JOYSTICK = "joystick"
  MODE_DPAD = "dpad"
  DPAD_STEP_MPS = 0.5
  DPAD_BUTTONS = {
    "up": 12,
    "right": 13,
    "down": 14,
    "left": 15,
  }

  def __init__(self, cfg: dict, dpad_step_mps: float = DPAD_STEP_MPS):
    command_ranges = cfg["command_ranges"]
    self._min_limits = np.array(
      [
        command_ranges["lin_vel_x"][0],
        command_ranges["lin_vel_y"][0],
        command_ranges["ang_vel_yaw"][0],
      ],
      dtype=np.float32,
    )
    self._max_limits = np.array(
      [
        command_ranges["lin_vel_x"][1],
        command_ranges["lin_vel_y"][1],
        command_ranges["ang_vel_yaw"][1],
      ],
      dtype=np.float32,
    )
    self.dpad_step_mps = float(dpad_step_mps)
    self.mode = self.MODE_JOYSTICK
    self._dpad_lin_vel_x_mps = 0.0
    self._prev_dpad = {name: False for name in self.DPAD_BUTTONS}

  @property
  def dpad_lin_vel_x_mps(self) -> float:
    return self._dpad_lin_vel_x_mps

  def update_from_buttons(self, buttons, raw_joystick_commands=None) -> bool:
    """Update command mode/state from D-pad rising edges."""
    curr_dpad = {
      name: _button_is_pressed(buttons, index, name)
      for name, index in self.DPAD_BUTTONS.items()
    }
    changed = False

    if self._rising_edge(curr_dpad, "right"):
      if self.mode == self.MODE_JOYSTICK:
        if raw_joystick_commands is not None:
          joystick_command = self._map_joystick_command(raw_joystick_commands)
          self._dpad_lin_vel_x_mps = self._clamp_dpad_speed(joystick_command[0])
        self.mode = self.MODE_DPAD
      else:
        self.mode = self.MODE_JOYSTICK
      changed = True

    if self.mode == self.MODE_DPAD:
      if self._rising_edge(curr_dpad, "up"):
        self._dpad_lin_vel_x_mps = self._clamp_dpad_speed(
          self._dpad_lin_vel_x_mps + self.dpad_step_mps
        )
        changed = True
      if self._rising_edge(curr_dpad, "down"):
        self._dpad_lin_vel_x_mps = self._clamp_dpad_speed(
          self._dpad_lin_vel_x_mps - self.dpad_step_mps
        )
        changed = True
      if self._rising_edge(curr_dpad, "left"):
        self._dpad_lin_vel_x_mps = 0.0
        changed = True

    self._prev_dpad = curr_dpad
    return changed

  def compose_command(self, raw_joystick_commands) -> np.ndarray:
    joystick_command = self._map_joystick_command(raw_joystick_commands)
    if self.mode == self.MODE_JOYSTICK:
      return joystick_command
    return np.array(
      [
        self._dpad_lin_vel_x_mps,
        0.0,
        joystick_command[2],
      ],
      dtype=np.float32,
    )

  def _rising_edge(self, curr_dpad: dict[str, bool], name: str) -> bool:
    return curr_dpad[name] and not self._prev_dpad[name]

  def _map_joystick_command(self, raw_joystick_commands) -> np.ndarray:
    return map_normalized_velocity_commands_to_range(
      raw_joystick_commands,
      self._min_limits,
      self._max_limits,
    )

  def _clamp_dpad_speed(self, value: float) -> float:
    max_forward = max(0.0, float(self._max_limits[0]))
    return float(np.clip(value, 0.0, max_forward))


class VelocityCommandSensor(BaseSensor):
  """Velocity command observation from the deploy velocity controller."""

  dim = 3

  def get_data(self):
    command = self.interface.get_velocity_command_mps()
    obs_scales = self.interface.cfg["obs_scales"]
    scaled = np.array(
      [
        command[0] * obs_scales["lin_vel"],
        command[1] * obs_scales["lin_vel"],
        command[2] * obs_scales["ang_vel"],
      ],
      dtype=np.float32,
    )
    return scaled * self.scale


class VelocityCommandSidecarLogger:
  """CSV sidecar aligned one-to-one with canonical DataCollector rows."""

  HEADER = [
    "sample_index",
    "iteration_count",
    "control_time_s",
    "wall_time_s",
    "command_mode",
    "lin_vel_x_mps",
    "lin_vel_y_mps",
    "ang_vel_yaw_radps",
  ]

  def __init__(self, output_dir: str, control_dt: float):
    self.path = os.path.join(output_dir, "velocity_commands.csv")
    self.control_dt = float(control_dt)
    self.sample_index = 0
    self._file = open(self.path, "w", newline="")
    self._writer = csv.writer(self._file)
    self._writer.writerow(self.HEADER)

  def write_step(self, ctrl) -> None:
    command = ctrl.get_last_velocity_command_mps()
    self._writer.writerow(
      [
        self.sample_index,
        int(ctrl.iteration_count),
        float(ctrl.iteration_count) * self.control_dt,
        time.time(),
        ctrl.velocity_command_mode,
        float(command[0]),
        float(command[1]),
        float(command[2]),
      ]
    )
    self.sample_index += 1

  def close(self) -> None:
    if self._file.closed:
      return
    self._file.flush()
    self._file.close()


class VelocityController(BaseController):
  """MJLab custom velocity policy controller (purely proprioceptive).

  Matches the current MJLab custom velocity actor observations.
  """

  policy_joint_order = "mujoco"

  def build_sensors(self):
    cfg = self.cfg
    self._velocity_command_state = VelocityCommandState(cfg)
    self._last_velocity_command_mps = np.zeros(3, dtype=np.float32)
    self._velocity_sensor = VelocityCommandSensor(self, scale=1.0)
    actor_terms = cfg.get("actor_observation_terms", [])
    sensor_builders = {
      "base_lin_vel": lambda: RootLinVelSensor(
        self.robot, scale=cfg.get("base_lin_vel_scale", 1.0)
      ),
      "base_ang_vel": lambda: RootAngVelSensor(self.robot, scale=cfg["ang_vel_scale"]),
      "projected_gravity": lambda: GravitySensor(self.robot, scale=1.0),
      "gravity": lambda: GravitySensor(self.robot, scale=1.0),
      "joint_pos_rel": lambda: JointPosSensor(
        self.robot, scale=1.0, joint_order="mujoco"
      ),
      "joint_pos": lambda: JointPosSensor(self.robot, scale=1.0, joint_order="mujoco"),
      "joint_vel_rel": lambda: JointVelSensor(
        self.robot, scale=cfg["dof_vel_scale"], joint_order="mujoco"
      ),
      "joint_vel": lambda: JointVelSensor(
        self.robot, scale=cfg["dof_vel_scale"], joint_order="mujoco"
      ),
      "last_action": lambda: LastActionSensor(self.robot),
      "actions": lambda: LastActionSensor(self.robot),
      "velocity_commands": lambda: self._velocity_sensor,
      "command": lambda: self._velocity_sensor,
    }
    sensors = []
    unsupported_terms = []
    for term_name in actor_terms:
      builder = sensor_builders.get(term_name)
      if builder is None:
        unsupported_terms.append(term_name)
        continue
      sensors.append(builder())

    if sensors:
      if unsupported_terms:
        print(
          "[Velocity] Ignoring unsupported actor observation terms: "
          + ", ".join(unsupported_terms)
        )
      return sensors

    return [
      RootAngVelSensor(self.robot, scale=cfg["ang_vel_scale"]),
      GravitySensor(self.robot, scale=1.0),
      JointPosSensor(self.robot, scale=1.0, joint_order="mujoco"),
      JointVelSensor(self.robot, scale=cfg["dof_vel_scale"], joint_order="mujoco"),
      LastActionSensor(self.robot),
      self._velocity_sensor,
    ]

  @property
  def velocity_command_mode(self) -> str:
    return self._velocity_command_state.mode

  def handle_runtime_buttons(
    self,
    buttons,
    *,
    pose_initialised: bool,
    policy_running: bool,
  ) -> None:
    del pose_initialised, policy_running
    changed = self._velocity_command_state.update_from_buttons(
      buttons,
      raw_joystick_commands=self.robot.get_joystick_commands(),
    )
    if not changed:
      return
    command = self.get_velocity_command_mps()
    print(
      "[VelocityCmd] "
      f"mode={self.velocity_command_mode} "
      f"cmd=({command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f})"
    )

  def get_velocity_command_mps(self) -> np.ndarray:
    command = self._velocity_command_state.compose_command(
      self.robot.get_joystick_commands()
    )
    self._last_velocity_command_mps = command.astype(np.float32, copy=True)
    return self._last_velocity_command_mps.copy()

  def get_last_velocity_command_mps(self) -> np.ndarray:
    return self._last_velocity_command_mps.copy()


def main():
  args = parse_args()
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

  controller = VelocityController(
    interface_type=args.interface,
    policy_name=args.policy,
    task_name=TASK_NAME,
    render=args.render,
    timestamp=timestamp,
  )

  collector = None
  velocity_command_logger = None
  contact_force_window = deque()
  last_contact_force_log_time = time.time()
  if args.collect:
    from deploy.data_collector import DataCollector

    policy_path = os.path.join(os.path.dirname(__file__), "models", args.policy)
    policy_dir = os.path.dirname(policy_path)
    output_dir = args.output_dir or os.path.join(
      policy_dir, "real_data", f"velocity_{timestamp}"
    )
    collector = DataCollector(
      output_dir=output_dir,
      robot_configuration=controller.robot_configuration,
      kinematics=None,
      control_dt=controller.cfg["control_dt"],
      collection_mode="wtw",
    )
    velocity_command_logger = VelocityCommandSidecarLogger(
      output_dir=output_dir,
      control_dt=controller.cfg["control_dt"],
    )
    print(f"[Velocity] Data collection enabled → {output_dir}")
    print(f"[Velocity] Command sidecar → {velocity_command_logger.path}")

  def runtime_callback(ctrl: VelocityController):
    nonlocal last_contact_force_log_time

    if args.interface == "mujoco" and hasattr(ctrl.robot, "get_max_foot_contact_load"):
      now = time.time()
      contact_force_window.append((now, float(ctrl.robot.get_max_foot_contact_load())))
      cutoff = now - 1.0
      while contact_force_window and contact_force_window[0][0] < cutoff:
        contact_force_window.popleft()
      if now - last_contact_force_log_time >= 1.0 and contact_force_window:
        values = np.fromiter(
          (force for _, force in contact_force_window),
          dtype=np.float64,
        )
        print(
          f"[VelocityContact] last_1s max={values.max():.1f}N mean={values.mean():.1f}N"
        )
        last_contact_force_log_time = now

    if collector is None or ctrl.iteration_count % ctrl.decimation != 0:
      return

    dof_pos_hw = ctrl.robot.get_joint_pos(joint_order="hw")
    dof_vel_hw = ctrl.robot.get_joint_vel(joint_order="hw")
    ang_vel = ctrl.robot.get_root_ang_vel()
    gravity = ctrl.robot.get_gravity_vector()
    base_height = ctrl.robot.get_root_height()
    state = np.zeros(96)
    # Match the canonical hardware schema used by downstream MJLab tools:
    # lin_vel is left zero for postprocess, then ang_vel, gravity, joints,
    # joint velocities, and reconstructed torque follow in fixed slots.
    state[3:6] = ang_vel
    state[6:9] = gravity
    state[9:38] = ctrl.robot_configuration.remap_joint_array(
      dof_pos_hw, from_order="hw", to_order="mujoco"
    )
    state[38:67] = ctrl.robot_configuration.remap_joint_array(
      dof_vel_hw, from_order="hw", to_order="mujoco"
    )
    state[67:96] = ctrl.robot_configuration.remap_joint_array(
      ctrl.last_reconstructed_torque_hw, from_order="hw", to_order="mujoco"
    )

    action_29 = ctrl.robot_configuration.remap_joint_array(
      ctrl.last_joint_position_target_hw,
      from_order="hw",
      to_order="mujoco",
    )
    ext = collector.compute_placeholder_extension(base_height)
    contact = np.zeros(2)
    termination = np.zeros(1)

    collector.write_step(state, action_29, ext, contact, termination)
    if velocity_command_logger is not None:
      velocity_command_logger.write_step(ctrl)

  try:
    controller.run(callback=runtime_callback)
  finally:
    try:
      if collector is not None:
        collector.save_meta()
    finally:
      try:
        if velocity_command_logger is not None:
          velocity_command_logger.close()
      finally:
        controller.shutdown()


if __name__ == "__main__":
  main()
