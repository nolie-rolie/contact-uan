"""Deploy script for MJLab force-control policies.

Trains in MJLab (force_control task). Uses MuJoCo joint order.
Newer force policies may omit base_lin_vel, so per-frame obs can be either 99D
or 96D. History length and active observation terms come from env.yaml.

Usage:
    uv run --locked python deploy/deploy_force_control.py \
        --policy /path/to/your/force-control-policy.jit [options]
"""

# ruff: noqa: E402
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy.base_controller import BaseController
from deploy.sensors.force_command_sensor import ForceCommandSensor
from deploy.sensors.gravity_sensor import GravitySensor
from deploy.sensors.joint_pos_sensor import JointPosSensor
from deploy.sensors.joint_vel_sensor import JointVelSensor
from deploy.sensors.last_action_sensor import LastActionSensor
from deploy.sensors.root_ang_vel_sensor import RootAngVelSensor
from deploy.sensors.root_lin_vel_sensor import RootLinVelSensor
from deploy.utils.force_reach_utils import (
  clip_xy_force_to_reach,
)

INTERFACE_TYPE = "mujoco"
TASK_NAME = "MJLab-G1-ForceControl"


class TrainingStyleForceCommandSampler:
  """Single-robot force command sampler matching MJLab training timing."""

  def __init__(self, sampling_cfg: dict, control_dt: float):
    self.force_lo = np.asarray(sampling_cfg["force_range"][0], dtype=float)
    self.force_hi = np.asarray(sampling_cfg["force_range"][1], dtype=float)
    stiffness_range = sampling_cfg.get("stiffness_range")
    self.stiffness_range = (
      None if not stiffness_range else tuple(float(v) for v in stiffness_range)
    )
    self.max_xy_reach_m = sampling_cfg.get("max_xy_reach_m")
    if self.max_xy_reach_m is not None:
      self.max_xy_reach_m = float(self.max_xy_reach_m)
    self.resampling_time_range = tuple(
      float(v) for v in sampling_cfg["resampling_time_range"]
    )
    self.sample_zero_force_probability = float(
      sampling_cfg["sample_zero_force_probability"]
    )
    self.force_ramp_duration_range_s = tuple(
      float(v) for v in sampling_cfg["force_ramp_duration_range_s"]
    )
    self.force_hold_duration_s = float(sampling_cfg["force_hold_duration_s"])
    self.startup_zero_force_duration_s = float(
      sampling_cfg["startup_zero_force_duration_s"]
    )
    self.control_dt = float(control_dt)
    self._rng = np.random.default_rng()
    self._hold_steps = int(self.force_hold_duration_s / self.control_dt)
    self._peak_force = np.zeros(3, dtype=float)
    self._current_force = np.zeros(3, dtype=float)
    self._current_stiffness = (
      None if self.stiffness_range is None else float(self.stiffness_range[0])
    )
    self._time_left_s = 0.0
    self._steps_since_resample = 0
    self._ramp_duration_steps = 1
    self.reset(log=False)

  def reset(self, *, log: bool = True) -> np.ndarray:
    self._peak_force[:] = 0.0
    self._current_force[:] = 0.0
    self._current_stiffness = (
      None if self.stiffness_range is None else float(self.stiffness_range[0])
    )
    self._steps_since_resample = 0
    self._ramp_duration_steps = 1
    self._time_left_s = self.startup_zero_force_duration_s
    if self._time_left_s <= 0.0:
      self._resample_command()
    elif log:
      print(
        "[ForceSampler] Restarted with startup zero-force window "
        f"{self.startup_zero_force_duration_s:.2f}s"
      )
    return self.current_force()

  def current_force(self) -> np.ndarray:
    return self._current_force.copy()

  def current_clip_stiffness(self) -> float | None:
    return self._current_stiffness

  def step(self) -> np.ndarray:
    self._time_left_s -= self.control_dt
    if self._time_left_s <= 0.0:
      self._resample_command()

    factor = self._compute_ramp_factor()
    self._current_force = self._peak_force * factor
    self._steps_since_resample += 1
    return self.current_force()

  def _compute_ramp_factor(self) -> float:
    ramp_steps = max(self._ramp_duration_steps, 1)
    step = float(self._steps_since_resample)
    phase_end_hold = ramp_steps + self._hold_steps
    phase_end_down = phase_end_hold + ramp_steps

    factor = min(max(step / ramp_steps, 0.0), 1.0)
    if ramp_steps <= step < phase_end_hold:
      factor = 1.0
    elif phase_end_hold <= step < phase_end_down:
      factor = 1.0 - min(max((step - phase_end_hold) / ramp_steps, 0.0), 1.0)
    elif step >= phase_end_down:
      factor = 0.0
    return float(factor)

  def _sample_duration(self, low: float, high: float) -> float:
    if high <= low:
      return float(low)
    return float(self._rng.uniform(low, high))

  def _format_force(self, force_xyz: np.ndarray) -> str:
    return f"({force_xyz[0]:+.2f}, {force_xyz[1]:+.2f}, {force_xyz[2]:+.2f})"

  def _resample_command(self) -> None:
    self._time_left_s = self._sample_duration(*self.resampling_time_range)
    peak_force = self._rng.uniform(self.force_lo, self.force_hi)
    if self._rng.random() < self.sample_zero_force_probability:
      peak_force[:] = 0.0
    if self.stiffness_range is not None:
      self._current_stiffness = self._sample_duration(*self.stiffness_range)
    else:
      self._current_stiffness = None
    self._peak_force = clip_xy_force_to_reach(
      peak_force,
      stiffness=self._current_stiffness,
      max_xy_reach_m=self.max_xy_reach_m,
    )
    self._steps_since_resample = 0
    ramp_s = self._sample_duration(*self.force_ramp_duration_range_s)
    self._ramp_duration_steps = max(1, int(ramp_s / self.control_dt))
    stiffness_msg = ""
    if self._current_stiffness is not None:
      stiffness_msg = f" stiffness={self._current_stiffness:.2f}N/m "
    print(
      "[ForceSampler] New sampled peak command "
      f"{self._format_force(self._peak_force)} N "
      f"{stiffness_msg}"
      f"ramp={ramp_s:.2f}s hold={self.force_hold_duration_s:.2f}s "
      f"next_resample={self._time_left_s:.2f}s"
    )


def parse_args():
  p = argparse.ArgumentParser(description="Deploy MJLab force-control policy on G1")
  p.add_argument("--policy", required=True, help="Path to your exported .jit policy.")
  p.add_argument("--interface", default=INTERFACE_TYPE)
  p.add_argument("--collect", action="store_true", help="Enable CSV data collection")
  p.add_argument("--output_dir", default=None)
  p.add_argument(
    "--ee_mass",
    type=float,
    default=0.0,
    help="End-effector added mass (kg) for data collection extension",
  )
  p.add_argument("--render", action="store_true")
  return p.parse_args()


class ForceControlController(BaseController):
  """MJLab force-control policy controller.

  Reads base_lin_vel from LCM vBody (requires lcm_interface fix).
  Reads force command from joystick or programmatic set_force_command().
  """

  policy_joint_order = "mujoco"

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    sampling_cfg = self.cfg.get("force_command_sampling")
    if sampling_cfg is None:
      raise ValueError("Force-control deploy requires force_command_sampling config")
    self._force_sampler = TrainingStyleForceCommandSampler(
      sampling_cfg=sampling_cfg,
      control_dt=self.cfg["control_dt"],
    )
    self._force_command_mode = "manual"
    self._prev_r1_pressed = False

  def build_sensors(self):
    cfg = self.cfg
    force_scale = cfg.get("force_scale", 1.0 / 30.0)
    force_max_abs = tuple(cfg.get("force_command_max_abs", (30.0, 30.0, 30.0)))
    force_sampling_cfg = cfg.get("force_command_sampling", {})
    stiffness_range = force_sampling_cfg.get("stiffness_range")
    manual_clip_stiffness = None if not stiffness_range else float(stiffness_range[0])
    self._force_sensor = ForceCommandSensor(
      self.robot,
      scale=force_scale,
      max_force_xyz=force_max_abs,
      max_xy_reach_m=force_sampling_cfg.get("max_xy_reach_m"),
      clip_stiffness=manual_clip_stiffness,
    )
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
      "force_command": lambda: self._force_sensor,
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
          "[ForceControl] Ignoring unsupported actor observation terms: "
          + ", ".join(unsupported_terms)
        )
      return sensors

    return [
      RootLinVelSensor(self.robot, scale=cfg.get("base_lin_vel_scale", 1.0)),
      RootAngVelSensor(self.robot, scale=cfg["ang_vel_scale"]),
      GravitySensor(self.robot, scale=1.0),
      JointPosSensor(self.robot, scale=1.0, joint_order="mujoco"),
      JointVelSensor(self.robot, scale=cfg["dof_vel_scale"], joint_order="mujoco"),
      LastActionSensor(self.robot),
      self._force_sensor,
    ]

  def _enable_sampled_force_commands(self) -> None:
    self._force_command_mode = "sampled"
    reset_force = self._force_sampler.reset()
    self._force_sensor.set_clip_stiffness(self._force_sampler.current_clip_stiffness())
    self._force_sensor.set_force_command(reset_force)
    print("Force command mode: sampled (R1 toggle)")

  def _disable_sampled_force_commands(self) -> None:
    self._force_command_mode = "manual"
    force_sampling_cfg = self.cfg.get("force_command_sampling", {})
    stiffness_range = force_sampling_cfg.get("stiffness_range")
    self._force_sensor.set_clip_stiffness(
      None if not stiffness_range else float(stiffness_range[0])
    )
    self._force_sensor.clear_programmatic()
    print("Force command mode: manual (R1 toggle)")

  def _toggle_force_command_mode(self) -> None:
    if self._force_command_mode == "sampled":
      self._disable_sampled_force_commands()
    else:
      self._enable_sampled_force_commands()

  def handle_runtime_buttons(
    self,
    buttons,
    *,
    pose_initialised: bool,
    policy_running: bool,
  ) -> None:
    del pose_initialised, policy_running
    r1_pressed = bool(buttons[0][1])
    if r1_pressed and not self._prev_r1_pressed:
      self._toggle_force_command_mode()
    self._prev_r1_pressed = r1_pressed

  def pre_policy_step(self) -> None:
    if self._force_command_mode != "sampled":
      return
    force_command = self._force_sampler.step()
    self._force_sensor.set_clip_stiffness(self._force_sampler.current_clip_stiffness())
    self._force_sensor.set_force_command(force_command)


def main():
  args = parse_args()
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

  controller = ForceControlController(
    interface_type=args.interface,
    policy_name=args.policy,
    task_name=TASK_NAME,
    render=args.render,
    timestamp=timestamp,
  )
  if args.interface != "mujoco":
    print(
      "[ForceControl] Manual controls: d-pad up/down = Fx +/-1N, "
      "d-pad right/left = Fz +/-1N, A = zero"
    )

  collector = None
  if args.collect:
    from deploy.data_collector import DataCollector

    policy_path = os.path.join(os.path.dirname(__file__), "models", args.policy)
    policy_dir = os.path.dirname(policy_path)
    output_dir = args.output_dir or os.path.join(
      policy_dir, "real_data", f"force_control_{timestamp}"
    )
    collector = DataCollector(
      output_dir=output_dir,
      robot_configuration=controller.robot_configuration,
      kinematics=None,
      control_dt=controller.cfg["control_dt"],
      collection_mode="force_control",
      ee_mass=args.ee_mass,
    )
    print(f"[ForceControl] Data collection enabled → {output_dir}")

  def collect_callback(ctrl: ForceControlController):
    if collector is None or ctrl.iteration_count % ctrl.decimation != 0:
      return

    dof_pos_hw = ctrl.robot.get_joint_pos(joint_order="hw")
    dof_vel_hw = ctrl.robot.get_joint_vel(joint_order="hw")
    ang_vel = ctrl.robot.get_root_ang_vel()
    gravity = ctrl.robot.get_gravity_vector()
    base_height = ctrl.robot.get_root_height()
    lin_vel = ctrl.robot.get_root_lin_vel()

    state = np.zeros(96)
    state[0:3] = lin_vel
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
    # Get raw (unscaled) force command for the extension vector
    force_xyz = ctrl._force_sensor.get_raw_force()
    ext = collector.compute_placeholder_extension(base_height, force_xyz=force_xyz)
    contact = np.zeros(2)
    termination = np.zeros(1)

    collector.write_step(state, action_29, ext, contact, termination)

  try:
    controller.run(callback=collect_callback if args.collect else None)
  finally:
    if collector is not None:
      collector.save_meta()
    controller.shutdown()


if __name__ == "__main__":
  main()
