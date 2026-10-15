"""Deploy script for WTW (Walk These Ways) policies.

Trains in MJLab. Uses MuJoCo joint order.

Usage:
    uv run --locked python deploy/deploy_wtw.py \
        --policy /path/to/your/wtw-policy.jit [options]

The --collect flag enables CSV data collection in the release row schema.
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
from deploy.sensors.foot_swing_height_sensor import FootSwingHeightCommandSensor
from deploy.sensors.gait_frequency_sensor import GaitFrequencyCommandSensor
from deploy.sensors.gait_phase_sensor import GaitPhaseSensor
from deploy.sensors.gravity_sensor import GravitySensor
from deploy.sensors.joint_pos_sensor import JointPosSensor
from deploy.sensors.joint_vel_sensor import JointVelSensor
from deploy.sensors.joystick_velocity_sensor import JoystickVelocitySensor
from deploy.sensors.last_action_sensor import LastActionSensor
from deploy.sensors.root_ang_vel_sensor import RootAngVelSensor

INTERFACE_TYPE = "mujoco"
TASK_NAME = "MJLab-G1-WTW"


def parse_args():
  p = argparse.ArgumentParser(description="Deploy WTW policy on G1")
  p.add_argument(
    "--policy",
    required=True,
    help=(
      "Policy path. Supports absolute, repo-relative, and deploy/models-relative paths."
    ),
  )
  p.add_argument("--interface", default=INTERFACE_TYPE)
  p.add_argument("--collect", action="store_true", help="Enable CSV data collection")
  p.add_argument(
    "--output_dir",
    default=None,
    help=(
      "Output directory for collected data (default: auto-named under checkpoint dir)"
    ),
  )
  p.add_argument("--render", action="store_true")
  return p.parse_args()


class WTWController(BaseController):
  """WTW (Walk These Ways) policy controller.

  Obs (per frame, 12 + nj*3 = 99D):
    ang_vel(3), gravity(3), joint_pos(29), joint_vel(29), last_action(29),
    velocity_cmd(3), gait_freq(1), swing_height(1), gait_phase(1)
  """

  policy_joint_order = "mujoco"

  def build_sensors(self):
    cfg = self.cfg
    velocity_sensor = JoystickVelocitySensor(self.robot, scale=1.0)
    gait_freq_sensor = GaitFrequencyCommandSensor(
      self.robot, scale=1.0, default_value=0.0
    )
    foot_swing_sensor = FootSwingHeightCommandSensor(
      self.robot, scale=1.0, default_value=0.0
    )
    gait_phase_sensor = GaitPhaseSensor(
      self.robot,
      scale=1.0,
      gait_freq_sensor=gait_freq_sensor,
      velocity_sensor=velocity_sensor,
      swing_height_sensor=foot_swing_sensor,
      freq_range=tuple(cfg.get("gait_phase_freq_range", (1.0, 2.5))),
      dt=cfg["control_dt"],
    )
    return [
      RootAngVelSensor(self.robot, scale=cfg["ang_vel_scale"]),
      GravitySensor(self.robot, scale=1.0),
      JointPosSensor(self.robot, scale=1.0, joint_order="mujoco"),
      JointVelSensor(self.robot, scale=cfg["dof_vel_scale"], joint_order="mujoco"),
      LastActionSensor(self.robot),
      velocity_sensor,
      gait_freq_sensor,
      foot_swing_sensor,
      gait_phase_sensor,
    ]


def main():
  args = parse_args()

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

  controller = WTWController(
    interface_type=args.interface,
    policy_name=args.policy,
    task_name=TASK_NAME,
    render=args.render,
    timestamp=timestamp,
  )

  collector = None
  if args.collect:
    from deploy.data_collector import DataCollector

    policy_dir = os.path.dirname(controller.policy_path)
    output_dir = args.output_dir or os.path.join(
      policy_dir, "real_data", f"wtw_{timestamp}"
    )
    collector = DataCollector(
      output_dir=output_dir,
      robot_configuration=controller.robot_configuration,
      kinematics=None,
      control_dt=controller.cfg["control_dt"],
      collection_mode="wtw",
    )
    print(f"[WTW] Data collection enabled → {output_dir}")

  def collect_callback(ctrl: WTWController):
    if collector is None or ctrl.iteration_count % ctrl.decimation != 0:
      return

    dof_pos_hw = ctrl.robot.get_joint_pos(joint_order="hw")
    dof_vel_hw = ctrl.robot.get_joint_vel(joint_order="hw")
    ang_vel = ctrl.robot.get_root_ang_vel()
    gravity = ctrl.robot.get_gravity_vector()
    base_height = ctrl.robot.get_root_height()

    # State: zeros for lin_vel (filled by postprocessor); torque is the
    # reconstructed commanded PD torque + tau_ff.
    # Store joint signals in canonical MuJoCo order for downstream MJLab tools.
    state = np.zeros(96)
    state[3:6] = ang_vel
    state[6:9] = gravity
    state[9:38] = ctrl.robot_configuration.remap_joint_array(
      dof_pos_hw, from_order="hw", to_order="mujoco"
    )
    state[38:67] = ctrl.robot_configuration.remap_joint_array(
      dof_vel_hw, from_order="hw", to_order="mujoco"
    )
    # state[0:3] = zeros (base_lin_vel, filled by postprocessor)
    state[67:96] = ctrl.robot_configuration.remap_joint_array(
      ctrl.last_reconstructed_torque_hw, from_order="hw", to_order="mujoco"
    )

    ext = collector.compute_placeholder_extension(base_height)
    contact = np.zeros(2)
    termination = np.zeros(1)

    # Store the final clipped joint position targets actually sent to
    # hardware, in canonical MuJoCo order.
    action_29 = ctrl.robot_configuration.remap_joint_array(
      ctrl.last_joint_position_target_hw,
      from_order="hw",
      to_order="mujoco",
    )
    collector.write_step(state, action_29, ext, contact, termination)

  try:
    controller.run(callback=collect_callback if args.collect else None)
  finally:
    if collector is not None:
      collector.save_meta()
    controller.shutdown()


if __name__ == "__main__":
  main()
