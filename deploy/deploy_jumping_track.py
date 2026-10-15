"""Deploy script for corrected JumpingTrack task policies.

Actor obs:
  motion_anchor_height             (1)  — reference torso height
  motion_anchor_projected_gravity  (3)  — gravity in reference torso frame
  motion_joint_pos                (29)  — reference joint positions
  motion_joint_vel                (29)  — reference joint velocities
  motion_anchor_lin_vel_b          (3)  — reference torso-frame linear velocity
  motion_anchor_ang_vel_b          (3)  — reference torso-frame angular velocity
  projected_gravity                (3)  — gravity measured in the robot root frame
  base_ang_vel                     (3)  — from IMU
  joint_pos                       (29)  — relative to default
  joint_vel                       (29)
  last_action                     (29)
Total: 161D (no history).

macOS usage:
    uv run --locked mjpython deploy/deploy_jumping_track.py \
      --policy /path/to/your/jumping-policy.jit --render
"""

# ruff: noqa: E402
import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy.base_controller import BaseController
from deploy.sensors.gravity_sensor import GravitySensor
from deploy.sensors.joint_pos_sensor import JointPosSensor
from deploy.sensors.joint_vel_sensor import JointVelSensor
from deploy.sensors.last_action_sensor import LastActionSensor
from deploy.sensors.motion_anchor_ang_vel_sensor import MotionAnchorAngVelSensor
from deploy.sensors.motion_anchor_height_sensor import MotionAnchorHeightSensor
from deploy.sensors.motion_anchor_lin_vel_sensor import MotionAnchorLinVelSensor
from deploy.sensors.motion_anchor_projected_gravity_sensor import (
  MotionAnchorProjectedGravitySensor,
)
from deploy.sensors.motion_clip_player import MotionClipPlayer
from deploy.sensors.motion_joint_pos_sensor import MotionJointPosSensor
from deploy.sensors.motion_joint_vel_sensor import MotionJointVelSensor
from deploy.sensors.root_ang_vel_sensor import RootAngVelSensor

INTERFACE_TYPE = "mujoco"
# The motion and padding must match the selected policy's training clip.
MOTION_FILE = "motions/02_04_jump_balance_pad50.npz"  # pad50 clip (land + hold)

TASK_NAME = "MJLab-G1-JumpingTrack"


def parse_args():
  p = argparse.ArgumentParser(description="Deploy corrected JumpingTrack policy on G1")
  p.add_argument(
    "--policy",
    required=True,
    help="Policy .jit path, either absolute or relative to deploy/models/.",
  )
  p.add_argument("--interface", default=INTERFACE_TYPE)
  p.add_argument(
    "--motion-file",
    default=MOTION_FILE,
    help="Path to the padded .npz motion clip.",
  )
  p.add_argument("--render", action="store_true")
  p.add_argument("--headless", action="store_true")
  return p.parse_args()


class JumpingTrackController(BaseController):
  """Corrected 161-D JumpingTrack policy controller."""

  policy_joint_order = "mujoco"
  TRIGGER_BUTTON = 8

  def __init__(self, motion_file: str, **kwargs):
    self._motion_file = motion_file
    self._player: MotionClipPlayer | None = None
    self._prev_trigger = 0
    super().__init__(**kwargs)

  def build_sensors(self):
    cfg = self.cfg
    player = MotionClipPlayer(self._motion_file)
    self._player = player

    return [
      MotionAnchorHeightSensor(self.robot, player),
      MotionAnchorProjectedGravitySensor(self.robot, player),
      MotionJointPosSensor(self.robot, player),
      MotionJointVelSensor(self.robot, player),
      MotionAnchorLinVelSensor(self.robot, player),
      MotionAnchorAngVelSensor(self.robot, player),
      GravitySensor(self.robot),
      RootAngVelSensor(self.robot, scale=cfg.get("ang_vel_scale", 1.0)),
      JointPosSensor(self.robot, scale=1.0, joint_order="mujoco"),
      JointVelSensor(
        self.robot, scale=cfg.get("dof_vel_scale", 1.0), joint_order="mujoco"
      ),
      LastActionSensor(self.robot),
    ]

  def _arm_clip(self) -> None:
    assert self._player is not None
    self._player.rewind()

  def on_pose_initialized(self) -> None:
    if self._player is not None:
      self._arm_clip()
      print(f"Jump armed. Press A (button {self.TRIGGER_BUTTON}) to jump.")

  def handle_runtime_buttons(
    self, buttons, *, pose_initialised: bool, policy_running: bool
  ) -> None:
    pressed = buttons[self.TRIGGER_BUTTON][1] == 1
    rising_edge = pressed and self._prev_trigger == 0
    self._prev_trigger = 1 if pressed else 0
    if (
      policy_running
      and rising_edge
      and self._player is not None
      and self._player.is_armed
    ):
      print("Jump triggered.")
      self._arm_clip()
      self._player.trigger()

  def pre_policy_step(self) -> None:
    if self._player is not None:
      self._player.advance()


def main():
  args = parse_args()

  controller = JumpingTrackController(
    motion_file=args.motion_file,
    interface_type=args.interface,
    policy_name=args.policy,
    task_name=TASK_NAME,
    render=args.render,
    headless=args.headless,
  )

  try:
    controller.run()
  finally:
    controller.shutdown()


if __name__ == "__main__":
  main()
