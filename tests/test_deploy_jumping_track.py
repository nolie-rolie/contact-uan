from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from deploy import export_mjlab_checkpoint_to_jit as exporter
from deploy.base_controller import BaseController
from deploy.deploy_jumping_track import (
  MOTION_FILE,
  JumpingTrackController,
  parse_args,
)
from deploy.export_mjlab_checkpoint_to_jit import POLICY_SPECS, _infer_history_len
from deploy.sensors.gravity_sensor import GravitySensor
from deploy.sensors.motion_anchor_ang_vel_sensor import MotionAnchorAngVelSensor
from deploy.sensors.motion_anchor_height_sensor import MotionAnchorHeightSensor
from deploy.sensors.motion_anchor_lin_vel_sensor import MotionAnchorLinVelSensor
from deploy.sensors.motion_anchor_projected_gravity_sensor import (
  MotionAnchorProjectedGravitySensor,
)
from deploy.sensors.motion_clip_player import MotionClipPlayer
from deploy.sensors.motion_joint_pos_sensor import MotionJointPosSensor
from deploy.sensors.motion_joint_vel_sensor import MotionJointVelSensor


def _motion_file(tmp_path: Path) -> Path:
  joint_pos = np.arange(58, dtype=np.float32).reshape(2, 29)
  joint_vel = joint_pos + 100.0
  body_pos = np.zeros((2, 30, 3), dtype=np.float32)
  body_quat = np.zeros((2, 30, 4), dtype=np.float32)
  body_quat[..., 0] = 1.0
  body_pos[:, 15, 2] = (0.7, 1.1)
  body_lin_vel = np.zeros((2, 30, 3), dtype=np.float32)
  body_ang_vel = np.zeros((2, 30, 3), dtype=np.float32)
  body_lin_vel[1, 15] = (1.0, 2.0, 3.0)
  body_ang_vel[1, 15] = (4.0, 5.0, 6.0)
  path = tmp_path / "motion.npz"
  np.savez(
    path,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    body_pos_w=body_pos,
    body_quat_w=body_quat,
    body_lin_vel_w=body_lin_vel,
    body_ang_vel_w=body_ang_vel,
  )
  return path


def test_motion_player_exposes_jumping_track_reference(tmp_path: Path) -> None:
  player = MotionClipPlayer(str(_motion_file(tmp_path)))
  half_turn = np.array([2.0**-0.5, 0.0, 0.0, 2.0**-0.5], dtype=np.float32)
  player.align_to_robot(np.array([2.0, 3.0, 0.7], dtype=np.float32), half_turn)
  player.trigger()
  player.advance()

  assert player.anchor_height == pytest.approx(1.1)
  np.testing.assert_allclose(player.anchor_projected_gravity, [0.0, 0.0, -1.0])
  np.testing.assert_allclose(player.anchor_lin_vel_b, [1.0, 2.0, 3.0], atol=1e-6)
  np.testing.assert_allclose(player.anchor_ang_vel_b, [4.0, 5.0, 6.0], atol=1e-6)
  assert player.anchor_projected_gravity.dtype == np.float32
  assert player.anchor_lin_vel_b.dtype == np.float32
  assert player.anchor_ang_vel_b.dtype == np.float32


def test_reference_sensors_match_player_values(tmp_path: Path) -> None:
  player = MotionClipPlayer(str(_motion_file(tmp_path)))
  player.trigger()
  player.advance()
  interface = object()
  sensors = (
    MotionAnchorHeightSensor(interface, player),
    MotionAnchorProjectedGravitySensor(interface, player),
    MotionJointPosSensor(interface, player),
    MotionJointVelSensor(interface, player),
    MotionAnchorLinVelSensor(interface, player),
    MotionAnchorAngVelSensor(interface, player),
  )
  expected_shapes = ((1,), (3,), (29,), (29,), (3,), (3,))

  for sensor, shape in zip(sensors, expected_shapes, strict=True):
    value = sensor.get_data()
    assert value.shape == shape
    assert value.dtype == np.float32


class _MeasuredRobot:
  def get_gravity_vector(self):
    return np.array([0.5, 0.0, -0.8660254], dtype=np.float32)

  def get_root_ang_vel(self):
    return np.zeros(3, dtype=np.float32)


def _build_controller_sensors(controller_type, tmp_path: Path):
  controller = object.__new__(controller_type)
  controller.cfg = {
    "ang_vel_scale": 1.0,
    "dof_vel_scale": 1.0,
  }
  controller.robot = _MeasuredRobot()
  controller._motion_file = str(_motion_file(tmp_path))
  return controller.build_sensors()


def test_controller_builds_exact_161d_sensor_order(tmp_path: Path) -> None:
  sensors = _build_controller_sensors(JumpingTrackController, tmp_path)
  dims = [29 if sensor.dim == "nj" else int(sensor.dim) for sensor in sensors]

  assert dims == [1, 3, 29, 29, 3, 3, 3, 3, 29, 29, 29]
  assert sum(dims) == 161
  assert isinstance(sensors[6], GravitySensor)
  np.testing.assert_allclose(sensors[1].get_data(), [0.0, 0.0, -1.0], atol=1e-6)
  np.testing.assert_allclose(sensors[6].get_data(), [0.5, 0.0, -0.8660254], atol=1e-6)
  assert not np.array_equal(sensors[1].get_data(), sensors[6].get_data())
  assert issubclass(JumpingTrackController, BaseController)


def test_cli_requires_explicit_policy(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(sys, "argv", ["deploy_jumping_track.py"])
  with pytest.raises(SystemExit) as error:
    parse_args()
  assert error.value.code == 2


def test_cli_keeps_simulation_and_motion_defaults(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    sys, "argv", ["deploy_jumping_track.py", "--policy", "custom.jit"]
  )
  args = parse_args()
  assert args.policy == "custom.jit"
  assert args.interface == "mujoco"
  assert args.motion_file == MOTION_FILE


def test_cli_accepts_policy_and_motion_overrides(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "deploy_jumping_track.py",
      "--policy",
      "/tmp/custom.jit",
      "--motion-file",
      "/tmp/custom-motion.npz",
      "--headless",
    ],
  )
  args = parse_args()
  assert args.policy == "/tmp/custom.jit"
  assert args.motion_file == "/tmp/custom-motion.npz"
  assert args.headless


def test_exporter_supports_161d_observations() -> None:
  spec = POLICY_SPECS["jumping_track"]

  assert spec["frame_dim"] == 161
  assert spec["default_joint_order_target"] == "mujoco"
  assert _infer_history_len(161, spec["frame_dim"], override=None) == 1


def test_exporter_rejects_isaaclab_jump_before_loading_checkpoint(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "export_mjlab_checkpoint_to_jit.py",
      "--policy-type",
      "jumping_track",
      "--checkpoint",
      "/tmp/unused-model.pt",
      "--joint-order-target",
      "isaaclab",
    ],
  )

  def fail_if_loaded(_path: str) -> dict:
    pytest.fail("Unsupported joint order must be rejected before checkpoint loading")

  monkeypatch.setattr(exporter, "_safe_load_checkpoint", fail_if_loaded)
  with pytest.raises(ValueError, match="supports only MuJoCo joint order"):
    exporter.main()
