"""Offline tests of deployment torque clipping; no robot interface is created."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from deploy.base_controller import BaseController, compute_torque_speed_limits


def test_torque_speed_limits_reduce_motoring_but_preserve_braking_torque():
  lower, upper = compute_torque_speed_limits(
    joint_vel=np.array([[0.0, 10.0, 60.0], [-5.0, -20.0, -15.0]]),
    saturation_effort=np.array([10.0, 20.0, 30.0]),
    velocity_limit=np.array([10.0, 20.0, 30.0]),
  )

  np.testing.assert_allclose(upper, [[10.0, 10.0, 0.0], [10.0, 20.0, 30.0]])
  np.testing.assert_allclose(lower, [[-10.0, -20.0, -30.0], [-5.0, 0.0, -15.0]])


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_high_speed_allows_braking_without_requiring_reverse_torque(direction):
  lower, upper = compute_torque_speed_limits(
    joint_vel=direction * np.array([20.0, 25.0, 100.0]),
    saturation_effort=np.array([139.0]),
    velocity_limit=np.array([20.0]),
  )

  expected_lower, expected_upper = (-139.0, 0.0) if direction > 0 else (0.0, 139.0)
  np.testing.assert_array_equal(lower, np.full(3, expected_lower))
  np.testing.assert_array_equal(upper, np.full(3, expected_upper))


@pytest.mark.parametrize(
  ("stall", "speed"), [(0.0, 20.0), (-1.0, 20.0), (139.0, 0.0), (139.0, -1.0)]
)
def test_torque_speed_limits_reject_nonpositive_motor_parameters(stall, speed):
  with pytest.raises(ValueError, match="must be positive"):
    compute_torque_speed_limits(
      joint_vel=np.array([0.0]),
      saturation_effort=np.array([stall]),
      velocity_limit=np.array([speed]),
    )


def test_execute_policy_bounds_sent_targets_in_damping_and_overspeed_cases():
  joint_pos = np.zeros(4)
  joint_vel = np.array([12.0, -12.0, 20.0, -25.0])
  kp = np.full(4, 99.09843)
  kd = np.full(4, 6.30880)
  robot = SimpleNamespace(
    get_joint_pos=Mock(return_value=joint_pos),
    get_joint_vel=Mock(return_value=joint_vel),
    apply_command=Mock(),
  )
  robot_configuration = SimpleNamespace(
    num_joints=4,
    remap_joint_array=lambda values, **kwargs: values.copy(),
    joint_default_positions=Mock(return_value=np.zeros(4)),
    joint_stiffnesses=Mock(return_value=kp),
    joint_dampings=Mock(return_value=kd),
    joint_saturated_torques=Mock(return_value=np.full(4, 139.0)),
    joint_max_velocities=Mock(return_value=np.full(4, 20.0)),
  )
  # Call the numerical policy-step method on plain test doubles. Bypass the
  # controller constructor, which would initialize a real or simulated interface.
  controller = cast(
    BaseController,
    SimpleNamespace(
      compute_observation=Mock(return_value=[]),
      prev_action=np.array([0.01, -0.01, 2.0, -2.0]),
      cfg={"control": {"action_scale": np.ones(4)}, "offset_actions": False},
      robot=robot,
      robot_configuration=robot_configuration,
      policy_joint_order="mujoco",
      last_joint_position_target_hw=np.zeros(4),
      last_reconstructed_torque_hw=np.zeros(4),
    ),
  )

  returned_target = BaseController.execute_policy(controller, previous_hold=True)

  robot.apply_command.assert_called_once()
  command = robot.apply_command.call_args.kwargs
  sent_pd_torque = (
    command["kp"] * (command["q"] - joint_pos)
    + command["kd"] * (command["dq"] - joint_vel)
    + command["tau"]
  )
  # Damping-induced braking is allowed up to stall torque. At and above the
  # speed limit, a motoring request clips to zero rather than reverse torque.
  expected_torque = np.array([-74.7146157, 74.7146157, 0.0, 0.0])
  np.testing.assert_allclose(sent_pd_torque, expected_torque, atol=1e-12)
  np.testing.assert_allclose(controller.last_reconstructed_torque_hw, expected_torque)
  np.testing.assert_allclose(controller.last_joint_position_target_hw, command["q"])
  np.testing.assert_allclose(returned_target, command["q"])
