"""Regression tests for the shared training target-clipping rule."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from deploy.base_controller import compute_torque_speed_limits

from mjlab.tasks.velocity.mdp.actions import (
  TorqueSpeedClippedJointPositionAction,
  _clip_position_targets,
)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_target_clipping_limits_damping_dominated_torque(
  direction: float,
) -> None:
  # G1 knee parameters, with a small target error while moving at 24 rad/s.
  # Damping reverses the PD torque relative to the position-error selector.
  joint_pos = torch.zeros((1, 1), dtype=torch.float64)
  target = torch.full_like(joint_pos, direction * 0.01)
  joint_vel = torch.full_like(joint_pos, direction * 24.0)
  kp = torch.full_like(joint_pos, 99.09843)
  kd = torch.full_like(joint_pos, 6.30880)

  clipped = _clip_position_targets(
    position_target=target,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    kp=kp,
    kd=kd,
    saturation_effort=torch.full_like(joint_pos, 139.0),
    velocity_limit=torch.full_like(joint_pos, 20.0),
  )

  # The old position-error selector left |torque| at 150.420 Nm here.
  # Clipping both bounds must enforce the opposite torque bound as well.
  assert not torch.equal(clipped, target)
  pd_torque = kp * (clipped - joint_pos) - kd * joint_vel
  assert pd_torque.item() == pytest.approx(-direction * 139.0)


def test_training_and_deployment_match_asymmetric_envelope() -> None:
  # Heterogeneous joints and batches cover both velocity signs, stall,
  # no-load speed, overspeed, braking, and targets already inside the bounds.
  saturation = np.array([25.0, 88.0, 139.0, 5.0, 50.0])
  speed = np.array([37.0, 32.0, 20.0, 22.0, 37.0])
  kp = np.array([14.25, 40.18, 99.10, 16.78, 28.50])
  kd = np.array([0.91, 2.56, 6.31, 1.07, 1.81])
  speed_fraction = np.array([-1.5, -1.0, -0.4, 0.0, 0.4, 1.0, 1.5])[:, None]
  velocity = speed_fraction * speed
  q = np.arange(velocity.size).reshape(velocity.shape) * 0.01
  nominal_torque = np.array([-200.0, -1.0, 0.0, 1.0, 200.0])
  target = q + (nominal_torque + kd * velocity) / kp

  clipped = _clip_position_targets(
    position_target=torch.from_numpy(target),
    joint_pos=torch.from_numpy(q),
    joint_vel=torch.from_numpy(velocity),
    kp=torch.from_numpy(kp),
    kd=torch.from_numpy(kd),
    saturation_effort=torch.from_numpy(saturation),
    velocity_limit=torch.from_numpy(speed),
  ).numpy()
  lower_tau, upper_tau = compute_torque_speed_limits(velocity, saturation, speed)
  deployment_target = np.clip(
    target,
    q + (lower_tau + kd * velocity) / kp,
    q + (upper_tau + kd * velocity) / kp,
  )
  np.testing.assert_allclose(clipped, deployment_target, atol=1e-12)

  # Independently specified paper envelope for these seven normalized speeds.
  paper_lower = np.array([0.0, 0.0, -0.6, -1.0, -1.0, -1.0, -1.0])[:, None] * saturation
  paper_upper = np.array([1.0, 1.0, 1.0, 1.0, 0.6, 0.0, 0.0])[:, None] * saturation
  expected_torque = np.clip(nominal_torque, paper_lower, paper_upper)
  actual_torque = kp * (clipped - q) - kd * velocity
  np.testing.assert_allclose(actual_torque, expected_torque, atol=1e-12)
  np.testing.assert_allclose(upper_tau, paper_upper, atol=1e-12)
  np.testing.assert_allclose(lower_tau, paper_lower, atol=1e-12)


def test_training_holds_clipped_target_until_next_control_step() -> None:
  # Exercise the real action methods without constructing an environment.
  action: Any = object.__new__(TorqueSpeedClippedJointPositionAction)
  action.cfg = SimpleNamespace(clip=None)
  action._raw_actions = torch.zeros((1, 1))
  action._processed_actions = torch.zeros((1, 1))
  action._last_clipped_target = torch.zeros((1, 1))
  action._scale = 1.0
  action._offset = 0.0
  action._target_ids = torch.tensor([0])
  action._saturation_effort = torch.tensor([[139.0]])
  action._velocity_limit = torch.tensor([[20.0]])
  action._current_pd_gains = Mock(
    return_value=(torch.tensor([[99.09843]]), torch.tensor([[6.30880]]))
  )
  action._entity = SimpleNamespace(
    data=SimpleNamespace(
      joint_pos=torch.zeros((1, 1)),
      joint_vel=torch.tensor([[24.0]]),
      encoder_bias=torch.zeros((1, 1)),
    ),
    set_joint_position_target=Mock(),
  )

  action.process_actions(torch.tensor([[0.01]]))
  first_target = action._last_clipped_target.clone()
  action.apply_actions()
  # Physics state can change between control ticks; the commanded target holds.
  action._entity.data.joint_vel.fill_(-24.0)
  action.apply_actions()
  for call in action._entity.set_joint_position_target.call_args_list:
    torch.testing.assert_close(call.args[0], first_target)
  action._current_pd_gains.assert_called_once()

  action.process_actions(torch.tensor([[0.01]]))
  assert not torch.equal(action._last_clipped_target, first_target)
