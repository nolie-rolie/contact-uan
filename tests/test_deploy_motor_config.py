"""Saved training motor limits resolve offline, without opening a robot interface."""

import numpy as np
import pytest
from deploy.base_controller import parse_hardware_cfg
from deploy.robots.g1_config import G1Config


def _saved_env_cfg(**action_limits):
  return {
    "actions": {"joint_pos": {"scale": 0.5, **action_limits}},
    "scene": {
      "entities": {
        "robot": {
          "articulation": {
            "actuators": [
              {"target_names_expr": [".*"], "stiffness": 40.0, "damping": 2.0}
            ]
          }
        }
      }
    },
  }


@pytest.mark.parametrize("mapped_field", ["saturation_effort", "velocity_limit"])
def test_saved_motor_limits_resolve_scalar_and_regex_in_each_joint_order(mapped_field):
  scalar_field = (
    "velocity_limit" if mapped_field == "saturation_effort" else "saturation_effort"
  )
  cfg = parse_hardware_cfg(
    _saved_env_cfg(
      **{
        mapped_field: {"left_.*": 11.0, "right_.*": 22.0, "waist_.*": 33.0},
        scalar_field: 17.0,
      }
    )
  )
  robot = G1Config(dt=cfg["control_dt"])
  robot.load_from_cfg(cfg)

  getters = {
    "saturation_effort": robot.joint_saturated_torques,
    "velocity_limit": robot.joint_max_velocities,
  }
  for order in ("hw", "mujoco", "isaaclab"):
    expected = [
      11.0 if name.startswith("left_") else 22.0 if name.startswith("right_") else 33.0
      for name in robot.joint_names(order)
    ]
    np.testing.assert_array_equal(getters[mapped_field](order), expected)
    np.testing.assert_array_equal(getters[scalar_field](order), np.full(29, 17.0))


def test_missing_motor_overrides_restore_defaults_without_changing_other_instances():
  untouched = G1Config(dt=0.02)
  default_torque = untouched.joint_saturated_torques("hw").copy()
  default_speed = untouched.joint_max_velocities("hw").copy()
  overridden = G1Config(dt=0.02)
  overridden.load_from_cfg(
    parse_hardware_cfg(_saved_env_cfg(saturation_effort=9.0, velocity_limit=13.0))
  )

  for robot in (untouched, G1Config(dt=0.02)):
    robot.load_from_cfg(parse_hardware_cfg(_saved_env_cfg()))
    np.testing.assert_array_equal(robot.joint_saturated_torques("hw"), default_torque)
    np.testing.assert_array_equal(robot.joint_max_velocities("hw"), default_speed)
  np.testing.assert_array_equal(
    overridden.joint_saturated_torques("hw"), np.full(29, 9.0)
  )
  np.testing.assert_array_equal(
    overridden.joint_max_velocities("hw"), np.full(29, 13.0)
  )

  overridden.load_from_cfg(parse_hardware_cfg(_saved_env_cfg()))
  np.testing.assert_array_equal(
    overridden.joint_saturated_torques("hw"), default_torque
  )
  np.testing.assert_array_equal(overridden.joint_max_velocities("hw"), default_speed)


@pytest.mark.parametrize("field", ["saturation_effort", "velocity_limit"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), {".*": float("nan")}])
def test_saved_motor_limits_reject_nonpositive_or_nonfinite_values(field, value):
  robot = G1Config(dt=0.02)
  with pytest.raises(ValueError, match=f"{field} must be finite and strictly positive"):
    robot.load_from_cfg(parse_hardware_cfg(_saved_env_cfg(**{field: value})))


@pytest.mark.parametrize("value", [{"left_.*": 10.0}, {".*": 10.0, "left_.*": 20.0}])
def test_saved_motor_limits_require_one_regex_match_per_joint(value):
  robot = G1Config(dt=0.02)
  with pytest.raises(ValueError, match="Expected exactly one torque-speed match"):
    robot.load_from_cfg(parse_hardware_cfg(_saved_env_cfg(velocity_limit=value)))
