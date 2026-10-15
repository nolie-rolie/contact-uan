"""Verify the fixed bridge gain contract without any LCM transport or robot."""

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from deploy.robots.g1_config import G1Config
from deploy.utils.lcm_bridge_config import (
  load_lcm_bridge_pd_gains,
  validate_lcm_pd_gains,
)

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ARTICULATION


def _canonical_gains():
  names = G1Config(dt=0.02).joint_names("hw")
  gains = []
  for field in ("stiffness", "damping"):
    values = []
    for name in names:
      matches = [
        getattr(actuator, field)
        for actuator in G1_ARTICULATION.actuators
        if any(re.fullmatch(pattern, name) for pattern in actuator.target_names_expr)
      ]
      assert len(matches) == 1
      values.append(matches[0])
    gains.append(np.asarray(values))
  return names, gains[0], gains[1]


def test_bridge_source_gains_match_all_nominal_joints_at_wire_precision():
  names, stiffnesses, dampings = _canonical_gains()
  bridge_stiffnesses, bridge_dampings = load_lcm_bridge_pd_gains()
  assert len(names) == 29
  np.testing.assert_array_equal(bridge_stiffnesses, stiffnesses.astype(np.float32))
  np.testing.assert_array_equal(bridge_dampings, dampings.astype(np.float32))
  validate_lcm_pd_gains(stiffnesses, dampings, names)


@pytest.mark.parametrize("field", ["Kp", "Kd"])
def test_gain_override_is_rejected_with_joint_name(field):
  names, stiffnesses, dampings = _canonical_gains()
  values = stiffnesses if field == "Kp" else dampings
  values[0] *= 1.1
  with pytest.raises(ValueError, match=f"{field}.*{names[0]}.*Gain overrides"):
    validate_lcm_pd_gains(stiffnesses, dampings, names)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_gains_are_rejected(value):
  names, stiffnesses, dampings = _canonical_gains()
  stiffnesses[0] = value
  with pytest.raises(ValueError, match="finite positive gains"):
    validate_lcm_pd_gains(stiffnesses, dampings, names)


def test_unreadable_or_changed_bridge_source_is_rejected(tmp_path):
  path = tmp_path / "g1_control.cpp"
  with pytest.raises(ValueError, match="Cannot verify fixed LCM bridge gains"):
    load_lcm_bridge_pd_gains(path)
  path.write_text("const int G1_NUM_MOTOR = 29;\n")
  with pytest.raises(ValueError, match="missing fixed Kp array"):
    load_lcm_bridge_pd_gains(path)


def test_interface_rejects_gain_mismatch_before_constructing_transport(monkeypatch):
  names, stiffnesses, dampings = _canonical_gains()
  stiffnesses[0] *= 1.1
  repo = Path(__file__).resolve().parents[1]
  monkeypatch.syspath_prepend(
    str(repo / "third-party/openhomie/HomieDeploy/unitree_sdk2")
  )
  # This fake module cannot open a socket; transport construction is forbidden.
  lcm_stub = ModuleType("lcm")
  create_transport = Mock(side_effect=AssertionError("LCM must not be constructed"))
  lcm_stub.__dict__["LCM"] = create_transport
  monkeypatch.setitem(sys.modules, "lcm", lcm_stub)
  spec = importlib.util.spec_from_file_location(
    "_lcm_interface_guard_test", repo / "deploy/interfaces/lcm_interface.py"
  )
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  robot_configuration = SimpleNamespace(
    joint_stiffnesses=Mock(return_value=stiffnesses),
    joint_dampings=Mock(return_value=dampings),
    joint_names=Mock(return_value=names),
  )
  with pytest.raises(ValueError, match="Gain overrides are unsupported"):
    module.G1LCMInterface(robot_configuration, {}, "offline-test")
  create_transport.assert_not_called()
