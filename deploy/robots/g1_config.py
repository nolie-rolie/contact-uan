import os
import re

import numpy as np

_G1_TORQUE_SPEED_SATURATION_EFFORT_BY_REGEX = {
  ".*_elbow_joint": 25.0,
  ".*_shoulder_pitch_joint": 25.0,
  ".*_shoulder_roll_joint": 25.0,
  ".*_shoulder_yaw_joint": 25.0,
  ".*_wrist_roll_joint": 25.0,
  ".*_hip_pitch_joint": 88.0,
  ".*_hip_yaw_joint": 88.0,
  "waist_yaw_joint": 88.0,
  ".*_hip_roll_joint": 139.0,
  ".*_knee_joint": 139.0,
  ".*_wrist_pitch_joint": 5.0,
  ".*_wrist_yaw_joint": 5.0,
  "waist_pitch_joint": 50.0,
  "waist_roll_joint": 50.0,
  ".*_ankle_pitch_joint": 50.0,
  ".*_ankle_roll_joint": 50.0,
}

_G1_TORQUE_SPEED_VELOCITY_LIMIT_BY_REGEX = {
  ".*_elbow_joint": 37.0,
  ".*_shoulder_pitch_joint": 37.0,
  ".*_shoulder_roll_joint": 37.0,
  ".*_shoulder_yaw_joint": 37.0,
  ".*_wrist_roll_joint": 37.0,
  ".*_hip_pitch_joint": 32.0,
  ".*_hip_yaw_joint": 32.0,
  "waist_yaw_joint": 32.0,
  ".*_hip_roll_joint": 20.0,
  ".*_knee_joint": 20.0,
  ".*_wrist_pitch_joint": 22.0,
  ".*_wrist_yaw_joint": 22.0,
  "waist_pitch_joint": 37.0,
  "waist_roll_joint": 37.0,
  ".*_ankle_pitch_joint": 37.0,
  ".*_ankle_roll_joint": 37.0,
}


def _expand_joint_regex_values(joint_names, values_by_regex):
  resolved = {}
  for joint_name in joint_names:
    matches = [
      value
      for joint_regex, value in values_by_regex.items()
      if re.fullmatch(joint_regex, joint_name)
    ]
    if len(matches) != 1:
      raise ValueError(
        f"Expected exactly one torque-speed match for {joint_name}, got {matches}"
      )
    resolved[joint_name] = matches[0]
  return resolved


def _resolve_motor_parameter(
  value: float | dict[str, float], joint_names: list[str], field_name: str
) -> dict[str, float]:
  """Resolve a saved action limit without importing the training framework."""
  if isinstance(value, (int, float)):
    resolved = dict.fromkeys(joint_names, float(value))
  elif isinstance(value, dict):
    resolved = {
      name: float(item)
      for name, item in _expand_joint_regex_values(joint_names, value).items()
    }
  else:
    raise TypeError(f"{field_name} must be a scalar or a joint-regex mapping.")
  if any(not np.isfinite(item) or item <= 0.0 for item in resolved.values()):
    raise ValueError(
      f"{field_name} must be finite and strictly positive for every joint."
    )
  return resolved


class G1Config:
  num_joints = num_joints_hw = 29

  _joint_map_isaaclab = [
    "left_hip_pitch_joint",  # 0
    "right_hip_pitch_joint",  # 1
    "waist_yaw_joint",  # 2
    "left_hip_roll_joint",  # 3
    "right_hip_roll_joint",  # 4
    "waist_roll_joint",  # 5
    "left_hip_yaw_joint",  # 6
    "right_hip_yaw_joint",  # 7
    "waist_pitch_joint",  # 8
    "left_knee_joint",  # 9
    "right_knee_joint",  # 10
    "left_shoulder_pitch_joint",  # 11
    "right_shoulder_pitch_joint",  # 12
    "left_ankle_pitch_joint",  # 13
    "right_ankle_pitch_joint",  # 14
    "left_shoulder_roll_joint",  # 15
    "right_shoulder_roll_joint",  # 16
    "left_ankle_roll_joint",  # 17
    "right_ankle_roll_joint",  # 18
    "left_shoulder_yaw_joint",  # 19
    "right_shoulder_yaw_joint",  # 20
    "left_elbow_joint",  # 21
    "right_elbow_joint",  # 22
    "left_wrist_roll_joint",  # 23
    "right_wrist_roll_joint",  # 24
    "left_wrist_pitch_joint",  # 25
    "right_wrist_pitch_joint",  # 26
    "left_wrist_yaw_joint",  # 27
    "right_wrist_yaw_joint",  # 28
  ]
  _joint_map_hw = [
    "left_hip_pitch_joint",  # 0
    "left_hip_roll_joint",  # 1
    "left_hip_yaw_joint",  # 2
    "left_knee_joint",  # 3
    "left_ankle_pitch_joint",  # 4
    "left_ankle_roll_joint",  # 5
    "right_hip_pitch_joint",  # 6
    "right_hip_roll_joint",  # 7
    "right_hip_yaw_joint",  # 8
    "right_knee_joint",  # 9
    "right_ankle_pitch_joint",  # 10
    "right_ankle_roll_joint",  # 11
    "waist_yaw_joint",  # 12
    "waist_roll_joint",  # 13
    "waist_pitch_joint",  # 14
    "left_shoulder_pitch_joint",  # 15
    "left_shoulder_roll_joint",  # 16
    "left_shoulder_yaw_joint",  # 17
    "left_elbow_joint",  # 18
    "left_wrist_roll_joint",  # 19
    "left_wrist_pitch_joint",  # 20
    "left_wrist_yaw_joint",  # 21
    "right_shoulder_pitch_joint",  # 22
    "right_shoulder_roll_joint",  # 23
    "right_shoulder_yaw_joint",  # 24
    "right_elbow_joint",  # 25
    "right_wrist_roll_joint",  # 26
    "right_wrist_pitch_joint",  # 27
    "right_wrist_yaw_joint",  # 28
  ]

  _joint_map_mujoco = [
    "left_hip_pitch_joint",  # 0
    "left_hip_roll_joint",  # 1
    "left_hip_yaw_joint",  # 2
    "left_knee_joint",  # 3
    "left_ankle_pitch_joint",  # 4
    "left_ankle_roll_joint",  # 5
    "right_hip_pitch_joint",  # 6
    "right_hip_roll_joint",  # 7
    "right_hip_yaw_joint",  # 8
    "right_knee_joint",  # 9
    "right_ankle_pitch_joint",  # 10
    "right_ankle_roll_joint",  # 11
    "waist_yaw_joint",  # 12
    "waist_roll_joint",  # 13
    "waist_pitch_joint",  # 14
    "left_shoulder_pitch_joint",  # 15
    "left_shoulder_roll_joint",  # 16
    "left_shoulder_yaw_joint",  # 17
    "left_elbow_joint",  # 18
    "left_wrist_roll_joint",  # 19
    "left_wrist_pitch_joint",  # 20
    "left_wrist_yaw_joint",  # 21
    "right_shoulder_pitch_joint",  # 22
    "right_shoulder_roll_joint",  # 23
    "right_shoulder_yaw_joint",  # 24
    "right_elbow_joint",  # 25
    "right_wrist_roll_joint",  # 26
    "right_wrist_pitch_joint",  # 27
    "right_wrist_yaw_joint",  # 28
  ]

  _joint_default_positions = {
    "left_hip_pitch_joint": -0.20,
    "right_hip_pitch_joint": -0.20,
    "left_knee_joint": 0.42,
    "right_knee_joint": 0.42,
    "left_ankle_pitch_joint": -0.23,
    "right_ankle_pitch_joint": -0.23,
    "left_elbow_joint": 0.87,
    "right_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.16,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.16,
    "right_shoulder_pitch_joint": 0.35,
  }
  for joint in _joint_map_hw:
    if joint not in _joint_default_positions:
      _joint_default_positions[joint] = 0

  _joint_stiffnesses = {
    "left_hip_pitch_joint": 200,  # 0
    "left_hip_roll_joint": 150,  # 1
    "left_hip_yaw_joint": 150,  # 2
    "left_knee_joint": 200,  # 3
    "left_ankle_pitch_joint": 20,  # 4
    "left_ankle_roll_joint": 20,  # 5
    "right_hip_pitch_joint": 200,  # 6
    "right_hip_roll_joint": 150,  # 7
    "right_hip_yaw_joint": 150,  # 8
    "right_knee_joint": 200,  # 9
    "right_ankle_pitch_joint": 50,  # 10
    "right_ankle_roll_joint": 50,  # 11
    "waist_yaw_joint": 200,  # 12
    "waist_roll_joint": 200,  # 13
    "waist_pitch_joint": 200,  # 14
    "left_shoulder_pitch_joint": 40,  # 15
    "left_shoulder_roll_joint": 40,  # 16
    "left_shoulder_yaw_joint": 40,  # 17
    "left_elbow_joint": 40,  # 18
    "left_wrist_roll_joint": 5,  # 19
    "left_wrist_pitch_joint": 5,  # 20
    "left_wrist_yaw_joint": 5,  # 21
    "right_shoulder_pitch_joint": 40,  # 22
    "right_shoulder_roll_joint": 40,  # 23
    "right_shoulder_yaw_joint": 40,  # 24
    "right_elbow_joint": 40,  # 25
    "right_wrist_roll_joint": 5,  # 26
    "right_wrist_pitch_joint": 5,  # 27
    "right_wrist_yaw_joint": 5,  # 28
  }
  _joint_dampings = {
    "left_hip_pitch_joint": 5,  # 0
    "left_hip_roll_joint": 5,  # 1
    "left_hip_yaw_joint": 5,  # 2
    "left_knee_joint": 5,  # 3
    "left_ankle_pitch_joint": 2,  # 4
    "left_ankle_roll_joint": 2,  # 5
    "right_hip_pitch_joint": 5,  # 6
    "right_hip_roll_joint": 5,  # 7
    "right_hip_yaw_joint": 5,  # 8
    "right_knee_joint": 5,  # 9
    "right_ankle_pitch_joint": 2,  # 10
    "right_ankle_roll_joint": 2,  # 11
    "waist_yaw_joint": 5,  # 12
    "waist_roll_joint": 5,  # 13
    "waist_pitch_joint": 5,  # 14
    "left_shoulder_pitch_joint": 10,  # 15
    "left_shoulder_roll_joint": 10,  # 16
    "left_shoulder_yaw_joint": 10,  # 17
    "left_elbow_joint": 10,  # 18
    "left_wrist_roll_joint": 1,  # 19
    "left_wrist_pitch_joint": 1,  # 20
    "left_wrist_yaw_joint": 1,  # 21
    "right_shoulder_pitch_joint": 10,  # 22
    "right_shoulder_roll_joint": 10,  # 23
    "right_shoulder_yaw_joint": 10,  # 24
    "right_elbow_joint": 10,  # 25
    "right_wrist_roll_joint": 1,  # 26
    "right_wrist_pitch_joint": 1,  # 27
    "right_wrist_yaw_joint": 1,  # 28
  }
  _joint_max_velocities = _expand_joint_regex_values(
    _joint_map_isaaclab,
    _G1_TORQUE_SPEED_VELOCITY_LIMIT_BY_REGEX,
  )

  _joint_saturated_torques = _expand_joint_regex_values(
    _joint_map_isaaclab,
    _G1_TORQUE_SPEED_SATURATION_EFFORT_BY_REGEX,
  )

  _joint_lower_limits = {
    "left_hip_pitch_joint": -2.53,  # 0
    "left_hip_roll_joint": -0.52,  # 1
    "left_hip_yaw_joint": -2.75,  # 2
    "left_knee_joint": -0.08,  # 3
    "left_ankle_pitch_joint": -0.87,  # 4
    "left_ankle_roll_joint": -0.26,  # 5
    "right_hip_pitch_joint": -2.53,  # 6
    "right_hip_roll_joint": -2.96,  # 7
    "right_hip_yaw_joint": -2.75,  # 8
    "right_knee_joint": -0.087,  # 9
    "right_ankle_pitch_joint": -0.87,  # 10
    "right_ankle_roll_joint": -0.26,  # 11
    "waist_yaw_joint": -2.61,  # 12
    "waist_roll_joint": -0.52,  # 13
    "waist_pitch_joint": -0.52,  # 14
    "left_shoulder_pitch_joint": -3.08,  # 15
    "left_shoulder_roll_joint": -1.58,  # 16
    "left_shoulder_yaw_joint": -2.61,  # 17
    "left_elbow_joint": -1.04,  # 18
    "left_wrist_roll_joint": -1.97,  # 19
    "left_wrist_pitch_joint": -1.61,  # 20
    "left_wrist_yaw_joint": -1.61,  # 21
    "right_shoulder_pitch_joint": -3.08,  # 22
    "right_shoulder_roll_joint": -2.25,  # 23
    "right_shoulder_yaw_joint": -2.61,  # 24
    "right_elbow_joint": -1.04,  # 25
    "right_wrist_roll_joint": -1.97,  # 26
    "right_wrist_pitch_joint": -1.61,  # 27
    "right_wrist_yaw_joint": -1.61,  # 28
  }

  _joint_upper_limits = {
    "left_hip_pitch_joint": 2.87,  # 0
    "left_hip_roll_joint": 2.96,  # 1
    "left_hip_yaw_joint": 2.75,  # 2
    "left_knee_joint": 2.87,  # 3
    "left_ankle_pitch_joint": 0.52,  # 4
    "left_ankle_roll_joint": 0.26,  # 5
    "right_hip_pitch_joint": 2.87,  # 6
    "right_hip_roll_joint": 0.52,  # 7
    "right_hip_yaw_joint": 2.75,  # 8
    "right_knee_joint": 2.87,  # 9
    "right_ankle_pitch_joint": 0.52,  # 10
    "right_ankle_roll_joint": 0.26,  # 11
    "waist_yaw_joint": 2.61,  # 12
    "waist_roll_joint": 0.52,  # 13
    "waist_pitch_joint": 0.52,  # 14
    "left_shoulder_pitch_joint": 2.67,  # 15
    "left_shoulder_roll_joint": 2.25,  # 16
    "left_shoulder_yaw_joint": 2.61,  # 17
    "left_elbow_joint": 2.09,  # 18
    "left_wrist_roll_joint": 1.97,  # 19
    "left_wrist_pitch_joint": 1.61,  # 20
    "left_wrist_yaw_joint": 1.61,  # 21
    "right_shoulder_pitch_joint": 2.67,  # 22
    "right_shoulder_roll_joint": 1.58,  # 23
    "right_shoulder_yaw_joint": 2.61,  # 24
    "right_elbow_joint": 2.09,  # 25
    "right_wrist_roll_joint": 1.97,  # 26
    "right_wrist_pitch_joint": 1.61,  # 27
    "right_wrist_yaw_joint": 1.61,  # 28
  }

  _mujoco_xml_paths = {
    "default": f"{os.path.dirname(os.path.realpath(__file__))}/../assets/g1/g1_29dof.xml",
  }

  def __init__(self, dt):
    self.dt = dt

  def load_from_cfg(self, cfg):
    import re

    self._joint_stiffnesses = {k: None for k in self._joint_map_isaaclab}
    self._joint_dampings = {k: None for k in self._joint_map_isaaclab}
    self._joint_default_positions = dict(type(self)._joint_default_positions)
    self._joint_saturated_torques = _resolve_motor_parameter(
      cfg["control"].get("saturation_effort", type(self)._joint_saturated_torques),
      self._joint_map_isaaclab,
      "saturation_effort",
    )
    self._joint_max_velocities = _resolve_motor_parameter(
      cfg["control"].get("velocity_limit", type(self)._joint_max_velocities),
      self._joint_map_isaaclab,
      "velocity_limit",
    )

    for joint_names_expr, stiffness in cfg["control"]["stiffnesses"].items():
      # regex to match joint names
      for joint_name in self._joint_map_isaaclab:
        if re.match(joint_names_expr, joint_name):
          self._joint_stiffnesses[joint_name] = stiffness
    for joint_names_expr, damping in cfg["control"]["dampings"].items():
      for joint_name in self._joint_map_isaaclab:
        if re.match(joint_names_expr, joint_name):
          self._joint_dampings[joint_name] = damping

    print("Loaded joint stiffnesses and dampings:")
    print(self._joint_stiffnesses)
    print(self._joint_dampings)

    # ensure all joints have stiffness and damping values
    for joint_name in self._joint_map_isaaclab:
      if self._joint_stiffnesses[joint_name] is None:
        raise ValueError(f"Missing stiffness value for joint {joint_name}")
      if self._joint_dampings[joint_name] is None:
        raise ValueError(f"Missing damping value for joint {joint_name}")

    # Override default joint positions from training config if available
    if "default_joint_positions" in cfg["control"]:
      for joint_names_expr, value in cfg["control"]["default_joint_positions"].items():
        for joint_name in self._joint_map_isaaclab:
          if re.match(joint_names_expr, joint_name):
            self._joint_default_positions[joint_name] = value
      print("Loaded default joint positions from training config:")
      print({k: v for k, v in self._joint_default_positions.items() if v != 0.0})

  def _joint_param(self, dict, joint_order):
    # warning, it might be slow to do this every time
    if joint_order == "isaaclab":
      return np.array([dict[joint] for joint in self._joint_map_isaaclab])
    elif joint_order == "hw":
      return np.array([dict[joint] for joint in self._joint_map_hw])
    elif joint_order == "mujoco":
      return np.array([dict[joint] for joint in self._joint_map_mujoco])
    else:
      raise ValueError("Unknown joint_order")

  def joint_names(self, joint_order):
    if joint_order == "isaaclab":
      return self._joint_map_isaaclab
    elif joint_order == "hw":
      return self._joint_map_hw
    elif joint_order == "mujoco":
      return self._joint_map_mujoco
    else:
      raise ValueError("Unknown joint_order")

  def joint_default_positions(self, joint_order):
    return self._joint_param(self._joint_default_positions, joint_order)

  def joint_stiffnesses(self, joint_order):
    return self._joint_param(self._joint_stiffnesses, joint_order)

  def joint_dampings(self, joint_order):
    return self._joint_param(self._joint_dampings, joint_order)

  def joint_max_velocities(self, joint_order):
    return self._joint_param(self._joint_max_velocities, joint_order)

  def joint_saturated_torques(self, joint_order):
    return self._joint_param(self._joint_saturated_torques, joint_order)

  def joint_lower_limits(self, joint_order):
    return self._joint_param(self._joint_lower_limits, joint_order)

  def joint_upper_limits(self, joint_order):
    return self._joint_param(self._joint_upper_limits, joint_order)

  def compute_reindex_mapping(self, from_order, to_order):
    if from_order == to_order:
      return list(range(self.num_joints))
    else:
      return [
        self.joint_names(from_order).index(joint)
        for joint in self.joint_names(to_order)
      ]

  def remap_joint_array(self, joint_array, from_order, to_order):
    assert len(joint_array) == self.num_joints, "joint_array must have 29 elements"
    mapping = self.compute_reindex_mapping(from_order, to_order)
    return np.array([joint_array[i] for i in mapping])


if __name__ == "__main__":
  g1_config = G1Config(0.005)

  # print all the remappings

  print("isaaclab -> hw")
  print(g1_config.compute_reindex_mapping("isaaclab", "hw"))
  print("hw -> isaaclab")
  print(g1_config.compute_reindex_mapping("hw", "isaaclab"))
  print("mujoco -> hw")
  print(g1_config.compute_reindex_mapping("mujoco", "hw"))
  print("hw -> mujoco")
  print(g1_config.compute_reindex_mapping("hw", "mujoco"))
  print("isaaclab -> mujoco")
  print(g1_config.compute_reindex_mapping("isaaclab", "mujoco"))
  print("mujoco -> isaaclab")
  print(g1_config.compute_reindex_mapping("mujoco", "isaaclab"))
