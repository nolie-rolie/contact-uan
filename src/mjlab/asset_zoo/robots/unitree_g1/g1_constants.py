"""Unitree G1 constants."""

import re
from collections.abc import Sequence
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

G1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
)
assert G1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  asset_dir = G1_XML.parent / "assets"
  for path in asset_dir.glob("*"):
    if path.is_file():
      assets[f"{meshdir}/{path.name}" if meshdir else path.name] = path.read_bytes()
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Motor specs (from Unitree).
ROTOR_INERTIAS_5020 = (
  0.139e-4,
  0.017e-4,
  0.169e-4,
)
GEARS_5020 = (
  1,
  1 + (46 / 18),
  1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (
  0.489e-4,
  0.098e-4,
  0.533e-4,
)
GEARS_7520_14 = (
  1,
  4.5,
  1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (
  0.489e-4,
  0.109e-4,
  0.738e-4,
)
GEARS_7520_22 = (
  1,
  4.5,
  5,
)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (
  0.068e-4,
  0.0,
  0.0,
)
GEARS_4010 = (
  1,
  5,
  5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

ACTUATOR_5020 = ElectricActuator(
  reflected_inertia=ARMATURE_5020,
  velocity_limit=37.0,
  effort_limit=25.0,
)
ACTUATOR_7520_14 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=32.0,
  effort_limit=88.0,
)
ACTUATOR_7520_22 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_22,
  velocity_limit=20.0,
  effort_limit=139.0,
)
ACTUATOR_4010 = ElectricActuator(
  reflected_inertia=ARMATURE_4010,
  velocity_limit=22.0,
  effort_limit=5.0,
)

G1_TORQUE_SPEED_SATURATION_EFFORT: dict[str, float] = {
  ".*_elbow_joint": ACTUATOR_5020.effort_limit,
  ".*_shoulder_pitch_joint": ACTUATOR_5020.effort_limit,
  ".*_shoulder_roll_joint": ACTUATOR_5020.effort_limit,
  ".*_shoulder_yaw_joint": ACTUATOR_5020.effort_limit,
  ".*_wrist_roll_joint": ACTUATOR_5020.effort_limit,
  ".*_hip_pitch_joint": ACTUATOR_7520_14.effort_limit,
  ".*_hip_yaw_joint": ACTUATOR_7520_14.effort_limit,
  "waist_yaw_joint": ACTUATOR_7520_14.effort_limit,
  ".*_hip_roll_joint": ACTUATOR_7520_22.effort_limit,
  ".*_knee_joint": ACTUATOR_7520_22.effort_limit,
  ".*_wrist_pitch_joint": ACTUATOR_4010.effort_limit,
  ".*_wrist_yaw_joint": ACTUATOR_4010.effort_limit,
  "waist_pitch_joint": ACTUATOR_5020.effort_limit * 2,
  "waist_roll_joint": ACTUATOR_5020.effort_limit * 2,
  ".*_ankle_pitch_joint": ACTUATOR_5020.effort_limit * 2,
  ".*_ankle_roll_joint": ACTUATOR_5020.effort_limit * 2,
}

G1_TORQUE_SPEED_VELOCITY_LIMIT: dict[str, float] = {
  ".*_elbow_joint": ACTUATOR_5020.velocity_limit,
  ".*_shoulder_pitch_joint": ACTUATOR_5020.velocity_limit,
  ".*_shoulder_roll_joint": ACTUATOR_5020.velocity_limit,
  ".*_shoulder_yaw_joint": ACTUATOR_5020.velocity_limit,
  ".*_wrist_roll_joint": ACTUATOR_5020.velocity_limit,
  ".*_hip_pitch_joint": ACTUATOR_7520_14.velocity_limit,
  ".*_hip_yaw_joint": ACTUATOR_7520_14.velocity_limit,
  "waist_yaw_joint": ACTUATOR_7520_14.velocity_limit,
  ".*_hip_roll_joint": ACTUATOR_7520_22.velocity_limit,
  ".*_knee_joint": ACTUATOR_7520_22.velocity_limit,
  ".*_wrist_pitch_joint": ACTUATOR_4010.velocity_limit,
  ".*_wrist_yaw_joint": ACTUATOR_4010.velocity_limit,
  "waist_pitch_joint": ACTUATOR_5020.velocity_limit,
  "waist_roll_joint": ACTUATOR_5020.velocity_limit,
  ".*_ankle_pitch_joint": ACTUATOR_5020.velocity_limit,
  ".*_ankle_roll_joint": ACTUATOR_5020.velocity_limit,
}

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

G1_ACTUATOR_5020 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)
G1_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
G1_ACTUATOR_4010 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=STIFFNESS_4010,
  damping=DAMPING_4010,
  effort_limit=ACTUATOR_4010.effort_limit,
  armature=ACTUATOR_4010.reflected_inertia,
)

# Waist pitch/roll and ankles are 4-bar linkages with 2 5020 actuators.
# Due to the parallel linkage, the effective armature at the ankle and waist joints
# is configuration dependent. Since the exact geometry of the linkage is unknown, we
# assume a nominal 1:1 gear ratio. Under this assumption, the joint armature in the
# nominal configuration is approximated as the sum of the 2 actuators' armatures.
G1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)
G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)

G1_UAN_ACTUATOR_GROUP_NAMES = (
  "g1_5020_arms",
  "g1_7520_14",
  "g1_7520_22",
  "g1_4010",
  "g1_5020_doubled_linkage",
)

# Native MuJoCo order: left leg, right leg, waist, left arm, right arm.
G1_UAN_JOINT_TO_ACTUATOR_MAP = (
  1,
  2,
  1,
  2,
  4,
  4,
  1,
  2,
  1,
  2,
  4,
  4,
  1,
  4,
  4,
  0,
  0,
  0,
  0,
  0,
  3,
  3,
  0,
  0,
  0,
  0,
  0,
  3,
  3,
)

G1_UAN_ACTUATOR_GROUP_PATTERNS = (
  G1_ACTUATOR_5020.target_names_expr,
  G1_ACTUATOR_7520_14.target_names_expr,
  G1_ACTUATOR_7520_22.target_names_expr,
  G1_ACTUATOR_4010.target_names_expr,
  G1_ACTUATOR_WAIST.target_names_expr + G1_ACTUATOR_ANKLE.target_names_expr,
)


def build_g1_joint_to_actuator_map(
  joint_names: Sequence[str],
  num_actuators: int,
) -> tuple[list[int], str, tuple[str, ...]]:
  """Build the five actuator-family groups used by Contact-UAN."""
  joint_names = tuple(joint_names)
  if len(joint_names) != 29:
    raise ValueError(f"Expected 29 G1 joints, got {len(joint_names)}: {joint_names}")
  if num_actuators != 5:
    raise ValueError("Contact-UAN requires five actuator-family networks.")

  joint_to_group: list[int] = []
  unmatched: list[str] = []
  ambiguous: list[tuple[str, list[int]]] = []
  for joint_name in joint_names:
    matches = [
      group_idx
      for group_idx, patterns in enumerate(G1_UAN_ACTUATOR_GROUP_PATTERNS)
      if any(re.fullmatch(pattern, joint_name) for pattern in patterns)
    ]
    if len(matches) == 1:
      joint_to_group.append(matches[0])
    elif len(matches) == 0:
      unmatched.append(joint_name)
      joint_to_group.append(-1)
    else:
      ambiguous.append((joint_name, matches))
      joint_to_group.append(matches[0])

  if unmatched or ambiguous:
    raise ValueError(
      "Failed to build 5-group G1 joint_to_actuator_map. "
      f"unmatched={unmatched}, ambiguous={ambiguous}"
    )

  group_ids = set(joint_to_group)
  expected_ids = set(range(5))
  if group_ids != expected_ids:
    raise ValueError(
      f"Expected all 5 actuator groups to be used, got {sorted(group_ids)}"
    )

  return joint_to_group, "g1_actuator_family_5", G1_UAN_ACTUATOR_GROUP_NAMES


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.783675),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 1.28,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.76),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_robot_cfg() -> EntityCfg:
  """Get a fresh G1 robot configuration instance (upright pose)."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION,
  )


G1_ACTION_SCALE: dict[str, float] = {}
for a in G1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    G1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_robot_cfg())

  viewer.launch(robot.spec.compile())
