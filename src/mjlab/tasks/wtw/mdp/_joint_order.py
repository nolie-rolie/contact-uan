"""Isaac Lab <-> MuJoCo joint order permutation for Unitree G1.

MuJoCo XML order (kinematic chain): left leg, right leg, waist, left arm,
right arm.  Isaac Lab order: interleaved left/right by joint type.

ISAACLAB_TO_MUJOCO[mj_idx] = isaaclab_idx
    Usage: joint_pos_mujoco = joint_pos_isaaclab[ISAACLAB_TO_MUJOCO]

MUJOCO_TO_ISAACLAB[il_idx] = mj_idx
    Usage: joint_pos_isaaclab = joint_pos_mujoco[MUJOCO_TO_ISAACLAB]
"""

import numpy as np

ISAACLAB_TO_MUJOCO = np.array(
  [
    0,
    3,
    6,
    9,
    13,
    17,  # left leg
    1,
    4,
    7,
    10,
    14,
    18,  # right leg
    2,
    5,
    8,  # waist
    11,
    15,
    19,
    21,
    23,
    25,
    27,  # left arm
    12,
    16,
    20,
    22,
    24,
    26,
    28,  # right arm
  ]
)

MUJOCO_TO_ISAACLAB = np.argsort(ISAACLAB_TO_MUJOCO)
