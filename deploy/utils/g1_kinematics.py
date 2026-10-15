"""Headless MuJoCo FK helper for G1 foot heights and base height.

Uses the G1 MuJoCo model to compute foot positions and approximate
foot z-velocities from joint positions alone (no rendering or full sim).
No new dependencies — MuJoCo is already required by the deploy package.
"""

import os

import mujoco
import numpy as np

# Body names used to read foot positions from MuJoCo data.xpos
_LEFT_FOOT_BODY = "left_ankle_roll_link"
_RIGHT_FOOT_BODY = "right_ankle_roll_link"
_BASE_BODY = "pelvis"

# Default XML path relative to this file
_DEFAULT_XML = os.path.join(
  os.path.dirname(os.path.realpath(__file__)),
  "../assets/g1/g1_29dof.xml",
)


class G1KinematicsHelper:
  """Headless MuJoCo FK for G1 foot heights and base height.

  Loads the G1 MuJoCo model and, on each call to ``compute()``, sets
  the free-joint base pose and joint positions then runs
  ``mj_kinematics`` to compute body positions without a full physics
  step.

  Args:
      xml_path: Path to the G1 MuJoCo XML (defaults to the bundled
          ``g1_29dof.xml``).
  """

  def __init__(self, xml_path: str | None = None):
    if xml_path is None:
      xml_path = _DEFAULT_XML
    self.model = mujoco.MjModel.from_xml_path(xml_path)
    self.data = mujoco.MjData(self.model)

    # Cache body IDs for fast lookup
    self._left_foot_id = self.model.body(_LEFT_FOOT_BODY).id
    self._right_foot_id = self.model.body(_RIGHT_FOOT_BODY).id
    self._base_id = self.model.body(_BASE_BODY).id

    # Cache qpos addresses for the robot's 29 actuated joints.
    self._nq_free = 7  # free joint: 3 pos + 4 quat
    self._joint_qpos_addrs = []
    for joint_idx in range(self.model.njnt):
      joint_name = self.model.joint(joint_idx).name
      if joint_name is None:
        continue
      if joint_name == "floating_base_joint":
        continue
      self._joint_qpos_addrs.append(int(self.model.jnt_qposadr[joint_idx]))
    self._nj = len(self._joint_qpos_addrs)

    # Saved previous foot z positions for finite-difference velocity
    self._prev_foot_z = np.zeros(2)
    self._prev_dt_valid = False

  def compute(
    self,
    joint_pos_hw: np.ndarray,
    base_height: float,
    dt: float | None = None,
  ) -> dict:
    """Run FK and return foot heights, foot z-velocities, and base height.

    Args:
        joint_pos_hw: Joint positions in hardware (mujoco) order, shape (29,).
        base_height: Base height estimate (m) used to set the free-joint z
            position. Typically from LCM ``p[2]``.
        dt: Control timestep in seconds, used to estimate foot z-velocity
            via finite differences. If None, foot_z_vel is returned as zeros.

    Returns:
        dict with keys:
            ``foot_heights`` — shape (2,), [left, right] heights above ground (m).
            ``foot_z_vel`` — shape (2,), [left, right] foot z-velocities (m/s).
                Zero if ``dt`` is None or on the first call.
            ``base_height`` — scalar, the provided base_height value.
    """
    if len(joint_pos_hw) != self._nj:
      raise ValueError(
        f"Expected {self._nj} hardware joints for FK, got {len(joint_pos_hw)}"
      )

    # Set free-joint position: place base at provided height, identity quat
    self.data.qpos[:] = 0.0
    self.data.qpos[:3] = [0.0, 0.0, base_height]
    self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z
    self.data.qpos[self._joint_qpos_addrs] = joint_pos_hw

    # Run kinematics only (no physics integration)
    mujoco.mj_kinematics(self.model, self.data)

    left_z = float(self.data.xpos[self._left_foot_id, 2])
    right_z = float(self.data.xpos[self._right_foot_id, 2])

    foot_heights = np.array([left_z, right_z])

    # Finite-difference foot z-velocity
    if dt is not None and dt > 0.0 and self._prev_dt_valid:
      foot_z_vel = (foot_heights - self._prev_foot_z) / dt
    else:
      foot_z_vel = np.zeros(2)

    self._prev_foot_z = foot_heights.copy()
    self._prev_dt_valid = True

    return {
      "foot_heights": foot_heights,
      "foot_z_vel": foot_z_vel,
      "base_height": base_height,
    }
