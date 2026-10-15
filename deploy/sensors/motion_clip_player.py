"""Motion clip player for tracking-based deployment.

Loads a .npz motion clip, advances it one step per control tick, and exposes
reference joint angles/velocities and anchor body pose for the deploy sensors.
"""

import numpy as np


def _quat_wxyz_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
  """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
  trace = R[0, 0] + R[1, 1] + R[2, 2]
  if trace > 0:
    s = 0.5 / np.sqrt(trace + 1.0)
    w = 0.25 / s
    x = (R[2, 1] - R[1, 2]) * s
    y = (R[0, 2] - R[2, 0]) * s
    z = (R[1, 0] - R[0, 1]) * s
  elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
    s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
    w = (R[2, 1] - R[1, 2]) / s
    x = 0.25 * s
    y = (R[0, 1] + R[1, 0]) / s
    z = (R[0, 2] + R[2, 0]) / s
  elif R[1, 1] > R[2, 2]:
    s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
    w = (R[0, 2] - R[2, 0]) / s
    x = (R[0, 1] + R[1, 0]) / s
    y = 0.25 * s
    z = (R[1, 2] + R[2, 1]) / s
  else:
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    w = (R[1, 0] - R[0, 1]) / s
    x = (R[0, 2] + R[2, 0]) / s
    y = (R[1, 2] + R[2, 1]) / s
    z = 0.25 * s
  return np.array([w, x, y, z], dtype=np.float32)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
  """Conjugate of quaternion [w, x, y, z]."""
  return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def _yaw_from_quat(q: np.ndarray) -> float:
  """Heading (rotation about world +Z) of quaternion [w, x, y, z]."""
  w, x, y, z = q
  return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _quat_about_z(yaw: float) -> np.ndarray:
  """Quaternion [w, x, y, z] for a rotation of ``yaw`` about world +Z."""
  return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float32)


def _rot_z(yaw: float) -> np.ndarray:
  """3x3 rotation matrix for a rotation of ``yaw`` about world +Z."""
  c, s = np.cos(yaw), np.sin(yaw)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
  """Hamilton product q1 * q2, both [w, x, y, z]."""
  w1, x1, y1, z1 = q1
  w2, x2, y2, z2 = q2
  return np.array(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    dtype=np.float32,
  )


def _rotation_matrix_from_quat(q: np.ndarray) -> np.ndarray:
  """3x3 rotation matrix from quaternion [w, x, y, z]."""
  w, x, y, z = q
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ],
    dtype=np.float32,
  )


def torso_quat_from_base_and_waist(
  base_quat_wxyz: np.ndarray,
  waist_yaw: float,
  waist_roll: float,
  waist_pitch: float,
) -> np.ndarray:
  """G1 torso_link world orientation from the base orientation + waist encoders.

  R_torso = R_base @ Rz(waist_yaw) @ Rx(waist_roll) @ Ry(waist_pitch), matching
  the g1 waist chain pelvis -> waist_yaw(Z) -> waist_roll(X) -> torso(pitch,Y).
  Validated to machine precision against MuJoCo forward kinematics. Both the
  hardware (pelvis IMU) and mujoco (base qpos) interfaces use this so they
  compute the torso anchor orientation identically. Returns [w, x, y, z].
  """
  cy, sy = np.cos(waist_yaw), np.sin(waist_yaw)
  cr, sr = np.cos(waist_roll), np.sin(waist_roll)
  cp, sp = np.cos(waist_pitch), np.sin(waist_pitch)
  rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
  rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
  ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
  r_torso = _rotation_matrix_from_quat(base_quat_wxyz) @ rz @ rx @ ry
  return _quat_wxyz_from_rotation_matrix(r_torso).astype(np.float32)


# Full G1 body order as stored in the motion clip's body_pos_w / body_quat_w
# (the npz has all 30 bodies, matching the mjlab robot body_names ordering).
# The anchor index MUST index into this 30-body layout, not a subset.
_ANCHOR_BODY_NAMES = (
  "pelvis",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "waist_yaw_link",
  "waist_roll_link",
  "torso_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_link",
  "left_wrist_roll_link",
  "left_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
  "right_wrist_roll_link",
  "right_wrist_pitch_link",
  "right_wrist_yaw_link",
)
_ANCHOR_IDX = _ANCHOR_BODY_NAMES.index("torso_link")  # 15


class MotionClipPlayer:
  """Loads a .npz motion clip and replays it one step per call to advance().

  The clip is aligned to the robot's initial torso pose so that relative
  anchor errors are computed correctly regardless of where the robot stands.
  """

  def __init__(self, motion_file: str, pad_start: int = 0, pad_end: int = 0):
    data = np.load(motion_file)
    self.joint_pos = data["joint_pos"].astype(np.float32)  # (T, nj)
    self.joint_vel = data["joint_vel"].astype(np.float32)  # (T, nj)
    # body_pos_w / body_quat_w are stored for ALL bodies; we index anchor.
    self._body_pos_w = data["body_pos_w"].astype(np.float32)  # (T, B, 3)
    self._body_quat_w = data["body_quat_w"].astype(np.float32)  # (T, B, 4) [w,x,y,z]
    self._body_lin_vel_w = data["body_lin_vel_w"].astype(np.float32)
    self._body_ang_vel_w = data["body_ang_vel_w"].astype(np.float32)
    self.T = self.joint_pos.shape[0]
    self._pad_start = pad_start
    self._pad_end = pad_end
    self._total_len = pad_start + self.T + pad_end
    self._t = 0
    # Playback gate: hold on the current frame until triggered, then advance
    # once through the clip and hold on the final frame.
    self._playing = False
    # Rigid (yaw-only) transform mapping clip world frame -> robot world frame.
    # Identity until align_to_robot() is called.
    self._align_R = np.eye(3, dtype=np.float32)
    self._align_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    self._clip_anchor0_pos = np.zeros(3, dtype=np.float32)
    self._robot_anchor0_pos = np.zeros(3, dtype=np.float32)

  def align_to_robot(
    self,
    robot_anchor_pos: np.ndarray,
    robot_anchor_quat: np.ndarray,
  ) -> None:
    """Call once at startup with the robot's current torso pose.

    Builds a yaw-only rigid transform so the clip's first-frame anchor maps
    exactly onto the robot's torso position and heading. This zeroes the
    anchor position error and the heading component of the orientation error
    at the moment the policy starts; residual pitch/roll is left to the robot
    being initialized into the clip's first-frame joint configuration. Only
    yaw is aligned (not full orientation) so the jump trajectory is not tilted
    by transient pitch/roll in the robot's starting pose.
    """
    clip_pos0 = self._body_pos_w[0, _ANCHOR_IDX]
    clip_quat0 = self._body_quat_w[0, _ANCHOR_IDX]
    dyaw = _yaw_from_quat(robot_anchor_quat) - _yaw_from_quat(clip_quat0)
    self._align_R = _rot_z(dyaw)
    self._align_quat = _quat_about_z(dyaw)
    self._clip_anchor0_pos = clip_pos0.astype(np.float32)
    self._robot_anchor0_pos = robot_anchor_pos.astype(np.float32)

  def advance(self) -> None:
    """Advance one frame while playing; hold otherwise.

    Holds on the first frame until ``trigger()`` is called, plays through the
    clip exactly once, then recycles back to the first frame and holds there
    (re-armed). Holding the clip's final frame leaves the policy with a
    static, out-of-distribution reference; recycling to the first pose returns
    it to the stable "ready" stance it can hold indefinitely.
    """
    if not self._playing:
      return
    if self._t < self._total_len - 1:
      self._t += 1
    else:
      self._t = 0  # recycle to the first pose
      self._playing = False  # hold there, re-armed

  def trigger(self) -> None:
    """Start playback from the current (held) frame."""
    self._playing = True

  @property
  def is_playing(self) -> bool:
    return self._playing

  @property
  def is_armed(self) -> bool:
    """True when playback is held (always on the first frame), ready to trigger."""
    return not self._playing

  def rewind(self) -> None:
    """Reset playback to the first frame and re-arm the playback gate."""
    self._t = 0
    self._playing = False

  def _frame_idx(self, t: int) -> int:
    if t < self._pad_start:
      return 0
    elif t < self._pad_start + self.T:
      return t - self._pad_start
    else:
      return self.T - 1

  @property
  def current_joint_pos(self) -> np.ndarray:
    return self.joint_pos[self._frame_idx(self._t)]

  @property
  def current_joint_vel(self) -> np.ndarray:
    return self.joint_vel[self._frame_idx(self._t)]

  @property
  def anchor_pos_w(self) -> np.ndarray:
    pos = self._body_pos_w[self._frame_idx(self._t), _ANCHOR_IDX]
    return self._align_R @ (pos - self._clip_anchor0_pos) + self._robot_anchor0_pos

  @property
  def anchor_quat_w(self) -> np.ndarray:
    quat = self._body_quat_w[self._frame_idx(self._t), _ANCHOR_IDX]  # [w, x, y, z]
    return _quat_multiply(self._align_quat, quat)

  @property
  def anchor_height(self) -> np.float32:
    """Reference torso height relative to the motion's ground origin."""
    return self._body_pos_w[self._frame_idx(self._t), _ANCHOR_IDX, 2]

  @property
  def anchor_projected_gravity(self) -> np.ndarray:
    """World gravity direction expressed in the reference torso frame."""
    R = _rotation_matrix_from_quat(self.anchor_quat_w)
    return (R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)

  @property
  def anchor_lin_vel_b(self) -> np.ndarray:
    """Reference torso linear velocity expressed in its body frame."""
    velocity_w = self._body_lin_vel_w[self._frame_idx(self._t), _ANCHOR_IDX]
    aligned_velocity_w = self._align_R @ velocity_w
    R = _rotation_matrix_from_quat(self.anchor_quat_w)
    return (R.T @ aligned_velocity_w).astype(np.float32)

  @property
  def anchor_ang_vel_b(self) -> np.ndarray:
    """Reference torso angular velocity expressed in its body frame."""
    velocity_w = self._body_ang_vel_w[self._frame_idx(self._t), _ANCHOR_IDX]
    aligned_velocity_w = self._align_R @ velocity_w
    R = _rotation_matrix_from_quat(self.anchor_quat_w)
    return (R.T @ aligned_velocity_w).astype(np.float32)

  def anchor_pos_b(
    self,
    robot_anchor_pos: np.ndarray,
    robot_anchor_quat: np.ndarray,
  ) -> np.ndarray:
    """Motion anchor position relative to robot anchor frame (3D)."""
    R = _rotation_matrix_from_quat(robot_anchor_quat)
    return R.T @ (self.anchor_pos_w - robot_anchor_pos)

  def anchor_ori_b_6d(
    self,
    robot_anchor_quat: np.ndarray,
  ) -> np.ndarray:
    """Motion anchor orientation relative to robot anchor, as 6D (first 2 cols of R)."""
    q_rel = _quat_multiply(_quat_conjugate(robot_anchor_quat), self.anchor_quat_w)
    R_rel = _rotation_matrix_from_quat(q_rel)
    # mjlab: matrix_from_quat(ori)[..., :2].reshape(-1) — row-major flatten of
    # the first two COLUMNS, i.e. [R00,R01,R10,R11,R20,R21]. No transpose.
    return R_rel[:, :2].reshape(-1)  # shape (6,)
