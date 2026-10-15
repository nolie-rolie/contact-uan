import numpy as np
import math


def get_rpy_from_quaternion(q):
  w, x, y, z = q
  r = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
  p = np.arcsin(2 * (w * y - z * x))
  y = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
  return np.array([r, p, y])


def get_rotation_matrix_from_rpy(rpy):
  """
  Get rotation matrix from the given quaternion.
  Args:
      q (np.array[float[4]]): quaternion [w,x,y,z]
  Returns:
      np.array[float[3,3]]: rotation matrix.
  """
  r, p, y = rpy
  R_x = np.array(
    [[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]]
  )

  R_y = np.array(
    [[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]]
  )

  R_z = np.array(
    [[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]]
  )

  rot = np.dot(R_z, np.dot(R_y, R_x))
  return rot


def get_rotation_matrix_from_quaternion(q):
  return get_rotation_matrix_from_rpy(get_rpy_from_quaternion(q))
