"""Quaternion transforms shared by estimator collection and evaluation."""

import torch


def quat_rotate(q_wxyz, v):
  """Rotate vector v by quaternion q (wxyz convention). Batched."""
  w, x, y, z = q_wxyz[:, 0:1], q_wxyz[:, 1:2], q_wxyz[:, 2:3], q_wxyz[:, 3:4]
  t = 2.0 * torch.cross(torch.cat([x, y, z], dim=-1), v, dim=-1)
  return v + w * t + torch.cross(torch.cat([x, y, z], dim=-1), t, dim=-1)


def quat_rotate_inverse(q_wxyz, v):
  """Rotate vector v by inverse of quaternion q (wxyz convention). Batched."""
  q_inv = q_wxyz.clone()
  q_inv[:, 1:4] = -q_inv[:, 1:4]
  return quat_rotate(q_inv, v)
