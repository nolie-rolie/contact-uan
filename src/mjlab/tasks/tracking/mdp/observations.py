from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_anchor_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference anchor height relative to the environment origin."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.anchor_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]


def motion_anchor_projected_gravity(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """World gravity expressed in the reference anchor frame."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return quat_apply_inverse(command.anchor_quat_w, command.robot.data.gravity_vec_w)


def motion_joint_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Joint positions from the current reference-motion frame."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.joint_pos


def motion_joint_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Joint velocities from the current reference-motion frame."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.joint_vel


def motion_anchor_lin_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference anchor linear velocity expressed in its body frame."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return quat_apply_inverse(command.anchor_quat_w, command.anchor_lin_vel_w)


def motion_anchor_ang_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference anchor angular velocity expressed in its body frame."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return quat_apply_inverse(command.anchor_quat_w, command.anchor_ang_vel_w)


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)
