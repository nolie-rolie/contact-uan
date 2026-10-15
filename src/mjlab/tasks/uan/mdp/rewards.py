"""Reward terms for UAN replay rollout tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .actions import ReplayResidualJointPositionAction
from .replay_command import ReplayWindowCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def joint_position_mse(
  env: ManagerBasedRlEnv,
  command_name: str,
  action_name: str,
  mse_clip: float | None = None,
) -> torch.Tensor:
  command = cast(ReplayWindowCommand, env.command_manager.get_term(command_name))
  action = cast(
    ReplayResidualJointPositionAction,
    env.action_manager.get_term(action_name),
  )
  joint_pos = command.robot.data.joint_pos[:, action.target_ids]
  joint_error = joint_pos - command.current_gt_joint_pos
  command.record_joint_position_error(joint_error, action.target_names)
  mse = joint_error.square().mean(dim=-1)
  if mse_clip is not None:
    mse = mse.clamp(max=float(mse_clip))
  return mse


def joint_velocity_mse(
  env: ManagerBasedRlEnv,
  command_name: str,
  action_name: str,
  mse_clip: float | None = None,
) -> torch.Tensor:
  command = cast(ReplayWindowCommand, env.command_manager.get_term(command_name))
  action = cast(
    ReplayResidualJointPositionAction,
    env.action_manager.get_term(action_name),
  )
  joint_vel = command.robot.data.joint_vel[:, action.target_ids]
  recorded_joint_vel = command.current_gt_joint_vel
  mse = (joint_vel - recorded_joint_vel).square().mean(dim=-1)
  if mse_clip is not None:
    mse = mse.clamp(max=float(mse_clip))
  return mse


def residual_l2(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  action = cast(
    ReplayResidualJointPositionAction,
    env.action_manager.get_term(action_name),
  )
  return action.processed_residual.square().mean(dim=-1)


def residual_rate_l2(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  action = cast(
    ReplayResidualJointPositionAction,
    env.action_manager.get_term(action_name),
  )
  return action.processed_residual_delta.square().mean(dim=-1)
