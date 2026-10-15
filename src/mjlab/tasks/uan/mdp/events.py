"""Reset events for UAN replay rollout tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

from .actions import ReplayResidualJointPositionAction
from .replay_command import ReplayWindowCommand
from .utils import env_ids_tensor


def reset_to_recorded_trajectory_state(
  env: ManagerBasedRlEnv,
  env_ids,
  command_name: str,
  action_name: str,
) -> None:
  """Sample replay windows and reset MJLab state to the pre-action row."""
  env_ids_t = env_ids_tensor(env, env_ids)
  if env_ids_t.numel() == 0:
    return

  command = cast(ReplayWindowCommand, env.command_manager.get_term(command_name))
  action = cast(
    ReplayResidualJointPositionAction,
    env.action_manager.get_term(action_name),
  )

  starts = command.reset_to_random_window(env_ids_t)
  command.write_recorded_state(
    env_ids_t,
    command.action_state_row_ids(starts),
    action.target_ids,
  )
