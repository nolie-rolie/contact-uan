"""Task registry system for managing environment registration and creation."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import TypeVar

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlBaseRunnerCfg


_EnvCfgLike = ManagerBasedRlEnvCfg | Callable[[], ManagerBasedRlEnvCfg]
_RlCfgLike = RslRlBaseRunnerCfg | Callable[[], RslRlBaseRunnerCfg]
_T = TypeVar("_T")


@dataclass
class _TaskCfg:
  env_cfg: _EnvCfgLike
  play_env_cfg: _EnvCfgLike
  rl_cfg: _RlCfgLike
  runner_cls: type | None


# Private module-level registry: task_id -> task config.
_REGISTRY: dict[str, _TaskCfg] = {}


def register_mjlab_task(
  task_id: str,
  env_cfg: _EnvCfgLike,
  play_env_cfg: _EnvCfgLike,
  rl_cfg: _RlCfgLike,
  runner_cls: type | None = None,
) -> None:
  """Register an environment task.

  Args:
    task_id: Unique task identifier (e.g., "Mjlab-Velocity-Rough-Unitree-Go1").
    env_cfg: Environment configuration used for training.
    play_env_cfg: Environment configuration in "play" mode.
    rl_cfg: RL runner configuration.
    runner_cls: Optional custom runner class. If None, uses OnPolicyRunner.
  """
  if task_id in _REGISTRY:
    raise ValueError(f"Task '{task_id}' is already registered")
  _REGISTRY[task_id] = _TaskCfg(env_cfg, play_env_cfg, rl_cfg, runner_cls)


def _resolve_cfg(cfg_or_factory: _T | Callable[[], _T]) -> _T:
  return cfg_or_factory() if callable(cfg_or_factory) else cfg_or_factory


def list_tasks() -> list[str]:
  """List all registered task IDs."""
  return sorted(_REGISTRY.keys())


def load_env_cfg(task_name: str, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Load environment configuration for a task.

  Returns a deep copy to prevent mutation of the registered config.
  """
  cfg_or_factory = (
    _REGISTRY[task_name].env_cfg if not play else _REGISTRY[task_name].play_env_cfg
  )
  return deepcopy(_resolve_cfg(cfg_or_factory))


def load_rl_cfg(task_name: str) -> RslRlBaseRunnerCfg:
  """Load RL configuration for a task.

  Returns a deep copy to prevent mutation of the registered config.
  """
  return deepcopy(_resolve_cfg(_REGISTRY[task_name].rl_cfg))


def load_runner_cls(task_name: str) -> type | None:
  """Load the runner class for a task.

  If None, the default OnPolicyRunner will be used.
  """
  return _REGISTRY[task_name].runner_cls
