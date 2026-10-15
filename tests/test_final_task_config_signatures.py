"""Regression signatures for final public task configs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from mjlab.tasks.force_control.force_control_env_cfg import (
  make_force_control_env_cfg,
  make_force_control_eval_env_cfg,
  make_force_control_uan_env_cfg,
  make_force_control_uan_eval_env_cfg,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.uan.mdp.actions import ReplayResidualJointPositionActionCfg
from mjlab.tasks.utils.g1_variants import has_uan_checkpoint_env


def _task_id(*parts: str) -> str:
  return "-".join(parts)


REMOVED_TASK_IDS = {
  "Mjlab-WTW-HardTarget-Unitree-G1",
  "Mjlab-Velocity-Custom-HardTarget-Unitree-G1",
  "Mjlab-ForceControl-HardTargetEval-Unitree-G1",
  "Mjlab-UAN-Rollout-HardTarget-Debug-Unitree-G1",
  "Mjlab-WTW-UAN-Unitree-G1",
  _task_id("Mjlab", "Custom" + "ForceControl", "Unitree", "G1"),
  _task_id("Mjlab", "Custom" + "ForceControl", "Baseline", "Unitree", "G1"),
  _task_id("Mjlab", "Custom" + "ForceControl", "UAN", "Unitree", "G1"),
  _task_id(
    "Mjlab",
    "Custom" + "ForceControl",
    "HardTargetEval",
    "Unitree",
    "G1",
  ),
  _task_id("Mjlab", "ForceControl", "Baseline", "Unitree", "G1"),
  _task_id("Mjlab", "Velocity", "Custom", "Baseline", "Unitree", "G1"),
  _task_id("Mjlab", "WTW", "Flat", "Unitree", "G1"),
  _task_id("Mjlab", "WTW", "UAN", "Flat", "Unitree", "G1"),
  _task_id("Mjlab", "WTW", "Baseline", "Unitree", "G1"),
  _task_id("Mjlab", "WTW", "Quiet", "Unitree", "G1"),
  _task_id("Mjlab", "WTW", "UAN", "Quiet", "Unitree", "G1"),
  _task_id("Mjlab", "WTW", "Arm" + "Motion", "Unitree", "G1"),
}

EXPECTED_SIGNATURE_HASHES = {
  "wtw_train": "5defa37b14f98a4f44a79a6816c79f0d82167ba3e024e61a2f12b16479fe5524",
  "wtw_play": "0e6ec83c14b275eb13ac6461ee310e1f389048c53ac7e3f0b807199040279723",
  "force_train": "7a96fc1f04244195e2f14a15580ad13a38179278a0b2af444574a6f84c0b5475",
  "force_play": "d40b87052b4edb35b2674df865f29b971445028bfc713fc3b04d2596e64eefa4",
  "force_uan_train": "195ad995c595ded3c4d24875cfd833a2b797fba1e7b7c495b9c9d51a6d9c0c29",
  "force_uan_play": "62914ccc638e36814cc2c009b0342b43c76ef189bfff35cc81e3393fa1998e7f",
}


def _qualified_name(obj: Any) -> str:
  return (
    f"{getattr(obj, '__module__', type(obj).__module__)}."
    f"{getattr(obj, '__qualname__', type(obj).__qualname__)}"
  )


def _normalize(value: Any) -> Any:
  if value is None or isinstance(value, str | int | float | bool):
    return value
  if callable(value):
    return _qualified_name(value)
  if dataclasses.is_dataclass(value):
    return {
      "class": type(value).__name__,
      **{
        field.name: _normalize(getattr(value, field.name))
        for field in dataclasses.fields(value)
      },
    }
  if isinstance(value, Mapping):
    return {
      str(key): _normalize(item)
      for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
    }
  if isinstance(value, tuple | list):
    return [_normalize(item) for item in value]
  if isinstance(value, slice):
    return {"slice": [value.start, value.stop, value.step]}
  return repr(value)


def _config_signature(cfg: Any) -> dict[str, Any]:
  robot = cfg.scene.entities.get("robot")
  init_state = getattr(robot, "init_state", None)
  return {
    "scene_num_envs": cfg.scene.num_envs,
    "scene_extent": cfg.scene.extent,
    "terrain": _normalize(cfg.scene.terrain),
    "sensor_names": [sensor.name for sensor in (cfg.scene.sensors or ())],
    "sensors": _normalize(cfg.scene.sensors or ()),
    "robot_init_state": _normalize(init_state),
    "decimation": cfg.decimation,
    "episode_length_s": cfg.episode_length_s,
    "actions": _normalize(cfg.actions),
    "commands": _normalize(cfg.commands),
    "observations": _normalize(cfg.observations),
    "rewards": _normalize(cfg.rewards),
    "events": _normalize(cfg.events),
    "terminations": _normalize(cfg.terminations),
    "curriculum": _normalize(cfg.curriculum),
    "metrics": _normalize(cfg.metrics),
  }


def _signature_hash(cfg: Any) -> str:
  payload = json.dumps(
    _config_signature(cfg),
    sort_keys=True,
    separators=(",", ":"),
  )
  return hashlib.sha256(payload.encode()).hexdigest()


def test_removed_public_task_ids_are_absent() -> None:
  tasks = set(list_tasks())
  assert REMOVED_TASK_IDS.isdisjoint(tasks)


def test_canonical_public_task_ids_are_present() -> None:
  tasks = set(list_tasks())
  assert "Mjlab-ForceControl-Unitree-G1" in tasks
  assert "Mjlab-WTW-Unitree-G1" in tasks
  if has_uan_checkpoint_env():
    assert "Mjlab-ForceControl-UAN-Unitree-G1" in tasks


def test_final_wtw_config_matches_previous_baseline_behavior() -> None:
  assert (
    _signature_hash(load_env_cfg("Mjlab-WTW-Unitree-G1"))
    == (EXPECTED_SIGNATURE_HASHES["wtw_train"])
  )
  assert (
    _signature_hash(load_env_cfg("Mjlab-WTW-Unitree-G1", play=True))
    == (EXPECTED_SIGNATURE_HASHES["wtw_play"])
  )


def test_g1_uan_replay_uses_verified_residual_limit() -> None:
  for play in (False, True):
    cfg = load_env_cfg("Mjlab-UAN-Rollout-Unitree-G1", play=play)
    action = cfg.actions["residual_target"]
    assert isinstance(action, ReplayResidualJointPositionActionCfg)
    assert action.residual_clip == 1.0


def test_final_force_control_config_matches_previous_custom_behavior() -> None:
  assert (
    _signature_hash(make_force_control_env_cfg())
    == (EXPECTED_SIGNATURE_HASHES["force_train"])
  )
  assert (
    _signature_hash(make_force_control_eval_env_cfg())
    == (EXPECTED_SIGNATURE_HASHES["force_play"])
  )
  assert (
    _signature_hash(make_force_control_uan_env_cfg("dummy.pt"))
    == (EXPECTED_SIGNATURE_HASHES["force_uan_train"])
  )
  assert (
    _signature_hash(make_force_control_uan_eval_env_cfg("dummy.pt"))
    == (EXPECTED_SIGNATURE_HASHES["force_uan_play"])
  )
