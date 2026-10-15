"""Tests for motor saturation metrics."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import torch

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.metrics import motor_saturation_fraction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _make_env(
  qfrc_actuator: list[list[float]],
  joint_vel: list[list[float]],
  *,
  action_saturation_effort: float | None = None,
  action_velocity_limit: float | None = None,
  static_limits: list[float] | None = None,
) -> "ManagerBasedRlEnv":
  qfrc = torch.tensor(qfrc_actuator, dtype=torch.float32)
  vel = torch.tensor(joint_vel, dtype=torch.float32)
  num_envs, num_joints = qfrc.shape
  joint_names = [f"joint_{idx}" for idx in range(num_joints)]

  asset = SimpleNamespace(
    num_joints=num_joints,
    joint_names=joint_names,
    actuators=[],
    data=SimpleNamespace(qfrc_actuator=qfrc, joint_vel=vel),
  )

  action = SimpleNamespace(
    target_ids=torch.arange(num_joints, dtype=torch.long),
    target_names=joint_names,
    _global_ctrl_ids=torch.arange(num_joints, dtype=torch.long),
  )
  if action_saturation_effort is not None:
    action._saturation_effort = torch.full(
      (num_envs, num_joints),
      float(action_saturation_effort),
      dtype=torch.float32,
    )
  if action_velocity_limit is not None:
    action._velocity_limit = torch.full(
      (num_envs, num_joints),
      float(action_velocity_limit),
      dtype=torch.float32,
    )

  if static_limits is None:
    static_limits = [1.0] * num_joints
  actuator_forcerange = torch.zeros((num_envs, num_joints, 2), dtype=torch.float32)
  actuator_forcerange[:, :, 0] = -torch.tensor(static_limits, dtype=torch.float32)
  actuator_forcerange[:, :, 1] = torch.tensor(static_limits, dtype=torch.float32)

  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene={"robot": asset},
    action_manager=SimpleNamespace(get_term=Mock(return_value=action)),
    sim=SimpleNamespace(model=SimpleNamespace(actuator_forcerange=actuator_forcerange)),
  )
  return cast("ManagerBasedRlEnv", env)


def test_motor_saturation_fraction_keeps_static_metric_behavior():
  env = _make_env(
    qfrc_actuator=[[9.6, 4.0]],
    joint_vel=[[0.0, 0.0]],
    static_limits=[10.0, 5.0],
  )
  cfg = MetricsTermCfg(
    func=motor_saturation_fraction,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "action_name": "joint_pos",
      "mode": "static",
      "threshold": 0.95,
    },
  )
  metric = motor_saturation_fraction(cfg, env)

  result = metric(env, **cfg.params)

  assert torch.allclose(result, torch.tensor([0.5]))
