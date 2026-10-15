"""Offline checks for the released Contact-UAN architecture and PPO integration."""

from __future__ import annotations

import copy
import dataclasses
import math
from pathlib import Path
from typing import Any, cast
from xml.etree import ElementTree

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_UAN_JOINT_TO_ACTUATOR_MAP,
  G1_XML,
  build_g1_joint_to_actuator_map,
)
from mjlab.rl.uan_actor import UANActor
from mjlab.tasks.uan.config.g1.rl_cfg import unitree_g1_uan_rollout_runner_cfg
from mjlab.tasks.uan.mdp.utils import resolve_uan_obs_block_names


def _observations(batch: int = 4) -> TensorDict:
  return TensorDict(
    {"actor": torch.randn(batch, 174), "critic": torch.randn(batch, 483)},
    batch_size=[batch],
  )


def _actor(obs: TensorDict, **kwargs) -> UANActor:
  return UANActor(obs, {"actor": ["actor"]}, "actor", **kwargs)


def _metadata(actor: UANActor) -> dict:
  return {
    "actor_state_dict": actor.state_dict(),
    "num_joints": 29,
    "num_actuators": 5,
    "num_blocks": 2,
    "uan_history_length": 3,
    "uan_obs_block_names": ["target_pos_error", "joint_vel"],
    "joint_to_actuator_map": list(G1_UAN_JOINT_TO_ACTUATOR_MAP),
    "joint_order": "mujoco",
    "residual_units": "rad",
    "activation": "elu",
    "use_bias": True,
  }


def test_canonical_mapping_matches_g1_xml_actuator_families() -> None:
  names = [
    joint.attrib["name"]
    for joint in ElementTree.parse(G1_XML).findall(".//worldbody//joint")
  ]
  mapping, _, groups = build_g1_joint_to_actuator_map(names, 5)
  assert tuple(mapping) == G1_UAN_JOINT_TO_ACTUATOR_MAP
  assert len(groups) == 5
  with pytest.raises(ValueError, match="five"):
    build_g1_joint_to_actuator_map(names, 29)


@pytest.mark.parametrize(
  "kwargs",
  [
    {"num_actuators": 29},
    {"num_blocks": 3},
    {"history_length": 10},
    {"joint_to_actuator_map": [0] * 29},
    {"obs_normalization": True},
    {"use_bias": False},
    {"hidden_dims": (64, 64)},
  ],
)
def test_actor_rejects_nonpaper_architectures(kwargs: dict) -> None:
  with pytest.raises(ValueError, match="Contact-UAN"):
    _actor(_observations(), **kwargs)


def test_checkpoint_and_script_export_preserve_predictions(tmp_path: Path) -> None:
  obs = _observations()
  actor = _actor(obs).eval()
  expected = actor(obs)
  checkpoint = tmp_path / "uan.pt"
  torch.save(_metadata(actor), checkpoint)
  loaded = UANActor.from_checkpoint(str(checkpoint))
  torch.testing.assert_close(
    loaded(TensorDict({"uan_obs": obs["actor"]}, batch_size=[4])), expected
  )
  exported = torch.jit.script(loaded.as_jit())
  jit_path = tmp_path / "uan.jit"
  exported.save(str(jit_path))
  torch.testing.assert_close(torch.jit.load(str(jit_path))(obs["actor"]), expected)
  altered = _metadata(actor)
  altered["actor_state_dict"]["joint_to_actuator"] = torch.zeros(29, dtype=torch.long)
  with pytest.raises(ValueError, match="canonical"):
    actor.load_state_dict(altered["actor_state_dict"])
  for bad_metadata in (
    {"num_blocks": 3},
    {"uan_history_length": 10},
    {"uan_obs_block_names": ["target_pos_error", "joint_vel", "joint_pos_rel"]},
  ):
    with pytest.raises(ValueError, match="Contact-UAN"):
      resolve_uan_obs_block_names(bad_metadata)


def test_standard_ppo_updates_and_resumes_paper_actor() -> None:
  old_threads = torch.get_num_threads()
  torch.set_num_threads(1)
  try:
    torch.manual_seed(17)
    cfg = unitree_g1_uan_rollout_runner_cfg()
    assert cfg.algorithm.class_name == "PPO"
    obs = _observations()
    actor = _actor(obs, distribution_cfg=cfg.actor.distribution_cfg)
    critic = MLPModel(
      obs, {"critic": ["critic"]}, "critic", 1, hidden_dims=cfg.critic.hidden_dims
    )
    storage = RolloutStorage("rl", 4, 4, obs, [29], "cpu")
    algorithm_cfg = dataclasses.asdict(cfg.algorithm)
    algorithm_cfg.pop("class_name")
    algorithm_cfg.pop("share_cnn_encoders")
    algorithm = PPO(cast(Any, actor), critic, storage, **algorithm_cfg)
    before = actor.layer_weights[0].detach().clone()
    for _ in range(4):
      with torch.no_grad():
        actions = algorithm.act(obs)
        next_obs = _observations()
        algorithm.process_env_step(
          next_obs, -actions.square().sum(-1), torch.zeros(4), {}
        )
      obs = next_obs
    algorithm.compute_returns(obs)
    losses = algorithm.update()
    assert all(math.isfinite(float(value)) for value in losses.values())
    assert not torch.equal(before, actor.layer_weights[0])
    checkpoint = copy.deepcopy(algorithm.save())
    expected = actor(obs).detach().clone()
    with torch.no_grad():
      actor.layer_weights[0].add_(1)
    assert algorithm.load(checkpoint, load_cfg=None, strict=True)
    torch.testing.assert_close(actor(obs), expected)
  finally:
    torch.set_num_threads(old_threads)
