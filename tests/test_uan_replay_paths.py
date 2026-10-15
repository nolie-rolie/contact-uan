"""Replay inputs must be supplied explicitly; no environment is constructed."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scripts.uan_pipeline.eval_uan_replay_mjlab import (
  checkpoint_shares_rollout_actor_shape,
)
from scripts.uan_pipeline.render_uan_replay_side_by_side_mjlab import parse_args

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_UAN_JOINT_TO_ACTUATOR_MAP
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.uan.mdp.replay_command import ReplayWindowCommandCfg
from mjlab.tasks.uan.mdp.utils import load_replay_array


@pytest.mark.parametrize("play", [False, True])
def test_registered_uan_config_has_no_experiment_data_default(play: bool) -> None:
  cfg = load_env_cfg("Mjlab-UAN-Rollout-Unitree-G1", play=play)
  replay = cfg.commands["replay"]
  assert isinstance(replay, ReplayWindowCommandCfg)
  assert replay.data_dir == ""


def test_missing_replay_path_does_not_read_the_working_directory(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.chdir(tmp_path)
  expected = np.arange(12, dtype=np.float32).reshape(3, 4)
  np.save(tmp_path / "all_data.npy", expected)

  with pytest.raises(ValueError, match="--env.commands.replay.data-dir"):
    load_replay_array("")
  np.testing.assert_array_equal(load_replay_array(str(tmp_path)), expected)


def test_replay_renderer_requires_explicit_data_directory(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(sys, "argv", ["render_uan_replay_side_by_side_mjlab.py"])
  with pytest.raises(SystemExit) as error:
    parse_args()
  assert error.value.code == 2

  monkeypatch.setattr(sys, "argv", ["render", "--data-dir", str(tmp_path)])
  assert parse_args().data_dir == str(tmp_path)


def test_replay_eval_defaults_missing_metadata_to_paper_actuator_groups() -> None:
  explicit = {
    "num_actuators": 5,
    "joint_to_actuator_map": list(G1_UAN_JOINT_TO_ACTUATOR_MAP),
  }
  assert checkpoint_shares_rollout_actor_shape({}, explicit)
  assert checkpoint_shares_rollout_actor_shape(
    {"num_actuators": None, "joint_to_actuator_map": None}, explicit
  )


@pytest.mark.parametrize(
  "metadata",
  [{"num_actuators": 29}, {"num_actuators": 5, "joint_to_actuator_map": [0] * 29}],
)
def test_replay_eval_rejects_unsupported_actuator_groups(metadata: dict) -> None:
  with pytest.raises(ValueError, match="actuator-family"):
    checkpoint_shares_rollout_actor_shape(metadata, {})
