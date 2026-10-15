from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.uan_pipeline.collect_estimator_data import (
  make_state_target_layout,
  parse_args,
)
from scripts.uan_pipeline.estimator_config import COMPONENT_TARGETS, HISTORY_LENGTH


def test_collection_uses_paper_history() -> None:
  assert (
    parse_args(["--checkpoint_dir", "wtw", "--iterations", "1000"]).history_length
    == HISTORY_LENGTH
    == 3
  )
  with pytest.raises(SystemExit):
    parse_args(
      ["--checkpoint_dir", "wtw", "--iterations", "1000", "--history_length", "10"]
    )


@pytest.mark.parametrize(
  "flag",
  [
    "--no_estimate_foot_heights",
    "--estimate_joint_vel",
    "--estimate_projected_gravity",
  ],
)
def test_collection_rejects_alternative_target_options(flag: str) -> None:
  with pytest.raises(SystemExit):
    parse_args(["--checkpoint_dir", "wtw", "--iterations", "1000", flag])


def test_collection_layout_contains_every_paper_component() -> None:
  layout = make_state_target_layout(
    include_foot_heights=True,
    include_foot_contact_load=True,
    include_stance_foot_orientation=True,
  )
  assert len(layout) == len(set(layout)) == 19
  for targets in COMPONENT_TARGETS.values():
    assert set(targets).issubset(layout)


@pytest.mark.parametrize(
  "flag", ["--force_control", "--force_control_uan", "--velocity_custom"]
)
def test_collection_rejects_non_wtw_policies(flag: str) -> None:
  with pytest.raises(SystemExit):
    parse_args(["--checkpoint_dir", "wtw", "--iterations", "1000", flag])


def test_collection_requires_wtw_checkpoint_selection() -> None:
  with pytest.raises(SystemExit):
    parse_args([])
