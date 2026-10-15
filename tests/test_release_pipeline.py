from __future__ import annotations

import numpy as np

from mjlab.tasks.uan.config.g1.rl_cfg import unitree_g1_uan_rollout_runner_cfg
from mjlab.tasks.uan.mdp.utils import build_valid_starts


def test_release_replay_valid_starts_are_pre_action() -> None:
  term = np.zeros(12, dtype=np.float32)
  contact = np.zeros((12, 2), dtype=np.float32)
  contact[3, 0] = 1.0
  contact[4, 0] = 1.0

  starts = build_valid_starts(
    term,
    history_horizon=3,
    rollout_horizon=4,
    contact=contact,
    require_contact_at_reset=True,
    contact_threshold=0.8,
  )

  assert starts.tolist() == [3, 4]


def test_release_uan_default_actor_is_shared5() -> None:
  cfg = unitree_g1_uan_rollout_runner_cfg()
  assert cfg.actor.num_actuators == 5
