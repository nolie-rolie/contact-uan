"""Curriculum functions for force control training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def ramp_force_tracking_reward(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  term_name: str,
  initial_weight: float = 0.0,
  final_weight: float = 3.0,
  min_mean_episode_length: float = 500.0,
  warmup_env_steps: int = 0,
  ramp_steps: int = 50_000,
  min_ready_fraction: float = 0.15,
) -> dict[str, torch.Tensor]:
  """Gate on episode length, then linearly ramp a reward weight.

  The weight stays at *initial_weight* until either the smoothed mean episode
  length reaches *min_mean_episode_length* **or** at least
  *min_ready_fraction* of environments exceed the threshold (whichever
  comes first), and *warmup_env_steps* have elapsed.  Once activated, the
  weight linearly ramps to *final_weight* over *ramp_steps*.
  """
  # Lazy-init per-term state on the env instance.
  state_attr = "_force_curriculum_state"
  if not hasattr(env, state_attr):
    setattr(env, state_attr, {})
  state_dict: dict = getattr(env, state_attr)

  if term_name not in state_dict:
    state_dict[term_name] = {
      "activated": False,
      "activation_step": None,
    }
  state = state_dict[term_name]

  # Use only terminated envs' episode lengths (env_ids are the envs that
  # just reset, so their episode_length_buf still has the completed length).
  if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
    if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
      ep_buf = env.episode_length_buf[env_ids].float()
    else:
      ep_buf = env.episode_length_buf.float()
    mean_ep = float(ep_buf.mean().item())
    ready_frac = float((ep_buf >= min_mean_episode_length).float().mean().item())
  else:
    mean_ep = 0.0
    ready_frac = 0.0

  # Gate-on check.
  if (not state["activated"]) and env.common_step_counter >= warmup_env_steps:
    if mean_ep >= min_mean_episode_length or ready_frac >= min_ready_fraction:
      state["activated"] = True
      state["activation_step"] = int(env.common_step_counter)

  # Compute weight.
  if not state["activated"]:
    weight = initial_weight
  else:
    elapsed = max(int(env.common_step_counter) - int(state["activation_step"]), 0)
    ratio = min(elapsed / max(int(ramp_steps), 1), 1.0)
    weight = initial_weight + ratio * (final_weight - initial_weight)

  # Apply to reward manager.
  reward_cfg = env.reward_manager.get_term_cfg(term_name)
  reward_cfg.weight = weight

  return {"weight": torch.tensor(weight)}
