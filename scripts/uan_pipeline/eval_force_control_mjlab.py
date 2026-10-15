"""Evaluate MJLab force-control checkpoints in MJLab env variants.

Reports:
  - initial-episode survival over a fixed-duration rollout
  - yaw-frame force tracking errors
  - overall timeout/failure counts across all observed episodes
  - mean failure time for non-timeout terminations

Usage:
  uv run python scripts/uan_pipeline/eval_force_control_mjlab.py \
    --checkpoint "logs/rsl_rl/g1_force_control/<run>/model_29999.pt" \
    --env_variant eval \
    --num_envs 32 \
    --duration_s 20.0
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import types
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.force_control.config.g1.env_cfgs import (
  make_force_control_eval_env_cfg,
  make_force_control_uan_eval_env_cfg,
)
from mjlab.tasks.force_control.mdp.commands import ForceFieldCommand

EvaluationResult = dict[
  str, float | int | str | None | dict[str, int] | dict[str, np.ndarray]
]


def _safe_load_checkpoint(path: str) -> dict:
  class _Dummy:
    def __init__(self, *args, **kwargs):
      pass

  class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
      try:
        return super().find_class(module, name)
      except (ModuleNotFoundError, ImportError, AttributeError):
        return _Dummy

  stub_mod = types.ModuleType("_pickle_stub")
  stub_mod.__dict__.update(Unpickler=_StubUnpickler, load=pickle.load, dump=pickle.dump)

  return torch.load(
    path,
    map_location="cpu",
    weights_only=False,
    pickle_module=stub_mod,
  )


def load_policy(checkpoint_path: str, device: torch.device) -> nn.Module:
  if checkpoint_path.endswith(".jit"):
    policy = torch.jit.load(checkpoint_path, map_location=device)
    policy.eval()
    return policy

  ckpt = _safe_load_checkpoint(checkpoint_path)
  sd = ckpt["actor_state_dict"]
  linear_keys = sorted(
    [k for k in sd if k.startswith("mlp.") and k.endswith(".weight")],
    key=lambda k: int(k.split(".")[1]),
  )
  layers: list[nn.Module] = []
  for i, wk in enumerate(linear_keys):
    idx = int(wk.split(".")[1])
    linear = nn.Linear(sd[wk].shape[1], sd[wk].shape[0])
    linear.weight.data = sd[wk]
    linear.bias.data = sd[f"mlp.{idx}.bias"]
    layers.append(linear)
    if i < len(linear_keys) - 1:
      layers.append(nn.ELU())
  mlp = nn.Sequential(*layers).to(device).eval()

  obs_mean = sd.get("obs_normalizer._mean")
  obs_std = sd.get("obs_normalizer._std")

  class _NormalizedPolicy(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, model, mean, std):
      super().__init__()
      self.model = model
      self.has_norm = mean is not None and std is not None
      if self.has_norm:
        self.register_buffer("mean", mean.to(device))
        self.register_buffer("std", std.to(device))
      else:
        self.register_buffer("mean", torch.zeros(1, device=device))
        self.register_buffer("std", torch.ones(1, device=device))

    def forward(self, obs):
      if self.has_norm:
        # Match rsl_rl.modules.EmpiricalNormalization exactly. Its default
        # epsilon is large enough to noticeably affect policy actions.
        obs = (obs - self.mean) / (self.std + 1e-2)
      return self.model(obs)

  return _NormalizedPolicy(mlp, obs_mean, obs_std).to(device).eval()


def make_env_cfg(
  env_variant: str,
  uan_checkpoint: str | None = None,
):
  if env_variant == "eval":
    return make_force_control_eval_env_cfg()
  if env_variant == "uan_eval":
    if not uan_checkpoint:
      raise ValueError("--uan_checkpoint is required when --env_variant=uan_eval")
    return make_force_control_uan_eval_env_cfg(
      uan_checkpoint_path=uan_checkpoint,
    )
  raise ValueError(f"Unsupported env variant: {env_variant}")


def _empty_force_stats(prefix: str = "") -> dict[str, float | int]:
  return {
    f"{prefix}force_error_yaw_norm_mean": float("nan"),
    f"{prefix}force_error_yaw_norm_rmse": float("nan"),
    f"{prefix}force_error_yaw_norm_median": float("nan"),
    f"{prefix}force_error_yaw_norm_p95": float("nan"),
    f"{prefix}force_error_yaw_norm_max": float("nan"),
    f"{prefix}force_error_yaw_x_rmse": float("nan"),
    f"{prefix}force_error_yaw_y_rmse": float("nan"),
    f"{prefix}force_error_yaw_z_rmse": float("nan"),
    f"{prefix}target_force_yaw_norm_mean": float("nan"),
    f"{prefix}applied_force_yaw_norm_mean": float("nan"),
    f"{prefix}track_force_ee_raw_mean": float("nan"),
    f"{prefix}force_sample_count": 0,
  }


def _stats_from_flat_tensors(
  err: torch.Tensor,
  tgt: torch.Tensor,
  app: torch.Tensor,
  raw_track_force: torch.Tensor | None,
  prefix: str = "",
) -> dict[str, float | int]:
  if err.numel() == 0:
    return _empty_force_stats(prefix=prefix)
  err_norm = torch.linalg.norm(err, dim=-1)
  axis_rmse = torch.sqrt(torch.mean(err.square(), dim=0))

  stats: dict[str, float | int] = {
    f"{prefix}force_error_yaw_norm_mean": float(err_norm.mean().item()),
    f"{prefix}force_error_yaw_norm_rmse": float(
      torch.sqrt(torch.mean(err_norm.square())).item()
    ),
    f"{prefix}force_error_yaw_norm_median": float(
      torch.quantile(err_norm, 0.50).item()
    ),
    f"{prefix}force_error_yaw_norm_p95": float(torch.quantile(err_norm, 0.95).item()),
    f"{prefix}force_error_yaw_norm_max": float(err_norm.max().item()),
    f"{prefix}force_error_yaw_x_rmse": float(axis_rmse[0].item()),
    f"{prefix}force_error_yaw_y_rmse": float(axis_rmse[1].item()),
    f"{prefix}force_error_yaw_z_rmse": float(axis_rmse[2].item()),
    f"{prefix}target_force_yaw_norm_mean": float(
      torch.linalg.norm(tgt, dim=-1).mean().item()
    ),
    f"{prefix}applied_force_yaw_norm_mean": float(
      torch.linalg.norm(app, dim=-1).mean().item()
    ),
    f"{prefix}force_sample_count": int(err.shape[0]),
  }

  if raw_track_force is not None and raw_track_force.numel() > 0:
    stats[f"{prefix}track_force_ee_raw_mean"] = float(raw_track_force.mean().item())
  else:
    stats[f"{prefix}track_force_ee_raw_mean"] = float("nan")
  return stats


def _compute_force_stats(
  errors: list[torch.Tensor],
  targets: list[torch.Tensor],
  applied: list[torch.Tensor],
  track_force_rewards: list[torch.Tensor],
  active_force_threshold_n: float,
  prefix: str = "",
) -> dict[str, float | int]:
  if not errors:
    return {
      **_empty_force_stats(prefix=prefix),
      **_empty_force_stats(prefix=f"{prefix}active_"),
      f"{prefix}active_force_threshold_n": float(active_force_threshold_n),
    }

  err = torch.cat([v.reshape(-1, 3) for v in errors], dim=0)
  tgt = torch.cat([v.reshape(-1, 3) for v in targets], dim=0)
  app = torch.cat([v.reshape(-1, 3) for v in applied], dim=0)
  raw = (
    torch.cat([v.reshape(-1) for v in track_force_rewards], dim=0)
    if track_force_rewards
    else None
  )
  stats = _stats_from_flat_tensors(err, tgt, app, raw, prefix=prefix)
  active_mask = torch.linalg.norm(tgt, dim=-1) > float(active_force_threshold_n)
  active_raw = raw[active_mask] if raw is not None else None
  stats.update(
    _stats_from_flat_tensors(
      err[active_mask],
      tgt[active_mask],
      app[active_mask],
      active_raw,
      prefix=f"{prefix}active_",
    )
  )
  stats[f"{prefix}active_force_threshold_n"] = float(active_force_threshold_n)
  return stats


def _finite_np(values) -> np.ndarray:
  arr = np.asarray(values, dtype=np.float64).reshape(-1)
  return arr[np.isfinite(arr)]


def _bootstrap_ci(
  values,
  *,
  seed: int,
  num_bootstrap: int = 1000,
) -> tuple[float, float]:
  arr = _finite_np(values)
  if arr.size == 0:
    return float("nan"), float("nan")
  if arr.size == 1:
    value = float(arr[0])
    return value, value
  rng = np.random.default_rng(seed)
  means = rng.choice(arr, size=(num_bootstrap, arr.size), replace=True).mean(axis=1)
  return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _value_stats(values, prefix: str, *, seed: int) -> dict[str, float | int]:
  arr = _finite_np(values)
  if arr.size == 0:
    return {
      f"{prefix}_mean": float("nan"),
      f"{prefix}_std": float("nan"),
      f"{prefix}_ci95_low": float("nan"),
      f"{prefix}_ci95_high": float("nan"),
      f"{prefix}_env_count": 0,
    }
  ci_low, ci_high = _bootstrap_ci(arr, seed=seed)
  return {
    f"{prefix}_mean": float(arr.mean()),
    f"{prefix}_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    f"{prefix}_ci95_low": ci_low,
    f"{prefix}_ci95_high": ci_high,
    f"{prefix}_env_count": int(arr.size),
  }


def _rmse_stats(
  sum_sq: torch.Tensor,
  count: torch.Tensor,
  prefix: str,
  *,
  seed: int,
) -> dict[str, float | int]:
  sum_sq_np = sum_sq.detach().cpu().numpy().astype(np.float64)
  count_np = count.detach().cpu().numpy().astype(np.float64)
  valid = count_np > 0
  values = np.full_like(count_np, np.nan, dtype=np.float64)
  values[valid] = np.sqrt(sum_sq_np[valid] / count_np[valid])
  pooled = (
    float(np.sqrt(sum_sq_np[valid].sum() / count_np[valid].sum()))
    if valid.any()
    else float("nan")
  )
  return {
    prefix: pooled,
    f"{prefix}_sample_count": int(count_np[valid].sum()) if valid.any() else 0,
    **_value_stats(values, prefix, seed=seed),
  }


def _mean_accumulator_stats(
  total: torch.Tensor,
  count: torch.Tensor,
  prefix: str,
  *,
  seed: int,
) -> dict[str, float | int]:
  total_np = total.detach().cpu().numpy().astype(np.float64)
  count_np = count.detach().cpu().numpy().astype(np.float64)
  valid = count_np > 0
  values = np.full_like(count_np, np.nan, dtype=np.float64)
  values[valid] = total_np[valid] / count_np[valid]
  pooled = (
    float(total_np[valid].sum() / count_np[valid].sum())
    if valid.any()
    else float("nan")
  )
  return {
    prefix: pooled,
    f"{prefix}_sample_count": int(count_np[valid].sum()) if valid.any() else 0,
    **_value_stats(values, prefix, seed=seed),
  }


def _wilson_ci(
  successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
  if total <= 0:
    return float("nan"), float("nan")
  phat = successes / total
  denom = 1.0 + z**2 / total
  centre = phat + z**2 / (2.0 * total)
  margin = z * np.sqrt((phat * (1.0 - phat) + z**2 / (4.0 * total)) / total)
  return (
    float(max(0.0, (centre - margin) / denom)),
    float(min(1.0, (centre + margin) / denom)),
  )


def _raw_reward_term(env: ManagerBasedRlEnv, term_name: str) -> torch.Tensor | None:
  try:
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_index = env.reward_manager.active_terms.index(term_name)
    weighted = env.reward_manager._step_reward[:, term_index]
  except ValueError:
    return None
  if float(term_cfg.weight) == 0.0:
    return None
  return weighted / float(term_cfg.weight)


def evaluate(
  checkpoint: str,
  env_variant: str,
  num_envs: int,
  seconds: float,
  device: str,
  seed: int,
  uan_checkpoint: str | None = None,
  active_force_threshold_n: float = 1.0,
  save_per_env: bool = False,
) -> EvaluationResult:
  torch.manual_seed(seed)
  cfg = make_env_cfg(
    env_variant,
    uan_checkpoint=uan_checkpoint,
  )
  cfg.seed = seed
  cfg.scene.num_envs = num_envs
  cfg.scene.extent = max(float(num_envs) * 2.0, 2.0)
  cfg.episode_length_s = seconds + 1.0
  cfg.auto_reset = False

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  policy = load_policy(checkpoint, torch.device(device))
  force_cmd = cast(ForceFieldCommand, env.command_manager.get_term("desired_force"))

  total_steps = int(seconds / env.step_dt)
  obs_dict, _ = env.reset()
  obs = obs_dict["actor"]

  alive = torch.ones(num_envs, dtype=torch.bool, device=device)
  failure_time = torch.full(
    (num_envs,), float(seconds), dtype=torch.float, device=device
  )
  first_failure_reason = torch.full((num_envs,), -1, dtype=torch.long, device=device)
  term_names = list(env.termination_manager.active_terms)
  failure_counts = {name: 0 for name in term_names}

  raw_norm_sq = torch.zeros(num_envs, dtype=torch.float64, device=device)
  raw_count = torch.zeros(num_envs, dtype=torch.long, device=device)
  raw_active_norm_sq = torch.zeros_like(raw_norm_sq)
  raw_active_count = torch.zeros_like(raw_count)
  alive_norm_sq = torch.zeros_like(raw_norm_sq)
  alive_count = torch.zeros_like(raw_count)
  alive_active_norm_sq = torch.zeros_like(raw_norm_sq)
  alive_active_count = torch.zeros_like(raw_count)
  alive_x_sq = torch.zeros_like(raw_norm_sq)
  alive_y_sq = torch.zeros_like(raw_norm_sq)
  alive_z_sq = torch.zeros_like(raw_norm_sq)
  target_norm_sum = torch.zeros_like(raw_norm_sq)
  applied_norm_sum = torch.zeros_like(raw_norm_sq)
  force_norm_count = torch.zeros_like(raw_count)
  torque_sq_sum = torch.zeros_like(raw_norm_sq)
  positive_work_sum = torch.zeros_like(raw_norm_sq)
  effort_count = torch.zeros_like(raw_count)

  start = time.time()
  for step_idx in range(total_steps):
    with torch.inference_mode():
      action = policy(obs)
    obs_dict, _, terminated, truncated, _ = env.step(action)
    obs = obs_dict["actor"]
    done = terminated | truncated
    newly_failed = alive & done

    if newly_failed.any():
      failure_time[newly_failed] = (step_idx + 1) * env.step_dt
      for reason_idx, name in enumerate(term_names):
        reason_mask = env.termination_manager.get_term(name).to(torch.bool)
        reason_new = newly_failed & reason_mask
        failure_counts[name] += int(reason_new.sum().item())
        first_failure_reason[reason_new & (first_failure_reason < 0)] = reason_idx

    target_yaw = force_cmd.target_forces_b.detach()
    applied_yaw = force_cmd.forcefield_forces_b.detach()
    error_yaw = target_yaw - applied_yaw
    err_norm_sq = torch.sum(error_yaw.square(), dim=1)
    target_norm = torch.linalg.norm(target_yaw, dim=1)
    applied_norm = torch.linalg.norm(applied_yaw, dim=1)
    finite = (
      torch.isfinite(error_yaw).all(dim=1)
      & torch.isfinite(target_norm)
      & torch.isfinite(applied_norm)
    )
    active = target_norm > float(active_force_threshold_n)

    raw_norm_sq[finite] += err_norm_sq[finite].double()
    raw_count[finite] += 1
    raw_active_mask = finite & active
    raw_active_norm_sq[raw_active_mask] += err_norm_sq[raw_active_mask].double()
    raw_active_count[raw_active_mask] += 1
    target_norm_sum[finite] += target_norm[finite].double()
    applied_norm_sum[finite] += applied_norm[finite].double()
    force_norm_count[finite] += 1

    robot = env.scene["robot"]
    tau = robot.data.qfrc_actuator
    qvel = robot.data.joint_vel
    effort_dim = min(tau.shape[1], qvel.shape[1])
    tau = tau[:, :effort_dim]
    qvel = qvel[:, :effort_dim]
    torque_sq = torch.sum(tau.square(), dim=1)
    positive_work = torch.sum(torch.clamp(tau * qvel, min=0.0), dim=1)
    effort_finite = torch.isfinite(torque_sq) & torch.isfinite(positive_work)
    torque_sq_sum[effort_finite] += torque_sq[effort_finite].double()
    positive_work_sum[effort_finite] += positive_work[effort_finite].double()
    effort_count[effort_finite] += 1

    alive &= ~done
    alive_mask = alive & finite
    alive_norm_sq[alive_mask] += err_norm_sq[alive_mask].double()
    alive_count[alive_mask] += 1
    alive_x_sq[alive_mask] += error_yaw[alive_mask, 0].square().double()
    alive_y_sq[alive_mask] += error_yaw[alive_mask, 1].square().double()
    alive_z_sq[alive_mask] += error_yaw[alive_mask, 2].square().double()
    alive_active_mask = alive_mask & active
    alive_active_norm_sq[alive_active_mask] += err_norm_sq[alive_active_mask].double()
    alive_active_count[alive_active_mask] += 1

  wall_s = time.time() - start
  env.close()

  failed = ~alive
  num_survivors = int(alive.sum().item())
  num_failed = int(failed.sum().item())
  survival_ci_low, survival_ci_high = _wilson_ci(num_survivors, num_envs)
  failed_times = failure_time[failed]
  failure_time_np = failure_time.detach().cpu().numpy().astype(np.float64)
  survivor_count = raw_count.clone()
  survivor_norm_sq = raw_norm_sq.clone()
  survivor_active_count = raw_active_count.clone()
  survivor_active_norm_sq = raw_active_norm_sq.clone()
  survivor_count[failed] = 0
  survivor_norm_sq[failed] = 0.0
  survivor_active_count[failed] = 0
  survivor_active_norm_sq[failed] = 0.0

  force_stats = {
    **_rmse_stats(
      alive_norm_sq, alive_count, "alive_force_error_yaw_norm_rmse", seed=seed + 201
    ),
    **_rmse_stats(
      alive_active_norm_sq,
      alive_active_count,
      "alive_active_force_error_yaw_norm_rmse",
      seed=seed + 202,
    ),
    **_rmse_stats(
      raw_norm_sq, raw_count, "raw_force_error_yaw_norm_rmse", seed=seed + 203
    ),
    **_rmse_stats(
      raw_active_norm_sq,
      raw_active_count,
      "raw_active_force_error_yaw_norm_rmse",
      seed=seed + 204,
    ),
    **_rmse_stats(
      survivor_norm_sq,
      survivor_count,
      "survivor_force_error_yaw_norm_rmse",
      seed=seed + 205,
    ),
    **_rmse_stats(
      survivor_active_norm_sq,
      survivor_active_count,
      "survivor_active_force_error_yaw_norm_rmse",
      seed=seed + 206,
    ),
    **_rmse_stats(
      alive_x_sq, alive_count, "alive_force_error_yaw_x_rmse", seed=seed + 207
    ),
    **_rmse_stats(
      alive_y_sq, alive_count, "alive_force_error_yaw_y_rmse", seed=seed + 208
    ),
    **_rmse_stats(
      alive_z_sq, alive_count, "alive_force_error_yaw_z_rmse", seed=seed + 209
    ),
    **_mean_accumulator_stats(
      target_norm_sum, force_norm_count, "target_force_yaw_norm_mean", seed=seed + 210
    ),
    **_mean_accumulator_stats(
      applied_norm_sum, force_norm_count, "applied_force_yaw_norm_mean", seed=seed + 211
    ),
    **_mean_accumulator_stats(
      torque_sq_sum, effort_count, "mean_torque_squared", seed=seed + 212
    ),
    **_mean_accumulator_stats(
      positive_work_sum, effort_count, "mean_positive_work", seed=seed + 213
    ),
    "active_force_threshold_n": float(active_force_threshold_n),
  }

  result: EvaluationResult = {
    "checkpoint": checkpoint,
    "uan_checkpoint": uan_checkpoint,
    "env_variant": env_variant,
    "num_envs": num_envs,
    "seconds": seconds,
    "duration_s": seconds,
    "device": device,
    "seed": seed,
    "steps_requested": int(total_steps),
    "steps_executed": int(total_steps),
    "failures": num_failed,
    "timeouts": 0,
    "failure_counts": failure_counts,
    "total_episodes": num_envs,
    "initial_failures": num_failed,
    "initial_timeouts": 0,
    "initial_episode_survival": num_survivors / max(num_envs, 1),
    "survival_rate_ci95_low": survival_ci_low,
    "survival_rate_ci95_high": survival_ci_high,
    "failure_free_episodes": num_survivors / max(num_envs, 1),
    "mean_failure_time_s": float(np.mean(failure_time_np)),
    "median_failure_time_s": float(np.median(failure_time_np)),
    **_value_stats(failure_time_np, "failure_time_s", seed=seed + 214),
    "failed_mean_failure_time_s": float(failed_times.mean().item())
    if failed.any()
    else float("nan"),
    "failed_median_failure_time_s": float(failed_times.median().item())
    if failed.any()
    else float("nan"),
    "wall_s": wall_s,
    **force_stats,
  }
  result.update(
    {
      # Backward-compatible headline fields are alive-conditioned.
      "force_error_yaw_norm_rmse": result["alive_force_error_yaw_norm_rmse"],
      "active_force_error_yaw_norm_rmse": result[
        "alive_active_force_error_yaw_norm_rmse"
      ],
      "nonterminal_force_error_yaw_norm_rmse": result[
        "alive_force_error_yaw_norm_rmse"
      ],
      "nonterminal_force_error_yaw_norm_p95": float("nan"),
      "survivor_force_sample_count": result[
        "survivor_force_error_yaw_norm_rmse_sample_count"
      ],
      "force_sample_count": result["alive_force_error_yaw_norm_rmse_sample_count"],
      "force_error_yaw_x_rmse": result["alive_force_error_yaw_x_rmse"],
      "force_error_yaw_y_rmse": result["alive_force_error_yaw_y_rmse"],
      "force_error_yaw_z_rmse": result["alive_force_error_yaw_z_rmse"],
      "force_error_yaw_norm_p95": float("nan"),
      "active_force_error_yaw_norm_p95": float("nan"),
      "survivor_force_error_yaw_norm_p95": float("nan"),
    }
  )
  if save_per_env:
    raw_count_np = raw_count.detach().cpu().numpy()
    raw_norm_np = raw_norm_sq.detach().cpu().numpy()
    alive_count_np = alive_count.detach().cpu().numpy()
    alive_norm_np = alive_norm_sq.detach().cpu().numpy()
    result["per_env"] = {
      "failure_time_s": failure_time_np,
      "raw_force_error_yaw_norm_rmse": np.sqrt(
        np.divide(
          raw_norm_np,
          raw_count_np,
          out=np.full(num_envs, np.nan, dtype=np.float64),
          where=raw_count_np > 0,
        )
      ),
      "alive_force_error_yaw_norm_rmse": np.sqrt(
        np.divide(
          alive_norm_np,
          alive_count_np,
          out=np.full(num_envs, np.nan, dtype=np.float64),
          where=alive_count_np > 0,
        )
      ),
    }
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=str, required=True)
  parser.add_argument(
    "--env_variant",
    type=str,
    choices=("eval", "uan_eval"),
    required=True,
  )
  parser.add_argument(
    "--uan_checkpoint",
    type=str,
    default=None,
    help="UAN checkpoint path. Required for --env_variant=uan_eval.",
  )
  parser.add_argument("--num_envs", type=int, default=32)
  parser.add_argument("--seconds", type=float, default=None)
  parser.add_argument("--duration_s", type=float, default=None)
  parser.add_argument("--device", type=str, default="cpu")
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument(
    "--active_force_threshold_n",
    type=float,
    default=1.0,
    help="Target-force norm threshold for active-force tracking metrics.",
  )
  parser.add_argument(
    "--save_per_env",
    action="store_true",
    help="Include per-env diagnostic arrays in the JSON output.",
  )
  parser.add_argument("--output_json", type=str, default=None)
  args = parser.parse_args()

  seconds = args.duration_s if args.duration_s is not None else args.seconds
  if seconds is None:
    seconds = 20.0

  results = evaluate(
    checkpoint=args.checkpoint,
    env_variant=args.env_variant,
    num_envs=args.num_envs,
    seconds=float(seconds),
    device=args.device,
    seed=args.seed,
    uan_checkpoint=args.uan_checkpoint,
    active_force_threshold_n=args.active_force_threshold_n,
    save_per_env=args.save_per_env,
  )

  print(
    f"[{results['env_variant']}] "
    f"initial_survival={results['initial_episode_survival']:.4f} "
    f"failures={results['failures']} "
    f"timeouts={results['timeouts']} "
    f"total_episodes={results['total_episodes']}"
  )
  print(
    f"[{results['env_variant']}] "
    f"failure_free_episodes={results['failure_free_episodes']:.4f} "
    f"mean_failure_time_s={results['mean_failure_time_s']}"
  )
  print(
    f"[{results['env_variant']}] "
    f"force_yaw_rmse={results['force_error_yaw_norm_rmse']:.4f} "
    f"force_yaw_p95={results['force_error_yaw_norm_p95']:.4f} "
    f"axis_rmse=({results['force_error_yaw_x_rmse']:.4f}, "
    f"{results['force_error_yaw_y_rmse']:.4f}, "
    f"{results['force_error_yaw_z_rmse']:.4f})"
  )
  print(
    f"[{results['env_variant']}] "
    f"active_force_yaw_rmse={results['active_force_error_yaw_norm_rmse']:.4f} "
    f"active_force_yaw_p95={results['active_force_error_yaw_norm_p95']:.4f} "
    f"active_axis_rmse=({results['active_force_error_yaw_x_rmse']:.4f}, "
    f"{results['active_force_error_yaw_y_rmse']:.4f}, "
    f"{results['active_force_error_yaw_z_rmse']:.4f}) "
    f"active_samples={results['active_force_sample_count']}"
  )

  if args.output_json:
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
  main()
