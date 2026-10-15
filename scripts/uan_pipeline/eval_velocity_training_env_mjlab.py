"""Evaluate velocity policies in the custom G1 training environment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
for _name in (
  "UAN_CHECKPOINT_PATH",
  "UAN_CHECKPOINT_PATHS",
  "UAN_JIT_PATH",
  "UAN_JIT_PATHS",
):
  os.environ.pop(_name, None)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs/eval/velocity_training_env"
DEFAULT_SATURATION_THRESHOLD = 0.95
DEFAULT_EPS = 1.0e-6


def _jsonable(value: Any) -> Any:
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, torch.Tensor):
    value = value.detach().cpu()
    if value.numel() == 1:
      return value.item()
    return value.tolist()
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, dict):
    return {str(k): _jsonable(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(v) for v in value]
  return value


def _finite_np(values: Any) -> np.ndarray:
  arr = np.asarray(values, dtype=np.float64).reshape(-1)
  return arr[np.isfinite(arr)]


def _bootstrap_ci(
  values: Any,
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


def _value_stats(values: Any, prefix: str, *, seed: int) -> dict[str, float | int]:
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


def _wilson_ci(
  successes: int,
  total: int,
  z: float = 1.959963984540054,
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


def _resolve_named_values(
  value: float | dict[str, float],
  *,
  target_names: list[str],
  num_envs: int,
  device: str,
  field_name: str,
) -> torch.Tensor:
  if isinstance(value, (float, int)):
    resolved = torch.full(
      (num_envs, len(target_names)),
      float(value),
      dtype=torch.float,
      device=device,
    )
  elif isinstance(value, dict):
    resolved = torch.full(
      (num_envs, len(target_names)),
      float("nan"),
      dtype=torch.float,
      device=device,
    )
    for target_index, target_name in enumerate(target_names):
      matched_key = None
      matched_value = None
      for pattern, candidate_value in value.items():
        if re.fullmatch(pattern, target_name):
          if matched_key is not None:
            raise ValueError(
              f"Multiple {field_name} values match target {target_name!r}: "
              f"{matched_key!r} and {pattern!r}."
            )
          matched_key = pattern
          matched_value = float(candidate_value)
      if matched_value is not None:
        resolved[:, target_index] = matched_value
    if torch.isnan(resolved).any():
      missing = [
        name
        for index, name in enumerate(target_names)
        if torch.isnan(resolved[0, index]).item()
      ]
      raise ValueError(f"Missing {field_name} values for targets: {missing}.")
  else:
    raise TypeError(
      f"Unsupported {field_name} type: {type(value).__name__}. "
      "Expected float or dict[str, float]."
    )

  if torch.any(resolved <= 0.0):
    raise ValueError(f"{field_name} must be strictly positive for every target.")
  return resolved


def dynamic_saturation_fraction(
  torques: torch.Tensor,
  joint_vel: torch.Tensor,
  saturation_effort: torch.Tensor,
  velocity_limit: torch.Tensor,
  *,
  threshold: float = DEFAULT_SATURATION_THRESHOLD,
  eps: float = DEFAULT_EPS,
) -> torch.Tensor:
  """Return per-env fraction of joints above the dynamic torque-speed limit."""
  dynamic_limit = saturation_effort * torch.clamp(
    1.0 - torch.abs(joint_vel / velocity_limit),
    min=0.0,
  )
  usage = torch.abs(torques) / torch.clamp(dynamic_limit, min=eps)
  return (usage >= threshold).float().mean(dim=-1)


def sum_squared_torques(torques: torch.Tensor) -> torch.Tensor:
  """Return per-env sum of squared actuator torques."""
  return torch.sum(torques.square(), dim=-1)


def _accumulate_alive_mean(
  total: torch.Tensor,
  count: torch.Tensor,
  values: torch.Tensor,
  alive_mask: torch.Tensor,
) -> None:
  mask = alive_mask & torch.isfinite(values)
  total[mask] += values[mask].double()
  count[mask] += 1


def _resolve_checkpoint(path: str | Path) -> Path:
  checkpoint = Path(path).expanduser()
  if checkpoint.exists():
    return checkpoint.resolve()
  raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")


def _build_env(num_envs: int, duration_s: float, seed: int, device: str):
  import mjlab.tasks  # noqa: F401  # Populate registry side effects.
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.velocity.config.g1_custom.env_cfgs import unitree_g1_custom_env_cfg

  cfg = unitree_g1_custom_env_cfg(play=False)
  cfg.scene.num_envs = int(num_envs)
  cfg.seed = int(seed)
  cfg.episode_length_s = float(duration_s) + 1.0
  # Training-env eval can have early failures. Keep auto-reset enabled so one
  # failed env does not halt the vectorized rollout; first failures are still
  # recorded and failed envs stop contributing to metrics below.
  cfg.auto_reset = True
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def _load_policy(checkpoint: Path, wrapped, device: str):
  from mjlab.tasks.velocity.config.g1_custom.rl_cfg import (
    unitree_g1_custom_ppo_runner_cfg,
  )
  from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

  if checkpoint.suffix == ".jit":
    module = torch.jit.load(str(checkpoint), map_location=device)
    module.eval()

    def jit_policy(obs):
      actor_obs = obs["actor"] if isinstance(obs, dict) else obs.get("actor")
      return module(actor_obs.to(device))

    return jit_policy, "jit"

  agent_cfg = unitree_g1_custom_ppo_runner_cfg()
  runner = VelocityOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  return runner.get_inference_policy(device=device), "pt"


def _g1_dynamic_limit_tensors(env, asset) -> tuple[torch.Tensor, torch.Tensor]:
  from mjlab.asset_zoo.robots import (
    G1_TORQUE_SPEED_SATURATION_EFFORT,
    G1_TORQUE_SPEED_VELOCITY_LIMIT,
  )

  return (
    _resolve_named_values(
      G1_TORQUE_SPEED_SATURATION_EFFORT,
      target_names=asset.joint_names,
      num_envs=env.num_envs,
      device=env.device,
      field_name="saturation_effort",
    ),
    _resolve_named_values(
      G1_TORQUE_SPEED_VELOCITY_LIMIT,
      target_names=asset.joint_names,
      num_envs=env.num_envs,
      device=env.device,
      field_name="velocity_limit",
    ),
  )


def evaluate_one(
  *,
  policy_name: str,
  checkpoint: Path,
  num_envs: int,
  duration_s: float,
  seed: int,
  device: str,
  saturation_threshold: float = DEFAULT_SATURATION_THRESHOLD,
  save_per_env: bool = False,
) -> dict[str, Any]:
  from mjlab.rl import RslRlVecEnvWrapper

  env = _build_env(num_envs, duration_s, seed, device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
  try:
    policy, checkpoint_type = _load_policy(checkpoint, wrapped, device)
    obs = wrapped.get_observations()
    num_steps = int(round(duration_s / env.step_dt))

    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    failure_time = torch.full(
      (num_envs,),
      float(duration_s),
      dtype=torch.float,
      device=device,
    )
    term_names = list(env.termination_manager.active_terms)
    failure_counts = {name: 0 for name in term_names}

    xy_sq_sum = torch.zeros(num_envs, dtype=torch.float64, device=device)
    yaw_sq_sum = torch.zeros_like(xy_sq_sum)
    tracking_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    saturation_sum = torch.zeros_like(xy_sq_sum)
    torque_sq_sum = torch.zeros_like(xy_sq_sum)
    effort_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    robot = env.scene["robot"]
    joint_ids = torch.arange(robot.num_joints, dtype=torch.long, device=device)
    saturation_effort, velocity_limit = _g1_dynamic_limit_tensors(env, robot)

    with torch.inference_mode():
      for step_idx in range(num_steps):
        action = policy(obs)
        obs, _, dones, _ = wrapped.step(action)
        done = dones.to(torch.bool)
        pre_step_alive = alive.clone()
        newly_failed = alive & done

        if newly_failed.any():
          failure_time[newly_failed] = (step_idx + 1) * env.step_dt
          for name in term_names:
            reason_mask = env.termination_manager.get_term(name).to(torch.bool)
            failure_counts[name] += int((newly_failed & reason_mask).sum().item())

        metric_alive = pre_step_alive & ~done
        command = env.command_manager.get_command("twist")
        if command is None:
          raise RuntimeError("Velocity evaluation requires the twist command.")
        actual_lin = robot.data.root_link_lin_vel_b
        actual_yaw = robot.data.root_link_ang_vel_b[:, 2]
        xy_sq = torch.sum((command[:, :2] - actual_lin[:, :2]) ** 2, dim=1)
        yaw_sq = (command[:, 2] - actual_yaw) ** 2
        tracking_finite = metric_alive & torch.isfinite(xy_sq) & torch.isfinite(yaw_sq)
        xy_sq_sum[tracking_finite] += xy_sq[tracking_finite].double()
        yaw_sq_sum[tracking_finite] += yaw_sq[tracking_finite].double()
        tracking_count[tracking_finite] += 1

        torques = robot.data.qfrc_actuator[:, joint_ids]
        joint_vel = robot.data.joint_vel[:, joint_ids]
        sat_frac = dynamic_saturation_fraction(
          torques,
          joint_vel,
          saturation_effort,
          velocity_limit,
          threshold=saturation_threshold,
        )
        torque_sq = sum_squared_torques(torques)
        effort_finite = (
          metric_alive & torch.isfinite(sat_frac) & torch.isfinite(torque_sq)
        )
        saturation_sum[effort_finite] += sat_frac[effort_finite].double()
        torque_sq_sum[effort_finite] += torque_sq[effort_finite].double()
        effort_count[effort_finite] += 1

        alive &= ~done

    failed = ~alive
    failed_times = failure_time[failed]
    num_survivors = int(alive.sum().item())
    num_failed = int(failed.sum().item())
    failure_time_np = failure_time.detach().cpu().numpy().astype(np.float64)
    survival_ci_low, survival_ci_high = _wilson_ci(num_survivors, num_envs)

    result = {
      "policy": policy_name,
      "checkpoint": checkpoint,
      "checkpoint_type": checkpoint_type,
      "num_envs": int(num_envs),
      "duration_s": float(duration_s),
      "seed": int(seed),
      "device": device,
      "saturation_threshold": float(saturation_threshold),
      "steps_requested": int(num_steps),
      "steps_executed": int(num_steps),
      "survival_rate": float(num_survivors / max(num_envs, 1)),
      "survival_rate_ci95_low": survival_ci_low,
      "survival_rate_ci95_high": survival_ci_high,
      "num_survivors": num_survivors,
      "num_failed": num_failed,
      "failure_counts": failure_counts,
      "mean_failure_time_s": float(np.mean(failure_time_np)),
      "median_failure_time_s": float(np.median(failure_time_np)),
      "failed_mean_failure_time_s": (
        float(failed_times.mean().item()) if failed.any() else float("nan")
      ),
      "failed_median_failure_time_s": (
        float(failed_times.median().item()) if failed.any() else float("nan")
      ),
      **_value_stats(failure_time_np, "failure_time_s", seed=seed + 11),
      **_rmse_stats(
        xy_sq_sum,
        tracking_count,
        "alive_xy_velocity_rmse",
        seed=seed + 101,
      ),
      **_rmse_stats(
        yaw_sq_sum,
        tracking_count,
        "alive_yaw_rate_rmse",
        seed=seed + 102,
      ),
      **_mean_accumulator_stats(
        saturation_sum,
        effort_count,
        "dynamic_sat_frac_all_joints_0p95",
        seed=seed + 103,
      ),
      **_mean_accumulator_stats(
        torque_sq_sum,
        effort_count,
        "mean_torque_squared",
        seed=seed + 104,
      ),
      "tracking_sample_count": int(tracking_count.sum().item()),
      "effort_sample_count": int(effort_count.sum().item()),
    }

    if save_per_env:
      count_np = tracking_count.detach().cpu().numpy()
      effort_count_np = effort_count.detach().cpu().numpy()
      result["per_env"] = {
        "failure_time_s": failure_time_np,
        "alive_xy_velocity_rmse": np.sqrt(
          np.divide(
            xy_sq_sum.detach().cpu().numpy(),
            count_np,
            out=np.full(num_envs, np.nan, dtype=np.float64),
            where=count_np > 0,
          )
        ),
        "alive_yaw_rate_rmse": np.sqrt(
          np.divide(
            yaw_sq_sum.detach().cpu().numpy(),
            count_np,
            out=np.full(num_envs, np.nan, dtype=np.float64),
            where=count_np > 0,
          )
        ),
        "dynamic_sat_frac_all_joints_0p95": np.divide(
          saturation_sum.detach().cpu().numpy(),
          effort_count_np,
          out=np.full(num_envs, np.nan, dtype=np.float64),
          where=effort_count_np > 0,
        ),
        "mean_torque_squared": np.divide(
          torque_sq_sum.detach().cpu().numpy(),
          effort_count_np,
          out=np.full(num_envs, np.nan, dtype=np.float64),
          where=effort_count_np > 0,
        ),
      }

    return result
  finally:
    env.close()


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
  return {
    "policy": result["policy"],
    "checkpoint_type": result["checkpoint_type"],
    "survival_rate": result["survival_rate"],
    "survival_rate_ci95_low": result["survival_rate_ci95_low"],
    "survival_rate_ci95_high": result["survival_rate_ci95_high"],
    "num_failed": result["num_failed"],
    "mean_failure_time_s": result["mean_failure_time_s"],
    "median_failure_time_s": result["median_failure_time_s"],
    "failed_mean_failure_time_s": result["failed_mean_failure_time_s"],
    "alive_xy_velocity_rmse": result["alive_xy_velocity_rmse"],
    "alive_xy_velocity_rmse_mean": result["alive_xy_velocity_rmse_mean"],
    "alive_xy_velocity_rmse_std": result["alive_xy_velocity_rmse_std"],
    "alive_yaw_rate_rmse": result["alive_yaw_rate_rmse"],
    "alive_yaw_rate_rmse_mean": result["alive_yaw_rate_rmse_mean"],
    "alive_yaw_rate_rmse_std": result["alive_yaw_rate_rmse_std"],
    "dynamic_sat_frac_all_joints_0p95": result["dynamic_sat_frac_all_joints_0p95"],
    "dynamic_sat_frac_all_joints_0p95_mean": result[
      "dynamic_sat_frac_all_joints_0p95_mean"
    ],
    "dynamic_sat_frac_all_joints_0p95_std": result[
      "dynamic_sat_frac_all_joints_0p95_std"
    ],
    "mean_torque_squared": result["mean_torque_squared"],
    "mean_torque_squared_mean": result["mean_torque_squared_mean"],
    "mean_torque_squared_std": result["mean_torque_squared_std"],
    "tracking_sample_count": result["tracking_sample_count"],
    "effort_sample_count": result["effort_sample_count"],
    "checkpoint": str(result["checkpoint"]),
  }


def _parse_label_path_specs(specs: list[str], flag_name: str) -> dict[str, str]:
  parsed: dict[str, str] = {}
  for spec in specs:
    if "=" not in spec:
      raise ValueError(f"Invalid {flag_name} {spec!r}; expected POLICY=PATH.")
    label, path = spec.split("=", 1)
    if not label or not path:
      raise ValueError(f"Invalid {flag_name} {spec!r}; expected POLICY=PATH.")
    parsed[label] = path
  return parsed


def _manifest_row(label: str, checkpoint_spec: str) -> dict[str, Any]:
  checkpoint = _resolve_checkpoint(checkpoint_spec)
  return {
    "label": label,
    "run": checkpoint.parent.name,
    "checkpoint": str(checkpoint),
  }


def write_checkpoint_manifest(
  policies: dict[str, str],
  manifest_path: Path,
) -> None:
  rows = [_manifest_row(label, path) for label, path in policies.items()]
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  with manifest_path.open("w", newline="") as f:
    writer = csv.DictWriter(
      f,
      fieldnames=["label", "run", "checkpoint"],
    )
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--duration-s", type=float, default=20.0)
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--device", type=str, default="cuda:0")
  parser.add_argument(
    "--saturation-threshold",
    type=float,
    default=DEFAULT_SATURATION_THRESHOLD,
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
  )
  parser.add_argument(
    "--checkpoint",
    action="append",
    default=[],
    metavar="POLICY=PATH",
    help="Checkpoint path for a policy label. May be repeated.",
  )
  parser.add_argument(
    "--manifest-output",
    type=Path,
    default=None,
    help="Optional CSV manifest path for checkpoint metadata.",
  )
  parser.add_argument(
    "--manifest-only",
    action="store_true",
    help="Write --manifest-output and exit without running eval.",
  )
  parser.add_argument(
    "--save-per-env",
    action="store_true",
    help="Include per-env diagnostic arrays in each JSON output.",
  )
  args = parser.parse_args()

  try:
    policies = _parse_label_path_specs(args.checkpoint, "--checkpoint")
  except ValueError as exc:
    parser.error(str(exc))
  if not policies:
    parser.error("At least one --checkpoint POLICY=PATH is required.")

  if args.manifest_output is not None:
    write_checkpoint_manifest(policies, args.manifest_output)
    print(f"[Manifest] Wrote {len(policies)} rows to {args.manifest_output}")
  elif args.manifest_only:
    parser.error("--manifest-only requires --manifest-output.")

  if args.manifest_only:
    return

  args.output_dir.mkdir(parents=True, exist_ok=True)
  results = []
  for policy_name, checkpoint_spec in policies.items():
    checkpoint = _resolve_checkpoint(checkpoint_spec)
    print(f"[Eval] {policy_name}: {checkpoint}", flush=True)
    result = evaluate_one(
      policy_name=policy_name,
      checkpoint=checkpoint,
      num_envs=args.num_envs,
      duration_s=args.duration_s,
      seed=args.seed,
      device=args.device,
      saturation_threshold=args.saturation_threshold,
      save_per_env=args.save_per_env,
    )
    results.append(result)
    out_path = args.output_dir / f"{policy_name}.json"
    out_path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    print(
      "[Eval] "
      f"survival={result['survival_rate']:.3f} "
      f"xy_rmse={result['alive_xy_velocity_rmse']:.3f} "
      f"yaw_rmse={result['alive_yaw_rate_rmse']:.3f} "
      f"dyn_sat={result['dynamic_sat_frac_all_joints_0p95']:.5f} "
      f"torque_sq={result['mean_torque_squared']:.2f} "
      f"failed={result['num_failed']}",
      flush=True,
    )

  summary_path = args.output_dir / "summary.csv"
  rows = [_summary_row(result) for result in results]
  if rows:
    with summary_path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=list(rows[0]))
      writer.writeheader()
      writer.writerows(rows)

  print(f"[Eval] Wrote {len(results)} evals to {args.output_dir}", flush=True)


if __name__ == "__main__":
  main()
