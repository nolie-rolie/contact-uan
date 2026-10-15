#!/usr/bin/env python3
"""Evaluate MJLab WTW policies with common tracking and energy metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.wtw import mdp as wtw_mdp
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

DEFAULT_TASK = "Mjlab-WTW-Unitree-G1"


def str_to_bool(value: str | bool) -> bool:
  if isinstance(value, bool):
    return value
  normalized = value.strip().lower()
  if normalized in {"1", "true", "t", "yes", "y", "on"}:
    return True
  if normalized in {"0", "false", "f", "no", "n", "off"}:
    return False
  raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default=DEFAULT_TASK)
  parser.add_argument(
    "--checkpoint", required=True, help="WTW policy checkpoint (.pt)."
  )
  parser.add_argument("--num_envs", type=int, default=1024)
  parser.add_argument("--duration_s", type=float, default=20.0)
  parser.add_argument("--device", default=None)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--video", type=str_to_bool, nargs="?", const=True, default=False)
  parser.add_argument("--video_length", type=int, default=600)
  parser.add_argument("--video_count", type=int, default=3)
  parser.add_argument("--output_dir", default=None)
  return parser.parse_args()


def safe_name(value: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def default_output_dir(checkpoint: Path, task: str) -> Path:
  return checkpoint.parent / "eval_wtw" / f"{safe_name(task)}_{checkpoint.stem}"


def raw_reward_term(env: ManagerBasedRlEnv, term_name: str) -> torch.Tensor:
  term_cfg = env.reward_manager.get_term_cfg(term_name)
  if term_cfg.weight == 0.0:
    raise ValueError(f"Reward term {term_name!r} has zero weight.")
  term_index = env.reward_manager.active_terms.index(term_name)
  return env.reward_manager._step_reward[:, term_index] / float(term_cfg.weight)


def tensor_mean_or_nan(values: list[torch.Tensor]) -> float:
  if not values:
    return float("nan")
  flat = torch.cat([v.reshape(-1) for v in values])
  if flat.numel() == 0:
    return float("nan")
  return float(flat.mean().item())


def write_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")


def print_summary(summary: dict[str, Any]) -> None:
  rows = [
    ("task", summary["task"]),
    ("checkpoint", summary["checkpoint"]),
    ("num_envs", summary["num_envs"]),
    ("duration_s", f"{summary['duration_s']:.3f}"),
    ("linear_velocity_tracking", f"{summary['linear_velocity_tracking']:.6f}"),
    ("angular_velocity_tracking", f"{summary['angular_velocity_tracking']:.6f}"),
    ("survival_rate", f"{summary['survival_rate']:.6f}"),
    ("energy_mean_power", f"{summary['energy_mean_power']:.6f}"),
    ("energy_integrated", f"{summary['energy_integrated']:.6f}"),
  ]
  width = max(len(k) for k, _ in rows)
  print("\n[WTW Eval]")
  for key, value in rows:
    print(f"  {key:<{width}} : {value}")
  print(
    f"  summary_json{'':<{width - len('summary_json')}} : {summary['summary_json']}"
  )
  if summary["videos"]:
    print("  videos:")
    for path in summary["videos"]:
      print(f"    {path}")


def main() -> None:
  args = parse_args()

  checkpoint = Path(args.checkpoint).expanduser().resolve()
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
  if args.num_envs <= 0:
    raise ValueError("--num_envs must be positive.")
  if args.duration_s <= 0.0:
    raise ValueError("--duration_s must be positive.")
  if args.video and (args.video_length <= 0 or args.video_count <= 0):
    raise ValueError(
      "--video_length and --video_count must be positive when video is enabled."
    )

  # Import tasks after CLI parsing so externally set UAN_CHECKPOINT_PATH is already
  # visible before task registration runs.
  import mjlab.tasks  # noqa: F401

  configure_torch_backends()
  torch.manual_seed(int(args.seed))
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  output_dir = (
    Path(args.output_dir).expanduser().resolve()
    if args.output_dir is not None
    else default_output_dir(checkpoint, args.task)
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = int(args.num_envs)
  env_cfg.seed = int(args.seed)
  # Keep non-timeout terminations active for survival, but avoid artificial timeouts
  # during the requested eval window.
  env_cfg.episode_length_s = float(args.duration_s) + 1.0
  if args.video:
    env_cfg.viewer.width = 1280
    env_cfg.viewer.height = 720

  render_mode = "rgb_array" if args.video else None
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  if args.video:
    video_dir = output_dir / "videos"
    env_for_rl = VideoRecorder(
      base_env,
      video_folder=video_dir,
      step_trigger=lambda step: step % int(args.video_length) == 0
      and step < int(args.video_length) * int(args.video_count),
      video_length=int(args.video_length),
      name_prefix=f"eval-wtw-{checkpoint.stem}",
      disable_logger=False,
    )
  else:
    video_dir = None
    env_for_rl = base_env

  env = RslRlVecEnvWrapper(env_for_rl, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  total_steps = int(math.ceil(float(args.duration_s) / env.unwrapped.step_dt))
  alive = torch.ones(env.num_envs, device=env.unwrapped.device, dtype=torch.bool)
  lin_values: list[torch.Tensor] = []
  ang_values: list[torch.Tensor] = []
  energy_values: list[torch.Tensor] = []
  reward_values: list[torch.Tensor] = []

  obs = env.get_observations()
  with torch.inference_mode():
    for step in range(total_steps):
      actions = policy(obs)
      obs, reward, dones, extras = env.step(actions)
      base = env.unwrapped
      time_outs = extras.get("time_outs")
      if time_outs is None:
        time_outs = torch.zeros_like(dones, dtype=torch.bool)
      else:
        time_outs = time_outs.to(device=base.device, dtype=torch.bool)
      failures = dones.to(device=base.device, dtype=torch.bool) & ~time_outs

      alive_after_step = alive & ~failures
      if alive_after_step.any():
        lin_values.append(
          raw_reward_term(base, "track_linear_velocity")[alive_after_step].detach()
        )
        ang_values.append(
          raw_reward_term(base, "track_angular_velocity")[alive_after_step].detach()
        )
        energy_values.append(
          wtw_mdp.power_penalty(base, command_name="twist")[alive_after_step].detach()
        )
        reward_values.append(reward.to(device=base.device)[alive_after_step].detach())
      alive = alive_after_step

      if (step + 1) % max(1, total_steps // 10) == 0 or step + 1 == total_steps:
        print(
          f"[Eval] step={step + 1}/{total_steps} "
          f"survival={alive.float().mean().item():.3f} "
          f"lin={tensor_mean_or_nan(lin_values):.4f} "
          f"ang={tensor_mean_or_nan(ang_values):.4f}"
        )

  env.close()

  energy_mean_power = tensor_mean_or_nan(energy_values)
  summary_path = output_dir / "summary.json"
  videos = (
    [str(p) for p in sorted((video_dir or output_dir).glob("*.mp4"))]
    if args.video
    else []
  )
  summary = {
    "task": args.task,
    "checkpoint": str(checkpoint),
    "uan_checkpoint_env": os.environ.get("UAN_CHECKPOINT_PATH", ""),
    "num_envs": int(args.num_envs),
    "duration_s": float(args.duration_s),
    "step_dt": float(env.unwrapped.step_dt),
    "total_steps": int(total_steps),
    "seed": int(args.seed),
    "device": str(device),
    "linear_velocity_tracking": tensor_mean_or_nan(lin_values),
    "angular_velocity_tracking": tensor_mean_or_nan(ang_values),
    "survival_rate": float(alive.float().mean().item()),
    "energy_mean_power": energy_mean_power,
    "energy_integrated": float(energy_mean_power * float(args.duration_s)),
    "mean_reward": tensor_mean_or_nan(reward_values),
    "alive_sample_count": int(sum(v.numel() for v in lin_values)),
    "summary_json": str(summary_path),
    "videos": videos,
  }
  write_json(summary_path, summary)
  print_summary(summary)


if __name__ == "__main__":
  main()
