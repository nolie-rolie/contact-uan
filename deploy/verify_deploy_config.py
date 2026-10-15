"""Validate deploy-time config resolution against an MJLab env.yaml.

Examples:
    uv run --locked python deploy/verify_deploy_config.py \
        --policy /path/to/your/wtw-policy.jit \
        --policy-joint-order mujoco

    uv run --locked python deploy/verify_deploy_config.py \
        --policy /path/to/your/velocity-policy.jit \
        --policy-joint-order mujoco
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy.base_controller import (
  _load_policy_common_step_counter,
  _load_relaxed_yaml,
  parse_hardware_cfg,
  resolve_action_scale_array,
  resolve_policy_path,
  validate_runtime_config,
)
from deploy.robots.g1_config import G1Config


def _resolve_path(path: str) -> str:
  if os.path.isabs(path):
    return path

  base_dir = os.path.dirname(os.path.realpath(__file__))
  candidates = [
    os.path.abspath(path),
    os.path.abspath(os.path.join(base_dir, path)),
    os.path.abspath(os.path.join(base_dir, "models", path)),
  ]
  for candidate in candidates:
    if os.path.exists(candidate):
      return candidate
  return candidates[-1]


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--policy",
    help="Path to .jit/.pt policy file. env.yaml is inferred from its parent run directory.",
  )
  parser.add_argument(
    "--env-yaml",
    help="Direct path to env.yaml. Use this instead of --policy if preferred.",
  )
  parser.add_argument(
    "--policy-joint-order",
    required=True,
    choices=("isaaclab", "mujoco"),
    help="Joint order expected by the deploy script for this policy.",
  )
  parser.add_argument(
    "--policy-common-step-counter",
    type=int,
    help=(
      "Optional checkpoint common_step_counter override. Useful with "
      "--env-yaml when the task uses a command curriculum."
    ),
  )
  return parser.parse_args()


def main():
  args = parse_args()
  if bool(args.policy) == bool(args.env_yaml):
    raise SystemExit("Pass exactly one of --policy or --env-yaml.")

  if args.policy:
    policy_path = resolve_policy_path(args.policy)
    env_yaml_path = os.path.join(os.path.dirname(policy_path), "params", "env.yaml")
    policy_common_step_counter = _load_policy_common_step_counter(policy_path)
  else:
    env_yaml_path = _resolve_path(args.env_yaml)
    policy_common_step_counter = args.policy_common_step_counter

  env_cfg_raw = _load_relaxed_yaml(env_yaml_path)
  cfg = parse_hardware_cfg(
    env_cfg_raw,
    policy_common_step_counter=policy_common_step_counter,
  )
  robot = G1Config(dt=cfg["control_dt"])
  robot.load_from_cfg(cfg)
  cfg["control"]["action_scale"] = resolve_action_scale_array(
    cfg["control"]["action_scale"],
    robot.joint_names(args.policy_joint_order),
  )

  errors = validate_runtime_config(
    env_cfg_raw,
    cfg,
    robot,
    args.policy_joint_order,
    policy_common_step_counter=policy_common_step_counter,
  )

  print(f"env.yaml: {env_yaml_path}")
  print(f"policy_joint_order: {args.policy_joint_order}")
  print(f"policy_common_step_counter: {policy_common_step_counter}")
  print(
    "resolved summary: "
    f"history={cfg['history_length']} "
    f"control_dt={cfg['control_dt']} "
    f"offset_actions={cfg['offset_actions']} "
    f"base_lin_vel_scale={cfg.get('base_lin_vel_scale', 1.0)} "
    f"ang_vel_scale={cfg['ang_vel_scale']} "
    f"dof_vel_scale={cfg['dof_vel_scale']} "
    f"force_scale={cfg.get('force_scale')}"
  )
  print(f"resolved command_ranges: {cfg.get('command_ranges')}")

  if errors:
    print("verification: FAIL")
    for error in errors:
      print(f"- {error}")
    raise SystemExit(1)

  print("verification: PASS")


if __name__ == "__main__":
  main()
