"""Evaluate the separate base_lin state estimator on a plane in mjlab.

Loads state_estimator.jit and rolls out a policy in mjlab, building the
proprio history buffer exactly the way collect_estimator_data.py builds it,
then compares the three predicted body-frame linear velocities to simulation.
This rollout diagnostic accepts the base_lin component only.

Usage (from mjlab repo root):
    uv run python scripts/uan_pipeline/eval_estimator_mjlab.py \\
        --checkpoint_dir "logs/rsl_rl/g1_wtw/<run>/" \\
        --iteration 30000 \\
        --estimator_dir "<estimator_root>/base_lin" \\
        --num_envs 4096 --seconds 30.0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import cast

import torch

sys.path.insert(0, os.path.dirname(__file__))

from collect_estimator_data import apply_terrain_override
from estimator_bundle import validate_component_meta
from mujoco_sim_helpers import quat_rotate_inverse


def validate_estimator_meta(meta: dict, meta_path: str) -> None:
  """Require the supported three-output base-velocity component."""
  validate_component_meta(meta, "base_lin", Path(meta_path))


def read_base_linear_velocity_targets(robot_data) -> torch.Tensor:
  """Return the three body-frame velocity labels used to train base_lin."""
  return quat_rotate_inverse(
    robot_data.root_link_quat_w, robot_data.root_link_lin_vel_w
  )


def parse_args(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint_dir", type=str, required=True)
  parser.add_argument("--iteration", type=int, required=True)
  parser.add_argument("--estimator_dir", type=str, required=True)
  parser.add_argument("--num_envs", type=int, default=4096)
  parser.add_argument("--seconds", type=float, default=30.0)
  parser.add_argument("--device", type=str, default="cuda:0")
  parser.add_argument(
    "--terrain",
    choices=("plane",),
    default="plane",
    help="Plane terrain matching estimator training data.",
  )
  return parser.parse_args(argv)


def load_policy(checkpoint_path, device):
  """Load a policy and apply its saved observation normalization."""
  if checkpoint_path.endswith(".jit"):
    policy = torch.jit.load(checkpoint_path, map_location=device)
    policy.eval()
    return policy

  ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
  sd = ckpt["actor_state_dict"]
  import torch.nn as nn
  from rsl_rl.modules import MLP

  linear_keys = sorted(k for k in sd if k.startswith("mlp.") and k.endswith(".weight"))
  obs_dim = sd[linear_keys[0]].shape[1]
  hidden_dims = [sd[k].shape[0] for k in linear_keys[:-1]]
  action_dim = sd[linear_keys[-1]].shape[0]

  mlp = MLP(obs_dim, action_dim, hidden_dims, "elu")
  mlp_sd = {k[len("mlp.") :]: v for k, v in sd.items() if k.startswith("mlp.")}
  mlp.load_state_dict(mlp_sd)
  mlp.eval()
  mlp.to(device)

  obs_mean = sd.get("obs_normalizer._mean")
  obs_std = sd.get("obs_normalizer._std")

  class _NormalizedPolicy(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mlp, mean, std):
      super().__init__()
      self.mlp = mlp
      self.has_norm = mean is not None
      if mean is not None:
        self.register_buffer("mean", mean.to(device))
        self.register_buffer("std", std.to(device))
      else:
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("std", torch.ones(1))

    def forward(self, x):
      if self.has_norm:
        x = (x - self.mean) / (self.std + 1e-8)
      return self.mlp(x)

  policy = _NormalizedPolicy(mlp, obs_mean, obs_std).to(device).eval()
  print(f"  obs_normalizer applied: {policy.has_norm}")
  return policy


def main():
  args = parse_args()
  device = torch.device(args.device)
  os.environ.setdefault("MUJOCO_GL", "egl")

  # --- Load estimator ---
  est_jit_path = os.path.join(args.estimator_dir, "state_estimator.jit")
  est_meta_path = os.path.join(args.estimator_dir, "estimator_meta.json")
  with open(est_meta_path) as f:
    meta = json.load(f)
  validate_estimator_meta(meta, est_meta_path)
  estimator = torch.jit.load(est_jit_path, map_location=device)
  estimator.eval()
  H = meta["history_length"]
  P = meta["proprio_dim"]
  output_dim = int(meta["output_dim"])
  target_names = list(meta["target_layout"])
  scales = meta.get("proprio_scales", [0.25, 1.0, 1.0, 0.05, 1.0])
  SCALE_ANG_VEL = scales[0]
  SCALE_JOINT_VEL = scales[3]
  print(f"[Estimator] H={H}, P={P}, scales={scales}")
  print("[Estimator] training val RMSE:")
  for k, v in meta.get("per_component_rmse", {}).items():
    print(f"  {k:>12s}: {v:.6f}")

  # --- Load policy ---
  # Prefer .pt (mjlab-native, MJ joint order). The .jit file may have been
  # exported with IL↔MJ wrapping for use in IsaacLab — that would feed the
  # mjlab env IL-order outputs and cause the policy to fail.
  pt_path = os.path.join(args.checkpoint_dir, f"model_{args.iteration}.pt")
  jit_path = os.path.join(args.checkpoint_dir, f"model_{args.iteration}.jit")
  if os.path.exists(pt_path):
    policy = load_policy(pt_path, device)
    print(f"[Policy] Loaded {pt_path} (MJ-native)")
  elif os.path.exists(jit_path):
    policy = load_policy(jit_path, device)
    print(f"[Policy] Loaded {jit_path}")
  else:
    raise FileNotFoundError(f"No checkpoint for iter {args.iteration}")

  # --- Build env ---
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.envs.mdp.actions.actions import JointPositionAction
  from mjlab.tasks.wtw.wtw_env_cfg import make_wtw_env_cfg

  env_cfg = make_wtw_env_cfg()
  terrain_meta = apply_terrain_override(env_cfg, args.terrain)
  env_cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  N = env.num_envs
  total_steps = int(args.seconds / env.step_dt)
  joint_action_term = cast(
    JointPositionAction, env.action_manager.get_term("joint_pos")
  )
  print(
    f"[Env] N={N}, steps={total_steps}, dt={env.step_dt:.4f}s, "
    f"terrain={terrain_meta['terrain_type']} (override={terrain_meta['override']})"
  )

  robot_entity = env.scene.entities["robot"]
  robot_data = robot_entity.data
  ACTION_DIM = 29
  # --- Rollout with estimator ---
  obs_dict, _ = env.reset()
  prev_q_des = torch.zeros(N, ACTION_DIM, device=device)
  proprio_buf = torch.zeros(N, H, P, device=device)

  sq_err_sum = torch.zeros(output_dim, device=device)
  n_samples = 0
  term_count = 0

  for step in range(total_steps):
    # --- Build noisy proprio ---
    # Match collect_estimator_data.py exactly (training-time convention).
    quat_w = robot_data.root_link_quat_w
    ang_vel_w = robot_data.root_link_ang_vel_w
    ang_vel_b = quat_rotate_inverse(quat_w, ang_vel_w)
    proj_grav = robot_data.projected_gravity_b
    joint_pos = robot_data.joint_pos  # absolute
    joint_vel = robot_data.joint_vel

    # Training adds noise to proprio — mirror that so distribution matches.
    ang_vel_n = ang_vel_b + torch.empty_like(ang_vel_b).uniform_(-0.2, 0.2)
    proj_grav_n = proj_grav + torch.empty_like(proj_grav).uniform_(-0.01, 0.01)
    jpos_n = joint_pos + torch.empty_like(joint_pos).uniform_(-0.01, 0.01)
    jvel_n = joint_vel + torch.empty_like(joint_vel).uniform_(-1.5, 1.5)

    proprio_step = torch.cat(
      [
        ang_vel_n * SCALE_ANG_VEL,
        proj_grav_n,
        jpos_n,
        jvel_n * SCALE_JOINT_VEL,
        prev_q_des,
      ],
      dim=-1,
    )  # (N, 93)

    # --- Targets (ground truth) ---
    targets = read_base_linear_velocity_targets(robot_data)

    # Update rolling buffer
    proprio_buf = torch.roll(proprio_buf, shifts=-1, dims=1)
    proprio_buf[:, -1] = proprio_step

    # Run estimator (once buffer is full)
    if step >= H - 1:
      with torch.no_grad():
        flat_buf = proprio_buf.reshape(N, -1)
        est_out = estimator(flat_buf)
      sq_err = (est_out - targets) ** 2
      sq_err_sum += sq_err.sum(dim=0)
      n_samples += N

    # Step env
    actor_obs = obs_dict["actor"]
    with torch.no_grad():
      actions = policy(actor_obs)
    obs_dict, _, terminated, truncated, _ = env.step(actions)
    prev_q_des = joint_action_term._processed_actions.clone()

    dones = terminated | truncated
    if dones.any():
      proprio_buf[dones] = 0.0
      prev_q_des[dones] = 0.0
      term_count += dones.sum().item()

    if (step + 1) % 200 == 0:
      print(f"  step {step + 1}/{total_steps}")

  # --- Report ---
  rmse = torch.sqrt(sq_err_sum / n_samples).cpu()
  print("\n" + "=" * 60)
  print(f"Estimator RMSE on mjlab rollout ({n_samples:,} samples):")
  print("=" * 60)
  for name, v in zip(target_names, rmse, strict=True):
    print(f"  {name:>12s}: {v:.6f}")
  print(f"         mean: {rmse.mean():.6f}")
  print(f"  terminations: {term_count}")
  print("=" * 60)


if __name__ == "__main__":
  main()
