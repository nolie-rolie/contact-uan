"""Collect (noisy_proprio, ground_truth) pairs for training state estimators.

Rolls out trained WTW policies at multiple checkpoints in mjlab, recording:
  - Noisy proprioceptive observations (what a real robot would see)
  - Targets for all six independent Contact-UAN estimators: base velocity,
    foot contact, stance-foot height, and stance-foot orientation.

Checkpoints are split across envs so all policies run in parallel in a single pass.

Uniform observation noise is added before scaling:
  ang_vel:   ±0.2 rad/s  (uniform, before ×0.25 scale)
  proj_grav: ±0.05       (uniform)
  joint_pos: ±0.01 rad   (uniform)
  joint_vel: ±1.5 rad/s  (uniform, before ×0.05 scale)

Output: metadata .pt file and memory-mapped arrays with three-frame
proprioceptive histories (279 inputs) and all required labels. Angular velocity,
root height, and foot loads are retained as dataset diagnostics.

Usage (from repo root):
    uv run python scripts/uan_pipeline/collect_estimator_data.py \
        --checkpoint_dir "logs/rsl_rl/g1_wtw/<run>/" \
        --iterations 500,2000,5000 \
        --num_envs 4096 --seconds 20.0 \
        --output estimator_training_data.pt
"""

import argparse
import os
import sys
from typing import Any, cast

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from estimator_config import (
  HISTORY_LENGTH,
  PROPRIO_ACTION_REPRESENTATION,
  PROPRIO_DIM,
  PROPRIO_TERMS,
)
from mujoco_sim_helpers import quat_rotate_inverse

from mjlab.utils.lab_api.math import quat_apply_inverse

# Uniform observation noise amplitudes before scaling.
NOISE_ANG_VEL = 0.2  # ±0.2 rad/s
NOISE_PROJ_GRAV = 0.05  # ±0.05
NOISE_JOINT_POS = 0.01  # ±0.01 rad
NOISE_JOINT_VEL = 1.5  # ±1.5 rad/s

# Observation scales matching WTW actor terms
SCALE_ANG_VEL = 0.25
SCALE_JOINT_VEL = 0.05

STATE_TARGET_NAMES = [
  "lin_vel_x",
  "lin_vel_y",
  "lin_vel_z",
  "ang_vel_x",
  "ang_vel_y",
  "ang_vel_z",
  "root_height",
]
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_HEIGHT_TARGET_NAMES = ["left_foot_height", "right_foot_height"]
FOOT_HEIGHT_TARGET = "body_link_world_z"
PROJECTED_GRAVITY_TARGET_NAMES = ["proj_grav_x", "proj_grav_y", "proj_grav_z"]
JOINT_VEL_TARGET_NAMES = [f"joint_vel_{i:02d}" for i in range(29)]
FOOT_CONTACT_TARGET_NAMES = ["left_foot_contact", "right_foot_contact"]
FOOT_LOG_LOAD_TARGET_NAMES = ["left_foot_log_load", "right_foot_log_load"]
FOOT_CONTACT_LOAD_TARGET_NAMES = FOOT_CONTACT_TARGET_NAMES + FOOT_LOG_LOAD_TARGET_NAMES
STANCE_FOOT_PROJECTED_GRAVITY_TARGET_NAMES = [
  "left_foot_proj_grav_x",
  "left_foot_proj_grav_y",
  "left_foot_proj_grav_z",
  "right_foot_proj_grav_x",
  "right_foot_proj_grav_y",
  "right_foot_proj_grav_z",
]
STANCE_FOOT_PROJECTED_GRAVITY_TARGET = "body_link_projected_gravity"
FOOT_CONTACT_LOAD_SENSOR_NAME = "feet_ground_contact"
FOOT_CONTACT_LOAD_THRESHOLD_N = 10.0
GRAVITY_MPS2 = 9.81
ESTIMATOR_TERRAIN = "plane"


def make_state_target_layout(
  include_foot_heights: bool = True,
  include_projected_gravity: bool = False,
  include_joint_vel: bool = False,
  include_foot_contact_load: bool = False,
  include_stance_foot_orientation: bool = False,
) -> list[str]:
  names = list(STATE_TARGET_NAMES)
  if include_foot_heights:
    names.extend(FOOT_HEIGHT_TARGET_NAMES)
  if include_projected_gravity:
    names.extend(PROJECTED_GRAVITY_TARGET_NAMES)
  if include_joint_vel:
    names.extend(JOINT_VEL_TARGET_NAMES)
  if include_foot_contact_load:
    names.extend(FOOT_CONTACT_LOAD_TARGET_NAMES)
  if include_stance_foot_orientation:
    names.extend(STANCE_FOOT_PROJECTED_GRAVITY_TARGET_NAMES)
  return names


def resolve_foot_body_ids(robot_entity):
  foot_body_ids = robot_entity.find_bodies(list(FOOT_BODY_NAMES))[0]
  if len(foot_body_ids) != len(FOOT_BODY_NAMES):
    raise RuntimeError(
      f"Expected foot bodies {FOOT_BODY_NAMES}, found ids={foot_body_ids}"
    )
  return foot_body_ids


def _repo_root() -> str:
  return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _repo_relative(path: str) -> str | None:
  repo_root = _repo_root()
  path_abs = os.path.abspath(path)
  try:
    if os.path.commonpath([repo_root, path_abs]) != repo_root:
      return None
  except ValueError:
    return None
  return os.path.relpath(path_abs, repo_root)


def _describe_terrain(env_cfg) -> dict:
  terrain = getattr(env_cfg.scene, "terrain", None)
  if terrain is None:
    return {
      "terrain_type": None,
      "terrain_generator": None,
    }
  generator = getattr(terrain, "terrain_generator", None)
  return {
    "terrain_type": getattr(terrain, "terrain_type", None),
    "terrain_generator": None if generator is None else type(generator).__name__,
  }


def apply_terrain_override(env_cfg, terrain: str = ESTIMATOR_TERRAIN) -> dict:
  """Apply the fixed plane terrain used for estimator rollouts."""
  if terrain != ESTIMATOR_TERRAIN:
    raise ValueError(
      f"Estimator data collection uses terrain={ESTIMATOR_TERRAIN!r}; got {terrain!r}."
    )

  from mjlab.terrains import TerrainEntityCfg

  env_cfg.scene.terrain = TerrainEntityCfg(terrain_type=ESTIMATOR_TERRAIN)
  events = getattr(env_cfg, "events", None)
  if events is not None:
    if hasattr(events, "pop"):
      events.pop("randomize_terrain", None)
    elif hasattr(events, "randomize_terrain"):
      events.randomize_terrain = None

  meta = _describe_terrain(env_cfg)
  meta["override"] = ESTIMATOR_TERRAIN
  return meta


def parse_args(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(
    description="Collect estimator training data from mjlab policy rollouts."
  )
  parser.add_argument(
    "--checkpoint_dir",
    type=str,
    required=True,
    help="Directory containing model_<iter>.pt checkpoints.",
  )
  parser.add_argument(
    "--iterations",
    type=str,
    nargs="+",
    required=True,
    help="Checkpoint iterations (e.g. 500 2000 5000 or 500,2000,5000).",
  )
  parser.add_argument("--num_envs", type=int, default=4096)
  parser.add_argument("--seconds", type=float, default=20.0)
  parser.add_argument(
    "--history_length",
    type=int,
    choices=(HISTORY_LENGTH,),
    default=HISTORY_LENGTH,
    help="Three proprioceptive frames, as used by the paper's separate estimators.",
  )
  parser.add_argument("--device", type=str, default="cuda:0")
  parser.add_argument(
    "--terrain",
    choices=(ESTIMATOR_TERRAIN,),
    default=ESTIMATOR_TERRAIN,
    help=(
      "Estimator rollouts use fixed plane terrain. The option is retained "
      "for command compatibility."
    ),
  )
  parser.add_argument(
    "--output",
    type=str,
    default="estimator_training_data.pt",
    help="Output .pt file path.",
  )
  return parser.parse_args(argv)


def load_policy(checkpoint_path, device):
  """Load a WTW policy, including its observation normalization."""
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
        mean = cast(torch.Tensor, self._buffers["mean"])
        std = cast(torch.Tensor, self._buffers["std"])
        x = (x - mean) / (std + 1e-8)
      return self.mlp(x)

  policy = _NormalizedPolicy(mlp, obs_mean, obs_std).to(device).eval()
  print(
    f"  Loaded .pt: obs={obs_dim}, hidden={hidden_dims}, act={action_dim}, "
    f"normalizer={policy.has_norm}"
  )
  return policy


def build_noisy_proprio(robot_data, prev_q_des):
  """Build a noisy proprioceptive observation from raw sensor data.

  Returns: (N, 93) tensor.
  """
  quat_w = robot_data.root_link_quat_w
  ang_vel_w = robot_data.root_link_ang_vel_w
  ang_vel_b = quat_rotate_inverse(quat_w, ang_vel_w)

  proj_grav = robot_data.projected_gravity_b
  joint_pos_abs = robot_data.joint_pos
  joint_vel = robot_data.joint_vel

  ang_vel_noisy = ang_vel_b + torch.empty_like(ang_vel_b).uniform_(
    -NOISE_ANG_VEL, NOISE_ANG_VEL
  )
  proj_grav_noisy = proj_grav + torch.empty_like(proj_grav).uniform_(
    -NOISE_PROJ_GRAV, NOISE_PROJ_GRAV
  )
  joint_pos_noisy = joint_pos_abs + torch.empty_like(joint_pos_abs).uniform_(
    -NOISE_JOINT_POS, NOISE_JOINT_POS
  )
  joint_vel_noisy = joint_vel + torch.empty_like(joint_vel).uniform_(
    -NOISE_JOINT_VEL, NOISE_JOINT_VEL
  )

  return torch.cat(
    [
      ang_vel_noisy * SCALE_ANG_VEL,  # 3
      proj_grav_noisy,  # 3
      joint_pos_noisy,  # 29
      joint_vel_noisy * SCALE_JOINT_VEL,  # 29
      prev_q_des,  # 29 (no noise)
    ],
    dim=-1,
  )


def read_targets(
  robot_data,
  *,
  include_foot_heights: bool = True,
  include_projected_gravity: bool = False,
  include_joint_vel: bool = False,
  include_stance_foot_orientation: bool = False,
  foot_body_ids=None,
):
  """Read ground-truth estimation targets from the simulator.

  Returns either:
    (N, 7) tensor [lin_vel_b(3), ang_vel_b(3), root_height(1)]
    (N, 9+) tensor [..., left_foot_height_w, right_foot_height_w,
                    optional proj_grav(3), optional joint_vel(29)]
  """
  quat_w = robot_data.root_link_quat_w
  lin_vel_w = robot_data.root_link_lin_vel_w
  ang_vel_w = robot_data.root_link_ang_vel_w

  lin_vel_b = quat_rotate_inverse(quat_w, lin_vel_w)
  ang_vel_b = quat_rotate_inverse(quat_w, ang_vel_w)
  root_height = robot_data.root_link_pos_w[:, 2:3]

  targets = [lin_vel_b, ang_vel_b, root_height]
  if include_foot_heights:
    if foot_body_ids is None:
      raise ValueError("foot_body_ids are required when include_foot_heights=True")
    foot_heights = robot_data.body_link_pos_w[:, foot_body_ids, 2]
    targets.append(foot_heights)
  if include_projected_gravity:
    targets.append(robot_data.projected_gravity_b)
  if include_joint_vel:
    targets.append(robot_data.joint_vel)
  if include_stance_foot_orientation:
    if foot_body_ids is None:
      raise ValueError(
        "foot_body_ids are required when include_stance_foot_orientation=True"
      )
    targets.append(read_stance_foot_orientation_targets(robot_data, foot_body_ids))

  return torch.cat(targets, dim=-1)


def read_stance_foot_orientation_targets(robot_data, foot_body_ids) -> torch.Tensor:
  """Read projected gravity in left/right ankle-roll link frames."""
  foot_quat_w = robot_data.body_link_quat_w[:, foot_body_ids]
  gravity_w = robot_data.gravity_vec_w[:, None, :].expand(-1, len(foot_body_ids), -1)
  foot_gravity = quat_apply_inverse(
    foot_quat_w.reshape(-1, 4), gravity_w.reshape(-1, 3)
  ).reshape(foot_quat_w.shape[0], len(foot_body_ids), 3)
  foot_gravity = foot_gravity / foot_gravity.norm(dim=-1, keepdim=True).clamp_min(
    1.0e-8
  )
  return foot_gravity.reshape(foot_gravity.shape[0], -1)


def get_robot_mass_kg(env) -> torch.Tensor:
  """Return per-env robot mass in kg from the current MuJoCo model."""
  body_mass = env.sim.model.body_mass
  if not isinstance(body_mass, torch.Tensor):
    body_mass = torch.as_tensor(body_mass, device=env.device)
  body_mass = body_mass.to(env.device)
  if body_mass.ndim == 1:
    return body_mass.sum().expand(env.num_envs)
  return body_mass.sum(dim=-1)


def read_foot_contact_load_targets(
  env,
  robot_mass_kg: torch.Tensor | None = None,
  *,
  sensor_name: str = FOOT_CONTACT_LOAD_SENSOR_NAME,
  contact_threshold_n: float = FOOT_CONTACT_LOAD_THRESHOLD_N,
) -> torch.Tensor:
  """Read loaded-contact labels and log-normalized foot loads.

  Returns: (N, 4) tensor [left_contact, right_contact,
                          left_log_load, right_log_load].
  """
  sensor = env.scene[sensor_name]
  sensor_data = sensor.data
  if sensor_data.force is None:
    raise RuntimeError(f"Contact sensor {sensor_name!r} does not expose force data.")
  if robot_mass_kg is None:
    robot_mass_kg = get_robot_mass_kg(env)
  force_norm = sensor_data.force.norm(dim=-1)
  contact = (force_norm > contact_threshold_n).float()
  load = force_norm / (robot_mass_kg[:, None].clamp_min(1.0e-6) * GRAVITY_MPS2)
  log_load = torch.log1p(load.clamp_min(0.0))
  return torch.cat([contact, log_load], dim=-1)


def main():
  args = parse_args()
  device = torch.device(args.device)
  iterations = [int(x) for token in args.iterations for x in token.split(",")]

  os.environ.setdefault("MUJOCO_GL", "egl")

  # --- Load all policies first ---
  # Prefer the native checkpoint; exported policies must use MuJoCo joint order.
  policies = []
  valid_iterations = []
  valid_checkpoint_paths = []
  for ckpt_iter in iterations:
    pt_path = os.path.join(args.checkpoint_dir, f"model_{ckpt_iter}.pt")
    jit_path = os.path.join(args.checkpoint_dir, f"model_{ckpt_iter}.jit")

    if os.path.exists(pt_path):
      print(f"[Checkpoint] iter {ckpt_iter} (.pt)")
      policies.append(load_policy(pt_path, device))
      valid_iterations.append(ckpt_iter)
      valid_checkpoint_paths.append(os.path.abspath(pt_path))
    elif os.path.exists(jit_path):
      print(f"[Checkpoint] iter {ckpt_iter} (.jit)")
      policies.append(load_policy(jit_path, device))
      valid_iterations.append(ckpt_iter)
      valid_checkpoint_paths.append(os.path.abspath(jit_path))
    else:
      print(f"[WARNING] Skipping iter {ckpt_iter}: not found")

  if not policies:
    print("[ERROR] No valid checkpoints found!")
    return

  num_policies = len(policies)

  # --- Build environment ---
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.wtw.wtw_env_cfg import make_wtw_env_cfg

  # Retain training randomization and pushes while fixing the ground plane.
  env_cfg = make_wtw_env_cfg()
  terrain_meta = apply_terrain_override(env_cfg, ESTIMATOR_TERRAIN)
  env_cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  N = env.num_envs
  total_steps = int(args.seconds / env.step_dt)

  robot_data = env.scene.entities["robot"].data
  joint_action_term = env.action_manager.get_term("joint_pos")

  # --- Split envs across policies ---
  envs_per_policy = N // num_policies
  policy_env_ranges = []
  for i in range(num_policies):
    start = i * envs_per_policy
    end = (i + 1) * envs_per_policy if i < num_policies - 1 else N
    policy_env_ranges.append((start, end))

  print(f"\n[Config] envs={N}, steps={total_steps}, dt={env.step_dt:.4f}s")
  print(
    f"[Config] task=wtw, terrain={terrain_meta['terrain_type']} "
    f"(override={terrain_meta['override']})"
  )
  print(f"[Config] history_length={args.history_length}")
  print(
    f"[Config] Env assignment: "
    f"{[(it, s, e) for it, (s, e) in zip(valid_iterations, policy_env_ranges, strict=True)]}"
  )

  H = args.history_length
  ACTION_DIM = 29
  include_foot_heights = True
  include_projected_gravity = False
  include_joint_vel = False
  include_foot_contact_load = True
  include_stance_foot_orientation = True
  target_names = make_state_target_layout(
    include_foot_heights=include_foot_heights,
    include_projected_gravity=include_projected_gravity,
    include_joint_vel=include_joint_vel,
    include_foot_contact_load=include_foot_contact_load,
    include_stance_foot_orientation=include_stance_foot_orientation,
  )
  target_dim = len(target_names)
  robot_entity = env.scene.entities["robot"]
  foot_body_ids = (
    resolve_foot_body_ids(robot_entity)
    if include_foot_heights or include_stance_foot_orientation
    else None
  )
  robot_mass_kg = get_robot_mass_kg(env)
  if include_foot_heights:
    print(
      f"[Config] foot_height_targets={FOOT_BODY_NAMES} "
      f"({FOOT_HEIGHT_TARGET}), body_ids={foot_body_ids}"
    )
  else:
    print("[Config] foot_height_targets=disabled (7D target contract)")
  print(
    f"[Config] projected_gravity_targets={include_projected_gravity}, "
    f"joint_vel_targets={include_joint_vel}, "
    f"foot_contact_targets={include_foot_contact_load}, "
    f"stance_foot_orientation_targets={include_stance_foot_orientation}, "
    f"target_dim={target_dim}"
  )
  if include_foot_contact_load:
    if FOOT_CONTACT_LOAD_SENSOR_NAME not in env.scene.sensors:
      raise RuntimeError(
        f"Estimator collection requires contact sensor "
        f"{FOOT_CONTACT_LOAD_SENSOR_NAME!r}; available={list(env.scene.sensors)}"
      )
    print(
      f"[Config] foot_contact_sensor={FOOT_CONTACT_LOAD_SENSOR_NAME}, "
      f"contact_threshold={FOOT_CONTACT_LOAD_THRESHOLD_N:.1f} N, "
      f"mass_kg=[{robot_mass_kg.min().item():.3f}, {robot_mass_kg.max().item():.3f}]"
    )

  # --- Reset and allocate buffers ---
  obs_dict, _ = env.reset()
  prev_q_des = torch.zeros(N, ACTION_DIM, device=device)
  proprio_buf = torch.zeros(N, H, PROPRIO_DIM, device=device)

  term_count = 0  # cumulative termination/truncation events

  # --- Pre-allocate memory-mapped output files ---
  # Valid recording starts at step H-1 (once proprio buffer is full).
  valid_steps = total_steps - (H - 1)
  total_samples = valid_steps * N
  os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

  # Use a temp dir next to output for memmaps; merge into .pt at the end
  mmap_dir = args.output + ".mmap_tmp"
  os.makedirs(mmap_dir, exist_ok=True)

  # Save proprio already flattened (N, H*93) to avoid reshape copy at training time
  proprio_mmap = np.lib.format.open_memmap(
    os.path.join(mmap_dir, "proprio.npy"),
    mode="w+",
    dtype=np.float32,
    shape=(total_samples, H * PROPRIO_DIM),
  )
  target_mmap = np.lib.format.open_memmap(
    os.path.join(mmap_dir, "targets.npy"),
    mode="w+",
    dtype=np.float32,
    shape=(total_samples, target_dim),
  )
  env_id_mmap = np.lib.format.open_memmap(
    os.path.join(mmap_dir, "env_ids.npy"),
    mode="w+",
    dtype=np.int32,
    shape=(total_samples,),
  )
  episode_id_mmap = np.lib.format.open_memmap(
    os.path.join(mmap_dir, "episode_ids.npy"),
    mode="w+",
    dtype=np.int32,
    shape=(total_samples,),
  )
  terminated_mmap = np.lib.format.open_memmap(
    os.path.join(mmap_dir, "terminated.npy"),
    mode="w+",
    dtype=np.bool_,
    shape=(total_samples,),
  )
  truncated_mmap = np.lib.format.open_memmap(
    os.path.join(mmap_dir, "truncated.npy"),
    mode="w+",
    dtype=np.bool_,
    shape=(total_samples,),
  )
  dones_mmap = np.lib.format.open_memmap(
    os.path.join(mmap_dir, "dones.npy"),
    mode="w+",
    dtype=np.bool_,
    shape=(total_samples,),
  )
  mem_gb = (total_samples * (H * PROPRIO_DIM + target_dim + 1) * 4) / 1e9
  print(
    f"[Memmap] Pre-allocated {total_samples:,} samples "
    f"({valid_steps} steps × {N} envs) ≈ {mem_gb:.1f} GB on disk"
  )

  write_idx = 0  # current write position in memmaps
  env_ids_np = np.arange(N, dtype=np.int32)
  episode_ids_np = np.zeros(N, dtype=np.int32)
  terminated_count = 0
  truncated_count = 0

  # --- Main collection loop ---
  for step in range(total_steps):
    # Read noisy proprio and ground-truth targets
    proprio = build_noisy_proprio(robot_data, prev_q_des)
    targets = read_targets(
      robot_data,
      include_foot_heights=include_foot_heights,
      include_projected_gravity=False,
      include_joint_vel=False,
      include_stance_foot_orientation=False,
      foot_body_ids=foot_body_ids,
    )
    if include_projected_gravity:
      targets = torch.cat([targets, robot_data.projected_gravity_b], dim=-1)
    if include_joint_vel:
      targets = torch.cat([targets, robot_data.joint_vel], dim=-1)
    if include_foot_contact_load:
      foot_contact_load = read_foot_contact_load_targets(env)
      targets = torch.cat([targets, foot_contact_load], dim=-1)
    if include_stance_foot_orientation:
      stance_foot_orientation = read_stance_foot_orientation_targets(
        robot_data,
        foot_body_ids,
      )
      targets = torch.cat([targets, stance_foot_orientation], dim=-1)

    # Update rolling buffer
    proprio_buf = torch.roll(proprio_buf, shifts=-1, dims=1)
    proprio_buf[:, -1] = proprio

    # Only record after buffer is full — write directly to memmap.
    end_idx = write_idx + N
    if step >= H - 1:
      proprio_mmap[write_idx:end_idx] = proprio_buf.cpu().numpy().reshape(N, -1)
      target_mmap[write_idx:end_idx] = targets.cpu().numpy()
      env_id_mmap[write_idx:end_idx] = env_ids_np
      episode_id_mmap[write_idx:end_idx] = episode_ids_np

    # Dispatch observations to each policy group
    actor_obs = cast(torch.Tensor, obs_dict["actor"])
    actions = torch.zeros(N, ACTION_DIM, device=device)
    with torch.no_grad():
      for i, policy in enumerate(policies):
        start, end = policy_env_ranges[i]
        actions[start:end] = policy(actor_obs[start:end])

    obs_dict, _, terminated, truncated, _ = env.step(actions)
    prev_q_des = cast(Any, joint_action_term)._processed_actions.clone()
    dones = terminated | truncated
    terminated_np = terminated.cpu().numpy()
    truncated_np = truncated.cpu().numpy()
    dones_np = dones.cpu().numpy()

    if step >= H - 1:
      terminated_mmap[write_idx:end_idx] = terminated_np
      truncated_mmap[write_idx:end_idx] = truncated_np
      dones_mmap[write_idx:end_idx] = dones_np
      write_idx = end_idx

    # Reset history for terminated/truncated envs
    if dones.any():
      proprio_buf[dones] = 0.0
      prev_q_des[dones] = 0.0
      term_count += dones.sum().item()
      terminated_count += terminated.sum().item()
      truncated_count += truncated.sum().item()
      episode_ids_np[dones_np] += 1

    if (step + 1) % 200 == 0:
      term_rate = term_count / ((step + 1) * N)
      print(
        f"  step {step + 1}/{total_steps}  "
        f"terminations: {term_count} ({term_rate:.2%} per env-step)"
      )

  # --- Flush memmaps ---
  proprio_mmap.flush()
  target_mmap.flush()
  env_id_mmap.flush()
  episode_id_mmap.flush()
  terminated_mmap.flush()
  truncated_mmap.flush()
  dones_mmap.flush()

  assert write_idx == total_samples, (
    f"Expected {total_samples} samples, wrote {write_idx}"
  )

  print(f"\n[Summary] Total samples: {total_samples:,}")
  print(f"  proprio (flattened): ({total_samples}, {H * PROPRIO_DIM})")
  print(f"  targets: ({total_samples}, {target_dim})")
  print(f"  env_ids: ({total_samples},) (unique: {N})")
  print(f"  episode_ids: ({total_samples},)")
  print(
    f"  terminations: {term_count} "
    f"({term_count / (total_steps * N):.2%} per env-step over rollout)"
  )
  print(f"    terminated: {terminated_count}")
  print(f"    truncated: {truncated_count}")

  print("\n[Target stats]")
  # Compute stats in chunks to avoid loading full memmap into RAM
  STAT_CHUNK = 500_000
  for i, name in enumerate(target_names):
    vals = []
    for s in range(0, total_samples, STAT_CHUNK):
      vals.append(target_mmap[s : min(s + STAT_CHUNK, total_samples), i])
    col = np.concatenate(vals)
    print(
      f"  {name:>12s}: mean={col.mean():.4f}, "
      f"std={col.std():.4f}, "
      f"range=[{col.min():.4f}, {col.max():.4f}]"
    )

  # --- Save as directory of .npy files + metadata .pt ---
  # Saving a single .pt with the full proprio tensor would require ~28 GB
  # contiguous RAM. Instead, keep the memmap .npy files in place and save
  # a small metadata .pt alongside them. The training script loads the
  # .npy files as memmaps, so it never needs >1 batch in RAM at a time.
  output_dir = os.path.dirname(args.output) or "."

  # Move memmaps from temp dir to output dir
  import shutil

  for fname in [
    "proprio.npy",
    "targets.npy",
    "env_ids.npy",
    "episode_ids.npy",
    "terminated.npy",
    "truncated.npy",
    "dones.npy",
  ]:
    src = os.path.join(mmap_dir, fname)
    if os.path.exists(src):
      dst = os.path.join(output_dir, fname)
      shutil.move(src, dst)
  shutil.rmtree(mmap_dir, ignore_errors=True)
  del proprio_mmap, target_mmap, env_id_mmap

  # Save metadata as .pt (small, no large tensors)
  checkpoint_dir_abs = (
    os.path.abspath(args.checkpoint_dir) if args.checkpoint_dir is not None else None
  )
  checkpoint_dir_rel = (
    _repo_relative(args.checkpoint_dir) if args.checkpoint_dir is not None else None
  )
  meta_dict = {
    "num_samples": total_samples,
    "num_envs": N,
    "history_length": H,
    "proprio_dim": PROPRIO_DIM,
    "target_dim": target_dim,
    "target_layout": target_names,
    "estimate_foot_heights": include_foot_heights,
    "estimate_projected_gravity": include_projected_gravity,
    "estimate_joint_vel": include_joint_vel,
    "estimate_foot_contact": include_foot_contact_load,
    "estimate_stance_foot_orientation": include_stance_foot_orientation,
    "foot_body_names": (
      list(FOOT_BODY_NAMES)
      if include_foot_heights or include_stance_foot_orientation
      else []
    ),
    "foot_height_target": FOOT_HEIGHT_TARGET if include_foot_heights else None,
    "stance_foot_orientation_target": (
      STANCE_FOOT_PROJECTED_GRAVITY_TARGET if include_stance_foot_orientation else None
    ),
    "stance_foot_orientation_target_names": (
      STANCE_FOOT_PROJECTED_GRAVITY_TARGET_NAMES
      if include_stance_foot_orientation
      else []
    ),
    "foot_contact_load_sensor_name": (
      FOOT_CONTACT_LOAD_SENSOR_NAME if include_foot_contact_load else None
    ),
    "foot_contact_target_names": (
      FOOT_CONTACT_LOAD_TARGET_NAMES if include_foot_contact_load else []
    ),
    "foot_contact_load_threshold_n": (
      FOOT_CONTACT_LOAD_THRESHOLD_N if include_foot_contact_load else None
    ),
    "foot_contact_load_normalization": (
      {
        "type": "log1p_load",
        "formula": "log1p(force_norm / (robot_mass_kg * 9.81))",
        "robot_mass_source": "current_env_model_body_mass_per_sample",
        "initial_robot_mass_kg_min": float(robot_mass_kg.min().item()),
        "initial_robot_mass_kg_max": float(robot_mass_kg.max().item()),
        "gravity_mps2": GRAVITY_MPS2,
      }
      if include_foot_contact_load
      else None
    ),
    "proprio_terms": list(PROPRIO_TERMS),
    "proprio_action_representation": PROPRIO_ACTION_REPRESENTATION,
    "proprio_scales": [SCALE_ANG_VEL, 1.0, 1.0, SCALE_JOINT_VEL, 1.0],
    "noise_ranges": {
      "ang_vel": NOISE_ANG_VEL,
      "proj_grav": NOISE_PROJ_GRAV,
      "joint_pos": NOISE_JOINT_POS,
      "joint_vel": NOISE_JOINT_VEL,
    },
    "task": "wtw",
    "wtw_variant": "baseline",
    "checkpoint_dir": checkpoint_dir_abs,
    "checkpoint_dir_repo_relative": checkpoint_dir_rel,
    "requested_iterations": iterations,
    "valid_iterations": valid_iterations,
    "checkpoint_paths": valid_checkpoint_paths,
    "checkpoint_paths_repo_relative": [
      _repo_relative(path) or path for path in valid_checkpoint_paths
    ],
    "terrain_override": ESTIMATOR_TERRAIN,
    "terrain_type": terrain_meta["terrain_type"],
    "terrain": terrain_meta,
    "termination": {
      "dones": term_count,
      "terminated": terminated_count,
      "truncated": truncated_count,
      "dones_per_env_step": term_count / (total_steps * N),
      "terminated_per_env_step": terminated_count / (total_steps * N),
      "truncated_per_env_step": truncated_count / (total_steps * N),
      "failed_episode_count": terminated_count,
      "timeout_episode_count": truncated_count,
    },
    "collection": {
      "task": "wtw",
      "wtw_variant": "baseline",
      "checkpoint_dir": checkpoint_dir_abs,
      "checkpoint_dir_repo_relative": checkpoint_dir_rel,
      "requested_iterations": iterations,
      "valid_iterations": valid_iterations,
      "checkpoint_paths": valid_checkpoint_paths,
      "checkpoint_paths_repo_relative": [
        _repo_relative(path) or path for path in valid_checkpoint_paths
      ],
      "terrain": terrain_meta,
      "target_dim": target_dim,
      "target_layout": target_names,
      "estimate_foot_heights": include_foot_heights,
      "estimate_projected_gravity": include_projected_gravity,
      "estimate_joint_vel": include_joint_vel,
      "estimate_foot_contact": include_foot_contact_load,
      "estimate_stance_foot_orientation": include_stance_foot_orientation,
      "foot_body_names": (
        list(FOOT_BODY_NAMES)
        if include_foot_heights or include_stance_foot_orientation
        else []
      ),
      "foot_height_target": FOOT_HEIGHT_TARGET if include_foot_heights else None,
      "stance_foot_orientation_target": (
        STANCE_FOOT_PROJECTED_GRAVITY_TARGET
        if include_stance_foot_orientation
        else None
      ),
      "stance_foot_orientation_target_names": (
        STANCE_FOOT_PROJECTED_GRAVITY_TARGET_NAMES
        if include_stance_foot_orientation
        else []
      ),
      "foot_contact_load_sensor_name": (
        FOOT_CONTACT_LOAD_SENSOR_NAME if include_foot_contact_load else None
      ),
      "foot_contact_target_names": (
        FOOT_CONTACT_LOAD_TARGET_NAMES if include_foot_contact_load else []
      ),
      "foot_contact_load_threshold_n": (
        FOOT_CONTACT_LOAD_THRESHOLD_N if include_foot_contact_load else None
      ),
    },
    # Data file paths (relative to this .pt file)
    "proprio_file": "proprio.npy",
    "targets_file": "targets.npy",
    "env_ids_file": "env_ids.npy",
    "episode_ids_file": "episode_ids.npy",
    "terminated_file": "terminated.npy",
    "truncated_file": "truncated.npy",
    "dones_file": "dones.npy",
  }
  torch.save(meta_dict, args.output)
  print(f"\n[Saved] {args.output} (metadata)")
  print(f"[Saved] {output_dir}/proprio.npy ({total_samples}, {H * PROPRIO_DIM})")
  print(f"[Saved] {output_dir}/targets.npy ({total_samples}, {target_dim})")


if __name__ == "__main__":
  main()
