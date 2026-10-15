#!/usr/bin/env python3
"""Render one replay window as hardware playback beside simulation.

The left panel is a kinematic rendering of recorded hardware rows. The right
panel resets MJLab's UAN replay env to the same row and either applies zero
residual actions or a supplied UAN checkpoint for the requested horizon.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import cast

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

import mjlab.tasks  # noqa: F401
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.config import RslRlBaseRunnerCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.uan.mdp.actions import ReplayResidualJointPositionAction
from mjlab.tasks.uan.mdp.replay_command import (
  ReplayWindowCommand,
  ReplayWindowCommandCfg,
)
from mjlab.tasks.uan.uan_rollout_env_cfg import (
  sync_uan_rollout_config_from_checkpoint_metadata,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

STATE_DIM = 96
ACTION_DIM = 29
ROOT_HEIGHT_EXT_COL = 4

G1_HARDWARE_JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--data-dir",
    required=True,
    help="Processed replay directory containing all_data.npy.",
  )
  parser.add_argument("--task", default="Mjlab-UAN-Rollout-Unitree-G1")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--uan-checkpoint",
    default=None,
    help="Optional UAN checkpoint for the right-panel rollout.",
  )
  parser.add_argument("--horizon", type=int, default=24)
  parser.add_argument(
    "--extension-dim",
    type=int,
    default=None,
    help="Replay extension dimension override. Defaults to the task config value.",
  )
  parser.add_argument(
    "--require-contact-at-reset",
    nargs="?",
    const=True,
    default=None,
    type=lambda value: str(value).lower() in ("1", "true", "t", "yes", "y", "on"),
  )
  parser.add_argument("--contact-threshold", type=float, default=None)
  parser.add_argument("--window-start", type=int, default=None)
  parser.add_argument(
    "--window-selection",
    choices=("first", "even", "random"),
    default="random",
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output-dir", default=None)
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--height", type=int, default=480)
  parser.add_argument("--fps", type=int, default=50)
  parser.add_argument("--camera-distance", type=float, default=2.6)
  parser.add_argument("--camera-azimuth", type=float, default=135.0)
  parser.add_argument("--camera-elevation", type=float, default=-18.0)
  return parser.parse_args()


def select_window_start(
  valid_starts: torch.Tensor,
  *,
  explicit_start: int | None,
  selection: str,
  seed: int,
) -> int:
  valid_np = valid_starts.detach().cpu().numpy().astype(np.int64)
  if valid_np.size == 0:
    raise RuntimeError("Replay command has no valid starts.")

  if explicit_start is not None:
    explicit_start = int(explicit_start)
    if not np.any(valid_np == explicit_start):
      preview = ", ".join(str(int(x)) for x in valid_np[:8])
      raise ValueError(
        f"--window-start={explicit_start} is not valid. "
        f"First valid starts: [{preview}] ..."
      )
    return explicit_start

  if selection == "first":
    return int(valid_np[0])
  if selection == "even":
    return int(valid_np[valid_np.size // 2])
  if selection == "random":
    rng = np.random.default_rng(int(seed))
    return int(rng.choice(valid_np))
  raise ValueError(f"Unknown selection {selection!r}.")


def quat_from_projected_gravity(gravity_body: np.ndarray) -> np.ndarray:
  """Recover a yaw-free root quaternion from projected gravity."""
  world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)
  g = np.asarray(gravity_body, dtype=np.float64)
  norm = np.linalg.norm(g)
  if norm < 1.0e-8:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
  g = g / norm
  dot = float(np.clip(np.dot(g, world_gravity), -1.0, 1.0))
  if dot > 1.0 - 1.0e-8:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
  if dot < -1.0 + 1.0e-8:
    return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

  axis = np.cross(g, world_gravity)
  axis_norm = np.linalg.norm(axis)
  if axis_norm < 1.0e-8:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
  axis = axis / axis_norm
  half = 0.5 * np.arccos(dot)
  quat = np.array(
    [
      np.cos(half),
      axis[0] * np.sin(half),
      axis[1] * np.sin(half),
      axis[2] * np.sin(half),
    ],
    dtype=np.float64,
  )
  return quat / np.linalg.norm(quat)


def make_camera(args: argparse.Namespace) -> mujoco.MjvCamera:
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.distance = float(args.camera_distance)
  camera.azimuth = float(args.camera_azimuth)
  camera.elevation = float(args.camera_elevation)
  camera.lookat[:] = np.array([0.0, 0.0, 0.75], dtype=np.float64)
  return camera


def make_scene_option() -> mujoco.MjvOption:
  option = mujoco.MjvOption()
  option.geomgroup[2] = 1
  option.geomgroup[3] = 0
  return option


def hardware_row_ids(command, start_row: int, horizon: int) -> torch.Tensor:
  action_rows = torch.arange(
    int(start_row),
    int(start_row) + int(horizon),
    dtype=torch.long,
    device=command.device,
  )
  initial_row = command.action_state_row_ids(
    torch.as_tensor([int(start_row)], dtype=torch.long, device=command.device)
  )[0]
  target_rows = command.target_row_ids(action_rows)
  return torch.cat([initial_row[None], target_rows], dim=0)


def render_hardware_frames(
  *,
  command,
  row_ids: torch.Tensor,
  width: int,
  height: int,
  camera: mujoco.MjvCamera,
  option: mujoco.MjvOption,
) -> list[np.ndarray]:
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  data = mujoco.MjData(model)
  free_joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
  )
  if free_joint_id < 0:
    raise ValueError(f"floating_base_joint not found in {G1_XML}")
  free_qpos_adr = int(model.jnt_qposadr[free_joint_id])

  joint_qpos = []
  for joint_name in G1_HARDWARE_JOINT_NAMES:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
      raise ValueError(f"Joint {joint_name!r} not found in {G1_XML}")
    joint_qpos.append(int(model.jnt_qposadr[joint_id]))

  rows = row_ids.detach().cpu().numpy().astype(np.int64)
  state = command.state_raw[row_ids].detach().cpu().numpy()
  ext = command.ext_raw[row_ids].detach().cpu().numpy()

  frames: list[np.ndarray] = []
  with mujoco.Renderer(model, height=int(height), width=int(width)) as renderer:
    for state_row, ext_row in zip(state, ext, strict=True):
      root_height = (
        float(ext_row[ROOT_HEIGHT_EXT_COL])
        if ext_row.shape[0] > ROOT_HEIGHT_EXT_COL
        else float(command.cfg.default_base_height)
      )
      if not np.isfinite(root_height) or root_height <= 0.2:
        root_height = float(command.cfg.default_base_height)
      data.qpos[free_qpos_adr : free_qpos_adr + 3] = np.array(
        [0.0, 0.0, root_height], dtype=np.float64
      )
      data.qpos[free_qpos_adr + 3 : free_qpos_adr + 7] = quat_from_projected_gravity(
        state_row[6:9]
      )
      for joint_idx, qpos_adr in enumerate(joint_qpos):
        data.qpos[qpos_adr] = float(state_row[9 + joint_idx])
      mujoco.mj_forward(model, data)
      renderer.update_scene(data, camera=camera, scene_option=option)
      frames.append(renderer.render().copy())

  if len(frames) != len(rows):
    raise RuntimeError(f"Rendered {len(frames)} hardware frames for {len(rows)} rows.")
  return frames


def load_checkpoint_metadata(checkpoint: Path, *, device: str = "cpu") -> dict:
  metadata = torch.load(str(checkpoint), map_location=device, weights_only=False)
  if not isinstance(metadata, dict):
    raise ValueError(f"Checkpoint {checkpoint} did not load as a metadata dict.")
  return metadata


def load_policy(
  *,
  env: RslRlVecEnvWrapper,
  task_id: str,
  agent_cfg,
  checkpoint: Path,
  device: str,
):
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  return runner.get_inference_policy(device=device)


def render_sim_frames(
  *,
  env: ManagerBasedRlEnv | RslRlVecEnvWrapper,
  policy,
  start_row: int,
  horizon: int,
  width: int,
  height: int,
  camera_args: argparse.Namespace,
) -> tuple[list[np.ndarray], torch.Tensor]:
  base_env = env.unwrapped if isinstance(env, RslRlVecEnvWrapper) else env
  command = cast(ReplayWindowCommand, base_env.command_manager.get_term("replay"))
  command.valid_starts = torch.as_tensor(
    [int(start_row)],
    dtype=torch.long,
    device=command.device,
  )
  env_cfg = ViewerConfig(
    height=int(height),
    width=int(width),
    origin_type=ViewerConfig.OriginType.WORLD,
    lookat=(0.0, 0.0, 0.75),
    distance=float(camera_args.camera_distance),
    elevation=float(camera_args.camera_elevation),
    azimuth=float(camera_args.camera_azimuth),
    env_idx=0,
    max_extra_envs=0,
  )
  renderer = OffscreenRenderer(
    model=base_env.sim.mj_model,
    cfg=env_cfg,
    scene=base_env.scene,
  )
  frames: list[np.ndarray] = []
  joint_pos: list[torch.Tensor] = []
  try:
    renderer.initialize()
    obs, _ = env.reset()
    action_term = cast(
      ReplayResidualJointPositionAction,
      base_env.action_manager.get_term("residual_target"),
    )
    joint_pos.append(
      command.robot.data.joint_pos[:, action_term.target_ids][0].detach().clone()
    )
    renderer.update(base_env.sim.data)
    frames.append(renderer.render().copy())
    with torch.no_grad():
      for _ in range(int(horizon)):
        if policy is None:
          action = torch.zeros_like(base_env.action_manager.action)
          obs, _, terminated, truncated, _ = base_env.step(action)
          done = terminated | truncated
        else:
          action = policy(obs)
          if not isinstance(env, RslRlVecEnvWrapper):
            raise TypeError("Policy rendering requires the RL environment wrapper.")
          obs, _, done, _ = env.step(action)
        if bool(done[0].item()):
          raise RuntimeError(
            "Replay env reset before the requested horizon. "
            "Try a different --window-start."
          )
        joint_pos.append(
          command.robot.data.joint_pos[:, action_term.target_ids][0].detach().clone()
        )
        renderer.update(base_env.sim.data)
        frames.append(renderer.render().copy())
  finally:
    renderer.close()

  return frames, torch.stack(joint_pos, dim=0)


def build_env(
  args: argparse.Namespace,
  data_dir: Path,
) -> tuple[ManagerBasedRlEnv, RslRlBaseRunnerCfg | None]:
  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = None
  if args.uan_checkpoint is not None:
    agent_cfg = load_rl_cfg(args.task)
    checkpoint_metadata = load_checkpoint_metadata(
      Path(args.uan_checkpoint).expanduser().resolve(),
      device="cpu",
    )
    sync_uan_rollout_config_from_checkpoint_metadata(
      env_cfg=env_cfg,
      agent_cfg=agent_cfg,
      checkpoint_metadata=checkpoint_metadata,
    )
  env_cfg.scene.num_envs = 1
  if env_cfg.scene.terrain is not None:
    env_cfg.scene.terrain.num_envs = 1
  replay_cfg = cast(ReplayWindowCommandCfg, env_cfg.commands["replay"])
  replay_cfg.data_dir = str(data_dir)
  replay_cfg.rollout_horizon = int(args.horizon)
  if args.extension_dim is not None:
    replay_cfg.extension_dim = int(args.extension_dim)
  if args.require_contact_at_reset is not None:
    replay_cfg.require_contact_at_reset = bool(args.require_contact_at_reset)
  if args.contact_threshold is not None:
    replay_cfg.contact_threshold = float(args.contact_threshold)
  env_cfg.episode_length_s = (
    (int(args.horizon) + 1) * env_cfg.decimation * env_cfg.sim.mujoco.timestep
  )
  return ManagerBasedRlEnv(cfg=env_cfg, device=args.device), agent_cfg


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  writer = imageio.get_writer(
    str(path),
    fps=int(fps),
    codec="libx264",
    quality=8,
    pixelformat="yuv420p",
    macro_block_size=1,
  )
  try:
    for frame in frames:
      writer.append_data(frame)
  finally:
    writer.close()


def main() -> None:
  args = parse_args()
  configure_torch_backends()
  if args.horizon <= 0:
    raise ValueError(f"--horizon must be positive, got {args.horizon}.")

  data_dir = Path(args.data_dir).expanduser().resolve()
  if not (data_dir / "all_data.npy").exists():
    raise FileNotFoundError(f"Missing replay array: {data_dir / 'all_data.npy'}")
  output_dir = (
    Path(args.output_dir).expanduser().resolve()
    if args.output_dir is not None
    else data_dir.parent / "artifacts" / "side_by_side_window_videos"
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  checkpoint_path = (
    Path(args.uan_checkpoint).expanduser().resolve()
    if args.uan_checkpoint is not None
    else None
  )
  if checkpoint_path is not None and not checkpoint_path.exists():
    raise FileNotFoundError(f"Missing UAN checkpoint: {checkpoint_path}")

  env, agent_cfg = build_env(args, data_dir)
  try:
    wrapped_env = None
    policy = None
    if checkpoint_path is not None:
      if agent_cfg is None:
        raise RuntimeError("Internal error: missing agent cfg for UAN checkpoint.")
      wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
      policy = load_policy(
        env=wrapped_env,
        task_id=args.task,
        agent_cfg=agent_cfg,
        checkpoint=checkpoint_path,
        device=args.device,
      )
    command = cast(ReplayWindowCommand, env.command_manager.get_term("replay"))
    start_row = select_window_start(
      command.valid_starts,
      explicit_start=args.window_start,
      selection=args.window_selection,
      seed=args.seed,
    )
    row_ids = hardware_row_ids(command, start_row, args.horizon)
    camera = make_camera(args)
    option = make_scene_option()
    hardware_frames = render_hardware_frames(
      command=command,
      row_ids=row_ids,
      width=args.width,
      height=args.height,
      camera=camera,
      option=option,
    )
    sim_frames, sim_joint_pos = render_sim_frames(
      env=wrapped_env if wrapped_env is not None else env,
      policy=policy,
      start_row=start_row,
      horizon=args.horizon,
      width=args.width,
      height=args.height,
      camera_args=args,
    )
    if len(hardware_frames) != len(sim_frames):
      raise RuntimeError(
        f"Frame count mismatch: hardware={len(hardware_frames)} sim={len(sim_frames)}"
      )

    combined = [
      np.concatenate([left, right], axis=1)
      for left, right in zip(hardware_frames, sim_frames, strict=True)
    ]

    hardware_joint_pos = command.state_raw[row_ids, 9:38].detach().cpu()
    err = sim_joint_pos.cpu() - hardware_joint_pos
    joint_rmse = float(err[1:].square().mean().sqrt().item())
    final_joint_rmse = float(err[-1].square().mean().sqrt().item())

    suffix = "uan" if checkpoint_path is not None else "nouan"
    video_path = (
      output_dir / f"side_by_side_{suffix}_start_{start_row}_h{args.horizon}.mp4"
    )
    write_video(video_path, combined, args.fps)

    summary = {
      "data_dir": str(data_dir),
      "task": args.task,
      "output_video": str(video_path),
      "start_row": int(start_row),
      "horizon": int(args.horizon),
      "num_frames": len(combined),
      "fps": int(args.fps),
      "width_per_panel": int(args.width),
      "height": int(args.height),
      "row_ids": [int(x) for x in row_ids.detach().cpu().numpy().tolist()],
      "joint_pos_rmse_rad": joint_rmse,
      "final_step_joint_pos_rmse_rad": final_joint_rmse,
      "left_panel": "recorded hardware playback",
      "right_panel": (
        "UAN policy sim rollout"
        if checkpoint_path is not None
        else "no-UAN zero-residual sim rollout"
      ),
      "uan_checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
    }
    summary_path = (
      output_dir / f"side_by_side_{suffix}_start_{start_row}_h{args.horizon}.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[Video] {video_path}")
    print(f"[Summary] {summary_path}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
