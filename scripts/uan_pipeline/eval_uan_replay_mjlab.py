"""Evaluate trained UAN rollout policies on replay datasets.

Run from the release repo root:

  uv run --locked python scripts/uan_pipeline/eval_uan_replay_mjlab.py \
    --checkpoint "logs/rsl_rl/uan_rollout/<run>/model_1999.pt" \
    --dataset "wtw=<wtw_replay_root>/processed" \
    --reference-checkpoint "logs/rsl_rl/uan_rollout/<baseline-run>/model_4999.pt" \
    --include-zero-baseline

This script uses the release replay convention: row ``i`` is the state before
action ``i`` and the rollout target is row ``i + 1``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from tensordict import TensorDict

import mjlab.tasks  # noqa: F401
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_UAN_ACTUATOR_GROUP_NAMES,
  G1_UAN_JOINT_TO_ACTUATOR_MAP,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.config import RslRlBaseRunnerCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.uan.mdp.actions import ReplayResidualJointPositionAction
from mjlab.tasks.uan.mdp.observations import prime_actor_replay_history_buffers
from mjlab.tasks.uan.mdp.replay_command import (
  ReplayWindowCommand,
  ReplayWindowCommandCfg,
)
from mjlab.tasks.uan.mdp.utils import resolve_uan_obs_block_names
from mjlab.tasks.uan.uan_rollout_env_cfg import (
  sync_uan_rollout_config_from_checkpoint_metadata,
)
from mjlab.utils.torch import configure_torch_backends


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--checkpoint", required=True, help="Primary UAN checkpoint to evaluate."
  )
  parser.add_argument(
    "--dataset",
    action="append",
    required=True,
    help="Dataset spec in the form label=/abs/or/relative/path/to/processed",
  )
  parser.add_argument(
    "--reference-checkpoint",
    default=None,
    help="Optional reference UAN checkpoint for side-by-side comparison.",
  )
  parser.add_argument(
    "--include-zero-baseline",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Also evaluate the zero-residual baseline on each dataset.",
  )
  parser.add_argument("--task", default="Mjlab-UAN-Rollout-Unitree-G1")
  parser.add_argument("--device", default=None)
  parser.add_argument("--batch-size", type=int, default=1024)
  parser.add_argument("--rollout-horizon", type=int, default=24)
  parser.add_argument(
    "--require-contact-at-reset",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Require at least one replay contact label at the reset row.",
  )
  parser.add_argument("--contact-threshold", type=float, default=0.8)
  parser.add_argument(
    "--replay-horizon-for-env",
    type=int,
    default=25,
    help=(
      "Replay-window horizon used inside the env. Keep this one step longer "
      "than --rollout-horizon when you want the final post-step observation to remain valid."
    ),
  )
  parser.add_argument(
    "--max-windows",
    type=int,
    default=0,
    help="Optional cap on the number of valid replay windows to evaluate per dataset. 0 means all.",
  )
  parser.add_argument(
    "--output-dir",
    default=None,
    help="Optional output directory. Defaults to <checkpoint_dir>/replay_eval.",
  )
  return parser.parse_args()


def parse_dataset_specs(specs: list[str]) -> list[tuple[str, Path]]:
  datasets: list[tuple[str, Path]] = []
  for spec in specs:
    if "=" not in spec:
      raise ValueError(f"--dataset expects label=path, got {spec!r}.")
    label, raw_path = spec.split("=", maxsplit=1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
      raise ValueError(f"Dataset label must be non-empty: {spec!r}")
    if not path.exists():
      raise FileNotFoundError(path)
    datasets.append((label, path))
  return datasets


def write_json(path: Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )


def load_checkpoint_metadata(checkpoint: Path, *, device: str = "cpu") -> dict:
  metadata = torch.load(str(checkpoint), map_location=device, weights_only=False)
  if not isinstance(metadata, dict):
    raise ValueError(f"Checkpoint {checkpoint} did not load as a metadata dict.")
  return metadata


def configure_rollout_layout_from_checkpoint_metadata(
  *,
  env_cfg,
  agent_cfg,
  checkpoint_metadata: dict,
) -> tuple[str, ...]:
  return sync_uan_rollout_config_from_checkpoint_metadata(
    env_cfg=env_cfg,
    agent_cfg=agent_cfg,
    checkpoint_metadata=checkpoint_metadata,
  )


def checkpoint_shares_rollout_actor_shape(
  lhs_meta: dict,
  rhs_meta: dict,
) -> bool:
  lhs_block_names = resolve_uan_obs_block_names(lhs_meta)
  rhs_block_names = resolve_uan_obs_block_names(rhs_meta)
  if lhs_block_names != rhs_block_names:
    return False
  expected_count = len(G1_UAN_ACTUATOR_GROUP_NAMES)
  expected_map = list(G1_UAN_JOINT_TO_ACTUATOR_MAP)
  for metadata in (lhs_meta, rhs_meta):
    count = metadata.get("num_actuators")
    if count is not None and int(count) != expected_count:
      raise ValueError("Contact-UAN requires five actuator-family networks.")
    mapping = metadata.get("joint_to_actuator_map")
    if mapping is not None and list(mapping) != expected_map:
      raise ValueError("Contact-UAN requires the canonical actuator-family mapping.")
  return True


def build_env(
  *,
  task_id: str,
  data_dir: Path,
  device: str,
  batch_size: int,
  replay_horizon_for_env: int,
  require_contact_at_reset: bool,
  contact_threshold: float,
  checkpoint_metadata: dict,
) -> tuple[RslRlVecEnvWrapper, RslRlBaseRunnerCfg]:
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  configure_rollout_layout_from_checkpoint_metadata(
    env_cfg=env_cfg,
    agent_cfg=agent_cfg,
    checkpoint_metadata=checkpoint_metadata,
  )
  env_cfg.scene.num_envs = batch_size
  if env_cfg.scene.terrain is not None:
    env_cfg.scene.terrain.num_envs = batch_size
  replay_cfg = cast(ReplayWindowCommandCfg, env_cfg.commands["replay"])
  replay_cfg.data_dir = str(data_dir)
  replay_cfg.rollout_horizon = int(replay_horizon_for_env)
  replay_cfg.require_contact_at_reset = bool(require_contact_at_reset)
  replay_cfg.contact_threshold = float(contact_threshold)
  env_cfg.episode_length_s = (
    (int(replay_horizon_for_env) + 1) * env_cfg.decimation * env_cfg.sim.mujoco.timestep
  )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions), agent_cfg


def load_policy(
  *,
  env: RslRlVecEnvWrapper,
  task_id: str,
  checkpoint: Path,
  device: str,
  agent_cfg,
):
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  return runner, runner.get_inference_policy(device=device)


def set_explicit_starts(
  env: RslRlVecEnvWrapper, starts: torch.Tensor
) -> tuple[ReplayWindowCommand, ReplayResidualJointPositionAction, TensorDict]:
  base = env.unwrapped
  command = cast(ReplayWindowCommand, base.command_manager.get_term("replay"))
  action_term = cast(
    ReplayResidualJointPositionAction, base.action_manager.get_term("residual_target")
  )
  env_ids = torch.arange(starts.shape[0], device=base.device, dtype=torch.long)

  command.start_row[env_ids] = starts
  command.cursor[env_ids] = starts
  command.rollout_step[env_ids] = 0
  command.write_recorded_state(
    env_ids,
    command.action_state_row_ids(starts),
    action_term.target_ids,
  )
  # env.reset() has already populated observation history for a random replay
  # start. Clear those rows before priming the explicit replay history.
  base.observation_manager.reset(env_ids=env_ids)
  prime_actor_replay_history_buffers(base, env_ids=env_ids, command_name="replay")

  # Append the current frame at each explicit start.
  obs_dict = base.observation_manager.compute(update_history=True)
  return command, action_term, TensorDict(obs_dict, batch_size=[env.num_envs])


def evaluate_mode(
  *,
  env: RslRlVecEnvWrapper,
  valid_starts_all: np.ndarray,
  policy,
  mode_label: str,
  checkpoint_path: str | None,
  dataset_dir: Path,
  rollout_horizon: int,
  replay_horizon_for_env: int,
  batch_size: int,
  max_windows: int,
) -> dict:
  base = env.unwrapped
  command = cast(ReplayWindowCommand, base.command_manager.get_term("replay"))
  action_term = cast(
    ReplayResidualJointPositionAction, base.action_manager.get_term("residual_target")
  )

  starts_all = valid_starts_all
  if max_windows > 0:
    starts_all = starts_all[:max_windows]
  total_windows = int(starts_all.shape[0])

  total_sq_err = 0.0
  total_count = 0
  final_sq_err = 0.0
  final_count = 0
  per_window_rmse: list[float] = []
  residual_rms: list[float] = []
  delta_rms: list[float] = []

  for start_idx in range(0, total_windows, batch_size):
    starts_np = starts_all[start_idx : start_idx + batch_size]
    active_n = int(starts_np.shape[0])
    if active_n == 0:
      continue
    if active_n < batch_size:
      pad = np.repeat(starts_np[-1:], batch_size - active_n)
      starts_np = np.concatenate([starts_np, pad], axis=0)
    starts = torch.as_tensor(starts_np, dtype=torch.long, device=base.device)

    env.reset()
    command.valid_starts = starts.clone()
    command, action_term, obs = set_explicit_starts(env, starts)

    states = [command.robot.data.joint_pos[:, action_term.target_ids].detach().clone()]
    proc_resids = []
    proc_deltas = []

    with torch.no_grad():
      for _ in range(rollout_horizon):
        if policy is None:
          action = torch.zeros_like(base.action_manager.action)
        else:
          action = policy(obs)
        obs, _reward, done, _extras = env.step(action)
        if bool(done[:active_n].any().item()):
          raise RuntimeError(
            f"Unexpected done before horizon during mode={mode_label!r}. "
            "Check replay_horizon_for_env and episode-length overrides."
          )
        states.append(
          command.robot.data.joint_pos[:, action_term.target_ids].detach().clone()
        )
        proc_resids.append(action_term.processed_residual.detach().clone())
        proc_deltas.append(action_term.processed_residual_delta.detach().clone())

    sim = torch.stack(states, dim=0)[:, :active_n]
    proc_resids_t = torch.stack(proc_resids, dim=0)[:, :active_n]
    proc_deltas_t = torch.stack(proc_deltas, dim=0)[:, :active_n]
    starts_active = starts[:active_n]

    hardware_init = command.state_raw[command.action_state_row_ids(starts_active), 9:38]
    action_rows = (
      starts_active[:, None]
      + torch.arange(rollout_horizon, device=base.device, dtype=torch.long)[None, :]
    )
    target_rows = command.target_row_ids(action_rows)
    hardware_post = command.state_raw[target_rows, 9:38].transpose(0, 1)
    hardware = torch.cat([hardware_init.unsqueeze(0), hardware_post], dim=0)

    err = sim[1:] - hardware[1:]
    total_sq_err += float(err.square().sum().item())
    total_count += int(err.numel())
    final_sq_err += float(err[-1].square().sum().item())
    final_count += int(err[-1].numel())
    per_window_rmse.extend(
      err.square().mean(dim=(0, 2)).sqrt().detach().cpu().numpy().tolist()
    )
    residual_rms.extend(
      proc_resids_t.square().mean(dim=(0, 2)).sqrt().detach().cpu().numpy().tolist()
    )
    delta_rms.extend(
      proc_deltas_t.square().mean(dim=(0, 2)).sqrt().detach().cpu().numpy().tolist()
    )

  per_window_rmse_arr = np.asarray(per_window_rmse, dtype=np.float64)
  residual_rms_arr = np.asarray(residual_rms, dtype=np.float64)
  delta_rms_arr = np.asarray(delta_rms, dtype=np.float64)

  return {
    "mode": mode_label,
    "checkpoint": checkpoint_path,
    "data_dir": str(dataset_dir),
    "rollout_horizon": int(rollout_horizon),
    "replay_horizon_for_env": int(replay_horizon_for_env),
    "batch_size": int(batch_size),
    "num_valid_windows": total_windows,
    "global_joint_pos_rmse_h24_rad": float((total_sq_err / max(total_count, 1)) ** 0.5),
    "final_step_global_joint_pos_rmse_rad": float(
      (final_sq_err / max(final_count, 1)) ** 0.5
    ),
    "mean_window_joint_pos_rmse_h24_rad": float(per_window_rmse_arr.mean()),
    "p95_window_joint_pos_rmse_h24_rad": float(
      np.percentile(per_window_rmse_arr, 95.0)
    ),
    "max_window_joint_pos_rmse_h24_rad": float(per_window_rmse_arr.max()),
    "mean_residual_rms_rad": float(residual_rms_arr.mean()),
    "p95_residual_rms_rad": float(np.percentile(residual_rms_arr, 95.0)),
    "max_residual_rms_rad": float(residual_rms_arr.max()),
    "mean_delta_rms_rad": float(delta_rms_arr.mean()),
    "p95_delta_rms_rad": float(np.percentile(delta_rms_arr, 95.0)),
    "max_delta_rms_rad": float(delta_rms_arr.max()),
  }


def make_output_dir(checkpoint: Path, output_dir: str | None) -> Path:
  if output_dir is not None:
    return Path(output_dir).expanduser().resolve()
  return checkpoint.parent / "replay_eval"


def main() -> None:
  args = parse_args()
  configure_torch_backends()

  checkpoint = Path(args.checkpoint).expanduser().resolve()
  if not checkpoint.exists():
    raise FileNotFoundError(checkpoint)
  checkpoint_meta = load_checkpoint_metadata(checkpoint)
  reference_checkpoint = None
  reference_checkpoint_meta = None
  if args.reference_checkpoint is not None:
    reference_checkpoint = Path(args.reference_checkpoint).expanduser().resolve()
    if not reference_checkpoint.exists():
      raise FileNotFoundError(reference_checkpoint)
    reference_checkpoint_meta = load_checkpoint_metadata(reference_checkpoint)

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  datasets = parse_dataset_specs(args.dataset)
  out_dir = make_output_dir(checkpoint, args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  summary: dict[str, Any] = {
    "task": args.task,
    "device": device,
    "rollout_horizon": int(args.rollout_horizon),
    "replay_horizon_for_env": int(args.replay_horizon_for_env),
    "require_contact_at_reset": bool(args.require_contact_at_reset),
    "contact_threshold": float(args.contact_threshold),
    "batch_size": int(args.batch_size),
    "max_windows": int(args.max_windows),
    "primary_checkpoint": str(checkpoint),
    "reference_checkpoint": str(reference_checkpoint) if reference_checkpoint else None,
    "include_zero_baseline": bool(args.include_zero_baseline),
    "datasets": {},
  }

  for dataset_label, dataset_dir in datasets:
    print(f"=== Dataset: {dataset_label} ===", flush=True)
    env, primary_agent_cfg = build_env(
      task_id=args.task,
      data_dir=dataset_dir,
      device=device,
      batch_size=int(args.batch_size),
      replay_horizon_for_env=int(args.replay_horizon_for_env),
      require_contact_at_reset=bool(args.require_contact_at_reset),
      contact_threshold=float(args.contact_threshold),
      checkpoint_metadata=checkpoint_meta,
    )
    try:
      command = cast(
        ReplayWindowCommand, env.unwrapped.command_manager.get_term("replay")
      )
      valid_starts_all = command.valid_starts.detach().cpu().numpy().astype(np.int64)

      primary_runner, primary_policy = load_policy(
        env=env,
        task_id=args.task,
        checkpoint=checkpoint,
        device=device,
        agent_cfg=primary_agent_cfg,
      )
      del primary_runner
      primary_metrics = evaluate_mode(
        env=env,
        valid_starts_all=valid_starts_all,
        policy=primary_policy,
        mode_label="primary",
        checkpoint_path=str(checkpoint),
        dataset_dir=dataset_dir,
        rollout_horizon=int(args.rollout_horizon),
        replay_horizon_for_env=int(args.replay_horizon_for_env),
        batch_size=int(args.batch_size),
        max_windows=int(args.max_windows),
      )
      print(json.dumps(primary_metrics, indent=2), flush=True)

      dataset_payload: dict[str, object] = {"primary": primary_metrics}

      if reference_checkpoint is not None:
        assert reference_checkpoint_meta is not None
        if checkpoint_shares_rollout_actor_shape(
          checkpoint_meta, reference_checkpoint_meta
        ):
          ref_runner, ref_policy = load_policy(
            env=env,
            task_id=args.task,
            checkpoint=reference_checkpoint,
            device=device,
            agent_cfg=primary_agent_cfg,
          )
          del ref_runner
          reference_metrics = evaluate_mode(
            env=env,
            valid_starts_all=valid_starts_all,
            policy=ref_policy,
            mode_label="reference",
            checkpoint_path=str(reference_checkpoint),
            dataset_dir=dataset_dir,
            rollout_horizon=int(args.rollout_horizon),
            replay_horizon_for_env=int(args.replay_horizon_for_env),
            batch_size=int(args.batch_size),
            max_windows=int(args.max_windows),
          )
        else:
          ref_env, ref_agent_cfg = build_env(
            task_id=args.task,
            data_dir=dataset_dir,
            device=device,
            batch_size=int(args.batch_size),
            replay_horizon_for_env=int(args.replay_horizon_for_env),
            require_contact_at_reset=bool(args.require_contact_at_reset),
            contact_threshold=float(args.contact_threshold),
            checkpoint_metadata=reference_checkpoint_meta,
          )
          try:
            ref_command = cast(
              ReplayWindowCommand, ref_env.unwrapped.command_manager.get_term("replay")
            )
            ref_valid_starts_all = (
              ref_command.valid_starts.detach().cpu().numpy().astype(np.int64)
            )
            ref_runner, ref_policy = load_policy(
              env=ref_env,
              task_id=args.task,
              checkpoint=reference_checkpoint,
              device=device,
              agent_cfg=ref_agent_cfg,
            )
            del ref_runner
            reference_metrics = evaluate_mode(
              env=ref_env,
              valid_starts_all=ref_valid_starts_all,
              policy=ref_policy,
              mode_label="reference",
              checkpoint_path=str(reference_checkpoint),
              dataset_dir=dataset_dir,
              rollout_horizon=int(args.rollout_horizon),
              replay_horizon_for_env=int(args.replay_horizon_for_env),
              batch_size=int(args.batch_size),
              max_windows=int(args.max_windows),
            )
          finally:
            ref_env.close()
        dataset_payload["reference"] = reference_metrics

      if args.include_zero_baseline:
        zero_metrics = evaluate_mode(
          env=env,
          valid_starts_all=valid_starts_all,
          policy=None,
          mode_label="zero_baseline",
          checkpoint_path=None,
          dataset_dir=dataset_dir,
          rollout_horizon=int(args.rollout_horizon),
          replay_horizon_for_env=int(args.replay_horizon_for_env),
          batch_size=int(args.batch_size),
          max_windows=int(args.max_windows),
        )
        dataset_payload["zero_baseline"] = zero_metrics

      summary["datasets"][dataset_label] = dataset_payload
      write_json(out_dir / f"{dataset_label}.json", dataset_payload)
    finally:
      env.close()

  write_json(out_dir / "summary.json", summary)
  print(f"[Saved] {out_dir}")


if __name__ == "__main__":
  main()
