"""Tests for reference-conditioned G1 jumping-track tasks."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.tasks.jumping_track.config.g1.env_cfgs import (
  unitree_g1_jumping_track_baseline_env_cfg,
  unitree_g1_jumping_track_uan_env_cfg,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.tasks.utils.g1_variants import has_uan_checkpoint_env
from mjlab.tasks.velocity.mdp import (
  TorqueSpeedClippedJointPositionActionCfg,
  UANAugmentedTorqueSpeedClippedJointPositionActionCfg,
)
from mjlab.terrains import TerrainEntityCfg

REFERENCE_TERM_NAMES = (
  "motion_anchor_height",
  "motion_anchor_projected_gravity",
  "motion_joint_pos",
  "motion_joint_vel",
  "motion_anchor_lin_vel_b",
  "motion_anchor_ang_vel_b",
)

POSITIVE_TRACKING_TERMS = {
  "motion_global_root_pos",
  "motion_global_root_ori",
  "motion_body_pos",
  "motion_body_ori",
  "motion_body_lin_vel",
  "motion_body_ang_vel",
}

PADDED_JUMP_MOTION = (
  Path(__file__).resolve().parents[1] / "motions" / "02_04_jump_balance_pad50.npz"
)


class _MockScene(dict[str, Any]):
  env_origins: torch.Tensor


def _mock_env() -> SimpleNamespace:
  num_envs = 2
  half_turn_z = torch.tensor(2.0**-0.5)
  command = SimpleNamespace(
    joint_pos=torch.arange(58, dtype=torch.float32).reshape(num_envs, 29),
    joint_vel=torch.arange(58, dtype=torch.float32).reshape(num_envs, 29) + 100.0,
    anchor_pos_w=torch.tensor([[0.0, 0.0, 1.2], [0.0, 0.0, 2.5]]),
    anchor_quat_w=torch.tensor(
      [[1.0, 0.0, 0.0, 0.0], [half_turn_z, 0.0, 0.0, half_turn_z]]
    ),
    anchor_lin_vel_w=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    anchor_ang_vel_w=torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
    robot=SimpleNamespace(
      data=SimpleNamespace(
        gravity_vec_w=torch.tensor([[0.0, 0.0, -1.0]]).repeat(num_envs, 1)
      )
    ),
  )
  scene = _MockScene(robot=command.robot)
  scene.env_origins = torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 2.0]])
  return SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda _: command),
    scene=scene,
  )


def test_motion_reference_observations() -> None:
  env = cast(ManagerBasedRlEnv, _mock_env())
  command = env.command_manager.get_term("motion")
  command = cast(Any, command)

  torch.testing.assert_close(
    mdp.motion_anchor_height(env, "motion"), torch.tensor([[1.0], [0.5]])
  )
  torch.testing.assert_close(
    mdp.motion_anchor_projected_gravity(env, "motion"),
    torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
    atol=1e-6,
    rtol=0.0,
  )
  command.robot.data.projected_gravity_b = torch.tensor(
    [[0.5, 0.0, -0.8660254], [0.0, 0.5, -0.8660254]]
  )
  measured_gravity = envs_mdp.projected_gravity(env)
  reference_gravity = mdp.motion_anchor_projected_gravity(env, "motion")
  assert not torch.equal(measured_gravity, reference_gravity)
  torch.testing.assert_close(mdp.motion_joint_pos(env, "motion"), command.joint_pos)
  torch.testing.assert_close(mdp.motion_joint_vel(env, "motion"), command.joint_vel)
  torch.testing.assert_close(
    mdp.motion_anchor_lin_vel_b(env, "motion"),
    torch.tensor([[1.0, 2.0, 3.0], [5.0, -4.0, 6.0]]),
    atol=1e-6,
    rtol=0.0,
  )
  torch.testing.assert_close(
    mdp.motion_anchor_ang_vel_b(env, "motion"),
    torch.tensor([[7.0, 8.0, 9.0], [11.0, -10.0, 12.0]]),
    atol=1e-6,
    rtol=0.0,
  )


def test_jumping_track_baseline_config() -> None:
  cfg = unitree_g1_jumping_track_baseline_env_cfg()
  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms

  assert cfg.scene.num_envs == 16_384
  assert cfg.episode_length_s == 20.0
  assert tuple(actor_terms) == (
    *REFERENCE_TERM_NAMES,
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  )
  assert tuple(critic_terms) == (
    *REFERENCE_TERM_NAMES,
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "body_pos",
    "body_ori",
    "base_lin_vel",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  )
  assert actor_terms["projected_gravity"].func is envs_mdp.projected_gravity
  actor_gravity_noise = cast(Any, actor_terms["projected_gravity"].noise)
  assert actor_gravity_noise is not None
  assert actor_gravity_noise.n_min == -0.05
  assert actor_gravity_noise.n_max == 0.05
  assert critic_terms["projected_gravity"].func is envs_mdp.projected_gravity
  assert critic_terms["projected_gravity"].noise is None
  assert isinstance(cfg.actions["joint_pos"], TorqueSpeedClippedJointPositionActionCfg)
  assert cfg.curriculum == {}
  assert {"push_robot", "foot_friction", "encoder_bias", "base_com"} <= set(cfg.events)
  assert cfg.events["encoder_bias"].params["bias_range"] == (-0.01, 0.01)
  assert cfg.events["base_com"].params["ranges"] == {
    0: (-0.025, 0.025),
    1: (-0.05, 0.05),
    2: (-0.05, 0.05),
  }
  assert cfg.events["push_robot"].params["velocity_range"]["z"] == (-0.2, 0.2)


def test_jumping_track_uan_removes_robot_specific_dr() -> None:
  cfg = unitree_g1_jumping_track_uan_env_cfg(uan_checkpoint_path="/tmp/test-uan.pt")

  assert cfg.scene.num_envs == 16_384
  assert isinstance(
    cfg.actions["joint_pos"], UANAugmentedTorqueSpeedClippedJointPositionActionCfg
  )
  assert {"push_robot", "foot_friction"} <= set(cfg.events)
  assert {"encoder_bias", "base_com"}.isdisjoint(cfg.events)


def test_jumping_track_play_is_clean() -> None:
  for factory in (
    unitree_g1_jumping_track_baseline_env_cfg,
    unitree_g1_jumping_track_uan_env_cfg,
  ):
    kwargs = (
      {"uan_checkpoint_path": "/tmp/test-uan.pt"} if "uan" in factory.__name__ else {}
    )
    cfg = factory(play=True, **kwargs)
    motion = cfg.commands["motion"]
    assert isinstance(motion, MotionCommandCfg)
    assert cfg.scene.num_envs == 1
    assert cfg.episode_length_s == int(1e9)
    assert cfg.observations["actor"].enable_corruption is False
    assert {"push_robot", "foot_friction", "encoder_bias", "base_com"}.isdisjoint(
      cfg.events
    )
    assert motion.pose_range == {}
    assert motion.velocity_range == {}
    assert motion.sampling_mode == "start"


def test_jumping_track_registers_only_requested_tasks() -> None:
  task_ids = {task for task in list_tasks() if "JumpingTrack" in task}
  assert "Mjlab-JumpingTrack-Baseline-Unitree-G1" in task_ids
  if has_uan_checkpoint_env():
    assert "Mjlab-JumpingTrack-UAN-Unitree-G1" in task_ids
  assert all("HardTarget" not in task for task in task_ids)
  assert all("Custom" not in task for task in task_ids)
  cfg = load_env_cfg("Mjlab-JumpingTrack-Baseline-Unitree-G1")
  assert cfg.episode_length_s == 20.0


@pytest.mark.slow
def test_jumping_track_perfect_reference_tracking_rewards() -> None:
  """Exact reference states produce perfect rewards across resets and wrapping."""
  cfg = unitree_g1_jumping_track_baseline_env_cfg(play=True)
  cfg.scene.num_envs = 1
  cfg.scene.terrain = TerrainEntityCfg(terrain_type="plane")
  cfg.scene.sensors = ()
  cfg.events = {}
  cfg.terminations = {}
  cfg.rewards = {
    name: term for name, term in cfg.rewards.items() if name in POSITIVE_TRACKING_TERMS
  }

  motion_cfg = cfg.commands["motion"]
  assert isinstance(motion_cfg, MotionCommandCfg)
  motion_cfg.motion_file = str(PADDED_JUMP_MOTION)
  motion_cfg.sampling_mode = "start"
  motion_cfg.pose_range = {}
  motion_cfg.velocity_range = {}
  motion_cfg.joint_position_range = (0.0, 0.0)

  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
    command = cast(MotionCommand, env.command_manager.get_term("motion"))
    frame_count = command.motion.time_step_total

    max_raw_error = 0.0
    max_weighted_error = 0.0
    max_metric_error = 0.0
    worst_context = ""

    def check_current_state(frame: int, context: str) -> None:
      nonlocal max_metric_error, max_raw_error, max_weighted_error, worst_context

      env.scene.write_data_to_sim()
      env.sim.forward()
      command.update_relative_body_poses()
      command._update_metrics()

      env.reward_manager.compute(dt=env.step_dt)
      weighted_terms = dict(env.reward_manager.get_active_iterable_terms(0))
      for name in env.reward_manager.active_terms:
        term_cfg = env.reward_manager.get_term_cfg(name)
        raw_reward = term_cfg.func(env, **term_cfg.params).item()
        weighted_reward = weighted_terms[name][0]
        raw_error = abs(raw_reward - 1.0)
        weighted_error = abs(weighted_reward - term_cfg.weight)
        if max(raw_error, weighted_error) > max(max_raw_error, max_weighted_error):
          worst_context = (
            f"{context}, frame={frame}, command_frame={command.time_steps.item()}, "
            f"term={name}, raw={raw_reward}, weighted={weighted_reward}"
          )
        max_raw_error = max(max_raw_error, raw_error)
        max_weighted_error = max(max_weighted_error, weighted_error)

      for name, value in command.metrics.items():
        if not name.startswith("error_"):
          continue
        metric_error = torch.max(torch.abs(value)).item()
        if metric_error > max_metric_error:
          worst_context = (
            f"{context}, frame={frame}, command_frame={command.time_steps.item()}, "
            f"metric={name}, value={metric_error}"
          )
        max_metric_error = max(max_metric_error, metric_error)

    def check_frame(frame: int, context: str) -> None:
      command.reset_to_frame(env_ids, frame)
      check_current_state(frame, context)

    # A reset writes frame zero before command advancement prepares the next target.
    env.reset()
    assert command.time_steps.item() == 1
    command.time_steps[:] = 0
    check_current_state(0, "initial reset")

    for frame in range(frame_count):
      check_frame(frame, "full motion sweep")

    check_frame(frame_count // 2, "nonzero-frame reset")
    check_frame(frame_count - 1, "pre-wrap")
    command._update_command()
    assert command.time_steps.item() == 0
    check_current_state(0, "post-wrap")

    print(
      f"frames={frame_count}, max_raw_error={max_raw_error:.9g}, "
      f"max_weighted_error={max_weighted_error:.9g}, "
      f"max_metric_error={max_metric_error:.9g}"
    )
    assert max_raw_error <= 2.0e-4, (
      f"max raw reward error {max_raw_error} exceeded tolerance; {worst_context}"
    )
    assert max_weighted_error <= 2.0e-4, (
      f"max weighted reward error {max_weighted_error} exceeded tolerance; "
      f"{worst_context}"
    )
    assert max_metric_error <= 2.0e-6, (
      f"max command metric error {max_metric_error} exceeded tolerance; {worst_context}"
    )
  finally:
    env.close()
