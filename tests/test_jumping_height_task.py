"""Tests for fixed-objective G1 JumpingHeight tasks."""

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.jumping_height import mdp as jumping_mdp
from mjlab.tasks.jumping_height.config.g1.env_cfgs import (
  unitree_g1_jumping_height_baseline_env_cfg,
  unitree_g1_jumping_height_uan_env_cfg,
)
from mjlab.tasks.jumping_track.config.g1.env_cfgs import (
  unitree_g1_jumping_track_baseline_env_cfg,
)
from mjlab.tasks.jumping_track.config.g1.rl_cfg import (
  unitree_g1_jumping_track_ppo_runner_cfg,
)
from mjlab.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
)
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.utils.g1_variants import has_uan_checkpoint_env
from mjlab.tasks.velocity.mdp import (
  TorqueSpeedClippedJointPositionActionCfg,
  UANAugmentedTorqueSpeedClippedJointPositionActionCfg,
)
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

REFERENCE_TERM_NAMES = (
  "motion_anchor_height",
  "motion_anchor_projected_gravity",
  "motion_joint_pos",
  "motion_joint_vel",
  "motion_anchor_lin_vel_b",
  "motion_anchor_ang_vel_b",
)

EXPECTED_REWARD_WEIGHTS = {
  "motion_global_root_pos": 0.5,
  "motion_global_root_ori": 0.5,
  "motion_body_pos": 1.0,
  "motion_body_ori": 1.0,
  "motion_body_lin_vel": 1.0,
  "motion_body_ang_vel": 1.0,
  "jump_com_ballistic_rise": 5.0,
  "action_rate_l2": -0.1,
  "joint_limit": -10.0,
  "self_collisions": -10.0,
}

TRACKING_REWARD_NAMES = tuple(EXPECTED_REWARD_WEIGHTS)[:6]
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
PADDED_JUMP_MOTION = Path("motions/02_04_jump_balance_pad50.npz")


class _FakeScene(dict):
  def __init__(self, env_origins: torch.Tensor, **entities) -> None:
    super().__init__(entities)
    self.env_origins = env_origins


def _make_fake_motion_command(
  env_origins: torch.Tensor,
  reference_foot_z: torch.Tensor,
  robot_foot_z: torch.Tensor,
  reference_time_steps: torch.Tensor | None = None,
) -> tuple[MotionCommand, SimpleNamespace]:
  num_envs = env_origins.shape[0]
  command_any = cast(Any, object.__new__(MotionCommand))
  command_any.cfg = SimpleNamespace(body_names=FOOT_BODY_NAMES)
  command_any.time_steps = (
    torch.arange(num_envs) if reference_time_steps is None else reference_time_steps
  )
  num_reference_frames = reference_foot_z.shape[0]
  reference_body_pos = torch.zeros((num_reference_frames, 2, 3))
  reference_body_pos[:, :, 2] = reference_foot_z
  command_any.motion = SimpleNamespace(
    body_pos_w=reference_body_pos,
    body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 2] * num_reference_frames),
    joint_pos=torch.zeros((num_reference_frames, 1)),
    time_step_total=num_reference_frames,
  )
  command_any.body_indexes = torch.tensor([0, 1], dtype=torch.long)
  robot_body_pos = torch.zeros((num_envs, 2, 3))
  robot_body_pos[:, :, 2] = robot_foot_z + env_origins[:, 2].unsqueeze(1)
  command_any.robot = SimpleNamespace(
    data=SimpleNamespace(body_link_pos_w=robot_body_pos)
  )
  scene = _FakeScene(env_origins, robot=command_any.robot)
  command_any._env = SimpleNamespace(scene=scene)
  command = cast(MotionCommand, command_any)
  command_manager = SimpleNamespace(get_term=lambda _: command)
  return command, command_manager


def _set_fake_body_com_state(
  command: MotionCommand,
  body_com_z_velocity: torch.Tensor,
  body_masses: torch.Tensor,
) -> torch.Tensor:
  """Attach per-body COM velocities and per-world masses to a fake robot."""
  num_envs, num_bodies = body_com_z_velocity.shape
  body_ids = torch.arange(1, num_bodies + 1, dtype=torch.long)
  body_com_lin_vel_w = torch.zeros((num_envs, num_bodies, 3))
  body_com_lin_vel_w[:, :, 2] = body_com_z_velocity
  model_body_mass = torch.zeros((num_envs, num_bodies + 1))
  model_body_mass[:, body_ids] = body_masses
  robot_data = cast(Any, command.robot.data)
  robot_data.body_com_lin_vel_w = body_com_lin_vel_w
  robot_data.indexing = SimpleNamespace(body_ids=body_ids)
  robot_data.model = SimpleNamespace(body_mass=model_body_mass)
  return body_com_lin_vel_w


def _set_fake_foot_contacts(
  command: MotionCommand,
  found: torch.Tensor,
) -> torch.Tensor:
  """Attach mutable found-only foot contact data to a fake scene."""
  scene = cast(Any, command._env.scene)
  scene["feet_ground_contact"] = SimpleNamespace(data=SimpleNamespace(found=found))
  return found


def test_jumping_height_reward_and_termination_config() -> None:
  cfg = unitree_g1_jumping_height_baseline_env_cfg()
  tracking_cfg = unitree_g1_jumping_track_baseline_env_cfg()

  assert {name: term.weight for name, term in cfg.rewards.items()} == (
    EXPECTED_REWARD_WEIGHTS
  )
  for name in TRACKING_REWARD_NAMES:
    term = cfg.rewards[name]
    inherited_term = tracking_cfg.rewards[name]
    assert term.func is inherited_term.func
    assert term.params == inherited_term.params

  upward = cfg.rewards["jump_com_ballistic_rise"]
  assert upward.func is jumping_mdp.jump_com_ballistic_rise
  assert upward.params == {
    "command_name": "motion",
    "sensor_name": "feet_ground_contact",
    "foot_body_names": FOOT_BODY_NAMES,
    "airborne_height_threshold": 0.06,
    "contact_confirmation_steps": 2,
    "gravity": 9.81,
  }
  assert {
    "is_alive",
    "horizontal_root_velocity_l2",
    "pelvis_torso_roll_l2",
    "landing_impact",
    "joint_torques_normalized",
    "joint_symmetry",
    "waist_deviation",
  }.isdisjoint(cfg.rewards)
  assert tuple(cfg.terminations) == ("time_out", "low_height", "bad_orientation")
  low_height = cfg.terminations["low_height"]
  assert low_height.func is envs_mdp.root_height_below_minimum
  assert low_height.params == {"minimum_height": 0.30}
  assert low_height.time_out is False
  bad_orientation = cfg.terminations["bad_orientation"]
  assert bad_orientation.func is envs_mdp.bad_orientation
  assert bad_orientation.params["limit_angle"] == pytest.approx(math.radians(80.0))
  assert bad_orientation.time_out is False
  assert cfg.curriculum == {}

  assert tuple(cfg.metrics) == (
    "last_jump_peak_root_height",
    "last_takeoff_com_z_velocity",
    "last_jump_ballistic_rise",
    "jump_bonus_event_count",
    "jump_bonus_opportunity_consumed",
    "pelvis_roll_l2",
    "torso_roll_l2",
  )
  height_metric = cfg.metrics["last_jump_peak_root_height"]
  assert height_metric.func is jumping_mdp.last_jump_peak_root_height
  assert height_metric.params == {
    "command_name": "motion",
    "foot_body_names": FOOT_BODY_NAMES,
    "airborne_height_threshold": 0.06,
    "base_height": 0.79,
  }
  assert height_metric.reduce == "last"
  reward_metrics = {
    "last_takeoff_com_z_velocity": "last_takeoff_com_z_velocity",
    "last_jump_ballistic_rise": "last_ballistic_rise",
    "jump_bonus_event_count": "bonus_event_count",
    "jump_bonus_opportunity_consumed": "opportunity_consumed",
  }
  for metric_name, attribute in reward_metrics.items():
    metric = cfg.metrics[metric_name]
    assert metric.func is jumping_mdp.reward_term_state
    assert metric.params == {
      "reward_name": "jump_com_ballistic_rise",
      "attribute": attribute,
    }
    assert metric.reduce == "last"
  pelvis_roll = cfg.metrics["pelvis_roll_l2"]
  assert pelvis_roll.func is jumping_mdp.body_roll_l2
  assert pelvis_roll.params == {
    "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
  }
  assert pelvis_roll.reduce == "mean"
  torso_roll = cfg.metrics["torso_roll_l2"]
  assert torso_roll.func is jumping_mdp.body_roll_l2
  assert torso_roll.params == {
    "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
  }
  assert torso_roll.reduce == "mean"


def test_jumping_height_fall_termination_thresholds() -> None:
  tilt_angles = torch.deg2rad(torch.tensor([79.999, 80.0, 80.001], dtype=torch.float64))
  projected_gravity = torch.stack(
    [
      torch.sin(tilt_angles),
      torch.zeros_like(tilt_angles),
      -torch.cos(tilt_angles),
    ],
    dim=-1,
  )
  root_positions = torch.zeros((3, 3))
  root_positions[:, 2] = torch.tensor([0.299, 0.300, 0.301])
  robot = SimpleNamespace(
    data=SimpleNamespace(
      projected_gravity_b=projected_gravity,
      root_link_pos_w=root_positions,
    )
  )
  env = cast(ManagerBasedRlEnv, SimpleNamespace(scene={"robot": robot}))

  low_height = envs_mdp.root_height_below_minimum(env, minimum_height=0.30)
  bad_orientation = envs_mdp.bad_orientation(env, limit_angle=math.radians(80.0))

  assert low_height.tolist() == [True, False, False]
  assert bad_orientation.tolist() == [False, False, True]
  assert (low_height | bad_orientation).tolist() == [True, False, True]


def test_canonical_jump_stays_inside_fall_termination_limits() -> None:
  data = np.load(PADDED_JUMP_MOTION)
  root_pos = torch.from_numpy(data["body_pos_w"][:, 0])
  root_quat = torch.from_numpy(data["body_quat_w"][:, 0])
  cfg = unitree_g1_jumping_height_baseline_env_cfg()
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)

  lowest_perturbed_height = root_pos[:, 2].min() + motion.pose_range["z"][0]
  assert lowest_perturbed_height.item() > 0.30

  maximum_tilt = torch.tensor(0.0)
  for roll in (-0.1, 0.0, 0.1):
    for pitch in (-0.1, 0.0, 0.1):
      for yaw in (-0.2, 0.0, 0.2):
        delta = quat_from_euler_xyz(
          torch.tensor([roll]),
          torch.tensor([pitch]),
          torch.tensor([yaw]),
        )
        perturbed = quat_mul(delta.expand_as(root_quat), root_quat)
        root_up_z = 1.0 - 2.0 * (perturbed[:, 1].square() + perturbed[:, 2].square())
        tilt = torch.acos(root_up_z.clamp(-1.0, 1.0))
        maximum_tilt = torch.maximum(maximum_tilt, tilt.max())

  assert maximum_tilt.item() < math.radians(80.0)


def _make_ballistic_reward_fixture(
  num_envs: int,
) -> tuple[
  ManagerBasedRlEnv,
  MotionCommand,
  torch.Tensor,
  torch.Tensor,
  jumping_mdp.jump_com_ballistic_rise,
]:
  reference_foot_z = torch.tensor(
    [
      [0.03, 0.03],
      [0.03, 0.03],
      [0.10, 0.10],
      [0.10, 0.10],
      [0.10, 0.10],
      [0.10, 0.10],
      [0.10, 0.10],
      [0.03, 0.03],
      [0.03, 0.03],
      [0.10, 0.10],
      [0.10, 0.10],
    ]
  )
  command, command_manager = _make_fake_motion_command(
    torch.zeros((num_envs, 3)),
    reference_foot_z,
    torch.full((num_envs, 2), 0.03),
    reference_time_steps=torch.zeros(num_envs, dtype=torch.long),
  )
  body_com_lin_vel_w = _set_fake_body_com_state(
    command,
    torch.zeros((num_envs, 2)),
    torch.ones((num_envs, 2)),
  )
  contacts = _set_fake_foot_contacts(
    command, torch.ones((num_envs, 2), dtype=torch.long)
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene=command._env.scene,
      command_manager=command_manager,
      num_envs=num_envs,
      device="cpu",
      step_dt=0.02,
    ),
  )
  cfg = RewardTermCfg(
    func=jumping_mdp.jump_com_ballistic_rise,
    weight=EXPECTED_REWARD_WEIGHTS["jump_com_ballistic_rise"],
    params={
      "command_name": "motion",
      "sensor_name": "feet_ground_contact",
      "foot_body_names": FOOT_BODY_NAMES,
      "airborne_height_threshold": 0.06,
      "contact_confirmation_steps": 2,
      "gravity": 9.81,
    },
  )
  reward = jumping_mdp.jump_com_ballistic_rise(cfg, env)
  return env, command, contacts, body_com_lin_vel_w, reward


def test_jump_com_ballistic_rise_is_mass_weighted_and_dt_normalized() -> None:
  env, command, contacts, velocities, reward = _make_ballistic_reward_fixture(3)
  robot_data = cast(Any, command.robot.data)
  robot_data.model.body_mass[:, 1:] = torch.tensor([[1.0, 3.0], [1.0, 3.0], [2.0, 1.0]])

  # Two grounded samples arm and confirm the robot is standing on both feet.
  command.time_steps[:] = 0
  torch.testing.assert_close(reward(env), torch.zeros(3))
  command.time_steps[:] = 1
  torch.testing.assert_close(reward(env), torch.zeros(3))
  assert reward.takeoff_ready.tolist() == [True, True, True]

  # The first airborne sample stores takeoff velocity but does not yet pay.
  contacts.zero_()
  velocities[:, :, 2] = torch.tensor([[4.0, 0.0], [3.0, -1.0], [-1.0, 4.0]])
  command.time_steps[:] = 2
  torch.testing.assert_close(reward(env), torch.zeros(3))
  expected_velocity = torch.tensor([1.0, 0.0, 2.0 / 3.0])
  torch.testing.assert_close(reward.com_z_velocity, expected_velocity)

  # Confirmation pays v_z^2/(2g)/dt exactly once. Multiplying by the reward
  # manager's dt and weight recovers weight * ballistic rise.
  velocities[:, :, 2] = 20.0  # The stored first-airborne velocity must be used.
  command.time_steps[:] = 3
  raw_rate = reward(env)
  expected_rise = expected_velocity.clamp(min=0.0).square() / (2.0 * 9.81)
  torch.testing.assert_close(raw_rate, expected_rise / env.step_dt)
  weight = EXPECTED_REWARD_WEIGHTS["jump_com_ballistic_rise"]
  torch.testing.assert_close(raw_rate * env.step_dt * weight, expected_rise * weight)
  torch.testing.assert_close(reward.last_ballistic_rise, expected_rise)
  assert reward.opportunity_consumed.tolist() == [True, True, True]
  assert reward.bonus_event_count.tolist() == [1, 1, 1]

  command.time_steps[:] = 4
  torch.testing.assert_close(reward(env), torch.zeros(3))


def test_jump_com_ballistic_rise_rejects_chatter_and_one_foot_flight() -> None:
  env, command, contacts, velocities, reward = _make_ballistic_reward_fixture(1)
  velocities[:, :, 2] = 2.0
  for frame in (0, 1):
    command.time_steps[0] = frame
    torch.testing.assert_close(reward(env), torch.zeros(1))

  contacts[0] = torch.tensor([0, 1])
  command.time_steps[0] = 2
  torch.testing.assert_close(reward(env), torch.zeros(1))
  assert reward.airborne_streak.item() == 0

  contacts.zero_()
  command.time_steps[0] = 3
  torch.testing.assert_close(reward(env), torch.zeros(1))
  contacts[0, 0] = 1  # A single-step loss of shared flight is contact chatter.
  command.time_steps[0] = 4
  torch.testing.assert_close(reward(env), torch.zeros(1))
  assert reward.candidate_pending.item() is False

  contacts.zero_()
  command.time_steps[0] = 5
  torch.testing.assert_close(reward(env), torch.zeros(1))
  command.time_steps[0] = 6
  assert reward(env).item() == pytest.approx((2.0**2 / (2.0 * 9.81)) / 0.02)


def test_jump_com_ballistic_rise_does_not_rearm_after_reference_landing() -> None:
  env, command, contacts, velocities, reward = _make_ballistic_reward_fixture(2)
  velocities[:, :, 2] = torch.tensor([[2.0, 2.0], [1.0, 1.0]])
  for frame in (0, 1):
    command.time_steps[:] = frame
    torch.testing.assert_close(reward(env), torch.zeros(2))

  contacts.zero_()
  command.time_steps[:] = 2
  torch.testing.assert_close(reward(env), torch.zeros(2))
  command.time_steps[:] = 3
  value = reward(env)
  assert torch.all(value > 0.0)

  # Landing and taking off again during the same reference flight cannot pay.
  contacts.fill_(1)
  command.time_steps[:] = 4
  torch.testing.assert_close(reward(env), torch.zeros(2))
  contacts.zero_()
  command.time_steps[:] = 5
  torch.testing.assert_close(reward(env), torch.zeros(2))
  command.time_steps[:] = 6
  torch.testing.assert_close(reward(env), torch.zeros(2))

  # Reference landing and another complete robot jump in the same motion
  # segment do not open a second opportunity.
  contacts.fill_(1)
  command.time_steps[:] = 7
  torch.testing.assert_close(reward(env), torch.zeros(2))
  command.time_steps[:] = 8
  torch.testing.assert_close(reward(env), torch.zeros(2))
  contacts[0] = 0
  command.time_steps[:] = 9
  torch.testing.assert_close(reward(env), torch.zeros(2))
  command.time_steps[:] = 10
  second = reward(env)
  torch.testing.assert_close(second, torch.zeros(2))
  assert reward.opportunity_consumed.tolist() == [True, True]
  assert reward.bonus_event_count.tolist() == [1, 1]

  reward.reset(torch.tensor([0]))
  assert reward.bonus_event_count.tolist() == [0, 1]
  assert reward.previous_time_steps.tolist() == [-1, 10]


def test_body_roll_l2_sums_selected_squared_world_rolls() -> None:
  roll = torch.tensor(
    [
      [0.3, 1.2, -0.4],
      [-0.3, -1.2, 0.4],
      [0.0, 0.9, 0.0],
      [0.0, -0.9, 0.0],
    ]
  )
  pitch = torch.tensor(
    [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.5, 0.0, -0.7],
      [0.0, 0.0, 0.0],
    ]
  )
  yaw = torch.tensor(
    [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.8, 0.0, -1.1],
    ]
  )
  body_quat_w = quat_from_euler_xyz(
    roll.flatten(), pitch.flatten(), yaw.flatten()
  ).reshape(4, 3, 4)
  robot = SimpleNamespace(data=SimpleNamespace(body_link_quat_w=body_quat_w))
  env = cast(ManagerBasedRlEnv, SimpleNamespace(scene={"robot": robot}))
  asset_cfg = SceneEntityCfg("robot", body_ids=[0, 2])

  cost = jumping_mdp.body_roll_l2(env, asset_cfg=asset_cfg)

  torch.testing.assert_close(cost, torch.tensor([0.25, 0.25, 0.0, 0.0]))


def test_jump_com_ballistic_rise_rearms_after_command_resampling() -> None:
  env, command, contacts, velocities, reward = _make_ballistic_reward_fixture(1)
  velocities[:, :, 2] = 1.5
  for frame in (0, 1):
    command.time_steps[0] = frame
    reward(env)
  contacts.zero_()
  command.time_steps[0] = 2
  torch.testing.assert_close(reward(env), torch.zeros(1))
  command.time_steps[0] = 3
  first = reward(env)
  assert first.item() > 0.0
  assert reward.bonus_event_count.item() == 1

  # The motion command teleports/resamples at clip end. A discontinuous jump
  # into a grounded reference phase starts a new independent opportunity.
  contacts.fill_(1)
  command.time_steps[0] = 0
  torch.testing.assert_close(reward(env), torch.zeros(1))
  command.time_steps[0] = 1
  torch.testing.assert_close(reward(env), torch.zeros(1))
  contacts.zero_()
  command.time_steps[0] = 2
  torch.testing.assert_close(reward(env), torch.zeros(1))
  command.time_steps[0] = 3
  assert reward(env).item() > 0.0
  assert reward.bonus_event_count.item() == 2


def test_last_jump_peak_root_height_tracks_completed_robot_jumps() -> None:
  origins = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
  command, command_manager = _make_fake_motion_command(
    origins,
    torch.full((2, 2), 0.1),
    torch.full((2, 2), 0.03),
  )
  root_pose = torch.zeros((2, 7))
  root_pose[:, 2] = origins[:, 2] + 0.79
  robot_data = cast(Any, command.robot.data)
  robot_data.root_link_pose_w = root_pose
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene=command._env.scene,
      command_manager=command_manager,
      num_envs=2,
      device="cpu",
    ),
  )
  cfg = MetricsTermCfg(
    func=jumping_mdp.last_jump_peak_root_height,
    params={
      "command_name": "motion",
      "foot_body_names": FOOT_BODY_NAMES,
      "airborne_height_threshold": 0.06,
      "base_height": 0.79,
    },
  )
  metric = jumping_mdp.last_jump_peak_root_height(cfg, env)

  torch.testing.assert_close(metric(env), torch.zeros(2))
  command.robot.data.body_link_pos_w[0, :, 2] = 0.1
  command.robot.data.root_link_pose_w[0, 2] = 1.0
  torch.testing.assert_close(metric(env), torch.zeros(2))
  command.robot.data.root_link_pose_w[0, 2] = 1.2
  command.robot.data.body_link_pos_w[1, :, 2] = origins[1, 2] + 0.1
  command.robot.data.root_link_pose_w[1, 2] = origins[1, 2] + 1.1
  torch.testing.assert_close(metric(env), torch.zeros(2))
  command.robot.data.body_link_pos_w[0, :, 2] = 0.03
  command.robot.data.root_link_pose_w[1, 2] = origins[1, 2] + 1.3
  torch.testing.assert_close(metric(env), torch.tensor([0.41, 0.0]))
  command.robot.data.body_link_pos_w[1, :, 2] = origins[1, 2] + 0.03
  torch.testing.assert_close(metric(env), torch.tensor([0.41, 0.51]))

  command.robot.data.body_link_pos_w[0, :, 2] = 0.1
  command.robot.data.root_link_pose_w[0, 2] = 1.1
  torch.testing.assert_close(metric(env), torch.tensor([0.41, 0.51]))
  command.robot.data.root_link_pose_w[0, 2] = 1.4
  torch.testing.assert_close(metric(env), torch.tensor([0.41, 0.51]))
  command.robot.data.body_link_pos_w[0, :, 2] = 0.03
  torch.testing.assert_close(metric(env), torch.tensor([0.61, 0.51]))

  metric.reset(torch.tensor([0]))
  torch.testing.assert_close(metric.last_peak, torch.tensor([0.0, 0.51]))
  assert metric.was_airborne.tolist() == [False, False]


def test_jumping_height_sensor_and_observation_contract() -> None:
  cfg = unitree_g1_jumping_height_baseline_env_cfg()
  sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
  assert "feet_ground_contact" in sensors
  assert "self_collision" in sensors
  feet_contact = sensors["feet_ground_contact"]
  assert isinstance(feet_contact, ContactSensorCfg)
  assert feet_contact.primary.mode == "subtree"
  assert feet_contact.primary.pattern == (
    r"^(left_ankle_roll_link|right_ankle_roll_link)$"
  )
  assert feet_contact.primary.entity == "robot"
  assert feet_contact.secondary is not None
  assert feet_contact.secondary.mode == "body"
  assert feet_contact.secondary.pattern == "terrain"
  assert feet_contact.fields == ("found",)
  assert feet_contact.reduce == "none"
  assert feet_contact.num_slots == 1
  assert feet_contact.track_air_time is False

  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms
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

  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  reference_dim = 1 + 3 + 29 + 29 + 3 + 3
  actor_dim = reference_dim + 3 + 3 + 29 + 29 + 29
  critic_dim = (
    reference_dim
    + 3
    + 6
    + len(motion.body_names) * 3
    + len(motion.body_names) * 6
    + 3
    + 3
    + 3
    + 29
    + 29
    + 29
  )
  assert actor_dim == 161
  assert critic_dim == 299


@pytest.mark.slow
def test_reference_tracking_rewards_are_active_over_full_motion() -> None:
  cfg = unitree_g1_jumping_height_baseline_env_cfg(play=True)
  cfg.scene.num_envs = 1
  cfg.events = {}
  cfg.terminations = {}
  cfg.rewards = {
    name: term for name, term in cfg.rewards.items() if name in TRACKING_REWARD_NAMES
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
    command = cast(MotionCommand, env.command_manager.get_term("motion"))
    env_ids = torch.tensor([0], device=env.device)
    for frame in (0, 100, 250):
      command.reset_to_frame(env_ids, frame)
      env.scene.write_data_to_sim()
      env.sim.forward()
      command.update_relative_body_poses()
      env.reward_manager.compute(dt=env.step_dt)
      terms = dict(env.reward_manager.get_active_iterable_terms(0))
      for name in TRACKING_REWARD_NAMES:
        assert terms[name][0] == pytest.approx(EXPECTED_REWARD_WEIGHTS[name], abs=2e-4)
  finally:
    env.close()


def test_jumping_height_preserves_baseline_and_uan_variants() -> None:
  baseline = unitree_g1_jumping_height_baseline_env_cfg()
  uan = unitree_g1_jumping_height_uan_env_cfg(uan_checkpoint_path="/tmp/test-uan.pt")

  assert baseline.scene.num_envs == uan.scene.num_envs == 16_384
  assert isinstance(
    baseline.actions["joint_pos"], TorqueSpeedClippedJointPositionActionCfg
  )
  assert {"push_robot", "foot_friction", "encoder_bias", "base_com"} <= set(
    baseline.events
  )
  assert baseline.events["push_robot"].func.__name__ == "push_by_setting_velocity"
  assert "command_name" not in baseline.events["push_robot"].params

  assert isinstance(
    uan.actions["joint_pos"],
    UANAugmentedTorqueSpeedClippedJointPositionActionCfg,
  )
  assert {"push_robot", "foot_friction"} <= set(uan.events)
  assert {"encoder_bias", "base_com"}.isdisjoint(uan.events)
  assert uan.events["push_robot"].func.__name__ == "push_by_setting_velocity"


def test_jumping_height_play_is_clean() -> None:
  for factory in (
    unitree_g1_jumping_height_baseline_env_cfg,
    unitree_g1_jumping_height_uan_env_cfg,
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
    assert tuple(cfg.terminations) == (
      "time_out",
      "low_height",
      "bad_orientation",
    )


def test_jumping_height_registration_and_runner() -> None:
  task_ids = {task for task in list_tasks() if "JumpingHeight" in task}
  baseline_id = "Mjlab-JumpingHeight-Baseline-Unitree-G1"
  uan_id = "Mjlab-JumpingHeight-UAN-Unitree-G1"
  assert baseline_id in task_ids
  if has_uan_checkpoint_env():
    assert uan_id in task_ids
  else:
    assert uan_id not in task_ids

  cfg = load_env_cfg(baseline_id)
  agent = load_rl_cfg(baseline_id)
  assert isinstance(agent, RslRlOnPolicyRunnerCfg)
  assert cfg.episode_length_s == 20.0
  assert agent.experiment_name == "g1_jumping_track"
  assert agent.wandb_project == "mjlab-g1-jumping-track"
  assert agent.algorithm.entropy_coef == 0.005
  assert unitree_g1_jumping_track_ppo_runner_cfg().algorithm.entropy_coef == 0.005
  assert load_runner_cls(baseline_id) is MotionTrackingOnPolicyRunner
