"""Custom Unitree G1 velocity env: bumpy terrain, no heading."""

from __future__ import annotations

import math

from mjlab.asset_zoo.robots import (
  G1_TORQUE_SPEED_SATURATION_EFFORT,
  G1_TORQUE_SPEED_VELOCITY_LIMIT,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.utils.g1_variants import (
  G1_FOOT_GEOM_NAMES,
  apply_uan_training_dr,
  strip_variant_training_events_for_play,
  swap_in_torque_speed_clipped_action,
  swap_in_uan_torque_speed_clipped_action,
)
from mjlab.tasks.utils.terrain_profiles import BUMPY_TERRAINS_CFG
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def _g1_heel_and_motor_metrics() -> dict[str, MetricsTermCfg]:
  """Logging metrics shared by custom G1 velocity variants."""
  return {
    "heel_strike_angle_rad": MetricsTermCfg(
      func=velocity_mdp.heel_strike_angle_by_command,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg(
          "robot",
          body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        ),
        "command_name": "twist",
        "command_bins": velocity_mdp.DEFAULT_HEEL_STRIKE_COMMAND_BINS,
      },
    ),
    "motor_static_sat_frac_ankle": MetricsTermCfg(
      func=velocity_mdp.motor_saturation_fraction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
        ),
        "action_name": "joint_pos",
        "mode": "static",
        "threshold": 0.95,
      },
    ),
    "motor_dynamic_sat_frac_ankle": MetricsTermCfg(
      func=velocity_mdp.motor_saturation_fraction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
        ),
        "action_name": "joint_pos",
        "mode": "dynamic",
        "threshold": 1.0,
        "saturation_effort": G1_TORQUE_SPEED_SATURATION_EFFORT,
        "velocity_limit": G1_TORQUE_SPEED_VELOCITY_LIMIT,
      },
    ),
    "motor_static_sat_frac_hip": MetricsTermCfg(
      func=velocity_mdp.motor_saturation_fraction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            ".*_hip_pitch_joint",
            ".*_hip_roll_joint",
            ".*_hip_yaw_joint",
          ),
        ),
        "action_name": "joint_pos",
        "mode": "static",
        "threshold": 0.95,
      },
    ),
    "motor_dynamic_sat_frac_hip": MetricsTermCfg(
      func=velocity_mdp.motor_saturation_fraction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            ".*_hip_pitch_joint",
            ".*_hip_roll_joint",
            ".*_hip_yaw_joint",
          ),
        ),
        "action_name": "joint_pos",
        "mode": "dynamic",
        "threshold": 1.0,
        "saturation_effort": G1_TORQUE_SPEED_SATURATION_EFFORT,
        "velocity_limit": G1_TORQUE_SPEED_VELOCITY_LIMIT,
      },
    ),
  }


def unitree_g1_custom_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 velocity on bumpy terrain, no heading."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.scene.num_envs = 4096
  motion_epsilon = 1e-3

  swap_in_torque_speed_clipped_action(cfg)

  # 1. Remove heading mode.
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.heading_command = False
  twist.rel_heading_envs = 0.0
  twist.rel_turn_envs = 0.2
  twist.ranges.ang_vel_z = (-1.0, 1.0)
  twist.ranges.heading = None

  # 2. Swap flat terrain for bumpy.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = BUMPY_TERRAINS_CFG

  # Keep the lighter contact profile used by fast bumpy-terrain WTW training.
  cfg.sim.nconmax = None
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  # 3. Observation history
  cfg.observations["actor"].terms.pop("base_lin_vel", None)
  cfg.observations["actor"].history_length = 5
  cfg.observations["critic"].history_length = 5

  # 4. Heavier penalty on joint limits
  cfg.rewards["dof_pos_limits"].weight = -10.0

  # Track only yaw angular velocity for the custom task.
  cfg.rewards["track_angular_velocity"].func = velocity_mdp.track_angular_velocity_z
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.25)

  cfg.metrics.update(_g1_heel_and_motor_metrics())

  # Rely on explicit standing envs instead of a small-command standing deadzone.
  cfg.rewards["pose"].params["walking_threshold"] = motion_epsilon
  for reward_name in (
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "soft_landing",
  ):
    cfg.rewards[reward_name].params["command_threshold"] = motion_epsilon

  # 5. Low height termination (prevent sitting-down local minimum)
  cfg.terminations["low_height"] = TerminationTermCfg(
    func=velocity_mdp.low_height,
    params={"minimum_height": 0.5},
  )

  return cfg


def unitree_g1_custom_uan_env_cfg(
  play: bool = False,
  uan_checkpoint_path: str | None = None,
) -> ManagerBasedRlEnvCfg:
  """G1 custom velocity + shared UAN preset."""
  cfg = unitree_g1_custom_env_cfg(play=play)
  swap_in_uan_torque_speed_clipped_action(
    cfg,
    checkpoint_path=uan_checkpoint_path,
  )
  apply_uan_training_dr(cfg, foot_geom_names=G1_FOOT_GEOM_NAMES)
  if play:
    strip_variant_training_events_for_play(cfg)
  return cfg


def unitree_g1_custom_uan_eval_env_cfg(
  uan_checkpoint_path: str | None = None,
) -> ManagerBasedRlEnvCfg:
  """G1 custom velocity UAN eval variant."""
  return unitree_g1_custom_uan_env_cfg(
    play=True,
    uan_checkpoint_path=uan_checkpoint_path,
  )
