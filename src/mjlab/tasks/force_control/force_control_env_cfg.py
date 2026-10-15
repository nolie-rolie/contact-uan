"""Force control environment config factory.

Provides:
- make_force_control_env_cfg(): Force control training config
- make_force_control_uan_env_cfg(): Force control + shared UAN preset
- make_force_control_eval_env_cfg(): Eval variant
- make_force_control_uan_eval_env_cfg(): UAN eval variant
"""

from __future__ import annotations

import copy
import math
from typing import cast

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.force_control import mdp as fc_mdp
from mjlab.tasks.force_control.mdp.actions import ForceFieldActionCfg
from mjlab.tasks.utils.g1_variants import (
  G1_FOOT_GEOM_NAMES,
  apply_uan_training_dr,
  apply_velocity_baseline_dr,
  strip_variant_training_events_for_play,
  swap_in_uan_action,
)
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def _self_collision_sensor_cfg() -> ContactSensorCfg:
  return ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )


def _build_force_control_env_cfg(
  actions: dict[str, ActionTermCfg],
) -> ManagerBasedRlEnvCfg:
  """Build a force control env config with the given action config."""

  robot_cfg = get_g1_robot_cfg()
  robot_cfg.init_state = copy.deepcopy(robot_cfg.init_state)
  joint_pos = cast(dict[str, float], robot_cfg.init_state.joint_pos)

  # Standing pose: arms relaxed, slightly bent legs.
  joint_pos["left_shoulder_pitch_joint"] = 0.0
  joint_pos["right_shoulder_pitch_joint"] = 0.0
  joint_pos["left_shoulder_roll_joint"] = 0.0
  joint_pos["right_shoulder_roll_joint"] = 0.0
  joint_pos[".*_shoulder_yaw_joint"] = 0.0
  joint_pos[".*_elbow_joint"] = 0.0
  joint_pos[".*_wrist_.*joint"] = 0.0
  joint_pos[".*_hip_pitch_joint"] = -0.38
  joint_pos[".*_knee_joint"] = 0.82
  joint_pos[".*_ankle_pitch_joint"] = -0.44
  joint_pos[".*_hip_roll_joint"] = 0.0
  joint_pos[".*_hip_yaw_joint"] = 0.0
  joint_pos[".*_ankle_roll_joint"] = 0.0

  feet_contact = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  scene = SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": robot_cfg},
    sensors=(feet_contact, _self_collision_sensor_cfg()),
    num_envs=4096,
    extent=2.0,
  )

  max_force_z = 30.0
  force_scale = 1.0 / max_force_z

  # -- Observations --------------------------------------------------------
  base_lin_vel = ObservationTermCfg(
    func=envs_mdp.base_lin_vel,
    noise=Unoise(n_min=-0.5, n_max=0.5),
  )
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
      scale=0.25,
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos_rel": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel_rel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      scale=0.05,
    ),
    "last_action": ObservationTermCfg(func=envs_mdp.last_action),
    "force_command": ObservationTermCfg(
      func=fc_mdp.force_command,
      params={"command_name": "desired_force"},
      scale=force_scale,
    ),
  }

  critic_terms = {"base_lin_vel": base_lin_vel, **actor_terms}
  critic_terms["force_applied"] = ObservationTermCfg(
    func=fc_mdp.forcefield_force_applied,
    params={"command_name": "desired_force"},
    scale=force_scale,
  )
  critic_terms["foot_air_time"] = ObservationTermCfg(
    func=vel_mdp.foot_air_time,
    params={"sensor_name": "feet_ground_contact"},
  )
  critic_terms["foot_contact"] = ObservationTermCfg(
    func=vel_mdp.foot_contact,
    params={"sensor_name": "feet_ground_contact"},
  )
  critic_terms["foot_contact_forces"] = ObservationTermCfg(
    func=vel_mdp.foot_contact_forces,
    params={"sensor_name": "feet_ground_contact"},
  )

  obs_history_length = 20

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=obs_history_length,
      flatten_history_dim=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=obs_history_length,
      flatten_history_dim=True,
    ),
  }

  # -- Commands ------------------------------------------------------------
  commands: dict[str, CommandTermCfg] = {
    "desired_force": fc_mdp.ForceFieldCommandCfg(
      entity_name="robot",
      body_name="right_wrist_yaw_link",
      resampling_time_range=(10.0, 10.0),
      force_range=((-30.0, -30.0, -max_force_z), (30.0, 30.0, max_force_z)),
      max_applied_force=50.0,
      max_xy_reach_m=0.3,
      stiffness_range=(25.0, 400.0),
      damping_proportional_to_stiffness=0.1,
      sample_zero_force_probability=0.2,
      force_ramp_duration_range_s=(2.0, 4.0),
      force_hold_duration_s=2.0,
      startup_zero_force_duration_s=2.0,
      debug_vis=True,
    ),
  }

  # -- Events --------------------------------------------------------------
  push_velocity_range = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
  }

  events = {
    "reset_base": EventTermCfg(
      func=vel_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.06, 0.06),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {
          "x": (-0.2, 0.2),
          "y": (-0.2, 0.2),
          "z": (-0.1, 0.1),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.52, 0.52),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=vel_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.2, 0.2),
        "velocity_range": (-0.1, 0.1),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=vel_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": push_velocity_range},
    ),
    "physics_material": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
        "operation": "abs",
        "ranges": (0.5, 2.0),
      },
    ),
    "randomize_joint_parameters": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "add",
        "ranges": (0.01, 0.1),
      },
    ),
    "randomize_joint_damping": EventTermCfg(
      mode="reset",
      func=envs_mdp.dr.joint_damping,
      min_step_count_between_reset=720,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            ".*_hip_.*_joint",
            ".*_knee_joint",
            ".*_ankle_.*_joint",
            "waist_.*_joint",
          ),
        ),
        "operation": "add",
        "ranges": (0.0, 2.0),
      },
    ),
    # Force field application moved to ForceFieldAction (runs every substep).
  }

  # -- Rewards -------------------------------------------------------------
  rewards = {
    "track_force_ee": RewardTermCfg(
      func=fc_mdp.track_force_ee,
      weight=3.0,
      params={"command_name": "desired_force", "sigma": math.sqrt(50.0)},
    ),
    "track_lin_vel_xy_exp": RewardTermCfg(
      func=fc_mdp.track_lin_vel_xy_exp_zero,
      weight=2.0,
      params={"std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("robot")},
    ),
    "track_ang_vel_z_exp": RewardTermCfg(
      func=fc_mdp.track_ang_vel_z_exp_zero,
      weight=2.0,
      params={"std": math.sqrt(0.5), "asset_cfg": SceneEntityCfg("robot")},
    ),
    "upright": RewardTermCfg(
      func=vel_mdp.upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      },
    ),
    "pose": RewardTermCfg(
      func=fc_mdp.fixed_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "std": {
          r"left_.*|waist_.*|.*_hip_.*|.*_knee_.*|.*_ankle_.*|right_wrist_.*": 0.2,
          r"right_shoulder_.*": 0.2,
          r"right_elbow_.*": 0.2,
        },
      },
    ),
    "dof_pos_limits": RewardTermCfg(
      func=vel_mdp.joint_pos_limits,
      weight=-1.0,
    ),
    "action_rate_l2": RewardTermCfg(
      func=vel_mdp.action_rate_l2,
      weight=-0.1,
    ),
    "body_ang_vel": RewardTermCfg(
      func=vel_mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
    ),
    "angular_momentum": RewardTermCfg(
      func=vel_mdp.angular_momentum_penalty,
      weight=-0.02,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "self_collisions": RewardTermCfg(
      func=vel_mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }

  # -- Terminations --------------------------------------------------------
  terminations = {
    "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
    "bad_orientation": TerminationTermCfg(
      func=vel_mdp.bad_orientation,
      params={"limit_angle": math.pi / 4},
    ),
    "low_height": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.4},
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=None,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=50,
      ),
      contact_sensor_maxmatch=64,
    ),
    decimation=4,
    episode_length_s=20.0,
  )
  apply_velocity_baseline_dr(cfg, foot_geom_names=G1_FOOT_GEOM_NAMES)
  return cfg


# -- Eval overrides --------------------------------------------------------


def _apply_eval_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.episode_length_s = int(1e9)
  cfg.observations["actor"].enable_corruption = False
  strip_variant_training_events_for_play(cfg)


# -- Public factories ------------------------------------------------------


def make_force_control_env_cfg() -> ManagerBasedRlEnvCfg:
  """Force control env with joint position control."""
  return _build_force_control_env_cfg(
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=G1_ACTION_SCALE,
        use_default_offset=True,
      ),
      "force_field": ForceFieldActionCfg(),
    },
  )


def make_force_control_eval_env_cfg() -> ManagerBasedRlEnvCfg:
  """Force control eval env (no DR, no corruption)."""
  cfg = make_force_control_env_cfg()
  _apply_eval_overrides(cfg)
  return cfg


def make_force_control_uan_env_cfg(
  uan_checkpoint_path: str | None = None,
) -> ManagerBasedRlEnvCfg:
  """Force control env with the shared UAN action / DR preset."""
  cfg = make_force_control_env_cfg()
  swap_in_uan_action(
    cfg,
    checkpoint_path=uan_checkpoint_path,
  )
  apply_uan_training_dr(cfg, foot_geom_names=G1_FOOT_GEOM_NAMES)
  return cfg


def make_force_control_uan_eval_env_cfg(
  uan_checkpoint_path: str | None = None,
) -> ManagerBasedRlEnvCfg:
  """Force control eval env with UAN-augmented action."""
  cfg = make_force_control_uan_env_cfg(uan_checkpoint_path)
  _apply_eval_overrides(cfg)
  return cfg
