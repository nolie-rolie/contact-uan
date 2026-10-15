"""Manager-based UAN replay-rollout environment config."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.uan import mdp
from mjlab.tasks.uan.mdp.utils import (
  resolve_uan_obs_block_names,
  rollout_uan_obs_block_names,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig


@dataclass(kw_only=True)
class UANRolloutEnvCfg(ManagerBasedRlEnvCfg):
  """Named config type for UAN replay-rollout tasks."""

  def apply_actor_observation_layout(self) -> None:
    robot_joint_cfg = SceneEntityCfg("robot", joint_names=(".*",))
    actor_terms = {
      "target_pos_error": ObservationTermCfg(
        func=mdp.replay_target_pos_error,
        params={
          "command_name": "replay",
          "action_name": "residual_target",
        },
      ),
      "joint_vel": ObservationTermCfg(
        func=envs_mdp.joint_vel_rel,
        params={"asset_cfg": robot_joint_cfg},
        scale=0.05,
      ),
    }
    self.observations["actor"] = ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=3,
      flatten_history_dim=True,
    )

  def actor_obs_block_names(self) -> tuple[str, ...]:
    return rollout_uan_obs_block_names()


def sync_uan_rollout_actor_layout(env_cfg: ManagerBasedRlEnvCfg) -> None:
  apply_layout = getattr(env_cfg, "apply_actor_observation_layout", None)
  if callable(apply_layout):
    apply_layout()


def sync_uan_rollout_config_from_checkpoint_metadata(
  *,
  env_cfg: ManagerBasedRlEnvCfg,
  agent_cfg: Any,
  checkpoint_metadata: Mapping[str, Any],
) -> tuple[str, ...]:
  """Align rollout env/actor config with a saved UAN checkpoint."""
  is_uan_checkpoint = any(
    key in checkpoint_metadata
    for key in (
      "num_actuators",
      "joint_to_actuator_map",
      "uan_history_length",
      "uan_obs_block_names",
    )
  )
  if not is_uan_checkpoint:
    sync_uan_rollout_actor_layout(env_cfg)
    actor_cfg = getattr(agent_cfg, "actor", None)
    actor_block_names = cast(
      Callable[[], tuple[str, ...]] | None,
      getattr(env_cfg, "actor_obs_block_names", None),
    )
    if (
      actor_cfg is not None
      and callable(actor_block_names)
      and hasattr(actor_cfg, "num_blocks")
    ):
      actor_cfg.num_blocks = len(actor_block_names())
    return tuple(actor_block_names()) if callable(actor_block_names) else ()

  block_names = resolve_uan_obs_block_names(checkpoint_metadata)
  sync_uan_rollout_actor_layout(env_cfg)

  actor_cfg = getattr(agent_cfg, "actor", None)
  if actor_cfg is not None:
    if hasattr(actor_cfg, "num_blocks"):
      actor_cfg.num_blocks = len(block_names)
    if hasattr(actor_cfg, "num_actuators"):
      raw_num_actuators = checkpoint_metadata.get("num_actuators")
      if raw_num_actuators is not None:
        actor_cfg.num_actuators = int(raw_num_actuators)
    if hasattr(actor_cfg, "joint_to_actuator_map"):
      raw_joint_to_actuator_map = checkpoint_metadata.get("joint_to_actuator_map")
      actor_cfg.joint_to_actuator_map = (
        [int(idx) for idx in raw_joint_to_actuator_map]
        if raw_joint_to_actuator_map is not None
        else None
      )

  return block_names


def make_uan_rollout_env_cfg(
  *,
  num_envs: int = 4096,
) -> UANRolloutEnvCfg:
  rollout_horizon = 24
  decimation = 4
  physics_dt = 0.005
  robot_joint_cfg = SceneEntityCfg("robot", joint_names=(".*",))
  foot_body_cfg = SceneEntityCfg(
    "robot",
    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
  )

  scene = SceneCfg(
    terrain=TerrainEntityCfg(
      terrain_type="plane",
      num_envs=num_envs,
      env_spacing=2.0,
    ),
    entities={"robot": get_g1_robot_cfg()},
    sensors=(),
    num_envs=num_envs,
    extent=2.0,
  )

  cfg = UANRolloutEnvCfg(
    decimation=decimation,
    scene=scene,
    observations={
      "actor": ObservationGroupCfg(terms={}),
      "critic": ObservationGroupCfg(
        terms={
          "target_pos_error": ObservationTermCfg(
            func=mdp.replay_target_pos_error,
            params={
              "command_name": "replay",
              "action_name": "residual_target",
            },
          ),
          "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={"asset_cfg": robot_joint_cfg},
            scale=0.05,
          ),
          "joint_pos_rel": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={"asset_cfg": robot_joint_cfg},
          ),
          "recorded_target": ObservationTermCfg(
            func=mdp.recorded_target,
            params={"command_name": "replay"},
          ),
          "recorded_joint_vel": ObservationTermCfg(
            func=mdp.recorded_joint_vel,
            params={"command_name": "replay"},
            scale=0.05,
          ),
          "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
          ),
          "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.base_lin_vel,
          ),
          "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            scale=0.25,
          ),
          "root_height": ObservationTermCfg(
            func=mdp.root_height,
          ),
          "foot_pos_b": ObservationTermCfg(
            func=mdp.body_pos_b,
            params={"asset_cfg": foot_body_cfg},
          ),
        },
        concatenate_terms=True,
        enable_corruption=False,
        history_length=3,
        flatten_history_dim=True,
      ),
    },
    actions={
      "residual_target": mdp.ReplayResidualJointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        command_name="replay",
        residual_scale=1.0,
        residual_clip=1.0,
      )
    },
    events={
      "reset_to_recorded_trajectory_state": EventTermCfg(
        func=mdp.reset_to_recorded_trajectory_state,
        mode="reset",
        params={
          "command_name": "replay",
          "action_name": "residual_target",
        },
      )
    },
    rewards={
      "joint_position_mse": RewardTermCfg(
        func=mdp.joint_position_mse,
        weight=-1000.0,
        params={
          "command_name": "replay",
          "action_name": "residual_target",
          "mse_clip": None,
        },
      ),
      "joint_velocity_mse": RewardTermCfg(
        func=mdp.joint_velocity_mse,
        weight=-0.1,
        params={
          "command_name": "replay",
          "action_name": "residual_target",
          "mse_clip": None,
        },
      ),
      "residual_l2": RewardTermCfg(
        func=mdp.residual_l2,
        weight=0.0,
        params={"action_name": "residual_target"},
      ),
      "residual_rate_l2": RewardTermCfg(
        func=mdp.residual_rate_l2,
        weight=-0.01,
        params={"action_name": "residual_target"},
      ),
    },
    terminations={
      "time_out": TerminationTermCfg(
        func=envs_mdp.time_out,
        time_out=True,
      )
    },
    commands={
      "replay": mdp.ReplayWindowCommandCfg(
        data_dir="",
        extension_dim=15,
        history_horizon=3,
        rollout_horizon=rollout_horizon,
        require_contact_at_reset=True,
        contact_threshold=0.8,
      )
    },
    curriculum={},
    metrics={},
    sim=SimulationCfg(
      nconmax=80,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=physics_dt,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=50,
      ),
      contact_sensor_maxmatch=64,
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    episode_length_s=rollout_horizon * decimation * physics_dt,
    is_finite_horizon=False,
    scale_rewards_by_dt=False,
  )
  cfg.apply_actor_observation_layout()
  return cfg
