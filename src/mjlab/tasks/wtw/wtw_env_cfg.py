"""WTW environment config factory.

Provides factory functions for creating WTW walking environment configs:

- make_wtw_env_cfg(): WTW training config
- make_wtw_eval_env_cfg(): WTW play variant

All configs use WTW-compatible 297D observations (history=3, term-major
flattening), 20s episodes during training, decimation=4, and a custom
velocity-aligned reward stack with retained WTW gait shaping.
"""

import math

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
from mjlab.tasks.utils.g1_variants import (
  G1_FOOT_GEOM_NAMES,
  strip_variant_training_events_for_play,
)
from mjlab.tasks.utils.terrain_profiles import BUMPY_TERRAINS_CFG
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.wtw import mdp as wtw_mdp
from mjlab.tasks.wtw.mdp.commands import (
  FootSwingHeightCommandCfg,
  GaitFrequencyCommandCfg,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# Constants matching Isaac Lab WTW config
GAIT_FREQUENCY_RANGE = (1.0, 2.5)
SWING_HEIGHT_RANGE = (0.05, 0.1)
FOOT_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_RADII = [0.05, 0.05]
STANCE_WIDTH = 0.23701290
MOTION_EPSILON = 1e-3
POSE_STD_STANDING = {".*": 0.05}
POSE_STD_WALKING = {
  r".*hip_pitch.*": 0.3,
  r".*hip_roll.*": 0.15,
  r".*hip_yaw.*": 0.15,
  r".*knee.*": 0.35,
  r".*ankle_pitch.*": 0.25,
  r".*ankle_roll.*": 0.1,
  r".*waist_yaw.*": 0.2,
  r".*waist_roll.*": 0.08,
  r".*waist_pitch.*": 0.1,
  r".*shoulder_pitch.*": 0.15,
  r".*shoulder_roll.*": 0.15,
  r".*shoulder_yaw.*": 0.1,
  r".*elbow.*": 0.15,
  r".*wrist.*": 0.3,
}
POSE_STD_RUNNING = {
  r".*hip_pitch.*": 0.5,
  r".*hip_roll.*": 0.2,
  r".*hip_yaw.*": 0.2,
  r".*knee.*": 0.6,
  r".*ankle_pitch.*": 0.35,
  r".*ankle_roll.*": 0.15,
  r".*waist_yaw.*": 0.3,
  r".*waist_roll.*": 0.08,
  r".*waist_pitch.*": 0.2,
  r".*shoulder_pitch.*": 0.5,
  r".*shoulder_roll.*": 0.2,
  r".*shoulder_yaw.*": 0.15,
  r".*elbow.*": 0.35,
  r".*wrist.*": 0.3,
}
PUSH_VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.4, 0.4),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}
DEFAULT_WTW_NUM_ENVS = 4096


def _add_wtw_self_collision_reward(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add WTW self-collision sensing and reward without changing sim buffers."""
  sensor_names = {sensor.name for sensor in (cfg.scene.sensors or ())}
  if "self_collision" not in sensor_names:
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
      ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
      ),
    )

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=vel_mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": "self_collision", "force_threshold": 10.0},
  )


def _build_wtw_env_cfg(
  actions: dict[str, ActionTermCfg],
  *,
  enable_self_collision: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build a WTW env config with the given action config.

  Everything except actions is shared across all WTW variants.
  """

  # Scene
  feet_ground_cfg = ContactSensorCfg(
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
  sensors: list[ContactSensorCfg] = [feet_ground_cfg]
  if enable_self_collision:
    # Whole-body self-collision tracking needs much larger contact buffers and has a
    # noticeable throughput cost, so keep it opt-in for WTW training.
    sensors.append(
      ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
      )
    )

  scene = SceneCfg(
    terrain=TerrainEntityCfg(
      terrain_type="generator",
      terrain_generator=BUMPY_TERRAINS_CFG,
    ),
    entities={"robot": get_g1_robot_cfg()},
    sensors=tuple(sensors),
    num_envs=DEFAULT_WTW_NUM_ENVS,
    extent=2.0,
  )

  # Observations -- WTW 297D with history=3, term-major flattening
  #
  # Term order:
  #   base_ang_vel(3), projected_gravity(3), joint_pos_rel(29),
  #   joint_vel_rel(29), last_action(29), velocity_commands(3),
  #   gait_frequency(1), foot_swing_height(1), gait_phase(1)
  # Total per step: 99. With history_length=3: 297.
  # All joint data is in native MuJoCo order.

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
    "last_action": ObservationTermCfg(
      func=envs_mdp.last_action,
    ),
    "velocity_commands": ObservationTermCfg(
      func=vel_mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "gait_frequency": ObservationTermCfg(
      func=vel_mdp.generated_commands,
      params={"command_name": "gait_frequency"},
    ),
    "foot_swing_height": ObservationTermCfg(
      func=vel_mdp.generated_commands,
      params={"command_name": "foot_swing_height"},
    ),
    "gait_phase": ObservationTermCfg(
      func=wtw_mdp.gait_phase,
      params={
        "command_name": "twist",
        "freq_range": GAIT_FREQUENCY_RANGE,
      },
    ),
  }

  critic_terms = dict(actor_terms)
  critic_terms["base_lin_vel"] = ObservationTermCfg(
    func=envs_mdp.base_lin_vel,
  )
  critic_terms["root_height"] = ObservationTermCfg(
    func=wtw_mdp.root_height,
  )
  critic_terms["foot_contact"] = ObservationTermCfg(
    func=vel_mdp.foot_contact,
    params={"sensor_name": "feet_ground_contact"},
  )
  critic_terms["foot_contact_forces"] = ObservationTermCfg(
    func=vel_mdp.foot_contact_force_magnitude,
    params={"sensor_name": "feet_ground_contact"},
    scale=1.0 / 600.0,
  )
  critic_terms["foot_velocities"] = ObservationTermCfg(
    func=vel_mdp.foot_velocities,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=FOOT_NAMES,
      ),
    },
    scale=0.05,
  )

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=3,
      flatten_history_dim=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=3,
      flatten_history_dim=True,
    ),
  }

  # Commands
  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(5.0, 10.0),
      rel_standing_envs=0.2,
      rel_heading_envs=0.0,
      heading_command=False,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-1.0, 1.0),
      ),
    ),
    "gait_frequency": GaitFrequencyCommandCfg(
      resampling_time_range=(5.0, 10.0),
      frequency_range=(-1.0, 1.0),
    ),
    "foot_swing_height": FootSwingHeightCommandCfg(
      resampling_time_range=(5.0, 10.0),
      swing_height_range=(-1.0, 1.0),
    ),
  }

  # Events
  events = {
    "reset_base": EventTermCfg(
      func=vel_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=vel_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=vel_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": PUSH_VELOCITY_RANGE},
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=G1_FOOT_GEOM_NAMES),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
  }

  # Rewards
  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=vel_mdp.track_linear_velocity,
      weight=2.0,
      params={
        "command_name": "twist",
        "std": math.sqrt(0.25),
      },
    ),
    "track_angular_velocity": RewardTermCfg(
      func=vel_mdp.track_angular_velocity_z,
      weight=2.0,
      params={
        "command_name": "twist",
        "std": math.sqrt(0.25),
      },
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
      func=vel_mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": POSE_STD_STANDING,
        "std_walking": POSE_STD_WALKING,
        "std_running": POSE_STD_RUNNING,
        "walking_threshold": MOTION_EPSILON,
        "running_threshold": 1.5,
      },
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
    "gait": RewardTermCfg(
      func=wtw_mdp.biped_gait_phase,
      weight=-0.5,
      params={
        "command_name": "twist",
        "freq_range": GAIT_FREQUENCY_RANGE,
        "overlap": 0.25,
        "sensor_name": "feet_ground_contact",
      },
    ),
    "swing_height": RewardTermCfg(
      func=wtw_mdp.biped_swing_height,
      weight=-20.0,
      params={
        "command_name": "twist",
        "freq_range": GAIT_FREQUENCY_RANGE,
        "overlap": 0.25,
        "swing_height_range": SWING_HEIGHT_RANGE,
        "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_NAMES),
        "foot_radii": FOOT_RADII,
      },
    ),
    "raibert_heuristic": RewardTermCfg(
      func=wtw_mdp.biped_raibert_heuristic,
      weight=-5.0,
      params={
        "command_name": "twist",
        "freq_range": GAIT_FREQUENCY_RANGE,
        "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_NAMES),
        "stance_width": STANCE_WIDTH,
      },
    ),
    "action_rate_l2": RewardTermCfg(func=vel_mdp.action_rate_l2, weight=-0.1),
    "dof_pos_limits": RewardTermCfg(func=vel_mdp.joint_pos_limits, weight=-1.0),
    "soft_landing": RewardTermCfg(
      func=vel_mdp.soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": MOTION_EPSILON,
      },
    ),
  }
  if enable_self_collision:
    rewards["self_collisions"] = RewardTermCfg(
      func=vel_mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    )

  if enable_self_collision:
    sim = SimulationCfg(
      nconmax=70,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=500,
      ),
      contact_sensor_maxmatch=500,
    )
  else:
    sim = SimulationCfg(
      nconmax=None,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=50,
      ),
      contact_sensor_maxmatch=64,
    )

  # Terminations
  terminations = {
    "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
    "bad_orientation": TerminationTermCfg(
      func=vel_mdp.bad_orientation,
      params={"limit_angle": math.pi / 4},
    ),
  }

  return ManagerBasedRlEnvCfg(
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
    sim=sim,
    decimation=4,
    episode_length_s=20.0,
  )


# -- Eval overrides --------------------------------------------------------


def _apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  """Apply custom-velocity-style play overrides."""
  cfg.events.pop("push_robot", None)
  cfg.events["randomize_terrain"] = EventTermCfg(
    func=envs_mdp.randomize_terrain,
    mode="reset",
    params={},
  )
  cfg.episode_length_s = int(1e9)
  cfg.observations["actor"].enable_corruption = False

  assert cfg.scene.terrain is not None
  if cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.scene.terrain.terrain_generator.num_cols = 5
    cfg.scene.terrain.terrain_generator.num_rows = 5
    cfg.scene.terrain.terrain_generator.border_width = 10.0


# -- Public factories -----------------------------------------------------


def make_wtw_env_cfg(
  *,
  enable_self_collision: bool = True,
) -> ManagerBasedRlEnvCfg:
  """WTW env with vanilla joint position control."""
  cfg = _build_wtw_env_cfg(
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=G1_ACTION_SCALE,
        use_default_offset=True,
      ),
    },
    enable_self_collision=False,
  )
  if enable_self_collision:
    _add_wtw_self_collision_reward(cfg)
  return cfg


def make_wtw_eval_env_cfg(
  *,
  enable_self_collision: bool = True,
) -> ManagerBasedRlEnvCfg:
  """WTW play env without UAN."""
  cfg = make_wtw_env_cfg(enable_self_collision=enable_self_collision)
  _apply_play_overrides(cfg)
  strip_variant_training_events_for_play(cfg)
  return cfg
