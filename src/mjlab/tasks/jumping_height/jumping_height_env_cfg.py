"""Full motion tracking plus a one-shot ballistic-rise JumpingHeight objective."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.jumping_height import mdp

_TRACKING_REWARD_NAMES = (
  "motion_global_root_pos",
  "motion_global_root_ori",
  "motion_body_pos",
  "motion_body_ori",
  "motion_body_lin_vel",
  "motion_body_ang_vel",
)
_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_AIRBORNE_HEIGHT_THRESHOLD = 0.06
_CONTACT_CONFIRMATION_STEPS = 2
_ROOT_BASE_HEIGHT = 0.79
_MINIMUM_ROOT_HEIGHT = 0.30
_MAXIMUM_ROOT_TILT = math.radians(80.0)
_ROLL_BODY_NAMES = ("pelvis", "torso_link")


def apply_jumping_height_objective(cfg: ManagerBasedRlEnvCfg) -> None:
  """Apply full tracking plus one ballistic COM-rise bonus per jump."""
  tracking_rewards = {name: cfg.rewards[name] for name in _TRACKING_REWARD_NAMES}

  cfg.rewards = {
    **tracking_rewards,
    "jump_com_ballistic_rise": RewardTermCfg(
      func=mdp.jump_com_ballistic_rise,
      weight=5.0,
      params={
        "command_name": "motion",
        "sensor_name": "feet_ground_contact",
        "foot_body_names": _FOOT_BODY_NAMES,
        "airborne_height_threshold": _AIRBORNE_HEIGHT_THRESHOLD,
        "contact_confirmation_steps": _CONTACT_CONFIRMATION_STEPS,
        "gravity": 9.81,
      },
    ),
    "action_rate_l2": cfg.rewards["action_rate_l2"],
    "joint_limit": cfg.rewards["joint_limit"],
    "self_collisions": cfg.rewards["self_collisions"],
  }

  cfg.metrics["last_jump_peak_root_height"] = MetricsTermCfg(
    func=mdp.last_jump_peak_root_height,
    params={
      "command_name": "motion",
      "foot_body_names": _FOOT_BODY_NAMES,
      "airborne_height_threshold": _AIRBORNE_HEIGHT_THRESHOLD,
      "base_height": _ROOT_BASE_HEIGHT,
    },
    reduce="last",
  )
  reward_metric_attributes = {
    "last_takeoff_com_z_velocity": "last_takeoff_com_z_velocity",
    "last_jump_ballistic_rise": "last_ballistic_rise",
    "jump_bonus_event_count": "bonus_event_count",
    "jump_bonus_opportunity_consumed": "opportunity_consumed",
  }
  for metric_name, attribute in reward_metric_attributes.items():
    cfg.metrics[metric_name] = MetricsTermCfg(
      func=mdp.reward_term_state,
      params={
        "reward_name": "jump_com_ballistic_rise",
        "attribute": attribute,
      },
      reduce="last",
    )
  cfg.metrics["pelvis_roll_l2"] = MetricsTermCfg(
    func=mdp.body_roll_l2,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
    },
  )
  cfg.metrics["torso_roll_l2"] = MetricsTermCfg(
    func=mdp.body_roll_l2,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )

  for name in ("anchor_pos", "anchor_ori", "ee_body_pos"):
    cfg.terminations.pop(name, None)

  cfg.terminations["low_height"] = TerminationTermCfg(
    func=envs_mdp.root_height_below_minimum,
    params={"minimum_height": _MINIMUM_ROOT_HEIGHT},
  )
  cfg.terminations["bad_orientation"] = TerminationTermCfg(
    func=envs_mdp.bad_orientation,
    params={"limit_angle": _MAXIMUM_ROOT_TILT},
  )
