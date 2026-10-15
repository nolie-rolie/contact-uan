"""Shared helpers for G1 baseline and UAN task variants."""

from __future__ import annotations

import os

from mjlab.asset_zoo.robots import (
  G1_TORQUE_SPEED_SATURATION_EFFORT,
  G1_TORQUE_SPEED_VELOCITY_LIMIT,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as vel_mdp

G1_FOOT_GEOM_NAMES = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)

_BASELINE_HEAVY_EVENT_NAMES = (
  "add_base_mass",
  "add_link_mass",
  "randomize_base_com",
  "physics_material",
  "randomize_joint_parameters",
  "randomize_joint_damping",
  "randomize_armature",
  "randomize_friction",
  "randomize_encoder_bias",
  "randomize_pd_gains",
  "randomize_actuator_delay",
)

_PLAY_RANDOMIZATION_EVENT_NAMES = (
  "push_robot",
  "foot_friction",
  "encoder_bias",
  "base_com",
  *_BASELINE_HEAVY_EVENT_NAMES,
)


def has_uan_checkpoint_env() -> bool:
  """Return True when any supported UAN checkpoint env var is set."""
  return bool(os.environ.get("UAN_CHECKPOINT_PATH") or os.environ.get("UAN_JIT_PATH"))


def resolve_uan_checkpoint_path(explicit: str | None = None) -> str:
  """Resolve the UAN checkpoint path.

  Precedence:
  1. Explicit argument
  2. UAN_CHECKPOINT_PATH
  3. UAN_JIT_PATH
  """
  if explicit:
    return explicit

  for env_name in ("UAN_CHECKPOINT_PATH", "UAN_JIT_PATH"):
    value = os.environ.get(env_name, "")
    if value:
      return value

  raise ValueError(
    "No UAN checkpoint provided. Pass checkpoint_path or set "
    "UAN_CHECKPOINT_PATH / UAN_JIT_PATH."
  )


def _resolve_optional_uan_checkpoint_path(explicit: str | None = None) -> str | None:
  if explicit is not None:
    return resolve_uan_checkpoint_path(explicit)
  if has_uan_checkpoint_env():
    return resolve_uan_checkpoint_path()
  return None


def _pop_events(cfg: ManagerBasedRlEnvCfg, names: tuple[str, ...]) -> None:
  for name in names:
    cfg.events.pop(name, None)


def apply_velocity_baseline_dr(
  cfg: ManagerBasedRlEnvCfg,
  *,
  foot_geom_names: tuple[str, ...],
  base_body_names: tuple[str, ...] = ("torso_link",),
  keep_push_robot: bool = True,
) -> None:
  """Apply the default velocity-env DR recipe to a task config."""
  _pop_events(
    cfg,
    (
      *_BASELINE_HEAVY_EVENT_NAMES,
      "foot_friction",
      "encoder_bias",
      "base_com",
    ),
  )

  if keep_push_robot:
    cfg.events["push_robot"] = EventTermCfg(
      func=vel_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    )
  else:
    cfg.events.pop("push_robot", None)

  cfg.events["foot_friction"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
      "operation": "abs",
      "ranges": (0.3, 1.2),
      "shared_random": True,
    },
  )
  cfg.events["encoder_bias"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.encoder_bias,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "bias_range": (-0.015, 0.015),
    },
  )
  cfg.events["base_com"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.body_com_offset,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=base_body_names),
      "operation": "add",
      "ranges": {
        0: (-0.025, 0.025),
        1: (-0.025, 0.025),
        2: (-0.03, 0.03),
      },
    },
  )


def apply_uan_training_dr(
  cfg: ManagerBasedRlEnvCfg,
  *,
  foot_geom_names: tuple[str, ...],
  keep_push_robot: bool = True,
) -> None:
  """Apply the shared UAN training DR recipe."""
  apply_velocity_baseline_dr(
    cfg,
    foot_geom_names=foot_geom_names,
    keep_push_robot=keep_push_robot,
  )
  _pop_events(
    cfg,
    (
      "encoder_bias",
      "base_com",
      "randomize_armature",
      "randomize_friction",
      "randomize_encoder_bias",
      "randomize_pd_gains",
      "randomize_actuator_delay",
    ),
  )


def remove_robot_specific_training_dr(cfg: ManagerBasedRlEnvCfg) -> None:
  """Remove robot/model DR while preserving environment perturbations."""
  _pop_events(
    cfg,
    (
      "encoder_bias",
      "base_com",
      *_BASELINE_HEAVY_EVENT_NAMES,
    ),
  )


def swap_in_uan_action(
  cfg: ManagerBasedRlEnvCfg,
  *,
  checkpoint_path: str | None = None,
) -> None:
  """Replace the joint position action with the shared UAN action."""
  from mjlab.tasks.wtw.mdp.actions import UANAugmentedJointPositionActionCfg

  joint_pos_action = cfg.actions["joint_pos"]
  if not isinstance(joint_pos_action, JointPositionActionCfg):
    raise TypeError(
      "Expected cfg.actions['joint_pos'] to be JointPositionActionCfg, "
      f"got {type(joint_pos_action).__name__}."
    )

  cfg.actions["joint_pos"] = UANAugmentedJointPositionActionCfg(
    entity_name=joint_pos_action.entity_name,
    actuator_names=tuple(joint_pos_action.actuator_names),
    scale=joint_pos_action.scale,
    offset=joint_pos_action.offset,
    preserve_order=joint_pos_action.preserve_order,
    clip=joint_pos_action.clip,
    use_default_offset=joint_pos_action.use_default_offset,
    uan_checkpoint_path=_resolve_optional_uan_checkpoint_path(checkpoint_path),
  )


def swap_in_torque_speed_clipped_action(cfg: ManagerBasedRlEnvCfg) -> None:
  """Replace a joint-position action with the shared G1 clipped action."""
  joint_pos_action = cfg.actions["joint_pos"]
  if not isinstance(joint_pos_action, JointPositionActionCfg):
    raise TypeError(
      "Expected cfg.actions['joint_pos'] to be JointPositionActionCfg, "
      f"got {type(joint_pos_action).__name__}."
    )

  cfg.actions["joint_pos"] = vel_mdp.TorqueSpeedClippedJointPositionActionCfg(
    entity_name=joint_pos_action.entity_name,
    actuator_names=tuple(joint_pos_action.actuator_names),
    scale=joint_pos_action.scale,
    offset=joint_pos_action.offset,
    preserve_order=joint_pos_action.preserve_order,
    clip=joint_pos_action.clip,
    use_default_offset=joint_pos_action.use_default_offset,
    saturation_effort=G1_TORQUE_SPEED_SATURATION_EFFORT,
    velocity_limit=G1_TORQUE_SPEED_VELOCITY_LIMIT,
  )


def swap_in_uan_torque_speed_clipped_action(
  cfg: ManagerBasedRlEnvCfg,
  *,
  checkpoint_path: str | None = None,
) -> None:
  """Replace the joint position action with the clipped UAN action."""
  joint_pos_action = cfg.actions["joint_pos"]
  if not isinstance(joint_pos_action, vel_mdp.TorqueSpeedClippedJointPositionActionCfg):
    raise TypeError(
      "Expected cfg.actions['joint_pos'] to be "
      "TorqueSpeedClippedJointPositionActionCfg, "
      f"got {type(joint_pos_action).__name__}."
    )

  cfg.actions["joint_pos"] = (
    vel_mdp.UANAugmentedTorqueSpeedClippedJointPositionActionCfg(
      entity_name=joint_pos_action.entity_name,
      actuator_names=tuple(joint_pos_action.actuator_names),
      scale=joint_pos_action.scale,
      offset=joint_pos_action.offset,
      preserve_order=joint_pos_action.preserve_order,
      clip=joint_pos_action.clip,
      use_default_offset=joint_pos_action.use_default_offset,
      saturation_effort=joint_pos_action.saturation_effort,
      velocity_limit=joint_pos_action.velocity_limit,
      uan_checkpoint_path=_resolve_optional_uan_checkpoint_path(checkpoint_path),
    )
  )


def strip_variant_training_events_for_play(cfg: ManagerBasedRlEnvCfg) -> None:
  """Remove baseline/UAN training randomization from play configs."""
  _pop_events(cfg, _PLAY_RANDOMIZATION_EVENT_NAMES)
