"""Shared base controller for all per-task deploy scripts.

Provides:
  - YAML config parsing (``parse_hardware_cfg``)
  - Policy loading (JIT)
  - Obs-history buffer management
  - Action application with torque-based clipping
  - Main control loop (``run``)
  - Pose initialisation

Per-task scripts subclass ``BaseController`` and implement ``build_sensors()``.
Deploy scripts use MuJoCo joint order for policy observations and actions.
"""

import os
import pickle
import re
import time
import types
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml

_DEPLOY_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEPLOY_DIR.parent


class _RelaxedYamlLoader(yaml.SafeLoader):
  """YAML loader that ignores Python-specific object tags.

  MJLab training configs are dumped with tags like ``!!python/name:...`` and
  ``!!python/object/apply:...``. Deploy only needs the primitive mapping data,
  so we decode those tags into plain Python scalars/lists/dicts instead of
  importing the original modules.
  """


def _construct_relaxed_python(loader, suffix, node):
  if isinstance(node, yaml.MappingNode):
    return loader.construct_mapping(node, deep=True)
  if isinstance(node, yaml.SequenceNode):
    values = loader.construct_sequence(node, deep=True)
    if suffix.startswith("object/apply:") and len(values) == 1:
      return values[0]
    return values
  if suffix.startswith("name:"):
    return suffix[len("name:") :]
  return loader.construct_scalar(node)


_RelaxedYamlLoader.add_multi_constructor(
  "tag:yaml.org,2002:python/",
  _construct_relaxed_python,
)


def _load_relaxed_yaml(path: str) -> dict:
  with open(path, "r") as f:
    return yaml.load(f, Loader=_RelaxedYamlLoader)


def _load_relaxed_checkpoint(path: str) -> dict:
  """Load deploy metadata from training checkpoints without training imports."""

  class _Dummy:
    def __init__(self, *args, **kwargs):
      pass

  class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
      try:
        return super().find_class(module, name)
      except (ModuleNotFoundError, ImportError, AttributeError):
        return _Dummy

  stub_mod = types.ModuleType("_pickle_stub")
  stub_mod.__dict__["Unpickler"] = _StubUnpickler
  stub_mod.__dict__["load"] = pickle.load
  stub_mod.__dict__["dump"] = pickle.dump

  try:
    return torch.load(
      path,
      map_location="cpu",
      weights_only=False,
      pickle_module=stub_mod,
    )
  except TypeError:
    return torch.load(path, map_location="cpu", pickle_module=stub_mod)


def resolve_action_scale_array(action_scale, joint_names: list[str]) -> np.ndarray:
  """Resolve scalar / regex-map action scales into a per-joint array."""
  if isinstance(action_scale, dict):
    scale_array = np.ones(len(joint_names), dtype=np.float32)
    for idx, joint_name in enumerate(joint_names):
      for pattern, scale_val in action_scale.items():
        if re.match(pattern, joint_name):
          scale_array[idx] = float(scale_val)
          break
    return scale_array

  if np.isscalar(action_scale):
    return np.full(len(joint_names), float(cast(Any, action_scale)), dtype=np.float32)

  scale_array = np.asarray(action_scale, dtype=np.float32)
  if scale_array.shape != (len(joint_names),):
    raise ValueError(
      f"Expected action_scale to have shape ({len(joint_names)},), got {scale_array.shape}"
    )
  return scale_array


def _extract_observation_terms(env_cfg: dict) -> tuple[dict, dict]:
  observations = env_cfg.get("observations", {})
  obs_group = observations.get("actor") or observations.get("policy") or {}
  return obs_group, obs_group.get("terms", obs_group)


def _extract_robot_scene(env_cfg: dict) -> dict:
  scene_cfg = env_cfg.get("scene", {})
  robot_scene = scene_cfg.get("robot")
  if robot_scene is None:
    robot_scene = scene_cfg.get("entities", {}).get("robot", {})
  return robot_scene


def _extract_actuator_groups(robot_scene: dict):
  actuators = robot_scene.get("actuators")
  if actuators is None:
    actuators = robot_scene.get("articulation", {}).get("actuators", {})
  if isinstance(actuators, dict):
    return list(actuators.values())
  return list(actuators or [])


def _raw_scale(obs_terms: dict, term_name: str, default: float = 1.0) -> float:
  term = obs_terms.get(term_name)
  if isinstance(term, dict):
    scale = term.get("scale")
    return float(scale) if scale is not None else float(default)
  return float(default)


def _expand_regex_map(
  pattern_to_value: dict,
  joint_names: list[str],
  *,
  fill_value: float | None = None,
  base_values: np.ndarray | None = None,
) -> np.ndarray:
  if base_values is not None:
    values = np.asarray(base_values, dtype=np.float64).copy()
  else:
    if fill_value is None:
      values = np.full(len(joint_names), np.nan, dtype=np.float64)
    else:
      values = np.full(len(joint_names), float(fill_value), dtype=np.float64)

  for pattern, raw_value in pattern_to_value.items():
    value = float(raw_value)
    for idx, joint_name in enumerate(joint_names):
      if re.match(pattern, joint_name):
        values[idx] = value
  return values


def _resolve_checkpoint_path(policy_path: str) -> str | None:
  if policy_path.endswith(".pt") and os.path.exists(policy_path):
    return policy_path

  if policy_path.endswith(".jit"):
    pt_path = policy_path[:-4] + ".pt"
    if os.path.exists(pt_path):
      return pt_path

  return None


def resolve_policy_path(policy_name: str) -> str:
  """Resolve deploy policy paths from absolute, repo, or deploy-model roots."""
  policy_path = Path(policy_name).expanduser()
  if policy_path.is_absolute():
    return str(policy_path)

  candidates = [
    Path.cwd() / policy_path,
    _REPO_ROOT / policy_path,
    _DEPLOY_DIR / policy_path,
    _DEPLOY_DIR / "models" / policy_path,
  ]
  unique_candidates = []
  seen = set()
  for candidate in candidates:
    resolved = candidate.resolve()
    if resolved in seen:
      continue
    seen.add(resolved)
    unique_candidates.append(resolved)

  for candidate in unique_candidates:
    if candidate.exists():
      return str(candidate)

  searched = "\n- ".join(str(candidate) for candidate in unique_candidates)
  raise FileNotFoundError(
    f"Policy file not found for {policy_name!r}. Searched:\n- {searched}"
  )


def _load_policy_common_step_counter(policy_path: str) -> int | None:
  checkpoint_path = _resolve_checkpoint_path(policy_path)
  if checkpoint_path is None:
    return None

  ckpt = _load_relaxed_checkpoint(checkpoint_path)
  step = ckpt.get("infos", {}).get("env_state", {}).get("common_step_counter")
  return int(step) if step is not None else None


def compute_torque_speed_limits(
  joint_vel: np.ndarray,
  saturation_effort: np.ndarray,
  velocity_limit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Return asymmetric PD torque bounds from the torque-speed envelope.

  Motoring torque decreases linearly to zero at the no-load speed, while
  braking torque retains the stall limit. Beyond that speed the motoring
  bound remains zero, so reverse torque is permitted but never required.
  """
  saturation_effort = np.asarray(saturation_effort, dtype=np.float64)
  velocity_limit = np.asarray(velocity_limit, dtype=np.float64)
  joint_vel = np.asarray(joint_vel, dtype=np.float64)

  if np.any(saturation_effort <= 0.0):
    raise ValueError("saturation_effort must be positive for torque-speed clipping")
  if np.any(velocity_limit <= 0.0):
    raise ValueError("velocity_limit must be positive for torque-speed clipping")

  velocity_ratio = joint_vel / velocity_limit
  lower = -saturation_effort * (1.0 + np.clip(velocity_ratio, -1.0, 0.0))
  upper = saturation_effort * (1.0 - np.clip(velocity_ratio, 0.0, 1.0))
  return lower, upper


def _resolve_twist_command_ranges(
  env_cfg: dict,
  policy_common_step_counter: int | None,
) -> tuple[dict[str, list[float]], bool]:
  commands_cfg = env_cfg.get("commands", {})
  twist_cfg = commands_cfg.get("twist", {})
  ranges = twist_cfg.get("ranges", {})
  resolved_ranges = {
    "lin_vel_x": [float(v) for v in ranges.get("lin_vel_x", (-1.0, 1.0))],
    "lin_vel_y": [float(v) for v in ranges.get("lin_vel_y", (-1.0, 1.0))],
    "ang_vel_yaw": [float(v) for v in ranges.get("ang_vel_z", (-0.5, 0.5))],
  }

  curriculum_cfg = env_cfg.get("curriculum", {})
  velocity_stages = None
  for term_cfg in curriculum_cfg.values():
    if not isinstance(term_cfg, dict):
      continue
    params = term_cfg.get("params", {})
    if params.get("command_name") != "twist":
      continue
    if params.get("velocity_stages") is not None:
      velocity_stages = params["velocity_stages"]
      break

  if not velocity_stages:
    return resolved_ranges, False

  if policy_common_step_counter is None:
    raise ValueError(
      "env.yaml defines a twist command curriculum, but no checkpoint "
      "common_step_counter was available to resolve the active command range"
    )

  stage_key_map = {
    "lin_vel_x": "lin_vel_x",
    "lin_vel_y": "lin_vel_y",
    "ang_vel_z": "ang_vel_yaw",
  }
  for stage in velocity_stages:
    if policy_common_step_counter < int(stage.get("step", 0)):
      continue
    for stage_key, range_key in stage_key_map.items():
      stage_range = stage.get(stage_key)
      if stage_range is not None:
        resolved_ranges[range_key] = [float(v) for v in stage_range]

  return resolved_ranges, True


def validate_runtime_config(
  env_cfg_raw: dict,
  cfg: dict,
  robot_configuration,
  policy_joint_order: str,
  policy_common_step_counter: int | None = None,
) -> list[str]:
  """Compare resolved deploy values against the raw env.yaml contents."""
  errors: list[str] = []
  obs_group, obs_terms = _extract_observation_terms(env_cfg_raw)
  robot_scene = _extract_robot_scene(env_cfg_raw)
  isaaclab_joint_names = robot_configuration.joint_names("isaaclab")
  policy_joint_names = robot_configuration.joint_names(policy_joint_order)

  raw_expected_history_length = obs_group.get("history_length", 3)
  expected_history_length = (
    1 if raw_expected_history_length is None else int(raw_expected_history_length)
  )
  if cfg["history_length"] != expected_history_length:
    errors.append(
      f"history_length mismatch: got {cfg['history_length']}, expected {expected_history_length}"
    )

  sim_cfg = env_cfg_raw.get("sim", {})
  expected_control_dt = float(sim_cfg.get("dt", 0.005)) * int(
    env_cfg_raw.get("decimation", 4)
  )
  if not np.isclose(cfg["control_dt"], expected_control_dt):
    errors.append(
      f"control_dt mismatch: got {cfg['control_dt']}, expected {expected_control_dt}"
    )

  scale_checks = {
    "ang_vel_scale": _raw_scale(obs_terms, "base_ang_vel", default=1.0),
    "dof_vel_scale": _raw_scale(
      obs_terms,
      "joint_vel_rel",
      default=_raw_scale(obs_terms, "joint_vel", default=1.0),
    ),
    "base_lin_vel_scale": _raw_scale(obs_terms, "base_lin_vel", default=1.0),
  }
  if "force_command" in obs_terms:
    scale_checks["force_scale"] = _raw_scale(obs_terms, "force_command", default=1.0)

  for key, expected_value in scale_checks.items():
    actual_value = cfg.get(key, 1.0)
    if not np.isclose(actual_value, expected_value):
      errors.append(f"{key} mismatch: got {actual_value}, expected {expected_value}")

  expected_gait_phase_freq_range = (
    obs_terms.get("gait_phase", {}).get("params", {}).get("freq_range")
  )
  if expected_gait_phase_freq_range is not None:
    actual_freq_range = tuple(cfg.get("gait_phase_freq_range", ()))
    expected_freq_range = tuple(float(v) for v in expected_gait_phase_freq_range)
    if actual_freq_range != expected_freq_range:
      errors.append(
        f"gait_phase_freq_range mismatch: got {actual_freq_range}, expected {expected_freq_range}"
      )

  expected_gait_cmd_range = (
    env_cfg_raw.get("commands", {}).get("gait_frequency", {}).get("frequency_range")
  )
  if expected_gait_cmd_range is not None:
    actual_cmd_range = tuple(cfg.get("gait_frequency_command_range", ()))
    expected_cmd_range = tuple(float(v) for v in expected_gait_cmd_range)
    if actual_cmd_range != expected_cmd_range:
      errors.append(
        f"gait_frequency_command_range mismatch: got {actual_cmd_range}, expected {expected_cmd_range}"
      )
    if expected_cmd_range != (-1.0, 1.0):
      errors.append(
        "deploy gait_frequency command sensor assumes a raw command range of [-1, 1]"
      )

  expected_swing_cmd_range = (
    env_cfg_raw.get("commands", {})
    .get("foot_swing_height", {})
    .get("swing_height_range")
  )
  if expected_swing_cmd_range is not None:
    actual_cmd_range = tuple(cfg.get("foot_swing_height_command_range", ()))
    expected_cmd_range = tuple(float(v) for v in expected_swing_cmd_range)
    if actual_cmd_range != expected_cmd_range:
      errors.append(
        f"foot_swing_height_command_range mismatch: got {actual_cmd_range}, expected {expected_cmd_range}"
      )
    if expected_cmd_range != (-1.0, 1.0):
      errors.append(
        "deploy foot_swing_height command sensor assumes a raw command range of [-1, 1]"
      )

  expected_offset_actions = bool(
    env_cfg_raw.get("actions", {}).get("joint_pos", {}).get("use_default_offset", False)
  )
  if cfg["offset_actions"] != expected_offset_actions:
    errors.append(
      f"offset_actions mismatch: got {cfg['offset_actions']}, expected {expected_offset_actions}"
    )

  commands_cfg = env_cfg_raw.get("commands", {})
  expected_command_ranges = {}
  if commands_cfg.get("upper_body_joints") is not None:
    proc_ranges = commands_cfg["upper_body_joints"].get("proc_ranges", {})
    expected_command_ranges = {
      "lin_vel_x": [
        float(proc_ranges.get("min_vel", -1.0)),
        float(proc_ranges.get("max_vel", 1.0)),
      ],
      "lin_vel_y": [0.0, 0.0],
      "ang_vel_yaw": [
        float(proc_ranges.get("min_yaw_vel", -0.5)),
        float(proc_ranges.get("max_yaw_vel", 0.5)),
      ],
    }
  elif commands_cfg.get("base_velocity") is not None:
    ranges = commands_cfg["base_velocity"].get("ranges", {})
    expected_command_ranges = {
      "lin_vel_x": list(ranges.get("lin_vel_x", (-1.0, 1.0))),
      "lin_vel_y": list(ranges.get("lin_vel_y", (-1.0, 1.0))),
      "ang_vel_yaw": list(ranges.get("ang_vel_z", (-0.5, 0.5))),
    }
  elif commands_cfg.get("twist") is not None:
    try:
      expected_command_ranges, _ = _resolve_twist_command_ranges(
        env_cfg_raw,
        policy_common_step_counter,
      )
    except ValueError as exc:
      errors.append(str(exc))
      expected_command_ranges = {}
  elif commands_cfg.get("heading") is not None:
    heading_cfg = commands_cfg["heading"]
    ranges = heading_cfg.get("ranges", {})
    expected_command_ranges = {
      "tar_speed_min": float(ranges.get("tar_speed_min", 0.0)),
      "tar_speed_max": float(ranges.get("tar_speed_max", 1.0)),
      "rel_standing_envs": float(heading_cfg.get("rel_standing_envs", 0.0)),
    }

  for key, expected_value in expected_command_ranges.items():
    actual_value = cfg.get("command_ranges", {}).get(key)
    if actual_value != expected_value:
      errors.append(
        f"command_ranges[{key!r}] mismatch: got {actual_value}, expected {expected_value}"
      )

  expected_force_range = commands_cfg.get("desired_force", {}).get("force_range")
  if expected_force_range is not None:
    expected_force_range = [
      [float(v) for v in expected_force_range[0]],
      [float(v) for v in expected_force_range[1]],
    ]
    for lo, hi in zip(expected_force_range[0], expected_force_range[1], strict=True):
      if not np.isclose(abs(lo), abs(hi)):
        errors.append(
          "deploy force command sensor assumes symmetric force ranges about zero"
        )
    expected_force_max_abs = [
      max(abs(lo), abs(hi))
      for lo, hi in zip(expected_force_range[0], expected_force_range[1], strict=True)
    ]
    actual_force_max_abs = cfg.get("force_command_max_abs")
    if actual_force_max_abs is None or not np.allclose(
      actual_force_max_abs, expected_force_max_abs
    ):
      errors.append(
        f"force_command_max_abs mismatch: got {actual_force_max_abs}, expected {expected_force_max_abs}"
      )
    expected_sampling_cfg: dict[str, Any] = {
      "force_range": expected_force_range,
      "stiffness_range": [
        float(v)
        for v in commands_cfg["desired_force"].get(
          "stiffness_range",
          (),
        )
      ],
      "resampling_time_range": [
        float(v)
        for v in commands_cfg["desired_force"].get(
          "resampling_time_range",
          (10.0, 10.0),
        )
      ],
      "sample_zero_force_probability": float(
        commands_cfg["desired_force"].get("sample_zero_force_probability", 0.0)
      ),
      "force_ramp_duration_range_s": [
        float(v)
        for v in commands_cfg["desired_force"].get(
          "force_ramp_duration_range_s",
          (0.0, 0.0),
        )
      ],
      "force_hold_duration_s": float(
        commands_cfg["desired_force"].get("force_hold_duration_s", 0.0)
      ),
      "startup_zero_force_duration_s": float(
        commands_cfg["desired_force"].get("startup_zero_force_duration_s", 0.0)
      ),
      "max_xy_reach_m": commands_cfg["desired_force"].get("max_xy_reach_m"),
    }
    actual_sampling_cfg = cfg.get("force_command_sampling")
    if actual_sampling_cfg is None:
      errors.append("force_command_sampling missing from deploy cfg")
    else:
      if not np.allclose(
        actual_sampling_cfg.get("force_range", []),
        expected_sampling_cfg["force_range"],
      ):
        errors.append("force_command_sampling.force_range does not match env.yaml")
      if not np.allclose(
        actual_sampling_cfg.get("stiffness_range", []),
        expected_sampling_cfg["stiffness_range"],
      ):
        errors.append("force_command_sampling.stiffness_range does not match env.yaml")
      if not np.allclose(
        actual_sampling_cfg.get("resampling_time_range", []),
        expected_sampling_cfg["resampling_time_range"],
      ):
        errors.append(
          "force_command_sampling.resampling_time_range does not match env.yaml"
        )
      if not np.isclose(
        actual_sampling_cfg.get("sample_zero_force_probability", np.nan),
        expected_sampling_cfg["sample_zero_force_probability"],
      ):
        errors.append(
          "force_command_sampling.sample_zero_force_probability does not match env.yaml"
        )
      if not np.allclose(
        actual_sampling_cfg.get("force_ramp_duration_range_s", []),
        expected_sampling_cfg["force_ramp_duration_range_s"],
      ):
        errors.append(
          "force_command_sampling.force_ramp_duration_range_s does not match env.yaml"
        )
      if not np.isclose(
        actual_sampling_cfg.get("force_hold_duration_s", np.nan),
        expected_sampling_cfg["force_hold_duration_s"],
      ):
        errors.append(
          "force_command_sampling.force_hold_duration_s does not match env.yaml"
        )
      if not np.isclose(
        actual_sampling_cfg.get("startup_zero_force_duration_s", np.nan),
        expected_sampling_cfg["startup_zero_force_duration_s"],
      ):
        errors.append(
          "force_command_sampling.startup_zero_force_duration_s does not match env.yaml"
        )
      expected_max_xy_reach_m = expected_sampling_cfg["max_xy_reach_m"]
      actual_max_xy_reach_m = actual_sampling_cfg.get("max_xy_reach_m")
      if expected_max_xy_reach_m is None:
        if actual_max_xy_reach_m is not None:
          errors.append("force_command_sampling.max_xy_reach_m should be unset")
      elif actual_max_xy_reach_m is None or not np.isclose(
        float(actual_max_xy_reach_m),
        float(expected_max_xy_reach_m),
      ):
        errors.append("force_command_sampling.max_xy_reach_m does not match env.yaml")

  expected_action_scale = resolve_action_scale_array(
    env_cfg_raw.get("actions", {}).get("joint_pos", {}).get("scale", 0.5),
    policy_joint_names,
  )
  actual_action_scale = resolve_action_scale_array(
    cfg["control"]["action_scale"],
    policy_joint_names,
  )
  if not np.allclose(actual_action_scale, expected_action_scale):
    errors.append("action_scale array does not match env.yaml")

  actuator_groups = _extract_actuator_groups(robot_scene)
  expected_stiffnesses = np.full(len(isaaclab_joint_names), np.nan, dtype=np.float64)
  expected_dampings = np.full(len(isaaclab_joint_names), np.nan, dtype=np.float64)
  for group in actuator_groups:
    joint_exprs = group.get("joint_names_expr") or group.get("target_names_expr") or []
    stiffness = group.get("stiffness", {})
    damping = group.get("damping", {})
    for expr in joint_exprs:
      stiffness_val = stiffness[expr] if isinstance(stiffness, dict) else stiffness
      damping_val = damping[expr] if isinstance(damping, dict) else damping
      for idx, joint_name in enumerate(isaaclab_joint_names):
        if re.match(expr, joint_name):
          expected_stiffnesses[idx] = float(stiffness_val)
          expected_dampings[idx] = float(damping_val)

  if np.isnan(expected_stiffnesses).any():
    missing = [
      joint
      for joint, val in zip(isaaclab_joint_names, expected_stiffnesses, strict=True)
      if np.isnan(val)
    ]
    errors.append(f"env.yaml stiffness leaves joints unresolved: {missing}")
  if np.isnan(expected_dampings).any():
    missing = [
      joint
      for joint, val in zip(isaaclab_joint_names, expected_dampings, strict=True)
      if np.isnan(val)
    ]
    errors.append(f"env.yaml damping leaves joints unresolved: {missing}")

  if not np.allclose(
    robot_configuration.joint_stiffnesses("isaaclab"), expected_stiffnesses
  ):
    errors.append("resolved stiffness array does not match env.yaml")
  if not np.allclose(robot_configuration.joint_dampings("isaaclab"), expected_dampings):
    errors.append("resolved damping array does not match env.yaml")

  base_default_positions = np.array(
    [
      type(robot_configuration)._joint_default_positions[j]
      for j in isaaclab_joint_names
    ],
    dtype=np.float64,
  )
  init_joint_pos = robot_scene.get("init_state", {}).get("joint_pos", {})
  expected_default_positions = _expand_regex_map(
    init_joint_pos,
    isaaclab_joint_names,
    base_values=base_default_positions,
  )
  if not np.allclose(
    robot_configuration.joint_default_positions("isaaclab"),
    expected_default_positions,
  ):
    errors.append("resolved default joint positions do not match env.yaml overrides")

  init_state = robot_scene.get("init_state", {})
  expected_base_pos = init_state.get("pos")
  expected_base_rot = init_state.get("rot")
  actual_base_pos = cfg["control"].get("init_base_pos")
  actual_base_rot = cfg["control"].get("init_base_rot")
  if expected_base_pos is not None and not np.allclose(
    actual_base_pos, expected_base_pos
  ):
    errors.append(
      f"init_base_pos mismatch: got {actual_base_pos}, expected {list(expected_base_pos)}"
    )
  if expected_base_rot is not None and not np.allclose(
    actual_base_rot, expected_base_rot
  ):
    errors.append(
      f"init_base_rot mismatch: got {actual_base_rot}, expected {list(expected_base_rot)}"
    )

  return errors


class BaseController:
  """Base class shared by all per-task deploy scripts.

  Subclasses must override ``build_sensors()`` and set the expected policy
  joint order.
  """

  # Order expected by the JIT actor for observations/actions.
  policy_joint_order: str = "mujoco"

  def __init__(
    self,
    interface_type: str,
    policy_name: str,
    task_name: str,
    render: bool = False,
    headless: bool = False,
    stochastic: bool = False,
    timestamp: str | None = None,
  ):
    self.interface_type = interface_type
    self.policy_name = policy_name
    self.task_name = task_name
    self.stochastic = stochastic

    if timestamp is None:
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    self.timestamp = timestamp

    policy_path = resolve_policy_path(self.policy_name)
    self.policy_path = policy_path
    policy_dir = os.path.dirname(policy_path)

    # ------------------------------------------------------------------
    # Load env config
    # ------------------------------------------------------------------
    env_yaml_path = os.path.join(policy_dir, "params/env.yaml")
    env_cfg_raw = _load_relaxed_yaml(env_yaml_path)
    self.policy_common_step_counter = _load_policy_common_step_counter(policy_path)
    self.cfg = parse_hardware_cfg(
      env_cfg_raw,
      policy_common_step_counter=self.policy_common_step_counter,
    )

    self.iteration_count = 0
    self.decimation = 1  # policy runs every step (control_dt already set)

    # ------------------------------------------------------------------
    # Robot configuration
    # ------------------------------------------------------------------
    from deploy.robots.g1_config import G1Config

    self.robot_configuration = G1Config(dt=self.cfg["control_dt"])
    self.robot_configuration.load_from_cfg(self.cfg)
    self._resolve_action_scale()
    config_errors = validate_runtime_config(
      env_cfg_raw,
      self.cfg,
      self.robot_configuration,
      self.policy_joint_order,
      policy_common_step_counter=self.policy_common_step_counter,
    )
    if config_errors:
      raise ValueError(
        "Deploy config validation failed against env.yaml:\n- "
        + "\n- ".join(config_errors)
      )

    # ------------------------------------------------------------------
    # Robot interface
    # ------------------------------------------------------------------
    self.robot = self._make_interface(
      robot_configuration=self.robot_configuration,
      render=render,
      headless=headless,
    )

    # ------------------------------------------------------------------
    # Sensors (subclass provides)
    # ------------------------------------------------------------------
    self.sensors = self.build_sensors()

    # ------------------------------------------------------------------
    # Obs-history buffer
    # ------------------------------------------------------------------
    self.history_length = self.cfg["history_length"]
    zero_obs = [
      torch.zeros(self._sensor_dim(s), dtype=torch.float32) for s in self.sensors
    ]
    self.obs_history_buf = [list(zero_obs) for _ in range(self.history_length)]
    self._history_is_primed = False

    # Compute per-frame obs dim for the policy input tensor
    self.obs_dim_per_frame = sum(self._sensor_dim(s) for s in self.sensors)
    self.ob_buffer = torch.zeros(
      1, self.obs_dim_per_frame * self.history_length, device="cpu"
    )

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    self.policy = self._load_policy(policy_path)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    self.prev_action = np.zeros(self.robot_configuration.num_joints)
    self.last_joint_position_target_hw = np.zeros(self.robot_configuration.num_joints)
    self.last_reconstructed_torque_hw = np.zeros(self.robot_configuration.num_joints)

    # Check velocity reference limits (skip if no velocity commands)
    if "command_ranges" in self.cfg:
      self.check_vel_ref_limits()

  # -----------------------------------------------------------------------
  # Abstract interface — subclasses implement
  # -----------------------------------------------------------------------

  def build_sensors(self) -> list:
    """Return ordered list of sensor instances for this task."""
    raise NotImplementedError

  def handle_runtime_buttons(
    self,
    buttons,
    *,
    pose_initialised: bool,
    policy_running: bool,
  ) -> None:
    """Task-specific runtime button handling hook."""
    del buttons, pose_initialised, policy_running

  def pre_policy_step(self) -> None:
    """Task-specific hook that runs before a policy observation is built."""
    return None

  # -----------------------------------------------------------------------
  # Helpers
  # -----------------------------------------------------------------------

  def _sensor_dim(self, sensor) -> int:
    dim = sensor.dim
    if dim == "nj":
      return self.robot_configuration.num_joints
    if isinstance(dim, str) and "nj" in dim:
      return self.robot_configuration.num_joints + int(dim.split("+")[1])
    return int(dim)

  def _resolve_action_scale(self):
    self.cfg["control"]["action_scale"] = resolve_action_scale_array(
      self.cfg["control"]["action_scale"],
      self.robot_configuration.joint_names(self.policy_joint_order),
    )

  def _make_interface(self, robot_configuration, render, headless) -> Any:
    if self.interface_type == "dummy":
      from deploy.interfaces.dummy_interface import DummyInterface as Iface
    elif self.interface_type == "lcm":
      from deploy.interfaces.lcm_interface import G1LCMInterface as Iface
    elif self.interface_type == "mujoco":
      from deploy.interfaces.mujoco_interface import MujocoInterface as Iface
    else:
      raise ValueError(f"Unsupported interface_type: {self.interface_type!r}")
    iface_kwargs = {
      "robot_configuration": robot_configuration,
      "cfg": self.cfg,
      "task_name": self.task_name,
    }
    if self.interface_type == "mujoco":
      iface_kwargs["headless"] = headless
    robot = Iface(**iface_kwargs)
    robot.initialize(render=render)
    self.is_mujoco = self.interface_type == "mujoco"
    return robot

  def _load_policy(self, policy_path: str):
    torch.set_num_threads(1)
    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()
    torch.jit.optimize_for_inference(policy)

    action_std = None
    if self.stochastic:
      pt_name = os.path.basename(policy_path).split(".jit")[0] + ".pt"
      pt_path = os.path.join(os.path.dirname(policy_path), pt_name)
      if os.path.exists(pt_path):
        ckpt = _load_relaxed_checkpoint(pt_path)
        std = ckpt.get("model_state_dict", {}).get("std")
        if std is not None:
          action_std = std.detach().clone()

    ob_buffer = self.ob_buffer

    def mapped_policy(obs, info=None):
      del info
      ob_buffer.copy_(obs["obs_history"].cpu().reshape(1, -1))
      with torch.inference_mode():
        action = policy.forward(ob_buffer)
      if action_std is not None:
        action = action + torch.randn_like(action) * action_std
      return action

    return mapped_policy

  # -----------------------------------------------------------------------
  # Observation / action
  # -----------------------------------------------------------------------

  def compute_observation(self) -> list:
    """Return list of per-sensor numpy arrays."""
    obs_buf = []
    from deploy.sensors.last_action_sensor import LastActionSensor

    for sensor in self.sensors:
      if isinstance(sensor, LastActionSensor):
        data = sensor.get_data(self.prev_action)
      else:
        data = sensor.get_data()
      obs_buf.append(torch.tensor(data, dtype=torch.float32))
    return obs_buf

  def get_action(self, obs) -> np.ndarray:
    with torch.no_grad():
      action = self.policy(obs).squeeze(0).detach().numpy()
    return action

  # -----------------------------------------------------------------------
  # Control loop helpers
  # -----------------------------------------------------------------------

  def initialize_pose(self, duration: float = 2.0, realtime: bool = True):
    """Gradually move all joints to the default pose."""
    total_steps = int(duration / 0.005)
    initial_q = self.robot.get_joint_pos(joint_order="hw")
    final_q = self.robot_configuration.joint_default_positions(joint_order="hw")

    for step in range(total_steps):
      t = step / total_steps
      q_target = t * final_q + (1 - t) * initial_q
      self.robot.apply_command(
        kp=self.robot_configuration.joint_stiffnesses(joint_order="hw"),
        kd=self.robot_configuration.joint_dampings(joint_order="hw"),
        q=q_target,
        dq=np.zeros(self.robot_configuration.num_joints),
        tau=np.zeros(self.robot_configuration.num_joints),
        torque_mode=False,
      )
      if realtime:
        time.sleep(0.005)
    print("Pose initialized. Press X to deploy policy.")

  def check_vel_ref_limits(self):
    """Sanity-check velocity command range (no-op if no velocity commands)."""
    cr = self.cfg.get("command_ranges", {})
    if "lin_vel_x" in cr:
      assert cr["lin_vel_x"][0] <= cr["lin_vel_x"][1], "lin_vel_x limits invalid"
    if "lin_vel_y" in cr:
      assert cr["lin_vel_y"][0] <= cr["lin_vel_y"][1], "lin_vel_y limits invalid"

  def execute_policy(self, previous_hold: bool = False):
    """Compute obs, run policy, apply actions."""
    obs_buf = self.compute_observation()

    if not previous_hold:
      if not self._history_is_primed:
        # Match MJLab CircularBuffer startup behavior by backfilling the
        # full observation history with the first real frame instead of
        # feeding synthetic zero-history into a history-based policy.
        self.obs_history_buf = [
          [sensor_obs.clone() for sensor_obs in obs_buf]
          for _ in range(self.history_length)
        ]
        self._history_is_primed = True
      else:
        self.obs_history_buf = self.obs_history_buf[1:] + [obs_buf]
      obs_flat = torch.cat(
        [
          torch.cat([frame[i] for frame in self.obs_history_buf], dim=0)
          for i in range(len(obs_buf))
        ],
        dim=0,
      ).unsqueeze(0)
      obs = {"obs_history": obs_flat}
      action = self.get_action(obs)
    else:
      action = self.prev_action

    # Scale
    action_scale = self.cfg["control"]["action_scale"]
    scaled_action = action * action_scale

    # Remap from policy order to hw order
    hw_action = self.robot_configuration.remap_joint_array(
      scaled_action, from_order=self.policy_joint_order, to_order="hw"
    )

    # Offset by default positions
    default_positions = self.robot_configuration.joint_default_positions(
      joint_order="hw"
    )
    q_des = hw_action.copy()
    if self.cfg["offset_actions"]:
      q_des += default_positions

    # Clip both sides of the position interval satisfying the PD torque bounds.
    q = self.robot.get_joint_pos(joint_order="hw")
    dq = self.robot.get_joint_vel(joint_order="hw")
    joint_kps = self.robot_configuration.joint_stiffnesses(joint_order="hw")
    joint_kds = self.robot_configuration.joint_dampings(joint_order="hw")
    max_torques = self.robot_configuration.joint_saturated_torques(joint_order="hw")
    max_velocities = self.robot_configuration.joint_max_velocities(joint_order="hw")

    dq_des = np.zeros(self.robot_configuration.num_joints)
    tau_ff = np.zeros(self.robot_configuration.num_joints)
    min_torque, max_torque = compute_torque_speed_limits(
      joint_vel=dq,
      saturation_effort=max_torques,
      velocity_limit=max_velocities,
    )
    q_des_lower = q + (min_torque - tau_ff + joint_kds * dq) / joint_kps
    q_des_upper = q + (max_torque - tau_ff + joint_kds * dq) / joint_kps
    q_des_clipped = np.clip(q_des, q_des_lower, q_des_upper)
    reconstructed_torque = (
      joint_kps * (q_des_clipped - q) + joint_kds * (dq_des - dq) + tau_ff
    )

    self.robot.apply_command(
      kp=joint_kps,
      kd=joint_kds,
      q=q_des_clipped,
      dq=dq_des,
      tau=tau_ff,
      torque_mode=False,
    )

    self.last_joint_position_target_hw[:] = q_des_clipped
    self.last_reconstructed_torque_hw[:] = reconstructed_torque

    if not previous_hold:
      self.prev_action[:] = action

    # Return q_des_clipped in policy order for data collection callers.
    return self.robot_configuration.remap_joint_array(
      q_des_clipped, from_order="hw", to_order=self.policy_joint_order
    )

  def shutdown(self):
    print("Exiting by engaging damping mode and closing channels...")
    self.robot.damping_mode()
    time.sleep(0.1)
    self.robot.shutdown()
    print("Shutdown complete. Exiting.")

  def run(self, time_limit_s: float | None = None, callback=None):
    """Main control loop. Waits for L1 to initialise pose, X to run policy."""
    print("Waiting for L1 button press to assume initial pose...")
    pose_initialised = False
    self.iteration_count = 0
    last_log_time = time.time()

    while True:
      if not pose_initialised:
        buttons = self.robot.get_buttons()
        self.handle_runtime_buttons(
          buttons,
          pose_initialised=pose_initialised,
          policy_running=False,
        )
        if buttons[1][1] == 1:  # L1
          print("L1 detected. Assuming initial pose...")
          if self.interface_type == "mujoco":
            self.robot.lock_base()
          self.initialize_pose()
          if self.interface_type == "mujoco":
            self.robot.unlock_base()
          pose_initialised = True
        elif buttons[11][1] == 1:  # Y
          self.robot.damping_mode()
        else:
          time.sleep(0.0001)
          continue
      else:
        self.robot.repeat_last_command()
        buttons = self.robot.get_buttons()
        self.handle_runtime_buttons(
          buttons,
          pose_initialised=pose_initialised,
          policy_running=False,
        )

        if buttons[10][1] == 1:  # X — run policy
          print("X detected. Deploying policy...")
          loop_start = time.time()
          while True:
            if time_limit_s is not None:
              elapsed = self.iteration_count / (1.0 / self.cfg["control_dt"])
              if elapsed > time_limit_s:
                print(f"Time limit {time_limit_s}s reached.")
                self.robot.damping_mode()
                return

            self.pre_policy_step()
            previous_hold = self.iteration_count % self.decimation != 0
            self.execute_policy(previous_hold=previous_hold)
            self.iteration_count += 1

            if callback is not None:
              callback(self)

            ctrl_time = time.time()
            sleep_time = self.cfg["control_dt"] - (ctrl_time - loop_start)
            if sleep_time > 0:
              time.sleep(sleep_time)
              loop_start = ctrl_time + sleep_time
            else:
              time.sleep(0.0001)
              loop_start = ctrl_time - 0.0001

            buttons = self.robot.get_buttons()
            self.handle_runtime_buttons(
              buttons,
              pose_initialised=pose_initialised,
              policy_running=True,
            )
            if buttons[11][1] == 1:  # Y — stop
              self.robot.damping_mode()
              break
            elif buttons[9][1] == 1:  # B — re-init
              self.initialize_pose()
              break

            if self.iteration_count % 100 == 0:
              print(f"Frequency: {100 / (time.time() - last_log_time):.1f} Hz")
              last_log_time = time.time()


# ---------------------------------------------------------------------------
# YAML config parsing
# ---------------------------------------------------------------------------


def parse_hardware_cfg(
  env_cfg: dict,
  *,
  policy_common_step_counter: int | None = None,
) -> dict:
  """Parse hardware-relevant fields from env.yaml.

  Reads the actor observation group or the policy-group export layout.

  ``policy_common_step_counter`` is used to resolve MJLab velocity command
  curricula into the active command range for the exported checkpoint.

  Returns a flat dict with keys:
      control_dt, history_length, offset_actions,
      ang_vel_scale, dof_vel_scale, force_scale (optional),
      control.action_scale, control.stiffnesses, control.dampings,
      control.saturation_effort, control.velocity_limit (optional),
      control.default_joint_positions, control.init_base_pos (optional),
      control.init_base_rot (optional)
  """
  cfg: dict[str, Any] = {}

  # ------------------------------------------------------------------
  # Observation group
  # ------------------------------------------------------------------
  observations = env_cfg.get("observations", {})
  obs_group = observations.get("actor") or observations.get("policy") or {}
  obs_terms = obs_group.get("terms", obs_group)
  cfg["actor_observation_terms"] = list(obs_terms.keys())

  # history_length
  raw_history_length = obs_group.get("history_length", 3)
  cfg["history_length"] = 1 if raw_history_length is None else int(raw_history_length)

  # Per-term scales
  def _obs_scale(term_name: str, default: float | None = 1.0) -> float | None:
    term = obs_terms.get(term_name)
    if isinstance(term, dict):
      return float(cast(Any, term.get("scale", default) or default))
    return default

  cfg["ang_vel_scale"] = _obs_scale("base_ang_vel", default=1.0)
  cfg["base_lin_vel_scale"] = _obs_scale("base_lin_vel", default=1.0)
  cfg["dof_vel_scale"] = _obs_scale(
    "joint_vel_rel", default=_obs_scale("joint_vel", default=1.0)
  )
  # force_scale — only present for force control policies
  force_scale = _obs_scale("force_command", default=None)
  if force_scale is not None and force_scale != 1.0:
    cfg["force_scale"] = force_scale
  elif "force_command" in obs_terms:
    cfg["force_scale"] = _obs_scale("force_command", default=1.0)

  # Joystick sensors read command scales from obs_scales.
  velocity_cmd_scale = _obs_scale(
    "velocity_commands",
    default=_obs_scale("command", default=1.0),
  )
  cfg["obs_scales"] = {
    "lin_vel": float(cast(float, velocity_cmd_scale)),
    "ang_vel": float(cast(float, velocity_cmd_scale)),
    "dof_vel": float(cast(float, cfg["dof_vel_scale"])),
  }

  # ------------------------------------------------------------------
  # Command ranges
  # ------------------------------------------------------------------
  commands_cfg = env_cfg.get("commands", {})
  cfg["command_ranges"] = {}

  if commands_cfg.get("upper_body_joints") is not None:
    proc_ranges = commands_cfg["upper_body_joints"].get("proc_ranges", {})
    cfg["command_ranges"]["lin_vel_x"] = [
      float(proc_ranges.get("min_vel", -1.0)),
      float(proc_ranges.get("max_vel", 1.0)),
    ]
    cfg["command_ranges"]["lin_vel_y"] = [0.0, 0.0]
    cfg["command_ranges"]["ang_vel_yaw"] = [
      float(proc_ranges.get("min_yaw_vel", -0.5)),
      float(proc_ranges.get("max_yaw_vel", 0.5)),
    ]
  elif commands_cfg.get("base_velocity") is not None:
    ranges = commands_cfg["base_velocity"].get("ranges", {})
    cfg["command_ranges"]["lin_vel_x"] = list(ranges.get("lin_vel_x", (-1.0, 1.0)))
    cfg["command_ranges"]["lin_vel_y"] = list(ranges.get("lin_vel_y", (-1.0, 1.0)))
    cfg["command_ranges"]["ang_vel_yaw"] = list(ranges.get("ang_vel_z", (-0.5, 0.5)))
  elif commands_cfg.get("heading") is not None:
    heading_cfg = commands_cfg["heading"]
    ranges = heading_cfg.get("ranges", {})
    cfg["command_ranges"]["tar_speed_min"] = float(ranges.get("tar_speed_min", 0.0))
    cfg["command_ranges"]["tar_speed_max"] = float(ranges.get("tar_speed_max", 1.0))
    cfg["command_ranges"]["rel_standing_envs"] = float(
      heading_cfg.get("rel_standing_envs", 0.0)
    )
    cfg["command_type"] = "heading"
  elif commands_cfg.get("desired_force") is not None:
    cfg["command_type"] = "force_control"
    cfg["command_ranges"]["lin_vel_x"] = [-1.0, 1.0]
    cfg["command_ranges"]["lin_vel_y"] = [-1.0, 1.0]
    cfg["command_ranges"]["ang_vel_yaw"] = [-0.5, 0.5]
  elif commands_cfg.get("twist") is not None:
    cfg["command_ranges"], used_twist_curriculum = _resolve_twist_command_ranges(
      env_cfg,
      policy_common_step_counter,
    )
    if used_twist_curriculum:
      cfg["command_ranges_source"] = "twist_curriculum"
      cfg["policy_common_step_counter"] = policy_common_step_counter

  gait_phase_params = obs_terms.get("gait_phase", {}).get("params", {})
  if gait_phase_params.get("freq_range") is not None:
    cfg["gait_phase_freq_range"] = [float(v) for v in gait_phase_params["freq_range"]]

  gait_cmd_cfg = commands_cfg.get("gait_frequency", {})
  if gait_cmd_cfg.get("frequency_range") is not None:
    cfg["gait_frequency_command_range"] = [
      float(v) for v in gait_cmd_cfg["frequency_range"]
    ]

  swing_cmd_cfg = commands_cfg.get("foot_swing_height", {})
  if swing_cmd_cfg.get("swing_height_range") is not None:
    cfg["foot_swing_height_command_range"] = [
      float(v) for v in swing_cmd_cfg["swing_height_range"]
    ]

  desired_force_cfg = commands_cfg.get("desired_force", {})
  force_range = desired_force_cfg.get("force_range")
  if force_range is not None:
    cfg["force_command_max_abs"] = [
      max(abs(float(lo)), abs(float(hi)))
      for lo, hi in zip(force_range[0], force_range[1], strict=True)
    ]
    cfg["force_command_sampling"] = {
      "force_range": [
        [float(v) for v in force_range[0]],
        [float(v) for v in force_range[1]],
      ],
      "stiffness_range": [
        float(v) for v in desired_force_cfg.get("stiffness_range", ())
      ],
      "resampling_time_range": [
        float(v) for v in desired_force_cfg.get("resampling_time_range", (10.0, 10.0))
      ],
      "sample_zero_force_probability": float(
        desired_force_cfg.get("sample_zero_force_probability", 0.0)
      ),
      "force_ramp_duration_range_s": [
        float(v)
        for v in desired_force_cfg.get(
          "force_ramp_duration_range_s",
          (0.0, 0.0),
        )
      ],
      "force_hold_duration_s": float(
        desired_force_cfg.get("force_hold_duration_s", 0.0)
      ),
      "startup_zero_force_duration_s": float(
        desired_force_cfg.get("startup_zero_force_duration_s", 0.0)
      ),
      "max_xy_reach_m": (
        None
        if desired_force_cfg.get("max_xy_reach_m") is None
        else float(desired_force_cfg.get("max_xy_reach_m"))
      ),
    }

  # ------------------------------------------------------------------
  # Actions
  # ------------------------------------------------------------------
  actions_cfg = env_cfg.get("actions", {})
  joint_pos_action = actions_cfg.get("joint_pos", {})

  action_scale_raw = joint_pos_action.get("scale", 0.5)
  # Per-joint regex scales are resolved by the robot configuration loader.
  control_cfg: dict[str, Any] = {
    "action_scale": (
      action_scale_raw
      if isinstance(action_scale_raw, dict)
      else float(action_scale_raw)
    )
  }
  cfg["control"] = control_cfg

  for field in ("saturation_effort", "velocity_limit"):
    if field in joint_pos_action:
      cfg["control"][field] = joint_pos_action[field]

  cfg["offset_actions"] = bool(joint_pos_action.get("use_default_offset", False))

  # ------------------------------------------------------------------
  # Control rate
  # ------------------------------------------------------------------
  sim_cfg = env_cfg.get("sim", {})
  sim_dt = float(sim_cfg.get("dt", 0.005))
  decimation = int(env_cfg.get("decimation", 4))
  cfg["control_dt"] = sim_dt * decimation

  # ------------------------------------------------------------------
  # Robot scene cfg
  # ------------------------------------------------------------------
  scene_cfg = env_cfg.get("scene", {})
  robot_scene = scene_cfg.get("robot")
  if robot_scene is None:
    robot_scene = scene_cfg.get("entities", {}).get("robot", {})

  # ------------------------------------------------------------------
  # Gains / stiffness / damping
  # ------------------------------------------------------------------
  actuators = robot_scene.get("actuators")
  if actuators is None:
    actuators = robot_scene.get("articulation", {}).get("actuators", {})

  stiffnesses: dict[str, float] = {}
  dampings: dict[str, float] = {}
  if isinstance(actuators, dict):
    actuator_iter = actuators.values()
  else:
    actuator_iter = actuators or []

  for group in actuator_iter:
    k = group.get("stiffness", {})
    d = group.get("damping", {})
    joint_exprs = group.get("joint_names_expr") or group.get("target_names_expr") or []
    for expr in joint_exprs:
      stiffnesses[expr] = k[expr] if isinstance(k, dict) else float(k)
      dampings[expr] = d[expr] if isinstance(d, dict) else float(d)

  cfg["control"]["stiffnesses"] = stiffnesses
  cfg["control"]["dampings"] = dampings

  # ------------------------------------------------------------------
  # Default joint positions
  # ------------------------------------------------------------------
  init_state = robot_scene.get("init_state", {})
  init_joint_pos = init_state.get("joint_pos", {})
  if init_joint_pos:
    cfg["control"]["default_joint_positions"] = dict(init_joint_pos)

  # Base init pose
  init_base_pos = init_state.get("pos")
  init_base_rot = init_state.get("rot")
  if init_base_pos is not None:
    cfg["control"]["init_base_pos"] = list(init_base_pos)
  if init_base_rot is not None:
    cfg["control"]["init_base_rot"] = list(init_base_rot)

  # ------------------------------------------------------------------
  # MuJoCo sim cfg (passed to mujoco interface if needed)
  # ------------------------------------------------------------------
  mujoco_cfg = sim_cfg.get("mujoco", {})
  if mujoco_cfg:
    cfg["mujoco"] = mujoco_cfg

  return cfg
