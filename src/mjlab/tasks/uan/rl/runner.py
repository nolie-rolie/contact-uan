"""Task-local runner for UAN rollout checkpoints."""

from __future__ import annotations

import torch

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.uan.mdp.utils import resolve_g1_uan_actuator_sharing


class UANRolloutOnPolicyRunner(MjlabOnPolicyRunner):
  """Adds deployable UAN metadata without touching the default MJLab runner."""

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    self._configure_g1_uan_actor_sharing(env, train_cfg)
    super().__init__(env, train_cfg, log_dir, device)

  @staticmethod
  def _is_uan_actor_cfg(actor_cfg: dict | None) -> bool:
    if not isinstance(actor_cfg, dict):
      return False
    return str(actor_cfg.get("class_name", "")).endswith("UANActor")

  @classmethod
  def _configure_g1_uan_actor_sharing(cls, env, train_cfg: dict) -> None:
    actor_cfg = train_cfg.get("actor")
    if not cls._is_uan_actor_cfg(actor_cfg):
      return
    assert isinstance(actor_cfg, dict)
    num_actuators = int(actor_cfg.get("num_actuators", 5))
    robot = env.unwrapped.scene["robot"]
    joint_to_actuator_map, _, _ = resolve_g1_uan_actuator_sharing(
      joint_names=tuple(robot.joint_names),
      num_actuators=num_actuators,
      joint_to_actuator_map=actor_cfg.get("joint_to_actuator_map"),
    )
    actor_cfg["num_actuators"] = num_actuators
    actor_cfg["joint_to_actuator_map"] = joint_to_actuator_map

  def save(self, path: str, infos=None) -> None:
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    saved_dict = self.alg.save()
    saved_dict.update(self._uan_metadata())
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

  def _uan_metadata(self) -> dict:
    actor = getattr(self.alg, "actor", None)
    actor_cfg = self.cfg.get("actor", {})
    env_cfg = self.env.unwrapped.cfg
    action_cfg = env_cfg.actions.get("residual_target")
    if actor is None or actor.__class__.__name__ != "UANActor":
      return {}
    joint_to_actuator = actor.joint_to_actuator
    residual_clip = getattr(action_cfg, "residual_clip", None)
    actor_obs_cfg = env_cfg.observations.get("actor")
    actor_block_names = []
    if actor_obs_cfg is not None:
      actor_block_names = list(actor_obs_cfg.terms.keys())
    joint_to_actuator_list = joint_to_actuator.detach().cpu().tolist()
    joint_names = tuple(self.env.unwrapped.scene["robot"].joint_names)
    _, actuator_grouping, actuator_group_names = resolve_g1_uan_actuator_sharing(
      joint_names=joint_names,
      num_actuators=actor.num_actuators,
      joint_to_actuator_map=joint_to_actuator_list,
    )
    return {
      "num_joints": getattr(actor, "num_joints", 29),
      "num_actuators": getattr(actor, "num_actuators", 5),
      "joint_to_actuator_map": joint_to_actuator_list,
      "actuator_grouping": actuator_grouping,
      "actuator_group_names": actuator_group_names,
      "uan_history_length": getattr(actor, "history_length", 3),
      "num_blocks": getattr(actor, "num_blocks", 2),
      "uan_obs_block_names": actor_block_names,
      "activation": actor_cfg.get("activation", "elu"),
      "use_bias": getattr(actor, "use_bias", True),
      "uan_action_clip": residual_clip,
      "joint_order": "mujoco",
      "residual_units": "rad",
    }
