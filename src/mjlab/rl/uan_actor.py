"""Contact-UAN actor: five shared 6-128-128-1 ELU actuator networks.

Each network predicts a joint-target correction in radians from three frames
of target-position error and joint velocity. Joints share weights by G1 actuator
family, in native MuJoCo joint order.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any, cast

import torch
import torch.nn as nn
from rsl_rl.modules import HiddenState
from rsl_rl.modules.distribution import Distribution, GaussianDistribution
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories
from tensordict import TensorDict

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_UAN_JOINT_TO_ACTUATOR_MAP,
)


class UANActor(nn.Module):
  """Paper actuator-family model satisfying the rsl_rl model interface."""

  is_recurrent: bool = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    num_joints: int = 29,
    num_actuators: int = 5,
    joint_to_actuator_map: list[int] | None = None,
    history_length: int = 3,
    num_blocks: int = 2,
    hidden_dims: tuple[int, ...] | list[int] = (128, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    use_bias: bool = True,
    final_layer_init_uniform: float | None = None,
    final_layer_bias_zero: bool = False,
    distribution_cfg: dict | None = None,
  ) -> None:
    super().__init__()

    if (num_joints, num_actuators, history_length, num_blocks) != (29, 5, 3, 2):
      raise ValueError(
        "Contact-UAN requires 29 joints, five actuator families, three history "
        "frames, and two observation blocks."
      )
    if (
      tuple(hidden_dims) != (128, 128)
      or activation.lower() != "elu"
      or obs_normalization
      or not use_bias
    ):
      raise ValueError(
        "Contact-UAN uses two 128-unit ELU layers with biases and no observation "
        "normalization."
      )
    self.num_joints = num_joints
    self.num_actuators = num_actuators
    self.history_length = history_length
    self.num_blocks = num_blocks
    self.use_bias = use_bias

    # --- Observation groups ---
    self.obs_groups, self.obs_dim = self._get_obs_dim(
      obs,
      obs_groups,
      obs_set,
    )

    # --- Joint → actuator mapping ---
    if joint_to_actuator_map is None:
      joint_to_actuator_map = list(G1_UAN_JOINT_TO_ACTUATOR_MAP)
    j2a = torch.as_tensor(joint_to_actuator_map, dtype=torch.long)
    if tuple(joint_to_actuator_map) != G1_UAN_JOINT_TO_ACTUATOR_MAP:
      raise ValueError("Contact-UAN requires the canonical G1 actuator-family map.")
    if self.obs_dim != 174:
      raise ValueError(f"Contact-UAN requires 174 observations, got {self.obs_dim}.")
    self.register_buffer("joint_to_actuator", j2a, persistent=True)

    # --- Observation normalization ---
    self.obs_normalization = obs_normalization
    self.obs_normalizer = nn.Identity()

    # --- Distribution ---
    if distribution_cfg is not None:
      dist_cfg = dict(distribution_cfg)  # don't mutate caller's dict
      if dist_cfg.get("class_name") != "GaussianDistribution":
        raise ValueError("Contact-UAN uses a scalar Gaussian exploration distribution.")
      dist_cfg.pop("class_name")
      self.distribution: Distribution | None = GaussianDistribution(
        num_joints,
        **dist_cfg,
      )
    else:
      self.distribution = None
    self.output_per_joint = 1

    # --- Per-joint einsum layers ---
    act_fn = resolve_nn_activation(activation)
    self.activation = act_fn

    per_joint_in_dim = num_blocks * history_length
    layer_sizes = [per_joint_in_dim, *hidden_dims, self.output_per_joint]
    self.num_layers = len(layer_sizes) - 1

    self.layer_weights = nn.ParameterList()
    self.layer_biases = nn.ParameterList()
    for i in range(self.num_layers):
      in_f = layer_sizes[i]
      out_f = layer_sizes[i + 1]

      W = torch.empty(num_actuators, out_f, in_f)

      for a in range(num_actuators):
        nn.init.kaiming_uniform_(W[a], a=math.sqrt(5))

      self.layer_weights.append(nn.Parameter(W))
      if self.use_bias:
        b = torch.empty(num_actuators, out_f)
        for a in range(num_actuators):
          fan_in, _ = nn.init._calculate_fan_in_and_fan_out(W[a])
          bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
          nn.init.uniform_(b[a], -bound, bound)
        self.layer_biases.append(nn.Parameter(b))

    if final_layer_init_uniform is not None:
      bound = float(final_layer_init_uniform)
      with torch.no_grad():
        self.layer_weights[-1].uniform_(-bound, bound)
        if self.use_bias and final_layer_bias_zero:
          self.layer_biases[-1].zero_()

  # ------------------------------------------------------------------
  # Core per-joint forward
  # ------------------------------------------------------------------

  def _per_joint_forward(
    self,
    flat_obs: torch.Tensor,
  ) -> torch.Tensor:
    """Einsum-based per-joint forward.

    Args:
        flat_obs: ``[B, num_blocks * history * num_joints]``

    Returns:
        Joint-target corrections with shape ``[B, num_joints]``.
    """
    B = flat_obs.shape[0]
    J = self.num_joints

    # [B, num_blocks*H*J] → [B, per_joint_dim, J] → [B, J, per_joint_dim]
    pjd = flat_obs.shape[1] // J
    x = flat_obs.view(B, pjd, J).permute(0, 2, 1).contiguous()

    for i in range(self.num_layers):
      W = self.layer_weights[i][self.joint_to_actuator]  # [J, O, I]
      x = torch.einsum("joi,bji->bjo", W, x)
      if self.use_bias:
        b = self.layer_biases[i][self.joint_to_actuator]  # [J, O]
        x = x + b
      if i < self.num_layers - 1:
        x = self.activation(x)

    return x.squeeze(-1)

  # ------------------------------------------------------------------
  # MLPModel interface
  # ------------------------------------------------------------------

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    obs = (
      cast(TensorDict, unpad_trajectories(obs, masks))
      if masks is not None and not self.is_recurrent
      else obs
    )
    latent = self.get_latent(obs, masks, hidden_state)
    mlp_output = self._per_joint_forward(latent)

    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(mlp_output)
        return self.distribution.sample()
      return self.distribution.deterministic_output(mlp_output)
    return mlp_output

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    parts = [cast(torch.Tensor, obs[g]) for g in self.obs_groups]
    latent = torch.cat(parts, dim=-1)
    return self.obs_normalizer(latent)

  def reset(
    self,
    dones: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> None:
    pass

  def get_hidden_state(self) -> HiddenState:
    return None

  def detach_hidden_state(
    self,
    dones: torch.Tensor | None = None,
  ) -> None:
    pass

  # --- Distribution properties ---

  @property
  def output_mean(self) -> torch.Tensor:
    return self.distribution.mean  # type: ignore[union-attr]

  @property
  def output_std(self) -> torch.Tensor:
    return self.distribution.std  # type: ignore[union-attr]

  @property
  def output_entropy(self) -> torch.Tensor:
    return self.distribution.entropy  # type: ignore[union-attr]

  @property
  def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
    return self.distribution.params  # type: ignore[union-attr]

  def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    return self.distribution.log_prob(outputs)  # type: ignore[union-attr]

  def get_kl_divergence(
    self,
    old_params: tuple[torch.Tensor, ...],
    new_params: tuple[torch.Tensor, ...],
  ) -> torch.Tensor:
    return self.distribution.kl_divergence(  # type: ignore[union-attr]
      old_params,
      new_params,
    )

  def update_normalization(self, obs: TensorDict) -> None:
    pass

  def as_jit(self) -> nn.Module:
    return _JitUANActor(self)

  def load_state_dict(
    self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
  ):
    mapping = state_dict.get("joint_to_actuator")
    if mapping is None or mapping.tolist() != list(G1_UAN_JOINT_TO_ACTUATOR_MAP):
      raise ValueError("Checkpoint must use the canonical G1 actuator-family map.")
    return super().load_state_dict(state_dict, strict=strict, assign=assign)

  # --- Checkpoint loading ---

  @classmethod
  def from_checkpoint(
    cls,
    path: str,
    device: str | torch.device = "cpu",
  ) -> UANActor:
    """Load a paper-configuration checkpoint and validate its semantics."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt["actor_state_dict"]
    if ckpt.get("joint_order", "mujoco") != "mujoco":
      raise ValueError("Contact-UAN checkpoints must use MuJoCo joint order.")
    if ckpt.get("residual_units", "rad") != "rad":
      raise ValueError("Contact-UAN checkpoints must predict joint-target radians.")
    expected_blocks = ("target_pos_error", "joint_vel")
    if tuple(ckpt.get("uan_obs_block_names", expected_blocks)) != expected_blocks:
      raise ValueError("Contact-UAN checkpoints require target error and velocity.")
    mapping = ckpt.get("joint_to_actuator_map", G1_UAN_JOINT_TO_ACTUATOR_MAP)
    saved_mapping = sd.get("joint_to_actuator")
    if saved_mapping is None or saved_mapping.tolist() != list(mapping):
      raise ValueError("Checkpoint actuator mapping does not match its metadata.")
    distribution_cfg = None
    if "distribution.std_param" in sd:
      distribution_cfg = {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      }
    dummy_obs = TensorDict(
      {"uan_obs": torch.zeros(1, 174, device=device)}, batch_size=[1]
    )
    actor = cls(
      obs=dummy_obs,
      obs_groups={"actor": ["uan_obs"]},
      obs_set="actor",
      num_joints=ckpt.get("num_joints", 29),
      num_actuators=ckpt.get("num_actuators", 5),
      joint_to_actuator_map=list(mapping),
      history_length=ckpt.get("uan_history_length", 3),
      num_blocks=ckpt.get("num_blocks", 2),
      activation=ckpt.get("activation", "elu"),
      use_bias=ckpt.get("use_bias", True),
      distribution_cfg=distribution_cfg,
    )
    # Shape and key checks reject alternate architectures and normalization.
    actor.load_state_dict(sd, strict=True)
    actor.to(device)
    actor.eval()
    return actor

  # ------------------------------------------------------------------
  # Private helpers
  # ------------------------------------------------------------------

  @staticmethod
  def _get_obs_dim(
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
  ) -> tuple[list[str], int]:
    active = obs_groups[obs_set]
    dim = 0
    for g in active:
      if len(obs[g].shape) != 2:
        raise ValueError(
          f"UANActor only supports 1D obs, got shape {obs[g].shape} for '{g}'."
        )
      dim += obs[g].shape[-1]
    return active, dim


class _JitUANActor(nn.Module):
  """Deterministic TorchScript export of the paper actuator-family model."""

  def __init__(self, model: UANActor) -> None:
    super().__init__()
    self.num_joints = model.num_joints
    self.num_layers = model.num_layers
    self.activation = copy.deepcopy(model.activation)
    self.layer_weights = nn.ParameterList(
      [nn.Parameter(w.detach().clone()) for w in model.layer_weights]
    )
    self.layer_biases = nn.ParameterList(
      [nn.Parameter(b.detach().clone()) for b in model.layer_biases]
    )
    self.register_buffer(
      "joint_to_actuator",
      cast(torch.Tensor, model.joint_to_actuator).clone(),
      persistent=True,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.view(x.shape[0], 6, self.num_joints).permute(0, 2, 1).contiguous()
    for i, (weights, biases) in enumerate(zip(self.layer_weights, self.layer_biases)):  # noqa: B905
      x = torch.einsum("joi,bji->bjo", weights[self.joint_to_actuator], x)
      x = x + biases[self.joint_to_actuator]
      if i < self.num_layers - 1:
        x = self.activation(x)
    return x.squeeze(-1)

  @torch.jit.export
  def reset(self) -> None:
    pass
