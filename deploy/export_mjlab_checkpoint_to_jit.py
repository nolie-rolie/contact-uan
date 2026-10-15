"""Export MJLab actor checkpoints (.pt) to deploy-ready TorchScript (.jit).

Usage examples:
    uv run --locked python deploy/export_mjlab_checkpoint_to_jit.py \
        --policy-type wtw \
        --checkpoint /path/to/your/wtw-checkpoint.pt

    uv run --locked python deploy/export_mjlab_checkpoint_to_jit.py \
        --policy-type velocity \
        --checkpoint /path/to/your/velocity-checkpoint.pt

    uv run --locked python deploy/export_mjlab_checkpoint_to_jit.py \
        --policy-type force_control \
        --checkpoint /path/to/your/force-control-checkpoint.pt

"""

from __future__ import annotations

import argparse
import os
import pickle
import types
from typing import Any, TypedDict, cast

import torch
import torch.nn as nn
import yaml

ISAACLAB_TO_MUJOCO = [
  0,
  3,
  6,
  9,
  13,
  17,
  1,
  4,
  7,
  10,
  14,
  18,
  2,
  5,
  8,
  11,
  15,
  19,
  21,
  23,
  25,
  27,
  12,
  16,
  20,
  22,
  24,
  26,
  28,
]

MUJOCO_TO_ISAACLAB = [0] * 29
for _mj, _il in enumerate(ISAACLAB_TO_MUJOCO):
  MUJOCO_TO_ISAACLAB[_il] = _mj


class PolicySpec(TypedDict):
  frame_dim: int
  leading_identity_dims: tuple[int, ...]
  default_joint_order_target: str


POLICY_SPECS: dict[str, PolicySpec] = {
  "wtw": {
    "frame_dim": 99,
    "leading_identity_dims": (3, 3),
    "default_joint_order_target": "mujoco",
  },
  "velocity": {
    "frame_dim": 96,
    "leading_identity_dims": (3, 3),
    "default_joint_order_target": "mujoco",
  },
  "force_control": {
    "frame_dim": 99,
    "leading_identity_dims": (3, 3, 3),
    "default_joint_order_target": "mujoco",
  },
  "jumping_track": {
    "frame_dim": 161,
    "leading_identity_dims": (),
    "default_joint_order_target": "mujoco",
  },
}

EMPIRICAL_NORMALIZER_EPS = 1e-2


class PlainMjlabPolicy(nn.Module):
  """Deploy wrapper for policies that already use MuJoCo joint order."""

  def __init__(
    self,
    mlp: nn.Sequential,
    obs_mean: torch.Tensor | None = None,
    obs_std: torch.Tensor | None = None,
  ) -> None:
    super().__init__()
    self.mlp = mlp
    self.has_normalizer = obs_mean is not None and obs_std is not None
    if self.has_normalizer:
      self.register_buffer("obs_mean", obs_mean)
      self.register_buffer("obs_std", obs_std)
    else:
      self.register_buffer("obs_mean", torch.zeros(1))
      self.register_buffer("obs_std", torch.ones(1))

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    if self.has_normalizer:
      obs_mean = cast(torch.Tensor, self.obs_mean)
      obs_std = cast(torch.Tensor, self.obs_std)
      obs = (obs - obs_mean) / (obs_std + EMPIRICAL_NORMALIZER_EPS)
    return self.mlp(obs)


class MjlabToIsaacLabPolicy(nn.Module):
  """Wrapper for IsaacLab-order exports."""

  def __init__(
    self,
    mlp: nn.Sequential,
    obs_dim: int,
    history_len: int,
    leading_identity_dims: tuple[int, ...],
    obs_mean: torch.Tensor | None = None,
    obs_std: torch.Tensor | None = None,
    num_joints: int = 29,
  ) -> None:
    super().__init__()
    self.mlp = mlp
    self.has_normalizer = obs_mean is not None and obs_std is not None
    if self.has_normalizer:
      self.register_buffer("obs_mean", obs_mean)
      self.register_buffer("obs_std", obs_std)
    else:
      self.register_buffer("obs_mean", torch.zeros(1))
      self.register_buffer("obs_std", torch.ones(1))

    obs_perm: list[int] = []
    base = 0

    for dim in leading_identity_dims:
      for t in range(history_len):
        for d in range(dim):
          obs_perm.append(base + t * dim + d)
      base += dim * history_len

    for block in range(3):
      block_start = base + block * num_joints * history_len
      for t in range(history_len):
        for mj_j in range(num_joints):
          il_j = ISAACLAB_TO_MUJOCO[mj_j]
          obs_perm.append(block_start + t * num_joints + il_j)
    base += 3 * num_joints * history_len

    remaining = obs_dim - base
    for idx in range(remaining):
      obs_perm.append(base + idx)

    self.register_buffer("obs_perm", torch.tensor(obs_perm, dtype=torch.long))
    self.register_buffer("act_perm", torch.tensor(MUJOCO_TO_ISAACLAB, dtype=torch.long))

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    obs = obs[:, cast(torch.Tensor, self.obs_perm)]
    if self.has_normalizer:
      obs_mean = cast(torch.Tensor, self.obs_mean)
      obs_std = cast(torch.Tensor, self.obs_std)
      obs = (obs - obs_mean) / (obs_std + EMPIRICAL_NORMALIZER_EPS)
    out = self.mlp(obs)
    return out[:, cast(torch.Tensor, self.act_perm)]


def _safe_load_checkpoint(path: str) -> dict:
  class _Dummy:
    def __init__(self, *args, **kwargs):
      pass

  class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
      try:
        return super().find_class(module, name)
      except (ModuleNotFoundError, ImportError, AttributeError):
        return _Dummy

  stub_mod = cast(Any, types.ModuleType("_pickle_stub"))
  stub_mod.Unpickler = _StubUnpickler
  stub_mod.load = pickle.load
  stub_mod.dump = pickle.dump

  return torch.load(
    path,
    map_location="cpu",
    weights_only=False,
    pickle_module=stub_mod,
  )


def _load_activation(checkpoint_path: str, override: str | None) -> nn.Module:
  if override is not None:
    activation = override
  else:
    activation = "elu"
    agent_yaml_path = os.path.join(
      os.path.dirname(checkpoint_path), "params", "agent.yaml"
    )
    if os.path.exists(agent_yaml_path):
      with open(agent_yaml_path, "r") as f:
        agent_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
      activation = (
        agent_cfg.get("actor", {}).get("activation", "elu")
        if isinstance(agent_cfg, dict)
        else "elu"
      )

  activation = activation.lower()
  if activation == "elu":
    return nn.ELU()
  if activation == "relu":
    return nn.ReLU()
  if activation == "tanh":
    return nn.Tanh()
  raise ValueError(f"Unsupported activation: {activation}")


def _build_mlp(
  actor_state_dict: dict, activation: nn.Module
) -> tuple[nn.Sequential, int]:
  weight_keys = sorted(
    [k for k in actor_state_dict if k.startswith("mlp.") and k.endswith(".weight")],
    key=lambda k: int(k.split(".")[1]),
  )
  if not weight_keys:
    raise ValueError("Checkpoint does not contain actor MLP weights")

  layers: list[nn.Module] = []
  for index, weight_key in enumerate(weight_keys):
    layer_idx = int(weight_key.split(".")[1])
    weight = actor_state_dict[weight_key]
    bias = actor_state_dict[f"mlp.{layer_idx}.bias"]
    linear = nn.Linear(weight.shape[1], weight.shape[0])
    linear.weight.data = weight
    linear.bias.data = bias
    layers.append(linear)
    if index < len(weight_keys) - 1:
      layers.append(activation)

  obs_dim = actor_state_dict[weight_keys[0]].shape[1]
  return nn.Sequential(*layers).eval(), obs_dim


def _infer_history_len(obs_dim: int, frame_dim: int, override: int | None) -> int:
  if override is not None:
    return override
  if obs_dim % frame_dim != 0:
    raise ValueError(
      f"obs_dim={obs_dim} is not divisible by expected frame_dim={frame_dim}"
    )
  return obs_dim // frame_dim


def _default_output_path(checkpoint_path: str) -> str:
  if checkpoint_path.endswith(".pt"):
    return checkpoint_path[:-3] + ".jit"
  raise ValueError(f"Expected a .pt checkpoint, got: {checkpoint_path}")


def _resolve_checkpoint_path(checkpoint: str) -> str:
  if os.path.isabs(checkpoint):
    return checkpoint

  base_dir = os.path.dirname(os.path.realpath(__file__))
  candidates = [
    os.path.abspath(checkpoint),
    os.path.abspath(os.path.join(base_dir, checkpoint)),
    os.path.abspath(os.path.join(base_dir, "models", checkpoint)),
  ]
  for candidate in candidates:
    if os.path.exists(candidate):
      return candidate
  return candidates[-1]


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--policy-type",
    required=True,
    choices=sorted(POLICY_SPECS.keys()),
    help="Deploy policy family to export for.",
  )
  parser.add_argument(
    "--checkpoint",
    required=True,
    help="Path to MJLab actor checkpoint (.pt).",
  )
  parser.add_argument(
    "--output",
    default=None,
    help="Output .jit path (defaults to sibling model_<iter>.jit).",
  )
  parser.add_argument(
    "--joint-order-target",
    choices=("auto", "mujoco", "isaaclab"),
    default="auto",
    help=(
      "Target deploy joint order. Auto defaults to MuJoCo for the deployment scripts."
    ),
  )
  parser.add_argument(
    "--history-len",
    type=int,
    default=None,
    help="Override inferred observation history length.",
  )
  parser.add_argument(
    "--activation",
    default=None,
    help="Override actor activation (default: read params/agent.yaml, fallback to elu).",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  spec = POLICY_SPECS[args.policy_type]
  checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
  output_path = os.path.abspath(args.output or _default_output_path(checkpoint_path))

  joint_order_target = args.joint_order_target
  if joint_order_target == "auto":
    joint_order_target = spec["default_joint_order_target"]
  if args.policy_type == "jumping_track" and joint_order_target != "mujoco":
    raise ValueError("JumpingTrack export supports only MuJoCo joint order.")

  ckpt = _safe_load_checkpoint(checkpoint_path)
  actor_state_dict = ckpt["actor_state_dict"]
  activation = _load_activation(checkpoint_path, override=args.activation)
  mlp, obs_dim = _build_mlp(actor_state_dict, activation)
  history_len = _infer_history_len(
    obs_dim, spec["frame_dim"], override=args.history_len
  )
  obs_mean = actor_state_dict.get("obs_normalizer._mean")
  obs_std = actor_state_dict.get("obs_normalizer._std")

  if joint_order_target == "isaaclab":
    wrapper = MjlabToIsaacLabPolicy(
      mlp=mlp,
      obs_dim=obs_dim,
      history_len=history_len,
      leading_identity_dims=spec["leading_identity_dims"],
      obs_mean=obs_mean,
      obs_std=obs_std,
    )
  else:
    wrapper = PlainMjlabPolicy(
      mlp=mlp,
      obs_mean=obs_mean,
      obs_std=obs_std,
    )

  example = torch.zeros(1, obs_dim, dtype=torch.float32)
  traced = cast(
    torch.jit.ScriptModule,
    torch.jit.trace(wrapper.eval(), example, check_trace=False),
  )
  os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
  traced.save(output_path)

  print(f"[Saved] {output_path}")
  print(
    f"[Export] policy_type={args.policy_type} "
    f"joint_order_target={joint_order_target} "
    f"obs_dim={obs_dim} history_len={history_len} "
    f"normalized={obs_mean is not None and obs_std is not None}"
  )


if __name__ == "__main__":
  main()
