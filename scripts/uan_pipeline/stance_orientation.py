"""Reconstruct root orientation from contact-gated stance-foot estimates."""

from __future__ import annotations

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.uan.mdp.utils import write_recorded_state
from mjlab.utils.lab_api.math import quat_apply

NEUTRAL_PROJECTED_GRAVITY = (0.0, 0.0, -1.0)


def normalize_np(vec: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
  arr = np.asarray(vec, dtype=np.float32)
  return arr / np.maximum(np.linalg.norm(arr, axis=-1, keepdims=True), eps)


def normalize_torch(vec: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
  return vec / vec.norm(dim=-1, keepdim=True).clamp_min(eps)


def vector_angle_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  a_n = normalize_np(a)
  b_n = normalize_np(b)
  dot = np.clip(np.sum(a_n * b_n, axis=-1), -1.0, 1.0)
  return np.arccos(dot).astype(np.float32)


def contact_geometry_root_projected_gravity(
  env: ManagerBasedRlEnv,
  *,
  state_rows_np: np.ndarray,
  ext_rows_np: np.ndarray,
  foot_projected_gravity_np: np.ndarray,
  contact_prob_np: np.ndarray,
  joint_ids: torch.Tensor,
  foot_body_ids: list[int],
  contact_threshold: float,
  default_base_height: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
  """Convert stance-foot projected gravity predictions into root-frame labels.

  The root is written with neutral projected gravity so the ankle-roll body
  orientation read after FK is the foot orientation relative to the root. A
  foot-frame gravity vector can then be rotated into the root frame through
  that relative orientation.
  """
  state_rows = torch.as_tensor(state_rows_np, dtype=torch.float32, device=env.device)
  ext_rows = torch.as_tensor(ext_rows_np, dtype=torch.float32, device=env.device)
  foot_projected_gravity = torch.as_tensor(
    foot_projected_gravity_np, dtype=torch.float32, device=env.device
  )
  contact_prob = torch.as_tensor(
    contact_prob_np, dtype=torch.float32, device=env.device
  )
  n_rows = state_rows.shape[0]
  env_ids = torch.arange(n_rows, dtype=torch.long, device=env.device)
  row_ids = torch.arange(n_rows, dtype=torch.long, device=env.device)

  fallback_root_gravity = normalize_torch(state_rows[:, 6:9])
  state_neutral = state_rows.clone()
  state_neutral[:, 6:9] = torch.tensor(
    NEUTRAL_PROJECTED_GRAVITY, dtype=state_neutral.dtype, device=state_neutral.device
  )
  ext_neutral = ext_rows.clone()
  if ext_neutral.shape[1] < 5:
    pad = torch.zeros(
      ext_neutral.shape[0],
      5 - ext_neutral.shape[1],
      dtype=ext_neutral.dtype,
      device=ext_neutral.device,
    )
    ext_neutral = torch.cat([ext_neutral, pad], dim=-1)
  ext_neutral[:, 4] = 0.0

  write_recorded_state(
    env,
    env.scene["robot"],
    state_raw=state_neutral,
    ext_raw=ext_neutral,
    row_ids=row_ids,
    env_ids=env_ids,
    joint_ids=joint_ids,
    default_base_height=default_base_height,
    init_height_mode="root_height",
    foot_body_ids=None,
  )
  env.scene.write_data_to_sim()
  env.sim.forward()

  foot_ids = torch.as_tensor(foot_body_ids, dtype=torch.long, device=env.device)
  foot_quat_root = env.scene["robot"].data.body_link_quat_w[env_ids[:, None], foot_ids]
  foot_gravity = normalize_torch(foot_projected_gravity)
  root_candidates = quat_apply(
    foot_quat_root.reshape(-1, 4),
    foot_gravity.reshape(-1, 3),
  ).reshape(n_rows, 2, 3)
  root_candidates = normalize_torch(root_candidates)

  confident = contact_prob >= float(contact_threshold)
  weights = torch.where(
    confident, contact_prob.clamp_min(0.0), torch.zeros_like(contact_prob)
  )
  weight_sum = weights.sum(dim=1)
  used = weight_sum > 0.0
  weighted = (root_candidates * weights[..., None]).sum(dim=1)
  solved = normalize_torch(weighted / weight_sum.clamp_min(1.0e-6)[:, None])
  root_gravity = torch.where(used[:, None], solved, fallback_root_gravity)

  disagreement = vector_angle_rad(
    root_candidates[:, 0].detach().cpu().numpy(),
    root_candidates[:, 1].detach().cpu().numpy(),
  )

  return (
    root_gravity.detach().cpu().numpy().astype(np.float32),
    {
      "fallback_root_projected_gravity": fallback_root_gravity.detach()
      .cpu()
      .numpy()
      .astype(np.float32),
      "left_root_projected_gravity": root_candidates[:, 0]
      .detach()
      .cpu()
      .numpy()
      .astype(np.float32),
      "right_root_projected_gravity": root_candidates[:, 1]
      .detach()
      .cpu()
      .numpy()
      .astype(np.float32),
      "used_stance_orientation": used.detach().cpu().numpy().astype(bool),
      "active_orientation_contact_count": confident.sum(dim=1)
      .detach()
      .cpu()
      .numpy()
      .astype(np.float32),
      "double_stance_orientation_disagreement_rad": disagreement,
    },
  )
