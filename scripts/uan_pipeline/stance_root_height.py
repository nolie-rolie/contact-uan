"""Reconstruct root height from contact-gated stance-foot estimates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.tasks.uan.mdp.utils import write_recorded_state


@dataclass(frozen=True)
class RootHeightSolveCfg:
  contact_threshold: float = 0.5
  default_base_height: float = 0.75


def compute_foot_height_prior_z(
  env,
  *,
  state_rows: torch.Tensor,
  ext_rows: torch.Tensor,
  predicted_foot_heights: torch.Tensor,
  joint_ids: torch.Tensor,
  foot_body_ids: list[int],
  cfg: RootHeightSolveCfg,
) -> torch.Tensor:
  """Compute lowest-foot FK alignment prior from predicted foot heights."""
  n_rows = state_rows.shape[0]
  env_ids = torch.arange(n_rows, dtype=torch.long, device=env.device)
  ext_tmp = ext_rows.clone()
  if ext_tmp.shape[1] < 5:
    pad = torch.zeros(
      ext_tmp.shape[0],
      5 - ext_tmp.shape[1],
      dtype=ext_tmp.dtype,
      device=ext_tmp.device,
    )
    ext_tmp = torch.cat([ext_tmp, pad], dim=-1)
  ext_tmp[:, 0:2] = predicted_foot_heights
  row_ids = torch.arange(n_rows, dtype=torch.long, device=env.device)
  write_recorded_state(
    env,
    env.scene["robot"],
    state_raw=state_rows,
    ext_raw=ext_tmp,
    row_ids=row_ids,
    env_ids=env_ids,
    joint_ids=joint_ids,
    default_base_height=cfg.default_base_height,
    init_height_mode="lowest_foot",
    foot_body_ids=foot_body_ids,
  )
  env.scene.write_data_to_sim()
  env.sim.forward()
  return env.scene["robot"].data.root_link_pos_w[:n_rows, 2].clone()


def compute_contact_geometry_root_z(
  env,
  *,
  state_rows: torch.Tensor,
  ext_rows: torch.Tensor,
  predicted_foot_heights: torch.Tensor,
  contact_prob: torch.Tensor,
  joint_ids: torch.Tensor,
  foot_body_ids: list[int],
  cfg: RootHeightSolveCfg,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  """Compute root height from confident stance-foot geometry.

  Each foot gives an independent root-height candidate:

      root_z_i = predicted_foot_height_i - foot_fk_z_i(root_z=0)

  Rows with no confident stance foot fall back to the existing lowest-foot
  geometry prior.
  """
  n_rows = state_rows.shape[0]
  env_ids = torch.arange(n_rows, dtype=torch.long, device=env.device)
  fallback_z = compute_foot_height_prior_z(
    env,
    state_rows=state_rows,
    ext_rows=ext_rows,
    predicted_foot_heights=predicted_foot_heights,
    joint_ids=joint_ids,
    foot_body_ids=foot_body_ids,
    cfg=cfg,
  )

  ext_zero = ext_rows.clone()
  if ext_zero.shape[1] < 5:
    pad = torch.zeros(
      ext_zero.shape[0],
      5 - ext_zero.shape[1],
      dtype=ext_zero.dtype,
      device=ext_zero.device,
    )
    ext_zero = torch.cat([ext_zero, pad], dim=-1)
  ext_zero[:, 4] = 0.0
  row_ids = torch.arange(n_rows, dtype=torch.long, device=env.device)
  write_recorded_state(
    env,
    env.scene["robot"],
    state_raw=state_rows,
    ext_raw=ext_zero,
    row_ids=row_ids,
    env_ids=env_ids,
    joint_ids=joint_ids,
    default_base_height=cfg.default_base_height,
    init_height_mode="root_height",
    foot_body_ids=None,
  )
  env.scene.write_data_to_sim()
  env.sim.forward()

  foot_ids = torch.as_tensor(foot_body_ids, dtype=torch.long, device=env.device)
  foot_fk_z_at_zero = env.scene["robot"].data.body_link_pos_w[
    env_ids[:, None], foot_ids, 2
  ]
  root_candidates = predicted_foot_heights - foot_fk_z_at_zero
  confident = contact_prob >= cfg.contact_threshold
  weights = torch.where(
    confident, contact_prob.clamp_min(0.0), torch.zeros_like(contact_prob)
  )
  weight_sum = weights.sum(dim=1)
  used_stance_geometry = weight_sum > 0.0
  weighted_z = (root_candidates * weights).sum(dim=1) / weight_sum.clamp_min(1.0e-6)
  solved_z = torch.where(used_stance_geometry, weighted_z, fallback_z)
  double_stance_disagreement = (root_candidates[:, 0] - root_candidates[:, 1]).abs()

  diagnostics = {
    "fallback_root_z": fallback_z,
    "left_root_z": root_candidates[:, 0],
    "right_root_z": root_candidates[:, 1],
    "solved_root_z": solved_z,
    "used_stance_geometry": used_stance_geometry,
    "active_contact_count": confident.sum(dim=1).to(torch.float32),
    "double_stance_disagreement": double_stance_disagreement,
  }
  return solved_z, diagnostics
