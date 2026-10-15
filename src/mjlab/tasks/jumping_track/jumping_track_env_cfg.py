"""Base configuration for reference-conditioned jumping tracking."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.tracking import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def _reference_terms() -> dict[str, ObservationTermCfg]:
  params = {"command_name": "motion"}
  return {
    "motion_anchor_height": ObservationTermCfg(
      func=mdp.motion_anchor_height, params=params
    ),
    "motion_anchor_projected_gravity": ObservationTermCfg(
      func=mdp.motion_anchor_projected_gravity, params=params
    ),
    "motion_joint_pos": ObservationTermCfg(func=mdp.motion_joint_pos, params=params),
    "motion_joint_vel": ObservationTermCfg(func=mdp.motion_joint_vel, params=params),
    "motion_anchor_lin_vel_b": ObservationTermCfg(
      func=mdp.motion_anchor_lin_vel_b, params=params
    ),
    "motion_anchor_ang_vel_b": ObservationTermCfg(
      func=mdp.motion_anchor_ang_vel_b, params=params
    ),
  }


def _replace_command_with_reference_terms(
  terms: dict[str, ObservationTermCfg],
  *,
  remove: tuple[str, ...] = (),
) -> dict[str, ObservationTermCfg]:
  reference_terms = _reference_terms()
  return {
    key: value
    for name, term in terms.items()
    if name not in remove
    for key, value in (
      tuple(reference_terms.items()) if name == "command" else ((name, term),)
    )
  }


def _insert_term_after(
  terms: dict[str, ObservationTermCfg],
  *,
  after: str,
  name: str,
  term: ObservationTermCfg,
) -> dict[str, ObservationTermCfg]:
  """Insert an observation term after an existing term."""
  return {
    key: value
    for existing_name, existing_term in terms.items()
    for key, value in (
      ((existing_name, existing_term), (name, term))
      if existing_name == after
      else ((existing_name, existing_term),)
    )
  }


def apply_jumping_track_observations(cfg: ManagerBasedRlEnvCfg) -> None:
  """Replace tracking's reference command with jumping-track reference terms."""
  actor_terms = _replace_command_with_reference_terms(
    cfg.observations["actor"].terms,
    remove=("motion_anchor_pos_b", "motion_anchor_ori_b", "base_lin_vel"),
  )
  cfg.observations["actor"].terms = _insert_term_after(
    actor_terms,
    after="motion_anchor_ang_vel_b",
    name="projected_gravity",
    term=ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
  )

  critic_terms = _replace_command_with_reference_terms(cfg.observations["critic"].terms)
  cfg.observations["critic"].terms = _insert_term_after(
    critic_terms,
    after="base_lin_vel",
    name="projected_gravity",
    term=ObservationTermCfg(func=envs_mdp.projected_gravity),
  )
