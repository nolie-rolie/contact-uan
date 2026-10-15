"""Fixed-objective JumpingHeight configurations for Unitree G1."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.jumping_height.jumping_height_env_cfg import (
  apply_jumping_height_objective,
)
from mjlab.tasks.jumping_track.config.g1.env_cfgs import (
  unitree_g1_jumping_track_baseline_env_cfg,
  unitree_g1_jumping_track_uan_env_cfg,
)


def _add_feet_ground_contact_sensor(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add the found-only contact sensor used to confirm two-foot takeoff."""
  sensors = cfg.scene.sensors or ()
  if any(sensor.name == "feet_ground_contact" for sensor in sensors):
    raise ValueError("JumpingHeight already has a feet_ground_contact sensor.")
  cfg.scene.sensors = sensors + (
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(
        mode="subtree",
        pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
  )


def unitree_g1_jumping_height_baseline_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the fixed-objective JumpingHeight baseline."""
  cfg = unitree_g1_jumping_track_baseline_env_cfg(play=play)
  _add_feet_ground_contact_sensor(cfg)
  apply_jumping_height_objective(cfg)
  return cfg


def unitree_g1_jumping_height_uan_env_cfg(
  play: bool = False,
  uan_checkpoint_path: str | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create fixed-objective JumpingHeight with a frozen UAN actuator model."""
  cfg = unitree_g1_jumping_track_uan_env_cfg(
    play=play,
    uan_checkpoint_path=uan_checkpoint_path,
  )
  _add_feet_ground_contact_sensor(cfg)
  apply_jumping_height_objective(cfg)
  return cfg
