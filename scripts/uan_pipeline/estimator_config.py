"""Shared configuration for the six Contact-UAN state estimators."""

HISTORY_LENGTH = 3
PROPRIO_DIM = 93
PROPRIO_TERMS = (
  "ang_vel_b",
  "proj_grav",
  "joint_pos_abs",
  "joint_vel",
  "prev_q_des",
)
PROPRIO_ACTION_REPRESENTATION = "prev_q_des_absolute_joint_target"

COMPONENT_TARGETS = {
  "base_lin": ("lin_vel_x", "lin_vel_y", "lin_vel_z"),
  "foot_contact": ("left_foot_contact", "right_foot_contact"),
  "left_stance_foot_height": ("left_foot_height",),
  "right_stance_foot_height": ("right_foot_height",),
  "left_stance_foot_projected_gravity": (
    "left_foot_proj_grav_x",
    "left_foot_proj_grav_y",
    "left_foot_proj_grav_z",
  ),
  "right_stance_foot_projected_gravity": (
    "right_foot_proj_grav_x",
    "right_foot_proj_grav_y",
    "right_foot_proj_grav_z",
  ),
}

COMPONENT_HIDDEN_DIMS = {
  name: (128, 128) if name == "foot_contact" else (256, 256, 256)
  for name in COMPONENT_TARGETS
}

COMPONENT_SUBDIRS = {
  "base_lin": "base_lin",
  "foot_contact": "foot_contact",
  "left_stance_foot_height": "stance_foot_height/left_stance_foot_height",
  "right_stance_foot_height": "stance_foot_height/right_stance_foot_height",
  "left_stance_foot_projected_gravity": (
    "stance_foot_projected_gravity/left_stance_foot_projected_gravity"
  ),
  "right_stance_foot_projected_gravity": (
    "stance_foot_projected_gravity/right_stance_foot_projected_gravity"
  ),
}
