"""Offline compatibility checks for the G1 deployment simulation asset."""

import mujoco
import numpy as np
from deploy.robots.g1_config import G1Config
from deploy.utils.g1_kinematics import G1KinematicsHelper


def test_scene_matches_deployment_joint_and_contact_layout():
  model = mujoco.MjModel.from_xml_path(G1Config._mujoco_xml_paths["default"])

  assert (model.nq, model.nv, model.nu) == (36, 35, 29)
  joint_names = [model.joint(i).name for i in range(1, model.njnt)]
  assert joint_names == G1Config._joint_map_mujoco
  actuator_joint_names = [
    model.joint(int(joint_id)).name for joint_id in model.actuator_trnid[:, 0]
  ]
  assert actuator_joint_names == joint_names
  np.testing.assert_array_equal(model.jnt_qposadr[1:], np.arange(7, 36))
  np.testing.assert_array_equal(model.jnt_dofadr[1:], np.arange(6, 35))

  floor_id = model.geom("floor").id
  for side in ("left", "right"):
    foot_body_id = model.body(f"{side}_ankle_roll_link").id
    for index in range(1, 5):
      geom_id = model.geom(f"{side}_ankle_roll_{index}").id
      assert model.geom_bodyid[geom_id] == foot_body_id
      assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
      assert model.geom_size[geom_id, 0] > 0
      assert (
        model.geom_contype[geom_id] & model.geom_conaffinity[floor_id]
        or model.geom_contype[floor_id] & model.geom_conaffinity[geom_id]
      )


def test_kinematics_helper_accepts_hardware_joint_vector():
  helper = G1KinematicsHelper()
  joint_positions = np.linspace(-0.05, 0.05, G1Config.num_joints)
  initial = helper.compute(joint_positions, base_height=0.8, dt=0.02)
  raised = helper.compute(joint_positions, base_height=0.85, dt=0.02)

  assert initial["foot_heights"].shape == (2,)
  assert np.all(np.isfinite(initial["foot_heights"]))
  np.testing.assert_array_equal(initial["foot_z_vel"], np.zeros(2))
  np.testing.assert_allclose(raised["foot_heights"] - initial["foot_heights"], 0.05)
  np.testing.assert_allclose(raised["foot_z_vel"], 2.5)
  assert raised["base_height"] == 0.85
