"""Model assembly and kinematics for every configured arm."""

from __future__ import annotations

import numpy as np
import pytest

from vrcm.rcm import angle_between, rcm_error, rcm_length
from vrcm.specs import load_robot_spec

from helpers import random_configurations


def test_instrument_is_attached_to_the_flange(robot):
    """The tip must sit 0.30 m along the shaft from the instrument base."""
    kin = robot.kinematics()
    np.testing.assert_allclose(kin.d, (kin.p_tip - kin.p_base) / np.linalg.norm(kin.p_tip - kin.p_base), atol=1e-9)
    assert np.linalg.norm(kin.p_tip - kin.p_base) == pytest.approx(
        robot.spec.instrument.nominal_length, abs=1e-6
    )


def test_arm_joints_exclude_the_instrument_jaws(robot):
    assert robot.n_arm_joints == len(robot.spec.home_qpos)
    assert robot.model.nq == robot.n_arm_joints + 2  # jaw_a + jaw_b
    assert all("jaw" not in name for name in robot.arm_joint_names)


def test_home_configuration_sits_on_the_rcm_axis(robot):
    spec = load_robot_spec(robot.spec.name)
    robot.reset()
    kin = robot.kinematics()
    p_rcm = np.asarray(spec.trocar)
    assert np.linalg.norm(rcm_error(kin.p_tip, kin.d, p_rcm)) < 1e-5
    assert rcm_length(kin.p_tip, kin.d, p_rcm) == pytest.approx(spec.scene.home_insertion, abs=1e-5)
    # the instrument looks straight down into the cavity at the home pose
    assert angle_between(kin.d, np.array([0.0, 0.0, -1.0])) < np.deg2rad(0.5)
    assert robot.in_limits(kin.q)


def test_tip_jacobian_matches_finite_differences(robot):
    rng = np.random.default_rng(0)
    for q in random_configurations(robot, 3, seed=5, margin=0.4):
        kin = robot.kinematics(q)
        eps = 1e-7
        for i in range(robot.n_arm_joints):
            qp_ = q.copy()
            qp_[i] += eps
            qm_ = q.copy()
            qm_[i] -= eps
            kp_, km_ = robot.kinematics(qp_), robot.kinematics(qm_)
            np.testing.assert_allclose((kp_.p_tip - km_.p_tip) / (2 * eps), kin.Jv[:, i], atol=1e-6)
            dR = (kp_.R_tip - km_.R_tip) / (2 * eps)
            W = dR @ kin.R_tip.T
            omega = np.array([W[2, 1], W[0, 2], W[1, 0]])
            np.testing.assert_allclose(omega, kin.Jw[:, i], atol=1e-6)


def test_joint_limits_are_enforced_by_the_model(robot):
    assert robot.n_arm_joints == len(robot.joint_lower) == len(robot.joint_upper)
    assert np.all(robot.joint_lower < robot.joint_upper)
    assert (robot.joint_velocity_limits > 0).all()
    assert robot.in_limits(robot.spec.home_qpos)


def test_ik_recovers_a_reachable_pose(robot):
    """IK must recover poses close to the home configuration, where it started."""
    from vrcm.ik import ik_shaft

    rng = np.random.default_rng(7)
    q_home = np.asarray(robot.spec.home_qpos, dtype=float)
    solved = 0
    for _ in range(6):
        q_target = q_home + rng.uniform(-0.15, 0.15, size=robot.n_arm_joints)
        kin = robot.kinematics(q_target)
        res = ik_shaft(robot, kin.p_tip, kin.d, q0=q_home, max_iter=400)
        solved += int(res.converged)
    assert solved >= 5


def test_ik_multistart_solves_a_far_away_target(robot):
    """The multi-start solver handles a target in a different IK branch."""
    from vrcm.ik import solve_ik_multistart

    rng = np.random.default_rng(3)
    q_target = rng.uniform(robot.joint_lower + 0.4, robot.joint_upper - 0.4)
    kin = robot.kinematics(q_target)
    res = solve_ik_multistart(robot, kin.p_tip, kin.d, restarts=12, seed=1)
    assert res.converged, f"pos error {res.pos_error:.2e} m, shaft error {res.rot_error:.2e} rad"
    assert robot.in_limits(res.q)
