"""RCM geometry and the linearised RCM constraint."""

from __future__ import annotations

import numpy as np
import pytest

from vrcm.rcm import (
    angle_between,
    direction_error,
    direction_from_angles,
    insertion_jacobian,
    projector,
    rcm_error,
    rcm_jacobian,
    rcm_length,
    shaft_frame,
    shaft_point,
    skew,
    unit,
)

from helpers import random_configurations


def test_skew_matches_cross_product():
    rng = np.random.default_rng(0)
    for _ in range(10):
        a, b = rng.normal(size=3), rng.normal(size=3)
        np.testing.assert_allclose(skew(a) @ b, np.cross(a, b))


def test_rcm_error_is_zero_on_the_axis():
    p_rcm = np.array([0.4, 0.0, 0.4])
    d = unit(np.array([0.1, -0.2, -1.0]))
    for length in (0.05, 0.15, 0.3):
        p_tip = p_rcm + length * d
        assert np.linalg.norm(rcm_error(p_tip, d, p_rcm)) < 1e-12
        assert rcm_length(p_tip, d, p_rcm) == pytest.approx(length)
        np.testing.assert_allclose(shaft_point(p_tip, d, p_rcm), p_rcm, atol=1e-12)


def test_rcm_error_is_perpendicular_offset():
    p_rcm = np.zeros(3)
    d = np.array([0.0, 0.0, -1.0])
    p_tip = np.array([0.03, -0.02, 0.15])
    e = rcm_error(p_tip, d, p_rcm)
    assert e[2] == pytest.approx(0.0)
    np.testing.assert_allclose(e, [0.03, -0.02, 0.0])
    assert np.dot(e, d) == pytest.approx(0.0)


def test_projector_is_idempotent():
    d = unit(np.array([1.0, 2.0, -0.5]))
    P = projector(d)
    np.testing.assert_allclose(P @ P, P, atol=1e-12)
    np.testing.assert_allclose(P @ d, np.zeros(3), atol=1e-12)


def test_rcm_jacobian_derivative_matches_rcm_error_derivative():
    """``d/dt e_RCM ~= J_RCM qdot`` -- the heart of the controller."""
    from vrcm.robot import Robot
    from vrcm.specs import load_robot_spec

    rng = np.random.default_rng(1)
    eps = 1e-6
    for name in ("ur5e", "flexiv_rizon4"):
        robot = Robot(load_robot_spec(name))
        p_rcm = np.asarray(robot.spec.trocar)
        # the relation holds on the constraint manifold, i.e. with e_RCM = 0
        q = np.asarray(robot.spec.home_qpos, dtype=float)
        kin = robot.kinematics(q)
        e_home = rcm_error(kin.p_tip, kin.d, p_rcm)
        np.testing.assert_allclose(e_home, 0.0, atol=1e-6)
        l = rcm_length(kin.p_tip, kin.d, p_rcm)
        J = rcm_jacobian(kin.Jv, kin.Jw, kin.d, l)
        for _ in range(8):
            qdot = rng.normal(size=robot.n_arm_joints)
            kin_new = robot.kinematics(q + eps * qdot)
            e_new = rcm_error(kin_new.p_tip, kin_new.d, p_rcm)
            # note: the home configuration is on the manifold to ~0.1 um, so the
            # constant part e_home has to be removed before differencing
            predicted = (e_new - e_home) / eps
            np.testing.assert_allclose(predicted, J @ qdot, atol=2e-5)

        # away from the manifold the same relation holds to first order as well
        for q in random_configurations(robot, 3, seed=3):
            kin = robot.kinematics(q)
            l = rcm_length(kin.p_tip, kin.d, p_rcm)
            J = rcm_jacobian(kin.Jv, kin.Jw, kin.d, l)
            e_old = rcm_error(kin.p_tip, kin.d, p_rcm)
            qdot = rng.normal(size=robot.n_arm_joints)
            kin_new = robot.kinematics(q + eps * qdot)
            e_new = rcm_error(kin_new.p_tip, kin_new.d, p_rcm)
            predicted = (e_new - e_old) / eps
            np.testing.assert_allclose(predicted, J @ qdot, atol=1e-4)


def test_insertion_jacobian_matches_finite_difference():
    from vrcm.robot import Robot
    from vrcm.specs import load_robot_spec

    rng = np.random.default_rng(4)
    robot = Robot(load_robot_spec("ur5e"))
    q = robot.spec.home_qpos
    kin = robot.kinematics(q)
    row = insertion_jacobian(kin.Jv, kin.d)
    qdot = rng.normal(size=robot.n_arm_joints)
    eps = 1e-6
    kin_p = robot.kinematics(np.asarray(q) + eps * qdot)
    kin_m = robot.kinematics(np.asarray(q) - eps * qdot)
    # d^T J_v qdot is the rate at which the tip moves along the *current* shaft
    # direction, i.e. the insertion velocity v_ins of method.md section 5
    fd = float(kin.d @ ((kin_p.p_tip - kin_m.p_tip) / (2 * eps)))
    assert row @ qdot == pytest.approx(fd, abs=1e-7)


def test_direction_error_rotates_towards_the_target():
    """omega = d x d_des must move d towards d_des (method.md section 6 sign)."""
    d = np.array([0.0, 0.0, -1.0])
    d_des = unit(np.array([0.1, 0.0, -1.0]))
    omega = direction_error(d_des, d)
    d_new = unit(d + 1e-3 * np.cross(omega, d))
    assert angle_between(d_new, d_des) < angle_between(d, d_des)


def test_shaft_frame_axes_are_orthonormal():
    d = unit(np.array([0.2, -0.3, -0.9]))
    F = shaft_frame(d)
    np.testing.assert_allclose(F.T @ F, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(F[:, 2], d, atol=1e-12)


def test_direction_from_angles_recovers_the_nominal_direction():
    d = unit(np.array([0.0, 0.0, -1.0]))
    np.testing.assert_allclose(direction_from_angles(d, 0.0, 0.0), d, atol=1e-12)
    for pitch, yaw in [(0.2, 0.0), (0.0, -0.3), (0.15, 0.1)]:
        moved = direction_from_angles(d, pitch, yaw)
        assert angle_between(moved, d) > 1e-3
