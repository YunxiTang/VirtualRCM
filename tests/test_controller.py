"""Closed-loop behaviour of the V-RCM controller."""

from __future__ import annotations

import numpy as np
import pytest

from vrcm.controller import VrcmConfig, VrcmTarget
from vrcm.planner import ViewpointAction, ViewpointPlanner
from vrcm.rcm import angle_between, rcm_error, rcm_length
from vrcm.sim import SimConfig, VrcmSim
from vrcm.specs import load_robot_spec


def make_sim(name: str, mode: str = "kinematic", **cfg) -> VrcmSim:
    spec = load_robot_spec(name)
    controller = VrcmConfig(k_rcm=2.0, k_orient=3.0, k_insertion=2.0, lambda_reg=1e-4, **cfg)
    sim = VrcmSim(spec, controller, SimConfig(mode=mode, control_dt=0.01))
    sim.reset()
    return sim


@pytest.mark.parametrize("mode", ["kinematic", "dynamic"])
def test_rcm_is_enforced_during_a_full_manipulation(robot, mode):
    sim = make_sim(robot.spec.name, mode=mode)
    state = sim.state()
    planner = ViewpointPlanner(d_nominal=state["d"], depth_nominal=float(state["length"]))
    actions = [
        ViewpointAction(pitch=np.deg2rad(8), yaw=np.deg2rad(-6), insertion=0.015),
        ViewpointAction(pitch=np.deg2rad(-14), yaw=np.deg2rad(11), insertion=0.02),
        ViewpointAction(insertion=-0.03),
    ]
    log = None
    for action in actions:
        planner.command(action)
        log = sim.run(planner, 2.0, log=log)
    assert log is not None
    summary = log.summary()

    tolerance = 0.5 if mode == "kinematic" else 6.0  # [mm]
    assert summary["rcm_error_max_mm"] < tolerance

    # the viewpoint and the insertion depth must actually be reached
    assert angle_between(log.d[-1], log.d_des[-1]) < np.deg2rad(1.0)
    assert log.length[-1] == pytest.approx(log.depth_target[-1], abs=1e-3)


def test_rcm_error_feedback_pulls_a_displaced_shaft_back(robot):
    """Equation (8): a non-zero RCM error is driven back to zero."""
    sim = make_sim(robot.spec.name)
    # displace the shaft so that the trocar is 5 mm off the instrument axis
    q = np.asarray(robot.spec.home_qpos).copy()
    q[0] += 0.01
    sim.reset(q)
    start = sim.rcm_error_norm
    assert start > 1e-3

    planner = ViewpointPlanner(d_nominal=sim.state()["d"], depth_nominal=sim.insertion_depth)
    sim.run(planner, 4.0)
    assert sim.rcm_error_norm < 1e-4


def test_joint_limits_are_never_violated(robot):
    sim = make_sim(robot.spec.name)
    state = sim.state()
    planner = ViewpointPlanner(d_nominal=state["d"], depth_nominal=float(state["length"]),
                               max_pitch_rate=np.deg2rad(120), max_depth_rate=0.3)
    # ask for an aggressive action that would hit the limits if unconstrained
    planner.command(ViewpointAction(pitch=np.deg2rad(35), yaw=np.deg2rad(35), insertion=0.2))
    log = sim.run(planner, 3.0)
    q = np.asarray(log.q)
    assert np.all(q >= robot.joint_lower[None, :] - 1e-9)
    assert np.all(q <= robot.joint_upper[None, :] + 1e-9)
    assert np.linalg.norm(np.asarray(log.e_rcm), axis=1).max() < 1e-3


def test_insertion_limits_are_respected(robot):
    sim = make_sim(robot.spec.name)
    ins = robot.spec.instrument
    state = sim.state()
    planner = ViewpointPlanner(d_nominal=state["d"], depth_nominal=float(state["length"]),
                               depth_limits=(ins.min_insertion, ins.max_insertion), max_depth_rate=0.5)
    planner.command(ViewpointAction(insertion=1.0))  # ask for far more than allowed
    log = sim.run(planner, 5.0)
    lengths = np.asarray(log.length)
    assert lengths.max() <= ins.max_insertion + 1e-3
    assert lengths.min() >= ins.min_insertion - 1e-3


def test_guard_stops_motion_when_the_rcm_error_is_too_large(robot):
    sim = make_sim(robot.spec.name, max_rcm_error=1e-4)
    q = np.asarray(robot.spec.home_qpos).copy()
    q[0] += 0.01  # ~5 mm RCM error, above the guard
    sim.reset(q)
    result = sim.step(VrcmTarget(d_des=sim.state()["d"], insertion_velocity=0.0))
    assert not result.converged
    np.testing.assert_allclose(result.qdot, 0.0, atol=1e-12)
    assert any("guard" in note for note in result.notes)


def test_controller_reports_diagnostics(robot):
    sim = make_sim(robot.spec.name)
    result = sim.step(VrcmTarget(d_des=sim.state()["d"], insertion_velocity=0.01))
    assert result.backend in sim.controller.available_backends()
    assert result.iterations >= 1
    assert result.qdot.shape == (robot.n_arm_joints,)
    assert np.all(np.isfinite(result.qdot))
