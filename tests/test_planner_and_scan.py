"""Viewpoint planner and stop-and-go viewpoint search."""

from __future__ import annotations

import numpy as np
import pytest

from vrcm.planner import ViewpointAction, ViewpointPlanner
from vrcm.rcm import angle_between
from vrcm.sim import SimConfig, VrcmSim
from vrcm.specs import load_robot_spec
from vrcm.visibility import VisibilityModel, stop_and_go_scan


def test_planner_rate_limits_the_reference():
    planner = ViewpointPlanner(d_nominal=np.array([0.0, 0.0, -1.0]), depth_nominal=0.15,
                               max_pitch_rate=np.deg2rad(20), max_depth_rate=0.02)
    planner.command(ViewpointAction(pitch=np.deg2rad(30), insertion=0.1))
    dt = 0.01
    ref = planner.step(dt)
    assert ref.pitch == pytest.approx(np.deg2rad(0.2), abs=1e-9)
    assert ref.insertion_velocity == pytest.approx(0.02, rel=1e-9)
    assert not ref.settled


def test_planner_clamps_to_the_surgical_cone_and_depth_range():
    planner = ViewpointPlanner(d_nominal=np.array([0.0, 0.0, -1.0]), depth_nominal=0.15,
                               pitch_limit=np.deg2rad(20), depth_limits=(0.05, 0.2))
    planner.command(ViewpointAction(pitch=np.deg2rad(90), insertion=1.0))
    for _ in range(2000):
        ref = planner.step(0.01)
    assert ref.pitch == pytest.approx(np.deg2rad(20))
    assert ref.insertion_depth == pytest.approx(0.2)
    assert ref.settled


def test_planner_direction_moves_towards_the_goal():
    d0 = np.array([0.0, 0.0, -1.0])
    planner = ViewpointPlanner(d_nominal=d0, depth_nominal=0.15)
    planner.command(ViewpointAction(pitch=np.deg2rad(15), yaw=np.deg2rad(15)))
    start = planner.step(0.01).d_des
    for _ in range(500):
        ref = planner.step(0.01)
    assert angle_between(ref.d_des, start) > np.deg2rad(10)
    assert angle_between(ref.d_des, d0) < np.deg2rad(30)


def test_visibility_model_peaks_on_axis():
    target = np.array([0.5, 0.1, 0.3])
    model = VisibilityModel(target=target, ideal_distance=0.1, fov_deg=70)
    p_tip = target - 0.1 * np.array([0.0, 0.0, -1.0])
    on_axis = model.score(p_tip, np.array([0.0, 0.0, -1.0]))
    off_axis = model.score(p_tip, np.array([1.0, 0.0, 0.0]))
    assert on_axis > off_axis
    assert 0.0 <= off_axis <= on_axis <= 1.0


def test_stop_and_go_scan_improves_the_viewpoint():
    spec = load_robot_spec("ur5e")
    sim = VrcmSim(spec, sim_config=SimConfig(mode="kinematic", control_dt=0.01))
    sim.reset()
    state = sim.state()
    planner = ViewpointPlanner(d_nominal=state["d"], depth_nominal=float(state["length"]))
    visibility = VisibilityModel(
        target=np.asarray(spec.trocar) + np.array([0.05, 0.04, -0.18]),
        fov_deg=80.0, ideal_distance=0.09, distance_sigma=0.05,
    )
    start_score = visibility.score(state["p_tip"], state["d"])
    result = stop_and_go_scan(sim, planner, visibility, pitch_span=np.deg2rad(18),
                              yaw_span=np.deg2rad(18), depth_span=0.02, coarse=3, fine=3)
    assert result.best_viewpoint is not None
    assert result.best_score > start_score
    # the best viewpoint must be reachable through the V-RCM controller
    assert sim.rcm_error_norm < 5e-4

    final = sim.robot.kinematics()
    # the achieved viewpoint matches the selected one up to the residual tracking
    # error of the V-RCM controller
    assert visibility.score(final.p_tip, final.d) == pytest.approx(result.best_score, rel=5e-3)
