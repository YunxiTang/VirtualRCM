#!/usr/bin/env python
"""Browser-based viewer for the virtual RCM controller (viser + sliders).

Runs the V-RCM control loop and streams the MuJoCo scene to a web page, where
sliders set the goal viewpoint:

==========  ==================================================
control     action
==========  ==================================================
pitch       goal pitch of the shaft about the trocar [deg]
yaw         goal yaw of the shaft about the trocar [deg]
depth       goal insertion depth, trocar -> tip [mm]
Home        slide back to the nominal viewpoint
Hold        stop at the current reference (sliders jump to it)
Paused      freeze the controller
==========  ==================================================

A live plot under the controls shows the RCM error over the last
``--plot-window`` seconds of simulated time.

The planner rate-limits every slider move, so dragging a slider produces a smooth
motion that the controller tracks while keeping the shaft through the trocar.

The RCM frame is drawn at the trocar: a sphere at the RCM point, a triad in the
planner's shaft frame (z along the shaft, x/y the pitch/yaw axes) and a magenta
segment from the trocar to the closest point on the shaft axis (the RCM error,
only visible when it is non-zero).

    python scripts/demo_viser.py ur5e --mode dynamic      # then open http://localhost:8080

Needs ``pip install viser``; unlike the MuJoCo viewer it needs no display or
``mjpython`` and works over SSH with port forwarding.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demo_viewer import build_session, control_step  # noqa: E402
from vrcm.rcm import rcm_error, shaft_frame  # noqa: E402

VISUAL_GROUPS = (0, 1, 2)  # group 3 holds the collision geoms
RCM_AXIS_LENGTH = 0.04  # RCM frame triad [m]
RCM_AXIS_RADIUS = 0.0015
RCM_ERROR_MIN = 1e-5  # draw the error segment above this offset [m]


def mat_to_wxyz(xmat: np.ndarray) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.asarray(xmat, dtype=float).reshape(9))
    return quat


def geom_mesh(model: mujoco.MjModel, i: int) -> trimesh.Trimesh | None:
    """Triangle mesh of geom ``i`` in its own frame, or ``None`` if not drawable."""
    size = model.geom_size[i]
    kind = model.geom_type[i]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        m = model.geom_dataid[i]
        v0, nv = model.mesh_vertadr[m], model.mesh_vertnum[m]
        f0, nf = model.mesh_faceadr[m], model.mesh_facenum[m]
        return trimesh.Trimesh(model.mesh_vert[v0:v0 + nv], model.mesh_face[f0:f0 + nf], process=False)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * size)
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=size[0], height=2 * size[1])
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(radius=size[0], height=2 * size[1])
    if kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        sphere = trimesh.creation.icosphere(subdivisions=2)
        sphere.apply_scale(size if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID else size[0])
        return sphere
    return None  # planes are drawn as a grid, heightfields are not used


def geom_rgba(model: mujoco.MjModel, i: int) -> np.ndarray:
    mat = model.geom_matid[i]
    return model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[i]


class MujocoScene:
    """Mirror the visual geoms of a MuJoCo model into a viser scene."""

    def __init__(self, server: viser.ViserServer, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.data = data
        self.handles: dict[int, viser.MeshHandle] = {}
        for i in range(model.ngeom):
            if model.geom_group[i] not in VISUAL_GROUPS:
                continue
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
                extent = 2 * max(model.geom_size[i][:2].max(), 1.0)
                server.scene.add_grid("/mujoco/floor", width=extent, height=extent,
                                      position=data.geom_xpos[i])
                continue
            mesh = geom_mesh(model, i)
            if mesh is None:
                continue
            rgba = geom_rgba(model, i)
            handle = server.scene.add_mesh_simple(
                f"/mujoco/geom_{i}", mesh.vertices, mesh.faces,
                color=tuple(rgba[:3]), opacity=float(rgba[3]) if rgba[3] < 1 else None,
                position=data.geom_xpos[i], wxyz=mat_to_wxyz(data.geom_xmat[i]),
            )
            # only geoms that can move need per-frame pose updates
            if model.body_weldid[model.geom_bodyid[i]] != 0:
                self.handles[i] = handle

    def update(self) -> None:
        for i, handle in self.handles.items():
            handle.position = self.data.geom_xpos[i]
            handle.wxyz = mat_to_wxyz(self.data.geom_xmat[i])


class RcmMarkers:
    """Trocar sphere, shaft-frame triad and RCM error segment."""

    def __init__(self, server: viser.ViserServer, model: mujoco.MjModel, trocar: np.ndarray) -> None:
        self.trocar = trocar
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rcm")
        radius = float(model.site_size[site][0]) if site >= 0 else 0.006
        server.scene.add_icosphere("/rcm/point", radius=radius, color=(255, 50, 50), opacity=0.6,
                                   position=trocar)
        self.frame = server.scene.add_frame("/rcm/frame", axes_length=RCM_AXIS_LENGTH,
                                            axes_radius=RCM_AXIS_RADIUS, position=trocar)
        self.error = server.scene.add_line_segments(
            "/rcm/error", np.zeros((1, 2, 3)), colors=(255, 0, 255), thickness=3.0,
            thickness_units="screen", visible=False)

    def update(self, p_tip: np.ndarray, d: np.ndarray) -> None:
        frame = shaft_frame(d)
        self.frame.wxyz = mat_to_wxyz(frame)
        e = rcm_error(p_tip, frame[:, 2], self.trocar)
        visible = bool(np.linalg.norm(e) > RCM_ERROR_MIN)
        if visible:
            self.error.points = np.array([[self.trocar, self.trocar + e]])
        self.error.visible = visible


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("robot", nargs="?", default="ur5e")
    parser.add_argument("--mode", default="kinematic", choices=["kinematic", "dynamic"])
    parser.add_argument("--control-rate", type=float, default=100.0)
    parser.add_argument("--render-rate", type=float, default=30.0, help="scene updates per second")
    parser.add_argument("--plot-rate", type=float, default=10.0, help="RCM error plot updates per second")
    parser.add_argument("--plot-window", type=float, default=10.0, help="RCM error plot history [s]")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    sim, planner = build_session(args.robot, args.mode, args.control_rate)
    lock = threading.Lock()  # GUI callbacks run on viser's thread

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    scene = MujocoScene(server, sim.model, sim.data)
    markers = RcmMarkers(server, sim.model, sim.trocar)

    depth_home = planner.goal_depth
    with server.gui.add_folder("Viewpoint goal"):
        pitch = server.gui.add_slider("pitch [deg]", min=-np.degrees(planner.pitch_limit),
                                      max=np.degrees(planner.pitch_limit), step=0.5, initial_value=0.0)
        yaw = server.gui.add_slider("yaw [deg]", min=-np.degrees(planner.yaw_limit),
                                    max=np.degrees(planner.yaw_limit), step=0.5, initial_value=0.0)
        depth = server.gui.add_slider("depth [mm]", min=planner.depth_limits[0] * 1e3,
                                      max=planner.depth_limits[1] * 1e3, step=1.0,
                                      initial_value=depth_home * 1e3)
        home = server.gui.add_button("Home")
        hold = server.gui.add_button("Hold")
        paused = server.gui.add_checkbox("Paused", initial_value=False)
    with server.gui.add_folder("Status"):
        rcm_mm = server.gui.add_number("RCM error [mm]", initial_value=0.0, disabled=True)
        depth_mm = server.gui.add_number("depth [mm]", initial_value=depth_home * 1e3, disabled=True)
        ref_deg = server.gui.add_vector2("pitch / yaw ref [deg]", initial_value=(0.0, 0.0), disabled=True)
    history = deque(maxlen=max(int(args.plot_window * args.control_rate), 2))  # (t [s], error [mm])
    rcm_plot = server.gui.add_uplot(
        data=(np.zeros(0), np.zeros(0)),
        series=({"label": "t [s]"}, {"label": "RCM error [mm]", "stroke": "magenta", "width": 1.5}),
        title="RCM error",
        scales={"x": {"time": False}},
        aspect=1.6,
    )

    def set_goal(_=None) -> None:
        with lock:
            planner.goto(np.deg2rad(pitch.value), np.deg2rad(yaw.value), depth.value * 1e-3)

    def set_sliders(p: float, y: float, dep: float) -> None:
        pitch.value, yaw.value, depth.value = p, y, dep  # each assignment fires set_goal

    for slider in (pitch, yaw, depth):
        slider.on_update(set_goal)
    home.on_click(lambda _: set_sliders(0.0, 0.0, depth_home * 1e3))
    hold.on_click(lambda _: set_sliders(np.degrees(planner.pitch), np.degrees(planner.yaw), planner.depth * 1e3))

    dt = sim.config.control_dt
    render_every = max(int(round(args.control_rate / args.render_rate)), 1)
    plot_every = max(int(round(args.control_rate / args.plot_rate)), 1)
    tip = sim.robot.tip_site_id
    print(f"running {args.robot} in {args.mode} mode; open http://localhost:{args.port}")
    try:
        tick = 0
        while True:
            t0 = time.perf_counter()
            if not paused.value:
                with lock:
                    result = control_step(sim, planner, dt)
                history.append((sim.data.time, result.rcm_error_norm * 1e3))
                if tick % render_every == 0:
                    with server.atomic():
                        scene.update()
                        markers.update(sim.data.site_xpos[tip], sim.data.site_xmat[tip].reshape(3, 3)[:, 2])
                    rcm_mm.value = result.rcm_error_norm * 1e3
                    depth_mm.value = result.length * 1e3
                    ref_deg.value = (np.degrees(planner.pitch), np.degrees(planner.yaw))
                if tick % plot_every == 0:
                    rcm_plot.data = tuple(np.array(history).T)
                tick += 1
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
