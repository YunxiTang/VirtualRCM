#!/usr/bin/env python
"""Interactive MuJoCo viewer for the virtual RCM controller.

Keyboard control (focus the viewer window).  The MuJoCo viewer already owns every
letter (render/visualisation flags), digits ``0``-``5`` (geom groups), ``Left`` /
``Right`` (step back / forward), ``Page Up``, ``Tab``, ``[ ]``, ``+ -`` and ``F1``-``F5``,
so the demo only uses keys it leaves free, all present on a Mac keyboard:

==================  ==================================================
key                 action
==================  ==================================================
``Up`` / ``Down``   pitch the shaft forward / backward by 2 deg
``6`` / ``7``       yaw the shaft left / right by 2 deg
``8`` / ``9``       insert / retract by 5 mm
``Enter``           return to the trocar home viewpoint
``.``               stop tracking the current goal (hold position)
``F6``              toggle showing the RCM error in the console (``fn+F6`` on a Mac)
``Space``           pause / resume the controller (same as the viewer's own pause)
==================  ==================================================

The RCM frame is drawn in the scene: the ``rcm`` site at the trocar point, a triad
in the planner's shaft frame (z along the shaft, x/y the pitch/yaw axes), and a
magenta segment from the trocar to the closest point on the shaft axis (the RCM error, only visible when it is non-zero).

The script drives the simulation itself and renders through a *passive* MuJoCo
viewer handle, so the control loop runs at exactly ``--control-rate`` Hz.  Use
``--mode dynamic`` to see the actuator tracking error and the RCM feedback term
at work.

    python scripts/demo_viewer.py ur5e --mode dynamic

This script needs a display and the MuJoCo viewer package; it is not exercised
by the unit tests.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sysconfig
import sys
import time
from pathlib import Path

import glfw  # installed with the MuJoCo viewer; only its key-code constants are used
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrcm.controller import VrcmConfig, VrcmTarget  # noqa: E402
from vrcm.planner import ViewpointAction, ViewpointPlanner  # noqa: E402
from vrcm.rcm import rcm_error, shaft_frame  # noqa: E402
from vrcm.scene import RCM_SITE_GROUP  # noqa: E402
from vrcm.sim import SimConfig, VrcmSim  # noqa: E402
from vrcm.specs import load_robot_spec  # noqa: E402


def relaunch_under_mjpython() -> None:
    """Re-exec this script under ``mjpython`` when the platform needs it.

    The MuJoCo viewer must be launched from ``mjpython`` on macOS, and ``mjpython``
    loads ``libpython`` with ``dlopen``.  Interpreters that are not installed as a
    framework (for example the standalone builds managed by ``uv``) do not export
    ``libpython`` on the default ``dyld`` search path, so ``mjpython`` aborts with
    "Library not loaded: @rpath/libpython3.12.dylib".  Adding the interpreter's own
    ``LIBDIR`` to ``DYLD_FALLBACK_LIBRARY_PATH`` fixes that, and doing it here means
    the demo works with a plain ``python scripts/demo_viewer.py``.
    """
    if sys.platform != "darwin" or os.environ.get("_VRCM_MJPYTHON") == "1":
        return
    try:
        import mujoco.viewer as viewer
    except ImportError:
        return
    if getattr(viewer, "_MJPYTHON", None) is not None:
        return  # already running inside mjpython

    mjpython = shutil.which("mjpython") or str(Path(sys.executable).with_name("mjpython"))
    if not Path(mjpython).exists():
        print(
            "[demo_viewer] on macOS the viewer must be started with `mjpython`, which was not "
            "found on PATH; run `mjpython scripts/demo_viewer.py ...` manually.",
            file=sys.stderr,
        )
        return

    env = dict(os.environ)
    env["_VRCM_MJPYTHON"] = "1"
    libdir = sysconfig.get_config_var("LIBDIR")
    if libdir:
        existing = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{libdir}:{existing}" if existing else libdir
    os.execve(mjpython, [mjpython, str(Path(__file__).resolve()), *sys.argv[1:]], env)



RCM_AXIS_LENGTH = 0.04  # RCM frame triad [m]
RCM_AXIS_WIDTH = 0.0015
RCM_ERROR_MIN = 1e-5  # draw the error segment above this offset [m]
RCM_ERROR_RGBA = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)
RCM_AXIS_RGBA = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.4, 1.0, 1.0]], dtype=np.float32)


def build_session(robot_name: str, mode: str, control_rate: float = 100.0):
    """Create the simulation and planner used by the interactive demo."""
    spec = load_robot_spec(robot_name)
    sim = VrcmSim(
        spec,
        VrcmConfig(k_rcm=2.0, k_orient=3.0, k_insertion=2.0, lambda_reg=1e-4),
        SimConfig(mode=mode, control_dt=1.0 / control_rate),
    )
    sim.reset()
    state = sim.state()
    planner = ViewpointPlanner(
        d_nominal=state["d"], depth_nominal=float(state["length"]), max_depth_rate=0.08
    )
    return sim, planner


def handle_key(keycode: int, planner: ViewpointPlanner, ui: dict, step_rad: float, insert_m: float) -> None:
    """Translate a viewer key press (a GLFW key code) into a viewpoint action."""

    def toggle(flag: str) -> None:
        ui[flag] = not ui[flag]

    actions = {
        glfw.KEY_UP: lambda: planner.command(ViewpointAction(pitch=step_rad)),
        glfw.KEY_DOWN: lambda: planner.command(ViewpointAction(pitch=-step_rad)),
        glfw.KEY_6: lambda: planner.command(ViewpointAction(yaw=step_rad)),
        glfw.KEY_7: lambda: planner.command(ViewpointAction(yaw=-step_rad)),
        glfw.KEY_8: lambda: planner.command(ViewpointAction(insertion=insert_m)),
        glfw.KEY_9: lambda: planner.command(ViewpointAction(insertion=-insert_m)),
        glfw.KEY_ENTER: planner.reset,
        glfw.KEY_PERIOD: lambda: planner.goto(planner.pitch, planner.yaw, planner.depth),
        glfw.KEY_F6: lambda: toggle("log_rcm"),
        glfw.KEY_SPACE: lambda: toggle("paused"),
    }
    action = actions.get(keycode)
    if action is not None:
        action()


def draw_rcm_markers(scn: mujoco.MjvScene, trocar: np.ndarray, p_tip: np.ndarray, d: np.ndarray) -> None:
    """Draw the moving part of the RCM frame into a viewer's ``user_scn``.

    * RGB triad at the trocar in the planner's shaft frame (z along the shaft),
    * magenta segment from the trocar to the closest point on the shaft axis.

    The fixed trocar point itself is the scene's ``rcm`` site.
    """
    scn.ngeom = 0

    def connect(geom_type, width, a, b, rgba) -> None:
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, geom_type, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
        mujoco.mjv_connector(g, geom_type, width, a, b)
        scn.ngeom += 1

    frame = shaft_frame(d)
    for axis, rgba in zip(frame.T, RCM_AXIS_RGBA):
        connect(mujoco.mjtGeom.mjGEOM_ARROW, RCM_AXIS_WIDTH, trocar, trocar + RCM_AXIS_LENGTH * axis, rgba)

    e = rcm_error(p_tip, frame[:, 2], trocar)
    if np.linalg.norm(e) > RCM_ERROR_MIN:
        connect(mujoco.mjtGeom.mjGEOM_CAPSULE, RCM_AXIS_WIDTH / 2, trocar, trocar + e, RCM_ERROR_RGBA)


def control_step(sim, planner, dt: float):
    """One V-RCM control cycle: planner reference -> QP -> MuJoCo step."""
    ref = planner.step(dt)
    return sim.step(
        VrcmTarget(
            d_des=ref.d_des,
            insertion_depth=ref.insertion_depth,
            insertion_velocity=ref.insertion_velocity,
            roll_velocity=ref.roll_velocity,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", nargs="?", default="ur5e")
    parser.add_argument("--mode", default="kinematic", choices=["kinematic", "dynamic"])
    parser.add_argument("--control-rate", type=float, default=100.0)
    parser.add_argument("--step", type=float, default=2.0, help="pitch/yaw increment [deg]")
    parser.add_argument("--insert-step", type=float, default=5.0, help="insertion increment [mm]")
    parser.add_argument("--no-relaunch", action="store_true",
                        help="do not re-exec under mjpython on macOS")
    args = parser.parse_args()

    if not args.no_relaunch:
        relaunch_under_mjpython()

    try:
        import mujoco.viewer
    except ImportError:
        print("the MuJoCo viewer is not available in this environment", file=sys.stderr)
        return 1

    sim, planner = build_session(args.robot, args.mode, args.control_rate)
    ui = {"log_rcm": False, "paused": False}
    step_rad = np.deg2rad(args.step)
    insert_m = args.insert_step * 1e-3

    def key_callback(keycode: int) -> None:
        handle_key(keycode, planner, ui, step_rad, insert_m)

    dt = sim.config.control_dt
    print(f"running {args.robot} in {args.mode} mode; see --help for the key bindings")
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=key_callback) as viewer:
            viewer.opt.sitegroup[RCM_SITE_GROUP] = 1
            while viewer.is_running():
                t0 = time.perf_counter()
                if not ui["paused"]:
                    result = control_step(sim, planner, dt)
                    if ui["log_rcm"]:
                        print(f"\rrcm={result.rcm_error_norm * 1e3:8.4f} mm  "
                              f"depth={result.length * 1e3:7.2f} mm  "
                              f"pitch={np.degrees(planner.pitch):+6.2f} deg  "
                              f"yaw={np.degrees(planner.yaw):+6.2f} deg  ",
                              end="", flush=True)
                    # sim.step leaves the tip site pose current; no need for sim.state()
                    tip = sim.robot.tip_site_id
                    with viewer.lock():
                        draw_rcm_markers(viewer.user_scn, sim.trocar, sim.data.site_xpos[tip],
                                         sim.data.site_xmat[tip].reshape(3, 3)[:, 2])
                viewer.sync()
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    except RuntimeError as exc:
        if "mjpython" in str(exc):
            libdir = sysconfig.get_config_var("LIBDIR") or ""
            print(
                "On macOS the MuJoCo viewer has to run under mjpython.  For interpreters "
                "that do not export libpython on the default search path (uv-managed "
                "builds), prefix the command with the library directory:\n\n"
                f"  DYLD_FALLBACK_LIBRARY_PATH={libdir} mjpython scripts/demo_viewer.py "
                f"{args.robot} --mode {args.mode}\n",
                file=sys.stderr,
            )
            return 1
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
