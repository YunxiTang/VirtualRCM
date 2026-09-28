#!/usr/bin/env python
"""Head-less validation of the V-RCM controller for both arms.

Runs, for every robot config and both execution modes:

1. a scripted viewpoint sequence ``(pitch, yaw, insertion)`` -- the interface of
   section 10 of ``method.md``;
2. a coarse-to-fine stop-and-go viewpoint search driven by a geometric
   visibility model -- the loop of section 11;

writes CSV logs and PNG plots into ``results/`` and prints a summary table with
the RCM error, viewpoint tracking and insertion depth.

    python scripts/run_headless.py --modes kinematic dynamic --robots ur5e flexiv_rizon4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrcm.controller import VrcmConfig  # noqa: E402
from vrcm.planner import ViewpointAction, ViewpointPlanner  # noqa: E402
from vrcm.sim import SimConfig, VrcmSim  # noqa: E402
from vrcm.specs import PROJECT_ROOT, available_robots, load_robot_spec  # noqa: E402
from vrcm.visibility import VisibilityModel, stop_and_go_scan  # noqa: E402

RESULTS = PROJECT_ROOT / "results"


def scripted_sequence() -> list[ViewpointAction]:
    """A laparoscopic camera tour: tilt, sweep, zoom in, retract."""
    return [
        ViewpointAction(pitch=np.deg2rad(6.0), yaw=np.deg2rad(-4.0), insertion=0.01),
        ViewpointAction(pitch=np.deg2rad(-12.0), yaw=np.deg2rad(9.0), insertion=0.02),
        ViewpointAction(pitch=np.deg2rad(4.0), yaw=np.deg2rad(-14.0), insertion=-0.015),
        ViewpointAction(insertion=0.025),
        ViewpointAction(pitch=np.deg2rad(-5.0), yaw=np.deg2rad(5.0), insertion=-0.03),
    ]


def write_csv(path: Path, log) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["t", "qdot", "p_tip", "d", "d_des", "rcm_error", "rcm_error_norm_mm", "length", "depth_target"]
        )
        for i, t in enumerate(log.time):
            writer.writerow(
                [
                    f"{t:.5f}",
                    " ".join(f"{v:.6f}" for v in log.qdot[i]),
                    " ".join(f"{v:.6f}" for v in log.p_tip[i]),
                    " ".join(f"{v:.6f}" for v in log.d[i]),
                    " ".join(f"{v:.6f}" for v in log.d_des[i]),
                    " ".join(f"{v:.6f}" for v in log.e_rcm[i]),
                    f"{np.linalg.norm(log.e_rcm[i]) * 1e3:.6f}",
                    f"{log.length[i]:.6f}",
                    f"{log.depth_target[i]:.6f}",
                ]
            )


def plot(name: str, log, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    arr = log.arrays()
    t = arr["t"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(t, np.linalg.norm(arr["e_rcm"], axis=1) * 1e3, color="crimson")
    axes[0].set_ylabel("RCM error [mm]")
    axes[0].grid(alpha=0.3)
    ang = np.degrees(np.arccos(np.clip(np.sum(arr["d"] * arr["d_des"], axis=1), -1, 1)))
    axes[1].plot(t, ang, color="steelblue")
    axes[1].set_ylabel("shaft direction error [deg]")
    axes[1].grid(alpha=0.3)
    axes[2].plot(t, arr["length"] * 1e3, label="depth", color="seagreen")
    axes[2].plot(t, arr["depth_target"] * 1e3, "--", label="target", color="gray")
    axes[2].set_ylabel("insertion depth [mm]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    fig.suptitle(f"{name}: V-RCM closed loop")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_robot(name: str, mode: str, *, plot_results: bool) -> dict[str, float]:
    spec = load_robot_spec(name)
    cfg = VrcmConfig(
        k_rcm=2.0,
        k_orient=3.0,
        k_insertion=2.0,
        lambda_reg=1e-4,
        posture_gain=0.05 if len(spec.home_qpos) > 6 else 0.0,
    )
    sim = VrcmSim(spec, cfg, SimConfig(mode=mode, control_dt=0.01))
    sim.reset()
    state = sim.state()
    planner = ViewpointPlanner(d_nominal=state["d"], depth_nominal=float(state["length"]))

    log = None
    for action in scripted_sequence():
        planner.command(action)
        log = sim.run(planner, 1.5, log=log)
    assert log is not None

    summary = log.summary()
    tag = f"{name}_{mode}"
    write_csv(RESULTS / f"{tag}_scripted.csv", log)
    if plot_results:
        plot(f"{name} ({mode})", log, RESULTS / f"{tag}_scripted.png")

    # --- stop-and-go viewpoint search --------------------------------------
    planner.reset()
    # a lesion on the cavity floor, off to one side: the nominal "straight down"
    # viewpoint is not the best one, so the scan has to find a tilted viewpoint
    visibility = VisibilityModel(
        target=np.array(spec.trocar) + np.array([0.05, 0.04, -0.18]),
        fov_deg=80.0,
        ideal_distance=0.09,
        distance_sigma=0.05,
    )
    scan = stop_and_go_scan(
        sim, planner, visibility,
        pitch_span=np.deg2rad(18), yaw_span=np.deg2rad(18), depth_span=0.02,
        coarse=3, fine=3,
    )
    summary.update(
        {
            "scan_candidates": len(scan.candidates),
            "scan_best_score": scan.best_score,
            "scan_best_pitch_deg": scan.best_viewpoint[0] if scan.best_viewpoint else 0.0,
            "scan_best_yaw_deg": scan.best_viewpoint[1] if scan.best_viewpoint else 0.0,
            "scan_best_depth_mm": scan.best_viewpoint[2] * 1e3 if scan.best_viewpoint else 0.0,
        }
    )
    (RESULTS / f"{tag}_scan.txt").parent.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}_scan.txt").write_text(scan.table(limit=10) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", nargs="*", default=available_robots())
    parser.add_argument("--modes", nargs="*", default=["kinematic", "dynamic"])
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    rows: list[tuple[str, str, dict[str, float]]] = []
    for name in args.robots:
        for mode in args.modes:
            summary = run_robot(name, mode, plot_results=not args.no_plot)
            rows.append((name, mode, summary))

    header = f"{'robot':16s} {'mode':10s} {'rcm max[mm]':>11s} {'rcm rms[mm]':>11s} " \
             f"{'dir err[deg]':>12s} {'depth[min,max][m]':>19s} {'scan best':>10s}"
    print(header)
    print("-" * len(header))
    for name, mode, s in rows:
        print(
            f"{name:16s} {mode:10s} {s['rcm_error_max_mm']:11.4f} {s['rcm_error_rms_mm']:11.4f} "
            f"{s['direction_error_max_deg']:12.3f} "
            f"{s['depth_min']:.3f},{s['depth_max']:.3f}   {s['scan_best_score']:10.4f}"
        )
    print(f"\nlogs and plots written to {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
