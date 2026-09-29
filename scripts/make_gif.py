#!/usr/bin/env python
"""Render the README animation into ``figures/demo.gif``.

Both arms run the same circular viewpoint sweep in dynamic simulation.  Each
panel shows the arm, a close-up of the trocar with a fading trail of shaft axes
(all passing through the RCM), and a live readout; a strip at the bottom plots
the RCM error of both arms as it accumulates.  The sweep starts and ends at the
home viewpoint, so the GIF loops seamlessly.

    python scripts/make_gif.py            # needs ffmpeg on PATH
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_teaser import (  # noqa: E402
    RCM_RED, ROBOT_COLOR, ROBOT_LABEL, Projector, add_connector, add_sphere, hex_rgba, style_model,
)
from vrcm.controller import VrcmConfig  # noqa: E402
from vrcm.planner import ViewpointPlanner  # noqa: E402
from vrcm.sim import SimConfig, VrcmSim  # noqa: E402
from vrcm.specs import PROJECT_ROOT, load_robot_spec  # noqa: E402

OUT = PROJECT_ROOT / "figures" / "demo.gif"
ROBOTS = ("ur5e", "flexiv_rizon4")

# viewpoint sweep: ease out to the cone, one loop around it, ease back home
RADIUS = np.deg2rad(18.0)
T_HOLD, T_EASE, T_LOOP = 0.5, 1.4, 7.0
DURATION = 2 * T_HOLD + 2 * T_EASE + T_LOOP
DEPTH_SWING = 0.03

# layout (px)
PANEL_W, PANEL_H = 540, 450
HEADER_H = 62
PLOT_H = 150
INSET = 200
W, H = 2 * PANEL_W, HEADER_H + PANEL_H + PLOT_H
TRAIL = 30  # frames of shaft-axis history in the close-up
INK = (27, 27, 27)
MUTED = (110, 114, 120)
GRID = (228, 230, 233)


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def goal(t: float, depth0: float) -> tuple[float, float, float]:
    """Viewpoint goal (pitch, yaw, depth) at time ``t``."""
    t1 = T_HOLD + T_EASE
    t2 = t1 + T_LOOP
    if t < t1:
        return RADIUS * smoothstep((t - T_HOLD) / T_EASE), 0.0, depth0
    if t < t2:
        th = 2 * np.pi * (t - t1) / T_LOOP
        return RADIUS * np.cos(th), RADIUS * np.sin(th), depth0 + DEPTH_SWING * np.sin(2 * th)
    return RADIUS * (1.0 - smoothstep((t - t2) / T_EASE)), 0.0, depth0


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for path, index in (("/System/Library/Fonts/HelveticaNeue.ttc", 1 if bold else 0),
                        ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", 0)):
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default()


class View:
    """An offscreen camera whose frames get a white studio background."""

    def __init__(self, sim: VrcmSim, camera: dict, size: tuple[int, int], supersample: int = 2) -> None:
        self.sim = sim
        self.size = size
        h, w = size[1] * supersample, size[0] * supersample
        model = sim.model
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
        model.vis.global_.offheight = max(model.vis.global_.offheight, h)
        self.renderer = mujoco.Renderer(model, h, w, max_geom=5000)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = camera["lookat"]
        self.cam.distance = camera["distance"]
        self.cam.azimuth = camera["azimuth"]
        self.cam.elevation = camera["elevation"]
        self.opt = mujoco.MjvOption()
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def render(self, draw_extras, *, static_only: bool = False) -> tuple[Image.Image, Projector]:
        r = self.renderer
        r.update_scene(self.sim.data, camera=self.cam, scene_option=self.opt)
        scn = r.scene
        if static_only:  # static geoms are added first
            scn.ngeom = sum(1 for k in range(scn.ngeom) if scn.geoms[k].category == mujoco.mjtCatBit.mjCAT_STATIC)
        draw_extras(scn)
        img = r.render().copy()
        proj = Projector(scn.camera[0], *self.size)

        # segmentation pass to separate floor and background from content
        for k in range(scn.ngeom):
            scn.geoms[k].rgba[3] = 1.0
        objid = np.array([scn.geoms[k].objid for k in range(scn.ngeom)])
        objtype = np.array([scn.geoms[k].objtype for k in range(scn.ngeom)])
        scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
        scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1
        raw = r.render().astype(np.int64)
        scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
        scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0
        segid = raw[..., 0] + raw[..., 1] * 256 + raw[..., 2] * 65536 - 1
        valid = (segid >= 0) & (segid < scn.ngeom)
        background = ~valid
        on_floor = np.zeros_like(valid)
        on_floor[valid] = (objtype[segid[valid]] == mujoco.mjtObj.mjOBJ_GEOM) & (objid[segid[valid]] == self.floor)
        lum = img[on_floor].astype(float).mean(axis=1)
        lit = np.percentile(lum, 97) if lum.size else 255.0
        shade = np.clip(lum / lit, 0.0, 1.0)
        img[on_floor] = (255.0 - (1.0 - shade[:, None]) * 80.0).astype(np.uint8)
        img[background] = 255
        content = ~(background | on_floor)
        if content.any():  # clear the floor's far edge above everything else
            img[: np.nonzero(content.any(axis=1))[0].min() - 2] = 255
        return Image.fromarray(img).resize(self.size, Image.LANCZOS), proj


def make_sim(name: str) -> VrcmSim:
    spec = load_robot_spec(name)
    cfg = VrcmConfig(
        k_rcm=2.0, k_orient=3.0, k_insertion=2.0, lambda_reg=1e-4,
        posture_gain=0.05 if len(spec.home_qpos) > 6 else 0.0,
    )
    sim = VrcmSim(spec, cfg, SimConfig(mode="dynamic", control_dt=0.01))
    style_model(sim.model)
    trocar_mat = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_MATERIAL, "trocar_mat")
    sim.model.mat_rgba[trocar_mat, 3] = 0.45  # let the RCM point show through the port
    sim.model.vis.quality.shadowsize = 4096
    sim.reset()
    return sim


def draw_plot(canvas: Image.Image, traces: dict[str, list[tuple[float, float]]], t_now: float) -> None:
    d = ImageDraw.Draw(canvas)
    top = HEADER_H + PANEL_H
    x0, x1 = 86, W - 150
    y0, y1 = top + 26, top + PLOT_H - 34
    y_max = 0.5  # mm
    d.line([(0, top), (W, top)], fill=GRID, width=1)
    for v in (0.0, 0.2, 0.4):
        y = y1 - (y1 - y0) * v / y_max
        d.line([(x0, y), (x1, y)], fill=GRID if v else (190, 194, 199), width=1)
        d.text((x0 - 8, y), f"{v:.1f}", fill=MUTED, font=font(13), anchor="rm")
    d.text((22, (y0 + y1) / 2), "RCM\nerror\n[mm]", fill=MUTED, font=font(12), anchor="lm", spacing=2)
    for s in range(0, int(DURATION) + 1, 2):
        x = x0 + (x1 - x0) * s / DURATION
        d.text((x, y1 + 8), f"{s}s", fill=MUTED, font=font(12), anchor="mt")
    d.text((x0, top + 8), "RCM error in dynamic simulation, 100 Hz QP control", fill=INK, font=font(14, True))
    xc = x0 + (x1 - x0) * t_now / DURATION
    d.line([(xc, y0), (xc, y1)], fill=(205, 208, 212), width=1)
    for k, (name, pts) in enumerate(traces.items()):
        color = ROBOT_COLOR[name]
        xy = [(x0 + (x1 - x0) * t / DURATION, y1 - (y1 - y0) * min(e, y_max) / y_max) for t, e in pts]
        if len(xy) > 1:
            d.line(xy, fill=color, width=3, joint="curve")
        e_max = max(e for _, e in pts)
        ly = y0 + 10 + k * 38
        d.line([(x1 + 16, ly), (x1 + 34, ly)], fill=color, width=4)
        d.text((x1 + 40, ly), ROBOT_LABEL[name].split(" (")[0], fill=INK, font=font(13, True), anchor="lm")
        d.text((x1 + 40, ly + 17), f"max {e_max:.3f} mm", fill=MUTED, font=font(12), anchor="lm")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--still", type=float, help="only save the frame at this time [s] as a PNG to --out")
    args = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        print("ffmpeg is required", file=sys.stderr)
        return 1

    sims, planners, views, trails, traces = {}, {}, {}, {}, {}
    for name in ROBOTS:
        sim = make_sim(name)
        state = sim.state()
        planner = ViewpointPlanner(d_nominal=state["d"], depth_nominal=float(state["length"]))
        tx, ty, tz = sim.trocar
        wide = dict(lookat=(tx - 0.16, ty + 0.06, tz * 0.72), distance=1.6, azimuth=132, elevation=-20)
        close = dict(lookat=(tx, ty, tz - 0.01), distance=0.36, azimuth=132, elevation=-22)
        sims[name], planners[name] = sim, planner
        views[name] = (View(sim, wide, (PANEL_W, PANEL_H)), View(sim, close, (INSET, INSET)))
        trails[name], traces[name] = [], []

    dt = sims[ROBOTS[0]].config.control_dt
    per_frame = max(int(round(1.0 / (args.fps * dt))), 1)
    n_steps = int(round(DURATION / dt))
    depth0 = {n: planners[n].depth for n in ROBOTS}

    with tempfile.TemporaryDirectory() as tmp:
        frame = 0
        for i in range(n_steps + 1):
            t = i * dt
            for name in ROBOTS:
                pitch, yaw, depth = goal(t, depth0[name])
                planners[name].goto(pitch, yaw, depth)
                if i:
                    sims[name].run(planners[name], dt)
                traces[name].append((t, 1e3 * sims[name].rcm_error_norm))
            if (args.still is None and i % per_frame) or (args.still is not None and i != round(args.still / dt)):
                continue

            canvas = Image.new("RGB", (W, H), "white")
            d = ImageDraw.Draw(canvas)
            for k, name in enumerate(ROBOTS):
                sim = sims[name]
                kin = sim.robot.kinematics()
                trocar = sim.trocar
                trails[name] = (trails[name] + [(kin.p_tip.copy(), kin.d.copy())])[-TRAIL:]
                color = ROBOT_COLOR[name]

                def wide_extras(scn, kin=kin, trocar=trocar, color=color):
                    length = float(np.dot(trocar - kin.p_tip, -kin.d))
                    add_connector(scn, kin.p_tip, kin.p_tip - kin.d * (length + 0.14), 0.0025, hex_rgba(color, 0.9))
                    add_sphere(scn, trocar, 0.012, hex_rgba(RCM_RED, 0.95))

                def close_extras(scn, trail=trails[name], trocar=trocar, color=color):
                    n = len(trail)
                    for j, (p, dd) in enumerate(trail):
                        a = (j + 1) / n
                        length = float(np.dot(trocar - p, -dd))
                        add_connector(scn, p, p - dd * (length + 0.07), 0.0006 + 0.0006 * a,
                                      hex_rgba(color, 0.08 + 0.9 * a ** 4))
                    p, dd = trail[-1]
                    add_sphere(scn, p, 0.0022, hex_rgba(color))
                    add_sphere(scn, trocar, 0.0040, hex_rgba(RCM_RED, 0.95))

                x = k * PANEL_W
                wide_view, close_view = views[name]
                canvas.paste(wide_view.render(wide_extras)[0], (x, HEADER_H))
                inset, proj = close_view.render(close_extras, static_only=True)
                ix, iy = x + PANEL_W - INSET - 14, HEADER_H + PANEL_H - INSET - 12
                canvas.paste(inset, (ix, iy))
                d.rounded_rectangle([ix - 1, iy - 1, ix + INSET, iy + INSET], radius=6, outline=(200, 204, 209),
                                    width=2)
                d.text((ix + 8, iy + 6), "trocar close-up", fill=MUTED, font=font(12))
                u, v = proj(trocar)
                d.text((ix + u - 14, iy + v), "RCM", fill=RCM_RED, font=font(12, True), anchor="rm")

                d.text((x + 20, 14), ROBOT_LABEL[name], fill=INK, font=font(19, True))
                e_now = traces[name][-1][1]
                d.text((x + PANEL_W - 20, 16), f"RCM error  {e_now:.3f} mm", fill=color, font=font(15, True),
                       anchor="ra")
                ref = planners[name]
                d.text((x + 20, 40),
                       f"pitch {np.rad2deg(ref.pitch):+5.1f}°   yaw {np.rad2deg(ref.yaw):+5.1f}°   "
                       f"depth {1e3 * ref.depth:5.1f} mm",
                       fill=MUTED, font=font(13))
            d.line([(PANEL_W, 10), (PANEL_W, HEADER_H + PANEL_H - 10)], fill=GRID, width=1)
            draw_plot(canvas, traces, t)
            if args.still is not None:
                canvas.save(args.out)
                print(f"wrote {args.out}")
                return 0
            canvas.save(Path(tmp) / f"f{frame:04d}.png")
            frame += 1
            print(f"\rframe {frame}", end="", flush=True)
        print()

        args.out.parent.mkdir(parents=True, exist_ok=True)
        pattern = str(Path(tmp) / "f%04d.png")
        palette = str(Path(tmp) / "palette.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", pattern,
                        "-vf", "palettegen=max_colors=192:stats_mode=diff", palette], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", pattern,
                        "-i", palette, "-lavfi", "paletteuse=dither=sierra2_4a:diff_mode=rectangle",
                        "-loop", "0", str(args.out)], check=True)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {frame} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
