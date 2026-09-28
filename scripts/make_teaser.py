#!/usr/bin/env python
"""Render the project teaser figure into ``figures/teaser.{png,pdf}``.

Panels:

(a), (b) each arm on a clean background, with translucent "ghost" poses from a
         viewpoint tour and every shaft axis passing through the RCM point;
(c)      a close-up of the trocar with the pitch / yaw / insertion motions;
(d)      the RCM error over the scripted tour, read from ``results/*.csv``.

    python scripts/make_teaser.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrcm.controller import VrcmConfig  # noqa: E402
from vrcm.planner import ViewpointPlanner  # noqa: E402
from vrcm.sim import SimConfig, VrcmSim  # noqa: E402
from vrcm.specs import PROJECT_ROOT, load_robot_spec  # noqa: E402

OUT = PROJECT_ROOT / "figures"
RESULTS = PROJECT_ROOT / "results"

# viewpoint tour shown as ghosts: (pitch [deg], yaw [deg], depth [m])
TOUR = [(0.0, 0.0, 0.15), (26.0, 4.0, 0.11), (-22.0, 12.0, 0.19), (6.0, -28.0, 0.17), (-12.0, -20.0, 0.12)]
POSE_COLORS = ["#2b2d42", "#e07a1f", "#1f8a8a", "#7a4fb5", "#c23b6b"]
RCM_RED = "#d62828"
ROBOT_LABEL = {"ur5e": "UR5e (6-DOF)", "flexiv_rizon4": "Flexiv Rizon 4 (7-DOF)"}
ROBOT_COLOR = {"ur5e": "#1f6fb4", "flexiv_rizon4": "#e07a1f"}


def hex_rgba(color: str, alpha: float = 1.0) -> np.ndarray:
    return np.array([*matplotlib.colors.to_rgb(color), alpha], dtype=np.float32)


# ----------------------------------------------------------------- simulation
def make_sim(name: str) -> VrcmSim:
    spec = load_robot_spec(name)
    cfg = VrcmConfig(
        k_rcm=2.0, k_orient=3.0, k_insertion=2.0, lambda_reg=1e-4,
        posture_gain=0.05 if len(spec.home_qpos) > 6 else 0.0,
    )
    sim = VrcmSim(spec, cfg, SimConfig(mode="kinematic", control_dt=0.01))
    sim.reset()
    return sim


def tour_poses(sim: VrcmSim) -> list[np.ndarray]:
    """Drive the controller through TOUR and keep the settled joint pose of each stop."""
    state = sim.state()
    planner = ViewpointPlanner(d_nominal=state["d"], depth_nominal=float(state["length"]))
    poses = []
    for pitch, yaw, depth in TOUR:
        planner.goto(np.deg2rad(pitch), np.deg2rad(yaw), depth)
        sim.run(planner, 4.0, stop_when_settled=True)
        sim.run(planner, 0.5)
        poses.append(sim.robot.arm_qpos().copy())
    return poses


# ------------------------------------------------------------------ rendering
def style_model(model: mujoco.MjModel) -> None:
    """Studio look: plain light floor, softer phantom, no checker / skybox."""
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    model.geom_matid[floor] = -1
    model.geom_rgba[floor] = [1.0, 1.0, 1.0, 1.0]
    model.vis.rgba.haze[:] = [1.0, 1.0, 1.0, 1.0]
    for name, rgba in {
        "wall_mat": [0.93, 0.72, 0.64, 0.38],
        "cavity_mat": [0.88, 0.45, 0.42, 0.07],
        "trocar_mat": [0.25, 0.27, 0.30, 1.0],
    }.items():
        model.mat_rgba[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)] = rgba
    model.vis.headlight.ambient[:] = 0.42
    model.vis.headlight.diffuse[:] = 0.55
    model.vis.headlight.specular[:] = 0.05
    model.vis.quality.shadowsize = 8192


def add_connector(scn, a, b, width, rgba, geom_type=mujoco.mjtGeom.mjGEOM_CAPSULE) -> None:
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, geom_type, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
    mujoco.mjv_connector(g, geom_type, width, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def add_sphere(scn, p, r, rgba) -> None:
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, r), np.asarray(p, float), np.eye(3).ravel(), rgba)
    scn.ngeom += 1


def tip_axis(sim: VrcmSim, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sim.robot.set_arm_qpos(q)
    sim.robot.forward()
    tip = sim.robot.tip_site_id
    return sim.data.site_xpos[tip].copy(), sim.data.site_xmat[tip].reshape(3, 3)[:, 2].copy()


def render(
    sim: VrcmSim,
    poses: list[np.ndarray],
    camera: dict,
    size: tuple[int, int],
    *,
    main: int = 0,
    ghost_alpha: float = 0.16,
    axis_width: float = 0.0022,
    axis_above: float = 0.16,
    rcm_radius: float = 0.011,
    show_arm: bool = True,
) -> tuple[np.ndarray, "Projector", np.ndarray]:
    model, data = sim.model, sim.data
    h, w = size
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, h)
    renderer = mujoco.Renderer(model, h, w, max_geom=20000)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = camera["lookat"]
    cam.distance = camera["distance"]
    cam.azimuth = camera["azimuth"]
    cam.elevation = camera["elevation"]
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1

    # ghosts first, from a scratch MjData, then the main pose on top
    ghost_data = mujoco.MjData(model)
    renderer.update_scene(data, camera=cam, scene_option=opt)  # sets camera + static geoms
    scn = renderer.scene
    sim.robot.set_arm_qpos(poses[main])
    sim.robot.forward()
    renderer.update_scene(data, camera=cam, scene_option=opt)
    scn = renderer.scene
    if not show_arm:
        scn.ngeom = sum(1 for i in range(scn.ngeom) if scn.geoms[i].category == mujoco.mjtCatBit.mjCAT_STATIC)
    pert = mujoco.MjvPerturb()
    for i, q in enumerate(poses):
        if i == main or not show_arm:
            continue
        ghost_data.qpos[:] = data.qpos
        ghost_data.qpos[sim.robot._arm_qpos_adr] = q
        mujoco.mj_forward(model, ghost_data)
        start = scn.ngeom
        mujoco.mjv_addGeoms(model, ghost_data, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
        tint = hex_rgba(POSE_COLORS[i])
        for k in range(start, scn.ngeom):
            g = scn.geoms[k]
            g.rgba[:3] = 0.45 * g.rgba[:3] + 0.55 * tint[:3]
            g.rgba[3] = ghost_alpha
            g.matid = -1

    trocar = sim.trocar
    for i, q in enumerate(poses):
        p_tip, d = tip_axis(sim, q)
        length = float(np.dot(trocar - p_tip, -d))
        add_connector(scn, p_tip, p_tip - d * (length + axis_above), axis_width, hex_rgba(POSE_COLORS[i], 0.95))
        add_sphere(scn, p_tip, axis_width * 2.2, hex_rgba(POSE_COLORS[i]))
    add_sphere(scn, trocar, rcm_radius, hex_rgba(RCM_RED, 0.95))
    sim.robot.set_arm_qpos(poses[main])
    sim.robot.forward()

    img = renderer.render().copy()
    proj = Projector(scn.camera[0], w, h)

    # segmentation of the same scene, decoded by hand: the stock decoder fails on
    # the stray ID colours that edge blending produces
    for k in range(scn.ngeom):
        scn.geoms[k].rgba[3] = 1.0
    objid = np.array([scn.geoms[k].objid for k in range(scn.ngeom)])
    objtype = np.array([scn.geoms[k].objtype for k in range(scn.ngeom)])
    objid[objid < 0] = 10**6  # overlay geoms: content, never floor or background
    scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
    scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1
    raw = renderer.render().astype(np.int64)
    renderer.close()
    segid = raw[..., 0] + raw[..., 1] * 256 + raw[..., 2] * 65536 - 1
    valid = (segid >= 0) & (segid < scn.ngeom)
    seg = np.full(segid.shape + (2,), -1, dtype=np.int64)
    seg[valid, 0] = objid[segid[valid]]
    seg[valid, 1] = objtype[segid[valid]]

    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    background = seg[..., 0] < 0
    on_floor = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 0] == floor)
    # white studio background that keeps only the floor's shadows
    lum = img[on_floor].astype(float).mean(axis=1)
    lit = np.percentile(lum, 97) if lum.size else 255.0
    shade = np.clip(lum / lit, 0.0, 1.0)
    img[on_floor] = (255.0 - (1.0 - shade[:, None]) * 150.0).astype(np.uint8)
    img[background] = 255
    content = ~(background | on_floor)
    top = np.nonzero(content.any(axis=1))[0].min()
    img[:top][on_floor[:top]] = 255
    return img, proj, ~(background | on_floor)


class Projector:
    """World point -> pixel coordinates for the camera a frame was rendered with."""

    def __init__(self, glcam, w: int, h: int) -> None:
        self.pos = np.array(glcam.pos)
        self.fwd = np.array(glcam.forward)
        self.up = np.array(glcam.up)
        self.right = np.cross(self.fwd, self.up)
        self.near = glcam.frustum_near
        self.center = glcam.frustum_center
        self.bottom = glcam.frustum_bottom
        self.top = glcam.frustum_top
        # frustum_width is left at 0 until render time, which then derives it from the aspect ratio
        self.width = glcam.frustum_width or (self.top - self.bottom) / 2 * w / h
        self.w, self.h = w, h

    def __call__(self, p) -> np.ndarray:
        v = np.asarray(p, float) - self.pos
        x, y, z = v @ self.right, v @ self.up, v @ self.fwd
        xn = (self.near * x / z - self.center) / self.width
        yn = (self.near * y / z - (self.top + self.bottom) / 2) / ((self.top - self.bottom) / 2)
        return np.array([(xn + 1) / 2 * self.w, (1 - yn) / 2 * self.h])


def crop_to(img: np.ndarray, mask: np.ndarray, pad: float = 0.06, aspect: float | None = None) -> np.ndarray:
    """Crop ``img`` to the bounding box of ``mask`` plus a relative margin,
    widened to ``aspect`` (w / h) when given."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    m = int(pad * max(y1 - y0, x1 - x0))
    y0, y1, x0, x1 = y0 - m, y1 + m, x0 - m, x1 + m
    if aspect is not None:
        h, w = y1 - y0, x1 - x0
        if w / h < aspect:
            grow = int((aspect * h - w) / 2)
            x0, x1 = x0 - grow, x1 + grow
        else:
            grow = int((w / aspect - h) / 2)
            y0, y1 = y0 - grow, y1 + grow
    pad_img = np.pad(img, ((img.shape[0],) * 2, (img.shape[1],) * 2, (0, 0)), constant_values=255)
    oy, ox = img.shape[:2]
    return pad_img[y0 + oy:y1 + oy, x0 + ox:x1 + ox]


# -------------------------------------------------------------------- figure
def load_error(name: str, mode: str = "dynamic") -> tuple[np.ndarray, np.ndarray]:
    with open(RESULTS / f"{name}_{mode}_scripted.csv") as fh:
        rows = list(csv.DictReader(fh))
    t = np.array([float(r["t"]) for r in rows])
    e = np.array([float(r["rcm_error_norm_mm"]) for r in rows])
    return t, e


def panel_label(ax, text: str) -> None:
    ax.text(0.0, 1.0, text, transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold",
            color="#1b1b1b")


def main() -> int:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    })
    OUT.mkdir(parents=True, exist_ok=True)

    # figure geometry (inches): three square-ish render panels, then the plot
    fig_w, fig_h = 7.16, 2.55
    img_h = 0.78  # fraction of the figure height used by the renders
    img_bottom = 0.18
    panel_w = [0.22, 0.22, 0.25]
    aspects = [w * fig_w / (img_h * fig_h) for w in panel_w]

    renders = {}
    for k, name in enumerate(("ur5e", "flexiv_rizon4")):
        sim = make_sim(name)
        style_model(sim.model)
        poses = tour_poses(sim)
        tx, ty, tz = sim.trocar
        wide = dict(lookat=(tx - 0.22, ty + 0.02, tz * 0.62), distance=1.55, azimuth=132, elevation=-26)
        img, _, mask = render(sim, poses, wide, (1800, 1800))
        renders[name] = crop_to(img, mask, pad=0.04, aspect=aspects[k])
        if name == "ur5e":
            close = dict(lookat=(tx, ty, tz - 0.035), distance=0.46, azimuth=132, elevation=-26)
            img, proj, _ = render(sim, poses, close, (1500, 1500), axis_width=0.0014, rcm_radius=0.0058,
                                  axis_above=0.085, show_arm=False)
            # crop around the RCM
            u, v = proj(sim.trocar)
            ch = 1150
            cw = int(ch * aspects[2])
            x0, y0 = int(u - cw / 2), int(v - 0.36 * ch)
            renders["close"] = img[y0:y0 + ch, x0:x0 + cw]
            renders["close_proj"] = lambda p, proj=proj, x0=x0, y0=y0: proj(p) - np.array([x0, y0])
            renders["close_pts"] = dict(trocar=sim.trocar.copy(), poses=[tip_axis(sim, q) for q in poses])

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    x = 0.004
    axes = []
    for w in panel_w:
        axes.append(fig.add_axes([x, img_bottom, w, img_h]))
        x += w + 0.008
    ax_ur, ax_fx, ax_cl = axes
    for ax, key, label in ((ax_ur, "ur5e", "(a)"), (ax_fx, "flexiv_rizon4", "(b)")):
        ax.imshow(renders[key], interpolation="lanczos")
        ax.set_axis_off()
        panel_label(ax, label)
        ax.text(0.5, 0.0, ROBOT_LABEL[key], transform=ax.transAxes, ha="center", va="bottom", fontsize=8)

    # (c) close-up with annotations
    img = renders["close"]
    proj = renders["close_proj"]
    ax_cl.imshow(img, interpolation="lanczos")
    ax_cl.set_xlim(0, img.shape[1])
    ax_cl.set_ylim(img.shape[0], 0)
    ax_cl.set_axis_off()
    panel_label(ax_cl, "(c)")
    ax_cl.text(0.5, 0.0, "every shaft axis passes the RCM", transform=ax_cl.transAxes,
               ha="center", va="bottom", fontsize=8)
    trocar = renders["close_pts"]["trocar"]
    poses = renders["close_pts"]["poses"]
    c = proj(trocar)
    # pitch / yaw: arc hugging the upper ends of the shaft axes
    ends = np.array([proj(trocar - d * 0.085) for _, d in poses])
    ang = np.arctan2(ends[:, 1] - c[1], ends[:, 0] - c[0])
    r = 1.08 * np.linalg.norm(ends - c, axis=1).mean()
    t = np.linspace(ang.min() - 0.12, ang.max() + 0.12, 60)
    arc = c + r * np.stack([np.cos(t), np.sin(t)], axis=1)
    ax_cl.plot(arc[:, 0], arc[:, 1], color="#1b1b1b", lw=0.9, solid_capstyle="butt")
    for tail, head in ((arc[3], arc[0]), (arc[-4], arc[-1])):
        ax_cl.annotate("", xy=head, xytext=tail,
                       arrowprops=dict(arrowstyle="-|>", mutation_scale=7, color="#1b1b1b", lw=0.9,
                                       shrinkA=0, shrinkB=0))
    ax_cl.text(arc[-1][0] + 18, arc[-1][1], "pitch / yaw", fontsize=7.5, ha="left", va="center")
    ax_cl.annotate("RCM", xy=c, xytext=(c[0] - 0.24 * img.shape[1], c[1] - 0.10 * img.shape[0]),
                   fontsize=8, color=RCM_RED, fontweight="bold", ha="right", va="center",
                   arrowprops=dict(arrowstyle="-", color=RCM_RED, lw=0.8, shrinkB=9))
    # insertion: arrow next to the home shaft, inside the body
    p_tip, d = poses[0]
    side = np.array([0.0, 0.0, 0.0])
    a, b = proj(trocar + d * 0.035), proj(trocar + d * 0.12)
    off = np.array([-0.21 * img.shape[1], 0.0])
    ax_cl.add_patch(FancyArrowPatch(a + off, b + off, arrowstyle="<|-|>", mutation_scale=8, color="#1b1b1b",
                                    lw=0.9))
    ax_cl.text((a + off)[0] - 14, (a[1] + b[1]) / 2, "insertion", fontsize=7.5, ha="right", va="center",
               rotation=90)

    # (d) RCM error
    ax_pl = fig.add_axes([0.80, 0.37, 0.155, 0.47])
    for name in ("ur5e", "flexiv_rizon4"):
        t, e = load_error(name)
        ax_pl.plot(t, e, lw=1.0, color=ROBOT_COLOR[name], label=ROBOT_LABEL[name].split(" (")[0])
        ax_pl.axhline(e.max(), color=ROBOT_COLOR[name], lw=0.5, ls=(0, (2, 2)), alpha=0.6)
        ax_pl.text(7.6, e.max(), f"max\n{e.max():.2f}", color=ROBOT_COLOR[name], fontsize=6, ha="left",
                   va="center", clip_on=False, linespacing=1.0)
    ax_pl.set_xlabel("time [s]", labelpad=1)
    ax_pl.set_ylabel("RCM error [mm]", labelpad=2)
    ax_pl.set_xlim(0, 7.5)
    ax_pl.set_ylim(0, 0.26)
    ax_pl.spines[["top", "right"]].set_visible(False)
    ax_pl.tick_params(length=2, pad=1.5, labelsize=7)
    ax_pl.legend(frameon=False, loc="upper left", fontsize=6.5, ncol=2, columnspacing=0.8, handlelength=1.4, borderaxespad=0.1)
    fig.text(0.735, 0.97, "(d)", fontsize=10, fontweight="bold", va="top")
    fig.text(0.785, 0.965, "RCM error < 0.2 mm", fontsize=8, fontweight="bold", va="top")
    fig.text(0.785, 0.90, "dynamic simulation, camera tour", fontsize=6.5, va="top", color="#555")

    # pipeline strip across the bottom
    boxes = [
        ("viewpoint planner", "(pitch, yaw, insertion)"),
        ("V-RCM QP", "RCM + viewpoint + joint limits"),
        ("joint velocities $\\dot q$", "100 Hz control"),
        ("any arm", "UR5e, Rizon 4, ... via config"),
    ]
    bx, by, bh, gap = 0.012, 0.02, 0.12, 0.028
    bw = (1.0 - 2 * bx - gap * (len(boxes) - 1)) / len(boxes)
    for i, (title, sub) in enumerate(boxes):
        x0 = bx + i * (bw + gap)
        fc = "#fdecea" if i == 1 else "#f2f3f5"
        ec = RCM_RED if i == 1 else "#b3b8bd"
        fig.patches.append(FancyBboxPatch((x0, by), bw, bh, boxstyle="round,pad=0.0,rounding_size=0.01",
                                          transform=fig.transFigure, fc=fc, ec=ec, lw=0.7))
        fig.text(x0 + bw / 2, by + bh * 0.66, title, ha="center", va="center", fontsize=7.5, fontweight="bold")
        fig.text(x0 + bw / 2, by + bh * 0.28, sub, ha="center", va="center", fontsize=6.5, color="#555")
        if i < len(boxes) - 1:
            fig.patches.append(FancyArrowPatch((x0 + bw + 0.003, by + bh / 2), (x0 + bw + gap - 0.003, by + bh / 2),
                                               transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7,
                                               color="#5f6368", lw=0.8))

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"teaser.{ext}", dpi=300)
    print(f"wrote {OUT / 'teaser.png'} and .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
