"""Stop-and-go active viewpoint selection (section 11 of ``method.md``).

The upper level sees the surgical scene through the instrument (or an endoscope
sharing its shaft).  At every stop it

1. **scores** the current viewpoint with a visibility model,
2. **proposes** a coarse viewpoint action (pitch / yaw / insertion),
3. moves there with the V-RCM controller and stops,
4. refines the search around the best coarse viewpoint.

The visibility model here is geometric (distance + field-of-view cone + a
self-occlusion term), so the whole loop runs head-less and deterministically in
tests.  Swapping in a rendered or RGB-D based model only requires implementing
the same ``score(p_tip, d)`` interface -- see :class:`VisibilityModel`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vrcm.rcm import unit


@dataclass
class VisibilityModel:
    """Geometric stand-in for a perception/utility module.

    ``target`` is the point the surgeon/planner wants to look at.  The score is
    high when the target is centred in the field of view, at a comfortable
    distance, and not hidden behind the instrument shaft itself.
    """

    target: np.ndarray
    fov_deg: float = 40.0
    ideal_distance: float = 0.12
    distance_sigma: float = 0.06
    occlusion_radius: float = 0.02

    def score(self, p_tip: np.ndarray, d: np.ndarray) -> float:
        d = unit(d)
        to_target = np.asarray(self.target, dtype=float) - np.asarray(p_tip, dtype=float)
        dist = float(np.linalg.norm(to_target))
        if dist < 1e-9:
            return 0.0
        cos_angle = float(np.clip(np.dot(to_target / dist, d), -1.0, 1.0))
        angle = float(np.arccos(cos_angle))
        # smooth, monotone field-of-view weighting: 1 when centred, decaying over
        # roughly half the field of view
        centring = float(np.exp(-0.5 * (angle / (np.deg2rad(self.fov_deg) / 2.5)) ** 2))
        distance = np.exp(-0.5 * ((dist - self.ideal_distance) / self.distance_sigma) ** 2)
        lateral = dist * np.sin(angle)
        occlusion = 1.0 - np.exp(-0.5 * (lateral / self.occlusion_radius) ** 2)
        return float(centring * distance * (0.35 + 0.65 * occlusion))


@dataclass
class StopAndGoResult:
    """Record of one coarse-to-fine viewpoint search."""

    #: candidate viewpoints as ``(pitch_deg, yaw_deg, depth_mm)``
    candidates: list[tuple[float, float, float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)
    best_viewpoint: tuple[float, float, float] | None = None
    best_score: float = -np.inf
    #: search order for display
    def table(self, limit: int | None = None) -> str:
        rows = ["  phase  pitch[deg]  yaw[deg]  depth[mm]     score"]
        items = list(zip(self.phases, self.candidates, self.scores))
        if limit is not None:
            items = sorted(items, key=lambda it: -it[2])[:limit]
        for phase, cand, score in items:
            rows.append(f"  {phase:6s} {cand[0]:+10.2f} {cand[1]:+9.2f} {cand[2]:+10.2f}  {score:8.4f}")
        return "\n".join(rows)


def stop_and_go_scan(
    sim,
    planner,
    visibility: VisibilityModel,
    *,
    pitch_span: float = np.deg2rad(12.0),
    yaw_span: float = np.deg2rad(12.0),
    depth_span: float = 0.02,
    coarse: int = 3,
    fine: int = 3,
    fine_span_factor: float = 0.35,
) -> StopAndGoResult:
    """Coarse-to-fine viewpoint search executed through the V-RCM controller.

    Candidate viewpoints are *absolute* ``(pitch, yaw, depth)`` offsets from the
    planner's nominal direction.  For every candidate the instrument is moved
    there, brought to rest, and scored by ``visibility``: that is one
    stop-and-go cycle.
    """
    result = StopAndGoResult()
    pitch0, yaw0, depth0 = planner.goal_pitch, planner.goal_yaw, planner.goal_depth

    def evaluate(pitch: float, yaw: float, depth: float, phase: str) -> float:
        planner.goto(pitch, yaw, depth)
        sim.settle(planner)
        state = sim.robot.kinematics()
        score = visibility.score(state.p_tip, state.d)
        result.candidates.append((np.degrees(pitch), np.degrees(yaw), depth * 1e3))
        result.scores.append(score)
        result.phases.append(phase)
        if score > result.best_score:
            result.best_score = score
            result.best_viewpoint = (np.degrees(pitch), np.degrees(yaw), depth)
        return score

    def sweep(centre, span_p, span_y, span_d, n, phase) -> None:
        for dp in np.linspace(-span_p, span_p, n):
            for dy in np.linspace(-span_y, span_y, n):
                for dd in np.linspace(-span_d, span_d, n):
                    evaluate(centre[0] + float(dp), centre[1] + float(dy),
                             centre[2] + float(dd), phase)

    sweep((pitch0, yaw0, depth0), pitch_span, yaw_span, depth_span, coarse, "coarse")

    if result.best_viewpoint is not None and fine > 1:
        best = np.array([np.deg2rad(result.best_viewpoint[0]),
                         np.deg2rad(result.best_viewpoint[1]),
                         result.best_viewpoint[2]])
        sweep(best, pitch_span * fine_span_factor, yaw_span * fine_span_factor,
              depth_span * fine_span_factor, fine, "fine")

    if result.best_viewpoint is not None:
        planner.goto(np.deg2rad(result.best_viewpoint[0]),
                     np.deg2rad(result.best_viewpoint[1]),
                     result.best_viewpoint[2])
        sim.settle(planner)
    return result
