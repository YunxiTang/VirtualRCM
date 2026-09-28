"""Surgical viewpoint planner: ``(pitch, yaw, insertion)`` actions -> RCM targets.

This is the interface described in sections 10-11 of ``method.md``.  The
planner owns a *goal* viewpoint and produces a rate-limited *reference* that the
V-RCM controller tracks.  Because the reference is rate limited, a discrete
"stop-and-go" decision (move by 5 deg, wait, observe, decide again) is executed
as a smooth, continuous motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vrcm.rcm import direction_from_angles, unit


@dataclass
class ViewpointAction:
    """A viewpoint action ``a_t = [Δpitch, Δyaw, Δinsertion]`` (plus optional roll)."""

    pitch: float = 0.0
    yaw: float = 0.0
    insertion: float = 0.0
    roll: float = 0.0

    def copy(self) -> "ViewpointAction":
        return ViewpointAction(self.pitch, self.yaw, self.insertion, self.roll)

    def is_zero(self, tol: float = 1e-12) -> bool:
        return max(abs(self.pitch), abs(self.yaw), abs(self.insertion), abs(self.roll)) < tol


@dataclass
class ViewpointReference:
    """Continuous reference handed to the V-RCM controller."""

    d_des: np.ndarray
    insertion_depth: float
    insertion_velocity: float
    roll_velocity: float
    pitch: float
    yaw: float
    settled: bool


@dataclass
class ViewpointPlanner:
    """Rate-limited reference generator for coarse-to-fine viewpoint moves."""

    d_nominal: np.ndarray
    depth_nominal: float
    #: Limits of the surgical cone around the nominal direction.
    pitch_limit: float = np.deg2rad(35.0)
    yaw_limit: float = np.deg2rad(35.0)
    #: Instrument working range (trocar -> tip distance).
    depth_limits: tuple[float, float] = (0.05, 0.22)
    #: Slew limits.
    max_pitch_rate: float = np.deg2rad(25.0)
    max_yaw_rate: float = np.deg2rad(25.0)
    max_depth_rate: float = 0.04
    max_roll_rate: float = np.deg2rad(45.0)
    #: Tolerance for "settled".
    pitch_tol: float = np.deg2rad(0.05)
    depth_tol: float = 1e-4
    reference_axis: np.ndarray | None = None

    pitch: float = 0.0
    yaw: float = 0.0
    depth: float = 0.0
    roll: float = 0.0
    goal_pitch: float = 0.0
    goal_yaw: float = 0.0
    goal_depth: float = 0.0
    goal_roll: float = 0.0
    _history: list[ViewpointAction] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.d_nominal = unit(np.asarray(self.d_nominal, dtype=float))
        self.goal_depth = float(np.clip(self.depth_nominal, *self.depth_limits))
        self.depth = self.goal_depth

    # ------------------------------------------------------------------- goals
    def command(self, action: ViewpointAction) -> None:
        """Apply an incremental action to the goal viewpoint."""
        self.goal_pitch = float(np.clip(self.goal_pitch + action.pitch, -self.pitch_limit, self.pitch_limit))
        self.goal_yaw = float(np.clip(self.goal_yaw + action.yaw, -self.yaw_limit, self.yaw_limit))
        self.goal_depth = float(np.clip(self.goal_depth + action.insertion, *self.depth_limits))
        self.goal_roll += action.roll
        self._history.append(action.copy())

    def goto(self, pitch: float, yaw: float, depth: float | None = None, roll: float | None = None) -> None:
        """Set an absolute goal viewpoint."""
        self.goal_pitch = float(np.clip(pitch, -self.pitch_limit, self.pitch_limit))
        self.goal_yaw = float(np.clip(yaw, -self.yaw_limit, self.yaw_limit))
        if depth is not None:
            self.goal_depth = float(np.clip(depth, *self.depth_limits))
        if roll is not None:
            self.goal_roll = float(roll)

    def reset(self) -> None:
        self.pitch = self.yaw = self.roll = 0.0
        self.depth = self.goal_depth
        self.goal_pitch = self.goal_yaw = 0.0
        self._history.clear()

    # -------------------------------------------------------------------- step
    def step(self, dt: float) -> ViewpointReference:
        """Advance the reference by ``dt`` and return the new target."""
        d_pitch = float(np.clip(self.goal_pitch - self.pitch, -self.max_pitch_rate * dt, self.max_pitch_rate * dt))
        d_yaw = float(np.clip(self.goal_yaw - self.yaw, -self.max_yaw_rate * dt, self.max_yaw_rate * dt))
        d_depth = float(np.clip(self.goal_depth - self.depth, -self.max_depth_rate * dt, self.max_depth_rate * dt))
        d_roll = float(np.clip(self.goal_roll - self.roll, -self.max_roll_rate * dt, self.max_roll_rate * dt))

        self.pitch += d_pitch
        self.yaw += d_yaw
        self.depth += d_depth
        self.roll += d_roll

        d_des = direction_from_angles(self.d_nominal, self.pitch, self.yaw, self.reference_axis)
        settled = (
            abs(self.goal_pitch - self.pitch) < self.pitch_tol
            and abs(self.goal_yaw - self.yaw) < self.pitch_tol
            and abs(self.goal_depth - self.depth) < self.depth_tol
        )
        return ViewpointReference(
            d_des=d_des,
            insertion_depth=self.depth,
            insertion_velocity=d_depth / dt if dt > 0 else 0.0,
            roll_velocity=d_roll / dt if dt > 0 else 0.0,
            pitch=self.pitch,
            yaw=self.yaw,
            settled=settled,
        )

    # --------------------------------------------------------------- utilities
    def scan_actions(
        self,
        pitch_span: float = np.deg2rad(20.0),
        yaw_span: float = np.deg2rad(20.0),
        rows: int = 3,
        cols: int = 3,
        depth_steps: tuple[float, ...] = (0.0,),
    ) -> list[ViewpointAction]:
        """Coarse scan pattern around the current goal (used by the demo)."""
        actions: list[ViewpointAction] = []
        for dp in np.linspace(-pitch_span, pitch_span, rows):
            for dy in np.linspace(-yaw_span, yaw_span, cols):
                for dd in depth_steps:
                    actions.append(ViewpointAction(pitch=dp, yaw=dy, insertion=dd))
        return actions
