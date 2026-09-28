"""QP based Virtual RCM controller (sections 7-9 of ``method.md``).

Every control cycle solves, for the joint velocity ``qdot``::

    min   ||J_w qdot - omega_des||^2  +  lambda ||qdot||^2  +  posture task
    s.t.  u^T (J_v + l [d]_x J_w) qdot = -k_rcm (u^T e_rcm)      (2 rows)
          d^T J_v qdot = v_ins                                   (1 row)
          joint velocity / position rate box
          insertion depth rate limits
          collision avoidance rows (optional)

The two RCM rows are the projections of the RCM constraint on an orthonormal
basis of the plane perpendicular to the shaft, so the equality block is always
full rank (the raw ``J_RCM`` has rank 2 out of 3 rows by construction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from vrcm.qp import available_backends, least_squares_objective, resolve_backend, solve_qp
from vrcm.rcm import (
    direction_error,
    insertion_jacobian,
    rcm_error,
    rcm_jacobian,
    rcm_length,
    skew,
    unit,
)
from vrcm.robot import Kinematics, Robot


@dataclass
class VrcmConfig:
    """Controller gains and safety limits."""

    dt: float = 0.01
    #: RCM error feedback gain [1/s]; 0 enforces the constraint exactly per step.
    k_rcm: float = 1.0
    #: Shaft direction feedback gain [1/s].
    k_orient: float = 2.0
    #: Insertion depth feedback gain [1/s].
    k_insertion: float = 2.0
    #: Joint velocity regularisation.
    lambda_reg: float = 1e-3
    #: Joint position limit margin [rad].
    joint_limit_margin: float = 0.02
    #: Secondary posture task gain [1/s] (0 disables the task).
    posture_gain: float = 0.0
    #: Weight of the posture task relative to the viewpoint task.
    posture_weight: float = 0.05
    #: Keep this much distance to the joint limits in the posture task [rad].
    posture_margin: float = 0.15
    #: Guard: refuse to move if the RCM error is larger than this [m].
    max_rcm_error: float = 0.05
    #: Collision avoidance (experimental; numeric gradients of mj_geomDistance).
    collision_pairs: tuple[tuple[str, str], ...] = ()
    collision_safe_distance: float = 0.01
    collision_weight_scaling: float = 1.0
    qp_backend: str = "auto"


@dataclass
class VrcmTarget:
    """Per-cycle command produced by the viewpoint planner or the surgeon."""

    #: Desired shaft direction in the world frame.  ``None`` keeps the current one.
    d_des: np.ndarray | None = None
    #: Feed-forward insertion velocity [m/s], positive = deeper into the cavity.
    insertion_velocity: float = 0.0
    #: Absolute insertion depth target [m] from the trocar to the tip.
    insertion_depth: float | None = None
    #: Roll rate about the shaft axis [rad/s].
    roll_velocity: float = 0.0


@dataclass
class VrcmResult:
    qdot: np.ndarray
    kin: Kinematics
    d_des: np.ndarray
    omega_des: np.ndarray
    length: float
    rcm_error: np.ndarray
    rcm_error_norm: float
    insertion_velocity: float
    depth_error: float
    converged: bool
    backend: str
    iterations: int
    active_constraints: int
    limited: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    notes: tuple[str, ...] = ()


class VrcmController:
    """Task-space velocity controller enforcing the virtual RCM."""

    def __init__(self, robot: Robot, config: VrcmConfig | None = None) -> None:
        self.robot = robot
        self.config = config or VrcmConfig()
        self.config.dt = float(self.config.dt)
        self._lam = None
        self._collision_geoms = [
            (self.robot.geom_id(a), self.robot.geom_id(b)) for a, b in self.config.collision_pairs
        ]

    def reset(self) -> None:
        """Clear the warm start and any per-session state."""
        self._lam = None

    # ------------------------------------------------------------------ helpers
    @property
    def backend(self) -> str:
        return resolve_backend(self.config.qp_backend)

    def available_backends(self) -> list[str]:
        return available_backends()

    def _plane_basis(self, d: np.ndarray) -> np.ndarray:
        """Orthonormal basis (u, v) of the plane perpendicular to the shaft."""
        d = unit(d)
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(d, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u = unit(np.cross(d, ref))
        v = np.cross(d, u)
        return u, v

    def _desired_angular_velocity(self, d_des: np.ndarray, d: np.ndarray, roll_rate: float) -> np.ndarray:
        err = direction_error(d_des, d)
        if np.linalg.norm(err) < 1e-12 and np.dot(d_des, d) > 0:
            return roll_rate * d  # already aligned
        if np.linalg.norm(err) < 1e-9 and np.dot(d_des, d) < 0:
            # exactly antiparallel: choose an arbitrary perpendicular rotation axis
            u, _ = self._plane_basis(d)
            err = -np.pi * u
        return self.config.k_orient * err + roll_rate * d

    def _box(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Joint velocity box from velocity limits and joint position limits."""
        cfg = self.config
        vmax = self.robot.joint_velocity_limits
        lo = np.maximum(-vmax, (self.robot.joint_lower + cfg.joint_limit_margin - q) / cfg.dt)
        hi = np.minimum(vmax, (self.robot.joint_upper - cfg.joint_limit_margin - q) / cfg.dt)
        # A joint that is already outside the margin band may only move back
        # inside, and the box always contains zero so a safe fallback exists.
        lo = np.minimum(lo, 0.0)
        hi = np.maximum(hi, 0.0)
        return lo, hi

    def _collision_rows(self, kin: Kinematics, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Linearised distance constraints ``grad d^T qdot >= (d_safe - d)/dt``."""
        rows, rhs = [], []
        n = self.robot.n_arm_joints
        q0 = kin.q.copy()
        for g1, g2 in self._collision_geoms:
            dist, _ = self.robot.geom_distance(g1, g2, max_distance=max(self.config.collision_safe_distance * 5, 0.2))
            if dist > 0 and dist > self.config.collision_safe_distance:
                continue
            grad = np.zeros(n)
            eps = 1e-6
            for i in range(n):
                qp_ = q0.copy()
                qp_[i] += eps
                self.robot.kinematics(qp_)
                d_plus, _ = self.robot.geom_distance(
                    g1, g2, max_distance=max(self.config.collision_safe_distance * 5, 0.2)
                )
                grad[i] = (d_plus - dist) / eps
            self.robot.kinematics(q0)
            rows.append(-grad)
            rhs.append((dist - self.config.collision_safe_distance) / dt)
        if not rows:
            return np.zeros((0, n)), np.zeros(0)
        return np.vstack(rows), np.asarray(rhs)

    # --------------------------------------------------------------------- step
    def step(self, q: Sequence[float], target: VrcmTarget) -> VrcmResult:
        cfg = self.config
        robot = self.robot
        dt = cfg.dt

        kin = robot.kinematics(q)
        d = kin.d
        p_rcm = np.asarray(robot.spec.trocar, dtype=float)
        l = rcm_length(kin.p_tip, d, p_rcm)
        e_rcm = rcm_error(kin.p_tip, d, p_rcm)
        e_norm = float(np.linalg.norm(e_rcm))

        d_des = d if target.d_des is None else unit(np.asarray(target.d_des, dtype=float))
        omega_des = self._desired_angular_velocity(d_des, d, target.roll_velocity)

        depth_error = 0.0
        v_ins = target.insertion_velocity
        if target.insertion_depth is not None:
            depth_error = float(target.insertion_depth - l)
            v_ins += cfg.k_insertion * depth_error

        notes: list[str] = []
        ins = robot.spec.instrument
        if target.insertion_depth is not None:
            v_ins = float(np.clip(v_ins, -ins.insertion_rate, ins.insertion_rate))

        n = robot.n_arm_joints

        # ---- equality constraints ------------------------------------------
        J_rcm = rcm_jacobian(kin.Jv, kin.Jw, d, l)
        u, v = self._plane_basis(d)
        A_eq = np.vstack([u @ J_rcm, v @ J_rcm, insertion_jacobian(kin.Jv, d)])
        b_eq = np.concatenate([-cfg.k_rcm * np.array([u @ e_rcm, v @ e_rcm]), [v_ins]])

        # ---- objective ------------------------------------------------------
        tasks = [(kin.Jw, omega_des, 1.0)]
        if cfg.posture_gain > 0.0:
            q_ref = np.clip(kin.q, robot.joint_lower + cfg.posture_margin,
                            robot.joint_upper - cfg.posture_margin)
            tasks.append((np.eye(n), cfg.posture_gain * (q_ref - kin.q), cfg.posture_weight))
        H, g = least_squares_objective(tasks, n)
        H = H + cfg.lambda_reg * np.eye(n)

        # ---- inequality constraints ----------------------------------------
        lo, hi = self._box(kin.q)
        rows = [np.eye(n), -np.eye(n)]
        rhs = [hi, -lo]

        # insertion depth must stay inside the instrument working range
        ins_row = insertion_jacobian(kin.Jv, d)
        rows.append(ins_row)
        rhs.append(np.array([(ins.max_insertion - l) / dt]))
        rows.append(-ins_row)
        rhs.append(np.array([(l - ins.min_insertion) / dt]))

        if self._collision_geoms:
            c_rows, c_rhs = self._collision_rows(kin, dt)
            if c_rows.size:
                rows.append(c_rows)
                rhs.append(c_rhs)

        A_ub = np.vstack(rows)
        b_ub = np.concatenate(rhs)

        if e_norm > cfg.max_rcm_error:
            notes.append(f"rcm error {e_norm * 1000:.1f} mm exceeds guard, holding position")
            return self._idle(kin, d_des, omega_des, l, e_rcm, v_ins, depth_error, notes)

        result = solve_qp(
            H, g, A_eq, b_eq, A_ub, b_ub,
            lam0=self._lam, backend=cfg.qp_backend,
        )
        qdot = np.asarray(result.x, dtype=float)

        # ---- safety net: never send an infeasible or non-finite command -----
        violated = (A_ub @ qdot) - b_ub
        worst = float(np.max(violated)) if violated.size else 0.0
        if not np.all(np.isfinite(qdot)) or worst > 1e-6:
            notes.append(f"QP residual {worst:.2e}: clamping to the feasible box")
            qdot = np.clip(qdot, lo, hi)
        if result.infeasible:
            notes.append("QP reported an infeasible constraint set")
            qdot = np.zeros(n)
        elif not result.converged:
            notes.append(f"QP backend {result.backend} did not converge in {result.iterations} iterations")

        self._lam = result.lam_ub

        return VrcmResult(
            qdot=qdot,
            kin=kin,
            d_des=d_des,
            omega_des=omega_des,
            length=l,
            rcm_error=e_rcm,
            rcm_error_norm=e_norm,
            insertion_velocity=v_ins,
            depth_error=depth_error,
            converged=bool(result.converged),
            backend=result.backend,
            iterations=int(result.iterations),
            active_constraints=int(np.sum(result.active)) if result.lam_ub.size else 0,
            limited=np.abs(qdot) >= (robot.joint_velocity_limits - 1e-9),
            notes=tuple(notes),
        )

    def _idle(self, kin, d_des, omega_des, l, e_rcm, v_ins, depth_error, notes) -> VrcmResult:
        return VrcmResult(
            qdot=np.zeros(self.robot.n_arm_joints),
            kin=kin,
            d_des=d_des,
            omega_des=omega_des,
            length=l,
            rcm_error=e_rcm,
            rcm_error_norm=float(np.linalg.norm(e_rcm)),
            insertion_velocity=v_ins,
            depth_error=depth_error,
            converged=False,
            backend=self.backend,
            iterations=0,
            active_constraints=0,
            notes=tuple(notes),
        )
