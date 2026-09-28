"""Damped least-squares inverse kinematics and home-configuration generation.

Used for two things:

* placing the robot in a repeatable "parked at the trocar" configuration for a
  given instrument / trocar geometry (``solve_home``);
* an optional Cartesian tip-target mode used for comparisons and tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vrcm.rcm import unit
from vrcm.robot import Kinematics, Robot
from vrcm.specs import RobotSpec


def rotation_vector(R_err: np.ndarray) -> np.ndarray:
    """Axis-angle (rotation) vector of a rotation matrix."""
    R_err = np.asarray(R_err, dtype=float)
    cos_theta = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-9:
        return np.array([R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]]) / 2.0
    axis = np.array(
        [R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]]
    ) / (2.0 * np.sin(theta))
    return theta * axis


@dataclass
class IkResult:
    q: np.ndarray
    pos_error: float
    rot_error: float
    iterations: int
    converged: bool


def _solve_least_squares(
    residual,
    jacobian,
    q0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_iter: int,
    tol: float,
    posture_gain: float = 0.0,
    q_seed: np.ndarray | None = None,
) -> tuple[np.ndarray, float, int, bool]:
    """Bounded non-linear least squares on the joint vector.

    Uses SciPy's trust-region reflective solver when available (robust and
    bounded), and otherwise a Levenberg-Marquardt iteration with the same
    interface.
    """
    q_seed = q0.copy() if q_seed is None else np.asarray(q_seed, dtype=float)
    n = q0.shape[0]

    def full_residual(q: np.ndarray) -> np.ndarray:
        r = np.asarray(residual(q), dtype=float)
        if posture_gain > 0.0:
            r = np.concatenate([r, posture_gain * (q - q_seed)])
        return r

    def full_jacobian(q: np.ndarray) -> np.ndarray:
        J = np.asarray(jacobian(q), dtype=float)
        if posture_gain > 0.0:
            J = np.vstack([J, posture_gain * np.eye(n)])
        return J

    try:
        from scipy.optimize import least_squares

        sol = least_squares(
            full_residual, q0, jac=full_jacobian, bounds=(lower, upper),
            method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=max_iter,
        )
        cost = float(np.linalg.norm(residual(sol.x)))
        return np.asarray(sol.x, dtype=float), cost, int(sol.nfev), bool(cost < 1e-5 or sol.success)
    except ImportError:
        pass

    # --- dependency-free Levenberg-Marquardt fallback ----------------------
    q = np.array(q0, dtype=float)
    r = full_residual(q)
    cost = float(np.linalg.norm(r))
    lam = 1e-4
    nu = 2.0
    for it in range(1, max_iter + 1):
        if cost < tol:
            return q, cost, it, True
        J = full_jacobian(q)
        A = J.T @ J
        g = J.T @ r
        accepted = False
        for _ in range(30):
            try:
                dq = np.linalg.solve(A + lam * np.diag(np.diag(A)) + 1e-12 * np.eye(A.shape[0]), -g)
            except np.linalg.LinAlgError:
                dq = -g
            q_new = np.clip(q + dq, lower, upper)
            r_new = full_residual(q_new)
            cost_new = float(np.linalg.norm(r_new))
            if cost_new < cost:
                predicted = float(g @ dq + 0.5 * dq @ A @ dq)
                rho = (cost ** 2 - cost_new ** 2) / max(-2.0 * predicted, 1e-12)
                lam = lam * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                nu = 2.0
                q, r, cost = q_new, r_new, cost_new
                accepted = True
                break
            lam *= nu
            nu *= 2.0
        if not accepted:
            break
    return q, cost, it, bool(cost < tol)


def ik_pose(
    robot: Robot,
    p_des: np.ndarray,
    R_des: np.ndarray,
    q0: np.ndarray | None = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 400,
    damping: float = 0.02,
    step_size: float = 0.6,
    posture_gain: float = 0.02,
) -> IkResult:
    """Solve for the joint angles that put the instrument tip at ``p_des`` / ``R_des``.

    The redundancy of 7-DOF arms (and of the 6-DOF arms near singularities) is
    resolved by a posture term pulling towards the initial configuration.
    """
    q_seed = np.array(robot.spec.home_qpos if q0 is None else q0, dtype=float)
    p_des = np.asarray(p_des, dtype=float)
    R_des = np.asarray(R_des, dtype=float)

    # Residual is written as "current - desired" with a positively signed
    # Jacobian row, so that J == d(residual)/dq for the solvers below.
    def residual(q):
        kin = robot.kinematics(q)
        return np.concatenate([kin.p_tip - p_des, rotation_vector(kin.R_tip @ R_des.T)])

    def jacobian(q):
        kin = robot.kinematics(q)
        J = np.vstack([kin.Jv, kin.Jw])
        return J

    q, _, iterations, _ = _solve_least_squares(
        residual, jacobian, q_seed, robot.joint_lower, robot.joint_upper,
        max_iter=max_iter, tol=max(tol, 1e-9), posture_gain=posture_gain, q_seed=q_seed,
    )
    kin = robot.kinematics(q)
    return IkResult(
        q=q,
        pos_error=float(np.linalg.norm(kin.p_tip - p_des)),
        rot_error=float(np.linalg.norm(rotation_vector(R_des @ kin.R_tip.T))),
        iterations=iterations,
        converged=float(np.linalg.norm(kin.p_tip - p_des)) < tol
        and float(np.linalg.norm(rotation_vector(R_des @ kin.R_tip.T))) < tol,
    )


def ik_shaft(
    robot: Robot,
    p_des: np.ndarray,
    d_des: np.ndarray,
    q0: np.ndarray | None = None,
    *,
    tol_pos: float = 1e-6,
    tol_dir: float = 1e-6,
    max_iter: int = 600,
    damping: float = 1e-3,
    step_size: float = 0.8,
    posture_gain: float = 1e-3,
) -> IkResult:
    """Put the tip at ``p_des`` with the shaft aligned to ``d_des``.

    Only 5 task degrees of freedom are constrained (3 position + 2 alignment):
    the roll of the instrument about its own axis is free, which is exactly what
    the RCM constraint cares about.  A Levenberg-Marquardt damping schedule makes
    the iteration robust near singularities; the redundancy is resolved by a
    posture term towards ``q0``.
    """
    q_seed = np.array(robot.spec.home_qpos if q0 is None else q0, dtype=float)
    p_des = np.asarray(p_des, dtype=float)
    d_des = unit(np.asarray(d_des, dtype=float))
    u0, v0 = _orthonormal_plane(d_des)

    # Residual "current - desired" (see ik_pose for the sign convention).
    def residual(q: np.ndarray) -> np.ndarray:
        kin = robot.kinematics(q)
        return np.concatenate([
            kin.p_tip - p_des,
            [float(u0 @ kin.d), float(v0 @ kin.d)],
        ])

    def jacobian(q: np.ndarray) -> np.ndarray:
        kin = robot.kinematics(q)
        return np.vstack([
            kin.Jv,
            np.cross(kin.d, u0) @ kin.Jw,
            np.cross(kin.d, v0) @ kin.Jw,
        ])

    q, _, iterations, _ = _solve_least_squares(
        residual, jacobian, q_seed, robot.joint_lower, robot.joint_upper,
        max_iter=max_iter, tol=max(tol_pos, tol_dir), posture_gain=posture_gain, q_seed=q_seed,
    )
    kin = robot.kinematics(q)
    res = residual(q)
    pos_error = float(np.linalg.norm(res[:3]))
    dir_error = float(np.linalg.norm(res[3:]))
    return IkResult(
        q=q,
        pos_error=pos_error,
        rot_error=dir_error,
        iterations=iterations,
        converged=pos_error < tol_pos and dir_error < tol_dir,
    )


def _orthonormal_plane(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = unit(d)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(d, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = unit(np.cross(d, ref))
    v = np.cross(d, u)
    return u, v


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap joint angles to (-pi, pi] so joint-space distances are meaningful."""
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def solve_home(
    spec: RobotSpec,
    *,
    d_des: np.ndarray | None = None,
    q0: np.ndarray | None = None,
    restarts: int = 6,
    seed: int = 0,
) -> IkResult:
    """Configuration in which the shaft passes through the trocar at the nominal depth.

    A few random restarts are used because a 6-DOF arm may need a different IK
    branch than the one the seed configuration leads to.  The accepted solution
    is the one with the smallest error that keeps the largest distance to the
    joint limits.
    """
    robot = Robot(spec)
    d_des = np.array([0.0, 0.0, -1.0]) if d_des is None else unit(np.asarray(d_des, dtype=float))
    p_rcm = np.asarray(spec.scene.trocar, dtype=float)
    p_tip = p_rcm + spec.scene.home_insertion * d_des
    return solve_ik_multistart(
        robot, p_tip, d_des, q0=q0, restarts=restarts, seed=seed
    )


def solve_ik_multistart(
    robot: Robot,
    p_des: np.ndarray,
    d_des: np.ndarray,
    *,
    q0: np.ndarray | None = None,
    restarts: int = 6,
    seed: int = 0,
    polish: bool = True,
    **kwargs,
) -> IkResult:
    """Solve :func:`ik_shaft` from several seeds and keep the best solution.

    Inverse kinematics of a 6-DOF arm has several branches (including the
    "shaft pointing the other way" solution), and a single start can land in a
    local minimum, so a handful of restarts is the pragmatic choice for an
    offline home-pose calculation.
    """
    seeds = [np.asarray(robot.spec.home_qpos if q0 is None else q0, dtype=float)]
    rng = np.random.default_rng(seed)
    for _ in range(max(restarts - 1, 0)):
        lo = np.maximum(robot.joint_lower, -np.pi)
        hi = np.minimum(robot.joint_upper, np.pi)
        seeds.append(rng.uniform(lo, hi))

    best: IkResult | None = None
    best_score = np.inf
    q_ref = seeds[0]
    for s in seeds:
        res = ik_shaft(robot, p_des, d_des, q0=s, **kwargs)
        # among the solutions that solve the task, prefer the one that is close
        # to the reference configuration and far from the joint limits
        joint_travel = float(np.linalg.norm(_wrap(res.q - q_ref)))
        score = res.pos_error + res.rot_error + 1e-3 * joint_travel
        if res.converged:
            score -= 1e-3 * float(np.min(robot.limit_margin(res.q)))
        if score < best_score:
            best, best_score = res, score
        if res.converged and score <= 1e-6:
            break
    assert best is not None
    if polish:
        # The posture term trades task accuracy for staying close to the seed,
        # leaving a residual of the order of the posture weight; polishing
        # without it drives the tip exactly onto the RCM axis.
        refined = ik_shaft(robot, p_des, d_des, q0=best.q, posture_gain=0.0, **kwargs)
        if refined.pos_error <= best.pos_error and refined.rot_error <= best.rot_error + 1e-9:
            best = refined
    return best
