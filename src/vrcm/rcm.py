"""Remote centre of motion kinematics (equations 3-8 of ``method.md``).

Conventions
-----------
``p_tip``
    world position of the instrument tip (the reference point ``p_e``)
``d``
    unit vector along the instrument shaft, pointing from the flange towards the
    tip; ``d = R_tip[:, 2]``
``p_rcm``
    the fixed trocar / RCM point ``p_r``
``l``
    signed distance from the RCM point to the tip along the shaft, ``l = d . (p_tip - p_rcm)``

The RCM constraint is ``(I - d d^T) (p_tip - p_rcm) = 0``: the trocar must lie on
the shaft axis, while sliding along the axis (insertion) stays free.
"""

from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix ``[v]_x`` with ``[v]_x x = v x``."""
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("cannot normalise a zero vector")
    return v / n


def projector(d: np.ndarray) -> np.ndarray:
    """``I - d d^T``: projection onto the plane orthogonal to the shaft."""
    d = np.asarray(d, dtype=float).reshape(3)
    return np.eye(3) - np.outer(d, d)


def rcm_length(p_tip: np.ndarray, d: np.ndarray, p_rcm: np.ndarray) -> float:
    """Signed distance from the RCM point to the tip along the shaft."""
    return float(np.dot(np.asarray(d, dtype=float), np.asarray(p_tip, dtype=float) - np.asarray(p_rcm, dtype=float)))


def rcm_error(p_tip: np.ndarray, d: np.ndarray, p_rcm: np.ndarray) -> np.ndarray:
    """Perpendicular offset of the trocar from the shaft axis, ``e_RCM``."""
    r = np.asarray(p_tip, dtype=float) - np.asarray(p_rcm, dtype=float)
    d = np.asarray(d, dtype=float)
    return r - np.dot(d, r) * d


def shaft_point(p_tip: np.ndarray, d: np.ndarray, p_rcm: np.ndarray) -> np.ndarray:
    """Point on the shaft axis closest to the trocar (the *realised* RCM point)."""
    p_tip = np.asarray(p_tip, dtype=float)
    return p_tip - rcm_length(p_tip, d, p_rcm) * np.asarray(d, dtype=float)


def rcm_jacobian(Jv: np.ndarray, Jw: np.ndarray, d: np.ndarray, l: float) -> np.ndarray:
    """``J_RCM = (I - d d^T) (J_v + l [d]_x J_w)``."""
    P = projector(d)
    return P @ (np.asarray(Jv, dtype=float) + l * skew(d) @ np.asarray(Jw, dtype=float))


def insertion_jacobian(Jv: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Row vector ``d^T J_v`` that maps joint velocity to insertion velocity."""
    return np.asarray(d, dtype=float) @ np.asarray(Jv, dtype=float)


def direction_error(d_des: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Shaft direction error ``d x d_des`` used by the viewpoint controller.

    The shaft direction evolves as ``d_dot = omega x d``, so the angular velocity
    that rotates ``d`` towards ``d_des`` is proportional to ``d x d_des``.

    .. note::
       ``method.md`` (section 6) writes ``e_d = d_des x d`` together with
       ``omega_des = +k_R e_d``.  With the standard convention ``d_dot = omega x d``
       that rotates the shaft *away* from the target; the sign below is the one
       that makes the closed loop converge (equivalently ``omega_des = -k_R e_d``
       with the notation of ``method.md``).
    """
    return np.cross(np.asarray(d, dtype=float), np.asarray(d_des, dtype=float))


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a, b = unit(a), unit(b)
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about ``axis`` by ``angle`` radians."""
    k = unit(axis)
    K = skew(k)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def shaft_frame(d: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Orthonormal frame whose third axis is ``d``.

    Returns a 3x3 matrix ``F = [e_x e_y d]``; ``e_x`` is chosen to be
    perpendicular to both ``d`` and ``reference``.  This frame defines the
    pitch/yaw axes used by the viewpoint planner.
    """
    d = unit(d)
    if reference is None:
        reference = np.array([1.0, 0.0, 0.0])
    reference = np.asarray(reference, dtype=float)
    ex = np.cross(d, reference)
    if np.linalg.norm(ex) < 1e-8:
        reference = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(d, reference)) > 0.9:
            reference = np.array([0.0, 0.0, 1.0])
        ex = np.cross(d, reference)
    ex = unit(ex)
    ey = np.cross(d, ex)
    return np.column_stack([ex, ey, d])


def direction_from_angles(
    d_nominal: np.ndarray, pitch: float, yaw: float, reference: np.ndarray | None = None
) -> np.ndarray:
    """Shaft direction obtained by pitching/yawing the nominal direction.

    ``pitch`` rotates about the first axis of :func:`shaft_frame`, ``yaw`` about
    the second; both angles are absolute (not increments) relative to
    ``d_nominal``.
    """
    d_nominal = unit(d_nominal)
    F = shaft_frame(d_nominal, reference)
    ex, ey = F[:, 0], F[:, 1]
    R = rotation_about(ex, pitch) @ rotation_about(ey, yaw)
    return unit(R @ d_nominal)
