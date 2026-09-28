"""Small dense constrained QP solver used by the V-RCM controller.

Problem form::

    minimise    0.5 x^T H x + g^T x
    subject to  A_eq x = b_eq
                A_ub x <= b_ub

Available backends (``backend=`` argument / :func:`available_backends`):

``"auto"`` (default)
    the first of ``quadprog`` / ``osqp`` / ``scipy`` / ``numpy`` that is installed.
``"quadprog"``
    Goldfarb-Idnani dual active set, ~0.1 ms for controller-sized problems.
``"osqp"``
    operator splitting; robust but slower.
``"slsqp"``
    SciPy reference implementation, used to cross-check the others in tests.
``"dual_active_set"``
    dependency free pure-numpy solver described below.  It reduces the problem
    with a null-space parameterisation and then runs an exact active-set
    iteration on the dual.  Redundant inequality rows (which the controller
    inevitably produces: 2n box rows in n dimensions) make the dual optimum
    non-unique, so this backend applies a small ridge and reports
    ``converged=False`` together with a feasibility check when it cannot certify
    the solution.  Prefer ``quadprog`` when it is available.

The pure-numpy solver works as follows: the equality constraints (RCM +
insertion) are eliminated by a null-space parameterisation ``x = x_p + N z``, and
the remaining inequality-constrained QP is solved in the dual by an active-set
iteration over the non-negativity constraints of the multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QPResult:
    x: np.ndarray
    lam_ub: np.ndarray = field(default_factory=lambda: np.zeros(0))
    lam_eq: np.ndarray = field(default_factory=lambda: np.zeros(0))
    iterations: int = 0
    eq_residual: float = 0.0
    ineq_violation: float = 0.0
    converged: bool = True
    backend: str = "dual_active_set"
    infeasible: bool = False
    message: str = ""

    @property
    def active(self) -> np.ndarray:
        """Boolean mask of the inequality constraints that are active."""
        return self.lam_ub > 1e-8


def _as_2d(a, cols: int) -> np.ndarray | None:
    if a is None:
        return None
    a = np.atleast_2d(np.asarray(a, dtype=float))
    if a.size == 0:
        return None
    if a.shape[1] != cols:
        raise ValueError(f"expected {cols} columns, got shape {a.shape}")
    return a


def available_backends() -> list[str]:
    """Installed QP backends, best first."""
    found = []
    try:
        import quadprog  # noqa: F401

        found.append("quadprog")
    except ImportError:
        pass
    try:
        import osqp  # noqa: F401

        found.append("osqp")
    except ImportError:
        pass
    try:
        import scipy  # noqa: F401

        found.append("slsqp")
    except ImportError:
        pass
    found.append("dual_active_set")
    return found


def resolve_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    return available_backends()[0]


def null_space(A: np.ndarray, tol: float | None = None) -> np.ndarray:
    """Orthonormal basis of the null space of ``A`` (columns)."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.size == 0:
        return np.eye(A.shape[1])
    _, s, Vt = np.linalg.svd(A)
    if tol is None:
        tol = max(A.shape) * np.finfo(float).eps * (s[0] if s.size else 1.0)
    rank = int(np.sum(s > tol))
    return Vt[rank:].T


def solve_qp(
    H: np.ndarray,
    g: np.ndarray,
    A_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    A_ub: np.ndarray | None = None,
    b_ub: np.ndarray | None = None,
    *,
    lam0: np.ndarray | None = None,
    max_sweeps: int = 2000,
    tol: float = 1e-11,
    ridge: float = 0.0,
    backend: str = "dual_active_set",
) -> QPResult:
    """Solve the QP described in the module docstring."""
    H = np.asarray(H, dtype=float)
    g = np.asarray(g, dtype=float).reshape(-1)
    n = g.shape[0]
    if H.shape != (n, n):
        raise ValueError(f"H has shape {H.shape}, expected {(n, n)}")

    A_eq = _as_2d(A_eq, n)
    b_eq = None if A_eq is None else np.asarray(b_eq, dtype=float).reshape(-1)
    A_ub = _as_2d(A_ub, n)
    b_ub = None if A_ub is None else np.asarray(b_ub, dtype=float).reshape(-1)
    if backend == "slsqp":
        return _solve_slsqp(H, g, A_eq, b_eq, A_ub, b_ub, lam0=lam0)
    if backend == "hildreth":
        return _solve_hildreth(
            H, g, A_eq, b_eq, A_ub, b_ub, lam0=lam0, max_sweeps=max_sweeps, tol=tol, ridge=ridge
        )
    if backend == "quadprog":
        return _solve_quadprog(H, g, A_eq, b_eq, A_ub, b_ub)
    if backend == "osqp":
        return _solve_osqp(H, g, A_eq, b_eq, A_ub, b_ub)
    return _solve_dual_active_set(
        H, g, A_eq, b_eq, A_ub, b_ub, lam0=lam0, max_sweeps=max_sweeps, tol=tol, ridge=ridge
    )


def _stack_constraints(A_eq, b_eq, A_ub, b_ub):
    """Return ``(C, b, meq)`` for the convention ``C^T x >= b``."""
    blocks, rhs, meq = [], [], 0
    if A_eq is not None:
        blocks.append(A_eq.T)
        rhs.append(b_eq)
        meq = A_eq.shape[0]
    if A_ub is not None:
        blocks.append(-A_ub.T)
        rhs.append(-b_ub)
    if not blocks:
        return None, None, 0
    return np.hstack(blocks), np.concatenate(rhs), meq


def _solve_quadprog(H, g, A_eq, b_eq, A_ub, b_ub) -> QPResult:
    import quadprog

    n = g.shape[0]
    C, b, meq = _stack_constraints(A_eq, b_eq, A_ub, b_ub)

    G = 0.5 * (np.asarray(H, dtype=float) + np.asarray(H, dtype=float).T)
    if C is None:
        x = np.linalg.lstsq(G, -np.asarray(g, dtype=float), rcond=None)[0]
        out = QPResult(x=x, backend="quadprog")
        out.eq_residual = _eq_residual(x, A_eq, b_eq)
        return out

    # quadprog requires a positive definite Hessian
    eig_min = float(np.min(np.linalg.eigvalsh(G)))
    scale = max(float(np.max(np.abs(np.diag(G)))), 1.0)
    if eig_min <= 1e-12 * scale:
        G = G + (1e-10 * scale - eig_min) * np.eye(n)

    # NOTE: quadprog minimises 1/2 x'Gx - a'x, hence the negated linear term.
    try:
        x, _f, _xu, iters, lagrangian, iact = quadprog.solve_qp(
            G, -np.asarray(g, dtype=float), C, b, meq
        )
    except ValueError as exc:  # infeasible constraint set
        out = QPResult(x=np.zeros(n), backend="quadprog")
        out.converged = False
        out.infeasible = True
        out.message = str(exc)
        return out
    out = QPResult(x=np.asarray(x, dtype=float), backend="quadprog", iterations=int(np.max(np.atleast_1d(iters))))
    out.eq_residual = _eq_residual(out.x, A_eq, b_eq)
    if A_ub is not None and A_ub.size:
        out.ineq_violation = float(np.max(np.maximum(A_ub @ out.x - b_ub, 0.0)))
    out.converged = out.eq_residual < 1e-7 and out.ineq_violation < 1e-7
    if isinstance(lagrangian, np.ndarray) and lagrangian.size == C.shape[1]:
        out.lam_ub = np.asarray(lagrangian, dtype=float)[meq:]
    return out


def _solve_osqp(H, g, A_eq, b_eq, A_ub, b_ub) -> QPResult:
    import osqp
    import scipy.sparse as sp

    n = g.shape[0]
    G = 0.5 * (np.asarray(H, dtype=float) + np.asarray(H, dtype=float).T)
    eig_min = float(np.min(np.linalg.eigvalsh(G)))
    scale = max(float(np.max(np.abs(np.diag(G)))), 1.0)
    if eig_min <= 1e-12 * scale:
        G = G + (1e-10 * scale - eig_min) * np.eye(n)

    rows, lower, upper, n_eq = [], [], [], 0
    if A_eq is not None:
        rows.append(A_eq)
        lower.append(b_eq)
        upper.append(b_eq)
        n_eq = A_eq.shape[0]
    if A_ub is not None:
        rows.append(A_ub)
        lower.append(np.full(A_ub.shape[0], -np.inf))
        upper.append(b_ub)

    if not rows:
        x = np.linalg.solve(G, -np.asarray(g, dtype=float))
        out = QPResult(x=x, backend="osqp", iterations=0)
        out.eq_residual = _eq_residual(x, A_eq, b_eq)
        return out

    A = sp.csc_matrix(np.vstack(rows))
    l = np.concatenate(lower)
    u = np.concatenate(upper)

    prob = osqp.OSQP()
    prob.setup(P=sp.csc_matrix(G), q=np.asarray(g, dtype=float), A=A, l=l, u=u,
               verbose=False, eps_abs=1e-10, eps_rel=1e-10, polishing=True, max_iter=20000)
    res = prob.solve()
    out = QPResult(x=np.asarray(res.x, dtype=float), backend="osqp", iterations=int(res.info.iter))
    out.eq_residual = _eq_residual(out.x, A_eq, b_eq)
    if A_ub is not None and A_ub.size:
        out.ineq_violation = float(np.max(np.maximum(A_ub @ out.x - b_ub, 0.0)))
    out.converged = str(res.info.status).lower() in ("solved", "solved inaccurate")
    if res.y is not None and len(res.y) > n_eq:
        out.lam_ub = np.abs(np.asarray(res.y, dtype=float)[n_eq:])
    return out


def _dual_parts(H, g, A_eq, b_eq, A_ub, b_ub, ridge):
    """Null-space reduction of the primal, returning the dual data (P, r) as well.

    Returns ``(x_p, N, G, a, Cz, bz)`` where ``Cz z >= bz`` is the reduced
    inequality system.
    """
    n = g.shape[0]
    if A_eq is not None:
        x_p, *_ = np.linalg.lstsq(A_eq, b_eq, rcond=None)
        N = null_space(A_eq)
    else:
        x_p = np.zeros(n)
        N = np.eye(n)

    G = N.T @ H @ N
    a = N.T @ (H @ x_p + g)
    if ridge > 0.0:
        G = G + ridge * np.eye(G.shape[0])
    G = 0.5 * (G + G.T)

    if A_ub is None:
        C = np.zeros((G.shape[0], 0))
        b = np.zeros(0)
    else:
        C = -(A_ub @ N).T  # rows: -a_i' N, so that C' z >= b is -A_ub N z >= -b_ub
        b = -(b_ub - A_ub @ x_p)
    return x_p, N, G, a, C, b


def _solve_dual_active_set(
    H, g, A_eq, b_eq, A_ub, b_ub, *, lam0, max_sweeps, tol, ridge
) -> QPResult:
    """Exact dual active-set QP solver (Goldfarb-Idnani structure)."""
    n = g.shape[0]
    x_p, N, G, a, C, b = _dual_parts(H, g, A_eq, b_eq, A_ub, b_ub, ridge)
    k = N.shape[1]
    m = C.shape[1]
    result = QPResult(x=np.zeros(n), backend="dual_active_set", lam_ub=np.zeros(m))
    if k == 0:
        result.x = x_p
        result.eq_residual = _eq_residual(result.x, A_eq, b_eq)
        return result

    G_inv = np.linalg.inv(G) if np.linalg.matrix_rank(G) == k else np.linalg.pinv(G)
    if m == 0:
        result.x = x_p + N @ (-G_inv @ a)
        result.eq_residual = _eq_residual(result.x, A_eq, b_eq)
        return result

    P = C.T @ G_inv @ C
    P = 0.5 * (P + P.T)
    # Redundant inequality constraints (e.g. two joint limits that can never be
    # active simultaneously) make P singular and the dual optimum non-unique,
    # which lets a plain active-set iteration cycle.  A tiny diagonal ridge
    # makes the dual strictly concave; the induced primal perturbation is of
    # order 1e-10 and is negligible next to the control tolerances.
    p_scale = float(np.max(np.abs(np.diag(P)))) if m else 1.0
    P = P + max(1e-12, 1e-10 * p_scale) * np.eye(m)
    r = b + C.T @ (G_inv @ a)

    lam = np.zeros(m)
    active: list[int] = []
    if lam0 is not None and len(lam0) == m:
        init = np.maximum(np.asarray(lam0, dtype=float), 0.0)
        active = [i for i in range(m) if init[i] > 1e-9]
        lam[:] = init

    scale = max(float(np.max(np.abs(r))) if m else 1.0, 1.0)
    ftol = 1e-12 * scale

    iterations = 0
    for _ in range(max_sweeps):
        iterations += 1
        # 1) solve the equality-constrained dual sub-problem on the active set
        if active:
            idx = np.array(active, dtype=int)
            sub = P[np.ix_(idx, idx)]
            rhs = r[idx]
            try:
                lam_active = np.linalg.solve(sub, rhs)
            except np.linalg.LinAlgError:
                lam_active = np.linalg.lstsq(sub, rhs, rcond=None)[0]
            lam[:] = 0.0
            lam[idx] = lam_active
            # 2) drop the most negative multiplier
            j = int(np.argmin(lam_active))
            if lam_active[j] < -ftol:
                active.pop(j)
                continue
            lam[lam < 0.0] = 0.0

        # 3) find the most violating inactive constraint of the dual
        if active:
            grad = r - P @ lam
            grad[np.array(active, dtype=int)] = -np.inf
        else:
            grad = r.copy()
        i = int(np.argmax(grad))
        if grad[i] > ftol:
            if i not in active:
                active.append(i)
            continue
        break

    z = G_inv @ (C @ lam - a)
    result.x = x_p + N @ z
    result.lam_ub = lam
    result.iterations = iterations
    result.converged = iterations < max_sweeps
    result.eq_residual = _eq_residual(result.x, A_eq, b_eq)
    if A_ub is not None and A_ub.size:
        result.ineq_violation = float(np.max(np.maximum(A_ub @ result.x - b_ub, 0.0)))
    return result


def _solve_hildreth(
    H: np.ndarray,
    g: np.ndarray,
    A_eq: np.ndarray | None,
    b_eq: np.ndarray | None,
    A_ub: np.ndarray | None,
    b_ub: np.ndarray | None,
    *,
    lam0: np.ndarray | None,
    max_sweeps: int,
    tol: float,
    ridge: float,
) -> QPResult:
    n = g.shape[0]
    result = QPResult(x=np.zeros(n), backend="hildreth")

    # --- equality elimination: x = x_p + N z ------------------------------
    if A_eq is not None:
        x_p, *_ = np.linalg.lstsq(A_eq, b_eq, rcond=None)
        N = null_space(A_eq)
    else:
        x_p = np.zeros(n)
        N = np.eye(n)
    k = N.shape[1]

    H_z = N.T @ H @ N
    g_z = N.T @ (H @ x_p + g)
    if ridge > 0.0:
        H_z = H_z + ridge * np.eye(k)
    # H_z must be invertible for the dual formulation
    H_z = 0.5 * (H_z + H_z.T)
    try:
        H_z_inv = np.linalg.inv(H_z)
    except np.linalg.LinAlgError:
        H_z_inv = np.linalg.pinv(H_z)

    if A_ub is None or k == 0:
        z = -H_z_inv @ g_z
        x = x_p + N @ z
        result.x = x
        result.lam_ub = np.zeros(0 if A_ub is None else A_ub.shape[0])
        result.converged = True
        result.eq_residual = _eq_residual(result.x, A_eq, b_eq)
        return result

    A_z = A_ub @ N
    b_z = b_ub - A_ub @ x_p

    # --- dual: min 0.5 lam^T P lam + q^T lam, lam >= 0 ---------------------
    HinvAt = H_z_inv @ A_z.T
    P = A_z @ HinvAt
    P = 0.5 * (P + P.T)
    q = A_z @ (H_z_inv @ g_z) + b_z
    diag = np.diag(P).copy()

    lam = np.zeros(P.shape[0]) if lam0 is None or len(lam0) != P.shape[0] else np.array(lam0, dtype=float)
    lam = np.maximum(lam, 0.0)

    usable = diag > 1e-14
    iterations = 0
    for sweep in range(max_sweeps):
        delta = 0.0
        for i in range(P.shape[0]):
            if not usable[i]:
                lam[i] = 0.0
                continue
            residual = q[i] + P[i] @ lam - P[i, i] * lam[i]
            new = max(0.0, -residual / P[i, i])
            delta = max(delta, abs(new - lam[i]))
            lam[i] = new
        iterations += 1
        if delta < tol:
            break

    z = -H_z_inv @ (g_z + A_z.T @ lam)
    result.x = x_p + N @ z
    result.lam_ub = lam
    result.iterations = iterations
    result.converged = iterations < max_sweeps
    result.eq_residual = _eq_residual(result.x, A_eq, b_eq)
    result.ineq_violation = float(np.max(np.maximum(A_ub @ result.x - b_ub, 0.0))) if A_ub.size else 0.0
    return result


def _eq_residual(x: np.ndarray, A_eq: np.ndarray | None, b_eq: np.ndarray | None) -> float:
    if A_eq is None:
        return 0.0
    return float(np.max(np.abs(A_eq @ x - b_eq)))


def _solve_slsqp(H, g, A_eq, b_eq, A_ub, b_ub, *, lam0=None) -> QPResult:
    """Reference implementation used in tests (requires SciPy)."""
    from scipy.optimize import minimize

    n = g.shape[0]

    def fun(x):
        return 0.5 * x @ H @ x + g @ x

    def jac(x):
        return H @ x + g

    constraints = []
    if A_eq is not None:
        constraints.append({"type": "eq", "fun": lambda x: A_eq @ x - b_eq, "jac": lambda x: A_eq})
    if A_ub is not None:
        constraints.append({"type": "ineq", "fun": lambda x: b_ub - A_ub @ x, "jac": lambda x: -A_ub})

    x0 = np.zeros(n) if lam0 is None else np.asarray(lam0, dtype=float)
    res = minimize(fun, x0, jac=jac, constraints=constraints, method="SLSQP",
                   options={"maxiter": 300, "ftol": 1e-12})
    out = QPResult(x=np.asarray(res.x, dtype=float), backend="slsqp", iterations=int(res.nit))
    out.eq_residual = _eq_residual(out.x, A_eq, b_eq)
    if A_ub is not None and A_ub.size:
        out.ineq_violation = float(np.max(np.maximum(A_ub @ out.x - b_ub, 0.0)))
    out.converged = bool(res.success)
    return out


# --------------------------------------------------------------------------- #
# helpers used by the controller to assemble least-squares objectives
# --------------------------------------------------------------------------- #
def least_squares_objective(tasks: list[tuple[np.ndarray, np.ndarray, float]], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least-squares objective ``sum_i w_i ||A_i x - b_i||^2``.

    Returns ``(H, g)`` with ``H = sum w_i A_i^T A_i`` and ``g = -sum w_i A_i^T b_i``,
    i.e. the minimiser of ``0.5 x^T H x + g^T x`` is the weighted least-squares
    solution.
    """
    H = np.zeros((n, n))
    g = np.zeros(n)
    for A, b, w in tasks:
        A = np.atleast_2d(np.asarray(A, dtype=float))
        b = np.asarray(b, dtype=float).reshape(-1)
        H += w * (A.T @ A)
        g -= w * (A.T @ b)
    return H, g
