"""The QP solver must agree with a reference solver on every backend."""

from __future__ import annotations

import numpy as np
import pytest

from vrcm import qp

BACKENDS = [b for b in qp.available_backends() if b != "slsqp"]


def random_problem(rng, n_max: int = 8):
    n = int(rng.integers(2, n_max + 1))
    n_eq = int(rng.integers(0, min(3, n) + 1))
    n_ub = int(rng.integers(1, 12))
    A = rng.normal(size=(n + 2, n))
    H = A.T @ A + 1e-3 * np.eye(n)
    g = rng.normal(size=n)
    x0 = rng.normal(size=n)  # guaranteed feasible point
    A_eq = rng.normal(size=(n_eq, n))
    b_eq = A_eq @ x0
    A_ub = rng.normal(size=(n_ub, n))
    b_ub = A_ub @ x0 + rng.uniform(0.0, 1.5, size=n_ub)
    return H, g, A_eq, b_eq, A_ub, b_ub


def objective(H, g, x):
    return 0.5 * x @ H @ x + g @ x


@pytest.mark.parametrize("backend", BACKENDS)
def test_matches_reference_solver(backend):
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(60):
        H, g, A_eq, b_eq, A_ub, b_ub = random_problem(rng)
        got = qp.solve_qp(H, g, A_eq, b_eq, A_ub, b_ub, backend=backend)
        ref = qp.solve_qp(H, g, A_eq, b_eq, A_ub, b_ub, backend="slsqp")
        assert got.eq_residual < 1e-7
        assert got.ineq_violation < 1e-7
        worst = max(worst, objective(H, g, got.x) - objective(H, g, ref.x))
    assert worst < 1e-6


def test_unconstrained_solution_is_the_analytic_minimiser():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 4))
    H = A.T @ A + np.eye(4)
    g = rng.normal(size=4)
    for backend in BACKENDS:
        res = qp.solve_qp(H, g, backend=backend)
        np.testing.assert_allclose(res.x, -np.linalg.solve(H, g), atol=1e-8)


def test_equalities_are_satisfied_exactly():
    rng = np.random.default_rng(2)
    H = np.eye(5) * 2.0
    g = rng.normal(size=5)
    A_eq = np.array([[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0]])
    b_eq = np.array([0.3, -0.2])
    for backend in BACKENDS:
        res = qp.solve_qp(H, g, A_eq, b_eq, backend=backend)
        np.testing.assert_allclose(A_eq @ res.x, b_eq, atol=1e-9)


def test_rank_deficient_equalities_are_handled():
    """The RCM block has 3 rows but only rank 2; solvers must cope."""
    d = np.array([0.0, 0.0, -1.0])
    P = np.eye(3) - np.outer(d, d)
    A_eq = np.vstack([P, d]) @ np.stack([np.eye(3)[0], np.eye(3)[0], np.eye(3)[0]], axis=1)
    H = np.eye(3)
    g = np.array([1.0, -2.0, 0.5])
    b_eq = np.zeros(4)
    for backend in BACKENDS:
        res = qp.solve_qp(H, g, A_eq, b_eq, backend=backend)
        assert res.eq_residual < 1e-8


def test_infeasible_problem_is_reported():
    H = np.eye(2)
    g = np.zeros(2)
    A_ub = np.array([[1.0, 0.0], [-1.0, 0.0]])
    b_ub = np.array([-1.0, -1.0])  # x >= 1 and x <= -1
    res = qp.solve_qp(H, g, A_ub=A_ub, b_ub=b_ub, backend="quadprog")
    assert not res.converged


def test_least_squares_objective_helper():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(4, 3))
    b = rng.normal(size=4)
    H, g = qp.least_squares_objective([(A, b, 2.0)], 3)
    np.testing.assert_allclose(H, 2.0 * A.T @ A)
    np.testing.assert_allclose(g, -2.0 * A.T @ b)
