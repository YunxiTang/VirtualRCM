"""Small helpers shared by the test modules."""

from __future__ import annotations

import numpy as np

from vrcm.robot import Robot


def random_configurations(robot: Robot, n: int, seed: int = 0, margin: float = 0.15) -> np.ndarray:
    """``n`` random joint configurations that stay ``margin`` away from the limits."""
    rng = np.random.default_rng(seed)
    lo = robot.joint_lower + margin
    hi = robot.joint_upper - margin
    return rng.uniform(lo, hi, size=(n, robot.n_arm_joints))
