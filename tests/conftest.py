from __future__ import annotations

import numpy as np
import pytest

from vrcm.robot import Robot
from vrcm.specs import available_robots, load_robot_spec


@pytest.fixture(scope="session")
def robot_names() -> list[str]:
    return available_robots()


@pytest.fixture(scope="session", params=available_robots())
def robot(request) -> Robot:
    """A compiled model of every configured arm (UR5e, Flexiv Rizon 4)."""
    return Robot(load_robot_spec(request.param))


@pytest.fixture()
def fresh_robot(robot: Robot) -> Robot:
    robot.reset()
    return robot
