"""Virtual Remote Center of Motion (V-RCM) control for surgical robots in MuJoCo.

The package implements the method described in ``method.md``:

* :mod:`vrcm.robot`      -- MuJoCo model of a serial arm carrying an instrument
* :mod:`vrcm.rcm`        -- remote-centre-of-motion kinematics (error, Jacobian)
* :mod:`vrcm.qp`         -- dependency-free constrained QP solver
* :mod:`vrcm.controller` -- the QP based V-RCM task-space velocity controller
* :mod:`vrcm.planner`    -- surgical viewpoint actions (pitch / yaw / insertion)
* :mod:`vrcm.sim`        -- MuJoCo simulation environment and control loop
"""

from vrcm.controller import VrcmConfig, VrcmController, VrcmTarget
from vrcm.planner import ViewpointAction, ViewpointPlanner
from vrcm.rcm import rcm_error, rcm_jacobian, rcm_length, skew
from vrcm.robot import Robot
from vrcm.scene import build_scene_xml
from vrcm.sim import SimConfig, VrcmSim
from vrcm.specs import InstrumentSpec, RobotSpec, SceneSpec, load_robot_spec

__all__ = [
    "InstrumentSpec",
    "RobotSpec",
    "SceneSpec",
    "load_robot_spec",
    "build_scene_xml",
    "Robot",
    "VrcmController",
    "VrcmConfig",
    "VrcmTarget",
    "ViewpointAction",
    "ViewpointPlanner",
    "VrcmSim",
    "SimConfig",
    "rcm_error",
    "rcm_jacobian",
    "rcm_length",
    "skew",
]
