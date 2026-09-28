"""Thin, robot-agnostic wrapper around a compiled MuJoCo model.

Only the first ``spec.n_arm_joints`` joints of the model are treated as the arm;
everything appended by the instrument (the two jaw hinges) is kept out of the
control problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np

from vrcm.scene import build_scene_xml
from vrcm.specs import RobotSpec

#: MuJoCo >= 3.14 exposes mjtJoint as a plain Python Enum instead of int
#: constants, so enum members must be compared as ints to stay version agnostic.
_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)


@dataclass
class Kinematics:
    """Snapshot of the instrument frame produced by one forward-kinematics call."""

    q: np.ndarray  # arm joint positions, shape (n,)
    p_tip: np.ndarray  # world position of the tip site, shape (3,)
    R_tip: np.ndarray  # world rotation of the tip frame, shape (3, 3)
    d: np.ndarray  # shaft direction (tip frame z axis in world), shape (3,)
    Jv: np.ndarray  # translational Jacobian at the tip, shape (3, n)
    Jw: np.ndarray  # rotational Jacobian of the tip frame, shape (3, n)
    p_base: np.ndarray  # world position of the instrument base, shape (3,)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p_tip - self.p_base))


class Robot:
    """MuJoCo model of one arm with the surgical instrument rigidly attached."""

    def __init__(self, spec: RobotSpec, xml: str | None = None) -> None:
        self.spec = spec
        self.xml = xml if xml is not None else build_scene_xml(spec)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)

        self.arm_joint_names = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(self.model.njnt)
            if int(self.model.jnt_type[j]) in (_HINGE, _SLIDE)
        )
        n_arm = len(spec.home_qpos)
        if n_arm > len(self.arm_joint_names):
            raise ValueError(
                f"{spec.name}: home_qpos has {n_arm} entries but the model only exposes "
                f"{len(self.arm_joint_names)} joints"
            )
        self.n_arm_joints = n_arm
        self.arm_joint_names = self.arm_joint_names[:n_arm]

        self._arm_qpos_adr = np.array(
            [self.model.jnt_qposadr[j] for j in range(n_arm)], dtype=int
        )
        self._arm_dof_adr = np.array(
            [self.model.jnt_dofadr[j] for j in range(n_arm)], dtype=int
        )

        self.tip_site_id = self._site_id(spec.instrument.tip_site)
        self.base_site_id = self._site_id(spec.instrument.base_site)
        self.mid_site_id = self._site_id(spec.instrument.mid_site)

        #: actuator index driving each arm joint (-1 if the joint is unactuated)
        self.arm_actuator_ids = np.full(n_arm, -1, dtype=int)
        for act in range(self.model.nu):
            jid = int(self.model.actuator_trnid[act, 0])
            if jid < n_arm:
                self.arm_actuator_ids[jid] = act

        self.joint_lower = np.array([self.model.jnt_range[j][0] for j in range(n_arm)])
        self.joint_upper = np.array([self.model.jnt_range[j][1] for j in range(n_arm)])

        limits = spec.joint_velocity_limits
        defaults = np.full(n_arm, 1.5)
        self.joint_velocity_limits = np.array(limits, dtype=float) if limits else defaults
        if self.joint_velocity_limits.shape != (n_arm,):
            raise ValueError(
                f"{spec.name}: joint_velocity_limits has shape {self.joint_velocity_limits.shape}, expected ({n_arm},)"
            )

        self.reset()

    # ------------------------------------------------------------------ setup
    def _site_id(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"site {name!r} not found in the {self.spec.name} model")
        return sid

    @property
    def nq_arm(self) -> int:
        return self.n_arm_joints

    @property
    def nv(self) -> int:
        return self.model.nv

    def reset(self, q: Sequence[float] | None = None) -> None:
        """Reset the simulation state; defaults to the spec home configuration."""
        mujoco.mj_resetData(self.model, self.data)
        q = np.asarray(self.spec.home_qpos if q is None else q, dtype=float)
        self.set_arm_qpos(q)
        jaw_open = self.spec.instrument.jaw_closed
        self.set_jaw(jaw_open)
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------- state
    def set_arm_qpos(self, q: Sequence[float]) -> None:
        self.data.qpos[self._arm_qpos_adr] = np.asarray(q, dtype=float)

    def arm_qpos(self) -> np.ndarray:
        return np.array(self.data.qpos[self._arm_qpos_adr])

    def arm_qvel(self) -> np.ndarray:
        return np.array(self.data.qvel[self._arm_dof_adr])

    def set_jaw(self, angle: float) -> None:
        """Set both jaw angles (they stay coupled by the model equality constraint)."""
        for name in ("jaw_a_joint", "jaw_b_joint"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = angle if name == "jaw_a_joint" else -angle

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    # -------------------------------------------------------------- kinematics
    def kinematics(self, q: Sequence[float] | None = None) -> Kinematics:
        """Instrument frame, shaft direction and tip Jacobians at configuration ``q``."""
        if q is not None:
            self.set_arm_qpos(q)
        self.forward()

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.tip_site_id)

        R_tip = np.array(self.data.site_xmat[self.tip_site_id]).reshape(3, 3)
        return Kinematics(
            q=self.arm_qpos(),
            p_tip=np.array(self.data.site_xpos[self.tip_site_id]),
            R_tip=R_tip,
            d=R_tip[:, 2],
            Jv=jacp[:, self._arm_dof_adr],
            Jw=jacr[:, self._arm_dof_adr],
            p_base=np.array(self.data.site_xpos[self.base_site_id]),
        )

    def site_position(self, name: str) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return np.array(self.data.site_xpos[sid])

    # ------------------------------------------------------------- joint space
    def clamp_velocity(self, qdot: Sequence[float], scale: float = 1.0) -> np.ndarray:
        lim = scale * self.joint_velocity_limits
        return np.clip(np.asarray(qdot, dtype=float), -lim, lim)

    def limit_margin(self, q: Sequence[float]) -> np.ndarray:
        """Distance (rad) to the nearest joint limit, positive inside the range."""
        q = np.asarray(q, dtype=float)
        return np.minimum(q - self.joint_lower, self.joint_upper - q)

    def in_limits(self, q: Sequence[float], tol: float = 1e-6) -> bool:
        q = np.asarray(q, dtype=float)
        return bool(np.all(q >= self.joint_lower - tol) and np.all(q <= self.joint_upper + tol))

    # ------------------------------------------------------------ geom queries
    def geom_id(self, name: str) -> int:
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise ValueError(f"geom {name!r} not found in the {self.spec.name} model")
        return gid

    def geom_distance(self, geom1: int, geom2: int, max_distance: float = 1.0) -> tuple[float, np.ndarray]:
        """Signed distance between two geoms and the closest-point segment (world frame)."""
        fromto = np.zeros(6)
        dist = mujoco.mj_geomDistance(self.model, self.data, geom1, geom2, max_distance, fromto)
        return float(dist), fromto
