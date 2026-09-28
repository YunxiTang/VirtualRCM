"""MuJoCo simulation wrapper and the V-RCM control loop.

Two execution modes are supported:

``"kinematic"``
    the QP joint velocity is integrated directly (``qpos += qdot dt``), so the
    RCM error reflects *only* controller behaviour.  Use this for analysis and
    regression tests.
``"dynamic"``
    the joint velocity is converted into position set-points for the MuJoCo
    actuators and the model is stepped with ``mj_step``.  Actuator tracking error
    then shows up as RCM drift, which is the honest way to check the feedback
    term of equation (8) in ``method.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from vrcm.controller import VrcmConfig, VrcmController, VrcmResult, VrcmTarget
from vrcm.planner import ViewpointPlanner
from vrcm.rcm import rcm_error, rcm_length
from vrcm.robot import Robot
from vrcm.specs import RobotSpec


@dataclass
class SimConfig:
    mode: str = "kinematic"
    control_dt: float = 0.01
    #: physics substeps per control step (defaults to control_dt / model timestep)
    substeps: int | None = None
    #: jaw opening used in the simulation
    jaw: float | None = None
    #: stop integrating when a joint hits a hard limit (kinematic mode)
    respect_joint_limits: bool = True
    #: dynamic mode: feed the gravity/Coriolis torque forward through the PD
    #: set-points, so that the actuators behave like a gravity-compensated robot
    #: instead of sagging under their own weight.
    gravity_feedforward: bool = True
    #: dynamic mode: also feed the inertial torque ``M qddot`` forward, with
    #: ``qddot`` the change from the measured to the commanded joint velocity over
    #: one control step.  Removes most of the velocity lag of soft servos.
    acceleration_feedforward: bool = True
    #: dynamic mode: settle time (s) applied in :meth:`reset`.
    settle_time: float = 0.2


@dataclass
class SimLog:
    time: list[float] = field(default_factory=list)
    q: list[np.ndarray] = field(default_factory=list)
    qdot: list[np.ndarray] = field(default_factory=list)
    p_tip: list[np.ndarray] = field(default_factory=list)
    d: list[np.ndarray] = field(default_factory=list)
    d_des: list[np.ndarray] = field(default_factory=list)
    e_rcm: list[np.ndarray] = field(default_factory=list)
    length: list[float] = field(default_factory=list)
    depth_target: list[float] = field(default_factory=list)
    converged: list[bool] = field(default_factory=list)
    notes: list[tuple[str, ...]] = field(default_factory=list)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "t": np.asarray(self.time),
            "q": np.asarray(self.q),
            "qdot": np.asarray(self.qdot),
            "p_tip": np.asarray(self.p_tip),
            "d": np.asarray(self.d),
            "d_des": np.asarray(self.d_des),
            "e_rcm": np.asarray(self.e_rcm),
            "length": np.asarray(self.length),
            "depth_target": np.asarray(self.depth_target),
        }

    def summary(self) -> dict[str, float]:
        e = np.linalg.norm(np.asarray(self.e_rcm), axis=1) if self.e_rcm else np.zeros(0)
        d = np.asarray(self.d)
        dd = np.asarray(self.d_des)
        angle = np.degrees(np.arccos(np.clip(np.sum(d * dd, axis=1), -1, 1))) if len(d) else np.zeros(0)
        return {
            "steps": len(self.time),
            "rcm_error_max_mm": float(np.max(e) * 1000) if len(e) else 0.0,
            "rcm_error_rms_mm": float(np.sqrt(np.mean(e ** 2)) * 1000) if len(e) else 0.0,
            "direction_error_max_deg": float(np.max(angle)) if len(angle) else 0.0,
            "direction_error_rms_deg": float(np.sqrt(np.mean(angle ** 2))) if len(angle) else 0.0,
            "depth_min": float(np.min(self.length)) if self.length else 0.0,
            "depth_max": float(np.max(self.length)) if self.length else 0.0,
        }


class VrcmSim:
    """Simulation environment tying together the robot, controller and planner."""

    def __init__(
        self,
        spec: RobotSpec,
        controller_config: VrcmConfig | None = None,
        sim_config: SimConfig | None = None,
        robot: Robot | None = None,
    ) -> None:
        if isinstance(controller_config, SimConfig):
            raise TypeError(
                "VrcmSim(spec, controller_config, sim_config): pass a VrcmConfig as the "
                "second argument and a SimConfig as the third (or use keywords)"
            )
        if isinstance(sim_config, VrcmConfig):
            raise TypeError("the third argument of VrcmSim is a SimConfig, not a VrcmConfig")
        self.spec = spec
        self.robot = robot if robot is not None else Robot(spec)
        self.config = sim_config or SimConfig()
        cfg = controller_config or VrcmConfig()
        cfg.dt = self.config.control_dt
        self.controller = VrcmController(self.robot, cfg)
        self.model = self.robot.model
        self.data = self.robot.data
        self.trocar = np.asarray(spec.scene.trocar, dtype=float)
        if self.config.mode == "dynamic" and np.any(self.robot.arm_actuator_ids < 0):
            raise ValueError(f"{spec.name}: dynamic mode needs an actuator on every arm joint")
        self.n_substeps = self.config.substeps or max(
            int(round(self.config.control_dt / self.model.opt.timestep)), 1
        )

    # ------------------------------------------------------------------- state
    def reset(self, q: np.ndarray | None = None) -> None:
        self.robot.reset(q)
        self.controller.reset()
        if self.config.jaw is not None:
            self.robot.set_jaw(self.config.jaw)
            self._sync_jaw_actuator()
        self.robot.forward()
        if self.config.mode == "dynamic" and self.config.settle_time > 0.0:
            self._hold(self.config.settle_time)

    def _feedforward_ctrl(self, q_target: np.ndarray, qdot_des: np.ndarray | None = None) -> np.ndarray:
        """PD set-points that also cancel the gravity/Coriolis torque and inject
        the desired joint velocity.

        With a plain position servo the steady-state joint velocity is limited to
        ``kp * dt / kv * qdot_cmd`` (the PD only ever sees a lead of one control
        step), so the velocity feed-forward term ``kv * qdot_cmd / kp`` is what
        makes the actuator actually follow the QP solution.
        """
        acts = self.robot.arm_actuator_ids
        ctrl = np.array(q_target, dtype=float)
        if not self.config.gravity_feedforward:
            return ctrl
        tau = np.asarray(self.data.qfrc_bias)[self.robot._arm_dof_adr].copy()
        gravcomp = np.asarray(self.data.qfrc_gravcomp)[self.robot._arm_dof_adr]
        tau = tau - gravcomp  # Flexiv links already carry gravcomp="1"
        kp = self.model.actuator_gainprm[self.robot.arm_actuator_ids, 0]
        kv = -self.model.actuator_biasprm[self.robot.arm_actuator_ids, 2]
        if qdot_des is not None:
            qdot_des = np.asarray(qdot_des, dtype=float)
            tau = tau + kv * qdot_des
            if self.config.acceleration_feedforward:
                dofs = self.robot._arm_dof_adr
                mass = np.zeros((self.model.nv, self.model.nv))
                mujoco.mj_fullM(self.model, self.data, mass)
                qddot = (qdot_des - self.data.qvel[dofs]) / self.config.control_dt
                tau = tau + mass[np.ix_(dofs, dofs)] @ qddot
        ctrl = ctrl + tau / kp
        lower = self.model.actuator_ctrlrange[acts, 0]
        upper = self.model.actuator_ctrlrange[acts, 1]
        limited = self.model.actuator_ctrllimited[acts].astype(bool)
        return np.where(limited, np.clip(ctrl, lower, upper), ctrl)

    def _hold(self, duration: float) -> None:
        """Step the physics while holding the current joint configuration."""
        ctrl = self._feedforward_ctrl(self.robot.arm_qpos())
        self.data.ctrl[self.robot.arm_actuator_ids] = ctrl
        for _ in range(max(int(round(duration / self.model.opt.timestep)), 1)):
            mujoco.mj_step(self.model, self.data)

    def state(self) -> dict[str, np.ndarray | float]:
        kin = self.robot.kinematics()
        return {
            "q": kin.q,
            "p_tip": kin.p_tip,
            "d": kin.d,
            "length": rcm_length(kin.p_tip, kin.d, self.trocar),
            "rcm_error": rcm_error(kin.p_tip, kin.d, self.trocar),
        }

    @property
    def rcm_error_norm(self) -> float:
        s = self.state()
        return float(np.linalg.norm(s["rcm_error"]))

    @property
    def insertion_depth(self) -> float:
        return float(self.state()["length"])

    # -------------------------------------------------------------------- step
    def _sync_jaw_actuator(self) -> None:
        name_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "jaw_a")
        if name_id >= 0:
            self.data.ctrl[name_id] = self.spec.instrument.jaw_open

    def step(self, target: VrcmTarget) -> VrcmResult:
        result = self.controller.step(self.robot.arm_qpos(), target)
        if self.config.mode == "dynamic":
            q_des = self.robot.arm_qpos() + result.qdot * self.config.control_dt
            q_des = np.clip(q_des, self.robot.joint_lower, self.robot.joint_upper)
            self.data.ctrl[self.robot.arm_actuator_ids] = self._feedforward_ctrl(q_des, result.qdot)
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.model, self.data)
        else:
            for _ in range(self.n_substeps):
                dt = self.model.opt.timestep
                q = self.robot.arm_qpos() + result.qdot * dt
                if self.config.respect_joint_limits:
                    q = np.clip(q, self.robot.joint_lower, self.robot.joint_upper)
                self.data.qvel[self.robot._arm_dof_adr] = result.qdot
                self.robot.set_arm_qpos(q)
                mujoco.mj_forward(self.model, self.data)
            self.data.qvel[:] = 0.0
            self.data.qvel[self.robot._arm_dof_adr] = result.qdot
        return result

    # --------------------------------------------------------------------- run
    def settle(
        self,
        planner: ViewpointPlanner,
        *,
        max_time: float = 4.0,
        angle_tol: float = np.deg2rad(0.2),
        depth_tol: float = 1e-4,
    ) -> float:
        """Advance the loop until planner and controller have both converged.

        Returns the elapsed simulated time.  This is the "stop" half of a
        stop-and-go cycle: the next observation is only taken once the instrument
        is stationary, so the viewpoint that is scored is the one that was asked
        for.
        """
        dt = self.config.control_dt
        elapsed = 0.0
        while elapsed < max_time:
            ref = planner.step(dt)
            result = self.step(
                VrcmTarget(
                    d_des=ref.d_des,
                    insertion_depth=ref.insertion_depth,
                    insertion_velocity=ref.insertion_velocity,
                    roll_velocity=ref.roll_velocity,
                )
            )
            elapsed += dt
            state = self.robot.kinematics()
            angle_err = float(np.arccos(np.clip(np.dot(state.d, ref.d_des), -1.0, 1.0)))
            depth_err = abs(float(rcm_length(state.p_tip, state.d, self.trocar)) - ref.insertion_depth)
            if ref.settled and angle_err < angle_tol and depth_err < depth_tol:
                break
        return elapsed

    def reset_log_state(self) -> None:
        """No-op hook kept for symmetry with :meth:`reset`."""

    # --------------------------------------------------------------------- run
    def run(
        self,
        planner: ViewpointPlanner,
        duration: float,
        *,
        log: SimLog | None = None,
        callback=None,
        stop_when_settled: bool = False,
    ) -> SimLog:
        """Run the closed loop for ``duration`` seconds."""
        log = SimLog() if log is None else log
        dt = self.config.control_dt
        n_steps = int(round(duration / dt))
        t = len(log.time) * dt
        for _ in range(n_steps):
            ref = planner.step(dt)
            target = VrcmTarget(
                d_des=ref.d_des,
                insertion_depth=ref.insertion_depth,
                insertion_velocity=ref.insertion_velocity,
                roll_velocity=ref.roll_velocity,
            )
            result = self.step(target)
            log.time.append(t)
            log.q.append(self.robot.arm_qpos())
            log.qdot.append(result.qdot)
            log.p_tip.append(result.kin.p_tip)
            log.d.append(result.kin.d)
            log.d_des.append(ref.d_des)
            log.e_rcm.append(result.rcm_error)
            log.length.append(result.length)
            log.depth_target.append(ref.insertion_depth)
            log.converged.append(result.converged)
            log.notes.append(result.notes)
            if callback is not None:
                callback(t, result)
            t += dt
            if stop_when_settled and ref.settled:
                break
        return log
