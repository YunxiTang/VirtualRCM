"""Robot / instrument / scene specifications.

Everything that differs between the two supported arms (UR5e and Flexiv Rizon 4)
is described by a :class:`RobotSpec`, loaded from a JSON file in ``configs/``.
No robot specific code path exists anywhere else in the package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


@dataclass(frozen=True)
class InstrumentSpec:
    """An instrument mounted rigidly on the robot tool flange.

    The instrument frame has its origin on the flange mating face and its ``+z``
    axis along the shaft, so the tip site sits at ``z = nominal_length``.
    """

    xml: Path
    mount_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    tip_site: str = "instrument_tip"
    base_site: str = "instrument_base"
    mid_site: str = "instrument_shaft_mid"
    nominal_length: float = 0.30
    shaft_radius: float = 0.0035
    min_insertion: float = 0.03
    max_insertion: float = 0.22
    insertion_rate: float = 0.04  # m/s, default commanded insertion speed
    jaw_open: float = 0.35  # jaw joint angle used as the "open" pose
    jaw_closed: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InstrumentSpec":
        data = dict(data)
        data["xml"] = _resolve(data["xml"])
        for key in ("mount_pos", "mount_quat"):
            if key in data:
                data[key] = tuple(float(v) for v in data[key])
        return cls(**data)


@dataclass(frozen=True)
class SceneSpec:
    """Static environment: trocar (RCM) point, abdominal phantom, floor."""

    #: RCM / trocar point in the world frame.  Fixed for the whole session.
    trocar: tuple[float, float, float] = (0.35, 0.0, 0.40)
    #: Instrument tip distance below the trocar in the home pose of the robot.
    home_insertion: float = 0.15
    #: Abdominal wall slab: half sizes in x/y and thickness in z.
    wall_half_xy: tuple[float, float] = (0.20, 0.16)
    wall_thickness: float = 0.03
    #: Insufflated cavity below the wall (half sizes, floor of the cavity at trocar.z - depth).
    cavity_half_xy: tuple[float, float] = (0.17, 0.13)
    cavity_depth: float = 0.20
    #: Trocar port geometry (visual marker only).
    trocar_radius: float = 0.006
    trocar_length: float = 0.06
    #: Let the phantom collide with the robot.  Off by default: the trocar hole is
    #: not modelled, so contacts would fight the virtual RCM constraint.
    phantom_collision: bool = False
    floor: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneSpec":
        return cls(**data)


@dataclass(frozen=True)
class RobotSpec:
    """Everything needed to build and control one arm + instrument combination."""

    name: str
    robot_xml: Path
    flange_body: str
    instrument: InstrumentSpec
    scene: SceneSpec
    home_qpos: tuple[float, ...]
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    joint_velocity_limits: tuple[float, ...] | None = None
    #: Multiplies the damping ``kv`` of the arm's position actuators (dynamic mode).
    #: Applied to the compiled model, so the upstream MJCF stays untouched.
    actuator_kv_scale: float = 1.0
    sim_dt: float = 0.001
    control_dt: float = 0.01
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotSpec":
        data = dict(data)
        data["robot_xml"] = _resolve(data["robot_xml"])
        data["instrument"] = InstrumentSpec.from_dict(data["instrument"])
        data["scene"] = SceneSpec.from_dict(data.get("scene", {}))
        data["home_qpos"] = tuple(float(v) for v in data["home_qpos"])
        if data.get("joint_velocity_limits") is not None:
            data["joint_velocity_limits"] = tuple(float(v) for v in data["joint_velocity_limits"])
        for key in ("base_pos", "base_quat"):
            if key in data:
                data[key] = tuple(float(v) for v in data[key])
        return cls(**data)

    @property
    def trocar(self) -> tuple[float, float, float]:
        return self.scene.trocar

    def with_home(self, qpos: Sequence[float]) -> "RobotSpec":
        return replace(self, home_qpos=tuple(float(v) for v in qpos))


def available_robots(config_dir: Path | str = CONFIG_DIR) -> list[str]:
    """Names of all robot configs found in ``config_dir``."""
    return sorted(p.stem for p in Path(config_dir).glob("*.json"))


def load_robot_spec(name_or_path: str | Path, config_dir: Path | str = CONFIG_DIR) -> RobotSpec:
    """Load a :class:`RobotSpec` by config name (``"ur5e"``) or by file path."""
    path = Path(name_or_path)
    if path.suffix != ".json":
        path = Path(config_dir) / f"{path.name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no robot config at {path}")
    with open(path) as fh:
        return RobotSpec.from_dict(json.load(fh))
