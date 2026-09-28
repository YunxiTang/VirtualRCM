"""Assemble the MuJoCo scene: robot + instrument + surgical phantom.

The robot and instrument are provided as separate XML files.  This module merges
them into a single model without touching the upstream asset files:

* the instrument fragment (``assets/instruments/<name>/instrument.xml``) is copied
  into the tool-flange body of the robot, so the instrument is rigidly attached
  with an exact kinematic transform and never drifts;
* mesh paths are rewritten to absolute paths, so the generated model can be
  compiled from a string and does not depend on the working directory;
* the surgical environment (RCM/trocar marker, abdominal wall, cavity, floor) is
  added on top.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from vrcm.specs import InstrumentSpec, RobotSpec, SceneSpec

# Site group of the RCM marker; kept apart from the instrument sites (group 4) so a
# viewer can show it alone via ``opt.sitegroup[RCM_SITE_GROUP]``.
RCM_SITE_GROUP = 3

_TOP_LEVEL_ORDER = ("compiler", "option", "size", "visual", "statistic", "default")


def _parse(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def _abs_meshdir(xml_path: Path) -> Path:
    root = _parse(xml_path)
    compiler = root.find("compiler")
    meshdir = "."
    if compiler is not None and compiler.get("meshdir"):
        meshdir = compiler.get("meshdir")
    return (xml_path.parent / meshdir).resolve()


def _rewrite_asset_paths(asset: ET.Element, meshdir: Path) -> None:
    for mesh in asset.findall("mesh"):
        file = mesh.get("file")
        if file and not Path(file).is_absolute():
            mesh.set("file", str((meshdir / file).resolve()))


def _instrument_fragment(
    instrument: InstrumentSpec,
) -> tuple[ET.Element, list[ET.Element], dict[str, list[ET.Element]]]:
    """Return ``(body, assets, extras)`` of the instrument model.

    ``extras`` maps a top-level section name (``actuator``, ``equality``, ...) to
    the list of child elements that have to be merged into the same section of the
    robot model.
    """
    root = _parse(instrument.xml)
    meshdir = _abs_meshdir(instrument.xml)

    body = root.find("worldbody/body")
    if body is None:
        raise ValueError(f"{instrument.xml} does not contain a <worldbody><body>")
    body = copy.deepcopy(body)
    body.set("pos", " ".join(f"{v:.12g}" for v in instrument.mount_pos))
    body.set("quat", " ".join(f"{v:.12g}" for v in instrument.mount_quat))

    asset = root.find("asset")
    assets = copy.deepcopy(list(asset)) if asset is not None else []
    for element in assets:
        if element.tag == "mesh":
            file = element.get("file")
            if file and not Path(file).is_absolute():
                element.set("file", str((meshdir / file).resolve()))

    extras: dict[str, list[ET.Element]] = {}
    for tag in ("actuator", "equality", "tendon", "sensor"):
        section = root.find(tag)
        if section is not None:
            extras[tag] = copy.deepcopy(list(section))
    return body, assets, extras


def _scene_assets(scene: SceneSpec) -> list[ET.Element]:
    nodes: list[ET.Element] = [
        ET.fromstring(
            '<texture type="skybox" builtin="gradient" rgb1="0.32 0.52 0.70" '
            'rgb2="0.05 0.05 0.08" width="512" height="3072"/>'
        ),
        ET.fromstring(
            '<texture type="2d" name="floor_tex" builtin="checker" mark="edge" '
            'rgb1="0.22 0.26 0.30" rgb2="0.14 0.17 0.20" markrgb="0.55 0.58 0.60" '
            'width="300" height="300"/>'
        ),
        ET.fromstring('<material name="floor_mat" texture="floor_tex" texuniform="true" texrepeat="6 6" reflectance="0.1"/>'),
        ET.fromstring('<material name="wall_mat" rgba="0.92 0.66 0.58 0.45" specular="0.2"/>'),
        ET.fromstring('<material name="cavity_mat" rgba="0.85 0.30 0.30 0.10"/>'),
        ET.fromstring('<material name="trocar_mat" rgba="0.15 0.55 0.85 1"/>'),
    ]
    return nodes


def _phantom_body(scene: SceneSpec) -> ET.Element:
    """Abdominal wall slab, insufflated cavity and the trocar (RCM) port."""
    tx, ty, tz = scene.trocar
    contype = "1" if scene.phantom_collision else "0"
    conaffinity = "1" if scene.phantom_collision else "0"

    wall = ET.Element("body", {"name": "phantom"})
    ET.SubElement(
        wall,
        "geom",
        {
            "name": "abdominal_wall",
            "type": "box",
            "size": f"{scene.wall_half_xy[0]:.6g} {scene.wall_half_xy[1]:.6g} {scene.wall_thickness / 2:.6g}",
            "pos": f"{tx:.6g} {ty:.6g} {tz - scene.wall_thickness / 2:.6g}",
            "material": "wall_mat",
            "contype": contype,
            "conaffinity": conaffinity,
            "group": "1",
        },
    )
    cavity_z = tz - scene.wall_thickness - scene.cavity_depth / 2
    ET.SubElement(
        wall,
        "geom",
        {
            "name": "cavity",
            "type": "box",
            "size": f"{scene.cavity_half_xy[0]:.6g} {scene.cavity_half_xy[1]:.6g} {scene.cavity_depth / 2:.6g}",
            "pos": f"{tx:.6g} {ty:.6g} {cavity_z:.6g}",
            "material": "cavity_mat",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )
    # trocar port: a short cylinder sitting on the abdominal wall
    ET.SubElement(
        wall,
        "geom",
        {
            "name": "trocar_port",
            "type": "cylinder",
            "size": f"{scene.trocar_radius:.6g} {scene.trocar_length / 2:.6g}",
            "pos": f"{tx:.6g} {ty:.6g} {tz + scene.trocar_length / 2:.6g}",
            "material": "trocar_mat",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )
    ET.SubElement(
        wall,
        "site",
        {"name": "rcm", "pos": f"{tx:.6g} {ty:.6g} {tz:.6g}", "size": "0.006", "rgba": "1 0.2 0.2 0.55", "group": str(RCM_SITE_GROUP)},
    )
    return wall


def _worldbody(scene: SceneSpec, robot_bodies: list[ET.Element], base_pos, base_quat) -> ET.Element:
    world = ET.Element("worldbody")
    if scene.floor:
        ET.SubElement(world, "geom", {"name": "floor", "type": "plane", "size": "0 0 0.05", "material": "floor_mat"})
    world.append(_phantom_body(scene))
    ET.SubElement(world, "light", {"pos": "0.2 -0.6 1.6", "dir": "0 0 -1", "directional": "true", "castshadow": "true"})
    ET.SubElement(world, "light", {"pos": "0.9 0.8 1.2", "dir": "-0.4 -0.4 -1", "directional": "true", "diffuse": "0.4 0.4 0.4"})

    tx, ty, tz = scene.trocar
    ET.SubElement(
        world,
        "camera",
        {
            "name": "overview",
            "pos": f"{tx - 0.85:.6g} {ty - 0.75:.6g} {tz + 0.75:.6g}",
            "xyaxes": "0.66 -0.75 0 0.53 0.46 0.71",
            "fovy": "48",
        },
    )
    ET.SubElement(
        world,
        "camera",
        {
            "name": "instrument_view",
            "pos": f"{tx - 0.30:.6g} {ty - 0.28:.6g} {tz + 0.42:.6g}",
            "xyaxes": "0.68 -0.73 0 0.47 0.44 0.77",
            "fovy": "45",
        },
    )

    if any(abs(v) > 0 for v in base_pos) or tuple(base_quat) != (1.0, 0.0, 0.0, 0.0):
        mount = ET.SubElement(
            world,
            "body",
            {
                "name": "robot_mount",
                "pos": " ".join(f"{v:.12g}" for v in base_pos),
                "quat": " ".join(f"{v:.12g}" for v in base_quat),
            },
        )
        for body in robot_bodies:
            mount.append(copy.deepcopy(body))
    else:
        for body in robot_bodies:
            world.append(copy.deepcopy(body))
    return world


def build_scene_xml(spec: RobotSpec, *, attach_instrument: bool = True) -> str:
    """Build the full MuJoCo scene XML for ``spec`` as a string."""
    robot = _parse(spec.robot_xml)
    robot_meshdir = _abs_meshdir(spec.robot_xml)

    instrument_body, instrument_assets, instrument_extras = _instrument_fragment(spec.instrument)

    # ---- find the flange body and graft the instrument into it -------------
    flange = None
    for body in robot.iter("body"):
        if body.get("name") == spec.flange_body:
            flange = body
            break
    if flange is None:
        raise ValueError(f"flange body {spec.flange_body!r} not found in {spec.robot_xml}")
    if attach_instrument:
        flange.append(instrument_body)
    else:
        # debug mode: instrument left floating at the world origin
        flange.append(copy.deepcopy(instrument_body))

    root = ET.Element("mujoco", {"model": f"{spec.name}_vrcm"})

    # ---- compiler / option / visual / default (order matters for MuJoCo) ----
    for tag in _TOP_LEVEL_ORDER:
        section = robot.find(tag)
        if section is None:
            continue
        section = copy.deepcopy(section)
        if tag == "compiler":
            section.set("meshdir", str(robot_meshdir))
            section.set("angle", "radian")
            section.set("autolimits", "true")
        if tag == "option":
            section.set("timestep", f"{spec.sim_dt:.12g}")
            section.set("gravity", "0 0 -9.81")
            section.set("integrator", section.get("integrator", "implicitfast"))
            section.set("cone", "elliptic")
        root.append(section)
    if robot.find("option") is None:
        root.append(
            ET.fromstring(f'<option timestep="{spec.sim_dt:.12g}" integrator="implicitfast" cone="elliptic"/>')
        )
    root.append(
        ET.fromstring(
            '<visual>'
            '<headlight diffuse="0.6 0.6 0.6" ambient="0.28 0.28 0.28" specular="0.1 0.1 0.1"/>'
            '<rgba haze="0.15 0.25 0.35 1"/>'
            '<global azimuth="128" elevation="-16" offwidth="1280" offheight="960"/>'
            '<quality shadowsize="4096"/>'
            '</visual>'
        )
    )

    # ---- asset ------------------------------------------------------------
    asset = ET.Element("asset")
    robot_asset = robot.find("asset")
    if robot_asset is not None:
        for element in copy.deepcopy(list(robot_asset)):
            if element.tag == "mesh":
                file = element.get("file")
                if file and not Path(file).is_absolute():
                    element.set("file", str((robot_meshdir / file).resolve()))
            asset.append(element)
    for element in _scene_assets(spec.scene):
        asset.append(element)
    for element in instrument_assets:
        asset.append(element)
    root.append(asset)

    # ---- worldbody --------------------------------------------------------
    robot_world = robot.find("worldbody")
    robot_bodies = [copy.deepcopy(child) for child in robot_world if child.tag == "body"] if robot_world is not None else []
    extras = [copy.deepcopy(child) for child in robot_world if child.tag != "body"] if robot_world is not None else []
    world = _worldbody(spec.scene, robot_bodies, spec.base_pos, spec.base_quat)
    # lights declared by the robot model are kept as well
    for element in extras:
        if element.tag == "light":
            continue
        world.append(element)
    if not attach_instrument:
        world.append(instrument_body)
    root.append(world)

    # ---- actuator / equality ----------------------------------------------
    actuator = ET.Element("actuator")
    robot_actuator = robot.find("actuator")
    if robot_actuator is not None:
        for element in copy.deepcopy(list(robot_actuator)):
            actuator.append(element)
    equality = ET.Element("equality")
    for tag, children in instrument_extras.items():
        target = actuator if tag == "actuator" else equality
        if target is not actuator and tag != "equality":
            continue  # tendons / sensors of the instrument are not used
        for element in children:
            target.append(element)
    if len(actuator):
        root.append(actuator)
    if len(equality):
        root.append(equality)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def write_scene_xml(spec: RobotSpec, path: str | Path, **kwargs) -> Path:
    """Write the generated scene XML to ``path`` (useful for debugging / viewers)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_scene_xml(spec, **kwargs))
    return path
