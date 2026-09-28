#!/usr/bin/env python
"""Solve the "parked at the trocar" home configuration for each robot config.

For every config in ``configs/`` the script solves inverse kinematics so that

* the instrument shaft points along ``--shaft-direction`` (default: world -z,
  i.e. straight down into the abdominal cavity),
* the tip sits ``scene.home_insertion`` below the trocar,
* the tip frame equals the shaft frame used by the viewpoint planner,

then verifies the RCM residual and (optionally) writes ``home_qpos`` back into the
config file.

Usage::

    python scripts/generate_home.py                 # all configs, dry run
    python scripts/generate_home.py ur5e --write    # update configs/ur5e.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrcm.ik import solve_home  # noqa: E402
from vrcm.rcm import angle_between, rcm_error, rcm_length  # noqa: E402
from vrcm.robot import Robot  # noqa: E402
from vrcm.specs import CONFIG_DIR, available_robots, load_robot_spec  # noqa: E402


def report(spec_path: Path, d_des: np.ndarray, *, write: bool) -> bool:
    spec = load_robot_spec(spec_path)
    result = solve_home(spec, d_des=d_des)
    robot = Robot(spec)
    kin = robot.kinematics(result.q)
    p_rcm = np.asarray(spec.trocar)
    length = rcm_length(kin.p_tip, kin.d, p_rcm)
    err = np.linalg.norm(rcm_error(kin.p_tip, kin.d, p_rcm))
    tilt = np.degrees(angle_between(d_des, kin.d))

    print(f"[{spec.name}] IK {'converged' if result.converged else 'FAILED'} "
          f"({result.iterations} it, pos err {result.pos_error * 1e3:.3f} mm, "
          f"rot err {np.degrees(result.rot_error):.3f} deg)")
    print(f"    q      = [{', '.join(f'{v:.4f}' for v in result.q)}]")
    print(f"    tip    = {np.round(kin.p_tip, 4)}   shaft tilt wrt target: {tilt:.3f} deg")
    print(f"    depth  = {length:.4f} m (target {spec.scene.home_insertion:.4f}), "
          f"RCM residual = {err * 1e3:.3f} mm")
    ok = result.converged and err < 1e-4 and abs(length - spec.scene.home_insertion) < 1e-4
    if not ok:
        print("    !! home configuration rejected: trocar not on the shaft axis")

    print(f"    joint margin (rad): {np.round(robot.limit_margin(result.q), 3)}")
    if write:
        data = json.loads(Path(spec_path).read_text())
        data["home_qpos"] = [round(float(v), 6) for v in result.q]
        Path(spec_path).write_text(json.dumps(data, indent=2) + "\n")
        print(f"    -> wrote home_qpos to {spec_path}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robots", nargs="*", help="config names (default: all)")
    parser.add_argument("--write", action="store_true", help="update the config files")
    parser.add_argument("--shaft-direction", default="0,0,-1",
                        help="desired shaft direction in the world frame (default 0,0,-1)")
    args = parser.parse_args()

    d_des = np.array([float(v) for v in args.shaft_direction.split(",")])
    names = args.robots or available_robots()
    ok = True
    for name in names:
        ok &= report(CONFIG_DIR / f"{name}.json", d_des, write=args.write)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
