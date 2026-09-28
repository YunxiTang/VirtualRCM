# Virtual RCM

A **Virtual Remote Center of Motion (V-RCM)** controller for a surgical instrument
or endoscope mounted on a general-purpose arm, simulated in MuJoCo.

The instrument shaft is kept passing through a fixed trocar point while the tool
pitches and yaws about it and inserts or retracts along its axis. Each control
cycle solves a small constrained QP for the joint velocities. The QP combines the
RCM constraint, viewpoint and insertion tracking, and joint position and velocity
limits. An upper-level planner only issues `(pitch, yaw, insertion)` viewpoint
actions. The full derivation is in [method.md](method.md).

Two arms are supported. Both carry the `tom_dissector` instrument, and each arm is
described entirely by a JSON config:

| config | robot |
|---|---|
| [configs/ur5e.json](configs/ur5e.json) | Universal Robots UR5e (6 DOF, MuJoCo Menagerie) |
| [configs/flexiv_rizon4.json](configs/flexiv_rizon4.json) | Flexiv Rizon 4 (7 DOF) |

## Installation

```bash
pip install -e .            # numpy, mujoco
pip install -e ".[dev]"     # + pytest, scipy, matplotlib
pip install -e ".[viz]"     # + viser, for the browser viewer
```

or `pip install -r requirements.txt` to install everything. Python ≥ 3.10.

## Usage

### Browser viewer (sliders + live RCM error plot)

```bash
python scripts/demo_viser.py ur5e --mode dynamic
# open http://localhost:8080
```

The browser panel has three sections:

- **Viewpoint goal**: sliders for pitch and yaw [deg] and depth [mm], a **Home**
  button, a **Hold** button (stop at the current reference) and a **Paused** box.
  The planner rate-limits every change, so dragging a slider gives a smooth motion.
- **Status**: the current RCM error, insertion depth and pitch/yaw reference.
- **RCM error plot**: a scrolling plot of the RCM error [mm] over the last
  `--plot-window` seconds (10 by default) of simulated time.

The scene marks the RCM frame at the trocar with three elements:

- a red sphere at the RCM point;
- a triad in the shaft frame (z along the shaft, x/y the pitch/yaw axes);
- a magenta segment to the closest point on the shaft axis, i.e. the RCM error
  vector, drawn only when the error is non-zero.

This viewer needs no display and no `mjpython`, and works over SSH with port
forwarding.

Options: `--control-rate` (100 Hz), `--render-rate` (30 Hz), `--plot-rate`
(10 Hz), `--plot-window` (10 s), `--port` (8080).

### MuJoCo viewer (keyboard)

```bash
python scripts/demo_viewer.py ur5e --mode dynamic
```

The key bindings avoid every key the MuJoCo viewer reserves, and all of them exist
on a Mac keyboard:

| key | action |
|---|---|
| `Up` / `Down` | pitch ± 2° (`--step`) |
| `6` / `7` | yaw ± 2° |
| `8` / `9` | insert / retract 5 mm (`--insert-step`) |
| `Enter` | return to the home viewpoint |
| `.` | hold the current position |
| `F6` | toggle the RCM error readout in the console (`fn+F6` on a Mac) |
| `Space` | pause / resume |

The RCM frame is drawn the same way as in the browser viewer.

On macOS the viewer has to run under `mjpython`, and the script re-launches itself
under it automatically. With a uv-managed Python, `mjpython` can fail with
`Library not loaded: @rpath/libpython3.12.dylib`. The script works around this by
adding the interpreter's `LIBDIR` to `DYLD_FALLBACK_LIBRARY_PATH`. If you start
`mjpython` by hand, either set that variable yourself or symlink the library into
the venv:

```bash
ln -s "$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')/libpython3.12.dylib" .venv/
```

### Headless validation

```bash
python scripts/run_headless.py --robots ur5e flexiv_rizon4 --modes kinematic dynamic
```

The script runs two experiments for every robot and mode, and writes CSV logs,
plots and scan tables into [results/](results/). It then prints a summary with the
RCM error (max and RMS), the viewpoint direction error, the depth range and the
best scan score. The two experiments are:

- a scripted camera tour (method.md §10);
- a coarse-to-fine stop-and-go viewpoint search driven by a geometric visibility
  model (§11).

### Home configuration

```bash
python scripts/generate_home.py               # dry run for all configs
python scripts/generate_home.py ur5e --write  # write home_qpos into configs/ur5e.json
```

The script solves IK for the configuration "parked at the trocar": the shaft
points along `--shaft-direction` (world −z by default) and the tip sits at the
home insertion depth.

## Execution modes

| mode | behaviour |
|---|---|
| `kinematic` | the QP joint velocity is integrated directly, so the RCM error reflects only the controller; used for analysis and tests |
| `dynamic` | joint velocities become position set-points for the MuJoCo actuators and the model is stepped with `mj_step`, so actuator tracking error shows up as RCM drift that the feedback term has to correct. The set-points include gravity, velocity and acceleration feed-forward (`SimConfig.acceleration_feedforward`); soft servos can be stiffened per robot with `actuator_kv_scale` in the config (10 for the Flexiv) |

## Library

```python
from vrcm import VrcmConfig, VrcmSim, SimConfig, ViewpointPlanner, VrcmTarget, load_robot_spec
```

| module | contents |
|---|---|
| [vrcm/rcm.py](src/vrcm/rcm.py) | RCM kinematics: error, Jacobian, shaft frame |
| [vrcm/qp.py](src/vrcm/qp.py) | dependency-free dense constrained QP solver |
| [vrcm/controller.py](src/vrcm/controller.py) | the QP-based V-RCM velocity controller |
| [vrcm/planner.py](src/vrcm/planner.py) | viewpoint planner: `(pitch, yaw, insertion)` → rate-limited reference |
| [vrcm/visibility.py](src/vrcm/visibility.py) | visibility model and stop-and-go viewpoint search |
| [vrcm/robot.py](src/vrcm/robot.py) | robot-agnostic wrapper around the MuJoCo model |
| [vrcm/scene.py](src/vrcm/scene.py) | merges the robot, instrument and phantom into one scene |
| [vrcm/sim.py](src/vrcm/sim.py) | simulation environment and control loop |
| [vrcm/ik.py](src/vrcm/ik.py) | damped least-squares IK and home-pose solver |
| [vrcm/specs.py](src/vrcm/specs.py) | robot / instrument / scene specs loaded from `configs/` |

To add a robot, drop its MJCF into `assets/robots/`, write a config next to the
existing ones and run `generate_home.py --write`. The package has no
robot-specific code paths.

## Tests

```bash
pytest
```

## License

Apache 2.0, see [LICENSE](LICENSE). The robot and instrument assets under
[assets/](assets/) keep their own licenses.
