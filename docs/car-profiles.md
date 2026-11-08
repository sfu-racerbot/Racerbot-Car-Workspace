# Car profiles

> **Who this is for:** a new member checking which car a launch will use, or changing hardware calibration or algorithm settings.
> **Read first:** [Concepts](concepts.md) for ROS 2 programs and configuration files; [Hardware reference](hardware-reference.md) for measured values.
> **You'll be able to:** identify the selected car, edit its calibration, add a justified tuning override and prepare a third car's profile.
> **Time:** about 15 minutes to read; physical calibration takes longer.

Car 1 (`rb1`) and car 2 (`rb2`) run the same workspace (project folder). A car profile selects the hardware values without requiring separate copies of the driving code.

## Highlights

- **Serial-based selection on registered Jetsons.** A conflicting development override stops identity resolution.
- **Three configuration layers.** Shared settings load first, followed by that car's calibration and tuning files.
- **Checked tuning exceptions.** Every overridden setting needs a reason, and safety bounds cannot be relaxed through tuning.
- **Recorded configuration identity.** Run metadata includes the car ID and each calibration or tuning file's path and SHA-256 hash (a content fingerprint).

These behaviours are implemented in [`identity.py`](../src/racerbot_car/racerbot_car/identity.py), [`layers.py`](../src/racerbot_car/racerbot_car/layers.py), [`guards.py`](../src/racerbot_car/racerbot_car/guards.py) and [`run_metadata.py`](../src/race_diagnostics/race_diagnostics/run_metadata.py). The [package README](../src/racerbot_car/README.md) describes the Python interfaces.

## How the workspace knows which car it is

`racerbot_car.identity.current_car_id()` reads `/proc/device-tree/serial-number` and looks it up in [`cars/machines.yaml`](../src/racerbot_car/cars/machines.yaml). This file maps each quoted Jetson serial to a car ID.

On a registered car, leave `RACERBOT_CAR` unset. An empty override or one matching the serial is accepted; a conflicting non-empty value raises `CarMismatchError` rather than silently loading the other car's settings.

On a development machine or for the simulator, set `RACERBOT_CAR` to a known ID. The resolver accepts a known override when the serial is absent **or unregistered**; it is not proof that an unregistered machine is a development computer. Register a physical car's serial before using it.

Source: [`identity.py`: `read_serial`, `resolve_car_id`, `current_car_id`](../src/racerbot_car/racerbot_car/identity.py).

**Terminal 1 (development machine, workspace root):** after the usual ROS and workspace setup from [Concepts](concepts.md), select car 1 and ask the read-only doctor to report it:

```bash
export RACERBOT_CAR=rb1
ros2 run racerbot_doctor doctor
```

**Working when:** the first line is `Car rb1 (serial none), tuning overrides: none` on a machine without a Jetson serial or tuning files. Missing physical devices on a development machine can still make later hardware checks fail.

With tuning files, the doctor prints `Car ID (serial SERIAL), tuning overrides:` followed by one line per override: filename, node block, key, value and reason. `none` replaces an absent serial. Source: [`racerbot_doctor/cli.py`](../src/racerbot_doctor/racerbot_doctor/cli.py).

Bringup also calls `layers.launch_car_id()`, which checks connected USB IDs against the profiles before creating nodes (running ROS programs, see the [glossary](glossary.md)). A device distinctive to another car can cause `CarHardwareError`; an unplugged device alone is not a contradiction. Source: [`layers.py`](../src/racerbot_car/racerbot_car/layers.py) and [`hardware.py`](../src/racerbot_car/racerbot_car/hardware.py).

### Identity errors and what to do

The following message templates come from [`identity.resolve_car_id`](../src/racerbot_car/racerbot_car/identity.py). `SERIAL`, `ID` and `OTHER` stand for the actual values in the message.

| Error | Message | Action |
|---|---|---|
| `CarMismatchError` | `serial SERIAL identifies ID, but RACERBOT_CAR is OTHER; unset RACERBOT_CAR` | In that terminal run `unset RACERBOT_CAR`, then retry. Check the serial registration if it names the wrong physical car. |
| `UnknownCarError` | `unknown serial SERIAL; add to cars/machines.yaml:` followed by `"SERIAL": <car id>` | Identify the physical car, create its profile if needed, then add its quoted serial mapping. Replace the printed placeholder with its real ID. |
| `UnknownCarError` | `unknown RACERBOT_CAR 'OTHER'; known ids: [...]` | Correct the override to an ID in the printed list, or unset it on a registered car. |
| `UnknownCarError` | `no Jetson serial was found; set RACERBOT_CAR for a dev box or the simulator` | On a development machine, set a known ID as above. On a Jetson, investigate the missing serial rather than guessing the car. |

### Tests on a car

Every pytest run in this workspace hides the car's Jetson serial and its USB devices, so a test suite gives the same results on a car as on a development machine.

`racerbot_car` uses both host facts to refuse a contradicting car. `identity` checks the serial. `layers.launch_car_id` scans USB and refuses a launch when a device distinctive to another car is plugged in.

Without hiding them, a test that selects the other car, or no car, is refused on the car it runs on. On 2026-11-01 the serial refused about 100 tests on car 1 and 628 on car 2, and car 2's RealSense then refused 309 launch tests.

- The workspace [`conftest.py`](../conftest.py) points `racerbot_car.identity.SERIAL_PATH` and `racerbot_car.hardware.USB_DEVICES_ROOT` at paths that do not exist, for each test.
- The workspace [`pytest.ini`](../pytest.ini) makes the workspace root pytest's root directory for every suite, including under `colcon test`. Without it, a package with its own `setup.py` becomes its own root directory and pytest never loads that `conftest.py`.
- A package must not carry its own pytest configuration: a `pytest.ini`, a `[tool:pytest]` section in `setup.cfg`, `[tool.pytest.ini_options]` in `pyproject.toml` or `[pytest]` in `tox.ini`. pytest takes the nearest one as the root directory, so it never loads the workspace `conftest.py`, and that package's tests see the real hardware. Register a marker from the package's own `conftest.py` with `config.addinivalue_line('markers', ...)` instead, as `opponent_perception` does.
- [`test_no_package_pytest_config.py`](../src/racerbot_car/test/test_no_package_pytest_config.py) fails on any such file. Its one allowed exception is the vendored `teleop_tools/pytest.ini`, which is safe only while nothing under `teleop_tools` imports `racerbot_car`; the test checks that too. It was added on 2026-11-02, after `opponent_perception`'s `setup.cfg` had made that package its own root directory.
- A subprocess started by a test, such as a `ros2 launch` or a script run with `python3`, gets the same treatment from a test-only [`sitecustomize.py`](../tools/dev/test_support/hide_car_hardware/sitecustomize.py), which the fixture puts on `PYTHONPATH`. A test that sets `PYTHONPATH` for a subprocess must add to the inherited value, not replace it, or the subprocess sees the real hardware.

Production code has no switch to ignore the serial or the USB check; the override exists only inside a test run. Tests of `identity` or of the USB check set their own serial path or devices, which still take precedence.

## Where each car's values live

Replace `ID` with `rb1` or `rb2` in the paths below. Device fields are validated by [`profile.load_profile`](../src/racerbot_car/racerbot_car/profile.py); calibration keys are listed by node block in [`guards.CALIBRATION_KEYS`](../src/racerbot_car/racerbot_car/guards.py).

| Path under `src/racerbot_car/cars/ID/` | Contents |
|---|---|
| `car.yaml` | LiDAR (laser range scanner) transport and address, camera, dashboard site, distinctive USB IDs, optional hostnames and signed full-lock steering angles. See the [glossary](glossary.md). |
| `calibration/vesc.yaml` | VESC (motor and steering controller) gains, steering centre and servo clamps. Odometry estimates motion from wheel speed and steering; see the [glossary](glossary.md). |
| `calibration/sensors.yaml` | LiDAR serial port or Ethernet IP and port. |
| `calibration/joy_teleop.yaml` | Full-stick steering scale for manual driving. |
| `calibration/gap_follow.yaml` | Maximum steering angle and arc-guard correction limit. |
| `calibration/pure_pursuit.yaml`, `calibration/neural_racer.yaml` | Controller steering limits; car 2's `pure_pursuit` also sets its 2.7 m/s² online lateral ceiling. |
| `calibration/auto_map_race.yaml` | Car 2's planning physics, including 3.2 m/s² lateral acceleration and 0.22 s speed delay. |
| `calibration/odom_calibration.yaml` | Vehicle label, steering bound and fallback calibration values used by the wizard. |
| `tuning/CONFIG.yaml` | Justified per-car algorithm exceptions. Car 2 has `tuning/pure_pursuit.yaml`; car 1 has none. |
| `imu/car.yaml`, `imu/calib/*.json` | IMU (rotation and acceleration sensor) mount and calibration files. |

The two existing profiles are [`rb1/`](../src/racerbot_car/cars/rb1/) and [`rb2/`](../src/racerbot_car/cars/rb2/). The IMU launch selects `imu/car.yaml` in [`vehicle_state_launch.py`](../src/imu_bridge/launch/vehicle_state_launch.py); the [IMU tools guide](../tools/imu_calibration/README.md) covers calibration.

### How layers combine

A launch using `layers.layered()` / `car_params()` passes parameter files in this order:

1. The shared configuration, located by `layers.SHARED_CONFIGS` for code reading effective values.
2. `cars/ID/calibration/CONFIG.yaml`, if it exists.
3. `cars/ID/tuning/CONFIG.yaml`, if it exists.

An overlay is a YAML parameter file applied after another file. Use the shared file's **exact node block name** and a `ros__parameters` mapping: differently spelled blocks can stay separate in ROS rather than overriding each other.

`overlay_files()` returns existing overlays in calibration-then-tuning order. `layered()` checks required calibration keys and validates tuning before returning the list; `effective_config()` reads the resulting values for a named node. Source: [`layers.py`](../src/racerbot_car/racerbot_car/layers.py).

[`paths.py`](../src/racerbot_car/racerbot_car/paths.py) locates profiles in the source package, or its installed share directory. `RACERBOT_CARS_DIR` overrides that location. Data paths are `data/ID/KIND` under `RACERBOT_WS` (default `~/racerbot-ws`), with kinds `racerbot_runs`, `racerbot_auto`, `racerbot_sim` and `imu_calibration`; locating a path does not create it.

## Change calibration

1. Confirm the selected car with the doctor. Match the physical car and the profile before measuring anything.
2. Use [Odometry calibration](odom-calibration.md) for speed, servo centre, steering gain and travel. Edit device identity in `car.yaml` and the corresponding sensor overlay when replacing a sensor.
3. Keep calibration overlays within the node blocks and keys in `guards.CALIBRATION_KEYS`. Run the checks below before relaunching.

`guards.check_calibration_overlay()` rejects keys outside that allow-list. The calibration tests scan every `*/calibration/*.yaml`. At launch, `layered()` separately refuses missing required keys with `MissingCalibrationError`, naming the file and missing keys. Sources: [`guards.py`](../src/racerbot_car/racerbot_car/guards.py), [`layers.py`](../src/racerbot_car/racerbot_car/layers.py), [`test_guards_calibration.py`](../src/racerbot_car/test/test_guards_calibration.py).

### What the odometry wizard writes

By default, the wizard writes the selected car's **`calibration/vesc.yaml` and `calibration/joy_teleop.yaml`**, not the shared files. `CalibrationNode._resolve_config()` resolves those paths through `identity.current_car_id()` and refuses a missing overlay instead of falling back to shared config.

`_apply_parameters()` splits VESC updates from the stick scale, calls `vesc_yaml.apply()` / `teleop_yaml.apply()`, and stores backups under the session report directory's `backups/`. It restores completed writes if a later write fails.

Explicit `vesc_yaml_path` / `joy_teleop_yaml_path` parameters override the default destinations. Inspect those before applying a change: an explicitly supplied shared-file path can still direct a write there. The wizard does not automatically update the other controllers' limits or its fallback overlay.

Sources: [`calibration_node.py`](../src/odom_calibration/odom_calibration/calibration_node.py), [`vesc_yaml.py`](../src/odom_calibration/odom_calibration/vesc_yaml.py), [`teleop_yaml.py`](../src/odom_calibration/odom_calibration/teleop_yaml.py).

**Retain the LB deadman and the test order: static checks, wheels off the ground, then low-speed floor testing.** A correct YAML file does not establish that a new calibration drives correctly. Follow [Testing before it is on wheels](writing-your-own-node.md#testing-before-its-on-wheels).

## Add a tuning override

### Car 2's first tuning overlay

[`rb2/tuning/pure_pursuit.yaml`](../src/racerbot_car/cars/rb2/tuning/pure_pursuit.yaml) sets `steering_law: delay_lqr`. Shared settings load first, then car 2's calibration, then this tuning file, making delay LQR car 2's default race steering law.

The `# why:` comment cites car 2's measured steering delays: pure pursuit weaves at race speed, while delay LQR predicts through those delays. Car 1's steering chain has not been measured, so it retains the shared `pure_pursuit` default.

**Simulator-validated only, not driven.** The shipped builder and six seeds per course gave 11.40–11.47 s on 06a and 8.75–8.87 s on 06b, with zero contact and stop ticks. See [the results and calibration rationale](racing-autonomy.md#how-the-map-racer-beats-gap_follow-on-car-2).

### Create another override

Keep algorithm tuning shared unless there is a reason for a per-car exception. For example, a lower speed cap belongs in `tuning/gap_follow.yaml`, while a measured servo gain belongs in `calibration/vesc.yaml`.

1. Find the shared file through [`layers.SHARED_CONFIGS`](../src/racerbot_car/racerbot_car/layers.py), and check its node block and parameter name.
2. Create `cars/ID/tuning/CONFIG.yaml` with that same block and only the parameters that differ.
3. Put a non-empty `# why:` comment **directly above every parameter leaf**, including nested leaves. State the measurement or physical reason.
4. Run the checks below and the doctor. Confirm that the reason and value appear in its override list before driving.

This YAML is an illustrative lower cap for `src/racerbot_car/cars/rb1/tuning/gap_follow.yaml`, not a measured recommendation:

```yaml
gap_follow_node:
  ros__parameters:
    # why: Temporary lower cap while checking a replacement tire on rb1.
    max_speed: 1.0
```

[`guards.check_tuning_overlay()`](../src/racerbot_car/racerbot_car/guards.py) checks each leaf against the **shared file**, not the calibration overlay. Every key must exist in the shared file's same node block and must not be a calibration key for that config.

### Safety rules for tuning

The guard compares the leaf name (the part after the final dot for nested parameters). A bad overlay makes `layered()` raise `TuningOverlayError`, so launch refuses to continue. Source: [`guards.py`](../src/racerbot_car/racerbot_car/guards.py).

| Rule | Keys |
|---|---|
| `SAFETY_TRUE`: may only be `true` | `enable_deadman`, `enable_ttc`, `enable_swept_clearance` |
| `SAFETY_FIXED`: cannot appear in tuning | `deadman_button` |
| `SAFETY_CEILING`: cannot exceed the shared value | `max_speed`, `profile_max_speed`, `profile_max_brake`, `ttc_min_closing_speed`, and every name ending in `*_timeout_sec` |
| `SAFETY_FLOOR`: cannot go below the shared value | `ttc_threshold_sec`, `ttc_path_margin`, `car_width`, `car_length`, `optimize_car_width`, `emergency_stop_clearance`, `forward_stop_clearance`, `swept_clearance_margin`, `swept_clearance_max_length`, `emergency_stop_distance`, `profile_wall_clearance`, `narrow_corridor_clearance`, `open_corridor_clearance`, `overtake_min_side_clearance`, `body_clearance_fov_deg`, `emergency_arc_clearance`, `emergency_arc_length`, `emergency_hard_stop_distance` |

Ceiling and floor values must be numeric and not NaN (not-a-number). These checks prevent a per-car tuning file from weakening the listed safety bounds; they do not validate physical performance.

## Add a third car

Use `rb3` for the next car. [`identity.known_car_ids()`](../src/racerbot_car/racerbot_car/identity.py) discovers folders matching `rb` followed by digits **with a `car.yaml` file**; no central ID list needs editing.

1. Create `src/racerbot_car/cars/rb3/car.yaml` using an existing profile's structure, then replace its values with verified car 3 values. Required fields are `name`, `lidar`, `camera`, `dashboard_site`, `distinctive_usb_ids` and `steering.full_lock_rad.left/right`; `hostnames` is optional.
2. Choose a supported LiDAR kind (`serial` with `port` / `usb_id`, or `ethernet` with `ip` / integer `port` / `iface` / `host_ip`). Camera kind is `uvc` with `device`, `realsense` with `usb_id` / `topic`, or `none`.
3. Use lowercase `vvvv:pppp` USB IDs. Full-lock angles must be finite, left positive and right negative, each with magnitude strictly between zero and `0.8 rad`. Unknown profile fields are errors. Source: [`profile.load_profile()`](../src/racerbot_car/racerbot_car/profile.py).
4. Supply the seven calibration files listed above, as both existing cars do. Copy the block structure, then measure the new car's values; satisfy every required entry in `guards.CALIBRATION_KEYS`. Add `imu/car.yaml` and its referenced `imu/calib/` files for the IMU stack. Leave tuning absent unless justified.
5. Add the Jetson's actual serial to `cars/machines.yaml`, quoted so leading zeros survive. For example, replace `CAR3_SERIAL` in `"CAR3_SERIAL": rb3` with the serial from `/proc/device-tree/serial-number`. Duplicate serials, unquoted numeric serials and mappings to unknown IDs are rejected by `identity.load_machines()`.
6. Check downstream settings too. The dashboard launch selects `src/racerbot_launch/config/web_dashboard_rb3.yaml` using the **car ID**; it needs a matching settings file. Keep its site settings consistent with `car.yaml`'s `dashboard_site`. Source: [`dashboard_launch.py`](../src/racerbot_launch/launch/dashboard_launch.py).
7. Run the checks below, then follow the physical test order before driving.

The [steering-limit guard](../src/racerbot_launch/test/test_steering_limits_fit_servo.py) and [`tools/f1tenth_sim` tests](../tools/f1tenth_sim/tests/conftest.py) automatically include each known car with `calibration/vesc.yaml`. The historical hardware oracle in [`test_car_profiles.py`](../src/racerbot_launch/test/test_car_profiles.py) explicitly covers `rb1` and `rb2`; it does not automatically invent hardware expectations for `rb3`.

**Workspace-root development terminal:** run the profile and launch checks in an isolated ROS domain. The domain keeps test nodes off the car's normal ROS graph.

```bash
RB_WS=$PWD rb env ROS_DOMAIN_ID=65 python3 -m pytest src/racerbot_launch/test src/racerbot_car/test -q -p no:cacheprovider
```

**Working when:** pytest reports all tests passed. A failure names the profile, overlay or hardware expectation that needs attention. Fix the configuration rather than weakening the test.

## What a recorded run remembers

[`race_diagnostics.run_metadata.write_run_metadata()`](../src/race_diagnostics/race_diagnostics/run_metadata.py) writes `metadata/car.yaml` with `car_id` and an `overlays` list. Each entry contains the overlay path relative to the source root and its `sha256` content hash, covering existing calibration and tuning YAML files.

If identity cannot be resolved, the recorder writes `car_id: null` and an empty overlay list, with the identity error in `metadata/car_id.txt`. Overlay read failures are recorded in an `errors` list. Check these files before treating a run as evidence for a particular car's settings.
