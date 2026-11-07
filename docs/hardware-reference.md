# Hardware reference

> **Who this is for:** anyone looking up a port, address or calibration value for either car.
> **Read first:** nothing. This is a lookup table, not a tutorial.
> **What's in it:** VESC, LiDAR, joystick and network specifics, plus the gotchas that look like bugs and aren't.

The workspace runs on car 1 (`rb1`) and car 2 (`rb2`). Device selection and calibration live in `src/racerbot_car/cars/ID/`, where `ID` is the car's identifier. See [Car profiles](car-profiles.md) before changing a value.

## Shared stack and servo law

Both cars use ROS 2 Jazzy, a Jetson Orin Nano Super and a VESC motor and steering controller. A [node](glossary.md) is a running ROS program; a topic is its named message channel.

The shared VESC code uses `servo_position = gain × steering_angle + offset`. Each car supplies its own gain, offset and clamps in `calibration/vesc.yaml`. Steering angle is in radians; servo position is a unitless fraction.

The shared driving stack retains the LB deadman (hold-to-drive button). **Keep LB held whenever driving; never disable `enable_deadman`.** Follow the static checks, wheels-off-the-ground checks, then low-speed floor checks in [Writing your own node](writing-your-own-node.md).

The geometry below is shared configuration, measured on car 2 and assumed for car 1. The car-specific sections follow it.

## Physical dimensions used in config

> **Car 1: these are car 2's measurements.** Car 1 adopted them in October 2026. **TODO(verify on car 1):** measure the wheelbase, track width and LiDAR distance behind the front axle before treating them as car 1 measurements.

The chassis is a Traxxas Ford Fiesta ST Rally VXL 74276-4. Car 2 is not stock: its tape measurements on **2026-08-24** differed from the published figures.

| What was measured | Value | What the configs said before |
|---|---|---|
| Wheelbase (front axle centre to rear axle centre) | **0.36 m** | 0.324 m, the Traxxas published figure |
| Width (outer tire edge to outer tire edge) | **0.30 m** | 0.281 m, the published body width |
| LiDAR distance behind the front axle | **0.10 m** | estimated 0.10 m behind the nose, a different reference point |

`base_link` is the rear-axle coordinate frame (the origin for poses and collision checks, see the [glossary](glossary.md)). The LiDAR offset is therefore `0.36 − 0.10 = 0.26 m` forward, rather than the old 0.33 m estimate.

### Where each number lives

Change every consumer in a row together, or different parts of the stack model different geometry. These values remain shared rather than per-car calibration keys.

| Number | Files |
|---|---|
| Wheelbase `0.36 m` | `src/f1tenth_system/f1tenth_stack/config/vesc.yaml`, `src/pure_pursuit/config/{pure_pursuit,auto_map_race}.yaml`, `src/gap_follow/config/gap_follow.yaml`, `src/odom_calibration/config/odom_calibration.yaml`, `src/imu_bridge/config/vehicle_state.yaml`, `src/racerbot_sim/racerbot_sim/sim_bridge.py`, `tools/f1tenth_sim/sim_fidelity/calibration.py` |
| LiDAR offset `0.26 m` forward, `0.11 m` up | static transforms in the three `src/f1tenth_system/f1tenth_stack/launch/*bringup_launch.py` files, `src/racerbot_sim/launch/sim_bringup_launch.py`, `src/racerbot_launch/launch/dashboard_test_launch.py`; `laser_offset_x` in `gap_follow.yaml`, `pure_pursuit.yaml`, `auto_map_race.yaml`, `waypoint_recorder.yaml` and `src/racerbot_launch/config/web_dashboard_rb1.yaml` / `web_dashboard_rb2.yaml` |
| Collision width `0.31 m` | `gap_follow.yaml`, `pure_pursuit.yaml`, `auto_map_race.yaml` (`optimize_car_width`) and both simulators |

The dashboard tests compare the offsets with bringup transforms. The planner, recorder, controller and dashboard also use this offset to convert a LiDAR pose to a rear-axle pose; see [Localization](localization.md#which-point-on-the-car-the-position-means).

### The collision envelope is padded on purpose

`car_width` and `car_length` describe a rectangle larger than the measured chassis. Zero clearance means the padding is gone, rather than that the body is touching.

Car 2's run logs on **2026-08-25** showed false `emergency_clearance` stops at −10 mm to −13 mm with no contact. They tracked the old 14.5 mm-per-side padding, so width padding was cut to 5 mm per side: `0.30 + 2 × 0.005 = 0.31 m`.

Before the 2026-08-24 measurement, `0.31 m` also appeared in config, but came from adding 14.5 mm per side to the incorrect 0.281 m body width. The matching digits do not validate the earlier model.

Length remains **0.58 m**, with no bumper-to-bumper measurement. If the published total overhang of 0.211 m still applies, the chassis is about 0.571 m long, leaving roughly 4.5 mm padding at each end rather than the intended 22.5 mm.

**TODO(verify): measure rear-axle centre to the front bumper, and bumper to bumper on each car.** The assumed symmetric overhang puts the nose 0.21 m ahead of the LiDAR; the code subtracts that distance before forward stop decisions.

### What re-measuring invalidated

- Racing lines recorded with the old 0.33 m LiDAR offset are about 0.07 m out along the heading. Re-record them for racing.
- Odometry (wheel-based motion estimates, see the [glossary](glossary.md)) integrated heading as `speed × tan(steering) / wheelbase` with a wheelbase 10% too short. This overestimated yaw by about 10%. Re-run [odometry calibration](../src/odom_calibration/README.md).
- At the old 0.26 rad steering limit, minimum turning radius grew from 1.22 m to **1.35 m** (`0.36 / tan(0.26)`). Current car-specific radii are listed below. A longer wheelbase turns wider at the same angle, so an optimizer may correctly reject a formerly accepted course.
- Saved maps remain usable. The change is where `base_link` sits relative to the scan (one set of laser range readings, see the [glossary](glossary.md)) that built them.

## Car 1 (rb1)

### Profile sources

Car 1's device paths, USB ID, dashboard site and full-lock angles are in [`rb1/car.yaml`](../src/racerbot_car/cars/rb1/car.yaml). Its gains and clamps are in [`calibration/vesc.yaml`](../src/racerbot_car/cars/rb1/calibration/vesc.yaml).

The full-stick scale is in [`calibration/joy_teleop.yaml`](../src/racerbot_car/cars/rb1/calibration/joy_teleop.yaml). Controller limits are in [`calibration/gap_follow.yaml`](../src/racerbot_car/cars/rb1/calibration/gap_follow.yaml), [`pure_pursuit.yaml`](../src/racerbot_car/cars/rb1/calibration/pure_pursuit.yaml) and [`neural_racer.yaml`](../src/racerbot_car/cars/rb1/calibration/neural_racer.yaml). The planner's 0.30 rad limit remains shared in `src/pure_pursuit/config/auto_map_race.yaml`.

### Compute

- Jetson Orin Nano Super (Developer Kit), JetPack 7.2, L4T R39.2, Ubuntu 24.04.4 "noble", aarch64, 8GB RAM
- ROS2 **Jazzy**, installed natively (not in a container)
- Accounts: `racerbotcar-1` (sudo/admin) and `racermember1` (no sudo, ACL-scoped to this workspace only — see main [README.md](../README.md))

### VESC (motor + steering controller)

| | |
|---|---|
| Connection | USB, exposed as `/dev/sensors/vesc` (symlink to whatever `/dev/ttyACM*` it enumerates as) |
| udev rule | `/etc/udev/rules.d/99-vesc.rules` — matches idVendor `0483`, idProduct `5740` (STMicroelectronics Virtual COM Port), `MODE=0666` with no group, `SYMLINK+="sensors/vesc"`, so any user can open it without `dialout` or a re-login. Copy in the repo: `tools/udev/99-vesc.rules`; `sudo tools/udev/install.sh` installs all of car 1's rules |
| Firmware | 7.0 (confirmed via `vesc_driver_node`'s connection log) |
| Group access | Not needed for the VESC (`MODE=0666`). `racerbotcar-1` is in `dialout` anyway (checked 2026-10-24) |

**Config:** car 1's `calibration/vesc.yaml` over the shared `src/f1tenth_system/f1tenth_stack/config/vesc.yaml` (flattened excerpt, not a standalone ROS parameter file):
```yaml
speed_to_erpm_gain: 4600.0       # /** block: sets the motor command
speed_to_erpm_offset: 0.0
steering_angle_to_servo_gain: -0.83
steering_angle_to_servo_offset: 0.366
servo_min: 0.05
servo_max: 0.75
# vesc_to_odom_node block only:
speed_to_erpm_gain: -4040.0      # negated on purpose -- see below
wheelbase: 0.36   # meters -- car 2's measurement, assumed for car 1 (see Physical dimensions)
```
The servo formula: `servo_position = -0.83 × steering_angle + 0.366` (gain fitted 2026-10-21 from five floor half circles; the old -1.1135 under-predicted real steering by about 30%). Steering angle `0.0` → servo position `0.366` (center on this car, set 2026-10-21 from a drive-straight run: 16 cm left over 2.83 m at 0.35). This is what you'll see repeatedly if you inspect `/commands/servo/position` — `0.366` is neutral, not a bug. (Car 2's neutral is a different value; each car's servo is calibrated separately.)

**The odometry gain is negative on purpose.** `vesc_to_odom.cpp` hardcodes a negation of the eRPM the VESC reports, and this car's VESC already reports eRPM with the commanded sign, so with a positive gain `/odom` would read *negative* speed while the car drives forward. The `vesc_to_odom_node` block overrides the gain to `-4040.0` to cancel that (magnitude calibrated 2026-10-21 from five short tape runs, median of implied gains: 3959, 3936, 4044, 4088, 4179); the shared `/**` block keeps the positive `+4600.0` for the motor command. Until October 2026 car 1 carried this fix as an edit inside `vesc_to_odom.cpp` instead; it moved to the YAML to match car 2. **Never apply both** — they cancel, and `/odom` runs backwards.

**Motor command gain, 2026-11-02: `4600.0`, was `3650.0`.** The command gain sets how many eRPM `ackermann_to_vesc_node` asks for per m/s of `/drive` speed, so it decides how fast the car really goes for a given command. At 3650 car 1 delivered 0.79 of every commanded speed at racing pace, because it needs about 4600 eRPM per true m/s there. That cost about 4.4 s on a standing-start lap of the 2026-10-30 course (simulated 24.1 s against 19.7 s); the full attribution is in [the lap-gap report](rb1-lap-gap-2026-10-30.md).

- **Measured by:** `tools/run_analysis/loc.py` locating car 1's 2026-10-30 gap_follow runs on car 2's map of the same course (0.95 to 0.98 of beams within 10 cm), then `tools/run_analysis/speed_gain.py` comparing map distance with the integral of the `/drive` speed over each LB hold.
- **Result:** on the solo race-pace runs (`20261030-180531`, `20261030-190734`; 232 m), map distance over commanded distance was 0.793, so the gain that delivers the command is 3650 / 0.793 = 4604. Over all ten of that day's holds (two-car and slower ones included, 556 m) it is 4481.
- **Car 2 checks the method:** the same measurement on its five 2026-10-30 holds gives 1.013 at its 4614 gain, and the simulator's independent floor fit for car 2 gives 1.004.
- **Speed matters:** below 1.5 m/s car 1 needs about 3980 eRPM per true m/s, which is why the 2026-10-21 tape runs (2 to 2.6 m, slow) gave 4040 for odometry. Faster, the wheels slip more and it rises to about 4600. Commands below 1.5 m/s now run about 15% fast; car 2 does the same at its 4614.
- **VESC clamp:** 5.0 m/s × 4600 = 23000 eRPM, inside the shared `speed_max` of 23250, so gap_follow's 5.0 m/s top speed is not clipped. `src/racerbot_launch/test/test_speed_limits_fit_vesc.py` fails if any shipped top speed would be.

**Every commanded speed on car 1 is now about 26% faster in reality** (4600 / 3650), manual teleop's full stick included (5.0 m/s commanded, about 3.9 real before, about 5.0 now). The first run after this change follows [the gain check procedure](rb1-lap-gap-2026-10-30.md#first-run-after-the-gain-change): wheels off the ground with LB held, then the floor at low speed.

The odometry gain stays at 4040, so `/odom` reads 15 to 17% above true speed at racing speed (car 2's reads 3.5% above). That over-read makes TTC and the braking ceiling act early, which is the safe side; changing it is a separate decision, listed in the report.

**Steering limits, 2026-10-22.** Car 1's 2026-10-21 check circles at the `-0.83` gain gave **0.388 rad right and 0.401 rad left** for a 0.380 rad command (full stick, +2% and +5%). Left full stick lands on `servo_min` (0.366 − 0.83 × 0.38 = 0.051), so 0.38 rad is the most full stick can be without clipping. Car 2 has its own, different numbers (its own circles, 2026-10-16); they don't apply here.

- Full stick (`joy_teleop.yaml` scale) is **0.38 rad**, which both sides reach.
- Autonomy is held below that: `max_steering_angle` **0.33 rad = 19°** in `gap_follow.yaml` and `pure_pursuit.yaml` (1.05 m minimum turning radius). The race planner's `profile_max_steering_angle` is **0.30 rad = 17°** in `auto_map_race.yaml` (1.16 m), so the controller has steering left on the tightest planned corner. Both were 0.26 until the 2026-10-22 car 2 port. **Not yet validated on car 1:** wheels off the ground, then low speed.
- `src/racerbot_launch/test/test_steering_limits_fit_servo.py` fails if any of these four limits would be clipped by `servo_min`/`servo_max`, or if the planner is not below the controller.
- The calibration wizard can run the circle procedure itself: see [odom-calibration.md](odom-calibration.md#finding-how-far-the-steering-really-goes).
- Re-centring the servo horn, a different servo, or any linkage change invalidates all of this. Redo the drive-straight run and the circles (`docs/odom-calibration.md`).

**Servo output must be enabled in the VESC's own firmware config** — this is separate from anything in this repo and isn't visible to `ros2`/`colcon` at all. If a freshly-flashed or reset VESC has steering that does nothing despite `fault_code: 0` and a clean serial connection, check this first. Requires **VESC Tool** (official app, vesc-project.com) connected over USB — which means stopping the ROS bringup first, since only one process can hold the serial port. Full story in [troubleshooting.md](troubleshooting.md).

**The onboard IMU works, once it is switched on.** The note measured on 2026-08-19 (on car 2, and copied here) found `/sensors/imu/raw` publishing at 50 Hz with every field exactly `0.0` and no gravity on any axis: the IMU was disabled in the VESC's firmware. It has been turned on in **VESC Tool → App Settings → IMU** (a sample rate of 0 means off), and the team's October 2026 tests show it working on both cars. Like the servo-output flag above, it can only be read in VESC Tool, not from ROS.

If it ever reads all zeros again, it has been switched off: `imu_bridge`'s `/imu/status` then reports `fault` with an all-zero reason.

To re-check after changing anything in VESC Tool:

**Terminal 1** — the VESC driver on its own. Nothing publishes commands to it, so the car cannot move.

```bash
ros2 run vesc_driver vesc_driver_node --ros-args \
  --params-file install/f1tenth_stack/share/f1tenth_stack/config/vesc.yaml
```

**Working when:** it logs `Connected to VESC with firmware version 7.0` and keeps running. Leave it up.

**Terminal 2** — read the gyro.

```bash
ros2 topic echo /sensors/imu/raw --field angular_velocity
```

**Working when:** `z` moves when you rotate the car by hand, and the accelerometer shows roughly 9.81 on one axis at rest. All zeros means the IMU is still off.

Reading or writing the VESC's app configuration (including that servo-output flag) is **not possible from this ROS stack** — `vesc_driver` only implements motor/servo *control* commands (`COMM_SET_SERVO_POS`, `COMM_SET_RPM`, etc.), not the config read/write protocol (`COMM_GET_APPCONF`/`COMM_SET_APPCONF`) that VESC Tool uses. Don't go looking for a `ros2 service`/CLI way to do this — there isn't one here.

<a id="lidar--hokuyo-ust-30lx"></a>

### LiDAR: Hokuyo UST-30LX

Source: [`rb1/car.yaml`](../src/racerbot_car/cars/rb1/car.yaml) and [`calibration/sensors.yaml`](../src/racerbot_car/cars/rb1/calibration/sensors.yaml). Scan angles and rate settings remain in the shared `sensors.yaml`.

| | |
|---|---|
| Connection | USB (not Ethernet) — enumerates as `/dev/ttyACM*`, exposed as a stable `/dev/sensors/hokuyo` symlink |
| udev rule | `/etc/udev/rules.d/99-hokuyo.rules` — matches idVendor `15d1`, idProduct `0000` (Hokuyo Data Flex for USB, URG-Series USB Driver), `MODE=0666`, `SYMLINK+="sensors/hokuyo"`. Copy in the repo: `tools/udev/99-hokuyo.rules` (`sudo tools/udev/install.sh`) |
| Config | `src/racerbot_car/cars/rb1/calibration/sensors.yaml` sets `serial_port: "/dev/sensors/hokuyo"`; the effective configuration has no `ip_address` or `ip_port` |
| Scan rate | ~40Hz, ±180° (`angle_min`/`angle_max`: ±3.14 rad in `sensors.yaml`) |
| Frame | `laser`, offset from `base_link`: +0.26m forward, +0.11m up, no rotation — **car 2's measurement, assumed for car 1; not yet measured on this chassis** (see [Physical dimensions](#physical-dimensions-used-in-config)) |

If `/dev/sensors/hokuyo` doesn't show up after plugging the LiDAR in: `lsusb` should list `ID 15d1:0000`, and `ls -la /dev/ttyACM*` should show a device owned by group `dialout`/mode `0666`. `sudo udevadm control --reload-rules && sudo udevadm trigger` re-applies the rule without a reboot.

No static IP, dedicated NIC, or `nmcli` connection is needed for this car's LiDAR — that's a car 2–specific detail (Ethernet Hokuyo UST-10LX) that doesn't apply here.

<a id="camera--uvc-usb-webcam"></a>

### Camera: UVC USB webcam

Car 1's [`car.yaml`](../src/racerbot_car/cars/rb1/car.yaml) selects a UVC webcam at `/dev/video0` and dashboard site `rb1`. `ros2 launch racerbot_launch webcam_launch.py` starts `usb_cam_stream` from `src/web_dashboards`; `src/racerbot_launch/config/web_dashboard_rb1.yaml` also holds the stream's device setting.

The package defaults are 1280×720 at 30 FPS, served on port 9090. Use `v4l2-ctl --list-devices` to check enumeration changes. See [USB camera livestream](usb-camera-livestream.md).

<a id="joystick--logitech-f710"></a>

### Joystick: Logitech F710

| | |
|---|---|
| Mode | **Must be XInput** (switch on the back of the transceiver/controller). DirectInput mode produces no `/dev/input/js*` device on this system at all — silently, no error anywhere. |
| USB vendor:product (XInput) | `046d:c21f` |
| Deadman switch | LB (button index 4) — hold to enable manual driving |
| Speed axis | 1 (left stick, vertical) |
| Steering axis | 3 (right stick, horizontal) |
| RB (button index 5) | Bound to an `autonomous_control` profile in `joy_teleop.yaml` that currently does nothing functional — publishes to `/dev/null`. See [architecture.md](architecture.md#the-safety-model-read-this-before-writing-autonomy-code). |
| Group access | Joystick device nodes are group `input`; `racerbotcar-1` and `racermember1` are both members |

**Don't trust assumed Xbox-style axis numbering** — it was wrong here (upstream config had steering on axis 2, which is actually the left trigger on this pad in XInput mode, not the right stick). If you ever swap controllers, re-verify empirically:
```bash
ros2 topic echo /joy
```
watch which `axes[]` index moves as you work each stick/trigger, and which `buttons[]` index flips to `1` for each button, rather than assuming.

### Network

- `wlP1p1s0`: WiFi, normal DHCP, used for everything else (internet, SSH, etc.)
- `tailscale0`: Tailscale VPN interface, present for remote access if configured

## Car 2 (rb2)

Car 2's device, network, dashboard site and full-lock values are in [`rb2/car.yaml`](../src/racerbot_car/cars/rb2/car.yaml). Its parameter overlays live in [`rb2/calibration/`](../src/racerbot_car/cars/rb2/calibration/).

### Compute and access

Car 2's reference records a Jetson Orin Nano Super Developer Kit, JetPack 7.2, L4T R39.2, Ubuntu 24.04.4 (`noble`), aarch64 and 8 GB RAM. ROS 2 Jazzy runs natively.

Accounts are `racerbotcar-2` (administrator) and `racermember-2` (workspace-scoped access). The 2026-10-24 check found both in `dialout` and `input`.

### LiDAR: Hokuyo UST-10LX

Sources: [`car.yaml`](../src/racerbot_car/cars/rb2/car.yaml) and [`calibration/sensors.yaml`](../src/racerbot_car/cars/rb2/calibration/sensors.yaml). The shared `sensors.yaml` supplies the scan settings (about 40 Hz, ±3.14 rad).

| | |
|---|---|
| Connection | Ethernet (not USB) |
| Sensor's IP:port | `192.168.0.10:10940` (factory default, unchanged) |
| Jetson's wired NIC | `enP8p1s0` |
| NetworkManager profile | `hokuyo`, static IPv4 `192.168.0.15/24` |

**Terminal on car 2:** if the NetworkManager profile has not connected, run:

```bash
nmcli connection up hokuyo
ping 192.168.0.10   # should get replies once the LiDAR is powered and cabled in
```

**Working when:** ping receives replies. If a replacement Hokuyo has an unknown IP, `nmap -sn 192.168.0.0/24` with the `hokuyo` interface up lists reachable hosts on that subnet.

Car 2's other recorded interfaces are `wlP1p1s0` (WiFi with DHCP) and `tailscale0` (remote access when configured).

### Camera: RealSense D435i

[`car.yaml`](../src/racerbot_car/cars/rb2/car.yaml) selects RealSense USB `8086:0b3a`, image topic `/camera/camera/color/image_raw` and dashboard site `rb2`. `racerbot_launch/launch/realsense_camera_launch.py` starts the driver from `src/realsense-ros`.

`src/racerbot_launch/config/web_dashboard_rb2.yaml` selects the same image topic for the dashboard stream. See [RealSense camera](realsense-camera.md).

### VESC and steering

Car 2 uses USB `/dev/sensors/vesc` and firmware 7.0. Its 2026-10-24 on-car rule was `/etc/udev/rules.d/99-vesc.rules`, matching `0483:5740`, with `SYMLINK+="sensors/vesc"`, `GROUP="dialout"`, `MODE="0660"`. That installed rule differs from the repository's car 1 `MODE=0666` rule.

**Config** (`src/f1tenth_system/f1tenth_stack/config/vesc.yaml`):

The values below now come from [`rb2/calibration/vesc.yaml`](../src/racerbot_car/cars/rb2/calibration/vesc.yaml), layered over that shared file. This is a flattened excerpt, not a standalone ROS parameter file.
```yaml
speed_to_erpm_gain: 4614.0
speed_to_erpm_offset: 0.0
steering_angle_to_servo_gain: -0.78    # MEASURED 2026-10-16, see below
steering_angle_to_servo_offset: 0.5904
servo_min: 0.15
servo_max: 0.95
wheelbase: 0.36   # meters, MEASURED on this car 2026-08-24
```

The servo law is `servo_position = -0.78 × steering_angle + 0.5904`. At zero angle the servo is at `0.5904`, car 2's straight-ahead position.

**Car 2 steering calibration, 2026-10-16.** The old unmeasured gain `-1.2135` was about 1.5 times too large: a 19.5° command turned 29.2° left, while `servo_max: 0.85` limited the right side to 19.1°.

Raising the offset from 0.5304 to 0.5904 had corrected a left pull but used up right-hand travel. Half circles were tape-measured inner tire to inner tire. Rear-axle radius is `(inner diameter + 0.27 m track) / 2`; wheel angle is `atan(0.36 m / radius)`.

| Servo | Side | Inner diameter | Wheel angle |
|---|---|---|---|
| 0.178 | left | 1.02 m (twice) | 29.2° |
| 0.318 | left | 1.60 m | 21.1° |
| 0.85 | right | 1.81 m | 19.1° |
| 0.90 | right | 1.40 m | 23.3° |
| 0.95 | right | 1.23 m | 25.6° |
| 1.00 | right | 1.20 m | 26.1° |

The linkage is nonlinear. The `-0.78` fit covers 19° to 29°; angles below 19°, where autonomy mostly steers, were not measured. The right side gained only 0.5° from servo 0.95 to 1.00, within measurement error, so `servo_max` stops at 0.95. No buzzing, binding or heat was observed at 1.00.

The following full-stick limit comes from [`calibration/joy_teleop.yaml`](../src/racerbot_car/cars/rb2/calibration/joy_teleop.yaml). Controller limits come from [`gap_follow.yaml`](../src/racerbot_car/cars/rb2/calibration/gap_follow.yaml), [`pure_pursuit.yaml`](../src/racerbot_car/cars/rb2/calibration/pure_pursuit.yaml) and [`neural_racer.yaml`](../src/racerbot_car/cars/rb2/calibration/neural_racer.yaml). The planner limit remains shared in `src/pure_pursuit/config/auto_map_race.yaml`.

- Full stick (`joy_teleop.yaml` scale) is **0.436 rad = 25°**, which both sides reach. Check: both full-lock circles came out equal, 1.33 m inner = 24.2°.
- Autonomy is held below that: `max_steering_angle` **0.42 rad = 24°** in `gap_follow.yaml` and `pure_pursuit.yaml` (about 0.81 m minimum turning radius at the rear axle). Through the servo map that is 0.918 on the right and 0.263 on the left, which the 2026-10-16 circles put at 24.1° and 24.3°: inside the measured range, not extrapolated. It was 0.349 rad (20°) until 2026-11-07. At 0.349 the 2026-11-06 course's hairpin took full lock with about 0.1 m to spare, and the car wedged in it once (run `20261106-171030`). `neural_racer.yaml` stays at 0.349, since its policy never asks for more than 0.33. The race planner's `profile_max_steering_angle` is **0.30 rad = 17°** in `auto_map_race.yaml` (1.16 m), so the controller has steering left on the tightest planned corner.

Re-centring the horn, changing the servo or changing the linkage invalidates this calibration. Repeat the drive-straight run and circles in [Odometry calibration](odom-calibration.md#finding-how-far-the-steering-really-goes).

The negative odometry gain is in [`calibration/vesc.yaml`](../src/racerbot_car/cars/rb2/calibration/vesc.yaml). It cancels the driver's eRPM negation, as on car 1. **Never also negate the driver code**, or forward motion reads as reverse.

**Terminal on car 2, with bringup running:** check the effective gain. The verification line is retained from car 2's `docs/git-setup.md` at `9c2c2ed`:

```text
ros2 param get /vesc_to_odom_node speed_to_erpm_gain      # must be NEGATIVE (currently -4447.983786)
```

**Working when:** the value is `-4447.983786`.

The VESC firmware servo-output and IMU switches, described in car 1's section, apply to car 2 too. Stop bringup before connecting VESC Tool; the serial port can have only one owner.

### Joystick and IMU

Car 2 uses the same F710 XInput mapping as car 1: USB `046d:c21f`, LB button 4, speed axis 1 and steering axis 3. DirectInput produces no joystick device. Verify a replacement with `ros2 topic echo /joy`, watching each axis and button as it moves.

Car 2's BNO085 mount and calibration are in [`imu/car.yaml`](../src/racerbot_car/cars/rb2/imu/car.yaml) and [`imu/calib/`](../src/racerbot_car/cars/rb2/imu/calib/). Car 1's equivalents are under [`rb1/imu/`](../src/racerbot_car/cars/rb1/imu/). See [IMU differences](car1-vs-car2.md#imu) for the remaining car 2 measurements.
