# Car 1 vs car 2

> **Who this is for:** anyone choosing hardware settings for car 1 (`rb1`) or car 2 (`rb2`).
> **Read first:** [Car profiles](car-profiles.md) for identity and configuration layers.
> **You'll be able to:** find each car's hardware values and recognise measurements that still need checking.
> **Time:** about 5 minutes.

## The differences

Each cell names the hardware value and links to its profile source. The shared geometry and the steering measurements are in [Hardware reference](hardware-reference.md#physical-dimensions-used-in-config).

| Car 1 (`rb1`) | Car 2 (`rb2`) |
|---|---|
| **LiDAR:** Hokuyo UST-30LX, USB `15d1:0000`, `/dev/sensors/hokuyo`. [Device profile](../src/racerbot_car/cars/rb1/car.yaml), [sensor overlay](../src/racerbot_car/cars/rb1/calibration/sensors.yaml). | **LiDAR:** Hokuyo UST-10LX, Ethernet `192.168.0.10:10940`, `enP8p1s0`, host `192.168.0.15/24`. [Device profile](../src/racerbot_car/cars/rb2/car.yaml), [sensor overlay](../src/racerbot_car/cars/rb2/calibration/sensors.yaml). |
| **Camera:** UVC webcam, `/dev/video0`. [Device profile](../src/racerbot_car/cars/rb1/car.yaml). | **Camera:** RealSense D435i, USB `8086:0b3a`, image topic `/camera/camera/color/image_raw` (a named ROS message channel, see the [glossary](glossary.md)). [Device profile](../src/racerbot_car/cars/rb2/car.yaml). |
| **Servo:** position `= -0.83 × angle + 0.366`, clamps `0.05 … 0.75`; angle in radians, position unitless. [VESC overlay](../src/racerbot_car/cars/rb1/calibration/vesc.yaml). | **Servo:** position `= -0.78 × angle + 0.5904`, clamps `0.15 … 0.95`. [VESC overlay](../src/racerbot_car/cars/rb2/calibration/vesc.yaml). |
| **Speed gains:** command `4600.0` (was `3650.0` until 2026-11-02, which delivered 0.79 of every commanded speed; [measurement](hardware-reference.md#vesc-motor--steering-controller)), odometry `-4040.0` electrical rpm per m/s. [VESC overlay](../src/racerbot_car/cars/rb1/calibration/vesc.yaml). | **Speed gains:** command `4614.0`, odometry `-4447.983786` electrical rpm per m/s. [VESC overlay](../src/racerbot_car/cars/rb2/calibration/vesc.yaml). |
| **Steering limits:** full stick `0.38 rad` ([manual overlay](../src/racerbot_car/cars/rb1/calibration/joy_teleop.yaml)); autonomy `0.33 rad` ([gap follow](../src/racerbot_car/cars/rb1/calibration/gap_follow.yaml), [pure pursuit](../src/racerbot_car/cars/rb1/calibration/pure_pursuit.yaml), [neural racer](../src/racerbot_car/cars/rb1/calibration/neural_racer.yaml)). | **Steering limits:** full stick `0.436 rad` ([manual overlay](../src/racerbot_car/cars/rb2/calibration/joy_teleop.yaml)); autonomy `0.42 rad` ([gap follow](../src/racerbot_car/cars/rb2/calibration/gap_follow.yaml), [pure pursuit](../src/racerbot_car/cars/rb2/calibration/pure_pursuit.yaml)) since 2026-11-07, `0.349 rad` for the [neural racer](../src/racerbot_car/cars/rb2/calibration/neural_racer.yaml). |
| **Full lock:** left `0.401 rad` measured; right `-0.46265060240963857 rad` computed from the servo map, not measured. [Device profile and provenance](../src/racerbot_car/cars/rb1/car.yaml). | **Full lock:** left `0.5090708884223831 rad`, right `-0.44751997515716985 rad`, from 2026-10-16 tape circles. [Device profile and provenance](../src/racerbot_car/cars/rb2/car.yaml). |
| **BNO085 mount:** rotation `[92.44, 5.22, 90.11]°`, refined at rest 2026-10-24; yaw still needs a forward-push check. [IMU profile](../src/racerbot_car/cars/rb1/imu/car.yaml). | **BNO085 mount:** rotation `[90.80, 0.24, 0.24]°`, parked 2026-10-29; yaw and position are provisional. [IMU profile](../src/racerbot_car/cars/rb2/imu/car.yaml). |
| **Dashboard site:** `rb1`; remote hostnames `rd-rb1` / `ssh-rb1`. [Device profile](../src/racerbot_car/cars/rb1/car.yaml). | **Dashboard site:** `rb2`. [Device profile](../src/racerbot_car/cars/rb2/car.yaml). |

## These used to be ported

The two repositories used to receive changes ported by hand. Both cars now run this workspace: each hardware difference lives in its car's profile. See [Car profiles](car-profiles.md) for selecting a car and changing its values.

## IMU

An IMU (inertial measurement unit) measures rotation and acceleration. Both cars have the VESC's IMU and a BNO085 on the Jetson's I2C bus 7 at `0x4A`, rather than a USB microcontroller.

`src/imu_bridge/launch/vehicle_state_launch.py` starts `tools/imu_calibration/tools/bno_i2c_bridge.py` with `--pty /tmp/bno085 --no-dyncal`. The bridge turns I2C samples into the serial stream the ROS node (running program) reads. Only one bridge may own the sensor; an existing bridge is reused.

The node restarts its BNO driver after 2 s without samples, because the virtual port can appear after startup. It reads mount rotations from the selected car's `imu/car.yaml`, rather than the old `~/imu-calibration/config/car.yaml`.

**Keep the car still for the first 3 s**, while the startup gyro bias is measured. Moving it then contaminates the rotation estimate.

The shared defaults now apply to both cars (`map_course_launch.py`, `gap_follow_launch.py` and `vehicle_state.yaml`):

- `map_course` has `imu:=true`. `vehicle_state_launch.py` has `bno:=true`.
- `steering_damping` follows `imu` (default true), and `enable_steering_damping` is true.
- `gyro_odom` follows `imu`; `imu_lap_turns` follows `gyro_odom`. The mapper uses the gyro SLAM profile (`slam_gyro.yaml`) and its lap counter uses the same odometry (motion estimate). `imu:=false` turns those defaults off. SLAM builds a map while estimating position; see the [glossary](glossary.md).
- Gyro averaging for measured steering uses BNO and VESC weights of `1.0` each. Heading prefers BNO, then VESC; sustained disagreement falls back to the steering model until agreement returns.
- `/odom/fused` is published and recorded, but `gap_follow` uses it only with `fused_speed:=true`. Its gate requires `/imu/status` to be `ok` and fresh `/imu/data`, otherwise it falls back to `/odom`.
- `imu_all` and `fused_speed` default false in mapping and gap-follow launches. Gap follow's understeer, traction and impact-stop features also default off.

Car 2's **2026-10-31** calibration status remains provisional in [`rb2/imu/car.yaml`](../src/racerbot_car/cars/rb2/imu/car.yaml). Parked gyro bias and mount roll/pitch were measured on 2026-10-29; mount yaw, accelerometer fit (gravity was 1.7% low), chip position and synchronisation remain to be measured.

**Measure car 2's forward mount direction before using `fused_speed:=true`.** A placeholder yaw can swap forward and sideways acceleration. Use the mount forward-push procedure and 12-pose accelerometer fit in the [IMU tools guide](../tools/imu_calibration/README.md), then measure the chip position and update the lever arm too.

Car 1 also has unfinished measurements: its [`imu/car.yaml`](../src/racerbot_car/cars/rb1/imu/car.yaml) labels the VESC mount and calibration as another car's, the BNO height as unmeasured, and BNO yaw as needing a forward-push check.

## gap_follow course settings

Car 2's **2026-10-30** course settings are now shared in `src/gap_follow/config/gap_follow.yaml`: `swept_clearance_max_length: 5.0` m, `aim_depth_bias: 0.0`, `aim_depth_bias_fallback: 1.0` and `opponent_follow_floor: 0.5` m/s.

`time_trial` defaults false in `src/gap_follow/launch/gap_follow_launch.py`. The profile guard preserves these settings. See [the course measurements](gap-follow.md#speed-on-the-real-course-2026-10-30) for their reasons.

## IMU feature thresholds

Car 1's measured thresholds remain shared placeholders for car 2, labelled `CAR 1'S VALUES` in `src/gap_follow/config/gap_follow.yaml`. **Keep the features off on car 2 until car 2 records its own values.** Change the value and label together; `test_car_1_imu_values_stay_labelled_as_car_1s` guards that provenance.

- **Impact:** `imu_impact_threshold: 33.7257297468959` m/s², 1.5 times the 22.48 m/s² largest horizontal spike in car 1 run `20261029-132902`. Also in `IMPACT_DEFAULTS` in `gap_follow_node.py`. Record a wheels-off bump with car 2's BNO running before enabling [impact stop](gap-follow.md#impact-stop-imu).
- **Traction:** slip/release thresholds `0.246` / `0.123` m/s in the YAML and `TRACTION_DEFAULTS`, from 1.5 times the combined p99 of `/odom` minus `/odom/fused` in car 1 runs `20261026-194514` and `20261026-194913`. Repeat the percentile sweep on car 2 runs with `imu:=true`.
- **Understeer:** engage/release ratios `0.75` / `0.85` and steering lag `0.15 s` in the YAML and `UNDERSTEER_DEFAULTS`. Car 1's same two runs gave ratios of 0.84 to 0.97. Use `tools/run_analysis/grip_replay.py` and `yaw_tracking.py` on car 2 recordings to obtain its values.
- **Simulated noise:** `BNO_NOISE` / `VESC_NOISE` in `src/racerbot_sim/racerbot_sim/sim_imu.py` came from a stationary stretch of car 1 run `20261026-194514`. Measure gyro and accelerometer standard deviations from a stationary car 2 bag.
- **Simulated positions:** `BNO_X_OFFSET` / `VESC_X_OFFSET` in `sim_imu.py` and the corresponding `vehicle_state.yaml` offsets are `0.36` / `0.18 m` ahead of the rear axle, from car 1's mounts. Tape-measure car 2's mounts.
- **Mapping speed:** `mapping_max_speed` defaults to `3.5 m/s` in `map_course_launch.py`, from car 1's gyro SLAM profile replay-validated to about `3.3 m/s`. A faster car 2 mapping recording needs a clean `slam_report` before raising it.

The shared `auto_map_race.yaml` lap-time floor is `minimum_lap_duration_sec: 4.0`; its comment derives the bound from car 2's 0.99 m minimum turning radius (car 1's is 1.05 m).
