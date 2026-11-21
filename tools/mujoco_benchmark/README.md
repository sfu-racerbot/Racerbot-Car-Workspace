# MuJoCo / MuSHR experiment

**An open-floor dynamics test, not a replacement for the racing simulator.** This runs the existing University of Washington [MuSHR one-car scene](https://github.com/prl-mushr/mushr_mujoco_ros/blob/c86eef33dadb0e98fe916a55ac6aa1fc03af027e/models/one_car.xml) in stock MuJoCo. No model assets, alternative physics, viewer, policies or bag data are shipped here. No training was performed.

## Install (development machine or car computer, drivers OFF)

From the workspace root on Linux, with the system Python for your ROS distribution:

```bash
git clone https://github.com/prl-mushr/mushr_mujoco_ros "$HOME/mushr_mujoco_ros"
git -C "$HOME/mushr_mujoco_ros" checkout c86eef33dadb0e98fe916a55ac6aa1fc03af027e
export MUSHR_MODEL="$HOME/mushr_mujoco_ros/models/one_car.xml"
/usr/bin/python3 -m venv --system-site-packages "$HOME/.local/mushr-bench"
source "$HOME/.local/mushr-bench/bin/activate"
python -m pip install mujoco==3.3.7
```

Use the same model path and interpreter in subsequent terminals. The checkout stays **outside** the ROS workspace: it is a ROS **1** package and should not be built by colcon. Linux x86-64 tested; Jetson/ARM and its graphics driver have not been tested.

## Headless or live

1. **Headless, no ROS or display:** `python tools/mujoco_benchmark/throughput.py "$MUSHR_MODEL"`. This steps actual wheel contacts, reads IMU-site forward velocity and yaw rate at every step, and repeats the same seeded actions after reset.
2. **Native MuJoCo desktop:** `python -m mujoco.viewer --mjcf="$MUSHR_MODEL"`. Open **Control**, move **buddy_throttle_velocity** then **buddy_steering_pos**; compare the car to the stationary floor grid. **Simulation → Pause / Run / Reset** controls physics. This is a live simulation, not playback.
3. **Native Lichtblick Desktop:** run the adapter and the stock read-only Foxglove bridge below. No MuJoCo GUI needed. The feed is a separate live simulation, not a mirror of a separately launched MuJoCo viewer.

In **each ROS terminal**, use an unused domain, not the car's graph:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=79 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
# Humble only: export ROS_LOCALHOST_ONLY=1
export PYTHONPATH="$PWD/src/racerbot_sim:${PYTHONPATH:-}"
export MUSHR_MODEL="$HOME/mushr_mujoco_ros/models/one_car.xml"
```

```bash
# Terminal 1: only advances/publishes once the workspace hardware interlock clears.
"$HOME/.local/mushr-bench/bin/python" tools/mujoco_benchmark/live_feed.py "$MUSHR_MODEL"

# Terminal 2: local read-only access; don't enable clientPublish or bind to a LAN address.
ros2 launch foxglove_bridge foxglove_bridge_launch.xml address:=127.0.0.1 port:=18765 \
  capabilities:="[connectionGraph]" topic_whitelist:="['^/mujoco/(odom|imu)$','^/tf$']"

# Terminal 3 (optional): command ONLY the simulator. Ctrl-C to stop sending.
ros2 topic pub -r 20 /mujoco/command ackermann_msgs/msg/AckermannDriveStamped \
  '{drive: {speed: 1.5, steering_angle: 0.15}}'
```

In Lichtblick Desktop choose **Open connection → Foxglove WebSocket** and enter `ws://127.0.0.1:18765`. Use a stock Plot panel with `/mujoco/odom.twist.twist.linear.x` (m/s); use the 3D panel with fixed frame `mujoco/world` for transforms. If development is remote, keep the bridge on loopback and forward its port with SSH (`ssh -L 18765:127.0.0.1:18765 USER@HOST`). Stock ROS packages needed: `foxglove_bridge`, `ackermann_msgs`, `nav_msgs`, `sensor_msgs`, `tf2_ros`, `rclpy`. Commands expire after 0.25 s. Never run with physical drivers or remap this feed onto hardware topics.

### Why a thin adapter?

The model's [own ROS connector](https://github.com/prl-mushr/mushr_mujoco_ros/blob/c86eef33dadb0e98fe916a55ac6aa1fc03af027e/src/mushr_ros_connector.cpp) is ROS 1 + MuJoCo 2.00. The stock [ros2_control MuJoCo plugin](https://github.com/ros-controls/mujoco_ros2_control/blob/e254f77d7864233624bb81a344d9e720e50a18fe/mujoco_ros2_control/src/mujoco_system_interface.cpp#L1400-L1419) requires a joint named like each tendon actuator. MuSHR's four-wheel velocity tendon has **no** joint named `buddy_throttle_velocity`; it also ships no ros2_control URDF. Rather than author a new vehicle description, the adapter passes speed (m/s) and center steering angle (rad) directly to the named stock actuators, as upstream does. It exports body pose/twist, native accelerometer/gyro and TF only. Wall-clock stamps for live viewing; benchmark timing comes from the physics clock.

**Mapped-course limitation:** the [upstream map launch](https://github.com/prl-mushr/mushr_mujoco_ros/blob/c86eef33dadb0e98fe916a55ac6aa1fc03af027e/launch/one_car.launch) publishes an occupancy map separately; the physics scene contains only a plane. The scanner mesh does **not** emit LaserScan. Loading Racerbot's SLAM map (e.g. rb2 `20261111-151125`) into a viewer would not add collisions, scan noise, 1,081 rays or alignment to the recorded start. No mapped navigation, closed-loop lap, camera policy, or LiDAR RL task was tested here. Use a mapped simulator for those. Stock wheelbase is 0.2965 m versus the cars' 0.36 m; no Racerbot-specific physical model is assumed.

## Measurements

- Host: Intel Xeon x86-64 VM, 1 car, 1 CPU rollout, 10 ms native step. Three 10,000-step headless runs after 500 warm-up steps, 0.512 s median: **19,524 observed steps/s (~195× real time)**. Includes Python action assignment, stepping, forward sensor update and reading two scalars. Excludes model load, action generation, camera/LiDAR, ROS, learning and rendering. Not policy-training throughput.
- Native reset + forward: **34.5 µs** mean over 1,000 resets. Repeated seed 17 gave bit-identical sensor traces on this machine; seed 18 changed the trace. The seed controls the generated actions, **not** randomized dynamics. No sensor noise or domain randomization is supplied by this model.
- Before comparing a different plant, replayed the [published rb1 speed-loop finding](../run_analysis/results/2026-11-14-rb1-speed-loop.md) from the original Nov 11 155542 and 153721 MCAPs. The existing `fit_speed_loop.load_grids` and `scores` functions gave **0.36115 → 0.22228 m/s wheel-speed RMSE** over 11 holds, 39,705 samples. This reproduces the workspace's scalar speed-loop study, **not a MuJoCo score**. Wheel odometry is not independent ground truth.

Integration check: from the root, after the ROS environment above, install pytest in the virtual environment if needed and run `python -m pytest -q tools/mujoco_benchmark/test_live_feed.py` **without** another feed or live hardware on domain 79. Exercises suppression beside a driver, motion from Ackermann inputs, stale input stop and actual ROS sensor/TF outputs.
