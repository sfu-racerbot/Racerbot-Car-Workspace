# Gazebo Harmonic: November experiment

**Use this for ROS integration and inspecting sensors, not as a validated replacement for the Racerbot plant.**
The car, tires, drive system, LiDAR, renderer, ROS bridge and controller below are upstream software.
The benchmark tools assemble assets, feed actions and read state from the actual Gazebo server. No network training,
model weight changes, hardware launches or new run data.

## Install once (Ubuntu 24.04 / ROS 2 Jazzy)

Keep the experiment in a separate workspace. Do not install anything in a running car's driver stack.
These examples assume Bash and that the Racerbot checkout is the current directory.

```bash
export RB="$PWD" GZLAB="$HOME/gazebo-racerbot"
sudo apt install ros-jazzy-ros-gz ros-jazzy-xacro ros-jazzy-robot-state-publisher \
  python3-colcon-common-extensions python3-venv
mkdir -p "$GZLAB/ws/src" "$GZLAB/out"
git clone https://github.com/hippopond/f1tenth_testbed "$GZLAB/ws/src/f1tenth_testbed"
git -C "$GZLAB/ws/src/f1tenth_testbed" checkout 2c8c37ba96e2da21dd5373c8e49104b87111ba19
source /opt/ros/jazzy/setup.bash
(cd "$GZLAB/ws" && colcon build --packages-select f1tenth_testbed)
```

**Every terminal:** stop hardware bringup first. Use a dedicated, otherwise unused ROS domain, local
discovery, and a dedicated Gazebo partition. Do not replay bags into ROS. None of these commands should
publish on the car's drive topics. CPU-only x86-64 tested; not a Jetson performance claim.

```bash
export RB=/absolute/path/to/Racerbot-Car-Workspace GZLAB="$HOME/gazebo-racerbot"
export ROS_DOMAIN_ID=83 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export GZ_PARTITION=racerbot_gazebo_experiment GZ_IP=127.0.0.1
source /opt/ros/jazzy/setup.bash
source "$GZLAB/ws/install/setup.bash"
PKG=$(ros2 pkg prefix --share f1tenth_testbed)
OUT="$GZLAB/out"
```

## See the live car

**Upstream test track:** run `ros2 launch f1tenth_testbed sim.launch.py`. This starts the native Gazebo GUI,
car, robot description and bridge. Press Play if paused. For the missing odometry transform, in a second
terminal run `ros2 run ros_gz_bridge parameter_bridge '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'`.
The upstream bridge subscribes to an unused Gazebo topic instead of the model's actual `/tf`.
For a simulation-only controller, run `ros2 run f1tenth_testbed baseline_racer_node --ros-args -p speed:=1.0`
in the **isolated** environment. **Stop the simulator when stopping the controller:** the stock model holds
its last command; killing a controller is not a brake.

**Lichtblick Desktop:** use the common read-only live ROS/Foxglove bridge on the isolated simulation domain.
Select fixed frame `odom`, `/scan` in 3D, and plot `/odom.twist.twist.linear.x`. Check connectivity with
`ros2 run tf2_ros tf2_echo odom laser_link`. Live ROS feed verified; the native Gazebo GUI and Lichtblick
are different clients. Don't expose a writable bridge to the car. No HTML replayer or custom client.

## Use the race-day SLAM map, with or without a GUI

The map race (rb2 20261111-152132) used the line rebuilt from mapping run 151026.
Use **that same rebuild**, not the live 151125 package. The external converter makes wall contours, not a
surveyed 3D track: it assumes a zero-yaw map origin, skips image-boundary contours and invents 2 m wall
heights. No moving cars. Do not use wall contact here as evidence of real collision clearance.

```bash
git clone https://github.com/Adlink-ROS/map2gazebo "$GZLAB/map2gazebo"
git -C "$GZLAB/map2gazebo" checkout d9d85b91b9c1dd82d270466ebfb152fdc385d911
python3 -m venv "$GZLAB/converter"
"$GZLAB/converter/bin/pip" install 'numpy<2' trimesh==3.23.5 opencv-python-headless==4.10.0.84 \
  matplotlib pyyaml scipy networkx
MAP="$RB/data/rb2/racerbot_auto/20261111-151026-rebuilt"
mkdir -p "$OUT/models/map/meshes"
cp "$GZLAB/map2gazebo"/models/map/model.{sdf,config} "$OUT/models/map/"
"$GZLAB/converter/bin/python" "$GZLAB/map2gazebo/map2gazebo/map2gazebo_offline.py" \
  --map_dir "$MAP/map.pgm" --export_dir "$OUT/models/map/meshes"
python3 "$RB/tools/run_analysis/gazebo/prepare_world.py" "$OUT/november.sdf" --map-start "$MAP/start_pose.yaml"
export GZ_SIM_RESOURCE_PATH="$OUT/models:${GZ_SIM_RESOURCE_PATH:-}"
gz sim -r -s --headless-rendering --seed 13 "$OUT/november.sdf"
```

For a window instead, **stop** the server and replace the last command with
`gz sim -r --seed 13 "$OUT/november.sdf"`. On a host without a GPU the fallback is
`LIBGL_ALWAYS_SOFTWARE=1`. A GUI in a container also needs the host display and its X authorization;
do not disable access control with `xhost +`. Headless rendering needs EGL/Mesa or a compatible GPU
driver, **not** a display server.

In fresh isolated terminals, run these stock processes to get the same ROS interface on this map:

```bash
sed 's@/model/f1tenth_testbed/tf@/tf@' "$PKG/config/bridge.yaml" > "$OUT/bridge.yaml"
ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:="$OUT/bridge.yaml"
# Another terminal: sensor and joint frames + latched description for visualization
xacro "$PKG/urdf/f1tenth_testbed.urdf.xacro" > "$OUT/testbed.urdf"
ros2 run robot_state_publisher robot_state_publisher "$OUT/testbed.urdf" --ros-args -p use_sim_time:=true
# Another terminal: simulation-only upstream controller (or connect a read-only viewer)
ros2 run f1tenth_testbed baseline_racer_node --ros-args -p speed:=1.0
```

## Replay actions without ROS

Create a fresh floor world (no track contacts) then pass a CSV headed `speed_mps,steer_rad` with one
action per 25 ms. Each process starts at rest with a fresh Gazebo server and private Gazebo transport
partition, settles for 250 ms, then steps 25 physics ticks per action. Output includes t=0 and one end
sample per action. Compare the recorded wheel speed to **wheel_mps**, not chassis ground speed.

```bash
python3 "$RB/tools/run_analysis/gazebo/prepare_world.py" "$OUT/open.sdf"
python3 "$RB/tools/run_analysis/gazebo/replay.py" "$OUT/open.sdf" actions.csv "$OUT/trace.csv" --seed 13
```

The stock plugin accepts a Twist, not AckermannDriveStamped. This offline adapter maps a **requested tire
angle** to yaw-rate input with `v*tan(angle)/plugin_wheelbase`. It preserves the upstream plugin response,
including its zero-speed limitation. The model has 0.32 m between axles but omits `wheel_base` and
`kingpin_width`; Harmonic therefore computes steering and odometry with 1.0 m and 0.8 m respectively.
Do not treat stock yaw odometry as ground truth. The trace includes Gazebo chassis pose and yaw rate
separately. No VESC/servo/IMU model, measured steering feedback, delay/noise fit or Racerbot deadman/mux
is supplied by this model. Its instantaneous 1,080-beam ±2.35 rad, 10 m GPU LiDAR is **not** the
recordings' 1,081-beam physical scanner. A policy expecting the latter cannot just be loaded unchanged.

## Replay one shared November turning/braking block

Unlike the rest-start diagnostic above, this runner requires the frozen block's initial wheel-speed
proxy in m/s and exactly 480 mux commands. It writes one **post-step** body-forward speed and body yaw
rate per command, including the 80 warm-up ticks. Each block gets a new seed-0 server. Use a fresh
output filename for each block and retain its car, run and integer nanosecond block start when joining
the output to the independent masks. Never use Gazebo wheel speed or odometry yaw in place of a missing
physical reference.

In the isolated shell above, build the two missing native bindings (requires a compiler, CMake,
pybind11-dev and Jazzy's gz-sim development headers). Generate the open floor with
`prepare_world.py` as above; do not supply a map.

```bash
GZBENCH="$RB/tools/run_analysis/gazebo"
export CMAKE_PREFIX_PATH="$(find /opt/ros/jazzy/opt -mindepth 1 -maxdepth 1 | paste -sd :):$CMAKE_PREFIX_PATH"
cmake -S "$GZBENCH/moving_reset" -B "$OUT/bridge-build"
cmake --build "$OUT/bridge-build" -j2
export PYTHONPATH="$OUT/bridge-build:$PYTHONPATH"
# ACTIONS: frozen speed_mps,steer_rad CSV; V0: initial speed at block start
python3 "$GZBENCH/rollout.py" "$OUT/open.sdf" "$ACTIONS" "$V0" "$OUT/block-trace.csv"
```

The runner settles the vehicle, primes the stock actuator at the given speed, then sets pose, body
velocity and joint state with physics paused. It checks readback before the first action and releases
the one-shot velocity commands. Synthetic checks passed at 0, −1 and +2.5 m/s; three fresh positive-start
traces were byte-identical on this build. This is reset and repeatability evidence, not a November score.

For a separate input-clock audit, run `python3 "$GZBENCH/command_coverage.py" "$RB/data"` in the
analysis environment with NumPy and MCAP installed. Of 428 complete blocks, 415 passed the 100 ms command
age check. Missing initial wheel speed can reject additional blocks; this count does not measure
turning/braking coverage and does not use simulator outputs.

## Measurements (Gazebo 8.15.0, DART, Jazzy, x86-64 VM, 8 vCPUs, CPU rendering)

| Test | Result | Interpretation |
|---|---|---|
| Published rb1 Nov 11 speed-loop check | Shared/chosen wheel-speed RMS **0.36115 / 0.22228 m/s**, 39,705 moving samples | Re-read 155542 and 153721 MCAPs with the repository's `fit_speed_loop.load_grids` and `scores`; reproduced [the study](../results/2026-11-14-rb1-speed-loop.md). These are not Gazebo scores. |
| Gazebo development-only launch transient, rb1 155542 | Wheel-speed RMS **1.115 m/s**, mean signed error **+0.719 m/s**, 200 samples | First 1 s after **every** LB press lasting at least 1 s (5 holds, including one stationary). Rest at all starts. At 40 Hz, zero-order hold the last mux speed and steering at/before each tick; compare end-of-tick wheel speed with linearly interpolated bag-receive-time odometry. Per-hold RMS 2.154 / 0.758 / 0.745 / 0.000 / 0.672 m/s. No fitting, lag correction, map or controller feedback. Not the study's rolling-horizon score. |
| Native viewer | Car drove, stopped and paused on upstream test track; 0.64–0.72 simulated seconds per wall second | Live, software-rendered; not playback, SLAM-map performance or a held-out score. |
| Mapped-course algorithm smoke test | 8.15 s simulated in 10.18 s wall, 327 scans (**32.0 received scans/s**); 5.49 m Gazebo world-pose travel | Rebuilt rb2 151026 map + saved start pose. Stock testbed controller at 1 m/s; smoke interval includes controller startup/shutdown. All 1,080 ranges finite on all sampled scans. No lap, clearance or policy score. |
| Headless state-only stepping | 76–121 control steps/s; 2.45–2.51 s server creation, settling and discovery | Three 120-step runs on the open floor. No scan subscriber, GUI, ROS bridge, policy, optimizer or parallel workers. Timing excludes module imports and process launch; **not** sensor-training throughput. |
| Reset / seed | 2 s at 1 m/s, +0.1 rad, then 1 s stopped: byte-identical traces across two fresh seed-13 servers and one seed-14 server | Deterministic for this pulse on this build; changing the seed is **not** domain randomization. Native world reset on the map also returned the car to its saved coordinates and restored the static scan. |

The mapped test surfaced a shutdown issue in the upstream controller: SIGINT called ROS shutdown twice.
It does **not** send a stop before exiting. End the simulation or reset before another test. Keep the
standard bridge and GUI for algorithm debugging; neither this short run nor the development transient
validates using the uncalibrated model for training or deployment.
