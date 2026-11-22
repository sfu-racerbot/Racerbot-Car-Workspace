# Watch a live Gym simulation in Lichtblick Desktop

> **Who this is for:** anyone who wants to watch the simulator drive, on a laptop or the car computer, without touching a real car.
> **Read first:** [ros-simulator.md](../../../docs/ros-simulator.md) for what the simulator is and its hardware interlock, and [glossary.md](../../../docs/glossary.md) for words like *topic* and *bridge*.
> **You'll be able to:** run the two-car Gym race headless and watch it live in Lichtblick Desktop.
> **Time:** about 30 minutes the first time, then two terminals and one connection.

**Highlights**
- Stock parts only: Nav2 `map_server`, `foxglove_bridge` and four built-in Lichtblick panels. No extension.
- Read-only: the bridge listens on 127.0.0.1:8766 only, and clients can publish no topic, set no parameter and call no service (`check_feed.py` and `tools/racerbot_sim/tests/test_lichtblick_view.py` check this).
- The simulator runs the same way with or without the viewer, so closing Lichtblick never stops a run.
- Limits: tested on one Linux x86-64 host with Humble; Jetson/ARM64, Jazzy, macOS, Windows and a second physical device are untested.

**What you see:** stock F1TENTH Gym with club geometry, running the existing two-controller Racerbot race. Two small moving coordinate axes mark the cars' ground-truth poses on the actual simulated track grid. They are not individually color-coded. The separate laser view is in car 1’s laser frame; the plot shows its simulated wheel odometry and mux command (m/s). In /sim/status, sim_time_s counts physics steps in seconds. Message arrival times and ROS header stamps are not that clock. No /clock, SLAM-derived transform, calibrated-headless-model result or real-car telemetry is implied.

## One-time setup

These files are ordinary ROS launch configuration and a stock Lichtblick layout. There is no Lichtblick extension, custom bridge, recorded bag or simulator model here.

The simulator host needs ROS 2, the usual workspace build dependencies, and the pinned Gym from bash tools/f1tenth_sim/setup.sh. From the workspace root, in Bash:

```bash
source /opt/ros/humble/setup.bash  # tested Ubuntu 22.04 x86-64; use your installed ROS distro
sudo apt install ros-$ROS_DISTRO-foxglove-bridge ros-$ROS_DISTRO-nav2-map-server ros-$ROS_DISTRO-nav2-lifecycle-manager
bash tools/f1tenth_sim/setup.sh
colcon build --symlink-install --base-paths \
  src/racerbot_car src/drive_intent src/imu_bridge src/gap_follow \
  src/f1tenth_system/ackermann_mux src/f1tenth_system/f1tenth_stack \
  src/racerbot_launch src/race_diagnostics src/racerbot_sim --cmake-args -DBUILD_TESTING=OFF
```

Use [Lichtblick Desktop](https://github.com/lichtblick-suite/lichtblick/releases/tag/v1.31.0) on the viewing machine; choose its OS/architecture. The simulator can run on a different machine. Nothing here installs a service or requires an account. For an existing fully built workspace, skip the build.

## 1 · Simulator host, terminal A: headless

From the workspace root on an **idle simulator host**, with no real drivers running:

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
export ROS_DOMAIN_ID=79 ROS_LOCALHOST_ONLY=1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/tools/racerbot_sim/lichtblick/loopback-udp.xml"
export RACERBOT_CAR=rb1 PYTHONPATH="$PWD/.sim/python:$PYTHONPATH"
ros2 launch racerbot_sim sim_race_launch.py record:=false drive_topic_prefix:=/sim_only track:=indoor_oval timeout_s:=1800.0 laps:=1000.0
```

Reserve domain 79 for this test. Do not use a custom DDS discovery server, explicit remote peers, or a physical-driver graph. Keep the prefix and existing hardware interlocks; don't remap the sim commands back to /drive. The two controllers use the simulated joystick alone in this isolated graph. The simulator runs without Lichtblick and stops on its own race termination condition. If it finishes before you connect, relaunch it. Ctrl+C stops it.

## 2 · Simulator host, terminal B: optional read-only bridge

Wait until Gym's generated indoor_oval map exists, then, from the same checkout:

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/tools/racerbot_sim/lichtblick/loopback-udp.xml"
ros2 launch tools/racerbot_sim/lichtblick/sim_view.launch.xml \
  map_file:="$HOME/.ros/racerbot_sim/tracks/indoor_oval_map.yaml"
```

This launches stock Nav2 map_server for that geometry, its lifecycle manager, and foxglove_bridge. The bridge joins the local simulator domain and listens on **127.0.0.1:8766** only. The only optional bridge capability is connectionGraph: clients cannot publish topics, set node parameters, or call ROS services. Closing the viewer does not stop the simulator or publish a stop command. Stop the optional bridge with Ctrl+C.

## 3 · Viewing device: native desktop

In Lichtblick Desktop, open a **Foxglove WebSocket** connection to **ws://127.0.0.1:8766**. In Layouts, choose **Import from file** and select **tools/racerbot_sim/lichtblick/sim-layout.json**. Use the desktop app, not a browser or a bag file.

On the tested x86-64 VM, GPU acceleration was unavailable. The native v1.31.0 archive rendered the stock 3D panels with the following command from its extracted folder. On a machine with a working graphics driver, use the normal desktop launcher first. Only connect to the trusted local simulator feed.

```bash
DISPLAY=:0 ./lichtblick --use-gl=angle --use-angle=swiftshader \
  --enable-unsafe-swiftshader --ignore-gpu-blocklist
```

If the track appears but every dynamic panel stays blank after a simulator restart, stop both simulation and bridge, start A then B with the loopback-only Fast DDS profile in **both** terminals, and reopen the desktop connection. Do not open the bridge to the LAN as a workaround. The two-terminal profile above is the configuration that passed the native test.

**Separate simulator host (including an optional Jetson):** copy just the layout to your laptop. Using your normal SSH credentials on the laptop, keep this terminal open, then use the same loopback URL in Lichtblick:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8766:127.0.0.1:8766 user@sim-host
```

No LAN bind, firewall change, remote ROS discovery, or hardware bridge is needed. Forward a different *local* port if 8766 is occupied, and use that local port in the desktop client.

**Verification scope:** Linux x86-64, Ubuntu 22.04, Humble; `python3 tools/racerbot_sim/lichtblick/check_feed.py ~/.ros/racerbot_sim/tracks/indoor_oval_map.png` (ROS sourced, Python `websockets` installed) re-checks the live feed: it must print the PASS line. The SSH forward was exercised on one machine, not across two physical hosts. Jetson/ARM64, Jazzy, macOS and Windows were not run. On a Jetson the profile must match that machine and the normal hardware interlock must remain in force; never start real drivers or drive the car to try this. No Gazebo or calibrated headless model is part of this test.

## Other producers: compatibility notes, not confirmed with this layout

| Producer | Reported feed | What still needs a desktop test |
|---|---|---|
| [Gazebo testbed](https://github.com/hippopond/f1tenth_testbed/tree/2c8c37ba96e2da21dd5373c8e49104b87111ba19) | Stock ROS–Gazebo bridge; /clock, /odom, /scan (1,080 beams), odom → laser_link TF | Its own track/map, layout frames and native rendering |
| [AutoDRIVE 2026 IROS](https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing/releases/tag/2026-iros) | Engine first, then Humble devkit: /tf and /autodrive/roboracer_1/{lidar,odom,imu}; frame world | Native rendering and track/map; no working Unity view was reported |

Those feeds have different frame names and clocks. Do not reuse the Gym map for either engine.
