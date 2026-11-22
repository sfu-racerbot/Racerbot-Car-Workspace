# AutoDRIVE practice engine: headless or live in Lichtblick

> **Who this is for:** someone testing simulated range scans or a racing algorithm on a development machine.  
> **Read first:** [ROS concepts](../../../docs/concepts.md) — a *topic* carries messages between programs.  
> **You'll be able to:** launch the stock RoboRacer, inspect its live scans and repeat the November sensor comparison.  
> **Time:** about 10 minutes after downloading the engine and container.

**Verdict:** use this engine for live laser-scan (LiDAR) checks, not for scoring Racerbot turning or braking. See the [benchmark limits](#locked-november-turning-and-braking-test).

Run the simulator on an x86-64 Linux development host. The tested build is **not** a native ARM64/Jetson binary. A different device can view the feed; it need not run the engine.

## What worked

- Stock [IROS 2026 practice engine and devkit](https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing/releases/tag/2026-iros), without a GPU, display server or modified bridge.
- 1,081 actual ranges per scan, with live motion and soft reset.
- Stock Lichtblick Desktop connected to the live ROS feed through the stock Foxglove bridge. No replay file, HTML player, plugin or custom panel.
- No new physics, controller, vehicle model, map conversion or model training.

## Launch on a development host

Requires Docker, curl, unzip and Linux x86-64. No ROS installation on the host is needed. **Keep Docker's private network and IPC namespaces. Do not use host networking, pass through devices, connect to a car's ROS graph, or start hardware bringup.** Only the host's loopback ports are published.

**Terminal 1 — install and start the engine first.** Keep it running. The checksum is for this release's Linux practice archive, not the competition scene.

```bash
mkdir -p "$HOME/.cache/racerbot/autodrive-2026-iros"
cd "$HOME/.cache/racerbot/autodrive-2026-iros"
curl -fL -o practice.zip https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing/releases/download/2026-iros/autodrive_simulator_practice_linux.zip
echo '4ffa685fb83a48c3b9f2d2bb5f093aa9e6367f4eb98e2aa1365f57315ebd82f9  practice.zip' | sha256sum -c - &&
  unzip -o practice.zip && chmod +x "autodrive_simulator/AutoDRIVE Simulator.x86_64" &&
  env -u DISPLAY "./autodrive_simulator/AutoDRIVE Simulator.x86_64" \
    -batchmode -nographics -ip 127.0.0.1 -port 4575 -logFile "$PWD/practice.log"
```

**Terminal 2 — start the stock ROS 2 Humble bridge.** Let the engine initialize before starting this container. Connection-refused messages in the engine log before the bridge starts are expected.

```bash
docker run --rm --name racerbot-autodrive \
  -p 127.0.0.1:4575:4567 -p 127.0.0.1:8785:8765 \
  -e ROS_DOMAIN_ID=79 -e ROS_LOCALHOST_ONLY=1 --entrypoint /bin/bash \
  autodriveecosystem/autodrive_roboracer_api@sha256:6e4c29536b7283a1a7322473d46dab2d4ad5513cf40d1bdfc908ee9d408e26e9 \
  -c 'source /opt/ros/humble/setup.bash && source /home/autodrive_devkit/install/setup.bash && exec ros2 launch autodrive_roboracer bringup_headless.launch.py'
```

**Terminal 3 — check for actual data.** A [topic](../../../docs/glossary.md) is a named message channel. Seeing its name does not prove the engine is sending messages.

```bash
docker exec racerbot-autodrive bash -c 'source /opt/ros/humble/setup.bash && timeout 10 ros2 topic echo --once /autodrive/roboracer_1/lidar --field header'
```

**Working when:** you see a timestamp and `frame_id: lidar`. If the bridge reports `KeyError: V1 LIDAR Range Array` on connection, keep the simulation running and stop/relaunch **only Terminal 2**. We observed the devkit wait forever after a first packet that lacked ranges; once the engine had initialized the stock bridge worked.

To drive **only the simulated practice car**, use the stock keyboard control (teleop) below. Press lowercase `w` to increase throttle, `x` to zero both commands, and `r` to reset. Do not run this on a real car graph.

```bash
docker exec -it racerbot-autodrive bash -c 'source /opt/ros/humble/setup.bash && source /home/autodrive_devkit/install/setup.bash && ros2 run autodrive_roboracer teleop_keyboard'
```

**Shutdown:** zero the simulated commands with X, exit teleop, stop the engine in Terminal 1 (or terminate that process), then `docker stop racerbot-autodrive`. No device access or hardware configuration changes are needed.

## Watch it live, without enabling remote driving

For headless-only work, skip this section. To view it, install the standard bridge **inside that same devkit container**, not on the physical car network. In a separate terminal:

```bash
docker exec racerbot-autodrive bash -c 'apt-get update && apt-get install -y --no-install-recommends ros-humble-foxglove-bridge'
docker exec racerbot-autodrive bash -c 'source /opt/ros/humble/setup.bash && exec ros2 run foxglove_bridge foxglove_bridge --ros-args -p "capabilities:=[connectionGraph]"'
```

In [Lichtblick Desktop](https://github.com/lichtblick-suite/lichtblick/releases), open a **Foxglove WebSocket** connection to `ws://127.0.0.1:8785`. Add a stock **3D** panel, choose fixed frame `world` and enable `/autodrive/roboracer_1/lidar`. The positions of the laser returns and car frame come from the running engine's scans and `/tf` transforms. Odometry (estimated pose and motion) is on `/autodrive/roboracer_1/odom`.

From another development machine, first run `ssh -N -L 8785:127.0.0.1:8785 USER@ENGINE_HOST` on the **viewing** machine and use the same connection URL. This tunnel recipe is for a device with an SSH client and compatible viewer; no Jetson launch or two-host test was performed.

We tested Desktop 1.31.0. On this VM it needed software rendering (`LIBGL_ALWAYS_SOFTWARE=1` and flags `--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader`). Ordinary GPU-equipped clients need not use that fallback.

With the live view open, the stock keyboard controls reset the car to (0.8000, 3.1583) m, then moved it to about (0.091, −6.610) m. The scan changed as it moved. After stopping, timestamps kept advancing.

**This viewer is different from the Unity game window.** Native Unity graphics on this CPU machine showed magenta or black/corrupted frames. We did not measure the viewer's frame rate or interaction latency.

## What was measured

Workspace data: [November main](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/commit/76db7abecd8c66ca14bbc56d77e82331a08e3cb6). The two Car 1 runs and the Car 2 map race below are development references, **not holdouts**. No November 16 reference was fit, aligned or evaluated. No simulator parameter was changed.

| Sensor observations | Actual ranges/scan | Advertised max (m) | Observed delivery rate (Hz) |
|---|---:|---:|---:|
| Car 1 20261111-153721 | 1,081 | 60 | 40.04 |
| Car 1 20261111-155542 | 1,081 | 60 | 40.06 |
| Car 2 20261111-152132 (map race) | 1,081 | 30 | 39.72 |
| AutoDRIVE headless, idle, stock practice track | 1,081 | 10 | 19.96 |

Bag rates are the inverse median receive interval over each complete bag. The simulator rate is **delivered ROS messages per wall second**, not its 40 Hz sensor setting. Over 10.018 wall seconds, we received 200 each of scan, odometry, camera and lap timer.

The lap timer advanced 9.968 s between its first and last messages (roughly real time, not a physics-step benchmark). Camera frames were 192 × 108 and every sampled byte was 205: **no useful headless camera image**.

Before using the bag comparison, we reproduced the [published Car 1 race study's LB-held replay coverage](../results/2026-11-13-rb1-final-race-gap-follow.md): for 155542, 5,657 selected scans × 0.025 s = 141.4 s; for 153721, 4,699 = 117.5 s. This is controller replay time, not total button-held wall time or evidence of physical trajectory agreement.

**Reset and repeatability.** In three repeats, the stock soft reset returned to (0.8000, 3.1582) m. With zero steer, a 0.1 normalized throttle pulse moved 1.656, 1.598 and 1.641 m. Pulse lengths were 1.001, 1.003 and 1.015 wall seconds, respectively.

After restarting the engine, three more resets returned to the same rounded spawn. The same wall-timed pulse moved 1.616, 1.700 and 1.668 m. These were asynchronous trials, not synchronized or seeded repeats.

**Observation-to-command rate.** On one instance, a no-op subscriber issued zero steering/throttle once per scan: 189 command pairs in 10.013 wall seconds (18.88 Hz). This counts sends, not confirmed physical steps. No learning, inference, batched training or parallel scaling was measured. A run spanning a suspended VM was discarded.

**Why the mapped run has no fidelity score.** Car 2 run 152132's saved map-server and controller parameters select [151026-rebuilt](../../../data/rb2/racerbot_auto/20261111-151026-rebuilt): a 328 × 432 SLAM occupancy grid at 0.05 m/cell and its profiled racing line.

The stock practice release ships a different course and no PGM/YAML track loader. [Upstream's map workflow](https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Racetracks#custom-racetracks) traces/extrudes a 3D track for Unity; it is not a run-time map argument. No replacement course was substituted.

**Why the controls are not matched.** Racerbot commands speed in m/s and steering angle in radians via Ackermann messages. The [stock ROS devkit](https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing/tree/639579a7ac43e4e8f3b6f7e2eb9796b08f9f6de0/autodrive_devkit) accepts normalized throttle and steering as separate floats. Renaming a topic cannot supply a calibrated speed loop.

**Use for training setup:** asynchronous LiDAR observation checks and the stock teleop/viewer. No network training, physical driving or new run data.

## Locked November turning and braking test

**Not supported by the stock practice build.** The test requires a flat obstacle-free scene, seed 0, origin at zero heading and starting speed from the bag. At each 25 ms tick it requires a speed demand in m/s, steering in radians and the resulting body-forward speed and yaw rate. A controller or fitted lag is not allowed.

| Requirement | What the stock release did |
|---|---|
| Scene, seed and initial state | Soft reset to its practice course spawn; no seed or arbitrary pose/velocity setter in the devkit. |
| Physical-unit inputs | Normalized throttle/steer and Boolean reset only. A speed controller would change the experiment. |
| 25 ms post-step outputs | The bridge timestamps asynchronous odometry and IMU at publication; no synchronous step, step ID or simulation clock. |
| Forward speed and yaw rate | Actual ROS messages have odometry linear velocity and IMU angular velocity. Straight forward pulses gave positive odometry x velocity (~2.49 m/s) as world y decreased. No post-step samples verified. |

**Returned coverage:** neither car returned a matched block or tick for turning speed, turning yaw or braking speed. Every per-run RMSE and both per-car equal-weight means are **N/A**, not zero. No partial rollout, alternative track, interpolated engine sample, calibration or substitute dataset was scored.

The command-only audit below uses receive timestamps for `/ackermann_cmd` in every MCAP split. Anchor a 12 s window on the first command and every 10 s thereafter, including only windows whose end is at or before the last command. At all 480 tick starts, check that the latest command is at most 100 ms old.

| Car / partition | Runs | Candidate blocks | With fresh commands | Returned blocks / ticks (all metrics) | Runs with no engine result |
|---|---:|---:|---:|---:|---:|
| rb1 Nov 11 development | 4 | 70 | 67 | 0 / 0 | 4 |
| rb1 Nov 16 evaluation | 3 | 25 | 23 | 0 / 0 | 3 |
| rb2 Nov 11 development | 3 | 56 | 54 | 0 / 0 | 3 |
| rb2 Nov 11 evaluation | 7 | 277 | 271 | 0 / 0 | 7 |

The independently counted command-fresh blocks match all 17 accepted-block counts in the shared November v1 input archive (SHA-256 `c99cb8c70ff55848e26d958343f12b3de2fa11ed0b6a6468b11f1fe1b3f61350`). Across both cars, the engine returned none of the 415 required rollouts (199,200 replay ticks). It attempted none: the stock API cannot initialize or step them.

**Frozen reference coverage.** Each cell below gives **eligible blocks / eligible scored ticks**, summed over whole runs. Turning speed and turning yaw have identical counts here; they remain distinct metrics. Braking measures the response to a speed-demand reduction, not measured brake torque.

| Car / partition | Turning speed | Turning yaw | Braking speed | Runs with no eligible ticks in any metric |
|---|---:|---:|---:|---|
| rb1 development | 30 / 5,719 | 30 / 5,719 | 31 / 8,363 | 153409 (no full block) |
| rb1 evaluation | 12 / 1,345 | 12 / 1,345 | 13 / 1,791 | 095652 (one accepted block, no eligible ticks) |
| rb2 development | 37 / 7,963 | 37 / 7,963 | 38 / 9,295 | None |
| rb2 evaluation | 94 / 17,541 | 94 / 17,541 | 101 / 32,431 | None |

These are expected **reference** counts, not simulation samples. Returned metric blocks / ticks are 0 / 0 for every cell, including all runs with nonempty references. No RMSE or cross-car ranking follows from missing outputs.

**Prior use and configuration.** The two rb1 Nov 11 race-study runs and all three rb2 development runs appeared in earlier work; rb2 152132 informed the floor choice. The [post-race plan](../../../docs/superpowers/plans/2026-11-12-post-race-speed-program.md) includes rb2 evaluation runs 142604 and 154204. Evaluation means reserved for this experiment, not historically unseen.

The cars have different wheel/gyro calibrations. On rb1 Nov 16, 095058 had opponent awareness off and 095242 lacked the gap-follow parameter dump. No missing calibration was replaced using the other car, odometry yaw or a steering estimate. Mapping runs, the 155542 sensor reproduction, and the live-view demo are separate from the open-loop scores.

## Repeat the measurements

From the workspace root on the host, run the read-only bag reader (requires Python and `mcap-ros2-support`, tested at 0.5.7). It reads every split and writes nothing:

```bash
python3 tools/run_analysis/autodrive/november_reference.py data
```

With the engine and devkit running, **close teleop and any other command source**. This probe publishes only AutoDRIVE's stock throttle, steering and reset messages, inside the devkit's localhost-only domain. It prints the observation rates' inputs and three pulse outcomes as JSON lines.

```bash
docker cp tools/run_analysis/autodrive/probe.py racerbot-autodrive:/tmp/probe.py
docker exec racerbot-autodrive bash -c 'source /opt/ros/humble/setup.bash && python3 /tmp/probe.py'
```
