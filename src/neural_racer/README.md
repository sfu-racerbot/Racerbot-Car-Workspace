# neural_racer

> **Who this is for:** anyone running, testing or reviewing the learned racing policy on car 1.
> **Read first:** [the neural racer plan](../../docs/superpowers/specs/2026-10-23-neural-racer-plan.md) (sections A4 and A6), and [architecture.md](../../docs/architecture.md#workspace-policy-the-lb-deadman-button-is-mandatory-for-every-node-that-can-move-the-car) for the LB [deadman](../../docs/glossary.md#deadman) rule.
> **You'll be able to:** approve a trained model, run it in shadow mode beside `gap_follow`, and know what stops it.

`neural_racer_node` drives the car with a small neural network trained on FIR. It reads the [LiDAR](../../docs/glossary.md#lidar) scan, the car's speed and the IMU, and outputs steering and speed 40 times a second.

**This node can move the car.** It is driving code, and every rule in `CLAUDE.md`'s safety model applies to it.

## Highlights

- **Runs on the car's CPU in plain NumPy.** No PyTorch or GPU on the Jetson. Measured on car 1's Jetson, one core: the whole tick (observation, network, safety wrapper) takes **3.90 ms median, 4.03 ms p99**, 16% of the 25 ms tick.
- **Shadow mode is the default.** Until you pass `shadow:=false`, it publishes only on `/neural_racer/shadow_drive`, never on `/drive`. It refuses a shadow topic that reaches the motor (`/drive`, `/teleop`, `/ackermann_cmd`, `/commands/...`).
- **It only loads approved models.** A model must be in `~/.ros/neural_racer/models/manifest.yaml`, put there by `approve_model` after its hash, its observation spec and a 1e-5 golden check against PyTorch all pass. Driving `/drive` also needs both referee reports to say pass.
- **The network is never the last word.** A deterministic wrapper runs around it every tick: the LB deadman, input freshness, NaN checks, clipping, a steering-rate limit, `gap_follow`'s own LiDAR braking cap and flank creep (imported from `gap_logic`, not copied), and a cornering cap. Each step can only lower the speed.
- **The camera is pre-wired, not used.** The network has a camera slot whose weights start at exactly zero, so adding a camera later changes nothing until it is trained. `camera_enabled: true` is refused until Phase C exists.

### Why it exists

`gap_follow` reacts to what it sees but can't plan; `pure_pursuit` needs a map. This node aims to race well on lap 1 of an unseen track, use the lap-1 map from lap 2, and handle other cars. All of that is learned in simulation and judged by the calibrated CPU simulator before it reaches the car. See the plan for the whole design.

## Safety

- **Hold LB on the F710 the whole time.** Released, never received, or stale (`joy_timeout_sec`, 0.5 s) means a zero command and a reset of the network's memory. `enable_deadman: true` is the shipped value; never set it false.
- **A stale input is a stop.** The node commands zero, rather than guessing, when any of these hold:
  - a scan older than 75 ms
  - odometry older than 0.1 s
  - an IMU that is stale or reported faulted on `/imu/status` (`require_imu: true`)
  - in map mode, a pose older than 0.2 s
- **Follow the test order.** Static topic check → shadow mode → wheels off the ground → floor at `max_speed:=1.0`. The plan's Phase 7 and Phase 10 have the exact steps. Raise `max_speed` 0.5 m/s at a time, only after a clean run.

## Approving a model

**The current model is in this repo:** [models/](models/README.md) has its folder, its results and the commands to approve and run it. For any other model:

**On the car**, after copying an exported model folder from FIR (plan, B0 "Getting a model from FIR to the car"):

```bash
source /opt/ros/jazzy/setup.bash && source ~/racerbot-ws/install/setup.bash
ros2 run neural_racer approve_model ~/.ros/neural_racer/incoming/<hash> --shadow-only
```

**Working when:** it prints `approved <hash> for shadow mode only; golden check passed`.

**If it doesn't:** it prints `REFUSED:` and the reason: a hash that doesn't match, a spec version mismatch, or the NumPy forward pass disagreeing with PyTorch. Drop `--shadow-only` only once the model's `selection_report.json` and `referee_report.json` both pass.

## Running in shadow mode

**Build it** with `gap_follow`: the wrapper imports `gap_follow`'s braking geometry (`gap_logic`), so `gap_follow` must be built even when its node doesn't run.

```bash
colcon build --symlink-install --packages-up-to neural_racer
source /opt/ros/jazzy/setup.bash && source install/setup.bash
```

### Beside `gap_follow`, which drives

**Terminal 1** — bringup, as usual:

```bash
ros2 launch f1tenth_stack bringup_launch.py
```

**Terminal 2** — `gap_follow`, which is the one driving. Its launch also starts the IMU stack (`imu:=true` by default), the same `/imu/data` the network reads, so don't start `vehicle_state_launch.py` as well, or the IMU stack runs twice:

```bash
ros2 launch gap_follow gap_follow_launch.py
```

**Working when:** `ros2 topic echo /imu/status --once` shows `"state": "ok"`. Keep the car still for about 3 s after starting, for the gyro bias and the gravity check. If it shows `fault`, its `reason` names the problem ([The `/imu/status` verdict](../imu_bridge/README.md#the-imustatus-verdict)).

**Terminal 3** — the network, in shadow mode:

```bash
ros2 launch neural_racer neural_racer_launch.py model_hash:=<hash>
```

**Working when:** the log says `SHADOW -> /neural_racer/shadow_drive`, and `ros2 topic hz /neural_racer/shadow_drive` reads about 40 Hz while LB is held.

**If it doesn't:** a `REFUSING TO DRIVE:` line names the reason.

### On its own, with nothing driving

The network alone, with no `gap_follow`: nothing publishes `/drive`, so the car doesn't move even with LB held. Use it to check the network's output with the wheels off the ground, or with the car pushed by hand.

```bash
ros2 launch f1tenth_stack bringup_launch.py                       # terminal 1
ros2 launch imu_bridge vehicle_state_launch.py                    # terminal 2: the IMU stack; keep the car still ~3 s
ros2 launch neural_racer neural_racer_launch.py model_hash:=<hash>   # terminal 3
```

**Working when:** the same checks as above.

**For it to drive** (`shadow:=false max_speed:=1.0` on terminal 3), the model needs a full approval, with passing `selection_report.json` and `referee_report.json` (see [Approving a model](#approving-a-model)). With a shadow-only approval the node refuses: `REFUSING TO DRIVE: model refused`.

## Tests

**ROS-free tests** run unsourced, straight from the workspace:

```bash
python3 -m pytest src/neural_racer/test/ -v --ignore=src/neural_racer/test/test_neural_racer_node.py
```

**Node tests** need ROS sourced, and their own domain, never the car's graph:

```bash
source /opt/ros/jazzy/setup.bash && source ~/racerbot-ws/install/setup.bash
export ROS_DOMAIN_ID=79 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
python3 -m pytest src/neural_racer/test/ -v
```

**Working when:** every test passes. Unsourced, the ROS-free set is 153 tests. Two of them are the Jetson timing gates in `test_latency.py`, which fail on a busy dev box. Every node test publishes on `/test_only/drive`, and `test_no_test_here_builds_a_node_on_the_real_drive` enforces it.

`tools/neural_racer/check_mutations.py` proves the key tests fail against a deliberate break of the code they cover (`CLAUDE.md` Hard rule 1).
