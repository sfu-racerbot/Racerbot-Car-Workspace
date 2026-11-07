# Launch commands: the daily cheat sheet

**Who this is for:** anyone running the car today who already knows roughly what they want to do and just needs the command.
**Read first:** the [safety rules](#safety-every-time) below. If this is your first week, do [quickstart.md](quickstart.md) once.
**You'll be able to:** start any of the car's modes from a copy-paste command and tell whether it worked.
**Time:** a minute to find your command.

Every section follows the same pattern: what it does (one line), the terminals, the commands, and what you see when it works. The reasons and the details are in [operations.md](operations.md), linked from each section.

## Contents

- [Every terminal, first](#every-terminal-first)
- [Safety, every time](#safety-every-time)
- [Check the hardware](#check-the-hardware)
- [Drive by hand](#drive-by-hand)
- [gap_follow: drive itself, no map](#gap_follow-drive-itself-no-map)
- [Map a course: 3 laps, then stop](#map-a-course-3-laps-then-stop)
- [Race on a saved map](#race-on-a-saved-map)
- [Map and race in one go](#map-and-race-in-one-go)
- [Dashboard](#dashboard)
- [Record a run and analyse it](#record-a-run-and-analyse-it)
- [Simulator (no car)](#simulator-no-car)
- [Stop everything](#stop-everything)
- [Where things are saved](#where-things-are-saved)

## Every terminal, first

On a development machine or in the simulator, set `RACERBOT_CAR=rb1` or `RACERBOT_CAR=rb2` before launching. Without a Jetson serial or this setting, the error is `no Jetson serial was found; set RACERBOT_CAR for a dev box or the simulator`.

On either car, `racerbot_car.identity.current_car_id()` reads `/proc/device-tree/serial-number` and looks it up in `src/racerbot_car/cars/machines.yaml`. Leave `RACERBOT_CAR` unset or matching the serial.

A mismatch refuses every launch with `serial ... identifies rbN, but RACERBOT_CAR is ...; unset RACERBOT_CAR`. An unknown serial prints the exact line to add to `machines.yaml`.

In each terminal on the selected machine:

```bash
source /opt/ros/jazzy/setup.bash
source ~/racerbot-ws/install/setup.bash
```

**Working when:** nothing prints. If a `ros2` command later says "not found", you skipped this in that terminal.

## Safety, every time

- **Hold LB** on the F710 (XInput mode) whenever the car should move. Let go and it stops. This is true in every mode below, including the autonomous ones.
- **Wheels off the ground** the first time you run anything new or changed.
- **Never run `teleop_launch.py` together with an autonomous mode.** Its commands always win, so the car would just sit there (or do what the sticks say).
- Someone stands next to the battery switch whenever the car drives itself.

Why: [operations.md, safety checklist](operations.md#safety-checklist-every-time-before-powering-the-drive-motor).

## Check the hardware

**Any terminal:**

```bash
ros2 run racerbot_doctor doctor
```

**Working when:** the last line reads `0 FAIL`. The first line identifies the car, its serial and any tuning overrides. The check is read-only and safe any time.

## Drive by hand

**Terminal 1** (hardware; leave it running for everything below unless a section says otherwise):

```bash
ros2 launch f1tenth_stack bringup_launch.py
```

**Working when:** the VESC (motor controller) and the [LiDAR](glossary.md#lidar) come up without errors. It never moves the car by itself.

**Terminal 2:**

```bash
ros2 launch f1tenth_stack teleop_launch.py
```

**Working when:** holding LB and moving the sticks drives the car. Wiimote instead of the F710: [wiimote.md](wiimote.md). Details: [operations.md](operations.md#manual-driving-teleop).

## gap_follow: drive itself, no map

**Terminal 1:** `ros2 launch f1tenth_stack bringup_launch.py`

**Terminal 2:**

```bash
ros2 launch gap_follow gap_follow_launch.py
```

**Working when:** it prints a decision line every second, and the car drives itself while you hold LB.

**Other cars on the track too?** Opponent awareness is **on by default**, so the command above already runs it on every car. It is simulator-validated only. Do the [first run](#car-1-and-car-2-together-first-run) in order before you run two cars. To turn it off for one run, add `opponent_awareness:=false` to the end of the command above.

**Working when:** the decision lines show `opponent mode=follow` (or `clear`) while another car is moving ahead. For a moment after a followed car is lost, the line can show `opponent mode=follow_memory`.

It follows a slower car instead of squeezing past it, and passes only on a straight with room. On a track as narrow as ours it will mostly follow.

**Do not run two cars with opponent awareness turned off.** With it off, only 24 of 32 simulated two-car runs passed, with 3 contacts and 5 stalls ([gap-follow.md](gap-follow.md#several-cars-at-once)).

**One car, against the clock:** add `time_trial:=true`. It aims at the deepest point of each gap and raises `max_lateral_accel` to 3.0. It turns opponent awareness off, so **never use it with another car on the track**. Simulator-validated only: [gap-follow.md](gap-follow.md).

**Slower or faster for one run:** add `max_speed:=1.0` (any positive speed up to 5.0 m/s). The shared `gap_follow.yaml` sets 5.0 m/s: car 1's 2026-10-27 race defaults were ported to car 2 on 2026-10-29, replacing 3.5 m/s.

Below the YAML speed, the option also lowers `min_speed`, `corner_speed` and `corner_speed_wide` if they are above it. Above it, only the top speed on straights rises: corners remain capped by `corner_speed`, `corner_speed_wide` and `max_lateral_accel`.

Values above 5.0 or non-positive values are refused and nothing starts. When going faster on either car, test with the wheels off the ground first, then in open space. **Car 2 has never driven at the 5.0 default:** first use `max_speed:=1.0` in open space.

### Car 1 and car 2 together: first run

Opponent awareness is on by default, and it has only been tested in the simulator. Do these steps in order, and don't skip ahead. Each step is the cheapest place to find a problem the next one would find with a crash.

**Two people, one per car.** Each holds LB on **their own car's** F710 controller. Letting go of LB stops that car.

1. **Check the cars can't hear each other.** Both cars on, `bringup_launch.py` running on both. On each car, **any terminal**:

   ```bash
   ros2 topic info /joy
   ```

   **Working when:** `Publisher count: 1`. **If it says 2:** the cars share one ROS graph (the system of nodes and topics that can talk to each other, see [glossary.md](glossary.md)). Car 2's LB would then drive car 1, and each car would get the other's drive commands. Stop, and give the cars different `ROS_DOMAIN_ID` values ([concepts.md](concepts.md)).

2. **Wheels off the ground, both cars.** Car on a stand, wheels free. **Terminal 2** on each car:

   ```bash
   ros2 launch gap_follow gap_follow_launch.py opponent_awareness:=true max_speed:=1.0
   ```

   Both cars have `max_speed:=` (car 2 since the 2026-10-24 port from car 1).

   **Know this first:** the first LB press after each launch goes straight to `max_speed` with no ramp ([launch boost](gap-follow.md#launch-boost)), so keep hands and cables clear of the wheels. Add `launch_boost:=false` to turn that off.

   **Working when:** each car's wheels turn only while its own LB is held, and stop when it's released.

3. **Floor, low speed, open space.** Same commands, one car about 2 m behind the other, nobody standing on the track. Start somewhere at least 2 m wide, not the 1.2 m track.

   In the simulator, at these speeds in a 1.2 m corridor, 1–3 of every 8 runs touched a wall, with or without opponent awareness and even with one car alone (2026-10-23, `narrow_1p2`). In wider layouts there was no contact.

   **Working when:** the decision lines (one a second) of the car behind show `opponent mode=follow` and a `follow cap=`. **If either car shows `opponent mode=` with no other car moving,** it's mistaking a wall for a car: let go of LB, relaunch with `opponent_awareness:=false`, and keep that run's log.

## Map a course: 3 laps, then stop

`gap_follow` drives 3 laps while [SLAM](glossary.md#slam) builds the map. Then it saves the map and a [racing line](glossary.md#racing-line), stops the car, shuts everything down, and prints the commands to race on that map. Like every mode here it needs `bringup_launch.py` running first, and it refuses to start without it. Simulator-validated only; not yet run on the car.

**Before you start:** put the car on the track, pointing the way it should lap.

**Opponent awareness is off in mapping.** `map_course` starts `gap_follow` from `gap_follow.yaml`, but the mapping launches turn the layer off again, because mapping is always one car (decided 2026-10-28). Changing `enable_opponent_awareness` in the YAML does not change mapping.

To map with the layer on, add `opponent_awareness:=true` to the `map_course` command. If `opponent mode=` shows with no other car on the track, let go of LB, then relaunch without `opponent_awareness:=true`.

**Terminal 1** (leave it running for mapping *and* racing):

```bash
ros2 launch f1tenth_stack bringup_launch.py
```

**Working when:** the VESC and LiDAR start without errors and the car remains still.

**Terminal 2:**

```bash
ros2 run racerbot_launch map_course
```

**Working when:** the terminal starts mapping and reports lap progress as described below.

**This maps on the gyro by default on both cars.** It starts the BNO085 and the IMU side; SLAM takes its heading from the BNO, then the VESC if the BNO stops. If the bridge prints `sensor hub is not running`, power-cycle the BNO: the run still maps on the VESC's gyro alone.

On car 2, steering-derived heading was 15° out over 5 m of track on 2026-10-30, and run `184826`'s corrections reached 3.4 m.

**Keep the car still for the first 3 seconds** while the IMU's gyro bias is measured, **then hold LB for the whole run.** The car stays still for a moment at the start, until SLAM has recorded the start pose (`Start pose recorded: ...`). Releasing LB pauses mapping; holding it again carries on.

**Working when:** you see `Closed mapping lap 1/3`, `2/3` and `3/3`. The car stops just past the line, `Saved occupancy map successfully.` appears, Terminal 2 shuts down (Terminal 1 keeps running), and the last thing on screen is a block starting `MAPPING FINISHED.` with the race commands in it. Copy them from there.

After the launch stops, `map_course` **checks the map against the run's own scans**, whether the laps finished or you pressed `Ctrl+C`. It writes a folder named `<run>-rebuilt` in `data/<id>/racerbot_auto/`, and prints `GO:` or `NO-GO:` with the reason.

Here and below, replace `<id>` with `rb1` for car 1 or `rb2` for car 2. Replace the other angle-bracketed placeholders with the actual names too.

- **`GO`:** the map is good, and the race commands at the end point at that folder. Its racing line comes from where the car really was on the finished map, capped at **1.5 m/s** for the first drive. The commands show how to go faster.
- **`NO-GO`:** do not race on it. The most common reason is walls drawn twice. Map again, with the gyro on.

**What you see next depends on how the run ended:**

- **It finished by itself:** the check above, then the commands. Nothing else.
- **You pressed `Ctrl+C`, or it failed:** a line says it is stopping and nothing is driving the car. Allow up to 15 s for the nodes to shut down, followed by the check, a `RUN STOPPED` summary and the commands. The summary gives laps closed, verdict and recording location. Press `Ctrl+C` once: it is still working until the prompt comes back.

**The in-depth reports do not run by themselves**, because they take minutes. Add `analysis:=true` to run them straight after a run, or run them in a terminal at the [workspace](glossary.md) root on the recording, whose path is on the `Recording run to:` line:

```bash
python3 tools/run_analysis/slam_report.py ~/racerbot-ws/data/<id>/racerbot_runs/<time>
python3 tools/run_analysis/imu_report.py ~/racerbot-ws/data/<id>/racerbot_runs/<time>
ros2 run race_diagnostics summarize_run ~/racerbot-ws/data/<id>/racerbot_runs/<time>
```

**Working when:** each prints its report and saves it under that run's `analysis/` folder.

`rebuild:=false` skips the map check. If recording is disabled, there is nothing to check.

Mapping defaults to a 3.5 m/s speed cap. The gyro SLAM profile was validated by replay only up to about 3.3 m/s. Raise the default only after a recorded gyro run at a higher speed produces a clean `slam_report`.

**Then you can move the car.** Racing on this map can start from anywhere on the course, facing the way it drove while mapping. See [Race on a saved map](#race-on-a-saved-map).

Options, added to the end of the command:

| Option | When |
|---|---|
| `mapping_max_speed:=1.5` | a course nobody has driven yet |
| `laps:=4` | a different number of laps (at least 2) |
| `fused_speed:=true` | `gap_follow` brakes and accelerates on the IMU-fused speed (`/odom/fused`) instead of the wheel speed. Never more than 0.3 m/s from the wheel speed. **Not yet driven on:** do one normal run first. See [the IMU README](../src/imu_bridge/README.md#in-map_course) |
| `gyro_odom:=false` | SLAM maps on the steering model instead of the gyro. Car 2's replay measurements found this worse. The default, `true`, maps on **gyro odometry** (`/odom/imu`) and loads `slam_gyro.yaml`. Only the map changes: `gap_follow` drives the same either way. |
| `imu_lap_turns:=false` | Count laps in the steering model's `odom` instead of `odom_imu`. **Do not use this with the gyro:** a hard turn then reads as a SLAM correction, and laps stop closing. |
| `imu_all:=true` | Enable `fused_speed` as well. An explicit `false` overrides it. |
| `analysis:=true` | Run the in-depth reports after the launch exits. They take minutes. |
| `rebuild:=false` | Skip the map check after the launch exits. A run stopped early then keeps nothing. |
| `record_bag:=false` | don't save a bag. It is saved by default (a few hundred MB per run, in `~/racerbot-ws/data/<id>/racerbot_runs/`), because it's what lets a run be replayed and analysed afterwards |
| `imu:=false` | Run with no IMU at all: no BNO, no `vehicle_state`, and SLAM on the steering model. Maps worse; only for when the IMU side itself is broken. |
| `imu_hardware:=false` | use simulated IMU readings without starting real sensor readers; added for you when `map_course` detects the simulator |

With `imu_lap_turns` enabled, short odometry gaps use map heading provisionally. On return, the counter replaces that estimate with the gyro turn, avoiding double counting.

Reconciliation requires a gap of at most 0.25 s with provisional turning kept within 90°. Other gaps retain the map estimate because a wrapped heading difference can hide a full turn.

A lap must take at least 4 s, alongside the 300° turning gate. From the second lap on, it must also take at least half as long as the lap before. That way the minimum follows the course: on car 1's 29 m course at mapping speed it is about 8 s, and on a longer course it grows with it.

The former fixed 15 s blocked the first return on the simulator's roughly 29 m oval, forcing two circuits into each recorded lap.

Car 1's corrected `imu_all:=true` simulator run `20261029-170502` (recorded on car 1, not in this repo) recorded 363°, 360° and 359° (1.01, 1.00 and 1.00 revolutions). Its final lap passed and saved a racing profile. See [the simulator results](ros-simulator.md#simulated-imu) for the first-run diagnosis and repeat-run paths.

**If it says `the hardware is not running`:** start Terminal 1 first.

**If it doesn't finish:** `Ctrl+C` stops it, and you keep what it mapped: the map check above still runs. Less than one whole lap gives a map but no racing line, and it says so. A second `Ctrl+C` skips the check and prints the command to run it later. If the last lap fails the quality check, it drives up to two more by itself. Lap details: [operations.md](operations.md#reading-the-terminal-while-it-maps).

## Finishing the BNO085 calibration

**This car 2 procedure takes about 15 minutes and never drives the car.** Battery off is fine: the BNO runs from the Jetson. Nothing else may be using the BNO, so have no `map_course` or `gap_follow` running.

Mapping and steering damping do not need this. **`fused_speed:=true` does**, because until the mount yaw is measured the BNO's forward and sideways may be swapped.

**Terminal 1:** the bridge that reads the sensor.

```bash
cd ~/racerbot-ws/tools/imu_calibration
python3 tools/bno_i2c_bridge.py --pty /tmp/bno085 --no-dyncal
```

**Working when:** it keeps running with no error. **If it says `sensor hub is not running`:** power-cycle the BNO.

**Terminal 2:** the mount yaw, about a minute. On level ground, keep the car still for 3 s when it asks, then push it straight forward by hand.

```bash
cd ~/racerbot-ws/tools/imu_calibration
python3 tools/calibrate_imu.py --imu bno --step mount
```

**Working when:** it prints `still for ... s, then push detected` and an `rpy_deg`. Put that `rpy_deg` under `imus: bno:` in `src/racerbot_car/cars/rb2/imu/car.yaml`.

**Terminal 2:** the accelerometer, about 5 minutes. Hold the car still in 12 different orientations, about 2 s each: the 6 faces, then tilts in between. It captures each pose by itself.

```bash
python3 tools/calibrate_imu.py --imu bno --step accel
```

**Working when:** it reports 12 poses and writes `src/racerbot_car/cars/rb2/imu/calib/bno_calib.json`. Gravity should then read about 9.81 m/s², not 9.65.

**Last, with a tape measure:** measure how far car 2's BNO chip is ahead of the rear axle and its position from the car's centre.

TODO(verify): determine the correct per-car destination for `bno_x_offset` and how calibration tools consume `position` in `src/racerbot_car/cars/rb2/imu/car.yaml`. Do not write car 2's measurement into the shared `vehicle_state.yaml`.

## Testing the IMU in mapping (A/B)

On both cars the gyro is the default, so the "without" run below needs `gyro_odom:=false`.

The question: does SLAM make a better map when its odometry gets its heading from the gyro? The test is two mapping runs of the same course from the same spot, one without and one with `gyro_odom:=true`, both recorded. Everything but the odometry SLAM uses stays the same, so the difference between the two maps is the IMU's.

**Before you start:**
- The car on **battery**, on the track. Mark the start spot and direction with tape.
- Terminal 1: `ros2 launch f1tenth_stack bringup_launch.py`, left running for every step below.

**1. Check the BNO085 points forward** (once per car; 2 minutes). Car 2's roll and pitch were set from gravity on 2026-10-29, but which way is "forward" can only be seen while moving.

**Terminal 2:**

```bash
ros2 launch imu_bridge vehicle_state_launch.py
```

**Working when:** the IMU nodes start without repeated errors; leave this terminal running for the next check.

**Terminal 3**, then push the car forward by hand, firmly, a few times:

```bash
ros2 topic echo /imu/bno/data --field linear_acceleration
```

**Working when:** `x` jumps **positive** on each push, while `y` stays within about ±0.3. If `y` jumps instead of `x`, the mount's yaw is wrong: stop and redo `calibrate_imu.py --imu bno --step mount` in `~/racerbot-ws/tools/imu_calibration`. Then `Ctrl+C` terminals 2 and 3.

**2. Run A, steering odometry.** Car on the tape mark. **Terminal 2:**

```bash
ros2 run racerbot_launch map_course gyro_odom:=false
```

Keep the car still for the first 3 seconds, then hold LB as for any mapping run. **Working when:** it finishes as in [Map a course](#map-a-course-3-laps-then-stop), and prints `Recording run to: ~/racerbot-ws/data/<id>/racerbot_runs/<time>` near the start. Write that folder down as run A.

**3. Run B, gyro odometry.** Put the car back on the tape mark, facing the same way. **Terminal 2:**

```bash
ros2 run racerbot_launch map_course gyro_odom:=true
```

**Working when:** the same as run A, and the start of the output includes `publishing odom_imu -> odom` from `vehicle_state_node`. **If it logs `STOPPED publishing odom_imu -> odom`:** something else is giving `odom` a parent. `Ctrl+C`, and keep the log.

**4. Compare.** For each run's folder:

```bash
python3 tools/run_analysis/imu_report.py ~/racerbot-ws/data/<id>/racerbot_runs/<time>
```

**Working when:** it prints three sections: turn rate from each source, the steering lag and turning gain the gyro measured, and how much heading correction SLAM had to apply. Run B should need far less heading correction than run A.

Also compare the two saved maps on the dashboard (straight walls, no doubled corridors) and the `SLAM corrections absorbed` count each mapping log ends with. Both bags stay on the car, so the SLAM settings can be tuned afterwards without it.

## Race on a saved map

**Put the car anywhere on the course,** on open track and facing the racing direction (the way it drove while mapping). Keep it still until the race controller starts. The race launch matches the car's LiDAR scans (sweeps of distance readings, see [glossary](glossary.md)) against the whole saved map to find it, then starts the localizer there. Tested on recorded scans and in the simulator only; not yet driven.

**Terminal 1:** the `bringup_launch.py` from mapping, still running. Start it again if you closed it.

**Terminal 2**, pick one (the end-of-mapping printout has these with your run's folder already filled in):

```bash
# pure pursuit (the default racing controller)
ros2 launch racerbot_launch race_launch.py run_dir:=$HOME/racerbot-ws/data/<id>/racerbot_auto/<run>
# Stanley
ros2 launch racerbot_launch race_launch.py run_dir:=$HOME/racerbot-ws/data/<id>/racerbot_auto/<run> steering_law:=stanley
# LQR
ros2 launch racerbot_launch race_launch.py run_dir:=$HOME/racerbot-ws/data/<id>/racerbot_auto/<run> steering_law:=lqr
# MPC
ros2 launch racerbot_launch race_launch.py run_dir:=$HOME/racerbot-ws/data/<id>/racerbot_auto/<run> steering_law:=mpc
```

**Working when:** `Particle filter is at the start pose; racing can begin.` appears (the race controller only starts after it), then the car drives the racing line while you hold LB.

**If it can't find the car,** the log says why, and it keeps trying for 2 minutes. Nothing drives in the meantime.

| The log says | What to do |
|---|---|
| `ambiguous` | That spot looks like another part of the course. Move the car somewhere with a more distinctive view, such as near a bend. |
| `does not match this map` | Check this is the right map and that the course hasn't changed. |
| `faces against the racing direction` | Turn the car round. |
| `the car must be stationary` | Leave the car still. |

**To start exactly where mapping stopped instead,** put the car back on that spot and add `start:=start_pose`. Before racing that way, check it really is on the spot. This reads one scan and publishes nothing, so it cannot move the car:

```bash
ros2 run pure_pursuit check_start --ros-args -p run_dir:=$HOME/racerbot-ws/data/<id>/racerbot_auto/<run>
```

**Working when:** it prints `GO: car is 0.05 m and 2 deg from the start pose`. **If it prints `NO-GO`:** it says how far to move the car.

**Forgot the run folder?**

```bash
ls -t ~/racerbot-ws/data/<id>/racerbot_auto | head -3                    # newest first
cat ~/racerbot-ws/data/<id>/racerbot_auto/<run>/commands.txt             # that run's commands again
```

**In the simulator** (map, then race without moving the car): pure pursuit, LQR and MPC each raced 128–142 m with no contact. In one run Stanley swerved around something it wrongly took for another car, and brushed a wall. Start with pure pursuit.

What the steering laws are: [steering-laws.md](steering-laws.md). A better racing line from the same map: [operations.md, optimize the line](operations.md#2b-optional-better-optimize-the-line-instead-of-just-pacing-it).

## Map and race in one go

The same mapping, but it switches straight to racing as soon as lap 2 is clean, with no restart. Starts the hardware too. Opponent awareness is off during its mapping laps too; see [Map a course](#map-a-course-3-laps-then-stop). `opponent_awareness:=true` turns it on there.

**Terminal 1:**

```bash
ros2 launch racerbot_launch auto_map_race_launch.py
```

**Working when:** lap messages, then a handover summary, then `Transition complete: pure pursuit now has drive control.` Details: [operations.md](operations.md#automatic-map--raceline--race-recommended-for-a-new-course).

## Dashboard

**Any spare terminal**, alongside any mode:

```bash
ros2 launch racerbot_launch dashboard_launch.py
```

**Working when:** the car shows up at https://dashboard.sfuracerbot.ca. Details: [web-dashboard.md](web-dashboard.md).

## Record a run and analyse it

**Nothing to start: bringup records every algorithm run by itself** (since 2026-10-26). When an algorithm starts, the bringup terminal prints `recording ... -> ~/racerbot-ws/data/<id>/racerbot_runs/<ts>`; when it stops, `saved run ...`.

Manual driving is not recorded. Turn automatic recording off with `record:=false` on bringup. Details: [run-diagnostics.md, automatic recording](run-diagnostics.md#automatic-recording).

To record something else, such as a manual drive, start `ros2 launch race_diagnostics record_run.py` in its own terminal first. Bringup's recorder then stays out of the way.

Afterwards, in **any terminal**:

```bash
ros2 run race_diagnostics summarize_run ~/racerbot-ws/data/<id>/racerbot_runs/<ts>
python3 ~/racerbot-ws/tools/run_analysis/analyze_run.py ~/racerbot-ws/data/<id>/racerbot_runs/<ts>
```

**Working when:** the second writes `analysis/report.md` in the run folder, explaining every stop. The mapping commands above record a run by themselves. Details: [gap-follow.md, recording a floor run](gap-follow.md#recording-a-floor-run) and [run-analysis.md](run-analysis.md).

## Simulator (no car)

**Never on the car while its drivers are running.** The simulator fakes LB and the LiDAR. It refuses to start if it sees the real drivers, but don't rely on that.

```bash
# the whole map-then-race flow, in the simulator
ros2 launch racerbot_sim sim_auto_map_race_launch.py
# gap_follow on every indoor layout, no ROS (fast)
python3 ~/racerbot-ws/tools/f1tenth_sim/sweep_gap.py --track indoor --seeds 6
# several gap_follow cars at once (awareness is on by default; --set only restates it)
python3 ~/racerbot-ws/tools/f1tenth_sim/eval_gap_multi.py --track indoor_wide --cars 3 --speeds 1.8,3.5,3.5 --set enable_opponent_awareness=true
```

**Working when:** every line reads `pass N/N`, with `contact 0` (single car) or `wall 0  car-car 0` (several cars). Details: [ros-simulator.md](ros-simulator.md), [simulator.md](simulator.md).

## Stop everything

`Ctrl+C` in each terminal. If something is stuck:

```bash
pkill -f "joy_node|joy_teleop|vesc_driver_node|urg_node_driver|ackermann_mux|ackermann_to_vesc_node|vesc_to_odom_node|static_transform_publisher|gap_follow_node|slam_toolbox|particle_filter|pure_pursuit_node|waypoint_recorder_node"
```

Battery off **last**.

## Where things are saved

| What | Where |
|---|---|
| Maps, racing lines, `commands.txt` from mapping runs | `~/racerbot-ws/data/<id>/racerbot_auto/<YYYYMMDD-HHMMSS>/` |
| The same from the simulator | `~/racerbot-ws/data/<id>/racerbot_sim/auto/<YYYYMMDD-HHMMSS>/` |
| Run recordings (logs, events, optional bag) | `~/racerbot-ws/data/<id>/racerbot_runs/<ts>/` |
| Saving those three to GitHub after a session | `tools/save_runs.sh` (see [data/README.md](../data/README.md)) |
| Speeds and limits for `gap_follow` | `src/gap_follow/config/gap_follow.yaml` |
| Racing-line limits for mapping runs | `src/pure_pursuit/config/auto_map_race.yaml` |

Each car's physical values are in `src/racerbot_car/cars/<id>/`, including `car.yaml`, `calibration/*.yaml` and `imu/`. Shared configs hold the algorithm tuning used by both cars.
