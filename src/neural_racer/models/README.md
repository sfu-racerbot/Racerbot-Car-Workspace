# Neural racer models

> **Who this is for:** anyone putting a trained neural racer on car 1, or checking how good it is.
> **Read first:** [neural_racer's README](../README.md) (approval, shadow mode, safety) and the [plan's B0](../../../docs/superpowers/specs/2026-10-23-neural-racer-plan.md#getting-a-model-from-fir-to-the-car).
> **You'll be able to:** approve the current model and run it in shadow mode beside `gap_follow`, and know what it does and doesn't do yet.

Each folder is one exported model, exactly as `neural_racer_train.export` wrote it: `weights.npz`, `model_card.yaml` and `golden.npz`. The folder's name is the first 16 hex digits of `weights.npz`'s SHA-256. `approve_model` checks it, so don't edit the files.

## a0eef2af0a3757b6: current best (shadow mode only)

| | |
|---|---|
| Source | FIR run `league-v1-smoke`, role `main0`, update 1200. Privileged teacher (PPO), distilled into this LiDAR + IMU student (DAgger), then 3 hours of league self-play. Nov 3, 2026, on one H100. |
| Network | LiDAR conv encoder (32-64-64-64 channels), GRU memory (128), 646,274 parameters. |
| Inputs | **Observation spec v1: the car's current inputs.** It needs no change to `observation.py`. |
| Cornering cap | **Built in at 2.0 m/s², the car's `max_lateral_accel`** (card: `a_lat_folded: 2.0`). |
| Approval | **Shadow mode only.** It has no passing `selection_report.json` / `referee_report.json`, so `approve_model` won't approve it for `/drive`. |
| Driven on a car | Not yet. |

**About the cornering cap.** The network was trained with the cornering cap as an extra input (spec v2's 24th `ego` value). This export folds that input into the first layer at 2.0 m/s², so it computes exactly what the trained network does at that cap. Its golden check compares against the trained PyTorch network with the cap given as 2.0. To raise `max_lateral_accel`, re-export at the new value, or the network keeps driving as if the cap were 2.0:

```bash
python -m neural_racer_train.export --from-export <spec-2 model dir> <out dir> --car-spec 1 --a-lat 2.5
```

**It needs this branch's safety wrapper.** On `main`'s wrapper it finishes no time trial, 0 of 18. The braking cap read the nearest return in any direction, so on our 1.0–1.5 m wide courses it held the car to about 0.7 m/s on 798 of 800 ticks, and the car stalled within 12 m. This branch's wrapper reads `gap_follow`'s swept path instead: how far the body can drive along the arc it is steering.

**Why this one:** it finishes the most simulated time trials of any neural model we've benchmarked, and it's the fastest of the reliable ones. The newer league-v2 snapshots (Nov 9) did no better.

### How it does (simulation only)

These are runs in the calibrated CPU simulator (`tools/f1tenth_sim`, car 1's profile), 3-lap races, speed cap 5.05 m/s (the VESC limit). They're on this branch's car code (v1 observation, swept-path wrapper), against `main` 2fb626e's `gap_follow`, 3 runs per course.

**At the shipped 2.0 m/s² cap**, mean lap time and finishes:

| Course | Car 1's default physics | Nov 6 measured floor | gap_follow, default |
|---|---|---|---|
| car1_course_2026_10_26 | 3/3, 13.42 s | 3/3, 13.17 s | 3/3, 11.97 s |
| car2_course_2026_11_06a | **0/3** | 3/3, 14.07 s | 3/3, 12.79 s |
| car2_course_2026_11_06b | 3/3, 10.04 s | 3/3, 9.68 s | 2/3, 10.94 s |
| course_2026_10_30 | 3/3, 20.20 s | 3/3, 19.96 s | 3/3, 19.59 s |
| room_2026_07_27 | 3/3, 7.56 s | 3/3, 7.37 s | 3/3, 8.09 s |
| levine_blocked | 3/3, 18.04 s | **0/3** | 3/3, 17.71 s |
| **Finished** | **15/18** | **15/18** | **17/18** |

At this cap it's slower than `gap_follow` on car 1's course and about level elsewhere. The speed shows at a higher cap. The same trained network at 5.0 m/s² laps 15–22 % faster than `gap_follow` on our courses: car 1's course 11.02 s against 13.78 s, in FIR bench9 against `main` 9510526's slower, traffic-safe `gap_follow`. That needs a re-export at the higher cap, and a measured-grip decision first.

**Head to head** against `gap_follow` (local, `main` 2fb626e, cap 5.0, 24 races, scored on finishing order): 11 wins, 9 losses and 4 draws, with no contacts. It won only 2 of the 12 races it started second: it rarely overtakes. Against `main` 9510526's `gap_follow` over 12 tracks (FIR bench9), it caused 15 at-fault contacts per 100 races.

### How to run it

**On car 1**, with this branch pulled:

```bash
cd ~/racerbot-ws
colcon build --packages-select neural_racer
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 run neural_racer approve_model src/neural_racer/models/a0eef2af0a3757b6 --shadow-only
```

**Working when:** it prints `approved a0eef2af0a3757b6 for shadow mode only; golden check passed`.

**If it doesn't:** a `REFUSED:` line names the reason: a hash that doesn't match, a spec mismatch, or the golden check.

**Then** run it in shadow mode as in [Running in shadow mode](../README.md#running-in-shadow-mode), with this hash. `gap_follow` drives; the network only publishes on `/neural_racer/shadow_drive`:

```bash
ros2 launch neural_racer neural_racer_launch.py model_hash:=a0eef2af0a3757b6
```

**Working when:** the log says `SHADOW -> /neural_racer/shadow_drive`, and `ros2 topic hz /neural_racer/shadow_drive` reads about 40 Hz while LB is held.

**Record the run** so its commands can be compared with `gap_follow`'s offline:

```bash
ros2 bag record /scan /odom /imu/data /imu/status /joy /drive /neural_racer/shadow_drive
```

**Inputs it reads from the config:**
- `max_speed` is both the wrapper's speed cap and the network's speed-cap input. Keep the Phase 10 ladder: 1.0 m/s first.
- `max_lateral_accel` must stay 2.0 for this export (see above).

### Before it drives `/drive`

- **It needs passing reports.** `approve_model` refuses it for driving without a passing `selection_report.json` and `referee_report.json`, and this model has neither. That gate is the team's release decision. Don't work around it.
- **Re-measure the tick on car 1's Jetson.** The swept-path braking cap adds about 0.1–0.25 ms per tick on the pit PC. Run `test_latency.py` on the Jetson; the budget is 5 ms p99.
- **The wrapper's stopping guarantee runs out at about 3.4 m/s.** The braking cap reads 3.0 m of swept path (`swept_clearance_max_length`) at 3.0 m/s² (`max_braking_decel`). Above that speed, the wrapper can't promise a stop for what it sees.
- **It's been trained and tested only in simulation.** No real-car data has checked it yet.
