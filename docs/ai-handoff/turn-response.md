# Does a better turning-rate model make a better simulator?

> **Who this is for:** the next simulator maintainer or assistant, including Claude.
> **Read first:** [team rules](team-rules.md) and [the previous tyre experiment](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/108).
> **You'll be able to:** reproduce the November comparison and see why neither car's default changed.
> **Time:** five minutes to read; longer for offline replay.

**Decision: keep the existing racing physics.** Car 2 followed recorded turning commands more accurately, but when the controller steered itself on the club map, the simulated car lapped too quickly. Car 1 missed the turning-rate gate. Both candidates remain offline experiments; nothing here authorizes driving the car.

## What was frozen

Instead of estimating tyre forces, predict **yaw rate** (how fast the car rotates) from the simulated speed and steering. A gyroscope measured the target independently of those commands. The [identification literature](https://doi.org/10.3390/engproc2024079053) motivates fitting the response directly; the equations here are our own approximation.

Each car has four parameters: gain, steering trim, lateral-acceleration ceiling and response time. The candidate replaces the existing steering grip clamp and the yaw/slip equations. Gym still integrates position, steering and speed. Below 0.5 m/s Gym keeps its original equations, including reverse.

Fit only November development blocks. [Frozen manifests and implementation](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/commit/5ab932eb02aa2f6149e2715efd3f464038ca1f88) preceded held-out scoring. Both response times reached the 25 ms lower search bound; that is **not** a measured mechanical time constant.

The per-car gate required at least 10% lower mean turn-rate error, a majority of eligible drives improving, neither speed error rising over 2%, and every block complete. Passing meant “test transfer,” not “promote to default.”

## Commands replayed without feedback

Each cell is baseline → candidate, lower is better. Root mean square error (RMSE) weights large errors more heavily; the table then gives each nonempty drive equal weight. Speed is a wheel-derived estimate, **not** independent ground truth.

| Car / partition | Turn rate (rad/s) | Turn speed (m/s) | Braking speed (m/s) |
|---|---:|---:|---:|
| 1 development | 0.292 → 0.232 | 0.187 → 0.189 | 0.230 → 0.232 |
| 1 evaluation | 0.423 → 0.407 | 0.242 → 0.233 | 0.378 → 0.374 |
| 2 development | 0.236 → 0.109 | 0.119 → 0.114 | 0.128 → 0.125 |
| 2 evaluation | 0.313 → 0.175 | 0.170 → 0.171 | 0.219 → 0.219 |

- Car 1: both eligible drives improved, but by only 4.0% on average. The third had no scored turns.
- Car 2: all seven eligible drives improved; mean turn-rate RMSE fell 44.1%.
- Both models returned all 199,200 ticks across 415 accepted 12-second blocks. No October data entered this study.

## Path and sideways motion against the club maps

Replay the recorded steering and speed from the same trusted scan-to-map starts. Compare the final rear-axle position and direction over 1, 2 and 4 seconds. Slip is the angle between the car's heading and its estimated direction of travel at the assumed midpoint of the chassis.

| Car | Horizon | Windows | Final position error (m) | Slip RMSE (rad) | Final heading error (rad) |
|---|---:|---:|---:|---:|---:|
| 1 | 1 s | 53 | 0.260 → 0.293 | 0.110 → 0.141 | 0.167 → 0.174 |
| 1 | 2 s | 44 | 0.613 → 0.692 | 0.139 → 0.174 | 0.303 → 0.359 |
| 1 | 4 s | 24 | 1.323 → 1.652 | 0.137 → 0.163 | 0.517 → 0.705 |
| 2 | 1 s | 544 | 0.218 → 0.183 | 0.088 → 0.080 | 0.115 → 0.090 |
| 2 | 2 s | 450 | 0.503 → 0.389 | 0.091 → 0.081 | 0.177 → 0.140 |
| 2 | 4 s | 307 | 1.183 → 0.963 | 0.093 → 0.081 | 0.212 → 0.242 |

These are window-weighted means, not independent lap trials. All seven evaluation drives contributed for Car 2; only 095242 contributed for Car 1. This cross-check uses the upstream driving commands (`/drive`), not the final commands used in the gyro benchmark. At four seconds Car 2 ended closer but pointed farther from the recorded heading.

## When the controller steered itself

The unchanged race controller followed the rebuilt November 11 Car 2 map. Six paired seeds, three flying laps per seed, no contact for either model. Baseline mean lap: **10.046 s**. Candidate: **9.326 s**. The recorded comparison was **10.05 s**.

This is a regression against a known floor-calibration lap, not an independent lap-time validation. The candidate's assumption that the rear axle follows its heading does not measure how far the car slides sideways. Do not use it to validate spins, collision margins or faster racing.

## Replay from this revision

**Terminal — development computer**, from the workspace root after the [Gym setup](../../tools/f1tenth_sim/README.md). No ROS process or connection to a car is needed. Set `KIT` to the extracted private **canonical-inputs-v1** directory containing the frozen tables and scorer.

```bash
set -e
export PYTHONPATH="$PWD/.sim/python:$PWD/src/drive_intent${PYTHONPATH:+:$PYTHONPATH}"
OUT=$(mktemp -d)
SIM=tools/f1tenth_sim
for car in rb1 rb2; do
  python3 "$SIM/fit_turn_response.py" fit "$car" "$KIT/frozen" "$OUT/$car.json"
  cmp "$OUT/$car.json" "$SIM/turn_response_study/$car.json"
  python3 "$SIM/fit_turn_response.py" replay "$car" "$KIT/frozen" "$OUT/$car-base.csv"
  python3 "$SIM/fit_turn_response.py" replay "$car" "$KIT/frozen" "$OUT/$car-candidate.csv" \
    --model "$SIM/turn_response_study/$car.json"
done
for mode in base candidate; do
  { cat "$OUT/rb1-$mode.csv"; tail -n +2 "$OUT/rb2-$mode.csv"; } > "$OUT/$mode.csv"
  python3 "$KIT/score.py" "$KIT/frozen" "$OUT/$mode.csv" "$OUT/$mode-scores.json"
done
```

**Working when:** both manifests match on the reference Python 3.10/Gym stack and both score files report all expected blocks. A hash mismatch means the input kit differs: stop, do not disable the check. The real gyro and wheel samples are not fed back during replay.

For the mapped lap test use the [Car 2 comparison instructions](https://github.com/sfu-racerbot/racerbot-docs/blob/main/resources/CAR2_NOVEMBER_CORNERING.md) with six seeds. Omit the grip override in both arms; add `--turn-response tools/f1tenth_sim/turn_response_study/rb2.json` only for the candidate.

**Spatial replay:** the optional `tools/f1tenth_sim/turn_response_spatial.py` reuses the [SLAM rollout tools at the tested revision](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/tree/8b6e57907e0f667f4c156d0489e20edf8a0afa25). It needs that revision's `slam_rollout.py`, `run_analysis/slam_motion.py` and a bag reader (ROS or that revision's rosbags fallback). Supply the private November scan-to-map tracks, not raw wheel odometry.

**Terminal — same offline workspace**, with those tools available. List each evaluation run for the car, then rerun without `--model` to a different output file. This example runs one drive.

```bash
python3 tools/f1tenth_sim/turn_response_spatial.py rb2 "$LOC" "$OUT/spatial-one.json" \
  data/rb2/racerbot_runs/20261111-135453 --model tools/f1tenth_sim/turn_response_study/rb2.json
```

**Working when:** the JSON reports 1-, 2- and 4-second window counts. Never count empty coverage as zero error. Map matching uses a gyro/wheel motion prediction; these positions and slip estimates are useful cross-checks, not external motion capture.

## Where Claude should resume

1. Keep both defaults. Use the Car 2 candidate only for recorded-command turn-rate comparisons on this floor.
2. Preserve the private replay bundle with the canonical keys and both models, including Car 1's negative result. The baseline matched the reference kit's per-run RMSEs exactly.
3. Before fitting anything else, identify new, untouched whole drives. Reserve them **before** inspecting errors. Do not tune this model on the November evaluation runs or reuse the October holdouts.
4. Look for an independent measure of direction of travel and chassis heading, with trustworthy timing through corners, and a matched controller/map/lap reference. If missing, write a measurement plan rather than inferring slip from steering or running a car.
