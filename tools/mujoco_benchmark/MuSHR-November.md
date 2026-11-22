# MuSHR: ready for replay; no four-engine accuracy result yet

The flat-floor MuSHR model supports the specified moving reset, Ackermann inputs and post-step body measurements. The recording-side check below establishes a ceiling on eligible inputs. **It is not a turning/braking score.** No real-data rollouts have been attempted against frozen common inputs. No eligible subset, reference substitute, or cross-car ranking is reported.

## Input coverage

[November data](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/tree/76db7abecd8c66ca14bbc56d77e82331a08e3cb6/data), with no changes to that tree. Used all MCAP splits in all 17 named runs, integer receive timestamps on the mux topic, 12 s full windows every 10 s, and a latest-at-start age limit of 100 ms on all 480 inputs. No ref measurements or engine outputs contributed to these counts.

| Car / split | Runs | Full blocks | All commands fresh | Fresh input ticks (incl. warmup) | Runs already known empty |
|---|---:|---:|---:|---:|---:|
| rb1 development | 4 | 70 | 67 | 32,160 | 1 (153409) |
| rb1 evaluation | 3 | 25 | 23 | 11,040 | 0 |
| rb2 development | 3 | 56 | 54 | 25,920 | 0 |
| rb2 evaluation | 7 | 277 | 271 | 130,080 | 0 |

Thirteen blocks fail the command-age gate. **415 is an upper bound** before initial wheel speed, finite command values, and per-metric measurement quality, freshness and maneuver masks. A fresh command timestamp does not prove valid contents. A run not already known empty may still be empty for one or more metrics.

## Accuracy, coverage and missingness

For each of the four rows above and each of turning speed (m/s), turning yaw (rad/s) and braking speed (m/s): exact eligible blocks, nonempty runs and expected masked ticks are **N/A** until the common extractor supplies frozen masks and references. Returned scored ticks are **0**; all per-run RMSEs and equal-weight car means are **N/A**, not zero. No canonical rollouts attempted; no attempted rollout failed. For rb1 development 153409, expected ticks and blocks are already zero for all metrics. See `mushr-score-status.csv` for one row per run and metric (51 rows).

## What was tested

- MuJoCo 3.3.7 and [unmodified stock MuSHR](https://github.com/prl-mushr/mushr_mujoco_ros/tree/c86eef33dadb0e98fe916a55ac6aa1fc03af027e), one car on a plane. Same engine/model for both cars. Only solver timestep changed from 10 ms to 5 ms so five substeps reach every 25 ms boundary exactly in the replay grid.
- Full native reset per block: horizontal origin, identity attitude, zero steering and yaw, body speed initialized at the supplied value; four wheel rates initialized from speed divided by nominal axle height. No correction after reset. Speed and steering go directly to the named actuators as in the upstream ROS connector. Native actuator dynamics and limits remain active.
- Read local body-origin X velocity and Z angular velocity **after** each tick using MuJoCo forward kinematics; not wheel spin, finite-differenced pose, or steering-derived yaw.
- Seed 0 protocol: this worker has no random action generator, process noise or randomized parameters. Repeated identical inputs after both reused and fresh initialization gave bit-identical outputs. Synthetic block: initialize 1.7 m/s, command 2 m/s and +0.15 rad for 240 ticks, then zero both for 240. All 480 outputs returned over 12 s; positive yaw on left steering and near-zero final speed. This tests support and sign, not car fidelity.
- CSV rejects incomplete or misordered blocks, duplicates, non-finite inputs and numerical failures, preserves nanosecond keys as integers, and publishes no partial output on failure. Two tests passed; deliberately zeroing the moving reset and round-tripping the block key through float caused both tests to fail.

## Prior use and configuration — separate from accuracy

Checked run-ID mentions in the pinned workspace's analysis results and docs, and compared recorded configuration snapshots. rb1 153721/155542 and rb2 135934/142254/152132 appear in prior analyses. In the [post-race plan](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/blob/76db7abecd8c66ca14bbc56d77e82331a08e3cb6/docs/superpowers/plans/2026-11-12-post-race-speed-program.md), rb2 evaluation runs 142604 and 154204 already have published lap summaries; 142604 also appears in the grip-study task. The evaluation split is held out of this MuSHR setup, **not historically untouched**. Absence of a matching ID in those documents is not proof of no prior use. The earlier component studies supplied with the recipe have separate splits and outcomes. No evaluation outcome was used to tune or select the worker.

Where the runtime parameter snapshot exists, speed-to-ERPM / steering gain / steering offset were 4600 / −0.83 / 0.366 for rb1 and 4614 / −0.78 / 0.5904 for rb2. rb1 153409 is missing that runtime snapshot. rb1's Nov 16 overlay manifest adds auto-map-race calibration and pure-pursuit tuning versus Nov 11; the VESC overlay and copied IMU files compared here were unchanged within each car. rb2 has one overlay-manifest hash across these runs. Artifact hashes are not substitutes for verifying a calibrated source at a particular tick. See the per-run audit CSV for hashes and matched documents.

## Handoff

1. Apply the attached replay-worker patch on top of the [existing viewer PR](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/98). From the repository root, with the PR's MuJoCo environment active and its pinned model in `MUSHR_MODEL`, run `python -m pytest -q tools/mujoco_benchmark/test_four_engine.py`.
2. Sim Fidelity owns the prequalified blocks, starting speeds, masks and calibrated references. Give the worker the 480 ordered tick-start commands for each accepted block as CSV: `car,run,b_ns,step,speed_cmd,steer_cmd,initial_speed` (the last column is read only on step 0). Run `python tools/mujoco_benchmark/four_engine.py "$MUSHR_MODEL" inputs.csv predictions.csv`.
3. The common scorer must compare the exact expected and returned keys before reporting any RMSE; the worker validates the blocks supplied, not the absence of an entire block. No score on a convenient subset. Body speed is compared against a wheel proxy, not independent ground speed. Compare yaw only to a calibrated independent gyro. Use equal run weights within car, no pooling between cars.

The separate Nov 11 scalar-demo patch replays the historical straight-steering development comparison; it is not used by this worker. The mapped-course controller, real hardware, learning, and viewer tests are not part of this accuracy experiment. MuSHR's stock model does not provide map collisions or LaserScan. No code was pushed or merged in this follow-up.
