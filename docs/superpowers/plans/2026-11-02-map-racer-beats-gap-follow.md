# Map-based racer beats gap_follow: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Who this is for:** whoever builds this work, human or agent.
> **Read first:** [the design](../specs/2026-11-02-map-racer-beats-gap-follow-design.md), `CLAUDE.md`, `TEST_QUALITY_STANDARDS.md`, [sim-calibration.md](../../sim-calibration.md).
> **You'll be able to:** build each piece and know from its test or its measurement that it is done.
> **Time:** several working days of compute-bound work. The floor test plan (piece F) is one session per car, on the car, by a person.

**Goal:** the map-based stack (SLAM → particle filter → racing line → `pure_pursuit` / `auto_map_race` race phase) laps faster than `gap_follow` with zero contacts on our real courses, for rb1's and rb2's calibrations. It is proven on recorded data and in a simulator that was first checked against gap_follow's real laps.

**Architecture:** Six pieces, one branch and one PR each, none merged here. **A** makes the simulator trustworthy for both cars and makes racerbot_sim (the real nodes) the simulator of record. **B** and **C** measure SLAM and localization on the real bags and fix what fails. **D** builds one line-and-profile path from the map, used by both race routes, with margins sized from measured tracking error. **E** races it against gap_follow in simulation and ships the winning settings as defaults. **F** writes the floor test plan.

**Tech Stack:** Python 3.12, numpy, scipy, F1TENTH Gym (`.sim/`, pinned `bdaec14`), slam_toolbox 2.8.5, particle_filter + range_libc, rosbag2_py through `tools/run_analysis/runlog.py`, ROS 2 Jazzy via `rb`.

**Spec:** `docs/superpowers/specs/2026-11-02-map-racer-beats-gap-follow-design.md`

## Global Constraints

- Every node that can move the car keeps its LB deadman and its own four deny-path tests (LB held → non-zero; released, never received, stale → zero). Any task touching `pure_pursuit_node.py`, `auto_map_race_node.py` or `gap_follow_node.py` re-runs that node's deny-path tests and lists them in its commit message.
- Never relaxed: `enable_deadman`, TTC, swept clearance, `emergency_stop_clearance`, `emergency_stop_distance`, `body_contact`, `profile_wall_clearance` (0.20 m), `max_cross_track_error`.
- Contact is judged by `sim_fidelity`'s chassis-against-distance-transform check or `SimBridge.body_contact`/`car_contact`. **Never by Gym's collision flag.**
- Every new or changed test is seen to fail against a break (stub, return-the-expected-constant, and for math one of the CLAUDE.md mutations) before it counts. No test is weakened; a test that looks wrong is flagged to the tech lead, not edited.
- Every expected value in a test names its oracle: closed form, invariant, cited spec, or recorded measurement (run name).
- All ROS, pytest and simulator commands run as `RB_WS=<worktree> rb <cmd>` (domain 79, localhost). Simulations use `--workers 2`. Run `uptime` first and wait if load is over 6. `rb` queues heavy commands machine-wide.
- The worktree needs `ln -s ~/racerbot-ws/.sim .sim` (gitignored) for the Gym simulators, and `git submodule update --init src/particle_filter src/range_libc` for piece C.
- Car-specific simulator runs set `RACERBOT_CAR=rb1|rb2` or pass `--car`.
- **rb1's calibration files (`src/racerbot_car/cars/rb1/calibration/*`) are not edited.** Another session is changing car 1's speed calibration. Every rb1 result is marked provisional until task E5.
- Sim-to-real tolerance: simulated gap_follow's median lap within **±3%** of the real median lap for that car and course.
- Final head-to-head tables: 30 seeds × both directions. Tuning: at least 6 seeds.
- Real data is read-only. Outputs go to `~/.cache/racerbot/` or the job tmp dir, never into `data/`, except new fixtures under a package's `test/data/`.
- No floor runs and no hardware from these sessions.

## Review Focus

1. **A course pinch point narrower than car + 2 × margin** (car 2's course is 0.78 m at its narrowest): the line builder must fall back to the centre there and say so, never refuse the whole course and never put the line inside the margin elsewhere. Test in D2.
2. **`/odom/imu` absent when the particle filter is told to use it** (`imu:=false`, BNO missing): the filter must fall back to `/odom` and log it, never run without a motion model. Test in C3.
3. **A floor profile applied to a car it wasn't measured on:** must keep that car's steering, speed loop and odometry scales. Test in A1.
4. **A head-to-head seed with any contact:** that candidate is disqualified for the whole table, even if its median lap is fastest. Test in E1.
5. **A replay or simulation started beside real drivers:** must refuse (the existing `slam_replay` / `racerbot_sim` interlocks). The new `pf_replay.py` reuses the same check. Test in C2.

---

## File structure

| File | Piece | Responsibility |
|---|---|---|
| `tools/f1tenth_sim/sim_fidelity/calibration.py` | A | `Floor` splits floor values from car values |
| `tools/f1tenth_sim/validate_calibration.py` | A | `--car` on `sim`, so rb1 can be scored |
| `src/racerbot_sim/racerbot_sim/sim_bridge.py`, `gym_bridge_node.py`, `launch/sim_bringup_launch.py` | A | `floor:=` arg: the calibrated plant behind the real ROS topics |
| `tools/racerbot_sim/run_gap_follow_validation.py` | A | `--track/--floor/--car`, laps from ground truth |
| `src/racerbot_launch/config/slam_gyro.yaml` | B | changed only if a course fails replay |
| `tools/run_analysis/pf_replay.py` (new) | C | bag → real particle_filter on its own domain → pose track |
| `tools/run_analysis/pf_metrics.py` (new, no rclpy) | C | error against `loc.py` truth, by speed; delay |
| `tools/run_analysis/build_range_libc_cpu.sh` (new) | C | CPU range_libc into `~/.cache/racerbot/range_libc_cpu` for the dev box |
| `src/racerbot_launch/launch/race_launch.py` | C | particle filter motion model on `/odom/imu` with fallback |
| `src/pure_pursuit/pure_pursuit/race_line.py` (new, no rclpy) | D | map + seed lap → optimized-or-recorded line + profile, one path for both routes |
| `tools/run_analysis/rebuild_run.py`, `src/pure_pursuit/pure_pursuit/auto_map_race_node.py` | D | call `race_line.build_race_line` |
| `src/racerbot_car/cars/<id>/tuning/auto_map_race.yaml`, `pure_pursuit.yaml` (new) | D/E | per-car profile and law settings, each with `# why:` and the run or sweep it came from |
| `tools/f1tenth_sim/race_compare.py` (new) | E | gap_follow vs the map racer, seeds × directions, chassis contact |
| `docs/floor-test-map-racer.md` (new) | F | the floor test plan |

---

## Piece A: the simulator of record (branch `sim-of-record`)

### Task A1: A floor keeps the car's own values

**Files:**
- Modify: `tools/f1tenth_sim/sim_fidelity/calibration.py` (`Floor`, `FLOORS`, `with_floor`)
- Test: `tools/f1tenth_sim/tests/test_floor_car_split.py`

**Interfaces:**
- Produces: `Floor.measured_car: str` (`'rb2'` for `sfu_course_2026_10_30`), `Floor.floor_overrides: dict` (grip_curve, floor, map-wall values), `Floor.car_overrides: dict` (steering_linkage, steering_delay_ticks, speed_loop_*, speed_delay_ticks, speed_command_scale, odom_speed_scale, lidar_noise_*, lidar_dropout_rate, lidar_self_hits). `with_floor(name, base)` applies `floor_overrides` always, and `car_overrides` only when `base.car_id == floor.measured_car`, printing one line naming what it kept from the other car.

- [ ] **Step 1: Write the failing tests**
  - `test_floor_on_its_measured_car_is_what_it_was`: change-guard against the recorded measurement. `with_floor('sfu_course_2026_10_30', calibration_for_car('rb2'))` equals, field by field, the old single-dict override (copy today's dict into the test as the cited source, sim-calibration.md).
  - `test_floor_on_another_car_keeps_its_steering_and_speed`: `with_floor(..., calibration_for_car('rb1'))` has rb1's `steering_max_left == 0.401` (cars/rb1/car.yaml), no `steering_linkage` from rb2, `speed_command_scale == 1.0`, `odom_speed_scale == 1.0`, and rb2's `grip_curve`.
  - `test_every_floor_value_is_classified_once`: invariant over every `Floor` in `FLOORS`. The union of the two dicts' keys covers every key, and they don't intersect.
- [ ] **Step 2: Run, see them fail:** `RB_WS=<wt> rb python3 -m pytest tools/f1tenth_sim/tests/test_floor_car_split.py -v`. Then break `with_floor` (apply both dicts always) and confirm test 2 fails.
- [ ] **Step 3: Implement the split.** Keep `overrides` as a read-only property returning the merged dict, so existing callers are unchanged.
- [ ] **Step 4: Run the new tests and the whole tool suite green:** `rb python3 -m pytest tools/f1tenth_sim/tests/ -q`
- [ ] **Step 5: Commit** `sim_fidelity: a floor profile keeps another car's own steering and speed values`

### Task A2: Car 1's real laps

**Files:**
- Modify: `docs/sim-calibration.md` (new section "Car 1's real laps")

- [ ] **Step 1: Localize car 1's 30 Oct gap_follow runs on car 2's 30 Oct map:** `rb python3 tools/run_analysis/loc.py data/rb1/racerbot_runs/20261030-175418 ...-175701 ...-180531 ...-180746 ...-190734 --map data/rb2/racerbot_auto/20261030-185036-fixed`. Accept a run only at ≥ 0.90 of beams within 10 cm.
- [ ] **Step 2: Laps:** `rb python3 tools/run_analysis/laps.py <run> --line data/rb2/racerbot_auto/20261030-185036-fixed/raceline_raw.csv` for each accepted run.
- [ ] **Step 3: Car 1's 26 Oct course:** `loc.py` on `data/rb1/racerbot_runs/20261026-194514`, `-194714`, `-194913` with `--map data/rb1/racerbot_auto/20261026-194514-clean`, then `laps.py --line tools/f1tenth_sim/maps/car1_course_2026_10_26_centerline.csv`.
- [ ] **Step 4: Write the table** (run, fit, laps, median, max speed from the run's params) with the commands, and mark it "car 1 speed calibration as of c0c86b9".
- [ ] **Step 5: Commit** `docs: car 1's real gap_follow laps on both courses`

### Task A3: The no-ROS simulator reproduces gap_follow's real laps, per car and course

**Files:**
- Modify: `tools/f1tenth_sim/validate_calibration.py` (`--car` on `sim` and `fit-plant`, if absent)
- Test: `tools/f1tenth_sim/tests/test_validate_calibration_car.py`
- Modify: `docs/sim-calibration.md` (results table)

- [ ] **Step 1: Failing test** `test_sim_card_names_the_car_it_ran`: `validate_calibration.py sim --car rb1 ...` writes `car: rb1` in its JSON card, and its calibration has rb1's 0.401 left lock (cars/rb1/car.yaml). Break: ignore `--car`.
- [ ] **Step 2: Implement, run green.**
- [ ] **Step 3: Score three pairs, 6 seeds each:** rb2/`course_2026_10_30_scans`/`--params-from data/rb2/racerbot_runs/20261030-184826`; rb1/same track/`--params-from` the median rb1 run from A2; rb1/`car1_course_2026_10_26`/`--params-from data/rb1/racerbot_runs/20261026-194514`. All with `--floor sfu_course_2026_10_30`.
- [ ] **Step 4: Judge each against ±3%.** A miss is investigated with `trace_deficit.py` and `sectors.py` (superpowers:systematic-debugging) and fixed in the simulator or written down as a stated, signed difference. **For rb1 the fix may not touch rb1's calibration files.** A speed-scale miss on rb1 is recorded as "awaiting car 1's speed calibration".
- [ ] **Step 5: Commit** the table with every command.

### Task A4: racerbot_sim on the calibrated plant

**Files:**
- Modify: `src/racerbot_sim/racerbot_sim/sim_bridge.py` (`SimConfig.floor: str = ''`), `gym_bridge_node.py` (parameter `floor`), `src/racerbot_sim/launch/sim_bringup_launch.py`, `sim_race_launch.py`, `sim_auto_map_race_launch.py` (argument `floor`, default `''`)
- Test: `src/racerbot_sim/test/test_sim_bridge_floor.py`

**Interfaces:**
- Consumes: `sim_fidelity.with_floor`, `sim_fidelity.plant.FidelityPlant` (A1).
- Produces: `SimBridge(SimConfig(..., floor='sfu_course_2026_10_30', car='rb2'))` steps the same actuation, grip and odometry models as `FidelityPlant`. With `floor=''` its behaviour is unchanged.

- [ ] **Step 1: Failing tests** (rclpy-free, on `SimBridge` directly):
  - `test_no_floor_is_unchanged`: change-detector (labelled). 200 ticks of a fixed command script give the same states as before this task (record them before editing, in `test/data/`).
  - `test_floor_stop_matches_the_measured_stop`: recorded measurement. From 5.41 m/s on the wheels, a zero command stops in 3.39 m ± 0.10 m (run `20261030-175124`, sim-calibration.md).
  - `test_floor_steering_delay_is_two_ticks`: from the floor's `steering_delay_ticks = 2`. A steering step shows no yaw-rate response for 2 ticks, then does.
  - `test_floor_on_rb1_keeps_rb1_lock`: full-left command settles at 0.401 rad (cars/rb1/car.yaml).
- [ ] **Step 2: Run, see 2-4 fail.**
- [ ] **Step 3: Implement by reusing `FidelityPlant`'s actuation and grip components** (`sim_fidelity/actuation.py`, `grip.py`). Don't copy them. Keep `ttc_threshold = -inf` and `body_contact` as they are.
- [ ] **Step 4: Green, plus `rb python3 -m pytest src/racerbot_sim/test/ -q`** (sourced, domain 79). Compare the collected count with `grep -c '^def test'`.
- [ ] **Step 5: Commit**

### Task A5: Whole-stack gap_follow reproduces its real laps in racerbot_sim

**Files:**
- Modify: `tools/racerbot_sim/run_gap_follow_validation.py` (`--track`, `--track-directory`, `--floor`, `--car`, `--params-from RUN`, flying laps from `/sim/ground_truth_pose` against the track centreline)
- Test: `tools/racerbot_sim/tests/test_gap_follow_validation_laps.py` (pure lap-splitting on a synthetic ground-truth track: closed form, a circle at constant speed gives 2πR/v per lap; empty and single-lap input give no lap)

- [ ] **Step 1-2: Failing lap test, then implement, green.**
- [ ] **Step 3: Run the three pairs from A3,** 6 seeds, `--track-directory tools/f1tenth_sim/maps`.
- [ ] **Step 4: Judge each against ±3% of the real median (A2 / course-analysis.md).** This is the gate for piece E's racerbot_sim numbers. Record the result next to A3's.
- [ ] **Step 5: Commit, open PR A** (`sim-of-record`) with both tables in its body.

---

## Piece B: SLAM on the real bags (branch `slam-replay-both-cars`)

### Task B1: Replay every listed run on current `main`

**Files:**
- Modify: `docs/superpowers/specs/2026-11-02-map-racer-beats-gap-follow-design.md` (new "Results: SLAM" section)

- [ ] **Step 1: Whole launch, twice per run** (replays aren't deterministic): `rb python3 tools/run_analysis/replay_map_course.py <run>` for car 2's `20261030-184556`, `-184826`, `-185036`, `20261026-174911` and car 1's `20261026-174337`, `-180228`, `-194422`, `-194514`, `-194714`, `-194913`.
- [ ] **Step 2: Score:** `slam_score.py` against the reference map (car 2: `20261030-185036-fixed`; car 1: `20261026-194514-clean`, noting it's itself a SLAM map), plus `slam_report.py` for loop-closure jumps, and the `rebuild_run.py` verdict.
- [ ] **Step 3: Table per course:** walls within 10 cm, see-through %, largest correction, laps closed, heading at each closure, verdict. Car 1's gyro-off runs (`174337`, `180228`) are reported as "steering odometry, not tuned for".
- [ ] **Step 4: Decide.** Pass per course = GO verdict on both repeats, walls ≥ 0.85, no correction over 0.5 m, every closure heading under 15°. If every course passes, piece B ends here with the table (commit, PR). If not, B2.

### Task B2 (only if a course fails B1): a course-size rule for the SLAM profile

**Files:**
- Modify: `src/racerbot_launch/config/slam_gyro.yaml`
- Test: `src/racerbot_launch/test/test_slam_gyro_profile.py`

- [ ] **Step 1: Failing test** naming the failing course's measured node spacing, lap length and nearest parallel stretch (the B1 run name). The invariant: chain × spacing fits inside 2 × search radius, and the search radius is under that course's nearest parallel stretch. Must fail on today's values.
- [ ] **Step 2: Change only the values B1 implicates.** Re-run B1's replays on every course (a fix for car 1 must not break car 2's), twice each.
- [ ] **Step 3: Green, with all courses passing B1's criteria.** Commit with the before/after table. Open PR B.

---

## Piece C: localization on the real bags (branch `pf-on-real-bags`)

### Task C1: range_libc on the dev box, CPU only

**Files:**
- Create: `tools/run_analysis/build_range_libc_cpu.sh` (builds `src/range_libc/pywrapper` without CUDA into `~/.cache/racerbot/range_libc_cpu`, prints the `PYTHONPATH` line)

- [ ] **Step 1: Build:** `rb bash tools/run_analysis/build_range_libc_cpu.sh`. Working when `rb env PYTHONPATH=~/.cache/racerbot/range_libc_cpu python3 -c "import range_libc; print(range_libc.USE_CUDA)"` prints `0`.
- [ ] **Step 2: Commit.** The car keeps its CUDA build. Memory `range-libc-pywrapper-build` gets the CPU route added.

### Task C2: `pf_replay.py` and `pf_metrics.py`

**Files:**
- Create: `tools/run_analysis/pf_metrics.py` (no rclpy), `tools/run_analysis/pf_replay.py`
- Test: `tools/run_analysis/tests/test_pf_metrics.py`, `tools/run_analysis/tests/test_pf_replay_refuses.py`

**Interfaces:**
- Produces: `pf_metrics.errors(pf: PoseTrack, truth: PoseTrack, laser_offset_x: float = 0.26) -> ErrorTable` (per-sample position and heading error, truth interpolated to each pf stamp, pf moved from the LiDAR to the rear axle first); `pf_metrics.by_speed(table, edges=(0, 1, 2, 3, 4, 6)) -> list[SpeedBin]` (median, p95, max, count); `pf_metrics.delay(scan_stamps, pose_stamps) -> DelayStats`. `PoseTrack` = arrays `t, x, y, yaw`. `pf_replay.py RUN --map MAP [--odom odom|odom_imu] [--particles N] --out DIR` writes `pf_track.npz` and `pf_summary.json`.

- [ ] **Step 1: Failing tests**
  - `test_a_track_against_itself_has_zero_error`: invariant.
  - `test_a_known_offset_is_measured_exactly`: closed form. truth shifted 0.10 m in x gives 0.10 m position error ± 1e-9 m.
  - `test_the_laser_offset_is_removed_along_the_heading`: closed form. pf at the LiDAR with yaw 90° and truth at the rear axle give zero error.
  - `test_truth_is_interpolated_not_nearest`: closed form, linear motion.
  - `test_heading_error_wraps`: 179° against −179° is 2°.
  - edges: empty tracks give an empty table; NaN poses are excluded and counted; one sample gives one row; pf stamps outside the truth's span are dropped, not extrapolated.
  - `test_pf_replay_refuses_beside_drivers`: with a fake process table holding `urg_node` it exits non-zero naming it (reuses `slam_replay`'s `DRIVER_MARKERS` check).
- [ ] **Step 2: See them fail; implement; green.** Break checks: drop the laser offset, nearest-neighbour instead of interpolation.
- [ ] **Step 3: Commit.**

### Task C3: Measure, and point the filter at gyro odometry if it's better

**Files:**
- Modify: `src/racerbot_launch/launch/race_launch.py` (`pf_odom:=auto|odom|odom_imu`, default chosen by this task's numbers; `auto` uses `/odom/imu` when `vehicle_state_node` is up and falls back to `/odom` with a log line)
- Test: `src/racerbot_launch/test/test_race_launch_pf_odom.py`

- [ ] **Step 1: Truth tracks:** `loc.py` on car 2's `20261030-184556`, `-184826`, `-185036`, `-175728` (up to 5.4 m/s) with the 30 Oct map, and car 1's `20261026-194514`, `-194714`, `-194913` with `194514-clean`.
- [ ] **Step 2: Replays, `--odom odom` and `--odom odom_imu`, shipped `max_particles` 4000:** a table by speed bin. Position median/p95/max, heading p95, and delay (dev-box CPU, labelled not-Jetson).
- [ ] **Step 3: Failing launch tests:** `test_pf_reads_odom_imu_when_vehicle_state_runs`, `test_pf_falls_back_to_odom_without_vehicle_state` (Review Focus 2), `test_explicit_odom_is_obeyed`. Each asserts the remap the launch description builds. Break: hardcode `/odom`.
- [ ] **Step 4: Implement. The default is `auto` only if `odom_imu` beats `odom` on p95 position in every speed bin on every run; otherwise `odom`.** Green.
- [ ] **Step 5: Write the measured noise and delay into `sim_fidelity/calibration.py`** (`pose_noise_xy`, `pose_noise_yaw`, `pose_delay_ticks`) with provenance naming the runs, replacing the assumed values. Update the change-guard tests that pin them, flagging each changed expectation in the commit body (it's a spec change with a recorded measurement, not a weakening). Re-run A3's three pairs to confirm gap_follow's laps don't move.
- [ ] **Step 6: Commit, open PR C.**

---

## Piece D: the line, its speed and its steering (branch `raceline-from-map`)

Depends on A (simulator) and C (tracking noise). Its margins come from measured numbers.

### Task D1: Grip from the real runs

- [ ] **Step 1:** `rb python3 tools/run_analysis/gripcurve.py <run>` on car 2's fit runs (`SFU_COURSE_FIT_RUNS`) and car 1's `20261026-194514`, `-194913` and the A2 rb1 30 Oct runs.
- [ ] **Step 2: Per car,** the largest asked lateral acceleration whose achieved share is ≥ 0.90 becomes that car's starting `profile_max_lateral_accel`. Record the table in the design's results.

### Task D2: One line builder for both race routes

**Files:**
- Create: `src/pure_pursuit/pure_pursuit/race_line.py`
- Modify: `tools/run_analysis/rebuild_run.py` (`_write_lines` calls it), `src/pure_pursuit/pure_pursuit/auto_map_race_node.py` (race-phase profile calls it), `src/pure_pursuit/pure_pursuit/raceline_optimizer.py` (per-point margin)
- Test: `src/pure_pursuit/test/test_race_line.py`

**Interfaces:**
- Produces: `build_race_line(grid: OccupancyMap, seed_xy: np.ndarray, cfg: RaceLineConfig) -> RaceLine`. `RaceLineConfig` has: car_width, safety_margin, wall_clearance, max_steering_angle, wheelbase, v_max, v_min, a_lat_max, a_accel_max, a_brake_max, smoothing_passes, plus the optimizer's knobs, read from the same `auto_map_race.yaml` keys. `RaceLine` has: xy, speed, source (`'optimized'|'recorded'`), est_lap_s, min_clearance_m, pinch_points (indices where the margin was reduced), reasons (list of str).
- `safety_margin` default = measured p99 cross-track at race speed (D3/E) + 0.02 m map-wall error (sim-calibration.md, +1.0/+2.4 cm). Until D3 measures it, the existing `optimize_safety_margin` 0.10 is kept.

- [ ] **Step 1: Failing tests**
  - `test_every_point_keeps_the_wall_clearance`: invariant on `course_2026_10_30_scans` (and a synthetic oval). Clearance along the line ≥ `wall_clearance` everywhere.
  - `test_a_pinch_point_gets_the_centre_not_a_refusal`: a synthetic corridor with one 0.7 m neck (narrower than 0.31 + 2 × 0.10 + 0.20). The line passes through the neck's centre ± 0.02 m, `pinch_points` names it, and clearance elsewhere still meets the margin (Review Focus 1).
  - `test_the_faster_candidate_wins`: closed form on a circle of known radius. Optimized and recorded candidates get `est_lap_s` from `racing_math`, and `source` is the lower.
  - `test_both_routes_build_the_same_line`: `rebuild_run` and `auto_map_race_node`'s profile function, given the same grid, seed and config, return equal `xy` and `speed` to 1e-9.
  - `test_a_line_through_a_wall_is_refused`: a seed crossing an occupied cell, with `reasons` naming it. Keeps the existing hard refusal.
- [ ] **Step 2: See them fail; implement; green.** Mutation checks: flip the clearance comparison, drop the pinch fallback.
- [ ] **Step 3:** Run `src/pure_pursuit/test/` whole (sourced, domain 79), `auto_map_race`'s four deny-path tests included, and `tools/run_analysis/tests/`. Check the collected counts.
- [ ] **Step 4: Commit.**

### Task D3: Steering law and lookahead on the real courses

**Files:**
- Modify: `tools/f1tenth_sim/law_sweep.py`, `law_compare.py` (`--track`, `--floor`, `--car`, `--line`; report corner cross-track p95, over points with |κ| > 0.5 /m, alongside the mean)
- Test: `tools/f1tenth_sim/tests/test_law_compare_corner.py` (corner p95 on a synthetic trace: closed form; no corner points gives an explicit "no corners", not 0)

- [ ] **Step 1-2: Failing test, implement, green.**
- [ ] **Step 3: Tune on seeds 1000-1002,** each car and course, on D2's line with C3's measured pose noise. Candidates: pure pursuit with lookahead `min/max/gain` swept, Stanley, LQR, MPC.
- [ ] **Step 4: Judge once on seeds 12345-12349** with the unchanged promotion rule. The measured p99 cross-track of the winner becomes `safety_margin` (D2), and D2's line is rebuilt with it.
- [ ] **Step 5: Commit the table, open PR D.**

---

## Piece E: head-to-head and shipped defaults (branch `head-to-head-real-courses`)

### Task E1: `race_compare.py`

**Files:**
- Create: `tools/f1tenth_sim/race_compare.py` (candidates as named settings; tracks × cars × seeds × directions; chassis contact via `sim_fidelity`; JSONL rows, `--summary`)
- Test: `tools/f1tenth_sim/tests/test_race_compare_summary.py`

**Interfaces:**
- Produces: `summarise(rows: list[dict]) -> list[Verdict]`, where `Verdict` = candidate, car, track, laps, median_lap_s, min_clearance_m, contacts, disqualified: bool, reason.

- [ ] **Step 1: Failing tests:** `test_any_contact_disqualifies_the_candidate` (Review Focus 4: fastest median plus one contact gives disqualified), `test_no_lap_is_not_a_win`, `test_the_fastest_clean_candidate_is_named`, `test_a_seed_missing_from_one_candidate_is_reported`.
- [ ] **Step 2: Implement, green. Commit.**

### Task E2: Sweep, then the final tables

- [ ] **Step 1: No-ROS sweep, 6 seeds, `--workers 2`.** Per car and course, over `profile_max_lateral_accel` (from D1 up to the floor's 3.75), `profile_max_brake` (3.0 up to the floor's 4.5), `profile_max_speed` 4-5.5, `safety_margin`, and D3's law. gap_follow candidates: shipped default and time trial (`aim_depth_bias 1.0`, awareness off, lateral 3.25, steering at that car's limit).
- [ ] **Step 2: Final, 30 seeds × both directions,** for each car's fastest clean candidate against both gap_follow candidates and the real laps.
- [ ] **Step 3: Pass = beats gap_follow's real median lap on that car and course with zero contacts** (and gap_follow's best simulated setting, if the tech lead requires it). Record it in the design's results.

### Task E3: The whole stack in racerbot_sim

- [ ] **Step 1:** `tools/racerbot_sim/run_map_course_validation.py --track course_2026_10_30_scans --track-directory tools/f1tenth_sim/maps --floor sfu_course_2026_10_30 --car rb2` (add the three flags if missing, and set the range method so the CPU range_libc is accepted in simulation only), then the same for rb1 and for `car1_course_2026_10_26`. gap_follow maps, `rebuild_run.py` builds D2's line, and `race_launch.py run_dir:=` races it with the real particle filter and `pure_pursuit_node`.
- [ ] **Step 2:** `tools/racerbot_sim/run_auto_map_validation.py --scenario solo --track <each real course> --floor ... --car ...`.
- [ ] **Step 3: Flying laps from ground truth, contact from `body_contact`.** Only A5-validated pairs count. Pass = E2's criterion. A gap between E2 and E3 bigger than 3% is investigated before shipping (it's a wiring or node-vs-harness difference, sim-fidelity-audit F14).

### Task E4: Ship the winners as defaults

**Files:**
- Create/modify: `src/racerbot_car/cars/<id>/tuning/auto_map_race.yaml`, `.../tuning/pure_pursuit.yaml` (per-car values, each with `# why:` naming the E2/E3 table), shared defaults in `src/pure_pursuit/config/*.yaml` where both cars agree
- Test: `src/racerbot_launch/test/test_car_profiles.py` (each tuning value present and in range; `enable_deadman` and the safety values unchanged, checked against today's numbers)
- Docs: `docs/racing-autonomy.md` (narrow-course section rewritten with the new numbers), `docs/steering-laws.md` if the law changes, `docs/sim-calibration.md`, `CHANGELOG.md`

- [ ] **Step 1: Failing profile tests, implement, green** (full `src/racerbot_launch/test/`, `src/pure_pursuit/test/`).
- [ ] **Step 2: Commit, open PR E** with the final tables.

### Task E5: Re-run rb1 once car 1's speed calibration lands

- [ ] **Step 1:** When the other session's rb1 calibration change is merged, rebase A-E onto it and re-run A2, A3, A5, E2 (final only) and E3 for rb1.
- [ ] **Step 2:** Update rb1's tuning and tables. rb1 results stop being provisional only here.

---

## Piece F: the floor test plan (branch `floor-test-plan`)

### Task F1: `docs/floor-test-map-racer.md` and the car hand-off prompt

Use the `racerbot-docs-voice` and `novice-docs` skills.

- [ ] **Step 1: Per car, in order, each step with its command, "working when", abort signs and `laps.py`/`loc.py` afterwards:**
  1. Pull, build, run the tests (hand-off prompt, memory `car-handoff-prompt`), `racerbot_doctor`.
  2. Map: `map_course` (gap_follow, 3 laps), rebuild verdict GO, `check_start` GO.
  3. Wheels off, LB held: `race_launch.py run_dir:=...` at the 1.5 m/s file. Steering follows the line as the car is carried round, and releasing LB zeroes `/drive`. Measure particle filter delay on the Jetson here (C2's tool on that run's bag).
  4. Floor, low speed: 1.5 m/s file, 3 laps, a person holding LB. Abort on cross-track over 0.3 m (`/drive_intent`) or any `body_contact`.
  5. Floor, 2.5 m/s, then the shipped race file. Compare each run's laps with the simulator's prediction from E. Stop if real laps are more than 5% slower than predicted, and replay the run before going faster.
  6. `auto_map_race_launch.py` end to end, same steps.
- [ ] **Step 2: Fix the 2026-07-27 discrepancy** (git-setup.md says a `gap_follow` collision; localization.md blames pure pursuit) from the CHANGELOG and git history, correcting whichever doc is wrong.
- [ ] **Step 3: Commit, open PR F.**
