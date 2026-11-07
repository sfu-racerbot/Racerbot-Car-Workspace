# Map-based racers, competitive with gap_follow: implementation plan (revision of 2026-11-07)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Who this is for:** whoever builds this work, human or agent.
> **Read first:** [the 2026-11-02 design](../specs/2026-11-02-map-racer-beats-gap-follow-design.md) (still the constraints and anti-goals), [the 2026-11-02 plan](2026-11-02-map-racer-beats-gap-follow.md) (its pieces B, C and E are still wanted; this plan reorders the work), `CLAUDE.md`, `TEST_QUALITY_STANDARDS.md`.
> **You'll be able to:** fix what made the map-based racers lose on 2026-11-06, in the order that buys the most, and know from each test or measurement that it is done.

**Goal:** on car 2's courses, the map-based racers (`race_launch.py` and `auto_map_race`) lap significantly faster than gap_follow, race other cars, start from anywhere on the course, and `map_course` ends a run without crashing.

**Spec, revised by the tech lead on 2026-11-07:**
1. `map_course` must not cut the controls when mapping ends. It brakes the car to a stop under control.
2. The car can be put anywhere on the course, and the map-based racers localize there. It doesn't have to be the spot where `map_course` stopped.
3. Opponent detection and overtaking are part of the map-based racers.
4. The map-based racers run the course significantly faster than any other algorithm, and race properly against other cars.

## What 2026-11-06 showed (car 2, `main` 86e9ad8)

Runs copied from car 2 into `data/rb2/racerbot_runs/` (not committed). Located with `loc.py` and `laps.py` on that day's rebuilt maps.

| Run | Algorithm | Map | True laps |
|---|---|---|---|
| `170803`, `171030` | gap_follow | `164002-rebuilt`, 32.6 m | **11.3-12.0 s** (14 laps) |
| `164401` | pure pursuit | same | 21.6, 21.7 s |
| `175300` | pure pursuit | same | none: stalled by a false overtake |
| `180114` | MPC | `175837-rebuilt`, 31.6 m | none: stalled, steering saturated |
| `180341` | LQR | same | 19.5-21.8 s |

The causes, largest first:

1. **Every racer ran a 1.5 m/s profile.** `rebuild_run.py` writes `raceline_profiled.csv` at 1.5 m/s (a deliberate first-run cap), and `race_launch.py run_dir:=` loads that file. `intent_report.py`: profile speed held the car down 100% of the time on `164401` and `180341`.
2. **The line is gap_follow's path.** `rebuild_run` paces the recorded lap; it never runs the optimizer. Even its 4 m/s file estimates 11.8-12.6 s, a tie with gap_follow on paper with perfect tracking.
3. **Not enough steering.** The course needs curvature up to 1.07 /m (0.93 m radius). The planner allows 0.859 /m (0.30 rad) and the controller 0.349 rad, so the line from every `map_course` run was refused, and MPC sat at +0.349 rad through the bend at 1.5 m/s until `body_contact` stopped it (`180114`, 20.1-25.5 s). Car 2's measured full stick turns 0.423 rad both sides (sim-calibration.md), and its servo reaches 0.448 right and 0.509 left.
4. **A false opponent.** `175300` is a solo run, yet pure pursuit committed `overtake_left` at 7.5 s, steered into the wall, and `body_contact` latched until LB was released.
5. **`body_contact` stops hold until LB is released** (1.7-5.0 s stalls).
6. **`map_course` ends with an all-zero command.** `auto_map_race_node._publish_stop()` sends speed 0 **and steering 0** from the tick the last lap closes. In 4 of 5 runs gap_follow then logged the body inside a wall (clearance -0.009 to -0.146 m) about 1 s later, while asking for -0.28 to -0.35 rad.
7. **Start pose only where mapping stopped.** `seed_start_pose` seeds the particle filter from `start_pose.yaml`. `rebuild_run` warned three times that the car stopped 0.4-0.9 m off the line.

Also seen, not on the critical path: `slam_toolbox` failed to save the occupancy map on 3 of 5 runs (`Failed to spin map subscription`; the offline rebuild saved them), and gap_follow, `vehicle_state_node` and `imu_bridge_node` die with a pybind11 `RuntimeError` at shutdown.

## Global Constraints

The 2026-11-02 plan's Global Constraints all apply. In particular: the LB deadman and each node's four deny-path tests stay; `enable_deadman`, TTC, swept clearance, `emergency_stop_clearance`, `emergency_stop_distance`, `body_contact`'s threshold and `profile_wall_clearance` are never relaxed; contact is judged only by `sim_fidelity`'s chassis check or `SimBridge.body_contact`; every new test is seen to fail first; no test is weakened; ROS and simulators run via `RB_WS=<worktree> rb`; simulations at `--workers 2`; car 1's calibration files are not edited; no floor runs from this work.

One addition:
- **Steering limits are physical values.** A raised limit goes in that car's `calibration/` (car 2 only here), cites the measurement, and must pass `test_steering_limits_fit_servo.py`. Car 1's stay as they are.

## Review Focus

1. **The final lap closes on a straight with a wall close ahead,** and the controlled stop needs more room than there is: the stop must still end at zero, with gap_follow's emergency tiers in charge throughout, never "keep driving because the ramp says so".
2. **LB released during the controlled stop:** zero immediately, as everywhere else.
3. **A symmetric course** (an oval looks the same turned round): global localization must refuse when two distinct poses fit almost equally well, never pick one.
4. **A real car parked on the line at the start, or another car near the start:** global localization must still find the ego (map-subtraction residuals don't count as walls), and a stationary car ahead is an opponent, not a reason to refuse.
5. **A wall fragment the size of a car** (the 2026-11-06 false overtake): the overtake decision must require a track-relative, moving or persistently placed object off the walls, not a single scan's cluster.

---

## Task 1: `map_course` brakes to a stop, steering kept

**Branch:** `map-course-controlled-stop`

**Files:**
- Modify: `src/pure_pursuit/pure_pursuit/auto_map_race_node.py` (new state `stopping` between the last lap and `loading_profile` when `finish_after_mapping`, also before the racing handover)
- Modify: `src/pure_pursuit/config/auto_map_race.yaml` (`finish_stop_decel: 2.0` m/s², `finish_stop_timeout_sec: 4.0`, `finish_stop_speed_epsilon: 0.05` m/s)
- Test: `src/pure_pursuit/test/test_auto_map_race_controlled_stop.py`

**Behaviour:** on the closing lap, record `v0` = the last forwarded gap_follow speed. Each tick, forward gap_follow's command with speed `min(gap_follow speed, max(0, v0 - finish_stop_decel * t))` and **gap_follow's steering unchanged**. If gap_follow's command is stale or missing, publish zero (the existing `_command_status` path). Once the ramp reaches zero, or after `finish_stop_timeout_sec`, go to `loading_profile`, which publishes zeros as today. The map save may start during `stopping` (gap_follow doesn't use the SLAM pose).

- [ ] **Step 1: Failing tests** (node-level, `drive_topic:=/test_only/drive`):
  - `test_stop_keeps_gap_follow_steering`: closing at 3.0 m/s with gap_follow steering -0.30 rad, the next published command has steering -0.30 and speed ≤ 3.0 - 2.0 × dt (closed form).
  - `test_stop_reaches_zero_by_the_ramp`: commands fall by at least `finish_stop_decel × dt` per tick and reach 0 by `v0 / decel + 2 ticks` (closed form, 1.5 s from 3.0 m/s).
  - `test_stop_never_exceeds_gap_follow`: gap_follow asks 0 (its emergency tier) mid-ramp, and the published speed is 0 that tick.
  - `test_lb_release_during_stop_is_zero_now` (Review Focus 2): speed and steering 0 on the next tick.
  - `test_stale_gap_follow_during_stop_is_zero`.
  - `test_stop_times_out_to_zero`.
  - The node's existing four deny-path tests, re-run.
- [ ] **Step 2: Watch them fail** against today's code (steering 0 on the first stopping tick).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Green**, plus all of `src/pure_pursuit/test/` sourced at domain 79, with the collected count checked.
- [ ] **Step 5: Simulator check.** `tools/racerbot_sim/run_map_course_validation.py --track indoor_oval`: `body_contact` never fires after the last lap closes. Commit, PR.

## Task 2: Localize from anywhere on the course

**Branch:** `localize-anywhere`

**Files:**
- Modify: `src/pure_pursuit/pure_pursuit/map_localize.py` (`find_pose_anywhere(field, bx, by, ...) -> GlobalFix` with `pose`, `fit`, `runner_up_fit`, `runner_up_distance_m`, `ok`, `reason`)
- Modify: `src/pure_pursuit/pure_pursuit/seed_start_pose.py` (`mode: anywhere|start_pose`, default `anywhere` with `run_dir`; waits for 5 consecutive scans that agree within 0.10 m / 3°, seeds `/initialpose` with that pose, exits 0. Refusal exits non-zero, saying why and what to do)
- Modify: `src/racerbot_launch/launch/race_launch.py` (`start:=anywhere|start_pose`, default `anywhere`), `src/pure_pursuit/pure_pursuit/check_start.py` (reports the global fix)
- Test: `src/pure_pursuit/test/test_map_localize_anywhere.py`, `src/pure_pursuit/test/test_seed_anywhere.py`

**Algorithm:** `find_start` over every free cell more than `car_width/2` from a wall (no `near` prior), 0.10 m / 5° steps, `match_scan` refinement of the best 10. Accept when fit ≥ 0.85 **and** the best distinct candidate (> 1.0 m or > 30° away) fits at least 0.08 worse. Scan points more than 0.3 m from every wall are ignored in the fit (Review Focus 4).

- [ ] **Step 1: Failing tests:**
  - On the recorded `20261106-164002-rebuilt` map with real scans from run `164401` at 5 timestamps spread round the lap: pose within 0.15 m / 3° of `loc.py`'s truth (recorded measurement; fixture scans saved under `test/data/`).
  - A synthetic oval (point-symmetric): `ok is False`, and `reason` names ambiguity (Review Focus 3).
  - The same real scan plus a synthetic 0.3 × 0.5 m box 1 m ahead: still within 0.15 m.
  - A scan from a different map (`20261030-185036-fixed`): `ok is False`, fit too low.
  - Empty scan, all-`inf`, NaN beams, wrong beam count: `ok is False` with a reason, never a raise.
- [ ] **Step 2: Watch them fail. Step 3: Implement. Step 4: Green.**
- [ ] **Step 5: racerbot_sim check:** `race_launch.py run_dir:=` with the car spawned 10 m along the course from the saved start; the racer starts and laps with no contact. Commit, PR.

## Task 3: A racing line that can beat gap_follow

**Branch:** `raceline-from-map`

**Files:**
- Modify: `src/racerbot_car/cars/rb2/calibration/pure_pursuit.yaml` (`max_steering_angle` 0.349 → 0.42) and add `.../auto_map_race.yaml` (`profile_max_steering_angle` 0.30 → 0.39), citing "full stick turns 0.423 rad both sides, sim-calibration.md; servo limits 0.448 R / 0.509 L, 2026-10-16 circles". The planner stays below the controller so the controller has steering left to correct with.
- Create: `src/pure_pursuit/pure_pursuit/race_line.py` (the 2026-11-02 plan's D2 interface: `build_race_line(grid, seed_xy, cfg) -> RaceLine`, optimizer from the map's centreline with per-point margin and pinch fallback, the faster of optimized and recorded)
- Modify: `tools/run_analysis/rebuild_run.py` (uses `race_line`; writes `raceline_race.csv` at the validated race settings beside the staged 1.5 / 2.5 m/s files; `commands.txt` lists them in order)
- Modify: `auto_map_race_node.py` (race phase uses `race_line`)
- Test: `src/pure_pursuit/test/test_race_line.py` (the 2026-11-02 plan's D2 tests), `src/racerbot_launch/test/test_steering_limits_fit_servo.py` (passes with the new rb2 values, fails at 0.46)

- [ ] **Step 1: Import the 2026-11-06 courses as simulator tracks:** `tools/f1tenth_sim/track_import.py` for `20261106-164002-rebuilt` (32.6 m) and `20261106-184753-rebuilt` (22 m) → `car2_course_2026_11_06a`, `car2_course_2026_11_06b`.
- [ ] **Step 2: Validate the simulator there.** Simulated gap_follow on `_06a`, with run `171030`'s params, car 2, floor `sfu_course_2026_10_30`: median lap within ±3% of the real 11.6 s. A miss is investigated (`trace_deficit.py`, `sectors.py`) before anything else counts.
- [ ] **Step 3: Tests first, then `race_line.py`,** as in the 2026-11-02 plan's D2.
- [ ] **Step 4: Sweep in the no-ROS simulator, 6 seeds,** on both tracks: steering law (pure pursuit, LQR, Stanley, MPC), lookahead, `profile_max_lateral_accel` 2.5-3.75, `profile_max_brake` 3.0-4.5, top speed 4-5.5. Measured pose noise where task C of the 2026-11-02 plan has it, otherwise the 2 cm default. Pass: zero contacts on every seed and a median lap at least **10% faster** than simulated gap_follow's (the tech lead asked for significantly faster).
- [ ] **Step 5: Ship the winner** as `raceline_race.csv`'s settings and the shared defaults, with `# why:` citing the sweep. Commit, PR.

## Task 4: Opponents: no false overtakes, real overtakes

**Branch:** `map-racer-opponents`

**Files:**
- Modify: `src/pure_pursuit/pure_pursuit/pure_pursuit_node.py` opponent path, reusing `gap_follow/opponent_tracking.py`'s multi-car tracker (already shipped and validated in the simulator) instead of the single-cluster detector, behind map subtraction
- Test: `src/pure_pursuit/test/test_opponent_integration.py`, plus a replay test on the 2026-11-06 false overtake

- [ ] **Step 1: Replay fixture:** the scans, poses and map of `175300` around 7.0-8.5 s. Test: no overtake is committed on them (the false positive from 2026-11-06).
- [ ] **Step 2: Watch it fail on today's code.** Then the overtake requires a tracked object that persists for ≥ 0.3 s, lies ≥ 0.25 m from every mapped wall, and is on or within one car width of the racing line ahead (Review Focus 5).
- [ ] **Step 3: The simulator's traffic scenario** on `car2_course_2026_11_06a` with one and two slower scripted cars (racerbot_sim `sim_race_launch.py`, and `run_validation.py --scenario traffic`): zero car contact, zero wall contact, passes completed, lap time reported. Commit, PR.

## Task 5: `body_contact` recovers instead of waiting for LB

**Branch:** combined with Task 4 if the false overtake turns out to cause every latch; otherwise `body-contact-recovery`

- [ ] **Step 1:** Measure from the 2026-11-06 stalls whether `body_contact_escape` was refused by its directional gate, its clearance, or never tried.
- [ ] **Step 2:** If it was refused only because the contact was at the flank while steering into the wall, add a reverse-free recovery that steers *away* from the contact side at the escape crawl speed. It's gated by the same clearance check, never ahead of contact or stale-scan stops, and never past `body_contact`'s threshold. Its tests include the deny side: no escape when boxed in.

## Task 6: Race-day head-to-head and the floor plan

- [ ] Whole stack in racerbot_sim on `_06a` and `_06b`: `map_course` → rebuild → `race_launch.py run_dir:=` from a random spot (Task 2) → race, solo and with traffic. Laps from ground truth, contact from `body_contact`.
- [ ] `docs/floor-test-map-racer.md`, per the 2026-11-02 plan's piece F, using the staged files: 1.5 m/s, then 2.5, then `raceline_race.csv`. A hand-off prompt for car 2's Claude Code.
