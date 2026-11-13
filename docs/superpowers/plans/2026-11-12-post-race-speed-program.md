# Post-Race Speed Program Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make both cars lap faster and stop getting stuck, using what the 2026-11-11 race against UVic measured: fix the wall-alongside latch, make the map racer (pure pursuit) use the grip the cars really have, make the simulator match race day, bring gap follow up to the disparity extender's pace, then let the cars adapt to grip while driving.

**Architecture:** Five phases, in the order the tech lead set (the stuck fix first, then pure pursuit, then the rest). Every behaviour change is a pure function in a ROS-free module (`gap_logic.py`, `racing_math.py`) with closed-form tests, wired into the node behind a parameter whose default keeps today's behaviour, proven on recorded race-day scans and in the simulator, and only then handed to the tech lead for a floor test. Sonnet never drives a car.

**Tech Stack:** ROS 2 Jazzy, Python 3.12, numpy, pytest; `tools/run_analysis/` (bag analysis), `tools/f1tenth_sim/` (controller-math simulator), `racerbot_sim` (ROS-level simulator).

**Spec:** this document. Sections 1 to 3 are the spec (evidence, decisions, rules); section 4 onward is the plan. There is no separate design file.

**Roles:** Sonnet implements. Fable is the advisor (`/advisor` in the Sonnet session). The tech lead (Bryan) runs every floor test and approves every merge.

---

## 1. What race day measured

All numbers below are from the runs named. True position and speed come from `tools/run_analysis/loc.py` (LiDAR scan matching against the race-day map), not from the wheels.

Race-day map: `data/rb2/racerbot_auto/20261111-151026-rebuilt` (course line 28.17 m).

### 1.1 Lap times, same course

| Car | Algorithm | Run | Typical lap | Best pace |
|---|---|---|---|---|
| rb2 | pure pursuit, `steering_law:=delay_lqr` | `rb2 20261111-152132` | 9.9 to 10.2 s, 13 laps in a row | 9.3 s |
| rb2 | disparity extender (team A), testing | `rb2 20261111-142604` | 10.3 to 11.0 s, 20 laps | 9.7 s |
| rb2 | disparity extender, final race | `rb2 20261111-154204` | 10.6 to 16.1 s (traffic) | 10.1 s |
| rb2 | gap follow, `time_trial:=true` | `rb2 20261111-135934` | 11.8 to 12.5 s, with 15 s laps | 11.9 s |
| rb1 | gap follow, default | `rb1 20261111-153721` | 10.8 to 12.2 s | 9.6 s |
| rb1 | gap follow, default, final race | `rb1 20261111-155542` | 11.7 to 20.2 s | 10.3 s |

Pure pursuit was the fastest and most consistent thing run that day. Its planner estimated 9.48 s; it drove 10.05 s.

### 1.2 The wall-alongside latch (both controllers)

- **gap follow, rb1 final race, 87.6 to 95.9 s.** Wall 0.20 to 0.21 m from the LiDAR on the left (4 to 5 cm off the body), 0.82 m free ahead, car steering right (away). `gap_logic.swept_path_clearance` returned 0.00 for every steering angle from -0.33 to +0.33 rad. Its band is `car_width/2 + swept_clearance_margin` = 0.205 m, and a return inside the band alongside the body gives `along < 0`, which `max(0.0, nearest)` turns into 0. `braking_speed_limit(0, 0.12, ...)` is 0. The creep branch needs `forward_clearance <= forward_stop_clearance`, and the cone read 0.82 m. So the speed stayed at 0 for 8.3 s until the car was moved by hand.
- **gap follow, rb2, `20261111-142254`, 104.6 to 118.2 s:** same factor (`clearance cap`), 13.6 s.
- **pure pursuit, rb2, `20261111-152132`, 19.0 to 49.5 s:** `body_contact` with the wall 0.18 m from the LiDAR on the right (2.5 cm off the body), 0.63 m free ahead. `_reactive_override` Tier 0 stops for good when the closest contact is on the flank (not "directional"). 30.5 s, until moved by hand.

### 1.3 Grip and understeer

- `gripcurve.py` (asked = true speed squared times commanded curvature; achieved = true speed times gyro), rb2 pure pursuit run: asked 3.0 to 3.5 gives 2.84 m/s² achieved, 4 to 5 gives 3.05, 6 to 8 gives 3.25. The car delivers less curvature than the steering asks for, more so the harder it is asked.
- Same tool, rb2 disparity run: 4 to 5 gives 3.34, 6 to 8 gives 4.31. rb1 gap follow: 4 to 5 gives 3.58, 6 to 8 gives 4.63, 8 to 12 gives 5.92.
- Accelerometer cross-check (`linear_acceleration.y`, 0.2 s average, above 1.5 m/s): rb2 pure pursuit p95 3.83 m/s² on the VESC IMU (`/imu/vesc/data`) and 4.0 on the BNO085 (`/imu/data`); rb1 gap follow p95 6.0 m/s² (`/imu/data`).
- **Open question, Task 5 answers it:** the sustained grip limit is somewhere between 3.2 and about 4.5 m/s² on rb2 and may be higher on rb1. The high readings on the reactive runs may be slides (yaw without path curvature), not usable grip. Nothing in Phase 2 may assume a number above 3.2 until Task 5 measures it.
- **Measured by `grip_truth.py` (Task 5, steering window 0.10 rad, gyro basis):** rb2 (runs 152132, 135934, 142254, `/imu/vesc/data`) delivers 2.94 m/s² at a kinematic ask of 3.75 (bin 3.5-4.0) and falls beyond it (2.71 at 4-5, 2.80 at 5-6): the tyre is past its peak, steering more loses grip. rb1 (runs 153721, 155542) delivers 3.56 at an ask of 4.5 (bin 4.0-5.0, n = 53) and has not shown its peak; accelerometer over gyro is 1.08 on rb2 and 1.26 on rb1. Files: `tools/f1tenth_sim/sim_fidelity/sfu_course_2026_11_11_grip_rb{1,2}_vesc.json`.
- Pure pursuit plans at `profile_max_lateral_accel: 3.2` (rb2 calibration) and caps online at `max_lateral_accel: 3.1`. `lap_theory.py` on the path it drove: 9.78 s at 3.0, 9.18 s at 3.5, 8.70 s at 4.0. On the shortest safe line: 9.03 s at 3.0, 8.23 s at 4.0.
- `/drive_intent`: pure pursuit's speed was bound by its online `curve cap` 49% of the time, the profile 32%, the reactive layer 18%. The curve cap is computed from the *commanded* steering curvature (`_shape_normal_command`), which is larger than the curvature the car achieves.

### 1.4 Other findings

- **Pure pursuit's standing-start lap ran wide.** `rb2 20261111-152132`, 3.3 to 8.4 s: from rest it reached 4.9 m/s, entered a left-hander, the reactive layer cut in at 8.1 s and it stopped against the right wall at 8.4 s. The 13 flying laps after it were clean.
- **rb1's final race:** `follow cap` (opponent awareness) bound the speed 19% of the time and `curve cap` 49%. rb1 ran the default profile (`max_lateral_accel: 2.5`, steering limit 0.33 rad), rb2 ran `time_trial:=true` (3.0, 0.42 rad).
- **gap follow, rb2 time trial:** `clearance cap` (the swept-path braking ceiling) bound 27 to 36% of the time, `curve cap` 41 to 49%. The disparity extender has no braking ceiling and ran with `safety_node enable_ttc:=false`.
- **Measured response** (`car_measure.py`): rb2 servo 3 ticks delay, gain 15/s, rate 1.75 rad/s; speed loop 5 ticks delay, accel max 4.0, brake max 3.0 m/s². rb1 servo 3 ticks, gain 12/s, rate 1.75; speed loop 5 ticks, accel max 5.0, brake max 4.0. Wheel speed over true: rb2 1.019, rb1 1.042.
- **IMU topics exist only in some runs.** `imu_bridge_node` and `vehicle_state_node` ran for the map racer and for our gap follow (`152132`, `135934`), and both IMUs read fine there. Team A's launches (`142604`, `154204`) do not start them, so those bags hold only `/sensors/imu/raw` (the VESC's raw IMU, in g and deg/s, sensor axes). Any tool that needs calibrated IMU data on a team A run must convert the raw topic with the car's mount from `src/racerbot_car/cars/<id>/imu/car.yaml`.

## 2. Decisions the tech lead made (2026-11-12)

1. Next race is over a month away. Do all five phases; prioritize pure pursuit after the stuck fix.
2. Software changes, grippier tyres, and a higher VESC top speed are all allowed.
3. Light wall taps during floor testing are acceptable. Racing config still ships zero-contact in the simulator.
4. "Maximum speed for most of the lap" is not reachable on a 28 m course at this grip (5 m/s needs an 8 m radius at 3 m/s²). The target is instead: use all the grip the car has, everywhere.

## 3. Rules for the implementer

### Global Constraints

- Read `CLAUDE.md` first. Its Safety model and Test Quality sections bind every task here.
- **Never drive a car, launch bringup, or launch a driving node on a car.** Floor tests in section 9 are the tech lead's.
- Never set `enable_deadman: false`, never relax a guard in `racerbot_car/guards.py`, never touch `src/racerbot_a` (team A's, pin only).
- Run every ROS, pytest, colcon and simulator command as `rb <command>` on the dev box (domain 79, localhost discovery).
- One branch and one pull request per phase, branched from `origin/main`. Never push to `main`. Do not merge; the tech lead merges.
- Every new parameter defaults to today's behaviour. A change of a shipped default is its own commit with the simulator evidence in the message.
- Per-car physical values go in `src/racerbot_car/cars/<id>/calibration/`; a key must exist in the shared node block first. Run `rb python3 -m pytest src/racerbot_launch/test/test_car_profiles.py -q` after any profile change.
- Hard rule 1: break the code under each new test and watch it fail before calling the test done. Say in the commit message which mutation you used.
- Hard rule 2: a failing existing test is a finding. Task 3 changes one specified behaviour and names the test; anything else, stop and ask.
- Judge speed changes on the whole course suite (`tools/f1tenth_sim/course_suite/`), never one course. Zero contact, computed by the simulator's own chassis check, not gym's flag.
- The dev box is shared: one simulator job per seed, at most 6 at once, and not while load is above 5. The pit PC (`racerbot-pit-pc@100.73.207.60`, `~/racerbot-sim`) takes long suites.

### Starting state (2026-11-12)

- Open pull requests: #46 (simulator crash-course floor, a hypothesis; Phase 3 replaces it), #47 (rb2 planner lateral 3.2 to 2.8; see Task 8), #48 (line builder about 10 times faster; Task 8's sweeps want it merged), #50 (draft, see Task 4). Do not merge or close any of them; ask the tech lead about #48 before starting Task 8.
- rb2 is on branch `neural-racer/model-a0eef2af0a3757b6`, not `main`, with the edit in #50 uncommitted. Do not change rb2's checkout without the tech lead's word.
- Both cars hold runs that exist nowhere else (rb2 about 5.3 GB, rb1 about 0.8 GB). Never delete or move run data on a car.

### Getting the race-day data

rb1's four race-day runs are on the dev box under `data/rb1/racerbot_runs/20261111-*` (untracked). rb2's stay on the car; stream what a task needs into memory, never onto the dev box's disk (it is 94% full):

```bash
mkdir -p /dev/shm/rb2 && cd /dev/shm/rb2 && ssh -o BatchMode=yes racerbotcar-2@100.107.122.58 \
  'cd ~/racerbot-ws && tar cf - data/rb2/racerbot_auto/20261111-151026-rebuilt data/rb2/racerbot_runs/20261111-152132' | tar xf -
```

About 3 MB/s. `scp` hangs on car 2; use the `tar` pipe. Check `pgrep -af "bringup|_launch.py"` on the car first and do not read from a car that is driving. Locate a run before using `laps.py`, `gripcurve.py` or `car_measure.py`:

```bash
rb python3 tools/run_analysis/loc.py <run> --map /dev/shm/rb2/data/rb2/racerbot_auto/20261111-151026-rebuilt --workers 4
```

### When to ask the advisor (Fable)

Ask before acting, with the evidence attached, at each of these points:

1. Before writing the detailed steps for Phases 3, 4 and 5 (they are specified here as tasks with acceptance criteria; the step-level plan depends on what Phases 1 and 2 measure).
2. When Task 5's grip number is in, before choosing the lateral values Task 8 tries.
3. Any time a simulator result contradicts a race-day measurement in section 1.
4. Any existing test that fails and is not named in this plan.
5. Before proposing any change to a safety parameter, ceiling, or the reactive layers beyond what a task specifies.
6. If a task's acceptance criterion cannot be met after two honest attempts. Report what was tried; do not loosen the criterion.

### Review Focus

Failure modes the tasks' own tests might miss, most likely first. Each has its test in the task named.

1. A wall the car is *touching* (return inside the true body rectangle) must still stop it; the flank creep must never push along a wall. Tasks 2 and 3.
2. Steering *toward* the flank wall must still stop; only an arc that keeps the body clear may creep. Tasks 2 and 3.
3. Understeer compensation with a bad or empty curve (NaN, non-monotonic, one point) must refuse at startup or fall back to no compensation, never steer harder. Task 6.
4. Understeer compensation near full lock must clip at `max_steering_angle` and the speed cap must still hold the delivered lateral acceleration under the limit. Tasks 6 and 7.
5. LB released, `/joy` stale, or `/joy` never received during a flank creep: zero command. Tasks 2 and 3.

---

## 4. Phase 1: stop the wall-alongside latch

Branch: `unstick-flank`. One pull request.

**File structure:**
- `src/gap_follow/gap_follow/gap_logic.py`: add `body_arc_free_distance` (pure).
- `src/pure_pursuit/pure_pursuit/racing_math.py`: the same function, copied (the two packages do not import each other; `vehicle_boundary_distances` is already duplicated this way).
- `src/gap_follow/gap_follow/gap_follow_node.py`, `src/pure_pursuit/pure_pursuit/pure_pursuit_node.py`: the creep branch.
- `src/gap_follow/config/gap_follow.yaml`, `src/pure_pursuit/config/pure_pursuit.yaml`: two new parameters each.
- Tests and fixtures listed per task.

### Task 1: `body_arc_free_distance`

**Files:**
- Modify: `src/gap_follow/gap_follow/gap_logic.py` (add after `swept_path_clearance`)
- Modify: `src/pure_pursuit/pure_pursuit/racing_math.py` (identical copy)
- Test: `src/gap_follow/test/test_body_arc_free_distance.py` (new)
- Test: `src/pure_pursuit/test/test_body_arc_parity.py` (new)

**Interfaces:**
- Produces, in both modules:

```python
def body_arc_free_distance(clean: np.ndarray, valid: np.ndarray, angles: np.ndarray,
                           curvature: float, half_width: float, car_length: float,
                           wheelbase: float, laser_offset_x: float,
                           laser_offset_y: float = 0.0,
                           max_length: float = 1.0, step: float = 0.02) -> float:
    """Distance base_link can travel along a constant-curvature arc before any
    valid scan point lies inside the body rectangle. inf if none within
    max_length. 0.0 only if a point is inside the rectangle at the start."""
```

- The rectangle is `x in [rear, front]`, `|y| <= half_width` in base_link, with `front = wheelbase/2 + car_length/2`, `rear = wheelbase/2 - car_length/2` (same model as `swept_path_clearance`). Scan points are fixed in the starting base_link frame; the body moves. REP-103: positive curvature turns left.
- Approach: for travel `s = 0, step, 2*step, ... <= max_length`, the body pose is `(sin(k s)/k, (1 - cos(k s))/k, k s)` (straight line when `|k| < 1e-6`); transform all points into each pose's frame in one vectorized pass and test rectangle membership. Return the first `s` with a hit.
- Unlike `swept_path_clearance`, a point alongside the body but outside the rectangle is not an obstacle unless the body moves into it.

- [ ] **Step 1: Write the failing tests** in `test_body_arc_free_distance.py`. Geometry for all: `car_length=0.58, wheelbase=0.36, laser_offset_x=0.26`, so `front=0.47, rear=-0.11`. Build scans from points with a helper that converts base_link `(x, y)` points to `(range, angle)` from the LiDAR at `(0.26, 0)`. Each expected value below is a closed form; say so in the test.

| Test name | Setup | Assertion |
|---|---|---|
| `test_wall_alongside_outside_the_body_is_free_on_a_straight` | wall points `x in [-1, 3]` every 1 cm at `y=+0.20`; `curvature=0`, `half_width=0.175` | `== math.inf` |
| `test_wall_inside_the_body_rectangle_is_zero` | same wall at `y=+0.15`, `half_width=0.175` | `== 0.0` |
| `test_obstacle_ahead_on_a_straight_is_its_gap_to_the_bumper` | one point at `(0.47 + 0.30, 0.0)` | `== pytest.approx(0.30, abs=0.021)  # one 2 cm step` |
| `test_turning_away_from_a_flank_wall_is_free` | wall at `y=+0.20`; `curvature=-1/1.05`, `half_width=0.175`, `max_length=0.5` | `== math.inf` |
| `test_turning_toward_a_flank_wall_hits_at_the_front_corner` | wall at `y=+0.30`; `curvature=+1/1.05`, `half_width=0.175` | `approx(theta * R, abs=0.025)` with `R=1.05`, `rho=hypot(0.47, R-0.175)`, `phi0=atan2(0.47, R-0.175)`, `theta=acos((R-0.30)/rho)-phi0` |
| `test_mirror_symmetry` | the toward-wall case mirrored (`y -> -y`, `curvature -> -curvature`) | equal to the unmirrored result, `abs=1e-9` |
| `test_invalid_beams_are_not_obstacles` | a point inside the rectangle with `valid=False`; also `NaN` and `inf` ranges marked invalid | `== math.inf` |
| `test_empty_scan_is_free` | zero-length arrays | `== math.inf` |
| `test_all_zero_ranges_are_ignored` | all ranges 0.0, valid True | `== math.inf` (range 0 is not a return, as in `swept_path_clearance`) |
| `test_point_beyond_max_length_is_free` | point at `(0.47 + 0.60, 0)`, `max_length=0.5` | `== math.inf` |
| `test_mismatched_shapes_raise` | `angles` one longer | `pytest.raises(ValueError, match='matching shapes')` |
| `test_non_finite_or_non_positive_geometry_raises` | parametrize: `half_width=0`, `step=0`, `max_length=-1`, `curvature=nan` | `pytest.raises(ValueError, match=<the offending name>)` |

- [ ] **Step 2:** Run `rb python3 -m pytest src/gap_follow/test/test_body_arc_free_distance.py -v`. Expected: every test fails with `AttributeError: module 'gap_follow.gap_logic' has no attribute 'body_arc_free_distance'`.
- [ ] **Step 3:** Implement in `gap_logic.py`.
- [ ] **Step 4:** Run again; all pass. Then mutate and confirm failures: (a) drop the `x >= rear` bound, (b) flip the sign of the pose's `y`, (c) return `0.0` always, (d) return `math.inf` always. Each must fail at least one test. Revert.
- [ ] **Step 5:** Copy the function into `racing_math.py`. Write `test_body_arc_parity.py`: `test_matches_gap_follows_copy_on_random_scans` builds 200 seeded random scans (1081 beams, ranges 0.05 to 5 m, 5% invalid) with random curvature in `[-1.2, 1.2]` and asserts both modules return the same float (`==`, both are the same arithmetic). Import `gap_follow.gap_logic` inside the test with `pytest.importorskip('gap_follow.gap_logic')` so the unsourced run still collects; state in the test docstring that `rb` runs it.
- [ ] **Step 6:** Commit: `gap_logic, racing_math: body_arc_free_distance (rectangle swept along an arc)`.

### Task 2: gap follow creeps away from a flank wall

**Files:**
- Modify: `src/gap_follow/gap_follow/gap_follow_node.py` (parameters near line 247; speed stage near lines 1876 to 1916; intent factor near line 2127)
- Modify: `src/gap_follow/config/gap_follow.yaml`, `src/gap_follow/gap_follow/live_tuning.py` (not live-tunable: add neither parameter to the tunables; the live-tuning test must still pass)
- Create: `src/gap_follow/test/data/flank_latch_rb1_20261111-155542.npz`
- Test: `src/gap_follow/test/test_gap_follow_flank_creep_node.py` (new), `src/gap_follow/test/test_flank_creep_replay.py` (new)

**Interfaces:**
- Consumes: `gap_logic.body_arc_free_distance` (Task 1).
- Produces: parameters `flank_creep_length` (float, m, default `0.0` = off, shipped config `0.50`) and `flank_creep_margin` (float, m, default and shipped `0.02`); `/drive_intent` factor name `'flank creep'`.

**Behaviour (the whole specification):** in the speed stage, after `clearance_speed` is computed and after the existing `if creeping:` block, when all of these hold:

1. `flank_creep_length > 0`,
2. `clearance_speed <= 0.0` and `creeping` is False,
3. no emergency or TTC trip this tick (`emergency_tripped` and `ttc_tripped` are False),
4. `body_arc_free_distance(raw_window, window_valid, beam_angles, tan(steering_angle)/wheelbase, car_width/2 + flank_creep_margin, car_length, wheelbase, laser_offset_x, laser_offset_y, max_length=flank_creep_length)` is `inf`, where `steering_angle` is the slew-limited angle the rack will drive this tick,

then `clearance_speed = escape_creep_speed` and the tick is treated as `creeping` for everything downstream that reads it (ramp, launch boost, intent label `'flank creep'`). Otherwise nothing changes. Startup refuses `flank_creep_length` outside `[0, 2]` or `flank_creep_margin` outside `[0, 0.10]` with the same error style as `escape_creep_speed`.

- [ ] **Step 1: Save the fixture.** With rb1's run on the dev box and located (loc command in section 3):

```bash
rb python3 tools/run_analysis/scan_window_fixture.py data/rb1/racerbot_runs/20261111-155542 \
  --loc ~/.cache/racerbot/loc --from 88.5 --to 90.0 \
  --out src/gap_follow/test/data/flank_latch_rb1_20261111-155542.npz
```

Confirm it is under 1 MB and holds about 60 scans. Check `src/gap_follow/setup.py` does not need a `data_files` entry for tests to find it (tests read it by path relative to the test file).

- [ ] **Step 2: Write `test_flank_creep_replay.py`** (pure, no rclpy): for every scan in the fixture, with rb1's geometry (`car_width=0.31`, `swept_clearance_margin=0.05`, steering -0.20 rad):
  - `test_the_recorded_scans_reproduce_the_latch`: `swept_path_clearance(...) == 0.0` for at least 95% of scans (this pins the cause; label it `# recorded measurement: rb1 20261111-155542, 88.5-90.0 s`).
  - `test_the_body_arc_is_free_on_the_recorded_scans`: `body_arc_free_distance(..., half_width=0.175, max_length=0.5) == math.inf` for at least 95% of scans.
  - `test_steering_into_the_wall_is_not_free_on_the_recorded_scans`: with steering +0.33 rad, the result is finite and below 0.5 for at least 95% of scans.
- [ ] **Step 3: Write `test_gap_follow_flank_creep_node.py`**, following the node-construction pattern in `test_gap_follow_node.py` (`drive_topic:=/test_only/drive`, remap after any params file, LB faked through the test's own `/joy`). Synthetic scan: a straight wall 0.20 m left of the LiDAR, parallel to the car, otherwise open to 3 m.

| Test name | Assertion on the published command |
|---|---|
| `test_wall_5cm_off_the_flank_with_open_road_creeps` | `0 < speed <= 0.40` within 10 ticks, `flank_creep_length=0.5` |
| `test_flank_creep_off_by_default_keeps_the_stop` | node built with no `flank_creep_length`: speed `== 0.0` for 20 ticks |
| `test_wall_inside_the_body_still_stops` | wall 0.15 m from the LiDAR: speed `== 0.0` for 20 ticks |
| `test_steering_toward_the_flank_wall_still_stops` | wall 0.20 m left and the only gap far left (block ahead and right at 0.6 m): speed `== 0.0` |
| `test_flank_creep_reports_itself_in_drive_intent` | the bound factor's name is `'flank creep'` |
| `test_lb_released_during_flank_creep_is_zero` | creep established, then LB released: speed `== 0.0` on the next command |
| `test_joy_never_received_during_flank_creep_is_zero` | no `/joy` at all: speed `== 0.0` |
| `test_stale_joy_during_flank_creep_is_zero` | `/joy` stops; after the deadman timeout: speed `== 0.0` |
| `test_bad_flank_creep_parameters_refuse_to_start` | `flank_creep_length=-1`, `=3`, `flank_creep_margin=0.2`: constructing the node raises, `match='flank_creep'` |

- [ ] **Step 4:** Run both files. Expected: the creep tests fail (speed stays 0), the "still stops" tests pass already.
- [ ] **Step 5:** Implement. Add the two parameters to `gap_follow.yaml` with a comment citing the run and section 1.2 of this plan.
- [ ] **Step 6:** Run both files, then the whole package: `rb python3 -m pytest src/gap_follow/test/ -q`. Compare the collected count with `grep -c '^def test' src/gap_follow/test/*.py`. Mutations to confirm: (a) skip condition 3, (b) use `desired_steering` in place of `steering_angle`, (c) return `escape_creep_speed * 2`. `test_gap_mirror_parity.py` will fail if `tools/f1tenth_sim/gap_mirror.py` must mirror the change; mirror it there (that file is the simulator's copy of the node's logic) and add the same creep to its parity test.
- [ ] **Step 7:** Simulator check, no regressions: `rb python3 tools/f1tenth_sim/course_suite/run.py` with the gap follow arm (read `tools/f1tenth_sim/course_suite/README.md` for the exact flags). Acceptance: zero contact on every course, and no course slower than `main` by more than 0.5%.
- [ ] **Step 8:** Commit: `gap_follow: creep away from a wall alongside when the body's own arc is clear`.

### Task 3: pure pursuit creeps away from a flank wall

**Files:**
- Modify: `src/pure_pursuit/pure_pursuit/pure_pursuit_node.py` (`_reactive_override`, Tier 0, near lines 2373 to 2415)
- Modify: `src/pure_pursuit/config/pure_pursuit.yaml`
- Create: `src/pure_pursuit/test/data/flank_stop_rb2_20261111-152132.npz` (`scan_window_fixture.py --from 20.0 --to 21.5`, rb2's run streamed per section 3)
- Test: `src/pure_pursuit/test/test_flank_creep_node.py` (new); `src/pure_pursuit/test/test_opponent_integration.py` (one named change)

**Interfaces:**
- Consumes: `racing_math.body_arc_free_distance` (Task 1).
- Produces: parameters `flank_creep_length` (default `0.0` off, shipped `0.50`), `flank_creep_margin` (default and shipped `0.01`; the pure pursuit body floor is `emergency_stop_clearance: 0.03`, so the creep rectangle is the body plus 1 cm); reactive state `'flank_creep'`.

**Behaviour:** in Tier 0, where today a non-directional body contact returns `(None, 0.0, 'body_contact', detail)`: if `flank_creep_length > 0`, and `body_clearance > 0.0`, and `body_arc_free_distance(...)` along `desired_steering` with `half_width = car_width/2 + flank_creep_margin` and `max_length = flank_creep_length` is `inf`, return `(None, <creep speed>, 'flank_creep', detail)`. Use the creep speed `_emergency_escape` already uses (read it there; do not add a second creep-speed parameter). Steering is left to the tracking law (`None` override). Everything else in Tier 0 is unchanged.

**The existing test this changes (Hard rule 2, approved by the tech lead with this plan):** `test_wall_alongside_the_car_stops_it_although_the_forward_cone_is_clear` in `test_opponent_integration.py:605`. It asserts a flank wall stops the car. That stays true with the shipped-off default and whenever the body's arc is not free. Do **not** edit its assertions. If it fails after this task, the default is not off: fix the code. Add new tests beside it; do not replace it. Show the tech lead the final diff of that file in the pull request description.

- [ ] **Step 1:** Save the fixture (command above). Write `test_flank_creep_node.py` with the node pattern from `test_opponent_integration.py` (`drive_topic:=/test_only/drive`): the same nine cases as Task 2's table, with wall distance 0.18 m right for the creep case, 0.15 m for the touching case, and the state asserted as `'flank_creep'` in `/drive_intent`. Add `test_recorded_flank_stop_scans_are_free_along_the_arc` (pure): on the fixture, `body_arc_free_distance(..., half_width=0.165, curvature=tan(0.025)/0.36, max_length=0.5) == math.inf` for at least 95% of scans (`# recorded measurement: rb2 20261111-152132, 20.0-21.5 s`).
- [ ] **Step 2:** Run; creep tests fail, stop tests pass.
- [ ] **Step 3:** Implement and add the parameters to `pure_pursuit.yaml`.
- [ ] **Step 4:** `rb python3 -m pytest src/pure_pursuit/test/ -q`; compare collected count with `grep -c '^def test'`. Mutations: drop the `body_clearance > 0.0` condition; use `inf != inf`. Each must fail a test.
- [ ] **Step 5:** `rb python3 tools/run_analysis/tier1_replay.py` is for Tier 1; for Tier 0, write nothing new: run the course suite's map-racer arm. Acceptance: zero contact, no course slower than `main` by more than 0.5%.
- [ ] **Step 6:** Commit, push, open the pull request `Stop the wall-alongside latch (gap follow and pure pursuit)`. Body: section 1.2's evidence, the three run IDs, the suite table, and floor test F1 from section 9.

---

## 5. Phase 2: pure pursuit uses the grip the car has

Branch: `pp-grip`. One pull request per task group marked (PR).

### Task 4: race-day map and runs become simulator and test assets

**Files:**
- Create: `tools/f1tenth_sim/maps/car2_course_2026_11_11_*` via `rb python3 tools/f1tenth_sim/track_import.py` (read its `--help`; follow how `car2_course_2026_11_06a` was imported and registered in `maps/catalog.yaml` and the course suite)
- Create: `src/pure_pursuit/test/data/course_20261111/` with the map, `raceline_raw.csv`, and `standing_start_rb2_20261111-152132.npz` (`scan_window_fixture.py --from 3.0 --to 8.6`)

- [ ] **Finish draft PR #50 first** (`car2-guarded-recenter`): car 2's recovered `_guarded_move` edit to `raceline_optimizer.refine_centerline`, which the race-day map was built with. Add `test_refine_centerline_does_not_cross_an_island` to `src/pure_pursuit/test/test_raceline_optimizer.py`: an occupancy map of a 3 m wide ring track with a 0.5 m square pillar in the middle of one straight and a seed path passing 0.3 m beside it; assert every point of the refined line has `occ_map.clearance_at(x, y) > 1.5 * resolution` and that the line still closes. Prove it fails with `_guarded_move` stubbed to `return new`. Also assert the unobstructed-track result is unchanged (`np.allclose` against the same call on `main`'s function for a plain ring, `atol=1e-9  # m`). Run the pure_pursuit suite and the course suite's map-racer arm on that branch; zero contact and no course that built a line on `main` failing to build. Mark the PR ready and report.
- [x] Import the map, add it to the catalog and to the course suite's list.
- [ ] Baseline on `main`: run the map racer and gap follow (default and time trial) on it, 6 seeds. Record lap times in the pull request. **Acceptance for the import only:** the course builds, a lap closes, zero contact for gap follow default. The simulator's lap times versus section 1.1 are recorded, not gated, here; Phase 3 closes that gap.
- [x] Commit.

### Task 5: measure the sustained grip of each car (PR with Task 4)

**Files:**
- Create: `tools/run_analysis/grip_truth.py`, `tools/run_analysis/tests/test_grip_truth.py`

**Interfaces:**
- Produces: `grip_truth.steady_corners(t, speed, yaw_rate, lateral_accel, steering, *, min_speed=1.5, max_steering_change=0.03, window=0.3, max_long_accel=0.5) -> np.ndarray` (boolean mask of steady-cornering samples) and `grip_truth.curve(asked, delivered, bins) -> list[dict]`; CLI `grip_truth.py <located runs> [--imu-topic /imu/vesc/data] [--json out.json]` printing, per bin of asked lateral acceleration: sample count, median and p90 delivered by **accelerometer** and by **true speed times gyro**, and the ratio of the two.
- JSON schema: `{"car": str, "runs": [str], "imu_topic": str, "criteria": {...}, "curve": [{"asked": float, "delivered": float, "n": int}], "sustained_limit": float|null, "peak_asked": float|null, "accel_over_gyro": float|null}`. **Amended at advisor checkpoint 2 (2026-11-12):** `sustained_limit` is the largest GYRO-basis (true speed times gyro) median delivered among bins with n >= 50, `peak_asked` is that bin's centre, `accel_over_gyro` is the median accelerometer-over-gyro ratio over bins with n >= 50 (a roll indicator, not a gate). `curve[].delivered` is the accelerometer median, kept for reference.

Why the gyro basis: on steady corners true speed times gyro IS the centripetal acceleration; the accelerometer also reads body roll (g*sin(roll)), so it sits above the gyro (5-13% on rb2, 20-30% on the softer rb1). The first design (agreement within 15% as a gate) was wrong for steady samples and was replaced at checkpoint 2. Remove the accelerometer's bias and gravity leak with the median over straight samples (`|yaw_rate| < 0.05 rad/s`, speed above 1.5 m/s) of the same run. rb2's BNO lateral axis is unusable until its mount is calibrated (profile says PROVISIONAL); only `/imu/vesc/data` is used for rb2.

- [x] Tests first (`tools/run_analysis/tests/` runs unsourced; keep the math importable without rclpy, bag reading in `main()` only): synthetic steady circle (`speed=3`, `yaw_rate=1`, `lateral_accel=3`) is all steady and bins to delivered 3.0 `abs=0.01`; a steering ramp of 0.1 rad over 0.3 s is rejected; a braking segment of 2 m/s² is rejected; an accelerometer bias of 0.4 m/s² on a run with straights is removed to within 0.05; a simulated slide (gyro says 5, accelerometer says 3) is excluded from `sustained_limit`; empty input returns an empty curve and `sustained_limit` NaN; NaN samples are dropped, not propagated.
- [x] Implement; run on rb2 `152132`, `142604`, `135934` and rb1 `153721`, `155542`. Save the two JSON files under `tools/f1tenth_sim/sim_fidelity/` as `sfu_course_2026_11_11_grip_rb1.json` and `..._rb2.json`.
- [ ] **Advisor checkpoint 2:** send Fable the two tables. Outcome is one number per car, `sustained_limit`, and a decision whether floor test F2 (steady circles, section 9) is needed before Task 8.
- [x] Commit.

### Task 6: understeer compensation in the steering command (PR)

**Files:**
- Modify: `src/pure_pursuit/pure_pursuit/racing_math.py`, `src/pure_pursuit/pure_pursuit/pure_pursuit_node.py`, `src/pure_pursuit/config/pure_pursuit.yaml`
- Test: `src/pure_pursuit/test/test_understeer_compensation.py` (new, pure), `src/pure_pursuit/test/test_understeer_node.py` (new, node)

**Interfaces:**
- Produces in `racing_math.py`:

```python
@dataclasses.dataclass(frozen=True)
class GripCurve:
    asked: tuple[float, ...]      # kinematic lateral accel, m/s^2, strictly increasing, starts at 0
    delivered: tuple[float, ...]  # what the car achieves, strictly increasing, starts at 0, delivered <= asked

    def deliver(self, asked: float) -> float: ...   # piecewise linear; beyond the last point, flat
    def ask_for(self, wanted: float) -> float: ...  # inverse; beyond the last delivered value, the last asked value

def grip_curve_from_params(asked: Sequence[float], delivered: Sequence[float]) -> GripCurve | None:
    """None for two empty lists (compensation off). ValueError, naming the rule broken, for
    unequal lengths, fewer than 2 points, a non-finite value, non-increasing values,
    a first point other than (0, 0), or delivered > asked anywhere."""

def understeer_steering(steering: float, speed: float, wheelbase: float,
                        curve: GripCurve | None, max_steering_angle: float,
                        min_speed: float = 1.0) -> float:
    """The steering that makes the car follow the curvature `steering` asks for kinematically.
    Unchanged when curve is None, speed < min_speed, or non-finite input (returns `steering`
    clipped). Sign preserved. Clipped to +/- max_steering_angle."""
```

- Node parameters: `understeer_curve_asked`, `understeer_curve_delivered` (double arrays, default `[]`). Applied to the tracking law's steering, before the reactive override and before `_shape_normal_command`, only in the normal driving state (not while recovering, not to a reactive or escape steering). The uncompensated value is what `/drive_intent` reports as desired steering; add factor-free detail text `understeer +0.031rad`.

- [x] Pure tests (closed forms): identity curve `((0, 10), (0, 10))` returns the input steering (`abs=1e-12`); with `asked=(0, 2.5, 10)`, `delivered=(0, 2.5, 5.5)`, speed 3, wheelbase 0.36, a steering whose kinematic lateral is 4.0 comes back as the steering whose kinematic lateral is `2.5 + (4.0-2.5) * 7.5/3.0 = 6.25` (`abs=1e-6 rad`); below 2.5 unchanged; negative steering mirrors; result clips at 0.42; speed 0.5 returns input; `deliver(ask_for(a)) == a` over 50 values inside the curve (`abs=1e-9`); `ask_for` beyond the last delivered value returns the last asked value; each invalid curve raises `ValueError` with `match=` its rule; NaN steering or speed returns the clipped input, never NaN.
- [x] Node tests: empty arrays publish exactly the same steering as `main` on a fixed pose and line (compare two nodes); a curve makes the published steering larger in magnitude in a corner above 2.5 m/s² and identical on a straight; an invalid curve refuses to start (`match='understeer_curve'`); the four deadman deny-path tests on this node construction with the curve set.
- [x] Mutations: swap `asked` and `delivered` in `ask_for`; drop the clip; apply below `min_speed`.
- [ ] Commit.

### Task 7: the online curve cap uses delivered, not commanded, curvature (PR with Task 6)

**Files:** `pure_pursuit_node.py` (`_shape_normal_command`, lines 3368 on), test in `test_understeer_node.py`.

**Behaviour:** with a `GripCurve` set, `curve_speed` is the largest speed `v <= max_speed` with `curve.deliver(v² * online_curvature) <= max_lateral_accel`. Add `racing_math.delivered_curvature_speed_limit(curvature, max_lateral_accel, max_speed, curve) -> float`; with `curve=None` it returns exactly `curvature_speed_limit(...)`. Solve in closed form per linear segment (no iteration).

- [x] Tests: `curve=None` equals `curvature_speed_limit` over 100 seeded inputs (`==`); identity curve likewise (`abs=1e-9`); with the Task 6 example curve and `max_lateral_accel=3.1`, curvature 0.5/m: expected speed is `sqrt(ask_for(3.1)/0.5)` (`abs=1e-6`); zero curvature returns `max_speed`; curvature so large the limit is below any segment returns the kinematic answer of the first segment; NaN curvature returns 0.0 (command zero, the specified edge for driving code).
- [ ] Commit; open the pull request for Tasks 6 and 7 with both features **off by default**.

### Task 8: choose rb2's curve and lateral limits in the simulator (PR)

**Depends on:** Task 5's numbers, advisor checkpoint 2, and Phase 3's Task 11 (a simulator floor fitted to race day). If Phase 3 is not done, do Task 11 first.

- [ ] Put rb2's measured curve in `src/racerbot_car/cars/rb2/calibration/pure_pursuit.yaml` (`understeer_curve_asked`, `understeer_curve_delivered`), with a comment naming the JSON from Task 5.
- [ ] Sweep on the full course suite plus `car2_course_2026_11_11`, 6 seeds each, on the pit PC: `profile_max_lateral_accel` and online `max_lateral_accel` at `{current, sustained_limit - 0.4, sustained_limit - 0.2}` (three arms, compensation on) plus the control (current values, compensation off).
- [ ] **Acceptance:** an arm ships only with zero contact on every course and seed, a suite-mean lap gain of at least 3% over the control, no course slower than control, and the safety layer (reactive cap ticks) no busier than control. If no arm passes, ship compensation off and report.
- [ ] Open pull request #47 (rb2 planner lateral 3.2 to 2.8) conflicts with this task. Do not merge or close it; tell the tech lead which of the two the evidence supports.
- [ ] Commit the calibration change alone, with the suite table in the message. Floor ladder F3 follows.

### Task 9: the standing-start lap

- [ ] Diagnose, on `rb2 20261111-152132` from 3.0 to 8.6 s: (a) particle-filter pose (`/pf/viz/inferred_pose`) against `loc.py`'s pose, position and heading error over time; (b) commanded speed against the profile and against true speed; (c) cross-track error from `/drive_intent`'s detail text. Write the finding (which of localization, speed lead from rest, or line entry) in the pull request.
- [ ] **Advisor checkpoint:** send Fable the three traces before choosing the fix.
- [ ] Reproduce in the simulator on `car2_course_2026_11_11` starting at rest at the recorded start pose; write the test that fails; fix; the test passes. **Acceptance:** from rest at 5 start poses spread round the course, 6 seeds, the first lap has zero contact and is within 1.5 s of a flying lap.

### Task 9b: the map racer on car 1 (PR)

Car 1 has only ever raced gap follow. Two things have blocked it: its SLAM could not save a clean map (false loop closures, 2026-10-26 finding), and its steering chain was never measured, so it kept plain pure pursuit instead of `delay_lqr`. Race day supplies the measurement (section 1.4: servo 3 ticks, gain 12/s, rate 1.75 rad/s; speed loop 5 ticks, accel 5.0, brake 4.0 m/s²).

- [ ] **Can car 1 map today?** Replay rb1's race-day gap follow run through the mapping pipeline offline: `rb python3 tools/run_analysis/replay_map_course.py data/rb1/racerbot_runs/20261111-153721` (read its `--help`; `run_triage.py` first names where a clean replay can start, since the car was carried between holds). Compare the result with the race-day map using `slam_score.py`. Record: map built or not, and why not.
- [ ] **A map from the other car.** Find out what stops `race_launch.py` on rb1 from using a `run_dir` under `data/rb2/racerbot_auto/` (identity guards, `metadata/car.yaml`, the line's per-car physics baked into `raceline_profiled.csv`). The supported route is: copy the map and `raceline_raw.csv`, then rebuild the line for rb1 with `python3 -m pure_pursuit.race_line --run-dir <dir> --car rb1`. Make that one documented command (extend `pure_pursuit/race_line.py` or `run_commands.py`; do not weaken an identity guard) and test that the rebuilt line's speeds come from rb1's calibration, not rb2's.
- [ ] **rb1's calibration for the map racer:** in `src/racerbot_car/cars/rb1/`, give `delay_lqr` rb1's measured delays (find the `delay_lqr.*` parameter names in `pure_pursuit.yaml`; they default to rb2's values), set `steering_law: delay_lqr` in a tuning overlay with a `# why:` line citing the measurement, set rb1's `profile_max_accel`/`profile_max_brake` and lateral values from Task 5's rb1 result, and add rb1's `GripCurve`. Neither car has a `tuning/` directory for rb1 today; create it only for this justified override. Run the hardware guard.
- [ ] **Simulator:** with Phase 3's rb1 floor, run the suite's map-racer arm for rb1. Same acceptance as Task 8 (zero contact, no course slower than rb1's best gap follow, at least 3% faster on the suite mean).
- [ ] **Advisor checkpoint:** before the pull request, send Fable the mapping-replay result and the suite table. If car 1 still cannot map, the pull request says so plainly and ships the other-car-map route as the way to race.
- [ ] Floor ladder F3 on rb1 follows (section 9).

---

## 6. Phase 3: the simulator matches race day

Branch: `sim-2026-11-11`. Write the step-level plan with the advisor first (checkpoint 1).

### Task 10: fit a race-day floor per car
- `rb python3 tools/run_analysis/car_measure.py <located runs> --json <out>` for each car (section 1.4 has the headline values), then `fit_sim.py` open-loop on `rb2 152132` and `rb1 153721`.
- Add `Floor` entries `sfu_course_2026_11_11_rb2` and `..._rb1` in `tools/f1tenth_sim/sim_fidelity/calibration.py`, following `sfu_course_2026_11_06`, each with provenance naming its runs. Use `ProgressiveGrip` with Task 5's curve.
- **Acceptance:** `fit_sim.py` yaw-rate RMS and speed RMS no worse than the 2026-11-06 floor's recorded values in `sfu_course_2026_11_06_replay.json`.

### Task 11: closed-loop agreement
- On `car2_course_2026_11_11` with the race-day floor and the race-day configs (from each run's `metadata/params/`): map racer, gap follow time trial, gap follow default (rb1 floor).
- **Acceptance:** simulated median lap within 4% of section 1.1 for all three (10.05 s, 11.9 s, about 11.5 s). `simgrip.py` curve within 0.3 m/s² of Task 5's at every bin. If the map racer is off by more, the cause goes to the advisor (checkpoint 3) before any knob is turned.
- Add `tools/f1tenth_sim/tests/test_race_day_agreement.py` pinning the three lap times as recorded measurements with the 4% band.

### Task 12: the simulator reproduces both stuck cases
- Drive gap follow and the map racer at poses taken from the two fixtures (wall alongside, open ahead) with `flank_creep_length: 0.0`: the car must latch. With `0.50`: it must leave within 3 s without the chassis check reporting contact. This is the regression test that keeps Phase 1 fixed.

---

## 7. Phase 4: gap follow reaches the disparity extender's pace

Branch: `gap-follow-pace`. Write the step-level plan with the advisor first. The target: in the simulator on the race-day floor, gap follow time trial laps `car2_course_2026_11_11` in 10.6 s or better (the disparity extender's real typical lap), zero contact, with no course in the suite slower than today.

### Task 13: why the car got 5 cm from the wall
- rb1 `155542`, 85.0 to 88.0 s: at 87.0 s it steered +0.19 rad (left) with the left wall 0.62 m away; 0.5 s later the wall was at 0.31 m and 1.0 s later at 0.21 m. Save that window as a fixture; replay it through the node's logic with `tools/run_analysis/replay.py`; name the stage that chose left (gap selection, centering, arc guard, or corner hold). Fix that stage; the replay test asserts the minimum left clearance over the window stays above 0.10 m.
- Check whether the tech lead's "always stalls at the same point" matches this spot: use `spot.py --map <race-day map> --at <x,y of 87.5 s from loc>` on rb1 `153721` and `155542`, and list every stall in both runs by course position.

### Task 14: the braking ceiling
- `clearance cap` bound 27 to 36% of gap follow's time. `stops.py` on the race-day runs gives real stopping distances; rb1's speed loop brakes at 4.0 m/s², rb2's at 3.0, against `max_braking_decel: 3.0` and `braking_latency_sec: 0.25` for both.
- Per-car calibration of `max_braking_decel` from measured stops is a physical value, so it belongs in calibration; the guard treats braking keys as floors, so read `guards.py` first and take any conflict to the advisor (checkpoint 5). Sweep in the simulator; ship only with zero contact.

### Task 15: car 1's handicaps
- rb1 ran the default profile; rb2 ran time trial. Decide with the tech lead what rb1 should race with.
- `follow cap` bound 19% of rb1's final. Use `replay_opponent_gate.py` on `155542` to count how many follow episodes had a real car ahead (rb2 and UVic's cars were on track) and how many were walls or phantoms. Fix phantoms; leave real follows.
- rb1's steering limit (0.33 rad against rb2's 0.42) and its command gain are still open from earlier work; re-measuring them needs floor test F4.

---

## 8. Phase 5: adapt while driving

Branch: `live-grip`. Write the step-level plan with the advisor first. Constraint for everything here: an adaptive value may move only inside `[shipped value, a fixed ceiling in config]`; it can never raise a limit above its ceiling, never touch a deadman, TTC or clearance setting, and a stale or non-finite sensor returns it to the shipped value at once (the pattern `gap_follow`'s `enable_adaptive_grip` and `_understeer_cap` already follow; read `docs/gap-follow.md#understeer-check-imu`).

### Task 16: online understeer estimate for pure pursuit
- A pure estimator in `racing_math.py`: per bin of asked lateral acceleration, an exponential average of delivered over asked from the gyro and true-speed proxy (`/odom/fused` speed), only on steady-cornering samples (reuse Task 5's `steady_corners` rule). It scales the car's `GripCurve` within `[0.8, 1.2]` of the calibrated curve. Off by default.
- **Acceptance:** in the simulator with the floor's grip cut by 15% mid-run, cross-track error with the estimator on is at most half of what it is off, zero contact; with a NaN gyro it reverts to the calibrated curve within one tick.

### Task 17: online lateral limit for pure pursuit
- Port the logic of gap follow's adaptive grip to the map racer's online cap and profile scale: rise slowly while delivered tracks asked, cut at once on a slide, bounded as above.
- **Acceptance:** on the suite, no slower than fixed limits on the calibrated floor; on a floor 15% grippier, at least 2% faster; on a floor 15% slicker, zero contact where the fixed limits make contact.

### Task 18: tyres and top speed (tech lead's hardware)
- When tyres change: rerun Task 5 on a fresh run, update the calibration curve and the floor, rerun Task 8's sweep. No code change expected.
- Raising the VESC top speed above 5 m/s: the simulator's acceleration above 5.4 m/s is an estimate (`map-racer-margin` work, 2026-11-09). Needs floor test F5 before any config ships. Set `race_line_speed_floor_gain: 0.06` before raising it.

---

## 9. Floor tests (tech lead only)

Each is run after its pull request is merged and both cars are updated. Standard order every time: wheels off the ground with LB, then floor at low speed in open space, then the test.

| ID | After | What to do | Pass |
|---|---|---|---|
| F1 | Phase 1 | Park each car 5 cm from a wall, parallel, open road ahead. Launch gap follow, hold LB. Repeat with the map racer. Then park it touching the wall. | Creeps away at walking pace and resumes; touching the wall, it stays stopped. |
| F2 | Task 5, if the advisor asks | Steady circles by teleop at full left then full right stick, raising speed slowly until the car runs wide, with a recorder running (`docs/run-diagnostics.md`). | A clean grip number per car and direction from `grip_truth.py`. |
| F3 | Task 8 (rb2), Task 9b (rb1) | Map the course (rb1: or rebuild rb2's map for rb1), floor-test profile at 1.5 m/s, then race profile at `max_speed` 4.0, 4.5, 5.0 (live-tunable), 5 laps each. | Lap time falls with each step; taps are acceptable, a hard hit ends the ladder. |
| F4 | Task 15 | rb1 steering circles at full lock both ways (as on 2026-10-21) and a straight-line speed check. | rb1's real steering limit and gain. |
| F5 | Task 18 | Straight-line runs at 5.5, 6, 6.5 m/s on the longest straight available. | Measured acceleration above 5.4 m/s for the simulator. |

## 10. Done means

- Phase 1: neither controller sits beside a wall with open road ahead; both still stop when touching or steering into it. F1 passed on both cars.
- Phase 2: rb2's map racer is at least 3% faster on the suite mean with zero contact, and its first lap from rest is clean. rb1 runs the map racer too, on its own map or on one rebuilt from rb2's. F3 shows the gain on the floor on both cars.
- Phase 3: the simulator reproduces race-day lap times within 4% for three controllers.
- Phase 4: gap follow time trial at or under 10.6 s on the race-day course in the simulator, zero contact, suite not slower.
- Phase 5: the map racer holds its line when grip drops 15% and gains time when grip rises.
