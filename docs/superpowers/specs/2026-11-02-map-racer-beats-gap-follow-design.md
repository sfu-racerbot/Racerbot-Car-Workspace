# The map-based racer beats gap_follow on our real courses: design

> **Who this is for:** whoever builds the plan that implements this, human or agent, and the tech lead approving it.
> **Read first:** `CLAUDE.md`, [racing-autonomy.md](../../racing-autonomy.md), [sim-calibration.md](../../sim-calibration.md), [the SLAM reliability spec](2026-10-30-slam-reliability.md).
> **You'll be able to:** say what "faster than gap_follow" means here, what has to be built, in what order, and what was decided.
> **Time:** 10 minutes.

**Status:** design, 2026-11-02. Nothing here has run on the floor. No driving and no hardware in this work.

## Intent

The map-based stack (SLAM mapping, localization, `pure_pursuit`, `auto_map_race`'s race phase) exists so the car can brake early and carry speed through corners it already knows. On our real courses it has never shown that. The work is to make it measurably faster than `gap_follow` on the same course, with the same car calibration and zero contacts, for both cars. It has to be proven on recorded data and in a simulator that has itself been checked against real laps. Only then does a person take it to the floor.

We are here to win races (memory `aggressive-race-tuning`). So when the simulator shows a setting faster with zero contacts, it becomes the shipped default, not an opt-in.

## Where things stand (recon, 2026-11-02, `main` c0c86b9)

**Already fixed on `main`, so not work for this design:**
- `slam_gyro.yaml` (loaded with `gyro_odom:=true`, the `map_course` default) has a 3 m loop search, a chain of 10 and a weighted gyro prior (commit `ca79651`). On car 2's replays that gave 0.91 walls and under 0.3 m live error on every repeat. On car 1, an earlier version cut the worst correction from 2.45/2.75 m to 0.47 m (`194714`, `194913`).
- Lap closure happens at the closest approach to the start (`close_at_closest_approach: true`, `7ca6d9a`), not at the first tick inside the gate. The lap counter counts only corrections odometry did not see (`949cb6e`).
- `rebuild_run.py` builds a racing folder from any run's bag and gives a GO/NO-GO verdict. `replay_map_course.py` replays the whole `map_course` launch on a recording.

**Not done, and this design's job:**
1. **Car 1's recordings have never been replayed on the current SLAM profile.** The car-1 numbers above predate the gyro prior and chain 10.
2. **The map-based racer is far slower than gap_follow in the calibrated simulator.** Probe, 2026-11-02: `pure_course.py`, car 2, `course_2026_10_30_scans` plus floor `sfu_course_2026_10_30`, line `raceline_raw.csv`:

   | Top speed | Lateral | Lap | Contact | Notes |
   |---|---|---|---|---|
   | 4.0 | 2.5 | 29.35 s | 0 | max cross-track 0.12 m, 78 avoidance ticks |
   | 5.0 | 2.5 | 29.35 s | 0 | top speed changes nothing |
   | 4.0 or 5.0 | 3.25 | none | 0 | stalled: 1776 safety-stop ticks, cross-track up to 0.65 m |

   These laps start from a standing start, while gap_follow's are flying laps. That accounts for part of the difference, but not 9 s. gap_follow on the same simulator and floor: 20.56 s (its launched settings from run `184826`), 20.71 s (shipped default) and 17.74 s (best time trial). The line and its corner speeds limit the racer, not top speed.
3. **No floor race on record.** No recording since 2026-10-26 has `pure_pursuit_node` or `particle_filter` on the floor. Older folders were checked too: the Aug-Oct `racerbot_auto/*/events.jsonl` are diagnostic-probe logs with no race phase, and the 2026-07-27 folders hold profiled lines but no log. The docs disagree about 2026-07-27: `git-setup.md` calls it a `gap_follow` collision, while `localization.md` says a stale pose put pure pursuit into a wall. The plan corrects whichever is wrong.
4. **The simulator can't yet run car 1 on a measured floor.** The only floor profile, `sfu_course_2026_10_30`, also overrides **car 2's** steering linkage, speed loop and odometry scale. `--car rb1 --floor ...` therefore silently simulates car 2's steering.
5. **racerbot_sim (the only simulator that runs the real nodes) has uncalibrated physics:** stock Gym, 1-tick steering delay, no grip curve, no VESC speed loop. Its lap times can't be compared with real laps today. It can load the real course maps through `track_directory`.
6. **Localization in the no-ROS simulator is assumed:** 2 cm / 0.01 rad noise and a 2-tick delay. One real measurement exists: `particle_filter` on run `185036`'s scans gave 7 cm median, 22 cm p95 and 0.84° (`20261030-185036-fixed/commands.txt`). The filter's motion model reads `/odom`, whose yaw over-reads the gyro by 28-34%.
7. Both real courses are narrow. Car 2's 30 Oct course: 50.7 m, median width 1.42 m, narrowest 0.78 m. Car 1's 26 Oct course: 31.1 m, median 1.52 m, narrowest 1.06 m. racing-autonomy.md measured pure pursuit's corner error at 0.39-0.57 m at 2.5-3.0 m/s, which doesn't fit those widths. The probe's 0.12 m at lower speed suggests that figure is out of date, but it hasn't been re-measured at race speed.

## The bar

| Course (map) | Car | Real gap_follow | Source |
|---|---|---|---|
| 30 Oct, `data/rb2/racerbot_auto/20261030-185036-fixed`, 50.74 m line | rb2 | 20.6-20.9 s laps | `laps.py` on `data/rb2/racerbot_runs/20261030-185036` |
| 30 Oct, same map | rb1 | to be measured | car 1's gap_follow runs of that evening: `data/rb1/racerbot_runs/20261030-175418`, `-175701`, `-180531`, `-180746`, `-190734` |
| 26 Oct, `data/rb1/racerbot_auto/20261026-194514-clean`, about 31 m | rb1 | to be measured | `map_course` holds `20261026-194514`, `-194714`, `-194913` |

**Pass:** in the simulator of record, with that car's calibration on that course, the map-based racer laps faster than gap_follow's real laps, with zero contacts over every seed, judged by the chassis checks (`sim_fidelity`'s distance-transform check, or `SimBridge.body_contact`), never by Gym's collision flag.

**Also reported every time:** gap_follow's best zero-contact setting in the same simulator (its time trial, 17.74 s on rb2), so a "win" is never only a win over gap_follow's 3.5 m/s cap. Whether beating that is required is an open question, below.

## Simulator trust comes first

No comparison counts until simulated gap_follow on the same map is within a few percent of the real laps (the request said "within a few percent of 20.7 s"). Decided:

- **Tolerance: ±3%** of the real median lap, per car and course. (My call.)
- **rb2 on 30 Oct** passed this on 2026-10-31 (20.56 s against 20.70 s). It is re-run on current `main`, because gap_follow has changed since (launch boost, 2026-10-30).
- **rb1 on 30 Oct** needs the floor profile split into floor values (grip curve, LiDAR texture, map walls) and car values (steering linkage and trim, speed loop, command and odometry scales). Car 1 then uses its own car values from `cars/rb1/` on car 2's measured floor. Car 1's own runs from that evening validate it. **rb1's speed calibration is not changed here.** Another session is working on car 1's 2 s gap_follow deficit, probably `speed_to_erpm_gain` 3650 against the odometry's 4040. Until that lands, rb1 results are provisional and carry a named re-run step.
- **rb1 on 26 Oct** has no measured floor. The same split gives rb1's car values with the 30 Oct floor's grip, labelled as an assumption, and it's checked against car 1's real laps the same way.
- **racerbot_sim gets the calibrated plant.** It's the only simulator that runs the real `pure_pursuit_node`, `auto_map_race_node`, particle filter and `gap_follow_node`, so it's the simulator of record for lap times. `SimBridge` takes the same floor and car profile as `FidelityPlant` (reuse it rather than copy it). It has to reproduce gap_follow's real lap within ±3% before its head-to-head counts. The no-ROS simulator stays the fast inner loop for sweeps. (My call: without this, every whole-stack lap time rests on uncalibrated physics.)
- **Localization in both simulators uses measured numbers.** Pose noise and delay come from the particle filter replayed on real bags (below), not from the 2 cm guess.

## The work, by piece (one branch and one PR each, none merged here)

**A. Simulator of record.** Split floor from car values. Calibrate racerbot_sim's plant. Validate simulated gap_follow against real laps for rb2/30 Oct, rb1/30 Oct and rb1/26 Oct, and write down any gap that remains.

**B. SLAM on the real bags.** Replay every listed run through `replay_map_course.py` and `slam_replay.py` on current `main`: car 2's `20261030-184556`, `-184826`, `-185036` and `20261026-174911`; car 1's `20261026-174337`, `-180228`, `-194422`, `-194514`, `-194714` and `-194913`. Score them with `slam_metrics`/`slam_score`/`slam_report` and the rebuild verdict. Report per course: walls, see-through, loop-closure jumps, laps closed, and heading at closure. Change `slam_gyro.yaml` only if a course fails, and then only with a course-size rule tested as an invariant (chain × node spacing fits the search radius). Car 1's gyro-off runs (`174337`, `180228`) are reported, not tuned for, because gyro mapping is the default. The closure fix is verified on car 1's runs: heading at closure under the 15° quality limit.

**C. Localization on the real bags.** Run the real `particle_filter` (it needs `range_libc`, which is CUDA-only on the car, so build a CPU variant for the dev box) on the saved maps. Truth is `loc.py`'s scan-to-map track. Measure position and heading error against speed, the 30 Oct run that reached 5.4 m/s (`175728`) included, plus delay from scan stamp to pose. CPU timing here isn't the Jetson's, so record it as a dev-box figure and leave the Jetson timing to the floor plan. Try the filter's motion model on gyro odometry (`/odom/imu`) instead of steering odometry. Ship it if it's more accurate. Feed the measured noise and delay into both simulators.

**D. Line, speed and steering.**
- Grip from the real runs (`gripcurve.py`, both cars), turned into the profile's limits with a stated margin.
- The minimum-curvature optimizer (Phase 4b) run from the map's centerline, not gap_follow's path, with a wall margin = half the car's width + the measured p99 tracking error + the map's wall error. Where a pinch point leaves no room for that margin, the line takes the centre there instead of failing.
- A steering law and lookahead chosen with `law_compare.py`/`law_sweep.py` on the real courses with measured localization noise, judged on corner cross-track p95 at race speed. The existing promotion rule is unchanged.
- The same profile path is used by `rebuild_run.py` (the `race_launch.py run_dir:=` route) and by `auto_map_race`'s race phase, so both routes race the same thing.

**E. Head-to-head.** In the simulator of record, on each course with each car's calibration: gap_follow (shipped, and time trial) against the map-based racer (`race_launch.py run_dir:=` and `auto_map_race` end to end). At least 6 seeds while tuning; the final table uses 30 seeds in both directions, as sim-calibration.md does. Contact is judged by the chassis checks only. A setting that wins with zero contacts becomes the shipped default.

**F. Floor test plan,** written for the person at the car, per car: wheels off with LB held, then the floor at low speed, then race speed. Each step has the command, what "working" looks like, the abort signs, and `laps.py` after every run.

## Constraints

- Every node that can move the car keeps its LB deadman and its four deny-path tests. `enable_deadman`, TTC, swept clearance, emergency clearance, `emergency_stop_distance` and `body_contact` are never relaxed. Faster comes from the line, the profile, the steering law and localization, not from the safety net.
- CLAUDE.md's Test Quality rules apply to every new or changed test: each is seen to fail against a break, and no test is weakened.
- ROS, pytest and the simulators run through `rb <cmd>` (domain 79, localhost). Simulations run at `--workers 2`. Check the box's load first, because it's shared.
- rb1's calibration files are not edited.
- No floor runs, no hardware. The car gets a hand-off prompt (memory `car-handoff-prompt`), not commands from here.

## Non-goals

- Overtaking and opponent handling. Every comparison is solo.
- MPC on the Jetson beyond the existing benchmark. If a law needs `osqp`, it already has an install note.
- A minimum-time optimizer beyond minimum curvature. Only if D's measurements show minimum curvature is what limits us.
- Car 1's speed calibration.

## Anti-goals (a failure even if the numbers look good)

- A win that comes from relaxing a safety layer, or from a contact check that doesn't see the chassis.
- A win in a simulator that hasn't reproduced gap_follow's real lap on that course.
- rb1 results presented as final while car 1's speed calibration is still moving.
- A line that's faster only because it apexes inside the measured tracking error.

## Open questions for the tech lead

1. **Is "beats gap_follow's real 20.6-20.9 s" the bar, or must the map-based racer also beat gap_follow's best simulated setting (time trial, 17.74 s on rb2)?** My recommendation: require both. The real laps were capped at 3.5 m/s, and team A will not be.

## Decisions made here (check these)

- ±3% sim-to-real tolerance on lap time.
- racerbot_sim is calibrated and becomes the simulator of record. The no-ROS simulator stays the fast tuning loop.
- Car 1 on the 30 Oct floor uses car 2's measured grip with car 1's own steering and speed values. Labelled as an assumption until car 1's own floor is measured.
- Car 1's 26 Oct course is a second required course for rb1.
- A CPU build of `range_libc` on the dev box, for localization replay only. The car keeps its CUDA build.
- Six pieces, A to F, in the order A → (B, C in parallel) → D → E → F.
