# Gap follow time trial pace on the race-day course (plan Phase 4): where the 11.0 s goes

Simulator, `car2_course_2026_11_11`, floor `sfu_course_2026_11_11_rb2`, car 2's shipped gap follow plus `TIME_TRIAL`
(`gap_follow/launch_speed.py`), 6 seeds. Target in the plan: 10.6 s with zero contact, no course slower.

Baseline: **11.0 s** (10.98-11.05), no contact, minimum chassis clearance 0.150 m. Time per lap by the binding speed
cap: curve 5.4 s, clearance 3.3 s, acceleration 2.0 s, corner 0.2 s.

## Single-parameter changes (median lap, min clearance)

| change | lap (s) | min clearance (m) |
|---|---|---|
| none | 11.02 | 0.150 |
| `max_lateral_accel` 3.0 → 3.2 / 3.4 | 11.02 / 11.00 | 0.150 / 0.141 |
| `max_acceleration` 5 → 6 | 11.03 | 0.158 |
| `corner_speed` 2.14 → 2.5 | 11.04 | 0.158 |
| `swept_clearance_margin` 0.05 → 0.02 | 10.99 | 0.150 |
| `enable_corner_setup` true | 10.89 | **0.112** |
| `braking_latency_sec` 0.25 → 0.15 | 10.85 | 0.141 |

None reaches 10.6 s. The two that help most are either a measured value (braking latency is the car's command plus
speed-loop response, not a margin to tune) or trade 4 cm of clearance for 1% (corner setup). None is shipped.

## Why the lateral limit does not help

While curve-bound, the steering asks for a median curvature of 0.95 /m while the car's path (yaw rate over speed)
curves at 0.38 /m (0.5 s smoothed: 0.38). The steering changes sign 1.3-1.9 times a second above 1 m/s. Raising
`max_lateral_accel` by 13% raises the cap but not the lap: the car is already at the floor's grip in those corners,
and what the cap measures is the weave, not the corner. The simulated lap drives 30.4-31.0 m of a 29.24 m centre line.

So the remaining time is in the path, not the limits: a steadier steering command (less weave, a cap computed from
the curvature the car can deliver rather than what the gap asks for) is algorithm work, the kind Phase 4's plan
expected, and it changes how close gap follow runs to walls in corners. It needs the tech lead's direction before it
starts.

Reproduce: `eval_gap.run_track('car2_course_2026_11_11', params, seed, 90, calibration=with_floor(
'sfu_course_2026_11_11_rb2', calibration_for_car('rb2')), car_id='rb2', trace=...)` with
`params = gap_mirror.node_params(car_id='rb2') | TIME_TRIAL`.

## Update (2026-11-14, evening): standing against flying laps, and a car 2 time-trial step

The 11.0 s above is a **standing-start** lap. `eval_gap.run_track(..., max_laps=N)` now reports each lap
(`lap_times_s`): car 2's flying laps on this course are already **10.10-10.25 s** (median 10.16). Real race-day lap
times are flying laps, so the 10.6 s target compares against a flying lap; the simulator is optimistic for car 2's
gap follow, though: its flying mean speed (about 3.0 m/s) is about 10% above the real run's 2.71 m/s (see the
correction in `2026-11-14-rb2-race-day-agreement.md`). Pace changes are therefore judged by their gain against the
simulator's own baseline, on both laps.

Advisor's sweep (Fable 5.2, scratch mirror; repo untouched): smoothing the aim or capping on filtered or delivered
curvature was slower or crashed; the hairpins are grip-limited (yaw rate 1.77 rad/s at full lock whatever the cap).
What helped: a softer `steering_gain` with a faster command slew.

**Shipped for car 2 only (`launch_speed.TIME_TRIAL_BY_CAR`): `steering_gain` 0.8, `max_steering_rate` 4.0 with
`time_trial:=true`.** 16-course suite, 3 seeds, 2 laps each (median standing / flying):

| course | baseline | car 2 time trial |
|---|---|---|
| car2_course_2026_11_11 (rb2 floor) | 11.00 / 10.18 | 10.78 / 9.88 |
| indoor_oval, indoor_tight, indoor_wide | 9.28 / 8.35, 8.93 / 8.20, 11.12 / 10.20 | 9.03 / 8.12, 8.70 / 8.05, 10.85 / 10.07 |
| asb_10000, asb_sharp | 13.65 / 13.03, 13.90 / 13.22 | 13.47 / 12.80, 13.95 / 13.28 |
| narrow_1p2, narrow_1p0 | 9.38 / 8.62, 9.88 / 9.25 | 9.05 / 8.35, 9.50 / 8.90 |
| car1_course_2026_10_26 | 10.75 / 10.12 | 10.60 / 10.25 |
| car2_course_2026_11_06a, _scans | 11.40 / 10.47, 11.35 / 10.60 | 11.12 / 10.40, 11.10 / 10.47 |
| car2_course_2026_11_06b, _scans | 8.57 / 7.80, 8.53 / 7.67 | 8.40 / 7.62, 8.43 / 7.67 |
| course_2026_10_30, _scans | 17.30 / 16.25, 16.65 / 15.70 | 16.75 / 15.57, 16.23 / 15.05 |
| room_2026_07_27 | 6.80 / 5.92 | 6.30 / 5.47 |

Zero contact before and after on every course; minimum clearance on the race-day course 0.150 -> 0.100 m. Two small
regressions: asb_sharp +0.05 s, car1_course_2026_10_26 flying +0.13 s. Car 1 keeps the shared time trial: the same
change made contact on course_2026_10_30, its _scans version and asb_sharp with car 1's calibration.

## Update 2026-11-15: #71 reverted after the floor matrix

The suite above ran each course on one floor. `tools/f1tenth_sim/floor_matrix.py` (#79) runs every map course on car 2's
own calibration and on the race-day floor `sfu_course_2026_11_11_rb2`, each at grip scale 0.85, 1.0 and 1.15, 3 seeds,
2 laps from a standing start, and compares cell by cell (no new stall or contact, minimum chassis clearance no lower
than the reference's minus 0.03 m):

    python3 tools/f1tenth_sim/floor_matrix.py --car rb2 --time-trial \
        --arm 'shared:steering_gain=1.0,max_steering_rate=2.0' --arm 'p71:' \
        --track maps --floors car,sfu_course_2026_11_11_rb2 --grip 0.85,1.0,1.15 --seeds 3

#71 was worse in 32 of 72 cells. The ones that decide it, all on the race-day floor:

| course, grip | shared time trial | #71 |
|---|---|---|
| `course_2026_10_30`, 0.85 / 1.0 | 0 contact, 0.05 m | 3 / 2 of 3 contact |
| `course_2026_10_30_scans`, 0.85 | 0 contact | 3 of 3 contact |
| `car2_course_2026_11_06b`, 1.0 | 0 contact, 0.10 m | 2 of 3 contact |
| `car2_course_2026_11_06a(_scans)`, 0.85 | 2 / 1 stalls | 3 / 3 stalls |
| `car2_course_2026_11_11` (race day), 1.0 | 0.15 m | 0.10 m |

It is faster where it finishes (room 5.9 -> 5.5 s, 10-30 on car 2's own floor 16.25 -> 15.58 s), but the gain does not
survive a floor change, so `TIME_TRIAL_BY_CAR` is empty again; the per-car mechanism stays, and the time trial's
opponent-awareness refusal now always wins over a per-car entry.

The shared time trial is not clean either on the race-day floor: contact on `course_2026_10_30_scans` (1 of 3 at
grip 1.0, 3 of 3 at 1.15), on `course_2026_10_30` at 1.15 (1 of 3), and on `levine_obs` on both floors. That is open.
