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
