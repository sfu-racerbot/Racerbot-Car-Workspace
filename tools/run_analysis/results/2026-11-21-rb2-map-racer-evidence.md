# Car 2: what the November map-racer bags establish

**The Nov 9 heading finding partly reproduces.** LiDAR matched to the saved map agrees much more closely with the gyro than with steering-model odometry during turn-in. The reported 4–7° particle-filter lag depends on how timestamps are paired; the exact simultaneous 17° odometry discrepancy is not independently established. Three bags record a clearance stop in the same bend. That does not establish four wall impacts, an opponent, or what caused the incident.

## Nov 9 · primary run 190202

**Inputs:** [Original MCAP and run metadata](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/tree/76db7abecd8c66ca14bbc56d77e82331a08e3cb6/data/rb2/racerbot_runs/20261109-190202), [recorded controller parameters](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/blob/76db7abecd8c66ca14bbc56d77e82331a08e3cb6/data/rb2/racerbot_runs/20261109-190202/metadata/params/pure_pursuit_node.yaml), [map and racing line](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/tree/76db7abecd8c66ca14bbc56d77e82331a08e3cb6/data/rb2/racerbot_auto/20261109-185146-rebuilt), [claim under review](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/44). The recorded occupancy grid matches the saved map cell for cell (420 × 329 at 0.05 m/cell, including unknowns; same origin). All four Nov 9 bags’ controller startup messages name this same 289-waypoint line. In 190202 the filter used `/odom`; the controller used the filter’s laser-frame pose, steering law `delay_lqr`, a 5.0 m/s speed limit and 3.1 m/s² lateral-acceleration limit. Opponent overtaking and the local pass planner were off.

**Wheel-speed finding reproduced.** The recorded wheel-speed estimate fell from 4.909 to 3.066 m/s in 0.378 s. This confirms the published “4.91 to 3.07 in about 0.4 s”; the endpoint slope (4.87 m/s²) is not an independent measurement of chassis braking capacity.

| Heading comparison against scan-to-map alignment | Steering odometry: max error in heading change | Integrated gyro: same metric | Filter: median / max absolute heading error¹ |
|---|---:|---:|---:|
| Two seconds ending at 3.07 m/s (0.468–2.468 s; 80 scans) | 47.13° | 1.84° | 0.74° / 2.34° |
| Braking interval only (2.090–2.468 s; 15 scans) | 22.94° | 0.68° | 0.91° / 1.98° |

¹ Filter heading is already in the map frame; values use header-time interpolation. Gyro and odometry instead compare **change** in heading, aligned at the start of each row. Times are seconds from the first recorded LB-held joystick message, not from the start of the bag. The original analysis gives neither a time origin for its two-second window nor an alignment pose. These are independently defined windows, not a claim to have repeated its exact sample selection.

**Why the filter number changes:** In the later right turn at 3.05–3.50 s, using the latest filter heading received as of each scan gives a median 4.35° lag (peak 6.56°). Interpolating the filter by header time gives 2.79° (peak 4.65°) instead. The former comparison includes message age; neither is the exact pose known to have been consumed by the controller. Between these windows, filter, wheel, gyro, drive, and intent have 0.33–0.37 s gaps in bag receipt; the sensor headers also have gaps, while scan headers continue. Do not interpolate across that interval as if vehicle motion were measured.

**How the reference was built:** Match LiDAR endpoints to the exact map’s distance field at scan-midpoint time, using the recorded 0.26 m laser offset. Initialize from a global map search near the racing line, propagate with previous scan fits, then correct within-scan motion using velocities estimated from those fits. Neither gyro, wheel odometry nor filter poses drive this reference. A second, filter-seeded fit checks for a different solution. In the pre-gap two-second interval at least 98.8% of retained endpoints were within 0.10 m of map obstacles; the two initializations differed by at most 0.59° before motion correction. The reference is map-based, not surveyed ground truth. Integrate the raw VESC z gyro in its native deg/s after subtracting the parked two-second median bias; applying the recorded mount tilt changes its two-second heading change by less than 0.02°.

## What actually stopped

| Nov 9 bag | First recorded zero-speed body-clearance intent | Nearest waypoint · first clearance |
|---|---|---|
| [190202](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/tree/76db7abecd8c66ca14bbc56d77e82331a08e3cb6/data/rb2/racerbot_runs/20261109-190202) | 3.733 s; first actual zero-speed drive command 3.691 s | 107 · −0.162 m |
| [190356](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/tree/76db7abecd8c66ca14bbc56d77e82331a08e3cb6/data/rb2/racerbot_runs/20261109-190356) | 0.582 s; recording starts moving | 96 · −0.077 m |
| [190617](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/tree/76db7abecd8c66ca14bbc56d77e82331a08e3cb6/data/rb2/racerbot_runs/20261109-190617) | None recorded; already moving at first odometry. Avoidance recorded. | Not available |
| [190837](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/tree/76db7abecd8c66ca14bbc56d77e82331a08e3cb6/data/rb2/racerbot_runs/20261109-190837) | 3.007 s | 92 · −0.058 m |

In 190202 at 3.046 s the controller called a return at 0.70 m “map-unexplained” and capped speed at 1.0 m/s. Later, scan returns crossed the rectangular vehicle envelope’s 0.030 m stop threshold. A negative clearance means a return lies inside the *configured* envelope; it is not measured impact penetration. The bag does not identify the object. The different stop time in the saved run report is from replay through a **gap-follow** model; the figures above are the recorded **pure-pursuit** decisions and commands.

## Nov 11 · independent positive control for replay

[Run 152132](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/tree/76db7abecd8c66ca14bbc56d77e82331a08e3cb6/data/rb2/racerbot_runs/20261111-152132) reproduces the [published flank stop](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/blob/76db7abecd8c66ca14bbc56d77e82331a08e3cb6/tools/run_analysis/results/2026-11-14-map-racer-stuck-case.md): 30.496 s between first and last contiguous body-clearance intents, 1,220 zero-speed drive commands and zero wheel speed throughout. Across 1,218 raw scans the median body clearance was 0.022 m (range 0.015–0.028); the closest beam was always on the right (−84.5° to −44.5°). Within ±15° ahead the body had at least 0.715 m of clearance. This establishes a flank-triggered stop, not a guarantee the planned path ahead was clear.

Its recorded grid (328 × 432) and controller parameters select [151026-rebuilt](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/tree/76db7abecd8c66ca14bbc56d77e82331a08e3cb6/data/rb2/racerbot_auto/20261111-151026-rebuilt), with a 186-waypoint line, not the [151125](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/tree/76db7abecd8c66ca14bbc56d77e82331a08e3cb6/data/rb2/racerbot_auto/20261111-151125) line with 191. The filter used gyro odometry on Nov 11. A different course and different run are not a controlled test that changing odometry fixes the Nov 9 incident.

**Use in simulation:** Wheel-speed change, turn-in heading change, and the Nov 11 stop duration/clearance are measurable replay targets. Treat the [friction-circle interpretation](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/commit/c1ed0daca72a49e99817ba3f330c7d9f134a7207) as a hypothesis, not a measured crash cause or a validated fix. No car, simulator, controller or repository data was changed.

## Reproduce offline

From the workspace root with NumPy, SciPy, Pillow, PyYAML, mcap and mcap-ros2-support installed:

```sh
python -B tools/run_analysis/studies/rb2_map_racer_2026_11/replay.py
```

The script reads the tracked MCAPs in place, verifies the map and line before reporting each run, and prints measurements without saving new run data. No ROS graph, hardware, or simulator is required. To replay one bag, add e.g. `--run 20261109-190202`.
