# Car 1's U-turn contact (run 20261111-155542, 87.34 s): an obstacle the LiDAR never saw, not a spin

This corrects the 2026-11-14 ~02:00 reading (post-race ledger, PR #62 description) that car 1 spun at the U-turn because
the rear let go at about 4.4 m/s² of lateral demand. That reading took the yaw-rate jump for the cause of the contact; the
timing says it is the result. Plan task 13 ("why the car got 5 cm from the wall").

## What the run shows

Times are seconds after the event replay's start (85.60 s on intent_report's clock); contact at +1.74 s = 87.34 s
(`2026-11-13-corner-entry-study.md`, independent chassis check). Gyro is `/imu/data`, scans and `/drive` are the fixture
`src/gap_follow/test/data/steer_into_wall_rb1_20261111-155542.npz`.

| t (s) | wheel speed (m/s) | /drive steering (rad) | yaw rate (rad/s) | asked lateral v²tan(δ)/L | delivered v·r |
|---|---|---|---|---|---|
| 1.50 | 2.81 | +0.207 | 1.17 | 4.60 | 3.29 |
| 1.60 | 2.66 | +0.198 | 1.27 | 3.96 | 3.38 |
| 1.70 | 2.63 | +0.150 | 1.35 | 2.91 | 3.55 |
| 1.72 | 2.63 | +0.152 | 1.33 | | |
| **1.74** | | | **2.32** | contact | |
| 1.76 | | +0.100 | 4.36 | | |
| 1.80 | 2.42 | +0.020 | 5.18 | | |

- Up to the contact the car **understeers** (asked 4.0-4.6 m/s², delivered 3.3-3.4), the opposite of a spin.
- The yaw rate is steady at 1.33 rad/s until exactly the contact time, then rises 1.33 → 4.36 rad/s in 40 ms
  (about 75 rad/s²) while the steering straightens: an impact on the left side turning the car, not a tyre force.

## What the car could see

- `tools/run_analysis/steer_stages.py` (mirror replay, /drive parity: max 0.0055 rad over 32 scans, 86.6-87.4 s): the gap
  stage aimed 16-19° left the whole approach; centering was 0 by design (it fades out past 15° of bearing and under 1.5 m
  of forward depth); the arc guard corrected by up to -0.083 rad at 86.96-87.06 s and then found the arc clear.
- Re-running the swept-path check on each scan from +1.20 to +1.72 s, with the guard's ±90° window and with the full 270°
  scan, on the commanded arc and on the arc the gyro says the car drove: every one is clear (free length ≥ 1.84 m or
  inf). The body rectangle moved along the driven arc for 1 m meets no scan point either.
- Accumulating every scan from 1 s before the start to +1.70 s into world coordinates (the fixture's poses): at the
  contact pose the closest point the LiDAR **ever** returned is **0.17-0.18 m** from the chassis's left side.

So the car hit something no scan contains. The likeliest is part of the inner barrier below or through the LiDAR plane
(the scan plane is about 0.11 m up), which also fits two earlier findings: car 2's map is missing part of the inner
barrier at this U-turn, and car 1's beams end 0.1-0.15 m inside the barrier line its own runs suggest. Every stage acted
correctly on what it was given.

## Consequences

- **Brake assist (PR #62)** was motivated by an overspeed spin. With no spin there, its case rests only on the general
  speed-tracking overshoot, and the simulator says it costs time and clearance; it stays a draft.
- **Saturating tyres** (`tools/f1tenth_sim/sim_fidelity/tyres.py`, off by default) were built to reproduce the spin. With
  car 1's measured peak (about 4.7 m/s², friction 0.45-0.50) and any rear/front balance tried, the warm event replay
  peaks at 1.9-2.3 rad/s and makes no contact, as the reading above predicts. They stay available (`event_replay.py
  --tyres FRONT,REAR,SHAPE`) for limit-handling work; nothing uses them by default.
- **For the tech lead:** look at the inside of the U-turn barrier at about (-9.6, 11.5) on car 2's map (the inner end of
  the U-turn barrier) for anything low or thin, such as a foot, a base plate or a cable. If it is part of the track, the
  controllers need a wider inner margin there (`arc_guard_margin`, now 0.05 m) or the map racer's line needs that wall
  drawn in; neither can be chosen from the scans alone.

Reproduce:

    rb python3 tools/run_analysis/steer_stages.py data/rb1/racerbot_runs/20261111-155542 --from 86.6 --to 87.4
    rb python3 tools/f1tenth_sim/event_replay.py --seeds 2 --warm data/rb1/racerbot_runs/20261111-155542 [--tyres 0.48,0.44,1.6]
