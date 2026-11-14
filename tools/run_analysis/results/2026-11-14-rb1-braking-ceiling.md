# Car 1's braking ceiling (plan Task 14), 2026-11-14: keep max_braking_decel 3.0

gap_follow's clearance cap assumes it can brake at `max_braking_decel` (3.0 m/s²) after `braking_latency_sec`
(0.25 s). The question was whether car 1 brakes harder than that, so the cap could allow more speed.

## Real stops

`stops.py` on rb1 runs 20261111-153721 and -155542, stops from above 1.5 m/s with a speed command at or near zero.
Distances are measured on the wheels; the wall check disagreed on most, so they are wheel-measured.

| run | stop | wheel distance | model 3.0 / 3.5 / 4.0 m/s² (+0.25 s) |
|---|---|---|---|
| 153721 | 3.19 -> 0.27 m/s in 0.92 s | 1.78 m | 2.49 / 2.25 / 2.07 |
| 153721 | 2.84 -> 0.29 m/s in 0.90 s | 1.50 m | 2.05 / - / 1.71 |
| 153721 | 2.93 -> 0.28 m/s in 1.00 s | 1.87 m | 2.16 / 1.96 / **1.80** |
| 155542 | 2.84 -> 0.24 m/s in 0.70 s | 1.07 m | 2.06 / - / 1.72 |
| 155542 | 2.51 -> 0.28 m/s in 0.68 s | 0.92 m | 1.68 / - / 1.42 |
| 155542 | 2.17 -> 0.00 m/s in 0.70 s | 0.83 m | 1.33 / - / 1.13 |

A full stop (command near zero) is shorter than the 3.0 model on every stop, and every stop fits a 3.5 model;
one does not fit 4.0 (1.87 m against 1.80). That is braking to zero, where the excess over the command is large. For
small excesses car 1's VESC hardly brakes at all (`2026-11-13-corner-entry-study.md`), and the clearance cap's
commands are rarely zero.

## Simulator

gap_follow, rb1 with its fitted speed loop (`cars/rb1/sim/speed_loop.yaml`), opponent awareness on, one lap:

| max_braking_decel | car2_course_2026_11_11, 6 seeds | reversed, 4 seeds | indoor, 10 runs |
|---|---|---|---|
| 3.0 (shipped) | 11.74 s, min clearance 0.187 m | 10.91 s, 0.184 m | 11.55 s, 0.155 m, no contact |
| 3.5 | 11.72 s, 0.182 m | 11.08 s, 0.144 m | 11.63 s, **0.019 m** |
| 4.0 | 11.77 s, 0.182 m | 11.06 s, 0.140 m | 11.37 s, **contact on 1 of 10** |

No lap is faster by more than 0.02 s, both directions lose clearance, and 4.0 makes contact.

## Answer

Keep `max_braking_decel: 3.0` on car 1. The clearance cap is not what limits car 1's lap in the simulator once its
own speed loop is modelled, and the real stops give no margin for raising it at the commands the cap actually sends.
Car 2 was not re-measured here (its runs are on car 2).
