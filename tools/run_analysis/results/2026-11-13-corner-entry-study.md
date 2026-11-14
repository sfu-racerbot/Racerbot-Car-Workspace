# Corner-entry study: rb1 run 20261111-155542, the U-turn wall contact at 87.34 s (intent_report clock)

Measurement and harness only; no controller change. Tools: `tools/run_analysis/speed_tracking.py` (real runs and `eval_gap --trace` files),
`tools/f1tenth_sim/event_replay.py` (start the plant at the recorded state), `tools/run_analysis/corner_shed.py`.

## C1/C2: does the speed follow the command? (real 155542, real 153721, simulator rb1 car2_course_2026_11_11, 6 seeds, awareness on)

Command = /drive (real) or the controller's command (sim); speed = /odom wheel speed (real) or the measured speed (sim); LB-held spans only.

| | real 155542 | real 153721 | sim |
|---|---|---|---|
| a. dv/dt median, e = v - cmd 0.2-0.5 m/s (n) | -0.01 (1223) | -0.26 (848) | -1.08 (502) |
| a. e 0.5-1.0 | -1.08 (1585) | -1.06 (1381) | -1.96 (342) |
| a. e 1.0-1.5 | -2.25 (581) | -2.36 (526) | -3.23 (127) |
| a. e > 1.5 | -2.74 (494) | -2.63 (592) | -3.09 (42) |
| b. step-downs: n, reached within 0.3 m/s | 34, 21 | 28, 12 | 25, 25 |
| b. time to within 0.3 m/s, median / p90 (s) | 0.52 / 0.71 | 0.54 / 0.72 | 0.33 / 0.38 |
| b. not reached: remaining error at 0.4 s (m/s) | 13 events, 0.44-1.38 | 16 events, 0.34-1.36 | none |
| c. overshoots n (rise >= 1.0 in 0.6 s, fall >= 0.5 in 0.4 s) | 72 | 66 | 39 |
| c. max(v - cmd) over 1.0 s, median / p90 (m/s) | 1.54 / 2.67 | 1.62 / 2.62 | 0.88 / 1.72 |
| c. time v > cmd + 0.3, median / p90 (s) | 0.72 / 0.88 | 0.81 / 0.90 | 0.30 / 0.51 |
| d. command jitter at > 1.5 m/s, p50 / p90 (m/s per tick) | 0.084 / 0.439 | 0.098 / 0.389 | 0.093 / 0.259 |

Overshoot event times on 155542 include 86.11 and 87.32 (the approach to the contact) and 84.2. Part d's binding-cap-change fraction is not
computed: the run's recording has no decision log with the cap.

## C3: the recorded state replayed in the simulator

Start: fixture steer_into_wall_rb1_20261111-155542.npz index 44 = clock 85.606 s, pose x -7.358, y 9.138, yaw 8.021 rad (1.738), wheel speed
0.358 m/s, steering 0. Start chassis-to-map clearance on the sim's track map: 0.403 m (free space, >= 0.2 m). The mirror is cold-started
there with the fixture's recorded parameters (static gate off, as recorded), 3.0 s, 6 seeds (12345-12350) each.

| | real | sim, awareness on (6 seeds) | sim, awareness off (6 seeds) |
|---|---|---|---|
| speed at +0.7 s (m/s) | 2.51 | 2.26-2.33 | 2.26-2.33 |
| speed at +1.3 s (m/s) | 2.88 | 1.67-1.69 | 1.67-1.69 |
| max(v - cmd) (m/s) | 2.22 up to the contact (about 1.3 over 86.2-87.3 s) | 0.71-1.83 | 0.71 |
| minimum chassis-to-map clearance (m), at | contact at +1.74 s near x -9.6, y 11.5 | 0.180-0.206 at +2.5-2.7 s, x -10.45..-10.77, y 11.14-11.32 | 0.180-0.206 at +2.6-2.7 s |
| contact (independent chassis check) | yes | no (0 of 6) | no (0 of 6) |

Seed 12345, every 0.25 s (t from the start; real = recorded run), awareness on:

| t | real v | sim v | real cmd | sim cmd | real steer | sim steer |
|---|---|---|---|---|---|---|
| 0.00 | 0.36 | 0.36 | 0.50 | 0.00 | +0.019 | +0.000 |
| 0.25 | 0.72 | 0.91 | 1.76 | 2.88 | +0.032 | -0.000 |
| 0.50 | 1.95 | 2.02 | 2.99 | 2.60 | +0.099 | -0.000 |
| 0.75 | 2.43 | 2.16 | 1.69 | 1.62 | +0.304 | +0.250 |
| 1.00 | 2.64 | 1.80 | 2.22 | 1.62 | +0.181 | +0.330 |
| 1.25 | 2.85 | 1.68 | 2.57 | 1.62 | +0.029 | +0.330 |
| 1.50 | 2.81 | 1.67 | 2.07 | 2.08 | +0.207 | +0.166 |
| 1.75 | 2.48 | 2.28 | 1.96 | 2.71 | +0.121 | +0.000 |
| 2.00 | 1.73 | 2.65 | 0.71 | 2.17 | -0.330 | +0.189 |

(Full table to +3.0 s: `event_replay.py` output.) The cold-started controller commands 1.62 m/s from +0.75 s while the real node commanded
2.2-2.6, with the real node's own history; the simulated speed follows its command, the real speed does not.

## C4: the three answers

1. **Does the real car brake to a lower command, and how fast?** Only partly. For a speed excess of 0.2-0.5 m/s over the command it does
   not brake at all (median dv/dt -0.01 m/s^2 on 155542, -0.26 on 153721); for 0.5-1.0 m/s it brakes at -1.08 m/s^2; for 1.0-1.5 at -2.25;
   above 1.5 at -2.74. After a command step-down of >= 0.8 m/s that holds 0.4 s, the speed gets within 0.3 m/s of the new command in a median
   0.52 s, and in 13 of 34 events (155542) it has not by the time the command moves again (remaining error 0.44-1.38 m/s). The simulator
   brakes harder (-1.08 / -1.96 / -3.23 / -3.09) and always reaches (25 of 25, median 0.33 s).
2. **Does the simulator reproduce the overspeed and the contact from the real state?** No. Twelve runs (6 seeds, awareness on and off) start
   at the recorded pose and speed in free space (0.403 m) and none touches: minimum clearance 0.180-0.206 m about 2.6 s in, 1-1.2 m further
   round the turn than the real contact. Speed at +1.3 s is 1.68 m/s against the real 2.88: the cold-started controller commands 1.62 m/s there
   (real 2.2-2.6 with its history) and the simulated speed loop tracks it (max(v - cmd) 0.71-1.83 m/s against 2.22 real). The overshoot
   statistics above show the same direction: real max(v - cmd) 1.54 median against the simulator's 0.88.
3. **First clock time the real command was above what the car could still shed before the corner.** From `corner_shed.py` (the wheel speed
   integrated to the contact, 4.31 m from 84.5 s; v_max = sqrt(v_c^2 + 2 a d)): with the measured braking for a 0.5-1.0 m/s excess
   (a = 1.08 m/s^2) and a corner speed v_c = 1.5 m/s, the command first exceeded v_max at 86.13 s and the speed itself at 86.56 s. The
   neighbouring cases: a = 0.65 (the deceleration the car actually achieved at 86.2-87.3 s): command 86.04 s, speed 86.24 s (v_c 1.5); a = 2.25:
   86.80 s / 86.86 s; with v_c 2.0 and a = 1.08: 86.74 s / 86.76 s. The corner speed v_c is an assumption (the simulator passes this corner at
   2.1 m/s with 0.19 m of clearance); every other number is measured.

## Notes

- The real max(v - cmd) is 2.22 m/s over the whole approach (zero-order-hold /drive, wheel /odom), about 1.3 over 86.2-87.3 s.
- The replay is cold-started; the controller's state before 85.6 s (history of the 0.5 m/s crawl) is not reproduced.
