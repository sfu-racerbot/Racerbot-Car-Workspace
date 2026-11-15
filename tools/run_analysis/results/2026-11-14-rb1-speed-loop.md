# Car 1's speed loop in the simulator, and why the U-turn hit still does not reproduce (2026-11-14)

Follows `2026-11-13-corner-entry-study.md`, which found that rb1 hardly brakes for a small overspeed and that the
simulator, braking harder, did not reproduce the wall hit at 87.34 s of run 20261111-155542.

## 1. The fit

`tools/run_analysis/fit_speed_loop.py` replays the speed loop alone against both rb1 race runs (20261111-155542,
-153721): the recorded `/drive` speed in, the wheel speed out, on the plant's 0.025 s grid. It uses the same function
the plant calls (`sim_fidelity.actuation.speed_loop_accel`), with 1 s horizons restarted at the real speed every
0.25 s so that model error cannot drift. The score is the wheel-speed RMS over samples with a command of at least
0.3 m/s. The grid covers the acceleration gain and limit, the braking deadband, gain and limit, and the transport
delay (864 sets). The output, with the top ten, is `tools/f1tenth_sim/sim_fidelity/sfu_course_2026_11_11_speed_loop_rb1.json`.

| loop | accel gain, max | brake deadband, gain, max | delay | RMS all | accelerating | braking |
|---|---|---|---|---|---|---|
| shared (run 20261019-200703) | 5, 4.5 | 0, 4, 4 | 2 | 0.361 | 0.315 | 0.385 |
| best without a deadband | 14, 5.5 | 0, 2, 4 | 2 | 0.231 | 0.238 | 0.234 |
| **chosen** | **20, 5.5** | **0.2, 3, 4** | **3** | **0.222** | **0.242** | **0.216** |

Most of the gain is the acceleration side and the delay. The deadband adds a further 8% on braking samples.

Rows a-c of the corner-entry study, from a replay of each whole LB hold (no restarts):

| | real (both runs) | shared loop | chosen loop |
|---|---|---|---|
| a. braking median, excess 0.2-0.5 / 0.5-1.0 / 1.0-1.5 / >1.5 m/s | -0.09 / -1.06 / -2.31 / -2.65 | -1.11 / -2.33 / -3.61 / -3.23 | -0.48 / -1.20 / -2.54 / -3.00 |
| b. step-downs reached within 0.3 m/s; median time | 33 of 62; 0.52 s | 61 of 67; 0.35 s | 33 of 67; 0.65 s |
| c. overshoot after a ramp: median max(v - cmd), time over cmd + 0.3 | 1.56 m/s, 0.75 s | 1.24 m/s, 0.47 s | 1.66 m/s, 0.75 s |

Rejected on the same data:
- **A coasting deceleration inside the deadband** (0.3-1.0 m/s²): worse at every setting.
- **A PI loop with integral windup** (integral gain 2-8 /s², clamps 0.25-1 m/s): worse at every setting. The overshoot
  is explained by the delay plus the deadband, not by windup.

The chosen values are in `src/racerbot_car/cars/rb1/sim/speed_loop.yaml`. Only the simulator reads that file
(`sim_fidelity.calibration.car_speed_loop`), and car 2 has none, so it keeps the shared loop.

## 2. Closed loop: rb1 gap follow on car2_course_2026_11_11, 6 seeds, 1 lap, opponent awareness on

| loop | lap (s) per seed | mean | min body clearance (m) | contact | follow ticks |
|---|---|---|---|---|---|
| shared (main ec26682) | 13.53 14.88 12.75 15.05 15.98 12.80 | 14.17 | 0.172 0.170 0.178 0.161 0.150 0.164 | 0 | 82 132 48 159 183 55 |
| car 1's own | 11.75 11.38 12.50 11.50 11.13 12.20 | **11.74** | 0.216 0.202 0.190 0.189 0.213 0.187 | 0 | 60 52 99 57 31 86 |

With car 1's stronger acceleration the simulated car recovers from each slow-down sooner. The simulator was
under-rating car 1 by about 2.4 s a lap on this course.

## 3. The U-turn event replayed from the real state, warm

`tools/f1tenth_sim/event_replay.py --warm data/rb1/racerbot_runs/20261111-155542` first feeds the controller the run's
373 recorded scans from the LB press up to clock 85.606 s, the way `steer_stages.replay_window` does. It then closes
the loop from the recorded pose and speed. The plant's first two scans after a reset are zeros (its LiDAR delay line),
so the warm run plays the recorded `/drive` on those two ticks instead of feeding the warmed controller zeros. The
first closed-loop command is then 0.50 m/s and +0.037 rad, against the recorded 0.50 m/s and +0.019 rad (awareness
on). Real values: speed 2.51 m/s at +0.7 s and 2.88 at +1.3 s, peak overspeed max(v - cmd) 2.22 m/s, contact at
+1.74 s.

| awareness on, 6 seeds | speed at +0.7 s | speed at +1.3 s | max(v - cmd) | min chassis clearance | contact |
|---|---|---|---|---|---|
| shared loop | 0.51-2.38 | 0.51-1.95 | 0.60-1.89 | 0.141-0.180 m | 0 of 6 |
| car 1's own | 0.66-2.71 | 0.66-2.53 | 0.90-2.24 | 0.180-0.224 m | 0 of 6 |

Awareness off, car 1's own loop: 2.36-2.43 at +0.7 s, 2.30-2.69 at +1.3 s, no contact in 6 of 6. The low seeds
with awareness on are runs where the follow cap still held.

**The path matches, the wall does not.** Seed 12345 with car 1's loop follows the real path within 0.07-0.17 m and
is at (-9.59, 11.43) at +1.75 s, where the real car hit at (-9.62, 11.50). I placed the plant at each recorded pose
through the corner. At the contact pose (87.36 s) the simulator's map leaves **0.46 m of chassis clearance**, and
0.52-0.75 m over the 0.5 s before it. The real car's own scans there show the left wall 0.29-0.34 m from the LiDAR.
The simulated track, built from car 2's map of that day, has the inner wall of this corner 0.3-0.4 m further away
than car 1 found it. Either the barrier was moved between the two cars' runs, or there was an object at the inner
corner that car 2's map does not hold. The confirmed-static object of the follow study, at about (-8.1, 9.9), sits
at this corner. The speed model cannot reproduce a contact with a wall the track does not have.

## Answers

1. **Is the deadband model enough?** The deadband alone is not: most of the error was the acceleration side and the
   delay. The whole-loop fit (gain 20 up to 5.5 m/s², deadband 0.2, braking 3 /s up to 4 m/s², 3-tick delay)
   matches car 1's braking within 0.4 m/s² in every excess bin, its step-down reach (33 of 67 against 33 of 62) and
   its overshoot (1.66 m/s for 0.75 s against 1.56, 0.75). Wheel-speed error fell 38%.
2. **Does the event reproduce?** Speed and path now do: within 0.2 m/s at +0.7 s and within 0.17 m of the real path.
   The contact does not, because the simulated track's inner wall at that corner is 0.3-0.4 m further out than the
   real one. Next step: a track for this course built from car 1's own scans of the run, then re-run the event.
3. **What changes for rb1 in the simulator?** On the race-day course its lap falls from 14.17 to 11.74 s (6 seeds)
   with more clearance (minimum 0.187 against 0.150 m) and no contact. Car 2 is unchanged.
