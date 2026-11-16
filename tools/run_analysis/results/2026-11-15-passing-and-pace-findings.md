# Passing and pace findings, 2026-11-15

Simulator work on Fable's 2026-11-15 plan (jobs 2, 3, 6 and 7). Car 2 unless stated; race-day course
`car2_course_2026_11_11`, its planned line, the race-day floor `sfu_course_2026_11_11_rb2`.

## Real race-day laps, without localisation

Lap period from the autocorrelation of `/drive` steering; path per lap from true wheel distance over moving time.

| run | controller | lap | path per lap | speed while moving |
|---|---|---|---|---|
| 20261111-135934 | gap follow, time trial | 9.70 s | 26.0 m | 2.68 m/s |
| 20261111-142254 | gap follow | 9.85 s | 26.5 m | 2.69 m/s |
| 20261111-152132 | map racer | 10.05 s | 29.3 m | 2.91 m/s |

The gap-follow time trial was the fastest thing either car did on race day. The map racer is faster on speed but drives
about 13% more distance (its planned line is 27.88 m, and it tracks it about 5% wide, as the simulator also shows:
28.9 m). This is a candidate, not a conclusion: the methods differ slightly between runs, and whether the map
racer's lap is bound by its line length or by grip is not yet separated (the earlier-braking runs cut tracking width
without cutting the lap).

The simulator's gap-follow time trial is 4-5% **slow** against the car (flying 10.1-10.25 s against 9.70 s),
not optimistic: see `2026-11-14-rb2-race-day-agreement.md`, second correction.

## Car 2's speed loop refit (not adopted)

`fit_speed_loop.py` on 135934 with a wider grid picks accel gain 10, accel max 5.0, brake 4 / 4, delay 3 ticks.
Its 1-D replay error is 12-14% lower than the race-day floor's own loop on all three race-day runs (two out of sample).
Closed loop it moves the time trial toward the car (+4.5% to +3% of 9.70 s) and the map racer slightly away (-0.5% to
-1% of 10.05 s). Not adopted: the floor's loop is pinned to its `car_measure.py` measurement by
`test_floor_2026_11_11_rb2.py`, and the gain is small.

## Stopped-car bypass (draft #84, off)

On the race-day course a car that stalls on the line ends our race: 0 of 6 finish, both cars, planner on or off.
From each seed's first standstill a crawl path round it exists in ground truth (6 of 6, exact rectangles, 0.05 m),
but not under the planner's own checks:

| check at the six stall poses | seeds with a path |
|---|---|
| ground truth, exact rectangles, 0.05 m | 6 |
| planner stack (0.06 m + build slack, body bounding box, opponent box uncertainty) | 0 |
| without build slack | 0 |
| without car margins | 0 |
| without either | 3 |

## Gap-follow pace

Floor matrix (`floor_matrix.py`, 12 courses x {car 2's own floor, race-day floor} x grip 0.85 / 1.0 / 1.15 x 3
seeds, 2 laps), each arm against the shared time trial:

| arm | flying lap, median cell | verdict |
|---|---|---|
| #71: steering gain 0.8, slew 4.0 | 2-7% faster where it finishes | 32 worse cells: contact on both 10-30 courses and 06b, stalls on 06a; reverted (#81) |
| delay-predicted aim, 0.05 s (`target -= tau v tan(delta) / L`, prototype in the mirror only) | -2.2% (50 cells) | 30 worse cells: contact on both 10-30 courses and levine_obs, stalls on 06a/06b |

On the race-day course alone the aim delay gives 9.94 s against 10.18 s flying, with the steering weave down from
0.96 to 0.78 reversals per second, but the closest wall pass falls from 0.15 to 0.071 m. Both arms that tighten the
line fail on the same courses on the race-day floor; the shared time trial itself makes contact on
`course_2026_10_30_scans` and `levine_obs` there, so those courses are already at the edge. Car 1: no gain at 0.05 s,
contact at 0.1 s.

## Two-car baseline (race_two, #80)

Both cars, race-day course, 8 scenarios x 6 seeds: zero contact of any kind in 96 races, zero passes. Against a
stalled car nobody finishes; against a 60% car we follow it for the whole race.

## Update 2026-11-16: path length over flying laps, real wall clearance, tighter builder floors

Re-measured over whole flying laps only (steering-autocorrelation lap period; true wheel distance inside the
longest stretch without a stop, launch lap excluded):

| run | flying laps | lap | path per lap | speed |
|---|---|---|---|---|
| 20261111-135934, gap-follow time trial | 9 | 9.70 s | 26.64 m | 2.75 m/s |
| 20261111-152132, map racer | 12 | 10.05 s | 29.27 m | 2.91 m/s |

Chassis to nearest LiDAR return over those laps (body outline in the laser frame, the 15-17 beams that always
return under 0.5 m, the car itself, excluded):

| run | min | p1 | p5 | median | time under 0.16 m |
|---|---|---|---|---|---|
| gap-follow time trial | 0.042 m | 0.126 m | 0.139 m | 0.255 m | 9.4% |
| map racer | 0.061 m | 0.135 m | 0.177 m | 0.351 m | 3.2% |

Gap follow runs about 0.1 m nearer the walls and drives a 9% shorter path. The builder keeps the line's body
0.22 m (0.30 m where the steering is tight) from walls. Tightening those floors does not reproduce the gain in the
course suite (`course_suite/run.py`, 11 courses with laps, 3 seeds, floor `sfu_course_2026_11_06`):

| builder floors | flying lap vs shipped, median | contact |
|---|---|---|
| 0.22 / 0.30 (shipped) | - | none |
| 0.16 / 0.24 | -0.4% | `course_2026_10_30_scans` 3 of 3 |
| 0.12 / 0.18 | -0.2% | `asb_sharp`, `car1_course_2026_10_26` 3 of 3; `indoor_tight` 2 of 3; 10-30 3 of 3 |

So the map racer's longer real path is not explained by the clearance floors; what gap follow does differently
(cutting corners on its own sightlines) remains open.

**Correction (2026-11-16, later).** "Not explained by the clearance floors" overstates it. On the race-day course the
shipped floors build no line at all from the suite's mapping lap (every candidate short of the 0.30 m tight floor by
0.14-0.19 m), so that course has no shipped row; there 0.12 / 0.18 ran 10.02 s against 0.16 / 0.24's 10.25 s (-2.2%),
zero contact. The suite runs on the 2026-11-06 floor and records no clearance. And the simulator's clearance scale is
not the car's: on the race-day course and floor, chassis to nearest scan return over flying laps is

| | min | p1 | median |
|---|---|---|---|
| gap follow, car | 0.042 m | 0.126 m | 0.255 m |
| gap follow, simulator (3 seeds) | 0.131-0.139 m | 0.148-0.153 m | 0.366-0.373 m |
| map racer, car | 0.061 m | 0.135 m | 0.351 m |
| map racer, simulator (3 seeds) | 0.124-0.137 m | 0.150-0.170 m | 0.388-0.397 m |

The simulated gap follow keeps 0.11 m further from walls than the car (its longer path and the 4-5% it runs slow),
and both simulated tails are cleaner. Until the simulated world matches the car on this scale, clearance-only verdicts
in the simulator (the parked speed-ups) cannot be ranked.

## Car 1 cannot build a line on the race-day map (2026-11-16)

`python3 -m pure_pursuit.race_line --car rb1 --from-run-dir <car 2's mapping run>` (#63's adopt path), shipped code:

| car 2 mapping run | car 1 | car 2 |
|---|---|---|
| 20261111-143428 | min_time, 9.66 s | (not tried) |
| 20261111-151125 (the run car 2 raced on; the simulator's `car2_course_2026_11_11` map) | **no line**: every candidate fails (min_time: body 0.224 against the 0.30 m tight floor, curvature 1.083 against car 1's 0.879 1/m) | min_time, 9.44 s |

Car 1's planner curvature limit (0.879 1/m) is below what one corner of this line needs; car 1's own steering
(0.33 rad, about 0.95 1/m) and its race-day gap-follow laps say the car can drive the course. Before floor test F3
(car 1's map racer on car 2's map), either adopt a mapping run car 1's limits accept, or revisit car 1's planner
steering limit and tight floor. Nothing was changed.

**Correction (same day).** It depends on the recorded lap the builder starts from, not the map alone: a car 1 line on
the same frame built from a simulated gap-follow lap (Fable's, `min_time` 9.81 s, curvature 0.862 under car 1's 0.879)
passes. So: car 1 builds from 143428's lap and from a simulated lap, not from car 2's own recorded lap of 151125; checked
on the dev box only (the optimizer is known to differ on x86). With that simulated-lap line, #63's evidence reruns with
car 1's own steering-law parameters (#79): 6 seeds x 3 laps, first laps 10.60-10.65 s, flying 10.05-10.17 s, zero
contact, zero safety-tier ticks, closest chassis-to-map 0.071-0.10 m (#63 reported 10.65-10.78 / 10.12-10.25 s with the
shared servo values). Before F3, check that the line car 1 will race was built on the Jetson from the lap it will use.
