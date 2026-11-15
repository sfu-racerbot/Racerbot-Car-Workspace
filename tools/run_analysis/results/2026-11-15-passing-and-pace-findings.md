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
