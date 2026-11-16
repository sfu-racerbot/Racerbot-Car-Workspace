# The map racer's race-day flank stop in closed loop, and the rejoin that hit the next wall (plan Task 12)

rb2 run 20261111-152132 stopped for 30.5 s (19.0-49.5 s) with a wall 2.5 cm off its right flank and the road open
ahead. Phase 1 added flank creep to `pure_pursuit_node` (shipped `flank_creep_length: 0.50`). The simulator's copy of
the node's emergency tiers (`tools/f1tenth_sim/run_validation.apply_map_safety`) predated it, so every map-racer
simulation still latched there. This adds the creep to that mirror, closes the loop from the recorded stop, and fixes
what the closed loop found.

## Reproducing the stop

Fixture `src/pure_pursuit/test/data/flank_stop_rb2_20261111-152132.npz`: pose (-7.02, 9.27), heading 1.68 rad; its scans
give the body 0.021 m of clearance. Car 2's race-day map draws that wall about 7 cm closer: at the recorded pose the
simulated body overlaps it by 0.043 m, which is a touch and correctly stops whatever the creep says. Moving the car
0.05 m to its left gives 0.023 m, the recorded scan state, and that is where the simulation starts.

| `car2_course_2026_11_11`, rb2 floor, 3 s from rest | moved | chassis contact |
|---|---|---|
| flank creep off | 0.01 m (stays stopped, as on race day) | none |
| flank creep 0.5 m, as shipped | 1.05-1.10 m | **yes, 3 of 3 seeds** |
| flank creep 0.5 m + arc braking cap (below) | 0.73-0.81 m | none, 3 of 3 seeds (0.05-0.07 m left) |

## Why the creep led into the next wall

The creep crawls at 0.25 m/s until the body is clear, then hands back to the tracking law. Off the line, that law
steered full lock (0.42 rad) back toward it, turning the car about 68° left in 1.8 s and accelerating to 1.4 m/s
toward the wall ahead. Nothing between the line's speed and tier 1 depends on the clearance: in map mode walls are
"explained" and do not trigger avoidance, and tier 1 stops only when the swept path is inside a fixed 0.14 m. It first
tripped at 0.13 m free at 1.39 m/s, too late to stop with car 2's braking. The real node has the same tiers, so the
real car would do the same after a creep.

## The fix: the arc braking cap (`pure_pursuit_node`, on in `pure_pursuit.yaml`)

When tier 1 passes but the swept path (the same `_arc_clearance`, both arcs, 1 m horizon) is finite, the command is
capped at the speed from which the car, holding it for `arc_braking_response_sec` (0.25) and then braking at
`arc_braking_decel` (3.0 m/s², car 2's measured physical braking, as `following_brake`), still stops with
`emergency_arc_clearance` free: `racing_math.delayed_braking_speed_limit`. Beyond the horizon it is inf, so it is at
most about 1.64 m/s and only when something is inside 1 m of the body's path. It only lowers speed. The node's own
default is off (as `flank_creep_length`'s); the shipped YAML turns it on. The tuning guard lets an overlay only lower
`arc_braking_decel` and only raise `arc_braking_response_sec`.

Cost, simulator (`pure_course.py --law delay_lqr --safety map`, cap off against on):

| course, line | rb1 | rb2 |
|---|---|---|
| `car2_course_2026_11_11`, race-day line, 3 seeds × 3 laps | identical (flying 9.27-9.37 s); cap never engaged | identical (flying 9.92-10.0 s); 1223 ticks, cap never engaged |
| `car2_course_2026_11_06a_scans`, centre line, 5.0 / 3.0 | identical | 17.20 against 17.23 s |
| `car2_course_2026_11_06b_scans`, centre line | identical | **latched (2191 stop ticks) → finishes 12.8 s** |
| `car1_course_2026_10_26`, centre line | contact either way; lap finished only with the cap | 17.05 against 17.03 s |
| `course_2026_10_30_scans`, centre line | contact either way | contact either way |
| `room_2026_07_27`, centre line | **9.9 → 11.5 s** | 9.25 s both |

So: no cost on the lines either car races, one centre-line stress run that only finishes with it, and one (rb1 in the
small room, on its centre line at 5 m/s and 3.0 m/s²) that is 1.6 s slower. Centre-line runs at those limits are not
a configuration either car races; the room cost is recorded here in case they ever are.

Tests: `src/pure_pursuit/test/test_delayed_braking.py`, `test_arc_braking_cap_node.py`,
`tools/f1tenth_sim/tests/test_map_racer_stuck_case.py`.

**For the tech lead:** floor test F1 with the map racer now exercises this. After it crawls off the wall, it should
turn back toward its line slowly (well under 1.6 m/s) and stop short of any wall within 1 m, not run into it.
