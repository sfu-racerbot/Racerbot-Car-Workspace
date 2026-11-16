# Car 2's race-day agreement arms (plan Task 11): map racer agrees by selection; time trial agrees on speed, not on the recorded lap time

Plan Task 11 asks for the simulated median lap within 4% of race day for three arms. Car 1's gap follow is pinned
(`tools/f1tenth_sim/tests/test_race_day_agreement.py`, 11.74 s against 11.5 s). This note covers car 2's two arms and
why neither is pinned as a lap time.

## Map racer (rb2 20261111-152132, real flying laps 10.05 s)

Car 2's floor `sfu_course_2026_11_11_rb2` chose its grip scale (1.15) by this lap: the closed-loop selection kept cells
whose flying lap with the online cap on was in [9.95, 10.45] s (`tools/f1tenth_sim/sim_fidelity/sfu_course_2026_11_11_rb2_replay.json`,
`closed_loop_selection`). A test pinning it would test the selection rule, not an independent agreement, so none is added.

## Gap follow time trial (rb2 20261111-135934)

Simulator: `car2_course_2026_11_11`, floor `sfu_course_2026_11_11_rb2`, car 2's recorded parameters
(`metadata/params/gap_follow_node.yaml` of the run, read from car 2; parameters added since the race taken at their
off values: `flank_creep_length` 0, static gate 0/0), 6 seeds: **11.0 s** (10.98-11.05), no contact. With today's
values for those four parameters: the same.

The plan's 11.9 s (11.8-12.5 s typical) came from `loc.py` positions. On this run they are not good enough to time laps:

- no fixes at all over s = 14-18 m of the 29.24 m centre line, five to seventeen over 19-21 m;
- 679 fixes (34 s) inside one metre at s = 4 m while the wheel speed there has a median of 2.5 m/s (a stuck estimate);
- `laps.py` (which counts a jump as no progress) gives laps of 12.3, 13.5, 15.3, 13.3, 15.7, 13.7 and 15.5 s.

Measures that do not depend on localisation:

| | real 135934 (LB held) | simulator (3 seeds) |
|---|---|---|
| wheel distance / time | 295.8 m in 107.1 s | path 30.4-31.0 m per lap |
| mean speed | 2.76 m/s wheel, 2.71 m/s true (odom scale 1.0167) | 2.77-2.83 m/s true |
| time under 0.3 m/s | 0.7 s in 107 s | launch only |

Where localisation is good (s = 8-13 and 19-29 m), the commanded and the true speed per metre agree within about
0.2-0.3 m/s, with the simulator sometimes the slower one. Over a lap as long as the simulator's (about 30.8 m), the real
mean speed gives 11.1-11.4 s, 1-4% from the simulator's 11.0 s.

So the time-trial arm agrees on speed and probably on lap time, but the recorded lap time it would be checked against
is not a measurement. A test is not added: against 2.71 m/s the simulator sits at +2 to +4%, too close to the 4% band
to pin a number whose real side is inferred. A clean time-trial run with good localisation (or lap timing from the
start line, not from positions) would settle it.

Reproduce (needs the run, which stays on car 2): `rb python3 tools/run_analysis/laps.py <run> --line
tools/f1tenth_sim/maps/car2_course_2026_11_11_centerline.csv`; the simulator lap is `eval_gap.run_track` with the
settings above.

## Correction (2026-11-14, evening)

The simulator lap above (11.0 s, 2.77-2.83 m/s) is a **standing-start** lap; the real run's 2.71 m/s is over 107 s
of mostly flying laps. The simulator's flying laps are 10.10-10.25 s at about 3.0 m/s, so the time-trial arm is
about 10% optimistic, not 1-4%. The conclusion that the localisation-based 11.9 s is not a measurement stands; the
claim that the simulator "agrees on speed" does not.

## Second correction (2026-11-15): the time-trial arm agrees on speed; "10% optimistic" was a measuring error

Both the table's "path 30.4-31.0 m per lap" and the evening correction's "about 3.0 m/s" came from summing the
distance between consecutive `x, y` samples of `eval_gap`'s trace. Those are the controller's **estimated** pose
(`std_state`, with the plant's 2 cm white pose noise), not ground truth: on 7 cm steps the noise lengthens the path by
about 10%. Integrating the plant's true speed instead (same run: rb2, shared time trial, `sfu_course_2026_11_11_rb2`,
seed 12345, 4 laps) gives 82.4 m over 30.6 s of flying laps against 90.6 m from the noisy positions.

Speed while driving, ticks commanded above 1.5 m/s:

| | commanded | true |
|---|---|---|
| real 135934 (`/drive`; `/odom` / 1.0167) | 2.66 m/s | 2.73 m/s |
| simulator, flying laps | 2.56 m/s | 2.64 m/s (2.69 over all flying ticks) |

The simulator is 1-3% *slower* than the car, not 10% faster. The steering it commands is the same too: mean |steering|
0.166 rad real against 0.185, p90 0.42 both, 0.96 direction changes per second both. Binding shares (`intent_report.py`)
are curve cap 49%, clearance 27%, acceleration ceiling 24% on the car, matching Fable's simulator count (41-49% curve).

A speed-loop refit on this run (`fit_speed_loop.py`, wider grid: accel gain 10, max 5.0, brake 4 / 4, delay 3) lowers
the 1-D replay error by 12-14% against the race-day floor's loop on all three race-day runs (two out of sample), but
makes the closed-loop simulator faster still (standing 11.0 -> 10.78 s), so it is not adopted on that evidence alone.

**The real lap, without localisation:** the autocorrelation of `/drive` steering over the 109 s LB hold peaks at a lag of
9.70 s (r 0.78) and again at 19.40 s, so the car's flying laps were about **9.70 s**, against the simulator's 10.1-10.25
s. The car drove a shorter path (about 26.5 m at 2.73 m/s, against the simulator's 27.5 m) at about the same speed. For
the time trial the simulator is therefore 4-5% **slow**, not optimistic. It also means car 2's gap-follow time trial
(9.70 s) was faster on race day than its map racer (10.05 s flying).
