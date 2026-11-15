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
