# Static-confirmation gate (2026-11-13): decision FAIL, nothing implemented

Study: `gate_report` in `tools/run_analysis/range_rate_study.py` (same command as the earlier table; prints a `== GATE` block per run).
Rule measured: a track is static-confirmed when >= 4 raw steering-model residual samples lie among the newest 8 no older than 0.6 s
(sample only if 0 < dt <= 0.25 s) and median |residual| < S. Wall ticks = every tick of a wall episode (shipped follow cap binding);
the object on a tick is that episode's followed track. Missing samples never confirm.

| run | S | wall ticks confirmed (aggregate) | real episodes with a confirmed run > 0.15 s or > 5% of ticks |
|---|---|---|---|
| 155542 | 0.5 | 534/638 = 0.837 | 33.53 (9/30 = 30%, longest run 0.225 s); 97.60 (2/2 ticks, 0.05 s) |
| 155542 | 0.35 | 511/638 = 0.801 | 33.53 (9/30 = 30%, longest run 0.225 s) |
| 153721 | 0.5 | 445/488 = 0.912 | none (no real episodes) |
| 153721 | 0.35 | 418/488 = 0.857 | none |

Wall side passes at both S on both runs (>= 0.80). The real side fails at both S: episode 33.53 s has window medians down to
0.14 m/s for 9 of its 30 ticks (the followed object is nearly stationary along the line of sight at that moment), so no S that keeps
the wall fraction above 0.80 can reject it. Other real episodes: min full-window median 1.29 (38.75), 0.68 (86.89), 0.37 (97.60,
2 ticks), 1.16 (100.59), 2.87 (114.41), 2.03 (124.73); zero ticks confirmed on 5 of the 7 at both S.

Per-episode wall fractions (S = 0.5): lowest 0.56 (104.65), 0.59 (226.08 on 153721), 0.65 (69.24), 0.69 (34.31).
Neither S passes, so per the brief nothing was implemented.

## Per-track truth (2026-11-13, controller ruling)

The FAIL above came from the episode label. Episode 33.53 s (run 155542) holds two tracks. Track 99 (first 21 ticks) is the real car:
world position (-7.25, 9.23) -> (-7.88, 10.65) in 0.5 s, residual +2.2..+4.9 m/s, never confirmed. Track 100 (the 9 confirmed ticks) sits
at world (-8.12 +/- 0.02, 9.89 +/- 0.02) for 0.45 s while rb1 moves 0.6 m, tracker velocity 0.15-0.4 m/s: a static object whose 0.5 m/s
follow cap took rb1 from 1.7 to 0.33 m/s. Confirming it static is correct.

Truth label per tick (`track_truth`, independent of the residual): the same track id's world position 0.2 s earlier (nearest row within
0.03 s) more than 0.10 m away = moving; at most 0.06 m = static; otherwise unknown. Over all real-episode ticks of run 155542
(159 truly moving):

| S | truly-moving ticks confirmed | longest confirmed truly-moving run | truly-static ticks confirmed | wall ticks confirmed (155542 / 153721) |
|---|---|---|---|---|
| 0.5 | 3/159 (1.9%) | 0.050 s | 2/2 | 0.837 / 0.912 |
| 0.35 | 1/159 (0.6%) | 0.025 s | 2/2 | 0.801 / 0.857 |

PASS at S = 0.35 (wall >= 0.80 on both runs; moving confirmed <= 5%, no run > 0.15 s). The one offending tick at 0.35 is row 744, track 100,
the static object's first tick, whose 0.2 s look-back lands on track 99's neighbouring position. S = 0.5 also passes by this rule (3 ticks: that
one and two of track 27 at 97.60 s), but S = 0.35 is shipped.
