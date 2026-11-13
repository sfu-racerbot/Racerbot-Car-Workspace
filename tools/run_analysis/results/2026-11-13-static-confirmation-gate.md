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
