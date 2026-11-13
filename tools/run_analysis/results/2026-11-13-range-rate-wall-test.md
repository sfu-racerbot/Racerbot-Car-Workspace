# Range-rate wall test (2026-11-13): decision FAIL, not implemented

Study: `tools/run_analysis/range_rate_study.py` (hook added to `ego_yaw_study.run_variant`; shipped code untouched).
residual = d(near_range)/dt + v cos(b) + omega * laser_x * sin(b), b = bearing of the object's nearest beam, running median of 8 valid ticks (4 needed). NaN ticks (no associated detection / smoothing not yet filled) count as below T. Episodes are the shipped follow episodes (wall = U-turn inner corner, real = free-space objects that moved over 0.5 m).

## Decision table

| run | omega | T | wall episodes below T on >=95% of ticks | real episodes at/above T on >=70% of ticks | verdict |
|---|---|---|---|---|---|
| 20261111-155542 | gyro | 0.8 | 19/22 | 1/7 | FAIL |
| 20261111-155542 | gyro | 0.5 | 18/22 | 1/7 | FAIL |
| 20261111-155542 | steer | 0.8 | 19/22 | 1/7 | FAIL |
| 20261111-155542 | steer | 0.5 | 17/22 | 1/7 | FAIL |
| 20261111-153721 | gyro | 0.8 | 24/24 | 0/0 | FAIL |
| 20261111-153721 | gyro | 0.5 | 22/24 | 0/0 | FAIL |
| 20261111-153721 | steer | 0.8 | 24/24 | 0/0 | FAIL |
| 20261111-153721 | steer | 0.5 | 21/24 | 0/0 | FAIL |

Pass needed >=20/22 (final), >=22/24 (testing) AND >=6/7 real. Real episodes reach 1/7 at both thresholds, so neither T passes and nothing was implemented.

## Why the real episodes fail

Real-car residuals are large when measured (median 1.3-3.3 m/s on 5 of 7) but only 4 of 7 episodes keep an associated detection for most ticks: valid ticks per episode are 30/30, 4/54, 4/6, 0/2, 24/48, 48/70, 31/68. Fraction >=0.8 over all ticks: 0.70, 0.04, 0.00, 0.00, 0.50, 0.69, 0.46. The separation in the data that exists is wide (wall median 0.07-0.24, p90 mostly <0.4; real 1.3-3.3), so a test that tolerated missing samples might separate them, but that is a different rule from the one briefed and was not applied. One wall episode (104.65, p90 2.07) has a tail the 95% rule rejects.

Full per-episode tables (median, p10, p90, fractions) for both runs and both omega sources: see `range_rate_study.py` output; reproduce with the command in its docstring (data in the main checkout, loc cache `~/.cache/racerbot/loc`).
