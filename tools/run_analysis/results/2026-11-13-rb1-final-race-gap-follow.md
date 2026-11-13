# rb1 final race: what gap_follow's follow cap cost, and what it was following

Measurement only (Phase 4, Task 15); no code under test changed. Runs: 20261111-155542 (the final, several cars) and
20261111-153721 (testing, fewer cars). Reproduce:

    python3 tools/run_analysis/follow_cost.py data/rb1/racerbot_runs/20261111-155542 --loc ~/.cache/racerbot/loc \
        --map src/pure_pursuit/test/data/course_20261111/map.yaml \
        --line tools/f1tenth_sim/maps/car2_course_2026_11_11_planned_line.csv     # line file: PR #54

## Method and limits (read before the numbers)

- Each LB hold is replayed through `gap_mirror.GapController` with the run's recorded parameters (`steer_stages.py`).
  A tick is **under the follow cap** when the follow cap (follow law and alongside-abort ceiling over the tracked bodies)
  is below the next-lowest ceiling (curve, clearance, side, the corner cap on a fallback gap, and `max_speed` 5.0). An
  **episode** is a run of such ticks with gaps under 0.1 s. Only LB-held time counts (155542: 141.4 s; 153721: 117.5 s).
- The replay is exact for steering over the final's window (0.0164 rad max, `2026-11-13-rb1-steer-into-wall.md`) but looser
  over a whole run (steering RMS 0.045 rad, 7-9% of ticks more than 0.05 m/s off in speed). Episode counts and times are
  therefore close, not exact; the follow cap values (0.50 = `opponent_follow_floor`) match the recorded `/drive` speeds
  where checked (stall/slow-spot tables).
- The object that set the cap is the tracked body with the lowest follow-law cap that tick; its position is its
  base_link coordinates moved to the map by loc.py's pose. **Wall/static** = within 0.15 m of an occupied cell of the
  race-day map (`course_20261111/map.yaml`, the map loc.py used); otherwise free space = a real object. `moved` is the
  distance from its first to its last sighting in the episode (more than 0.5 m = it moved).
- "Progress lost" is in two readings: the brief's (sum over the ticks of next-lowest cap minus follow cap, times 0.025 s),
  and a counterfactual replay of the same scans with opponent awareness off (sum of the positive difference of the speed
  commands). **Both are command-speed measures in metres of command, an upper bound on distance lost: the car's real speed
  lagged both commands** (it was often 1 to 2 m/s under the next-lowest cap when the cap came on).

## Totals

| run | LB-held | under follow cap | episodes | progress lost (brief / counterfactual) |
|---|---|---|---|---|
| 155542 (final) | 141.4 s | **30.18 s (21%)** | 62 | **56.3 m / 54.5 m** |
| 153721 (testing) | 117.5 s | 13.48 s (11%) | 36 | 31.3 m / 30.9 m |

Split by what it was following:

| class | 155542 episodes | time | lost (brief / cf) | 153721 episodes | time | lost (brief / cf) |
|---|---|---|---|---|---|---|
| wall or static (on the map) | 30 | 20.18 s | 45.2 m / 44.5 m | 33 | 13.28 s | 31.2 m / 30.8 m |
| real object in free space | 22 | 9.08 s | 10.7 m / 9.7 m | 3 | 0.20 s | 0.09 m / 0.09 m |
|   of which moved over 0.5 m | 7 | 6.95 s | 9.2 m / 8.5 m | 0 | 0 | 0 |
|   of which did not move | 15 | 2.12 s | 1.45 m / 1.26 m | 3 | 0.20 s | 0.09 m |
| no object (follow memory / held car) | 10 | 0.93 s | 0.47 m / 0.28 m | 0 | 0 | 0 |

**About two thirds of the final's follow-cap time (20.2 of 30.2 s) and 80% of the progress lost (45 of 56 m) was spent
following a wall.** In the testing run 99% was. Only 7 episodes in the final (6.95 s, 9 m) followed something that moved in free
space, i.e. a car.

## The same place every time

Of the 30 wall episodes in the final, 22 (16.0 s, 39.1 m of the 45.2 m) are the object at **(-8.1, 9.9)** (course position
s = 14.2-16.2 m): the inner wall corner at the entrance of the U-turn, on the occupied cells (distance 0.00 m), seen from the
straight before it and held at the 0.5 m/s follow floor; 10 of them last 1.05-1.88 s each. The testing run has the same
corner in 24 of its 33 wall episodes (12.3 s of 13.3 s, 29.3 m of 31.2 m; 6 of them 1.2-1.9 s). It is also where rb1's slow
spots and its approach to the wall in `2026-11-13-rb1-steer-into-wall.md` start. Others: (-9.9, 10.9) at s 17.4-18.3 (also a wall), (-16, 12.4) at s 24.4 (a
real object that moved 1.0 m: a car), (-14, 4.4) at s 4.1 (a car that moved 1.75 m), (-10.7, 3.1) at s 8.5 (a car that moved
6.85 m, 1.75 s).

## Episodes, run 155542 (seconds from the first joy/odom/intent; `s` is the car's course position in m; caps in m/s)

```
  22.24  1.35  14.99   0.50  3.66   2.99   4.037  3.925   -8.11   9.89   0.00  0.15 [38]       wall/static
  24.20  0.05  15.89   0.50  1.68   1.18   0.059  0.059   -8.06  10.04   0.00  0.03 [38]       wall/static
  29.64  0.05   1.50   3.75  4.80   1.06   0.053  0.000  -15.38   4.31   0.25  0.21 [80]       free space (real object)
  29.72  0.05   1.65   3.82  5.00   1.18   0.059  0.000  -15.45   4.06   0.00  0.02 [80]       wall/static
  33.53  0.75  14.09   0.86  2.50   1.40   1.048  0.990   -7.72   9.90   0.35  1.12 [99, 100]  free space (real object)
  34.31  1.05  14.84   0.50  3.77   2.72   2.856  2.436   -8.11   9.89   0.00  0.03 [100]      wall/static
  35.71  0.18  15.74   0.50  2.02   1.57   0.275  0.275   -8.05  10.02   0.00  0.10 [100]      wall/static
  36.39  0.12  17.54   0.50  2.66   2.20   0.275  0.275   -9.96  10.92   0.05  0.20 [115]      wall/static
  38.75  1.35  24.43   0.79  1.99   1.24   1.677  1.673  -15.99  12.45   0.54  1.01 [128]      free space (real object)
  46.32  1.07  15.29   0.36  3.49   3.16   3.402  3.304   -8.09   9.93   0.00  0.10 [166]      wall/static
  48.01  0.05  16.19   0.50  2.31   1.81   0.091  0.091   -8.09  10.13   0.00  0.07 [166]      wall/static
  48.61  0.12  17.54   0.50  2.41   1.96   0.246  0.246   -9.94  10.89   0.05  0.12 [185]      wall/static
  57.92  0.53  15.44   0.50  3.48   2.99   1.571  1.564   -8.06  10.01   0.00  0.09 [233]      wall/static
  58.54  0.03  15.89   0.50  2.32   1.82   0.045  0.052   -8.04  10.04   0.00  0.00 [233]      wall/static
  58.60  0.03  15.89   0.50  2.18   1.68   0.042  0.042   -8.05  10.05   0.00  0.00 [233]      wall/static
  58.65  0.03  16.04   0.50  1.89   1.39   0.035  0.035   -8.06  10.07   0.00  0.00 [233]      wall/static
  58.70  0.03  16.04   0.50  1.70   1.20   0.030  0.030   -8.06  10.08   0.00  0.00 [233]      wall/static
  64.74  0.12   4.35   2.56  3.00   0.52   0.065  0.011  -13.99   3.08   0.00  0.14 [293]      wall/static
  65.20  0.03   4.50   2.69  3.45   0.76   0.019  0.000  -13.47   2.95   0.22  0.00 [293]      free space (real object)
  65.25  0.18   4.65   2.69  3.15   0.39   0.068  0.011  -13.36   2.74   0.27  0.28 [293]      free space (real object)
  65.45  0.03   4.95   2.84  3.23   0.39   0.010  0.000  -13.37   2.50   0.20  0.00 [293]      free space (real object)
  65.50  0.03   5.10   2.81  3.17   0.36   0.009  0.000  -13.39   2.39   0.16  0.00 [293]      free space (real object)
  65.60  0.03   5.25   2.87  2.95   0.08   0.002  0.000     nan    nan    nan   nan []         no object (memory/held)
  65.65  0.10   5.40   2.87  3.36   0.55   0.055  0.017     nan    nan    nan   nan []         no object (memory/held)
  66.03  0.03   6.00   2.87  3.04   0.18   0.004  0.004     nan    nan    nan   nan []         no object (memory/held)
  66.30  0.05   6.15   2.87  3.03   0.16   0.008  0.007     nan    nan    nan   nan []         no object (memory/held)
  69.24  1.88  14.99   0.50  3.39   2.20   4.119  4.436   -8.10   9.91   0.00  0.81 [305, 306] wall/static
  84.25  1.38  14.99   0.49  3.62   2.86   3.939  3.818   -8.10   9.88   0.00  0.12 [40]       wall/static
  86.89  0.15  17.54   1.54  2.30   1.14   0.171  0.171   -9.08  10.56   0.51  1.15 [61, 63]   free space (real object)
  87.16  0.12  18.44   2.01  2.11   0.13   0.016  0.016   -9.58  11.16   0.42  0.24 [63]       free space (real object)
  97.60  0.05  22.78   0.50  2.83   2.33   0.116  0.116  -12.09  11.60   0.46  0.63 [27]       free space (real object)
  97.80  1.03  24.43   1.29  1.86   0.66   0.681  0.681  -16.27  12.42   0.43  0.27 [36]       free space (real object)
 100.41  0.05   1.65   2.88  3.61   0.73   0.036  0.040  -15.18   5.21   0.74  0.09 [44]       free space (real object)
 100.49  0.05   2.10   2.86  3.18   0.32   0.016  0.016  -15.00   5.00   0.87  0.29 [44]       free space (real object)
 100.59  1.20   4.05   2.36  3.51   1.14   1.368  1.248  -14.04   4.43   0.98  1.75 [44]       free space (real object)
 104.65  1.78  14.84   0.53  2.13   1.49   2.648  2.715   -8.09   9.91   0.00  0.88 [59, 60]   wall/static
 106.82  0.12  16.04   0.50  2.06   1.64   0.205  0.205   -8.05  10.06   0.00  0.05 [59]       wall/static
 112.30  0.05   3.90   3.14  3.55   0.41   0.021  0.000  -13.47   3.17   0.05  0.19 [115]      wall/static
 112.52  0.05   4.50   3.14  3.45   0.31   0.016  0.002     nan    nan    nan   nan []         no object (memory/held)
 112.70  0.10   4.80   3.14  4.18   1.01   0.101  0.018     nan    nan    nan   nan []         no object (memory/held)
 112.84  0.08   5.10   3.14  3.83   0.99   0.074  0.025     nan    nan    nan   nan []         no object (memory/held)
 112.97  0.03   4.95   3.14  3.15   0.01   0.000  0.000     nan    nan    nan   nan []         no object (memory/held)
 113.04  0.03   4.95   3.14  3.49   0.35   0.009  0.009     nan    nan    nan   nan []         no object (memory/held)
 114.41  1.75   8.54   2.71  3.73   1.44   2.523  2.382  -10.74   3.06   1.03  6.85 [130]      free space (real object)
 116.18  0.45  11.69   2.82  3.26   0.44   0.199  0.199     nan    nan    nan   nan []         no object (memory/held)
 117.10  1.55  14.24   0.50  2.88   2.17   3.369  3.345   -8.15   9.86   0.00  0.54 [132, 133] wall/static
 119.02  0.18  15.74   0.50  1.75   1.40   0.246  0.246   -8.03  10.02   0.05  0.10 [132]      wall/static
 122.01  0.25  22.93   1.58  2.88   1.36   0.341  0.311  -13.71  14.00   0.53  0.23 [167]      free space (real object)
 124.66  0.05   0.75   2.30  2.58   0.27   0.014  0.014  -15.71   6.68   0.71  0.18 [180]      free space (real object)
 124.73  1.70   3.15   2.23  3.52   1.35   2.297  1.905  -14.59   5.37   0.65  2.23 [180]      free space (real object)
 129.44  1.55  14.84   0.50  3.35   2.50   3.882  3.868   -8.10   9.90   0.00  0.82 [192, 193] wall/static
 131.36  0.12  15.89   0.50  2.10   1.73   0.216  0.216   -8.05  10.06   0.00  0.10 [192]      wall/static
 132.16  0.15  18.14   0.50  2.15   1.65   0.247  0.247  -10.07  10.78   0.05  0.02 [208]      wall/static
 136.92  0.03   1.20   3.46  3.47   0.02   0.000  0.000  -14.47   5.32   0.57  0.00 [229]      free space (real object)
 136.97  0.10   1.50   3.49  3.88   0.35   0.035  0.023  -14.18   5.10   0.46  0.30 [229]      free space (real object)
 141.31  0.08  14.54   0.50  2.73   2.23   0.167  0.167   -8.11   9.88   0.00  0.02 [265]      wall/static
 141.43  1.50  15.59   0.50  3.43   2.54   3.814  3.925   -8.06   9.96   0.00  0.11 [265]      wall/static
 146.89  1.03  17.39   0.45  1.62   1.23   1.258  1.257   -7.48  12.06   0.07  0.00 [300]      wall/static
 149.88  0.03   2.40   4.09  4.52   0.43   0.011  0.000  -13.46   3.53   0.30  0.00 [324]      free space (real object)
 154.07  1.45  15.29   0.50  3.41   2.81   4.068  3.806   -8.06   9.93   0.00  0.06 [337]      wall/static
 158.76  0.12  18.89   0.52  1.62   1.16   0.145  0.144   -8.09  11.25   0.81  0.18 [377]      free space (real object)
 158.91  2.58  17.54   0.31  1.69   1.53   3.933  3.903   -7.21  11.22   0.07  1.08 [377]      wall/static
```

## Episodes, run 153721

```
  84.20  0.03  14.69   0.50  3.75   3.25   0.081  0.067   -8.11   9.96   0.00  0.00 [38]       wall/static
  84.27  0.50  15.44   0.50  3.45   2.99   1.496  1.485   -8.08   9.99   0.00  0.10 [38]       wall/static
  87.17  0.05  18.29   0.50  1.86   1.36   0.068  0.068  -10.01  10.84   0.00  0.06 [10]       wall/static
  87.70  0.05  19.48   0.50  2.17   1.67   0.084  0.084  -10.52  10.80   0.05  0.06 [23]       wall/static
  96.07  0.58  15.29   0.50  3.46   3.00   1.724  1.715   -8.08   9.96   0.00  0.13 [74]       wall/static
  96.74  0.03  15.89   0.50  2.44   1.94   0.049  0.051   -8.05  10.04   0.00  0.00 [74]       wall/static
  96.79  0.03  15.89   0.50  2.23   1.73   0.043  0.043   -8.05  10.03   0.00  0.00 [74]       wall/static
  96.84  0.03  15.89   0.50  1.94   1.44   0.036  0.036   -8.06  10.05   0.00  0.00 [74]       wall/static
  96.89  0.03  15.89   0.50  1.76   1.26   0.031  0.031   -8.06  10.05   0.00  0.00 [74]       wall/static
 106.68  1.28  14.84   0.50  3.21   2.63   3.356  3.206   -8.12   9.88   0.00  0.13 [165]      wall/static
 108.58  0.10  15.89   0.50  1.70   1.22   0.122  0.122   -8.04  10.03   0.00  0.01 [165]      wall/static
 109.30  0.10  17.69   0.50  2.44   2.01   0.201  0.201   -9.97  10.87   0.05  0.04 [191]      wall/static
 123.42  0.10  14.84   0.50  3.06   2.58   0.258  0.255   -8.13   9.99   0.00  0.04 [41]       wall/static
 123.57  0.48  15.44   0.50  3.47   2.96   1.405  1.400   -8.08   9.98   0.00  0.04 [41]       wall/static
 124.14  0.03  15.89   0.50  2.33   1.83   0.046  0.059   -8.06  10.06   0.00  0.00 [41]       wall/static
 124.19  0.03  15.89   0.50  2.32   1.82   0.046  0.054   -8.06  10.06   0.00  0.00 [41]       wall/static
 124.24  0.70  16.04   0.50  2.28   1.80   1.262  1.266   -8.06  10.09   0.00  0.08 [41]       wall/static
 180.39  0.48  14.84   0.50  3.53   2.95   1.404  1.371   -8.10   9.93   0.00  0.08 [24]       wall/static
 191.80  1.20  15.14   0.50  3.41   2.95   3.539  3.492   -8.09   9.92   0.00  0.17 [50]       wall/static
 200.61  0.10   6.45   0.50  2.46   2.03   0.203  0.185  -13.54   5.39   0.00  0.05 [101]      wall/static
 202.85  1.45  14.84   0.50  3.56   2.71   3.929  3.890   -8.13   9.88   0.00  0.12 [111]      wall/static
 204.77  0.15  15.74   0.50  1.98   1.57   0.236  0.236   -8.03  10.02   0.00  0.02 [111]      wall/static
 205.57  0.12  17.84   0.50  2.37   1.88   0.234  0.234   -9.91  10.84   0.00  0.09 [132]      wall/static
 214.73  1.40  14.84   0.50  3.65   2.64   3.689  3.613   -8.11   9.89   0.00  0.09 [10]       wall/static
 216.58  0.12  15.89   0.50  1.69   1.20   0.150  0.150   -8.05  10.04   0.00  0.03 [10]       wall/static
 217.33  0.15  18.14   0.50  2.39   1.87   0.280  0.280   -9.99  10.87   0.00  0.15 [40]       wall/static
 224.66  0.03   8.99   4.45  5.00   0.55   0.014  0.014   -6.12   6.65   0.45  0.00 [85]       free space (real object)
 224.74  0.08   9.44   3.97  4.36   0.31   0.023  0.023   -6.04   6.69   0.47  0.01 [85]       free space (real object)
 224.84  0.10  10.04   3.41  3.90   0.50   0.050  0.050   -6.12   6.58   0.47  0.18 [85]       free space (real object)
 226.08  1.88  14.84   0.23  1.62   1.29   2.420  2.412   -8.16   9.86   0.00  0.11 [92]       wall/static
 228.88  0.15  15.74   0.50  1.70   1.37   0.205  0.205   -8.04  10.03   0.00  0.04 [92]       wall/static
 237.64  0.03   9.74   3.17  4.05   0.88   0.022  0.022   -6.49   6.74   0.07  0.00 [183]      wall/static
 237.79  0.33  11.09   2.83  4.37   1.37   0.446  0.446   -6.35   6.94   0.10  0.00 [183]      wall/static
 239.01  1.43  15.29   0.42  3.31   2.59   3.692  3.655   -8.10   9.92   0.00  0.12 [188]      wall/static
 240.88  0.05  16.04   0.50  2.43   1.93   0.097  0.096   -8.05  10.05   0.00  0.03 [188]      wall/static
 241.40  0.15  17.39   0.50  2.68   2.28   0.342  0.336   -9.94  10.88   0.05  0.13 [219]      wall/static
```
