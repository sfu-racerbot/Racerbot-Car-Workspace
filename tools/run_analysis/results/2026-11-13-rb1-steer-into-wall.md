# rb1 20261111-155542: why the car turned into the left wall at 84.5-88.2 s

Diagnosis only (Phase 4, Task 13); no fix. Reproduce with
`python3 tools/run_analysis/steer_stages.py data/rb1/racerbot_runs/20261111-155542 --loc ~/.cache/racerbot/loc --from 84.5 --to 88.2`.
Times are seconds from the bag's first `/joy`, `/odom` or `/drive_intent` message (absolute 1794441344.154), the clock
`intent_report.py` prints; `scan_window_fixture.py` and `loc.py`-based tools that key on the first `/scan` or `/joy`
are 2.099 s later (the test fixture's window is 82.40-86.10 on that clock). The analysis keys on absolute receive time.

## Replay reproduces /drive

`steer_stages.py` runs `gap_mirror.GapController` (the node's own logic, held to the node by `test_gap_mirror_parity.py`)
with the run's recorded `metadata/params/gap_follow_node.yaml` (`flank_creep_length` 0 and `flank_creep_margin` 0.05 filled
in: the run's node predates flank creep), from the start of the LB hold that contains the window, the bag's scans, the
`/odom` speed the node had and the `/imu/data` gyro its steering damping read. Over the 148 scans of 84.5-88.2 s, all
paired with a published `/drive`: **steering error max 0.0164 rad, RMS 0.0026 rad**; the speed command matches `/drive` within 0.02 m/s on 139 of 148 scans (the other 9 are follow-cap ticks
at 86.89-87.26 s; at 86.99 s the replay's follow cap was 0.5 m/s against the node's 2.17; steering there is unaffected). Over the whole run (5657 ticks) the replay is looser (steering max 0.46 rad, RMS 0.045,
6.8% of ticks more than 0.05 m/s off in speed): the table is trustworthy for this window, the whole-run bound factors in
`2026-11-13-rb1-stalls-and-slow-spots.md` are attribution at the listed ticks, each checked against the recorded speed
command (last column there).

## Verdict

1. **The tick and the stage.** The turn toward the inside wall starts at **86.01-86.31 s in the gap-aim stage**. A wall end
   at 40-60 degrees left (the inside corner of the U-turn) passes the LiDAR; ranges at 40/50 degrees go from 0.75/0.69 m
   (85.90 s) to 3.82/0.67 (86.10) to 3.70/3.98 (86.20); the chosen gap widens from 26.2 to 54.5 degrees and its midpoint
   (aim_depth_bias 0) swings from +2.0 to +23.5 degrees left (86.31 s). Bearing term `steering_gain 1.0 x 23.5 deg =
   +0.41 rad`; with damping -0.08 the command reaches **+0.328 rad (the 0.33 limit) at 86.31 s with the left wall already
   at 0.46 m** (minimum raw range 60-120 degrees left). Centering is **0.000 from 86.01 s on**: its bias fades to zero at a
   15 degree aim bearing (`centering_zero_bearing_deg`), exactly when the aim passes it, and its maximum is only 0.08 rad
   anyway. The arc guard does nothing (0.000) until 86.96 s.
2. **It stays committed.** From 86.9 to 87.3 s the aim holds +16 to +19 degrees (a deep opening at 30-40 degrees left,
   8.96 m at 87.20 s) and the command +0.19 to +0.21 rad with centering 0.000, damping -0.05 to -0.14 and the arc guard
   only -0.025 to -0.083 for 5 ticks; the left range falls 1.00, 0.62, 0.50, 0.33 m (87.0 s: 0.62; 87.3 s: 0.31).
3. **The wall is touched at 87.34 s.** The gyro goes 1.39 to 4.94 rad/s in 0.13 s while the steering command is
   falling (0.18 to 0.055; kinematics give under 0.4 rad/s) and `/imu/data` shows horizontal acceleration 15.8 and then
   29.2 m/s^2 at 87.34 s (ax -15.8, ay 24.5; steady cornering is 3-6 m/s^2). That is a contact; the heading goes from
   167 to 241 degrees between 87.2 and 87.6 s (loc.py). (`enable_imu_impact_stop` was false in this run; its threshold 33.7 m/s^2 would not have
   tripped at 29.2.) The car then stopped 0.21 m from the wall by LiDAR at 88.0 s, the "5 cm" of body clearance.
4. **Not the opponent layer.** Re-running the same window with `enable_opponent_awareness=false`: steering differs by at
   most 0.0018 rad over all 148 scans (speed differs, that is the follow cap). The opponent mode is empty from 85.7 to
   86.5 s (the turn-in), then `follow` (the plain gap is used, never `clear`) with 1-2 tracked movers.
5. **What the scan looked like.** The U-turn's inside corner: to the stage the opening at 30-55 degrees left was a free,
   deep gap whose midpoint is the aim, with the left wall 0.46 m beside the car. It aimed at the inside of the corner.
   Beyond 87.6 s the exit lay at about -30 degrees right (3.5 m) and the car, pointing at the left wall (0.46 m ahead),
   counter-steered to -0.33.

## Per-scan table (warm replay; every 4th scan before 86.0 s, every 2nd to 86.6 s)

Columns: seconds from the bag's first joy/odom/intent; wheel speed m/s; left and right minimum raw range 60-120 degrees;
forward minimum raw range in the forward stop cone; chosen gap's bearing (deg) and width (deg); bearing term (steering
gain x bearing, with the speed-scheduled gain); centering; damping; clipped sum; arc-guard correction; commanded steering
(after the slew limit); recorded `/drive` steering; opponent mode; tracked movers; bound speed factor; commanded speed.
Rad unless noted. Sums and guard are exactly the node's order: bearing + centering + damping, clip, arc guard, slew.

```
t      v    L60-120 R60-120 fwd  gapB  gapW  bear   cent   damp   sum   guard  cmd    real   mode  nmov bind  spd
 84.50 1.64   1.17   0.65  1.24   -5.5  26.7 -0.096  0.057  0.011 -0.028  0.000 -0.028 -0.028 follow  1 follow 0.49
 84.70 0.95   1.21   0.60  1.11   -2.5  26.7 -0.044  0.077 -0.003  0.030  0.000  0.030  0.030 follow  0 follow 0.49
 84.90 0.56   1.25   0.58  1.07   -1.5  26.0 -0.026  0.080 -0.005  0.049  0.000  0.049  0.049 follow  0 follow 0.49
 85.10 0.50   1.26   0.57  1.08   -0.8  26.2 -0.013  0.080 -0.004  0.063  0.000  0.063  0.063 follow  0 follow 0.49
 85.30 0.33   1.27   0.57  1.34   -0.8  26.2 -0.013  0.080 -0.005  0.061  0.000  0.061  0.061 follow  0 follow 0.49
 85.50 0.31   1.29   0.58  1.33   -0.8  26.2 -0.013  0.080 -0.007  0.060  0.000  0.060  0.060 follow  0 follow 0.50
 85.71 0.41   0.71   0.58  1.32   -0.5  26.5 -0.009  0.016 -0.006  0.002  0.000  0.002  0.002         0 accel  1.00
 85.91 0.95   0.69   0.60  1.30    2.0  26.2  0.035  0.011 -0.003  0.044  0.000  0.044  0.044         0 accel  2.01
 86.01 1.41   0.62   0.62  1.27    5.0  29.5  0.087 -0.000 -0.005  0.082  0.000  0.082  0.082         0 accel  2.51
 86.11 1.96   0.52   0.67  1.25    7.7  32.5  0.135 -0.012 -0.025  0.099  0.000  0.099  0.099         0 accel  3.01
 86.21 2.40   0.47   0.74  1.24   14.0  41.2  0.244 -0.003 -0.037  0.205  0.000  0.205  0.205         0 curve  2.08
 86.31 2.51   0.46   0.71  1.23   23.5  54.5  0.410  0.000 -0.082  0.328  0.000  0.328  0.328         0 curve  1.63
 86.41 2.29   0.48   0.75  1.28   22.5  53.0  0.393  0.000 -0.169  0.224  0.000  0.224  0.224         0 curve  1.99
 86.51 2.56   0.54   0.81  1.48   17.7  51.2  0.310  0.000 -0.174  0.135  0.000  0.135  0.135 follow  1 curve  2.57
 86.61 2.64   0.62   0.88  1.57   16.0  52.2  0.279  0.000 -0.103  0.176  0.000  0.176  0.176 follow  1 curve  2.25
 86.63 2.66   0.67   0.90  1.56   16.0  52.7  0.279  0.000 -0.098  0.181  0.000  0.181  0.181 follow  1 curve  2.22
 86.66 2.66   0.71   0.92  1.55   15.0  51.2  0.262 -0.000 -0.107  0.154  0.000  0.154  0.154 follow  1 curve  2.40
 86.68 2.65   0.76   0.94  1.54   14.7  37.2  0.257 -0.000 -0.115  0.142  0.000  0.142  0.142 follow  1 curve  2.51
 86.71 2.70   0.81   0.95  1.54   18.0  39.0  0.314  0.000 -0.123  0.191  0.000  0.191  0.191 follow  1 curve  2.16
 86.73 2.72   0.86   0.97  1.52   13.2  29.7  0.231 -0.002 -0.125  0.104  0.000  0.141  0.144         0 accel  2.84
 86.76 2.74   0.93   0.98  1.51   12.5  29.0  0.218 -0.001 -0.121  0.096  0.000  0.096  0.096         0 accel  2.98
 86.78 2.76   0.99   1.00  1.54   11.5  26.7  0.201 -0.000 -0.123  0.077  0.000  0.077  0.077         0 cleara 3.03
 86.81 2.81   1.02   1.03  1.54    8.5  21.7  0.148 -0.001 -0.119  0.029  0.000  0.029  0.030         0 cleara 2.57
 86.83 2.85   0.98   1.05  1.58    7.7  20.2  0.135 -0.005 -0.109  0.021  0.000  0.021  0.021         0 cleara 2.52
 86.86 2.86   0.84   1.06  1.54    8.2  20.5  0.144 -0.016 -0.097  0.032  0.000  0.032  0.032         0 cleara 2.58
 86.89 2.88   0.80   1.07  1.45   12.7  29.2  0.223 -0.007 -0.090  0.126  0.000  0.085  0.091 follow  1 follow 1.48
 86.91 2.89   0.75   1.08  1.39   12.7  27.7  0.223 -0.008 -0.078  0.137  0.000  0.132  0.133 follow  1 follow 1.62
 86.94 2.89   0.72   1.09  1.29   13.2  26.2  0.231 -0.007 -0.051  0.173  0.000  0.173  0.173 follow  1 follow 1.61
 86.96 2.85   0.69   1.10  1.33   16.0  29.2  0.279  0.000 -0.047  0.232 -0.067  0.165  0.165 follow  1 follow 1.59
 86.99 2.79   0.66   1.08  1.27   18.7  31.0  0.327  0.000 -0.055  0.272 -0.083  0.189  0.189 follow  2 follow 0.50
 87.01 2.72   0.62   1.05  1.22   16.7  24.7  0.292  0.000 -0.079  0.214 -0.025  0.189  0.189 follow  2 follow 0.50
 87.04 2.75   0.58   1.03  1.18   17.7  23.2  0.310  0.000 -0.089  0.221 -0.032  0.189  0.189 follow  1 curve  2.17
 87.06 2.70   0.54   1.02  1.15   19.5  24.2  0.340  0.000 -0.106  0.234 -0.046  0.189  0.189 follow  1 curve  2.17
 87.09 2.83   0.51   1.01  1.14   18.5  20.2  0.323  0.000 -0.117  0.206  0.000  0.206  0.206 follow  1 curve  2.08
 87.11 2.80   0.50   1.00  1.13   18.7  18.0  0.327  0.000 -0.120  0.207  0.000  0.207  0.207 follow  1 curve  2.07
 87.14 2.74   0.44   0.76  1.14   19.0  15.7  0.332  0.000 -0.125  0.206  0.000  0.206  0.206 follow  1 curve  2.07
 87.16 2.70   0.43   0.74  1.13   18.7  15.5  0.327  0.000 -0.127  0.201  0.000  0.201  0.201 follow  2 follow 2.01
 87.19 2.69   0.35   0.72  1.12   17.7  12.5  0.310  0.000 -0.127  0.183  0.000  0.183  0.183 follow  2 follow 2.01
 87.21 2.65   0.33   0.71  1.12   19.0  12.7  0.332  0.000 -0.127  0.205  0.000  0.205  0.205 follow  2 follow 2.01
 87.24 2.65   0.38   0.72  1.12   19.0  12.5  0.332  0.000 -0.133  0.199  0.000  0.199  0.199 follow  2 follow 2.01
 87.26 2.64   0.34   0.74  1.12   18.5  11.5  0.323  0.000 -0.139  0.184  0.000  0.184  0.184 follow  2 follow 2.01
 87.29 2.63   0.31   0.79  1.12   17.0   9.0  0.297  0.000 -0.132  0.165  0.000  0.165  0.165 follow  2 curve  2.32
 87.31 2.63   0.32   0.81  1.13   16.0   7.2  0.279  0.000 -0.134  0.146  0.000  0.146  0.144 follow  2 curve  2.48
 87.34 2.63   0.31   0.80  1.11   17.2  11.2  0.301  0.000 -0.141  0.161  0.000  0.161  0.161 follow  1 curve  2.36
 87.36 2.41   0.29   0.80  1.13   14.5  11.0  0.253 -0.003 -0.405 -0.154  0.107  0.109  0.104 follow  1 cleara 1.79
 87.39 2.39   0.27   0.80  1.17    8.7  11.2  0.153 -0.032 -0.494 -0.330  0.000  0.059  0.055 follow  1 cleara 1.39
 87.41 2.43   0.30   0.80  0.86    2.5  11.7  0.044 -0.051 -0.526 -0.330  0.000  0.009  0.004 follow  1 cleara 1.47
 87.44 2.42   0.31   0.80  0.72   -3.5  12.7 -0.061 -0.043 -0.444 -0.330  0.000 -0.041 -0.045 follow  1 cleara 1.53
 87.46 2.19   0.30   0.79  0.66   -8.5  14.7 -0.148 -0.025 -0.425 -0.330  0.071 -0.093 -0.097 follow  1 cleara 1.68
 87.49 2.06   0.25   0.80  0.60  -15.0  14.7 -0.262  0.000 -0.461 -0.330  0.024 -0.141 -0.148 follow  1 cleara 1.68
 87.51 2.00   0.31   0.79  0.55  -21.5  15.7 -0.375  0.000 -0.460 -0.330  0.000 -0.191 -0.191 follow  1 curve  1.62
 87.54 1.90   0.31   0.80  0.51  -26.8  17.2 -0.467  0.000 -0.384 -0.330  0.000 -0.242 -0.252 follow  1 curve  1.62
 87.56 1.76   0.26   0.81  0.50  -32.0  17.2 -0.559  0.000 -0.305 -0.330  0.000 -0.292 -0.292 follow  1 curve  1.62
 87.59 1.73   0.19   0.83  0.48  -36.5  18.0 -0.637  0.000 -0.235 -0.330  0.000 -0.330 -0.330 follow  1 curve  1.62
 87.61 1.72   0.16   0.84  0.46  -39.5  19.0 -0.689  0.000 -0.180 -0.330  0.000 -0.330 -0.330 follow  1 cleara 0.22
 87.64 1.69   0.35   0.86  0.44  -41.8  19.5 -0.729  0.000 -0.115 -0.330  0.000 -0.330 -0.330 follow  1 cleara 0.01
 87.66 1.60   0.29   0.86  0.42  -43.0  21.0 -0.750  0.000 -0.057 -0.330  0.000 -0.330 -0.330 follow  1 cleara 0.00
 87.69 1.55   0.40   0.87  0.39  -43.5  22.0 -0.759  0.000  0.013 -0.330  0.000 -0.330 -0.330 follow  1 cleara 0.00
 87.71 1.53   0.43   0.87  0.38  -44.0  21.0 -0.768  0.000  0.063 -0.330  0.000 -0.330 -0.330 follow  1 cleara 0.00
 87.74 1.49   0.47   0.88  0.35  -42.5  22.7 -0.742  0.000  0.111 -0.330  0.000 -0.330 -0.330 follow  1 cleara 0.00
 87.76 1.35   0.42   0.88  0.34  -41.5  22.2 -0.724  0.000  0.142 -0.330  0.000 -0.330 -0.330 follow  1 creep  0.40
 87.79 1.26   0.36   0.88  0.32  -39.5  22.2 -0.689  0.000  0.133 -0.330  0.000 -0.330 -0.330 follow  1 creep  0.37
 87.81 1.19   0.33   0.89  0.30  -33.0   8.2 -0.576  0.000  0.124 -0.330  0.000 -0.330 -0.330 follow  1 creep  0.13
 87.84 1.13   0.30   0.89  0.29  -31.8   8.2 -0.554  0.000  0.119 -0.330  0.000 -0.330 -0.330 follow  1 creep  0.00
 87.86 1.11   0.26   0.90  0.29  -26.5  16.7 -0.463  0.000  0.101 -0.330  0.000 -0.330 -0.330 follow  1 creep  0.00
 87.89 1.08   0.25   0.91  0.28  -17.8  31.7 -0.310  0.000  0.093 -0.216  0.000 -0.280 -0.280 follow  1 creep  0.00
 87.92 1.04   0.24   0.91  0.29  -15.0  35.7 -0.262  0.000  0.075 -0.187  0.000 -0.229 -0.229 follow  1 creep  0.00
 87.94 0.93   0.23   0.91  0.29  -15.5  32.7 -0.271  0.000  0.058 -0.212  0.000 -0.212 -0.212 follow  1 creep  0.00
 87.97 0.87   0.22   0.91  0.29  -13.0  36.5 -0.227  0.000  0.047 -0.180  0.000 -0.180 -0.180         0 creep  0.00
 87.99 0.81   0.22   0.90  0.31    nan   nan    nan    nan    nan    nan  0.000 -0.180 -0.180         0 stop   0.00
 88.02 0.75   0.21   0.91  0.32  -13.3  34.7 -0.231  0.000  0.031 -0.200  0.000 -0.200 -0.200         0 creep  0.35
 88.04 0.67   0.21   0.90  0.34  -10.8  39.0 -0.188  0.000  0.032 -0.156  0.000 -0.156 -0.156         0 creep  0.40
 88.07 0.62   0.21   0.90  0.36    nan   nan    nan    nan    nan    nan  0.000 -0.156 -0.156         0 stop   0.00
 88.09 0.58   0.21   0.89  0.43  -10.5  39.2 -0.183 -0.029  0.027 -0.185  0.000 -0.185 -0.185         0 cleara 0.00
 88.12 0.52   0.20   0.89  0.73  -10.5  38.2 -0.183 -0.030  0.011 -0.202  0.000 -0.202 -0.202         0 cleara 0.00
 88.14 0.46   0.21   0.88  0.78  -10.0  39.0 -0.175 -0.032  0.016 -0.190  0.000 -0.190 -0.190         0 cleara 0.00
 88.17 0.41   0.21   0.88  0.80  -12.3  33.7 -0.214 -0.020  0.015 -0.218  0.000 -0.218 -0.219         0 cleara 0.00
 88.19 0.35   0.21   0.89  0.80   -9.8  38.2 -0.170 -0.034  0.010 -0.193  0.000 -0.193 -0.189         0 cleara 0.00
```
