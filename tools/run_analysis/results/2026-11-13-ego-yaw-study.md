# Does the tracker's ego yaw make walls look like cars? (rb1 20261111-155542 and -153721)

Measurement only; no behaviour change (`ego_yaw_study.py` patches `EgoPose.advance` in its own process). Replays each LB hold
through `gap_mirror.GapController` (recorded parameters) three ways: **steering** = as shipped (yaw rate = speed * tan(steering
estimate) / wheelbase); **gyro** = `/imu/data` angular_velocity.z for the yaw rate (the topic gap_follow's IMU features read;
a sample no older than `imu_timeout_sec` 0.1 s at the scan, else the steering model), wheel speed for translation; **truth** =
yaw rate and speed from loc.py's pose (central difference +-0.1 s). Episodes are the shipped variant's (follow_cost.py):
wall = the object at the U-turn inner corner (-8.1, 9.95), real = free-space objects that moved more than 0.5 m.
Per episode and variant: the track nearest the shipped object in the body frame (within 0.6 m): its world speed (median,
p90), fraction of ticks it is `moving`, fraction it feeds the follow law (moving or held), fraction of ticks any follow cap
applies; `path` = map-frame path length of the tracked point (the detection centroid) over the episode, m (it is a sum of
per-tick steps, so it includes jitter; net displacement is the `moved` column of the follow_cost results).

## A2 decision rule (applied as written)

Rule: implement A3 only if with gyro >= 18 of the 22 wall episodes lose the cap on >= 80% of ticks AND >= 6 of the 7 real
episodes keep it on >= 80%. Gyro fails it; truth fails it; so the rule says stop and report.

| run | variant | wall episodes losing the cap on >=80% of ticks | real episodes keeping it on >=80% |
|---|---|---|---|
| 155542 | steering (shipped) | 0 / 22 | 7 / 7 |
| 155542 | gyro | **2 / 22** (need 18) | 7 / 7 |
| 155542 | truth | **5 / 22** (need 18) | 6 / 7 |
| 153721 | steering | 0 / 24 | (no real moving episodes) |
| 153721 | gyro | 11 / 24 | - |
| 153721 | truth | 15 / 24 | - |

The brief's reading of "even truth fails" is "the cause is the sliding centroid". The data say it is both, plus a latch:

1. **Shipped ego inflates wall speeds a lot.** The wall track first becomes `moving` at 2.6-2.8 m range with a world speed of
   1.2-3.2 m/s (true yaw rate 0.7-1.7 rad/s), peaks 1.9-4.1 m/s (table "LATCH (steering)"). A 20% yaw error at 2.7 m and
   1.7 rad/s is 0.9 m/s, so the steering model's error (lag, lead, slip) is larger than the yaw-gain error alone.
2. **With true ego the wall is much slower but not slow enough.** Per-episode median speed 0.07-0.26 m/s, but the track
   still crosses the 0.6 m/s threshold: first `moving` at 0.60-0.66 m/s (range 1.9-2.5 m, or 0.5 m for four tracks that first count as moving only as the car passes), peak 0.64-0.84 m/s (the sliding
   centroid of a wall end: the tracked point's path is 0.13-1.8 m over the 1.0-1.9 s episodes, median 0.145 m over all 22).
3. **The cap outlives the spike.** Once a track has been `moving` it is a confirmed car and stays in `followed_body()` for
   `stopped_release_sec` (1.0 s = 40 ticks; "held" column) after the speed falls back, so a 15-35 tick excursion above 0.6
   m/s holds the 0.5 m/s follow floor for another second. With truth ego a wall track is `moving` for only 14-35 ticks of a
   110-250 tick life, yet it feeds the cap for 37-44 held ticks on 16 of 22 episodes (7 distinct tracks).
4. Real cars: speeds 2-3.5 m/s median in every variant; they keep the cap in 7/7 (gyro) and 6/7 (truth; the exception is a
   one-tick episode).

So a better ego yaw alone does not remove the wall episodes (gyro 2/22). What the data point to for the next design: the
moving threshold sits right at the wall's residual speed (0.6 vs peaks 0.64-0.84), and the 1.0 s hold converts any excursion
into a 1.4-1.9 s slow zone. Not implemented: A3 (the rule says stop).

## Full output, run 155542
```
WALL episodes at the U-turn inner corner: 22
   start   dur     s |                steering                |                  gyro                  |                 truth                 
                      |   vmed  vp90  moving  fed  cap_applies  path |   vmed  vp90  moving  fed  cap_applies  path |   vmed  vp90  moving  fed  cap_applies  path
   22.24  1.35  14.99 |  0.21  0.60   0.26  1.00    1.00    0.28 |  0.19  0.30   0.00  0.00    0.00    0.27 |  0.08  0.18   0.00  0.00    0.00    0.29
   24.20  0.05  15.89 |  0.61  0.62   1.00  1.00    1.00    0.03 |  0.76  0.77   1.00  1.00    1.00    0.03 |  0.50  0.50   0.00  0.00    0.00    0.03
   34.31  1.05  14.84 |  0.17  0.20   0.00  0.69    1.00    0.13 |  0.16  0.20   0.00  0.00    1.00    0.13 |  0.07  0.14   0.00  0.69    1.00    0.13
   35.71  0.18  15.74 |  0.74  0.75   1.00  1.00    1.00    0.17 |  0.75  0.86   0.86  0.86    0.86    0.17 |  0.54  0.55   0.00  0.00    0.00    0.16
   46.32  1.07  15.29 |  0.28  0.47   0.05  0.98    1.00    0.16 |  0.21  0.30   0.09  0.98    1.00    0.17 |  0.10  0.33   0.07  0.98    1.00    0.16
   48.01  0.05  16.19 |  0.66  0.67   1.00  1.00    1.00    0.02 |  0.74  0.76   1.00  1.00    1.00    0.02 |  0.71  0.72   1.00  1.00    1.00    0.02
   57.92  0.53  15.44 |  0.46  0.56   1.00  1.00    1.00    0.21 |  0.34  0.41   0.29  1.00    0.71    0.22 |  0.39  0.57   0.19  1.00    0.86    0.24
   58.54  0.03  15.89 |  0.46  0.46   1.00  1.00    1.00    0.00 |  0.49  0.49   0.00  1.00    1.00    0.00 |  0.29  0.29   0.00  1.00    1.00    0.00
   58.60  0.03  15.89 |  0.57  0.57   1.00  1.00    1.00    0.00 |  0.54  0.54   0.00  1.00    1.00    0.00 |  0.31  0.31   0.00  1.00    1.00    0.00
   58.65  0.03  16.04 |  0.61  0.61   1.00  1.00    1.00    0.00 |  0.54  0.54   0.00  1.00    1.00    0.00 |  0.28  0.28   0.00  1.00    1.00    0.00
   58.70  0.03  16.04 |  0.59  0.59   1.00  1.00    1.00    0.00 |  0.52  0.52   0.00  1.00    1.00    0.00 |  0.29  0.29   0.00  1.00    1.00    0.00
   69.24  1.88  14.99 |  0.19  0.74   0.17  0.71    1.00    1.37 |  0.19  0.39   0.04  0.04    1.00    1.37 |  0.09  0.30   0.04  0.04    1.00    1.37
   84.25  1.38  14.99 |  0.18  0.54   0.27  1.00    1.00    0.74 |  0.19  0.24   0.00  0.00    0.00    0.72 |  0.10  0.31   0.00  0.00    0.00    0.75
  104.65  1.78  14.84 |  0.21  3.10   0.20  0.76    1.00    1.77 |  0.18  2.59   0.20  0.20    1.00    1.78 |  0.24  2.91   0.30  0.76    1.00    1.80
  106.82  0.12  16.04 |  0.76  0.77   1.00  1.00    1.00    0.06 |  0.81  0.89   1.00  1.00    1.00    0.06 |  0.64  0.65   1.00  1.00    1.00    0.06
  117.10  1.55  14.24 |  0.21  0.54   0.35  1.00    1.00    1.11 |  0.18  0.30   0.06  0.06    0.81    1.08 |  0.07  0.30   0.06  0.06    0.81    1.09
  119.02  0.18  15.74 |  0.83  0.86   1.00  1.00    1.00    0.11 |  0.82  0.90   1.00  1.00    1.00    0.10 |  0.50  0.57   0.00  0.00    0.00    0.11
  129.44  1.55  14.84 |  0.16  0.62   0.19  0.84    1.00    1.27 |  0.19  0.36   0.06  0.06    1.00    1.27 |  0.12  0.34   0.08  0.73    1.00    1.30
  131.36  0.12  15.89 |  0.75  0.79   1.00  1.00    1.00    0.11 |  0.78  0.87   1.00  1.00    1.00    0.11 |  0.67  0.74   0.80  0.80    0.80    0.11
  141.31  0.08  14.54 |  0.36  0.40   1.00  1.00    1.00    0.02 |  0.41  0.44   1.00  1.00    1.00    0.02 |  0.18  0.22   0.00  1.00    1.00    0.02
  141.43  1.50  15.59 |  0.26  0.37   0.18  0.85    1.00    0.27 |  0.23  0.34   0.00  0.65    1.00    0.27 |  0.14  0.20   0.00  0.53    1.00    0.29
  154.07  1.45  15.29 |  0.22  0.56   0.31  1.00    1.00    0.56 |  0.22  0.32   0.12  0.81    0.81    0.55 |  0.13  0.23   0.02  0.69    0.79    0.58
  steering  episodes losing the cap on >=80% of ticks: 0/22; keeping it on >=80%: 22/22; median of per-episode speed median 0.41 m/s
  gyro      episodes losing the cap on >=80% of ticks: 2/22; keeping it on >=80%: 19/22; median of per-episode speed median 0.37 m/s
  truth     episodes losing the cap on >=80% of ticks: 5/22; keeping it on >=80%: 16/22; median of per-episode speed median 0.26 m/s
REAL moving-car episodes (moved > 0.5 m): 7
   start   dur     s |                steering                |                  gyro                  |                 truth                 
                      |   vmed  vp90  moving  fed  cap_applies  path |   vmed  vp90  moving  fed  cap_applies  path |   vmed  vp90  moving  fed  cap_applies  path
   33.53  0.75  14.09 |  3.68  5.26   0.70  1.00    1.00    2.77 |  3.21  3.91   0.70  0.70    1.00    2.74 |  3.35  3.90   0.77  1.00    1.00    2.73
   38.75  1.35  24.43 |  2.08  2.55   0.28  0.28    1.00    1.18 |  1.91  2.12   0.28  0.28    1.00    1.15 |  3.42  4.22   0.28  0.24    0.98    1.31
   86.89  0.15  17.54 |  2.04  2.12   1.00  1.00    1.00    1.30 |  1.97  2.01   1.00  1.00    1.00    1.31 |  2.17  2.38   1.00  1.00    1.00    1.20
   97.60  0.05  22.78 |  1.15  1.15   1.00  1.00    1.00    0.30 |  1.21  1.21   1.00  1.00    1.00    0.31 |  0.73  0.73   1.00  1.00    0.00    0.17
  100.59  1.20   4.05 |  2.38  3.37   0.67  0.67    1.00    1.91 |  2.43  2.96   0.67  0.67    1.00    1.91 |  3.01  4.60   0.67  0.67    1.00    1.92
  114.41  1.75   8.54 |  3.49  3.71   0.77  0.77    1.00    8.40 |  3.52  3.76   0.77  0.77    1.00    8.44 |  3.42  8.00   0.77  0.77    1.00    8.57
  124.73  1.70   3.15 |  3.13  4.03   0.46  0.46    1.00    2.61 |  3.21  3.99   0.46  0.46    1.00    2.62 |  3.07  4.94   0.46  0.46    1.00    2.66
  steering  episodes losing the cap on >=80% of ticks: 0/7; keeping it on >=80%: 7/7; median of per-episode speed median 2.38 m/s
  gyro      episodes losing the cap on >=80% of ticks: 0/7; keeping it on >=80%: 7/7; median of per-episode speed median 2.43 m/s
  truth     episodes losing the cap on >=80% of ticks: 1/7; keeping it on >=80%: 6/7; median of per-episode speed median 3.07 m/s
LATCH (steering): why the follow cap holds on the wall episodes
   start   dur  track  life moving held  first_moving(s from ep start)  speed  range  yaw_rate  peak_speed
   22.24  1.35     38   144     50    40        -0.45         3.18   2.70     1.71     3.20
   24.20  0.05     38   144     50    40        -2.41         3.18   2.70     1.71     3.20
   34.31  1.05    100   128     47    39        -0.78         3.00   2.63     1.24     3.57
   35.71  0.18    100   128     47    39        -2.18         3.00   2.63     1.24     3.57
   46.32  1.07    166   119     38    40        -0.55         2.77   2.67     1.28     3.78
   48.01  0.05    166   119     38    40        -2.23         2.77   2.67     1.28     3.78
   57.92  0.53    233   117     71    23        -0.70         1.61   2.71     1.03     3.95
   58.54  0.03    233   117     71    23        -1.33         1.61   2.71     1.03     3.95
   58.60  0.03    233   117     71    23        -1.38         1.61   2.71     1.03     3.95
   58.65  0.03    233   117     71    23        -1.43         1.61   2.71     1.03     3.95
   58.70  0.03    233   117     71    23        -1.48         1.61   2.71     1.03     3.95
   69.24  1.88    306   138     52    40        -0.48         1.23   2.63     0.82     2.60
   84.25  1.38     40   141     42    40        -0.43         2.27   2.65     1.17     3.26
  104.65  1.78     59   137     42    40        -0.27         2.15   2.64     1.16     4.07
  106.82  0.12     59   137     42    40        -2.44         2.15   2.64     1.16     4.07
  117.10  1.55    132   131     58    40        -0.20         1.88   2.77     1.69     1.88
  119.02  0.18    132   131     58    40        -2.12         1.88   2.77     1.69     1.88
  129.44  1.55    192   127     48    40        -0.47         2.40   2.58     0.72     3.54
  131.36  0.12    192   127     48    40        -2.39         2.40   2.58     0.72     3.54
  141.31  0.08    265   128     39    40        -0.57         3.21   2.69     1.53     3.97
  141.43  1.50    265   128     39    40        -0.70         3.21   2.69     1.53     3.97
  154.07  1.45    337   123     54    40        -0.60         1.88   2.68     1.18     3.75
LATCH (truth): why the follow cap holds on the wall episodes
   start   dur  track  life moving held  first_moving(s from ep start)  speed  range  yaw_rate  peak_speed
   22.24  1.35     40   254     23     0         2.08         0.64   0.49     1.52     1.88
   24.20  0.05     40   254     23     0         0.13         0.64   0.49     1.52     1.88
   34.31  1.05    101   128     23    40        -0.40         0.60   2.11     1.31     0.70
   35.71  0.18    101   128     23    40        -1.81         0.60   2.11     1.31     0.70
   46.32  1.07    167   117     35    40        -0.38         0.61   2.37     1.77     0.73
   48.01  0.05    167   117     35    40        -2.06         0.61   2.37     1.77     0.73
   57.92  0.53    238   111     29    44        -0.50         0.65   2.35     1.62     0.72
   58.54  0.03    238   111     29    44        -1.13         0.65   2.35     1.62     0.72
   58.60  0.03    238   111     29    44        -1.18         0.65   2.35     1.62     0.72
   58.65  0.03    238   111     29    44        -1.23         0.65   2.35     1.62     0.72
   58.70  0.03    238   111     29    44        -1.28         0.65   2.35     1.62     0.72
   69.24  1.88    310   136     22     0         2.03         0.63   0.54     0.67     0.72
   84.25  1.38     40   254     23     0       -59.93         0.64   0.49     1.52     1.88
  104.65  1.78     59   148     25    40         0.32         0.64   1.87     1.21     1.13
  106.82  0.12     59   148     25    40        -1.84         0.64   1.87     1.21     1.13
  117.10  1.55    161   127     15     0         2.24         0.66   0.56     1.72     0.84
  119.02  0.18    161   127     15     0         0.32         0.66   0.56     1.72     0.84
  129.44  1.55    223   127     23    40        -0.05         0.60   2.04     1.81     0.82
  131.36  0.12    223   127     23    40        -1.97         0.60   2.04     1.81     0.82
  141.31  0.08    306   122     14    37        -0.42         0.62   2.46     1.71     0.68
  141.43  1.50    306   122     14    37        -0.55         0.62   2.46     1.71     0.68
  154.07  1.45    379   123     28    40        -0.35         0.61   2.27     1.65     0.64
```

## Full output, run 153721
```
/usr/lib/python3/dist-packages/numpy/lib/nanfunctions.py:1215: RuntimeWarning: Mean of empty slice
  return np.nanmean(a, axis, out=out, keepdims=keepdims)
WALL episodes at the U-turn inner corner: 24
   start   dur     s |                steering                |                  gyro                  |                 truth                 
                      |   vmed  vp90  moving  fed  cap_applies  path |   vmed  vp90  moving  fed  cap_applies  path |   vmed  vp90  moving  fed  cap_applies  path
   84.20  0.03  14.69 |  0.37  0.37   1.00  1.00    1.00    0.00 |  0.38  0.38   1.00  1.00    1.00    0.00 |  0.31  0.31   1.00  1.00    0.00    0.00
   84.27  0.50  15.44 |  0.58  0.62   1.00  1.00    1.00    0.17 |  0.34  0.40   0.35  1.00    0.90    0.17 |  0.19  0.30   0.00  1.00    1.00    0.16
   96.07  0.58  15.29 |  0.50  0.57   1.00  1.00    1.00    0.19 |  0.34  0.42   0.00  0.00    0.00    0.19 |  0.22  0.36   0.00  0.00    0.00    0.20
   96.74  0.03  15.89 |  0.46  0.46   1.00  1.00    1.00    0.00 |  0.49  0.49   0.00  0.00    0.00    0.00 |  0.23  0.23   0.00  0.00    0.00    0.00
   96.79  0.03  15.89 |  0.55  0.55   1.00  1.00    1.00    0.00 |  0.54  0.54   0.00  0.00    0.00    0.00 |  0.23  0.23   0.00  0.00    0.00    0.00
   96.84  0.03  15.89 |  0.60  0.60   1.00  1.00    1.00    0.00 |  0.57  0.57   0.00  0.00    0.00    0.00 |  0.20  0.20   0.00  0.00    0.00    0.00
   96.89  0.03  15.89 |  0.57  0.57   1.00  1.00    1.00    0.00 |  0.54  0.54   0.00  0.00    0.00    0.00 |  0.20  0.20   0.00  0.00    0.00    0.00
  106.68  1.28  14.84 |  0.16  0.71   0.22  1.00    1.00    0.30 |  0.18  0.38   0.00  0.00    0.00    0.28 |  0.06  0.32   0.06  0.88    0.94    0.32
  108.58  0.10  15.89 |  0.64  0.66   1.00  1.00    1.00    0.08 |  0.75  0.81   1.00  1.00    1.00    0.08 |  0.35  0.45   0.00  0.00    0.00    0.09
  123.42  0.10  14.84 |  0.34  0.41   1.00  1.00    1.00    0.04 |  0.41  0.45   1.00  1.00    0.75    0.03 |  0.59  0.60   1.00  1.00    1.00    0.03
  123.57  0.48  15.44 |  0.44  0.56   1.00  1.00    1.00    0.22 |  0.37  0.42   0.11  1.00    0.89    0.23 |  0.31  0.51   0.00  1.00    1.00    0.22
  124.14  0.03  15.89 |  0.40  0.40   1.00  1.00    1.00    0.00 |  0.39  0.39   0.00  1.00    1.00    0.00 |  0.33  0.33   0.00  1.00    1.00    0.00
  124.19  0.03  15.89 |  0.43  0.43   1.00  1.00    1.00    0.00 |  0.36  0.36   0.00  1.00    1.00    0.00 |  0.28  0.28   0.00  1.00    1.00    0.00
  124.24  0.70  16.04 |  0.26  0.47   0.36  1.00    1.00    0.12 |  0.27  0.31   0.00  0.54    0.54    0.12 |  0.13  0.26   0.00  0.43    0.43    0.12
  180.39  0.48  14.84 |  0.39  0.76   0.37  1.00    1.00    0.15 |  0.27  0.42   0.00  0.00    0.00    0.14 |  0.21  0.34   0.00  0.00    0.00    0.15
  191.80  1.20  15.14 |  0.23  0.50   0.17  1.00    1.00    0.33 |  0.23  0.39   0.00  0.00    0.00    0.32 |  0.09  0.20   0.00  0.00    0.00    0.34
  202.85  1.45  14.84 |  0.17  0.79   0.31  1.00    1.00    0.47 |  0.17  0.26   0.00  0.00    0.00    0.46 |  0.08  0.28   0.00  0.00    0.00    0.48
  204.77  0.15  15.74 |  0.70  0.73   1.00  1.00    1.00    0.06 |  0.76  0.84   1.00  1.00    1.00    0.05 |  0.45  0.48   0.00  0.00    0.00    0.06
  214.73  1.40  14.84 |  0.16  0.58   0.23  1.00    1.00    0.33 |  0.18  0.29   0.00  0.00    0.00    0.35 |  0.09  0.26   0.00  0.00    0.00    0.34
  216.58  0.12  15.89 |  0.71  0.72   1.00  1.00    1.00    0.08 |  0.84  0.90   1.00  1.00    0.80    0.08 |  0.43  0.53   0.00  0.00    0.00    0.09
  226.08  1.88  14.84 |  0.19  0.31   0.05  0.59    1.00    0.45 |  0.15  0.26   0.00  0.00    0.00    0.44 |  0.14  0.22   0.00  0.00    0.00    0.45
  228.88  0.15  15.74 |  0.70  0.73   1.00  1.00    1.00    0.11 |  0.76  0.79   1.00  1.00    1.00    0.11 |  0.66  0.70   0.83  0.83    0.83    0.11
  239.01  1.43  15.29 |  0.19  0.59   0.25  1.00    1.00    0.47 |  0.17  0.25   0.04  0.74    0.84    0.48 |  0.14  0.30   0.00  0.65    0.84    0.50
  240.88  0.05  16.04 |  0.60  0.60   1.00  1.00    1.00    0.02 |  0.75  0.77   1.00  1.00    1.00    0.02 |  0.52  0.54   0.00  0.00    0.00    0.02
  steering  episodes losing the cap on >=80% of ticks: 0/24; keeping it on >=80%: 24/24; median of per-episode speed median 0.44 m/s
  gyro      episodes losing the cap on >=80% of ticks: 11/24; keeping it on >=80%: 11/24; median of per-episode speed median 0.37 m/s
  truth     episodes losing the cap on >=80% of ticks: 15/24; keeping it on >=80%: 8/24; median of per-episode speed median 0.22 m/s
REAL moving-car episodes (moved > 0.5 m): 0
   start   dur     s |                steering                |                  gyro                  |                 truth                 
                      |   vmed  vp90  moving  fed  cap_applies  path |   vmed  vp90  moving  fed  cap_applies  path |   vmed  vp90  moving  fed  cap_applies  path
  steering  episodes losing the cap on >=80% of ticks: 0/0; keeping it on >=80%: 0/0; median of per-episode speed median nan m/s
  gyro      episodes losing the cap on >=80% of ticks: 0/0; keeping it on >=80%: 0/0; median of per-episode speed median nan m/s
  truth     episodes losing the cap on >=80% of ticks: 0/0; keeping it on >=80%: 0/0; median of per-episode speed median nan m/s
LATCH (steering): why the follow cap holds on the wall episodes
   start   dur  track  life moving held  first_moving(s from ep start)  speed  range  yaw_rate  peak_speed
   84.20  0.03     38   110     48     0        -0.62         2.86   2.69     1.45     4.07
   84.27  0.50     38   110     48     0        -0.70         2.86   2.69     1.45     4.07
   96.07  0.58     74   141     83    22        -0.65         2.70   2.68     1.27     4.19
   96.74  0.03     74   141     83    22        -1.32         2.70   2.68     1.27     4.19
   96.79  0.03     74   141     83    22        -1.37         2.70   2.68     1.27     4.19
   96.84  0.03     74   141     83    22        -1.42         2.70   2.68     1.27     4.19
   96.89  0.03     74   141     83    22        -1.47         2.70   2.68     1.27     4.19
  106.68  1.28    165   135     48    40        -0.40         2.15   2.71     1.26     2.99
  108.58  0.10    165   135     48    40        -2.30         2.15   2.71     1.26     2.99
  123.42  0.10     41   166     92    32       -39.69         7.99   6.03     1.73     7.99
  123.57  0.48     41   166     92    32       -39.84         7.99   6.03     1.73     7.99
  124.14  0.03     41   166     92    32       -40.42         7.99   6.03     1.73     7.99
  124.19  0.03     41   166     92    32       -40.47         7.99   6.03     1.73     7.99
  124.24  0.70     41   166     92    32       -40.52         7.99   6.03     1.73     7.99
  180.39  0.48     24    95     29    12        -0.55         3.40   2.62     0.88     4.22
  191.80  1.20     50   161     37    40        -0.43         2.20   2.65     1.15     3.42
  202.85  1.45    111   138     55    40        -0.40         1.77   2.69     1.03     3.10
  204.77  0.15    111   138     55    40        -2.32         1.77   2.69     1.03     3.10
  214.73  1.40     10   185     65    45      -127.78         0.83   1.63     0.83     4.31
  216.58  0.12     10   185     65    45      -129.63         0.83   1.63     0.83     4.31
  226.08  1.88     92   171     39    40        -0.47         3.09   2.62     1.14     3.57
  228.88  0.15     92   171     39    40        -3.27         3.09   2.62     1.14     3.57
  239.01  1.43    188   132     55    43        -0.60         3.01   2.70     1.46     4.03
  240.88  0.05    188   132     55    43        -2.47         3.01   2.70     1.46     4.03
LATCH (truth): why the follow cap holds on the wall episodes
   start   dur  track  life moving held  first_moving(s from ep start)  speed  range  yaw_rate  peak_speed
   84.20  0.03     35   116     23    22        -0.55         0.60   2.55     1.67     0.77
   84.27  0.50     35   116     23    22        -0.62         0.60   2.55     1.67     0.77
   96.07  0.58     75   133      0     0          nan          nan    nan      nan     0.66
   96.74  0.03     75   133      0     0          nan          nan    nan      nan     0.66
   96.79  0.03     75   133      0     0          nan          nan    nan      nan     0.66
   96.84  0.03     75   133      0     0          nan          nan    nan      nan     0.66
   96.89  0.03     75   133      0     0          nan          nan    nan      nan     0.66
  106.68  1.28    163   138     17    45        -0.10         0.66   2.19     1.67     3.06
  108.58  0.10    163   138     17    45        -2.00         0.66   2.19     1.67     3.06
  123.42  0.10     39   151     20    40        -0.37         0.60   2.29     1.73     1.59
  123.57  0.48     39   151     20    40        -0.53         0.60   2.29     1.73     1.59
  124.14  0.03     39   151     20    40        -1.10         0.60   2.29     1.73     1.59
  124.19  0.03     39   151     20    40        -1.15         0.60   2.29     1.73     1.59
  124.24  0.70     39   151     20    40        -1.20         0.60   2.29     1.73     1.59
  180.39  0.48     24   117      0     0          nan          nan    nan      nan     0.58
  191.80  1.20     50   148     15     0         1.84         0.63   0.49     1.02     0.74
  202.85  1.45    111   142      8     0         2.25         0.70   0.54     1.96     0.78
  204.77  0.15    111   142      8     0         0.32         0.70   0.54     1.96     0.78
  214.73  1.40     10   200     31     0       -95.43         1.50   4.37     1.43     4.30
  216.58  0.12     10   200     31     0       -97.28         1.50   4.37     1.43     4.30
  226.08  1.88     95   170     15     0         2.82         0.64   0.52    -0.47     0.71
  228.88  0.15     95   170     15     0         0.02         0.64   0.52    -0.47     0.71
  239.01  1.43    192   141     30    37        -0.47         0.64   2.51     1.68     0.86
  240.88  0.05    192   141     30    37        -2.35         0.64   2.51     1.68     0.86
```
