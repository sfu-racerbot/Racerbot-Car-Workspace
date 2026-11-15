# rb1 stalls and slow spots, 20261111-153721 and 20261111-155542

Reproduce: `python3 tools/run_analysis/gap_stalls.py data/rb1/racerbot_runs/20261111-<run> --loc ~/.cache/racerbot/loc --line tools/f1tenth_sim/maps/car2_course_2026_11_11_planned_line.csv` (the line file comes with PR #54; `s` is the arc length along it from the race-day start pose, line length 27.88 m). Times are seconds from the first joy/odom/intent message. A stall is the published `/drive` speed exactly 0 for 0.5 s or more with LB held; a slow spot is true speed under 1.0 m/s, not a stall, more than 3 s after an LB press. Bound factor and side clearances come from the replay (steering parity with `/drive` over each whole run: 153721 max 0.42 rad RMS 0.044; 155542 max 0.46 rad RMS 0.045, so they are attributions at the listed ticks, each checked against the recorded speed command in the last column).

## 153721
```
/home/bryan/racerbot-ws/data/rb1/racerbot_runs/20261111-153721: line length 27.88 m
replay parity with /drive: 4699 ticks, 4699 paired, steering max 0.4186 rad rms 0.0435, speed median |err| 0.000 m/s, 8.8% of ticks more than 0.05 m/s off
STALLS (commanded speed 0 for 0.5 s or more, LB held)
   start    dur     s   off     x       y    bound               opponent   left  right   fwd   v   after_press replay/real cmd
SLOW SPOTS (true speed under 1.0 m/s, not a stall)
   start    dur     s   off     x       y    bound               opponent   left  right   fwd   v   after_press replay/real cmd
   96.92   0.07  16.04  0.11   -7.44   10.03  accel               clear       0.52   0.63  1.33 0.97   11.19   1.10/1.08
  107.16   1.07  14.99  0.09   -7.35    8.95  follow              follow      1.16   0.62  1.24 0.96   21.43   0.50/0.50
  124.09   0.67  15.89  0.12   -7.41    9.98  accel               clear       0.57   0.63  1.41 1.05    7.15   1.54/0.50
  192.39   0.87  15.29  0.07   -7.37    9.37  follow              follow      1.39   0.53  1.30 1.01   11.14   0.50/0.50
  203.28   1.22  14.84  0.07   -7.38    8.92  follow              follow      1.11   0.68  1.19 0.99   22.03   0.50/0.50
  215.15   1.03  14.84  0.21   -7.22    8.90  follow              follow      1.24   0.55  1.24 0.98    3.98   0.50/0.50
  226.43   2.02  14.84  0.07   -7.45    8.79  follow              follow      1.04   0.77  1.12 0.96   15.26   0.39/0.38
  239.46   1.07  15.29  0.17   -7.25    9.28  follow              follow      1.41   0.42  1.05 0.95   28.29   0.42/0.38
```

## 155542
```
/home/bryan/racerbot-ws/data/rb1/racerbot_runs/20261111-155542: line length 27.88 m
replay parity with /drive: 5657 ticks, 5657 paired, steering max 0.4603 rad rms 0.0448, speed median |err| 0.000 m/s, 6.8% of ticks more than 0.05 m/s off
STALLS (commanded speed 0 for 0.5 s or more, LB held)
   start    dur     s   off     x       y    bound               opponent   left  right   fwd   v   after_press replay/real cmd
   88.07   0.75  19.63  0.08  -10.50   11.28  stop(ttc_brake)                 0.21   0.90  0.36 0.62   11.83   0.00/0.00
   91.68   4.19  19.63  0.08  -10.49   11.28  clearance           launch      0.20   0.87  0.82 0.00    0.02   0.00/0.00
SLOW SPOTS (true speed under 1.0 m/s, not a stall)
   start    dur     s   off     x       y    bound               opponent   left  right   fwd   v   after_press replay/real cmd
   22.74   1.08  14.99  0.10   -7.34    9.08  follow              follow      1.23   0.59  1.15 1.02    5.27   0.50/0.50
   34.33   0.93  14.69  0.24   -7.21    8.75  follow              follow      1.24   0.58  0.98 1.00   16.86   0.50/0.50
   39.65   0.58  24.73  0.12  -14.73   12.20  follow              follow_memory  0.41   0.77  1.43 0.99   22.17   0.79/0.79
   46.83   0.88  15.29  0.16   -7.27    9.38  follow              follow      1.48   0.45  1.30 0.98   29.35   0.36/0.39
   69.66   1.13  14.99  0.20   -7.22    8.99  follow              follow      1.31   0.50  1.23 0.99   52.19   0.50/0.50
   84.68   1.23  14.99  0.08   -7.34    9.04  follow              follow      1.20   0.61  1.13 1.06    8.45   0.49/0.49
   87.94   0.10  19.63  0.03  -10.45   11.33  creep               follow      0.23   0.91  0.29 0.93   11.71   0.00/0.00
   95.90   0.15  19.93  0.24  -10.59   11.03  clearance           launch      0.20   0.99  1.68 0.00    4.24   2.49/0.12
  105.10   1.17  14.84  0.41   -7.02    8.85  follow              follow      1.66   0.33  0.70 1.06   13.43   0.52/0.52
  117.65   0.92  14.39  0.01   -7.53    8.43  follow              follow      0.82   0.94  1.28 1.00   25.99   0.55/0.55
  129.81   1.17  14.99  0.26   -7.17    8.94  follow              follow      1.35   0.47  1.15 1.05   38.15   0.50/0.50
  141.93   0.77  15.59  0.11   -7.34    9.62  follow              follow      0.76   0.49  1.33 1.03   50.26   0.50/0.50
  147.31   0.92  17.24  0.48   -8.54   10.67  follow              follow_memory  0.60   0.74  1.17 1.00   55.65   0.45/0.45
  154.55   1.07  15.44  0.21   -7.22    9.42  follow              follow      1.52   0.40  1.36 1.00   62.88   0.50/0.50
  159.14   2.32  17.54  0.29   -8.64   10.98  follow              follow      0.81   0.92  0.89 1.06   67.47   0.50/0.50
```

