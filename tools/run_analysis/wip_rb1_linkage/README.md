# WIP: car 1 (rb1) steering linkage calibration — handoff notes

Not finished; not wired into the simulator. Scripts are scratch analysis, ROS-free.

## Data used
- data/rb1/racerbot_runs/20261026-{174337,180228,194514,194714,194913} (fit) and
  20261030-{175418,175701,180531,180746,190734} (held out). Topics: /drive, /odom, /sensors/imu/raw (gyro z, deg/s, bias ~0.48), /steering/measured.
- 20261030-{175055,175304,190408,190519,190904} have no /steering/measured; 20261026-194422 never moved.
- Car 1's 2026-11-11 race runs (used for its speed loop) are not in data/.

## Gap found
- car 1 has no Floor profile (no linkage, trim, grip curve, odometry scale); docs/superpowers/plans/2026-11-12-post-race-speed-program.md Task 10 planned `sfu_course_2026_11_11_rb1` but it was never added.
- Gyro-implied rack angle, lateral accel < 2 m/s^2, steady steering, both days: command -0.330 -> -0.288 (right), +0.330 -> +0.326 (left), centre trim ~0.
  /steering/measured agrees on the asymmetry: -0.33 -> -0.254, +0.33 -> +0.305. rb1 gap follow clamps commands to +-0.33.
- /odom yaw over-reads the gyro by 6-30% (ratio 0.70-0.94).

## Replay results (replay.py: real /drive through the plant, 90 s, scored against the gyro where real and sim speed > 1 m/s)
| set | before (rb1 calibration) yaw RMS | best fit (right 0.80, left 1.05, delay 2) |
|---|---|---|
| fit, 10-26, mean of 5 | 0.186 rad/s | 0.145 rad/s |
| held out, 10-30, per run | .198 .190 .268 .274 .273 (mean 0.241) | .192 .183 .266 .291 .268 (mean 0.240) |
- Held-out gain is nil; the fit improvement does not transfer to the 10-30 day. Grid edges were hit (right 0.80, left 1.05): extend the grid.
- Speed RMS 0.39 (fit) / 0.61 (held out) is unchanged (only steering moved); wheel speed used as truth, odometry scale unmeasured for rb1.

## Next
1. Extend the grid; try grip_curve / grip scale for rb1 (fixture sfu_course_2026_11_11_grip_rb1_vesc.json) and trim; localized true speed (loc.py) for odom scale.
2. Only add a Floor if held-out yaw RMS improves; add test + docs/sim-calibration.md section, and a racerbot-docs PR.

## Environment
- No ROS on the VM: `pip install --user rosbags`; extract.py reads the MCAP bags into npz.
- tools/f1tenth_sim/setup.sh fails its origin check behind the git proxy; run its fetch/checkout/pip steps by hand, then
  PYTHONPATH=.sim/python:.sim/f1tenth_gym:src/gap_follow:src/pure_pursuit:tools/f1tenth_sim:tools NUMBA_CACHE_DIR=.sim/numba-cache
