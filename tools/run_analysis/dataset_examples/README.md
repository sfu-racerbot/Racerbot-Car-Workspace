# dataset_examples

> **Who this is for:** anyone who wants numbers from a recorded run for state estimation or pure pursuit work.
> **Read first:** [docs/dataset-examples.md](../../../docs/dataset-examples.md), the full guide: the maths, the topics, results on real runs and limitations.
> **You'll be able to:** find the script for each dataset use and run it.

One offline analysis per dataset use. [Odometry](../../../docs/glossary.md#odometry--odom),
the map pose and the IMU are compared the same way in every script (`common.py`).

| Script | Use | Needs |
|---|---|---|
| `localization.py` | odometry drift against the map pose; `--map` adds loc.py | `/odom`, `/odom/imu`, a map pose |
| `velocity_estimation.py` | wheel and Kalman speed against map speed; accelerometer check | `/odom`, a map pose, `/imu/data` |
| `steering_response.py` | bicycle model, delay, steering map, understeer by speed | `/ackermann_cmd`, `/odom`, `/imu/data` |
| `lateral_acceleration.py` | v·r against the accelerometer; grip curve and sustained limit | `/odom`, `/imu/data`, `/ackermann_cmd` |
| `algorithm_accuracy.py` | command tracking; path tracking against the run's own racing line | `/drive`, `/odom`, `/imu/data` (+ the particle filter's pose) |
| `run_all.py` | all five, reading the bag once | |
| `common.py` | shared: loading, time grid, driving mask, map pose in base_link, jumps, axis check | |

From the workspace root, no ROS needed (`pip install rosbags==0.10.11` instead):

```bash
python3 tools/run_analysis/dataset_examples/run_all.py data/rb2/racerbot_runs/20261111-152132
```

Tests: `python3 -m pytest tools/run_analysis/tests/test_dataset_examples.py`.
