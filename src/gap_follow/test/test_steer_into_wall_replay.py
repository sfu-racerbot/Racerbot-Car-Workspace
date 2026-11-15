"""Phase 4 Task 13: rb1 steered into the left wall at 84.5-88.2 s of run 20261111-155542 (pure, no rclpy).

Fixture steer_into_wall_rb1_20261111-155542.npz (tools/run_analysis/scan_window_fixture.py): the 148 scans of that
window in seconds from the bag's first /scan or /joy (82.40-86.10; intent_report.py's clock is 2.099 s earlier, so
84.5-88.2 there), with the wheel speed, the /imu/data gyro the node's steering damping read, the /drive the node
published and the run's gap_follow_node parameters. Diagnosis and the stage-by-stage table:
tools/run_analysis/results/2026-11-13-rb1-steer-into-wall.md.

The window is replayed through tools/f1tenth_sim/gap_mirror.GapController (the node's own per-scan logic, held to
the node by test_gap_mirror_parity.py) cold-started at the first scan, so the first ticks' steering differs from
the recorded /drive by the missing warm-up; the stage table in the results file is the warm replay, which
reproduces /drive within 0.0164 rad over the whole window.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'tools' / 'f1tenth_sim'))
import gap_mirror  # noqa: E402

FIXTURE = HERE / 'data' / 'steer_into_wall_rb1_20261111-155542.npz'
LEFT_BAND = (math.radians(60.0), math.radians(120.0))


class _Lidar:
    def __init__(self, data):
        self.angle_min = float(data['angle_min'])
        self.angle_increment = float(data['angle_increment'])
        self.range_min = float(data['range_min'])


@pytest.fixture(scope='module')
def recorded():
    return np.load(FIXTURE)


@pytest.fixture(scope='module')
def replay():
    data = np.load(FIXTURE)
    params = json.loads(str(data['params_json']))
    ctrl = gap_mirror.GapController(params, _Lidar(data))
    angles = float(data['angle_min']) + np.arange(data['ranges'].shape[1]) * float(data['angle_increment'])
    band = (angles >= LEFT_BAND[0]) & (angles <= LEFT_BAND[1])
    rows = []
    for n, scan in enumerate(data['ranges'].astype(np.float64)):
        yaw = float(data['yaw_rates'][n])
        steering, speed = ctrl.command(scan, float(data['odom_speeds'][n]), n, yaw_rate=None if math.isnan(yaw) else yaw)
        r = scan[band]
        r = r[np.isfinite(r) & (r > float(data['range_min'])) & (r < params['max_range'])]
        rows.append(dict(steering=steering, speed=speed, left=float(r.min()) if r.size else math.inf,
                         side_left=ctrl.telemetry.get('side_left', math.nan)))
    return rows


def test_the_replay_ran_the_whole_window_and_reproduces_the_wall_approach(replay, recorded):
    # recorded measurement: rb1 20261111-155542, 84.5-88.2 s. The node published up to +0.33 rad to the left
    # (/drive in the fixture) while the left wall closed in to 0.19 m; the cold-started replay, which lacks the
    # warm-up, commands above +0.25 rad on the same scans and sees the same wall.
    assert len(replay) == 148
    assert np.nanmax(recorded['drive_steering']) == pytest.approx(0.328, abs=0.005)    # rad, recorded
    assert max(r['steering'] for r in replay) > 0.25       # rad
    assert min(r['left'] for r in replay) < 0.25           # m


@pytest.mark.xfail(strict=True, reason='Phase 4 Task 13: car 1 steered into the left wall; see '
                   'docs/superpowers/plans/2026-11-12-post-race-speed-program.md; remove when fixed')
def test_the_left_clearance_stays_above_30_cm_over_the_window(replay):
    # Today the left side range (minimum raw range 60-120 degrees left) falls to 0.19 m at 87.99 s and the car
    # touched the wall at 87.34 s (gyro 1.4 -> 4.9 rad/s, horizontal acceleration 29 m/s^2 in /imu/data).
    # NOTE: the scans are recorded, so this reading of the window does not change with the controller; it
    # documents the recorded outcome the way the brief asked. A closed-loop version (scans synthesised from the
    # race-day map along the replayed car's own path) is the test that can flip.
    assert min(r['left'] for r in replay) > 0.30           # m
