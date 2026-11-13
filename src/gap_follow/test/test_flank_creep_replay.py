"""Replay of the recorded wall-alongside latch (pure, no rclpy).

Fixture: rb1 run 20261111-155542, 88.5-90.0 s (60 scans, loc.py poses), the
window inside the 8.3 s latch at 87.6-95.9 s (plan
docs/superpowers/plans/2026-11-12-post-race-speed-program.md section 1.2).
"""
import math
import os

import numpy as np
import pytest

from gap_follow import gap_logic

FIXTURE = os.path.join(os.path.dirname(__file__), 'data',
                       'flank_latch_rb1_20261111-155542.npz')
CAR_WIDTH, CAR_LENGTH, WHEELBASE, LASER_X = 0.31, 0.58, 0.36, 0.26
SWEPT_MARGIN = 0.05
MAX_RANGE = 10.0
FOV = math.radians(180.0)  # gap_follow.yaml forward_fov_deg


@pytest.fixture(scope='module')
def scans():
    data = np.load(FIXTURE)
    angle_min = float(data['angle_min'])
    inc = float(data['angle_increment'])
    out = []
    for ranges in data['ranges']:
        clean, valid = gap_logic.sanitize_ranges(
            ranges.astype(float), MAX_RANGE, float(data['range_min']))
        angles = angle_min + np.arange(len(ranges)) * inc
        window = np.abs(angles) <= FOV / 2.0
        out.append((clean[window], valid[window], angles[window]))
    assert len(out) >= 50
    return out


def _fraction(scans, predicate):
    return sum(1 for s in scans if predicate(*s)) / len(scans)


def _swept(steering):
    def f(clean, valid, angles):
        return gap_logic.swept_path_clearance(
            clean, valid, angles, math.tan(steering) / WHEELBASE,
            CAR_WIDTH / 2.0 + SWEPT_MARGIN, CAR_LENGTH, WHEELBASE, LASER_X,
            0.0, 5.0)
    return f


def _body(steering, half_width=0.175, max_length=0.5):
    def f(clean, valid, angles):
        return gap_logic.body_arc_free_distance(
            clean, valid, angles, math.tan(steering) / WHEELBASE, half_width,
            CAR_LENGTH, WHEELBASE, LASER_X, 0.0, max_length)
    return f


def test_the_recorded_scans_reproduce_the_latch(scans):
    # recorded measurement: rb1 20261111-155542, 88.5-90.0 s
    f = _swept(-0.20)
    assert _fraction(scans, lambda *s: f(*s) == 0.0) >= 0.95


def test_the_body_arc_is_free_on_the_recorded_scans(scans):
    # recorded measurement: rb1 20261111-155542, 88.5-90.0 s
    f = _body(-0.20)
    assert _fraction(scans, lambda *s: f(*s) == math.inf) >= 0.95


def test_steering_into_the_wall_is_not_free_on_the_recorded_scans(scans):
    # recorded measurement: rb1 20261111-155542, 88.5-90.0 s (wall on the left)
    f = _body(+0.33)
    assert _fraction(scans, lambda *s: math.isfinite(f(*s)) and f(*s) < 0.5) >= 0.95
