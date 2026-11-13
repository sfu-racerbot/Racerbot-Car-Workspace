"""track_truth of range_rate_study.py: the per-track moving/static label from world positions, independent of the radial
residual. Oracle: closed-form world transform of a point fixed in the world while the ego moves and yaws."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'gap_follow'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'f1tenth_sim'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'pure_pursuit'))
from run_analysis import range_rate_study as rr

DT = 0.025


def rows_for(world_of_t, n=20):
    """Ego drives along x at 2 m/s while yawing 1 rad/s; the track's world position is world_of_t(t); each row stores
    the track in the body frame (inverse of the transform under test)."""
    rows = []
    for i in range(n):
        t = i * DT
        x, y, yaw = 2.0 * t, 0.3 * t, 1.0 * t
        wx, wy = world_of_t(t)
        dx, dy = wx - x, wy - y
        bx, by = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
        rows.append(dict(t=t, x=x, y=y, yaw=yaw, tracks=[(bx, by, 0.0, False, False, False, 7, False, 0.0, 0)]))
    return rows


def test_a_point_fixed_in_the_world_is_static_while_the_ego_moves_and_yaws():
    rows = rows_for(lambda t: (3.0, 1.0))
    assert rr.track_truth(rows, 19, 7) == 'static'


def test_a_track_that_moves_20_cm_in_a_fifth_of_a_second_is_moving():
    rows = rows_for(lambda t: (3.0 + t * 1.0, 1.0))                      # 1 m/s: 0.2 m in 0.2 s
    assert rr.track_truth(rows, 19, 7) == 'moving'


def test_between_the_two_bounds_and_without_an_earlier_sample_is_unknown():
    mid = rows_for(lambda t: (3.0 + t * 0.4, 1.0))                       # 0.08 m in 0.2 s: between 0.06 and 0.10
    assert rr.track_truth(mid, 19, 7) == 'unknown'
    assert rr.track_truth(mid, 3, 7) == 'unknown'                        # nothing 0.2 s earlier
    assert rr.track_truth(rows_for(lambda t: (3.0, 1.0)), 19, 99) == 'unknown'     # an id not on the row
