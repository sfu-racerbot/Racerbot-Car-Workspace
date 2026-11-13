"""The pure helpers of gap_stalls.py and follow_cost.py (stall grouping, course position, follow episodes).

The bag-reading and replay parts are exercised by running the tools on the run's recording (results files); these
tests hold the arithmetic they rest on to closed forms. Run by the tool-suite command in CLAUDE.md.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))      # tools/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'gap_follow'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'f1tenth_sim'))
from run_analysis import follow_cost, gap_stalls


def test_groups_finds_runs_at_least_the_minimum_length():
    # oracle: closed form, ticks every 0.025 s; True at ticks 10..29 (a 0.475 s span) and 40..60 (0.5 s).
    t = np.arange(100) * 0.025
    mask = np.zeros(100, bool)
    mask[10:30] = True
    mask[40:61] = True
    runs = gap_stalls.groups(mask, t, 0.5)
    assert runs == [(40, 60)]                       # the 0.475 s run is under 0.5 s
    assert gap_stalls.groups(mask, t, 0.4) == [(10, 29), (40, 60)]


def test_groups_bridges_a_gap_shorter_than_the_allowance_but_not_a_longer_one():
    t = np.arange(40) * 0.025
    mask = np.ones(40, bool)
    mask[10:12] = False                              # a 0.075 s hole between samples at 0.225 and 0.3
    assert gap_stalls.groups(mask, t, 0.0, gap=0.06) == [(0, 9), (12, 39)]     # 0.075 s gap > 0.06 s: split
    assert gap_stalls.groups(mask, t, 0.0, gap=0.08) == [(0, 39)]


def test_groups_empty_and_all_false_give_nothing():
    assert gap_stalls.groups(np.zeros(5, bool), np.arange(5) * 0.025, 0.0) == []
    assert gap_stalls.groups(np.zeros(0, bool), np.zeros(0), 0.0) == []


def test_line_projector_places_a_point_by_arc_length_from_the_first_point(tmp_path):
    # oracle: closed form. A 4 x 2 m rectangle sampled every 1 m, first point (0, 0), counter-clockwise:
    # perimeter 12 m; (3, 0) is 3 m along, (4, 1) is 5 m along (3 + 1 + 1), (2, 2) is 8 m along.
    pts = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (3, 2), (2, 2), (1, 2), (0, 2), (0, 1)]
    path = tmp_path / 'line.csv'
    path.write_text('x,y,speed\n' + '\n'.join(f'{x},{y},3.0' for x, y in pts))
    project, length = gap_stalls.line_projector(path)
    assert length == pytest.approx(12.0, abs=1e-9)                  # m
    assert project(3.0, 0.0)[0] == pytest.approx(3.0, abs=1e-9)
    assert project(4.0, 1.0)[0] == pytest.approx(5.0, abs=1e-9)
    s, off = project(2.1, 2.2)
    assert s == pytest.approx(8.0, abs=1e-9) and off == pytest.approx(math.hypot(0.1, 0.2), abs=1e-9)


def _row(t, follow, lowest, stopped=False, obj=None, x=0.0, y=0.0, yaw=0.0):
    return dict(t=t, follow_cap_tel=follow, next_lowest=lowest, stopped=stopped, obj=obj, x=x, y=y, yaw=yaw)


def test_follow_episodes_count_only_ticks_where_the_follow_cap_is_below_the_next_lowest():
    # oracle: closed form. 8 ticks, follow cap 0.5 below a next-lowest 3.0 on ticks 2-5 (4 ticks = 0.1 s), a gap of
    # 0.2 s, then one more tick: two episodes. Object 2 m dead ahead of a car at (10, 5) heading +y: map (10, 7).
    obj = dict(bx=2.0, by=0.0, vx=0.0, vy=0.0, cap=0.5, track=7)
    rows = [_row(i * 0.025, math.inf, 3.0) for i in range(2)]
    rows += [_row((2 + i) * 0.025, 0.5, 3.0, obj=obj, x=10.0, y=5.0, yaw=math.pi / 2) for i in range(4)]
    rows += [_row((6 + i) * 0.025, math.inf, 3.0) for i in range(2)]
    rows += [_row(0.5, 0.5, 2.0, obj=obj, x=10.0, y=5.0, yaw=math.pi / 2)]
    eps = follow_cost.episodes_of(rows, 0.0, lambda x, y: (1.0, 0.0), lambda x, y: 0.05)
    assert len(eps) == 2
    first = eps[0]
    assert first['ticks'] == 4 and first['duration'] == pytest.approx(0.1, abs=1e-12)           # s
    assert first['given_up'] == pytest.approx(2.5, abs=1e-12)                                    # m/s
    assert first['metres_brief'] == pytest.approx(2.5 * 4 * follow_cost.DT, abs=1e-12)          # m
    assert (first['obj_x'], first['obj_y']) == pytest.approx((10.0, 7.0), abs=1e-9)             # m, map frame
    assert first['kind'] == 'wall/static' and first['tracks'] == [7]


def test_follow_episode_classification_uses_the_wall_distance_and_the_moved_distance():
    obj = dict(bx=2.0, by=0.0, vx=0.0, vy=0.0, cap=0.5, track=3)
    far = [_row(i * 0.025, 0.5, 3.0, obj=obj, x=float(i), y=0.0) for i in range(5)]      # the car drives 4 m
    eps = follow_cost.episodes_of(far, 0.0, lambda x, y: (0.0, 0.0), lambda x, y: 0.5)
    assert eps[0]['kind'].startswith('free') and eps[0]['moved'] == pytest.approx(4.0, abs=1e-9)   # m: the object moved with it
    edge = follow_cost.episodes_of(far, 0.0, lambda x, y: (0.0, 0.0), lambda x, y: follow_cost.WALL_DISTANCE)
    assert edge[0]['kind'] == 'wall/static'                                    # exactly 0.15 m counts as the wall


def test_a_stopped_tick_is_not_a_follow_tick():
    rows = [_row(i * 0.025, 0.5, 3.0, stopped=True) for i in range(4)]
    assert follow_cost.episodes_of(rows, 0.0, lambda x, y: (0.0, 0.0), lambda x, y: 1.0) == []
