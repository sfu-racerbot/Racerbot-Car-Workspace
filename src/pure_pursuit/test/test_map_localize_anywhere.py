"""Localizing a parked car from one scan, anywhere on a saved map (no ROS).

Until 2026-11-07 a race had to start where map_course stopped the car: the
particle filter was seeded from start_pose.yaml. The tech lead needs to put the
car anywhere on the course, often moved to the start line. find_pose_anywhere
answers "where is this scan on the map?" with no prior, and refuses when the
answer isn't unique (a symmetric course) or doesn't fit.

Runner: plain pytest, no ROS needed:
    python3 -m pytest src/pure_pursuit/test/test_map_localize_anywhere.py -v
"""

import math
from pathlib import Path

import numpy as np
import pytest

from pure_pursuit import map_localize
from pure_pursuit.occupancy_map import OccupancyMap

DATA = Path(__file__).parent / 'data'
COURSE = DATA / 'course_20261106a'
# oracle: recorded measurement. loc.py's located poses of run 20261106-164401
# (car 2) on its own day's rebuilt map, every chosen scan at fit >= 0.994.
SCANS = np.load(COURSE / 'scans_20261106-164401.npz')
POSITION_TOL_M = 0.15     # the particle filter spreads 0.5 m round a seed
HEADING_TOL_RAD = math.radians(3.0)


def _field(map_dir):
    return map_localize.WallField.from_occupancy(OccupancyMap.from_yaml(str(Path(map_dir) / 'map.yaml')))


@pytest.fixture(scope='module')
def field():
    return _field(COURSE)


def _localize(field, ranges):
    return map_localize.localize_scan(field, ranges, float(SCANS['angle_min']), float(SCANS['angle_increment']))


def _yaw_error(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


@pytest.mark.parametrize('index', range(len(SCANS['ranges'])))
def test_a_real_scan_is_found_where_loc_put_it(field, index):
    fix = _localize(field, SCANS['ranges'][index])
    truth = SCANS['poses'][index]
    assert fix.ok, fix.reason
    assert math.hypot(fix.pose[0] - truth[0], fix.pose[1] - truth[1]) < POSITION_TOL_M
    assert _yaw_error(fix.pose[2], truth[2]) < HEADING_TOL_RAD
    # Not a near-tie with some other place on the course.
    assert fix.runner_up_residual >= map_localize.AMBIGUITY_RATIO * fix.residual


def test_a_box_in_front_of_the_car_does_not_move_the_fix(field):
    """Review Focus 4: another car parked just ahead."""
    ranges = SCANS['ranges'][2].astype(float).copy()
    angles = float(SCANS['angle_min']) + float(SCANS['angle_increment']) * np.arange(len(ranges))
    ahead = np.abs(angles) < math.radians(9.0)   # a 0.3 m wide car at 1 m
    ranges[ahead] = 1.0
    fix = _localize(field, ranges)
    truth = SCANS['poses'][2]
    assert fix.ok, fix.reason
    assert math.hypot(fix.pose[0] - truth[0], fix.pose[1] - truth[1]) < POSITION_TOL_M


def test_a_scan_with_a_third_of_its_beams_missing_still_localizes(field):
    ranges = SCANS['ranges'][4].astype(float).copy()
    ranges[::3] = np.nan
    ranges[1::7] = np.inf
    fix = _localize(field, ranges)
    truth = SCANS['poses'][4]
    assert fix.ok, fix.reason
    assert math.hypot(fix.pose[0] - truth[0], fix.pose[1] - truth[1]) < POSITION_TOL_M


@pytest.fixture(scope='module')
def other_course():
    # Car 2's 2026-10-30 course: another 1.5 m wide SFU corridor loop.
    return _field(Path(__file__).resolve().parents[3] / 'data/rb2/racerbot_auto/20261030-185036-fixed')


@pytest.mark.parametrize('index', [0, 1, 4])
def test_a_scan_from_another_course_is_refused(other_course, index):
    """Scans 1 and 4 fit that course at 0.93 somewhere along a straight, above MIN_GLOBAL_FIT, so the
    refusal has to come from beams passing through its walls (measured 8.9-64%, 2026-11-07)."""
    fix = _localize(other_course, SCANS['ranges'][index])
    assert not fix.ok
    assert 'does not match this map' in fix.reason


def test_a_scan_ending_in_open_floor_is_refused(field):
    """Every beam ends 40% short of the walls (boxes all round, or a changed course): nothing crosses a
    wall, so only the fit can say this view is not this map."""
    fix = _localize(field, SCANS['ranges'][2].astype(float) * 0.6)
    assert not fix.ok
    assert 'does not match this map' in fix.reason
    # The fit check on its own, without the see-through check that also catches it above.
    short = map_localize.scan_points(SCANS['ranges'][2].astype(float) * 0.6, float(SCANS['angle_min']),
                                     float(SCANS['angle_increment']))
    alone = map_localize.find_pose_anywhere(field, *short)
    assert not alone.ok
    assert alone.fit < map_localize.MIN_GLOBAL_FIT
    assert f'is under {map_localize.MIN_GLOBAL_FIT:.2f}' in alone.reason


def test_the_shortlist_keeps_a_distant_equal_when_one_place_crowds_the_top():
    # oracle: closed form. Ten cells within 0.1 m of (0, 0) outscore everything; (5, 0) is next.
    xs = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.0, 0.0, 0.0, 0.0, 5.0])
    ys = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.04, 0.06, 0.08, 0.0])
    scores = np.array([[1.0 - 0.01 * i] for i in range(10)] + [[0.5]])
    picked = map_localize.spread_candidates(scores, xs, ys, np.array([0.0]), 2)
    assert picked == [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)]


@pytest.mark.parametrize('ranges', [
    np.zeros(0), np.full(1081, np.inf), np.full(1081, np.nan), np.zeros(1081)],
    ids=['empty', 'all-inf', 'all-nan', 'all-zero'])
def test_an_unusable_scan_is_refused_not_raised(field, ranges):
    fix = _localize(field, ranges)
    assert not fix.ok
    assert 'beams' in fix.reason


# --- A point-symmetric course: two places fit any scan equally ------------

def _oval_field(res=0.05):
    """A ring between two ellipses, symmetric under a half turn about the origin."""
    xs = np.arange(-6.0, 6.0, res) + res / 2
    ys = np.arange(-4.0, 4.0, res) + res / 2
    gx, gy = np.meshgrid(xs, ys)
    outer = (gx / 5.0) ** 2 + (gy / 3.0) ** 2 <= 1.0
    inner = (gx / 3.5) ** 2 + (gy / 1.5) ** 2 < 1.0
    free = outer & ~inner
    walls = ~free
    return map_localize.WallField(walls, free, res, -6.0, -4.0), free, res


def _cast(free, res, pose, beams=1081, fov=math.radians(270.0), laser_x=map_localize.LASER_X):
    """Ranges seen from the LiDAR of a base_link at `pose`, by marching each beam through the grid."""
    x0 = pose[0] + laser_x * math.cos(pose[2])
    y0 = pose[1] + laser_x * math.sin(pose[2])
    angles = -fov / 2 + fov * np.arange(beams) / (beams - 1)
    steps = np.arange(0.0, 10.0, res / 2)
    px = x0 + np.cos(pose[2] + angles)[:, None] * steps
    py = y0 + np.sin(pose[2] + angles)[:, None] * steps
    cols = ((px + 6.0) / res).astype(int).clip(0, free.shape[1] - 1)
    rows = ((py + 4.0) / res).astype(int).clip(0, free.shape[0] - 1)
    blocked = ~free[rows, cols]
    first = np.where(blocked.any(axis=1), blocked.argmax(axis=1), len(steps) - 1)
    return steps[first], -fov / 2, fov / (beams - 1)


def test_a_symmetric_course_is_refused_as_ambiguous():
    """Review Focus 3: the oval looks the same from (x, y, yaw) and (-x, -y, yaw + pi)."""
    field, free, res = _oval_field()
    ranges, angle_min, inc = _cast(free, res, (4.25, 0.0, math.pi / 2))
    fix = map_localize.find_pose_anywhere(field, *map_localize.scan_points(ranges, angle_min, inc))
    assert not fix.ok
    assert 'ambiguous' in fix.reason
    # Both candidates are the two symmetric poses (oracle: closed form, half-turn symmetry).
    for pose in (fix.pose, fix.runner_up_pose):
        assert min(math.hypot(pose[0] - 4.25, pose[1]), math.hypot(pose[0] + 4.25, pose[1])) < 0.2


def test_the_symmetric_twin_survives_a_short_shortlist():
    """Neighbouring cells and headings of one place score almost alike. Without spreading the shortlist
    over distinct places, four candidates can all be the same place and the twin is never compared."""
    field, free, res = _oval_field()
    ranges, angle_min, inc = _cast(free, res, (4.25, 0.0, math.pi / 2))
    fix = map_localize.find_pose_anywhere(field, *map_localize.scan_points(ranges, angle_min, inc), refine=4)
    assert not fix.ok
    assert 'ambiguous' in fix.reason


# --- Facing the racing direction ------------------------------------------

def _square_line():
    """Counter-clockwise square, 4 m a side."""
    side = np.linspace(0.0, 4.0, 41)[:-1]
    return np.vstack([np.column_stack([side, np.zeros_like(side)]),
                      np.column_stack([np.full_like(side, 4.0), side]),
                      np.column_stack([4.0 - side, np.full_like(side, 4.0)]),
                      np.column_stack([np.zeros_like(side), 4.0 - side])])


@pytest.mark.parametrize('yaw, expected_deg', [(0.0, 0.0), (math.pi / 2, 90.0), (math.pi, 180.0),
                                               (-math.pi / 4, 45.0)])
def test_heading_against_the_line_on_the_bottom_side(yaw, expected_deg):
    # oracle: closed form; the line runs +x along y = 0.
    error = map_localize.line_heading_error(_square_line(), (2.0, 0.1, yaw))
    assert math.degrees(error) == pytest.approx(expected_deg, abs=1.0)


def test_heading_against_the_line_uses_the_nearest_side():
    # On the right side the line runs +y; a car facing +y is aligned.
    assert math.degrees(map_localize.line_heading_error(_square_line(), (4.1, 2.0, math.pi / 2))) < 1.0
