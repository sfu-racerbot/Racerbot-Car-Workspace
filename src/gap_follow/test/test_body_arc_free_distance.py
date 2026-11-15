"""body_arc_free_distance: the body rectangle swept along an arc (plan
docs/superpowers/plans/2026-11-12-post-race-speed-program.md, Task 1).

Every expected value is a closed form recomputed here from the rectangle
x in [rear, front], |y| <= half_width in base_link.
"""
import math

import numpy as np
import pytest

from gap_follow import gap_logic

CAR_LENGTH = 0.58
WHEELBASE = 0.36
LASER_X = 0.26
FRONT = WHEELBASE / 2 + CAR_LENGTH / 2  # 0.47
HALF = 0.175


def scan_from_points(points, valid=None):
    """base_link (x, y) points -> (range, angle) seen from the LiDAR."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    dx = pts[:, 0] - LASER_X
    dy = pts[:, 1]
    ranges = np.hypot(dx, dy)
    angles = np.arctan2(dy, dx)
    ok = np.ones(len(pts), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    return ranges, ok, angles


def wall(y, x0=-1.0, x1=3.0):
    xs = np.arange(x0, x1, 0.01)
    return np.stack([xs, np.full_like(xs, y)], axis=1)


def free(points, curvature, half_width=HALF, **kw):
    r, v, a = scan_from_points(points)
    return gap_logic.body_arc_free_distance(
        r, v, a, curvature, half_width, CAR_LENGTH, WHEELBASE, LASER_X, **kw)


def test_wall_alongside_outside_the_body_is_free_on_a_straight():
    # closed form: a wall at |y|=0.20 > half_width 0.175 is never entered.
    assert free(wall(0.20), 0.0) == math.inf


def test_wall_inside_the_body_rectangle_is_zero():
    # closed form: |y|=0.15 <= 0.175 and x spans the body -> hit at s=0.
    assert free(wall(0.15), 0.0) == 0.0


def test_obstacle_ahead_on_a_straight_is_its_gap_to_the_bumper():
    # closed form: gap = 0.77 - front(0.47) = 0.30
    assert free([[FRONT + 0.30, 0.0]], 0.0) == pytest.approx(0.30, abs=0.021)  # one 2 cm step


def test_turning_away_from_a_flank_wall_is_free():
    # wall on the left, turning right: the body leaves it.
    assert free(wall(0.20), -1 / 1.05, max_length=0.5) == math.inf


def toward_wall_expected():
    R = 1.05
    rho = math.hypot(FRONT, R - HALF)
    phi0 = math.atan2(FRONT, R - HALF)
    theta = math.acos((R - 0.30) / rho) - phi0
    return theta * R


def test_turning_toward_a_flank_wall_hits_at_the_front_corner():
    # closed form: the front-left corner sits at radius rho from the turn
    # centre and reaches y=0.30 after theta = acos((R-0.30)/rho) - phi0.
    assert free(wall(0.30), 1 / 1.05) == pytest.approx(toward_wall_expected(), abs=0.025)  # 2.5 cm


def test_mirror_symmetry():
    left = free(wall(0.30), 1 / 1.05)
    right = free(wall(-0.30), -1 / 1.05)
    assert math.isfinite(left)
    assert right == pytest.approx(left, abs=1e-9)  # mirror image, same arithmetic


def test_invalid_beams_are_not_obstacles():
    pts = np.array([[0.2, 0.0], [0.3, 0.0], [0.4, 0.0]])
    r, _, a = scan_from_points(pts)
    r = r.copy()
    r[1] = np.nan
    r[2] = np.inf
    valid = np.array([False, False, False])
    assert gap_logic.body_arc_free_distance(
        r, valid, a, 0.0, HALF, CAR_LENGTH, WHEELBASE, LASER_X) == math.inf


def test_empty_scan_is_free():
    e = np.zeros(0)
    assert gap_logic.body_arc_free_distance(
        e, e.astype(bool), e, 0.0, HALF, CAR_LENGTH, WHEELBASE, LASER_X) == math.inf


def test_all_zero_ranges_are_ignored():
    n = 50
    assert gap_logic.body_arc_free_distance(
        np.zeros(n), np.ones(n, dtype=bool), np.linspace(-1, 1, n),
        0.0, HALF, CAR_LENGTH, WHEELBASE, LASER_X) == math.inf


def test_point_beyond_max_length_is_free():
    assert free([[FRONT + 0.60, 0.0]], 0.0, max_length=0.5) == math.inf


def test_mismatched_shapes_raise():
    r, v, a = scan_from_points(wall(0.2))
    with pytest.raises(ValueError, match='matching shapes'):
        gap_logic.body_arc_free_distance(
            r, v, np.append(a, 0.0), 0.0, HALF, CAR_LENGTH, WHEELBASE, LASER_X)


@pytest.mark.parametrize('name,value', [
    ('half_width', 0.0), ('step', 0.0), ('max_length', -1.0),
    ('curvature', float('nan'))])
def test_non_finite_or_non_positive_geometry_raises(name, value):
    kw = dict(curvature=0.0, half_width=HALF, max_length=1.0, step=0.02)
    kw[name] = value
    r, v, a = scan_from_points(wall(0.2))
    with pytest.raises(ValueError, match=name):
        gap_logic.body_arc_free_distance(
            r, v, a, kw['curvature'], kw['half_width'], CAR_LENGTH, WHEELBASE,
            LASER_X, max_length=kw['max_length'], step=kw['step'])
