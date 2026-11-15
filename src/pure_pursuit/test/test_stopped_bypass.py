"""pass_planner's stopped-car bypass (Bypass, crawl_rollout, bypass_entry), 2026-11-15. ROS-free; run by
`python3 -m pytest src/pure_pursuit/test/` and colcon test.

Oracles: the Bypass profile is linear interpolation plus the quintic return (closed forms recomputed here); the
corridor cases reuse test_pass_planner's synthetic stadium, whose first straight has walls at a known distance
(half_width - |y|), so whether a car fits beside a parked one is arithmetic; a found path is re-checked here
pose by pose with an independent body outline, not with the planner's own sampling.
"""
import math
from dataclasses import replace

import numpy as np
import pytest

from pure_pursuit import pass_planner as pp
from test_pass_planner import LINE, LIMITS, corridor, pose_behind

STOPPED_LIMITS = replace(LIMITS, wall_margin=0.06, car_margin=0.06, long_margin=0.08, steering_headroom=0.95,
                         spacing=0.05, body_grid=(17, 9))   # as pure_pursuit_node._pass_limits(stopped=True)
PARKED = pp.StalledOpponent(rear_arc=10.0, lateral=0.0, speed_bound=0.1, age=0.0)


def needed_half_width(limits=STOPPED_LIMITS, opponent=PARKED):
    """Narrowest corridor half-width with room beside a parked car centred on the line: its box half-width,
    the car margin, our width, the wall margin and build()'s sampling slack on a straight, side by side
    (closed form of the spec)."""
    _, _, half = pp.opponent_box(opponent, math.inf, limits)
    return half + limits.car_margin + limits.width + limits.wall_margin + pp.build_slack(limits, 0.0)


def outline(x, y, yaw, limits=LIMITS):
    """Our body's corners and edge midpoints at a rear-axle pose (independent of pass_planner.body_points)."""
    rear = (limits.wheelbase - limits.length) / 2
    pts = [(ax, ay) for ax in (rear, rear + limits.length / 2, rear + limits.length)
           for ay in (-limits.width / 2, 0.0, limits.width / 2)]
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([(x + c * ax - s * ay, y + s * ax + c * ay) for ax, ay in pts])


# --- Bypass profile -----------------------------------------------------------------------------------------

def test_bypass_lateral_interpolates_its_path_then_shifts_back_to_the_line():
    # On a straight-line path (d linear in s, slope 0.2) a natural cubic spline is that line exactly; the return
    # is the quintic Hermite from (0.5, slope 0.2) to (0, 0) over 2 m (closed forms).
    m = pp.Bypass(s_start=5.0, s_rel=(0.0, 1.0, 2.0), d=(0.1, 0.3, 0.5), back=2.0, side=1, target=0.5)
    s = np.array([5.0, 5.5, 7.0, 8.0, 9.0, 12.0])
    t = 0.5
    want = [0.1, 0.2, 0.5, 0.5 * (1 - 10 * t ** 3 + 15 * t ** 4 - 6 * t ** 5) + 0.2 * 2.0 * (t - 6 * t ** 3 + 8 * t ** 4 - 3 * t ** 5),
            0.0, 0.0]
    np.testing.assert_allclose(m.lateral(s, LINE), want, atol=1e-12)        # m
    assert (m.d_start, m.out, m.s_return) == (0.1, 2.0, 7.0)


def test_bypass_lateral_passes_through_every_path_point_with_continuous_curvature():
    m = pp.Bypass(0.0, (0.0, 0.5, 1.0, 1.5, 2.0), (0.0, 0.1, 0.35, 0.45, 0.45), 2.0, 1, 0.45)
    np.testing.assert_allclose(m.lateral(np.array(m.s_rel), LINE), m.d, atol=1e-12)
    s = np.linspace(0.01, 1.99, 199)
    second = np.diff(m.lateral(s, LINE), 2) / 0.01 ** 2
    assert np.max(np.abs(np.diff(second))) < 0.5      # no jump in d'' between neighbours (1/m per step)


def test_a_held_bypass_keeps_its_last_offset_for_good():
    m = pp.Bypass(5.0, (0.0, 1.0, 2.0), (0.1, 0.5, 0.4), 2.0, 1, 0.5).held()
    assert m.back == math.inf
    np.testing.assert_allclose(m.lateral(np.array([7.0, 9.0, 20.0]), LINE), [0.4, 0.4, 0.4], atol=1e-12)


def test_the_return_leaves_the_path_with_its_slope_and_arrives_flat():
    m = pp.Bypass(0.0, (0.0, 1.0, 2.0), (0.0, 0.2, 0.4), 2.0, 1, 0.4)
    eps = 1e-4
    d = lambda s: float(m.lateral(np.array([s]), LINE)[0])
    assert (d(2.0 + eps) - d(2.0 - eps)) / (2 * eps) == pytest.approx(0.2, abs=1e-3)   # slope continuous
    assert (d(4.0 - eps) - d(4.0 - 2 * eps)) / eps == pytest.approx(0.0, abs=1e-3)      # flat on arrival
    assert d(4.0) == pytest.approx(0.0, abs=1e-12)


def test_behind_its_start_the_bypass_is_its_first_offset():
    m = pp.Bypass(5.0, (0.0, 1.0), (0.2, 0.4), 1.0, 1, 0.4)
    assert m.lateral(np.array([4.0]), LINE)[0] == pytest.approx(0.2, abs=1e-12)


def test_a_manoeuvre_held_drops_only_its_shift_back():
    m = pp.Manoeuvre(1.0, 0.0, 0.5, 2.0, 6.0, 2.0, 1)
    assert m.held() == pp.Manoeuvre(1.0, 0.0, 0.5, 2.0, math.inf, 2.0, 1)


# --- crawl_rollout ------------------------------------------------------------------------------------------

def _check_path(path, walls, limits=STOPPED_LIMITS, opponent=PARKED):
    """Every rollout pose's outline clears the walls and the parked box by the margins (2 mm slack for the
    planner's denser body grid), and the last pose has its rear past the box front by 0.3 m."""
    lo, hi, half = pp.opponent_box(opponent, math.inf, limits)
    yaw = np.arctan2(np.diff(path[:, 1]), np.diff(path[:, 0]))
    for (x, y), a in zip(path[1:], yaw):
        pts = outline(x, y, a)
        assert walls(pts[:, 0], pts[:, 1]).min() >= limits.wall_margin - 0.002
        # the straight is the x axis from s=0, so arc = x and lateral = y here
        dx = np.maximum(np.maximum(lo - pts[:, 0], pts[:, 0] - hi), 0.0)
        dy = np.maximum(np.abs(pts[:, 1] - opponent.lateral) - half, 0.0)
        inside = (dx == 0) & (dy == 0)
        assert not inside.any() and np.hypot(dx, dy).min() >= limits.car_margin - 0.002
    rear = path[-1, 0] + (limits.wheelbase - limits.length) / 2
    assert rear >= hi + 0.3 - 0.11      # within one step (0.1 m) of the success line


def test_with_room_beside_it_a_crawl_path_round_the_parked_car_is_found():
    walls = corridor(needed_half_width() + 0.30)
    path = pp.crawl_rollout(LINE, pose_behind(1.5), PARKED, walls, STOPPED_LIMITS)
    assert path is not None
    assert tuple(path[0]) == pose_behind(1.5)[:2]
    _check_path(path, walls)


def test_with_no_room_beside_it_no_path_is_found():
    # One centimetre narrower than the car and both margins need, beside a car parked dead centre.
    walls = corridor(needed_half_width() - 0.01)
    assert pp.crawl_rollout(LINE, pose_behind(3.0), PARKED, walls, STOPPED_LIMITS) is None


def test_a_bigger_reachable_box_needs_more_room():
    # The moving-car bound (measured + 0.5 m/s) grows the box: the corridor that fits a parked car does not fit it.
    width = needed_half_width() + 0.30
    moving = replace(PARKED, speed_bound=4.0)
    assert needed_half_width(opponent=moving) > width
    assert pp.crawl_rollout(LINE, pose_behind(2.0), PARKED, corridor(width), STOPPED_LIMITS) is not None
    assert pp.crawl_rollout(LINE, pose_behind(2.0), moving, corridor(width), STOPPED_LIMITS) is None


def test_more_steering_headroom_than_the_limit_is_never_used():
    # A steering angle of zero only: no path round a car dead ahead, however wide the corridor.
    straight = replace(STOPPED_LIMITS, steering_headroom=0.0)
    assert pp.crawl_rollout(LINE, pose_behind(2.0), PARKED, corridor(3.0), straight) is None


def test_the_widest_clearance_path_wins():
    # Room on both sides, the left wider: the chosen path passes on the left (oracle: it maximises clearance).
    def walls(xs, ys):
        ys = np.asarray(ys, float)
        return np.minimum(1.7 - ys, needed_half_width() + 0.30 + ys)
    path = pp.crawl_rollout(LINE, pose_behind(1.5), PARKED, walls, STOPPED_LIMITS)
    assert path is not None
    k = int(np.argmin(np.abs(path[:, 0] - 10.3)))
    assert path[k, 1] > 0.0


# --- bypass_entry -------------------------------------------------------------------------------------------

def test_bypass_entry_returns_a_checked_plan_round_the_parked_car():
    walls = corridor(needed_half_width() + 0.30)
    plan, m, cap, why = pp.bypass_entry(LINE, pose_behind(1.5), 0.0, PARKED, walls, STOPPED_LIMITS, nominal=1.0)
    assert why == '' and isinstance(m, pp.Bypass)
    assert plan.mode == 'pass' and np.all(plan.wall[:20] >= STOPPED_LIMITS.wall_margin)
    assert math.isinf(pp.box_free_distance(plan, PARKED, LINE, STOPPED_LIMITS))
    assert cap >= 0.3
    assert abs(m.target) >= needed_half_width() - STOPPED_LIMITS.wall_margin - STOPPED_LIMITS.width / 2 - 0.05


def test_bypass_entry_says_why_when_there_is_no_room():
    walls = corridor(needed_half_width() - 0.01)
    plan, m, cap, why = pp.bypass_entry(LINE, pose_behind(3.0), 0.0, PARKED, walls, STOPPED_LIMITS, nominal=1.0)
    assert (plan, m, cap) == (None, None, None)
    assert why == 'bypass: no crawl path round the stopped car'
