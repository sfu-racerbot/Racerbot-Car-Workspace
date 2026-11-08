"""ROS-free planner oracles. Runner: pytest src/pure_pursuit/test/.

Rectangular track, analytic walls, and deliberately bounded synthetic opponents
keep these checks independent of any simulator collision verdict.
"""
from dataclasses import replace
import math

import numpy as np
import pytest

from pure_pursuit import pass_planner as pp, racing_math, tracking_laws


def scene():
    # 400 m rectangle, sampled densely enough to put the local scene on a straight.
    xy = np.vstack((np.column_stack((np.arange(0, 100, .15), np.zeros(667))),
                    [[100, 100], [0, 100]]))
    cumulative = racing_math.compute_cumulative_arc_length(
        racing_math.compute_segment_lengths(xy, closed=True))
    return xy, cumulative


def limits(**kwargs):
    return replace(pp.Limits(), tracking_margin=.05, opponent_uncertainty=.01,
                   opponent_speed_uncertainty=.001, opponent_lateral_accel=.000001,
                   opponent_max_speed=.01, **kwargs)


def search(**kwargs):
    xy, s = scene()
    args = dict(pose=(10., 0., 0.), speed=2., opponent=pp.Opponent(16., 0., 0.),
                wall_at=lambda x, y: 8 - abs(y), limits=limits())
    args.update(kwargs)
    return pp.plan(xy, s, **args)


def test_entry_cost_and_left_right_wall_selection():
    left = search(wall_at=lambda x, y: y + .7)
    right = search(wall_at=lambda x, y: .7 - y)
    # oracle: invariant -- mirrored half-planes must select opposite corridors.
    assert left is not None and right is not None
    assert left.side == 1 and right.side == -1
    assert left.path.xy[0, 1] == pytest.approx(0., abs=1e-9)  # m
    assert left.path.xy[-1, 1] > .65
    assert left.clearance > 0 and math.isfinite(left.cost)
    assert left.cost == pytest.approx(right.cost, abs=1e-6)  # seconds
    # oracle: closed form -- candidate cost integrates ds/v plus clearance penalty.
    travel = sum(np.linalg.norm(np.diff(left.path.xy, axis=0), axis=1)
                 / left.path.speed_profile[:-1])
    assert left.cost == pytest.approx(travel - 12 / .3 + .3 / left.clearance, abs=1e-9)


@pytest.mark.parametrize('value', [[], [[0., 0.]], [[0., 0.]] * 4,
                                [[0., 0.], [1., 0.], [math.nan, 1.], [0., 1.]]])
def test_degenerate_lines_decline(value):
    # oracle: spec -- absent, non-finite and degenerate paths cannot authorise movement.
    assert pp.plan(value, np.arange(len(value)), (0., 0., 0.), 1.,
                   pp.Opponent(3., 0., 0.), lambda x, y: 9.) is None


@pytest.mark.parametrize('kwargs', [dict(speed=math.nan), dict(speed=-1.),
                                  dict(pose=(0., math.inf, 0.)), dict(opponent=None),
                                  dict(opponent=pp.Opponent(3., 0., 0., age=-.1)),
                                  dict(wall_at=lambda x, y: math.nan),
                                  dict(wall_at=lambda x, y: 0.)])
def test_invalid_or_occupied_scene_declines(kwargs):
    # oracle: invariant -- unknown geometry never becomes a free corridor.
    assert search(**kwargs) is None


def test_whole_body_and_between_station_wall_check():
    path = pp.temporary_path(np.column_stack((np.arange(0., 3., .15), np.zeros(20))), np.ones(20))
    cfg = limits()
    # oracle: closed form -- a wall y=.20 intersects the inflated body despite free axle points.
    assert pp.wall_clearance(path, lambda x, y: .20 - y, cfg) < 0
    assert pp.wall_clearance(path, lambda x, y: 2. - abs(y), cfg) > 1.
    # Thin wall between successive axle stations, extending across the track.
    assert pp.wall_clearance(path, lambda x, y: abs(x - .075), cfg) < 0


def test_steering_grip_rate_and_physical_speed_bounds():
    x = np.arange(0., 6., .05)
    path = pp.temporary_path(np.column_stack((x, .1 * x * x)), np.full(len(x), 5.))
    cfg = limits()
    speeds = pp.feasible_speed(path, cfg)
    # oracle: physical inequalities from the brief; independently calculate each constraint.
    assert speeds is not None
    assert np.max(speeds ** 2 * abs(path.curvature)) <= 2.7 + 1e-9  # m/s²
    ds = np.linalg.norm(np.diff(path.xy, axis=0), axis=1)
    assert np.max(np.diff(speeds ** 2) / (2 * ds)) <= 4. + 1e-9
    assert np.max(-np.diff(speeds ** 2) / (2 * ds)) <= 3. + 1e-9
    rack = np.arctan(.36 * path.curvature)
    assert np.max(abs(np.diff(rack)) * np.minimum(speeds[:-1], speeds[1:]) / ds) <= 1. + 1e-9
    assert pp.feasible_speed(path, replace(cfg, steering=.001)) is None
    # A 10% grip change must have a numerical effect (not just an upper bound).
    faster = pp.feasible_speed(path, replace(cfg, lateral_accel=2.97))
    assert np.max(faster - speeds) > .05  # m/s


@pytest.mark.parametrize('speed', [0., 1.5, 4.])
def test_stopped_braking_and_weaving_opponent_tubes(speed):
    xy, cumulative = scene()
    path = pp.temporary_path(np.column_stack((np.linspace(10., 20., 80), np.ones(80))), np.full(80, 3.))
    times = np.linspace(0., 3., 80)
    cfg = limits()
    # oracle: invariant -- stopped car must remain occupied even if measured speed is positive.
    assert pp.opponent_clearance(path, times, xy, cumulative, pp.Opponent(16., 1., speed), cfg) < 0
    # A weave towards the adjacent corridor is occupied, not constant-velocity free space.
    assert pp.opponent_clearance(path, times, xy, cumulative, pp.Opponent(16., 0., speed),
                                 replace(cfg, opponent_lateral_accel=2.7)) < 0
    assert pp.opponent_clearance(path, times, xy, cumulative, pp.Opponent(60., 0., speed), cfg) > 0


def test_rejoin_is_checked_and_never_a_wrapped_temporary_path():
    clear = search(pose=(10., 1., 0.), opponent=pp.Opponent(2., 0., 0.), rejoin=True)
    blocked = search(pose=(10., 1., 0.), opponent=pp.Opponent(13., 0., 0.), rejoin=True)
    # oracle: invariant -- return crosses the line only when its swept corridor is free.
    assert clear is not None and blocked is None
    assert not clear.path.closed
    assert clear.path.xy[-1, 1] == pytest.approx(0., abs=1e-9)  # m
    assert tracking_laws.sample_at_arc(clear.path.curvature, clear.path.cumulative,
                                       1000., False) == clear.path.curvature[-1]


def test_frame_seam_and_rigid_transform():
    xy, cumulative = scene()
    a, t, n = pp.frame(xy, cumulative, -1.)
    total = cumulative[-1] + np.linalg.norm(xy[0] - xy[-1])
    b, _, _ = pp.frame(xy, cumulative, total - 1.)
    # oracle: closed form -- the closing segment is vertical at x=0, one metre from origin.
    assert a == pytest.approx([0., 1.], abs=1e-8)  # m
    assert b == pytest.approx(a, abs=1e-8)
    assert np.dot(t, n) == pytest.approx(0., abs=1e-12)
    rotation = np.array([[0., -1.], [1., 0.]])
    moved, _, _ = pp.frame(xy @ rotation.T + [5., 6.], cumulative, -1.)
    assert moved == pytest.approx(a @ rotation.T + [5., 6.], abs=1e-8)


def test_delayed_stop_rejects_too_short_horizon():
    # oracle: closed form -- at 10m/s response+braking is .45*10+100/6 > 12m.
    assert search(speed=10.) is None
    assert search(speed=2.) is not None


def test_delay_lqr_rollout_checks_real_executed_body_without_mutating_law():
    xy, cumulative = scene()
    candidate = search(opponent=pp.Opponent(60., 0., 0.))
    law = tracking_laws.DelayLqrLaw(tracking_laws.VehicleLimits(.36, .42, 1., .025),
                                   tracking_laws.DelayLqrParams())
    inp = tracking_laws.TrackingInput(10., 0., 0., 2., 0, 0.)
    # oracle: invariant -- real servo rollout succeeds in open space and fails on a wall.
    assert pp.rollout_clear(candidate, law, inp, xy, cumulative, pp.Opponent(60., 0., 0.),
                            lambda x, y: 9., limits())
    assert not pp.rollout_clear(candidate, law, inp, xy, cumulative, pp.Opponent(60., 0., 0.),
                                lambda x, y: 0., limits())
    assert len(law._history) == 0
