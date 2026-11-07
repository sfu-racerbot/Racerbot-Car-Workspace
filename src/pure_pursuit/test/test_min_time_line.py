"""ROS-free minimum-time line tests, collected by the pure_pursuit suite."""

import numpy as np
import pytest

from pure_pursuit.min_time_line import (
    CarPhysics, cone_clearance, footprint_clearance, lap_time, offsets_of,
    optimize_min_time, speed_profile,
)
from pure_pursuit.raceline_optimizer import path_frames, resample_closed_path, signed_curvature


def stadium():
    radius, straight, spacing = 2.0, 6.0, 0.1  # m
    x = np.arange(0.0, straight, spacing)
    angle = np.linspace(-np.pi / 2, np.pi / 2, 64, endpoint=False)
    return np.vstack([
        np.column_stack([x, np.full_like(x, -radius)]),
        np.column_stack([straight + radius * np.cos(angle), radius * np.sin(angle)]),
        np.column_stack([straight - x, np.full_like(x, radius)]),
        np.column_stack([-radius * np.cos(angle), -radius * np.sin(angle)]),
    ])


def test_stadium_acceleration_and_braking():
    xy = stadium()
    p = CarPhysics(v_max=8.0, steering_rate=100.0, steering_delay=0.0, speed_delay=0.0)
    speed, _, _ = speed_profile(xy, p)
    straight, radius = 6.0, 2.0  # m
    corner_speed = np.sqrt(p.a_lat_left * radius)
    x = xy[:60, 0]
    # oracle: closed form, braking v² <= v_corner² + 2*a_brake*d. Station 61
    # is the first with both neighbours on the constructed circular arc.
    chord = 2 * radius * np.sin(np.pi / (2 * 64))
    assert np.all(speed[:60] <= np.sqrt(corner_speed**2 + 2 * p.a_brake * (straight - x + chord))
                  + 1e-9)  # m/s, rounding tolerance
    # Acceleration and braking parabolas intersect when a_accel*x=a_brake*(L-x).
    peak = np.sqrt(corner_speed**2 + 2 * p.a_accel * p.a_brake * straight
                   / (p.a_accel + p.a_brake))
    assert peak < p.v_max
    assert np.max(speed[:60]) == pytest.approx(peak, rel=0.02)  # m/s, 2%


def test_speed_delay_moves_braking_earlier():
    xy = stadium()
    options = dict(v_max=8.0, steering_rate=100.0, steering_delay=0.0)
    immediate, _, _ = speed_profile(xy, CarPhysics(speed_delay=0.0, **options))
    delayed, _, _ = speed_profile(xy, CarPhysics(speed_delay=0.1, **options))
    # oracle: invariant: a delayed response must start braking earlier; compare
    # against the same threshold on the descending half of the straight.
    threshold = 4.0  # m/s, below both straight peaks and above corner speed
    assert immediate[:60].max() > threshold and delayed[:60].max() > threshold
    start = int(np.argmax(immediate[:60]))
    assert np.flatnonzero(delayed[start:60] < threshold)[0] < np.flatnonzero(
        immediate[start:60] < threshold)[0]


def test_steering_rate_limits_step():
    xy = stadium()
    p = CarPhysics(v_max=8.0, steering_rate=0.5, steering_delay=0.0, speed_delay=0.0,
                   v_min=0.01)
    speed, kappa, seg = speed_profile(xy, p)
    slower, _, _ = speed_profile(xy, CarPhysics(v_max=8.0, steering_rate=0.25,
                                                 steering_delay=0.0, speed_delay=0.0,
                                                 v_min=0.01))
    s = np.r_[0.0, np.cumsum(seg)]
    window = 2 * np.mean(seg)  # m; documented minimum window at zero delay
    periodic_kappa = np.r_[kappa, kappa[0]]
    ahead = np.interp((s[:-1] + window) % s[-1], s, periodic_kappa)
    behind = np.interp((s[:-1] - window) % s[-1], s, periodic_kappa)
    slope = np.maximum(abs(ahead - kappa), abs(kappa - behind)) / window
    near = np.arange(57, 64)  # geometric straight-to-arc transition
    # oracle: invariant v*|dkappa/ds| <= curvature_rate, module's documented window.
    assert np.max(slope[near]) > 0.1  # 1/m²; ensure the test sees the step
    assert np.all(speed[near] * slope[near] <= p.curvature_rate * 1.05)  # 1/(m s)
    assert np.min(slower[near] - speed[near]) < -0.1  # m/s


def test_offsets_constant_normal_shift():
    angle = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    center = 3.0 * np.column_stack((np.cos(angle), np.sin(angle)))  # m
    _, normals = path_frames(center)
    shift = 0.4  # m
    # oracle: invariant: projecting the parallel circle on its normal recovers shift.
    assert np.allclose(offsets_of(center, center + shift * normals), shift,
                       atol=0.005, rtol=0.0)  # m, 5 mm


def test_car_physics_validation_and_limits():
    # oracle: spec min_time_line.CarPhysics: positive finite values, nonnegative delays.
    positive = ('a_lat_left', 'a_lat_right', 'a_accel', 'a_brake', 'v_max', 'v_min',
                'wheelbase', 'max_steering_angle', 'linkage_gain', 'steering_rate')
    delays = ('steering_delay', 'speed_delay')
    for name in positive:
        for value in (0.0, -1.0, np.nan, np.inf, -np.inf):
            with pytest.raises(ValueError, match=name):
                CarPhysics(**{name: value})
    for name in delays:
        for value in (-1.0, np.nan, np.inf, -np.inf):
            with pytest.raises(ValueError, match=name):
                CarPhysics(**{name: value})
    with pytest.raises(ValueError, match='v_min.*v_max'):
        CarPhysics(v_min=2.0, v_max=1.0)
    physics = CarPhysics(steering_delay=0.0, speed_delay=0.0, linkage_gain=0.8,
                         steering_rate=0.7, wheelbase=0.4, max_steering_angle=0.3)
    # oracle: closed form, bicycle steering geometry and small-angle rack rate.
    assert physics.curvature_limit == pytest.approx(
        np.tan(physics.linkage_gain * physics.max_steering_angle) / physics.wheelbase,
        abs=1e-12)  # 1/m
    assert physics.curvature_rate == pytest.approx(
        physics.steering_rate * physics.linkage_gain / physics.wheelbase,
        abs=1e-12)  # 1/(m s)


def test_circle_speed_and_lap_time():
    radius = 2.0  # m
    angle = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    ccw = radius * np.column_stack([np.cos(angle), np.sin(angle)])
    physics = CarPhysics(steering_delay=0.0, speed_delay=0.0)
    for xy, grip in ((ccw, physics.a_lat_left), (ccw[::-1], physics.a_lat_right)):
        speed, curvature, _ = speed_profile(xy, physics)
        # oracle: closed form, a_lat=v²/R; constant curvature has zero rate.
        expected = min(np.sqrt(grip * radius), physics.v_max)
        assert np.allclose(speed, expected, rtol=0.01, atol=0.0)  # m/s, 1%
        assert np.all(np.sign(curvature) == (1 if grip == physics.a_lat_left else -1))
        assert lap_time(xy, physics) == pytest.approx(2 * np.pi * radius / expected,
                                                     rel=0.01)  # s, 1%


def test_speed_profile_invalid_paths():
    # oracle: spec min_time_line.speed_profile validates finite paths and distinct neighbours.
    physics = CarPhysics()
    for xy in (np.empty((0, 2)), np.array([[0.0, 0.0]]), np.zeros((3, 2)),
               np.array([[0., 0.], [1., 0.], [1., np.nan], [0., 1.]])):
        with pytest.raises(ValueError, match='at least 4 finite'):
            speed_profile(xy, physics)
    with pytest.raises(ValueError, match='duplicate consecutive'):
        speed_profile([[0, 0], [1, 0], [1, 0], [0, 1]], physics)


def test_cone_head_on_and_clear_range():
    xy = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [3, -2], [0, -2]])
    distance, laser_x, step, max_range = 0.7, 0.2, 0.01, 1.5  # m
    wall_x = xy[1, 0] + laser_x + distance
    field = lambda x, y: np.maximum(wall_x - x, 0.0)
    actual = cone_clearance(field, xy, laser_x=laser_x, step_floor=step)[1]
    # oracle: closed form, head-on ray meets the perpendicular wall at distance.
    assert distance - step <= actual <= distance + 1e-9  # m
    empty = lambda x, y: np.full(np.broadcast(x, y).shape, 10.0)
    assert cone_clearance(empty, xy, max_range=max_range)[1] == max_range  # m; range cap


def test_footprint_parallel_wall():
    xy = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [3, -2], [0, -2]])
    wall_y, half_width = 0.7, 0.155  # m
    field = lambda x, y: np.maximum(wall_y - y, 0.0)
    # oracle: closed form, nearest body edge is half_width nearer to the wall.
    assert footprint_clearance(field, xy, half_width=half_width)[1] == pytest.approx(
        wall_y - half_width, abs=0.001)  # m, 1 mm


def test_speed_delay_respects_future_planned_speeds():
    # Stadium: 6 m straights and 2 m radius semicircles, counter-clockwise.
    radius, straight, spacing = 2.0, 6.0, 0.1
    x = np.arange(0.0, straight, spacing)
    angle = np.linspace(-np.pi / 2, np.pi / 2, 64, endpoint=False)
    xy = np.vstack([
        np.column_stack([x, np.full_like(x, -radius)]),
        np.column_stack([straight + radius * np.cos(angle), radius * np.sin(angle)]),
        np.column_stack([straight - x, np.full_like(x, radius)]),
        np.column_stack([-radius * np.cos(angle), -radius * np.sin(angle)]),
    ])
    physics = CarPhysics(v_max=8.0, steering_delay=0.0, speed_delay=0.1)
    speed, _, seg = speed_profile(xy, physics)
    positions = np.concatenate(([0.0], np.cumsum(seg)))
    total = positions[-1]
    # oracle: invariant, min_time_line._respond_early docstring: every station
    # must be no faster than any station within speed * delay metres ahead.
    for i in range(len(speed)):
        distance = (positions[:-1] - positions[i]) % total
        future = speed[distance <= speed[i] * physics.speed_delay]
        assert speed[i] <= future.min() + 1e-10  # m/s, floating-point tolerance


def test_cone_clearance_parallel_wall():
    xy = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [3, -2], [0, -2]])
    wall_y = 0.5  # m
    step_floor = 0.01  # m
    field = lambda x, y: np.maximum(wall_y - y, 0.0)
    actual = cone_clearance(field, xy, laser_x=0.0, step_floor=step_floor)[1]
    # oracle: closed form; the +30 degree ray intersects y=wall_y at this range.
    sine = np.sin(np.deg2rad(30.0))
    expected = wall_y / sine
    assert expected - step_floor / sine <= actual <= expected + 1e-9  # m
