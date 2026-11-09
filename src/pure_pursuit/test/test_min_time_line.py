"""ROS-free minimum-time line tests, collected by the pure_pursuit suite."""
import math

from dataclasses import replace

import numpy as np
import pytest

from pure_pursuit.min_time_line import (
    CarPhysics, body_floor_at, cone_clearance, footprint_clearance, lap_time, offsets_of,
    optimize_min_time, speed_profile,
)
from pure_pursuit.raceline_optimizer import path_frames, resample_closed_path, signed_curvature
from pure_pursuit import min_time_line


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


def test_optimize_stadium_corridor():
    center = resample_closed_path(stadium(), 0.25, smooth=False)
    width = 1.0  # m, on each side of the radius-2 m stadium
    widths = np.full(len(center), width)

    def field(x, y):
        # Closed form: distance to the stadium centerline, then to either wall.
        radial = np.hypot(np.maximum(np.abs(x - 3.0) - 3.0, 0.0), y)
        return np.maximum(width - np.abs(radial - 2.0), 0.0)

    physics = CarPhysics(steering_delay=0.0, speed_delay=0.0)
    emergency, margin, body_floor = 0.35, 0.10, 0.08  # m
    result = optimize_min_time(center, widths, widths, physics, field,
                               emergency_stop_distance=emergency, cone_margin=margin,
                               body_floor=body_floor, control_spacing=2.0, max_iter=30)
    # oracle: invariants: optimize from center, improve time and enforce physical margins.
    assert result.ok, result.reason
    assert result.lap_s <= result.start_lap_s * 0.99  # s; at least 1% improvement
    assert np.all(cone_clearance(field, result.xy) >= emergency + margin)  # m
    assert np.all(footprint_clearance(field, result.xy) >= body_floor)  # m
    assert np.max(np.abs(signed_curvature(result.xy))) <= physics.curvature_limit + 0.02  # 1/m


def test_optimize_rejects_bad_widths():
    center = resample_closed_path(stadium(), 0.25, smooth=False)
    physics = CarPhysics()
    field = lambda x, y: np.ones(np.broadcast(x, y).shape)
    half_width, wall_margin = 0.155, 0.10  # m
    width = half_width + wall_margin - 0.01  # m, less than required on each side
    widths = np.full(len(center), width)
    result = optimize_min_time(center, widths, widths, physics, field,
                               half_width=half_width, wall_margin=wall_margin)
    # oracle: spec min_time_line.optimize_min_time rejects a corridor narrower than the car plus margins.
    assert not result.ok
    assert f'{2 * width:.2f} m wide' in result.reason
    with pytest.raises(ValueError, match='width_left and width_right'):
        optimize_min_time(center, widths[:-1], widths, physics, field)
    with pytest.raises(ValueError, match='width_left and width_right'):
        optimize_min_time(center, widths, widths[:-1], physics, field)


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
        abs=1e-12, rel=0.0)  # 1/m
    assert physics.curvature_rate == pytest.approx(
        physics.steering_rate * physics.linkage_gain / physics.wheelbase,
        abs=1e-12, rel=0.0)  # 1/(m s)


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
        wall_y - half_width, abs=0.001, rel=0.0)  # m, 1 mm


def test_speed_delay_respects_future_planned_speeds():
    xy = stadium()
    physics = CarPhysics(v_max=8.0, steering_delay=0.0, speed_delay=0.1)
    speed, _, seg = speed_profile(xy, physics)
    plan, _, _ = speed_profile(xy, replace(physics, speed_delay=0.0))
    positions = np.concatenate(([0.0], np.cumsum(seg)))
    total = positions[-1]
    # oracle: spec min_time_line.speed_profile, delay shifts the unshifted plan
    # by its own response-distance window, including wrap-around and v_min.
    for i in range(len(speed)):
        distance = (positions[:-1] - positions[i]) % total
        future = plan[distance <= plan[i] * physics.speed_delay]
        assert speed[i] == pytest.approx(max(physics.v_min, future.min()),
                                         abs=1e-9, rel=0.0)  # m/s


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


# --- body_floor_at (2026-11-07): the clearance floor rises near the steering limit -------------

def _circle(radius, n=400):
    ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])


def test_body_floor_is_constant_without_a_tight_floor():
    xy = stadium()
    for tight in (None, 0.22):
        floors = body_floor_at(xy, 0.22, tight, 1.0)
        assert floors.shape == (len(xy),)
        assert np.all(floors == 0.22)


@pytest.mark.parametrize('ratio,weight', [(0.4, 0.0), (0.5, 0.0), (0.7, 0.5), (0.9, 1.0), (1.2, 1.0)])
def test_body_floor_on_a_circle_follows_the_curvature_ratio(ratio, weight):
    # Closed form: weight = clip((k / limit - 0.5) / 0.4, 0, 1) with k = 1 / R on a circle (constant
    # curvature, so the corner window changes nothing). floor = 0.22 + 0.08 * weight.
    radius = 2.0
    limit = (1.0 / radius) / ratio
    floors = body_floor_at(_circle(radius), 0.22, 0.30, limit)
    assert floors == pytest.approx(np.full(len(floors), 0.22 + 0.08 * weight), abs=2e-3)  # m


def test_body_floor_covers_corner_entry_and_exit_but_not_mid_straight():
    # Stadium: 2 m semicircles (k = 0.5/m) and 6 m straights at 0.1 m spacing. With the limit at
    # 0.5 / 0.9 the corners are fully tight. Straight stations within window_m (0.6 m) of a corner
    # get the tight floor; the middle of a straight (3 m from either end) keeps the base floor.
    xy = stadium()
    floors = body_floor_at(xy, 0.22, 0.30, 0.5 / 0.9, window_m=0.6)
    straight = np.arange(0.0, 6.0, 0.1)
    n = len(straight)
    assert floors[n // 2] == pytest.approx(0.22, abs=1e-9)          # mid first straight
    assert floors[n - 3] == pytest.approx(0.30, abs=1e-9)          # 0.3 m before the first corner
    assert floors[n + 32] == pytest.approx(0.30, abs=1e-9)         # middle of the first corner
    assert floors[n - 15] == pytest.approx(0.22, abs=1e-9)         # 1.5 m before it


def test_body_floor_refuses_bad_inputs():
    with pytest.raises(ValueError, match='body_floor_tight'):
        body_floor_at(stadium(), 0.22, float('nan'), 1.0)
    with pytest.raises(ValueError, match='curvature_limit'):
        body_floor_at(stadium(), 0.22, 0.30, 0.0)


# --- _respond_early: braking points move earlier by the speed loop's response distance ---

def _respond_early_reference(speed, s, delay):
    """The definition, one station at a time: out[i] = min of the plan over [s_i, s_i + v_i * delay]."""
    total = s[-1]
    n = len(speed)
    pos = np.concatenate([s[:-1], s[:-1] + total])
    plan = np.concatenate([speed, speed])
    out = np.array(speed, dtype=float)
    for i in range(n):
        j = int(np.searchsorted(pos, pos[i] + speed[i] * delay, side='right'))
        out[i] = plan[i:max(j, i + 1)].min()
    return out


@pytest.mark.parametrize('seed', range(6))
def test_respond_early_matches_its_station_by_station_definition(seed):
    # oracle: the reference definition above (min over each station's response window, windows
    # wrapping round the closed lap), on random profiles, spacings and delays.
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 400))
    seg = rng.uniform(0.02, 0.5, n)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    speed = rng.uniform(0.3, 6.0, n)
    delay = float(rng.uniform(0.0, 1.5))
    got = min_time_line._respond_early(speed, s, delay)
    assert np.array_equal(got, _respond_early_reference(speed, s, delay))


def test_respond_early_moves_a_braking_point_back_by_speed_times_delay():
    # oracle: closed form, in exactly representable numbers. 0.125 m stations, 4 m/s then 2 m/s from
    # s = 5.0 m; delay 0.25 s, so a 4 m/s station's window is [s, s + 1.0] m, endpoint included (spec:
    # "anywhere within the next speed * delay metres"). Stations from s = 4.0 on reach the drop and
    # take 2 m/s; s = 3.875 reaches only 4.875 and keeps 4 m/s.
    seg = np.full(80, 0.125)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    station = s[:-1]
    speed = np.where(station < 5.0, 4.0, 2.0)
    out = min_time_line._respond_early(speed, s, 0.25)
    assert out[station == 3.875][0] == 4.0
    assert out[station == 4.0][0] == 2.0
    assert np.all(out[station < 4.0] == 4.0)
    assert np.all(out[station >= 4.0] == 2.0)
    assert np.array_equal(min_time_line._respond_early(speed, s, 0.0), speed)


# --- speed_floor_extra: body clearance that grows with planned speed ---

def _circle(radius, n=400):
    theta = np.arange(n) * (2 * np.pi / n)
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))


def test_speed_floor_extra_on_a_circle_is_gain_times_the_excess_corner_speed():
    # oracle: closed form. A circle has one curvature, so the planned speed is sqrt(a_lat * r) everywhere
    # (below v_max, no curvature change, no accel/brake zone): r = 4 m, 3.2 m/s^2 -> 3.5777 m/s, and the
    # extra room is 0.1 * (3.5777 - 3.0) = 0.0578 m at every station.
    phys = CarPhysics(a_lat_left=3.2, a_lat_right=3.2, v_max=7.0, max_steering_angle=0.38)
    extra = min_time_line.speed_floor_extra(_circle(4.0), phys, 0.1, 3.0)
    assert extra == pytest.approx(np.full(400, 0.1 * (math.sqrt(3.2 * 4.0) - 3.0)), abs=1e-3)  # m
    # above the corner speed: no extra room
    assert np.all(min_time_line.speed_floor_extra(_circle(4.0), phys, 0.1, 3.6) == 0.0)


def test_speed_floor_extra_with_zero_gain_is_zero_and_bad_gains_are_refused():
    # oracle: spec, gain 0 leaves the floor unchanged; a negative or non-finite gain is refused.
    phys = CarPhysics()
    assert np.all(min_time_line.speed_floor_extra(_circle(4.0), phys, 0.0, 3.0) == 0.0)
    for gain in (-0.01, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='speed floor gain must be finite'):
            min_time_line.speed_floor_extra(_circle(4.0), phys, gain, 3.0)
