"""ROS-free minimum-time line tests, collected by the pure_pursuit suite."""
import math
from pathlib import Path

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


# ============================================================================
# Combined grip (friction ellipse) in the speed profile (2026-11-12)
# ============================================================================
#
# Finding (rb2 20261111-152132 launch): the profile bounded lateral and longitudinal limits independently,
# a box, so it planned 3.0 braking together with 3.2 lateral = 4.4 m/s^2 on tyres that deliver about 3.

RACE_DAY_LINE = Path(__file__).parent / 'data' / 'course_20261111' / 'raceline_raw.csv'
ELLIPSE_SLACK = 0.05   # discretisation of a station step (the speed is taken at one end of it)


def race_day_line():
    from pure_pursuit.racing_math import load_xy_csv
    return resample_closed_path(load_xy_csv(RACE_DAY_LINE), 0.15)       # as the builder's min_time candidate is


def long_stadium(straight=40.0, radius=2.0, spacing=0.1):
    """Two straights of ``straight`` m joined by two constant-radius semicircles, counter-clockwise."""
    x = np.arange(0.0, straight, spacing)
    count = int(round(math.pi * radius / spacing))
    angle = np.linspace(-np.pi / 2, np.pi / 2, count, endpoint=False)
    return np.vstack([
        np.column_stack([x, np.full_like(x, -radius)]),
        np.column_stack([straight + radius * np.cos(angle), radius * np.sin(angle)]),
        np.column_stack([straight - x, np.full_like(x, radius)]),
        np.column_stack([-radius * np.cos(angle), -radius * np.sin(angle)]),
    ])


def ellipse_use(xy, physics):
    """(a_long / a_limit)^2 + (lateral / grip)^2 for each station step, and a mask of the unpinned ones."""
    speed, kappa, seg = speed_profile(xy, physics)
    grip = np.where(kappa >= 0.0, physics.a_lat_left, physics.a_lat_right)
    w = speed ** 2
    a_long = (np.roll(w, -1) - w) / (2.0 * seg)
    # The cornering that competes with a step is the cornering at the station the limit is read at: the one
    # being left when accelerating, the one being entered when braking (braking into a corner is limited by
    # the corner it reaches, not the straight it leaves). Measured at the left station for both, a braking
    # step into a corner on the raw race-day line reads up to 1.72.
    lateral = np.where(a_long > 0.0, w * np.abs(kappa) / grip,
                       np.roll(w, -1) * np.abs(np.roll(kappa, -1)) / np.roll(grip, -1))
    limit = np.where(a_long > 0.0, physics.a_accel, physics.a_brake)
    use = (a_long / limit) ** 2 + lateral ** 2
    pinned = np.zeros(len(speed), dtype=bool)
    for edge in (physics.v_min, physics.v_max):
        pinned |= (np.abs(speed - edge) < 1e-9) | (np.abs(np.roll(speed, -1) - edge) < 1e-9)
    return use, ~pinned


def stadium_physics(c):
    return CarPhysics(v_max=8.0, steering_rate=100.0, steering_delay=0.0, speed_delay=0.0, grip_coupling=c)


def race_physics(c):
    return CarPhysics(steering_delay=0.0, speed_delay=0.0, grip_coupling=c)


def test_coupling_zero_is_exactly_todays_profile():
    # Oracle: the unchanged closed-form helper, kept as _accel_brake_box, called directly.
    for xy, physics in ((stadium(), CarPhysics(v_max=8.0)), (race_day_line(), CarPhysics())):
        kappa = signed_curvature(xy)
        seg = min_time_line._segments(xy)
        grip = np.where(kappa >= 0.0, physics.a_lat_left, physics.a_lat_right)
        cap = np.minimum(np.sqrt(grip / np.maximum(np.abs(kappa), 1e-9)), physics.v_max)
        assert physics.grip_coupling == 0.0
        assert np.array_equal(min_time_line._accel_brake(cap, seg, physics, kappa, grip),
                              min_time_line._accel_brake_box(cap, seg, physics))
        # and through the whole profile: the same numbers as with the keyword never given
        assert np.array_equal(speed_profile(xy, physics)[0], speed_profile(xy, replace(physics, grip_coupling=0.0))[0])


@pytest.mark.parametrize('name,xy,physics', [('stadium', stadium(), stadium_physics(1.0)),
                                             ('race-day', race_day_line(), race_physics(1.0))])
def test_every_station_stays_inside_the_ellipse(name, xy, physics):
    # Oracle: the invariant itself, recomputed here; 5% over 1 for the discretisation of a station step.
    use, free = ellipse_use(xy, physics)
    assert free.sum() > 0.4 * len(xy), 'most stations must be unpinned for this to test anything'
    assert use[free].max() <= 1.0 + ELLIPSE_SLACK, (name, float(use[free].max()))


def test_box_profile_violates_the_ellipse_on_a_corner_entry():
    # The flaw, and proof the test above can fail: without coupling some station asks > 1.3 of the tyres.
    use, free = ellipse_use(race_day_line(), race_physics(0.0))
    assert use[free].max() > 1.3, float(use[free].max())


@pytest.mark.parametrize('xy,make', [(stadium(), stadium_physics), (race_day_line(), race_physics)])
def test_coupling_never_makes_the_lap_faster(xy, make):
    laps = [lap_time(xy, make(c)) for c in (0.0, 0.5, 1.0)]
    assert laps[2] >= laps[1] - 1e-9 >= laps[0] - 2e-9, laps       # s
    # and it is a real effect at every coupling: half coupling costs time, full costs more than half
    assert laps[1] - laps[0] > 1e-3 and laps[2] - laps[1] > 1e-3, laps


def test_straight_and_steady_corner_are_unchanged_but_the_exit_comes_later():
    xy = long_stadium()
    box, full = stadium_physics(0.0), stadium_physics(1.0)
    v0, kappa, seg = speed_profile(xy, box)
    v1, _, _ = speed_profile(xy, full)
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    straight, radius = 40.0, 2.0
    stop = box.v_max ** 2 / (2.0 * box.a_brake) + 1.0              # braking distance from v_max, plus 1 m
    far = (s > stop) & (s < straight - stop)                        # the first straight, far from both ends
    assert far.sum() > 100
    np.testing.assert_allclose(v1[far], v0[far], rtol=0.0, atol=1e-6)      # m/s
    corner = np.sqrt(box.a_lat_left * radius)
    arc = np.flatnonzero((s > straight + 0.3 * math.pi * radius) & (s < straight + 0.7 * math.pi * radius))
    assert len(arc) > 20
    np.testing.assert_allclose(v1[arc], v0[arc], rtol=0.0, atol=1e-6)      # m/s, the steady corner
    assert v0[arc] == pytest.approx(corner, abs=0.05)
    # Exit: the first station after the arc where speed passes corner + 0.5 m/s.
    after = np.flatnonzero(s > straight + math.pi * radius)

    def exit_s(v):
        return s[after[np.flatnonzero(v[after] > corner + 0.5)[0]]]
    assert exit_s(v1) > exit_s(v0)


def test_grip_coupling_is_validated():
    for bad in (-0.1, 1.1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='grip_coupling'):
            CarPhysics(grip_coupling=bad)
    assert CarPhysics(grip_coupling=1.0).grip_coupling == 1.0 and CarPhysics().grip_coupling == 0.0


# --- the fixed-point iteration (2026-11-13) --------------------------------------------------

def _sequential_reference(cap, seg, physics, kappa, grip):
    """The original sequential solution of the same problem, kept as the reference for the fixed point:
    forward / backward / forward passes over three laps with state-dependent limits (slow: pure python)."""
    n = len(cap)
    c = physics.grip_coupling
    a_acc, a_brk = physics.a_accel, physics.a_brake
    w = np.tile(np.asarray(cap, dtype=float) ** 2, 3).tolist()
    ds = np.tile(seg, 3).tolist()
    k = np.tile(np.abs(kappa), 3).tolist()
    g = np.tile(grip, 3).tolist()
    last = 3 * n - 1

    def eff(a, wj, j):
        use = min(1.0, (wj * k[j] / g[j]) ** 2)
        return a * math.sqrt(max(0.0, 1.0 - c * use))

    for _ in range(6):
        before = list(w)
        for j in range(1, last + 1):
            w[j] = min(w[j], w[j - 1] + 2.0 * eff(a_acc, w[j - 1], j - 1) * ds[j - 1])
        for j in range(last - 1, -1, -1):
            w[j] = min(w[j], w[j + 1] + 2.0 * eff(a_brk, w[j + 1], j + 1) * ds[j])
        for j in range(1, last + 1):
            w[j] = min(w[j], w[j - 1] + 2.0 * eff(a_acc, w[j - 1], j - 1) * ds[j - 1])
        if max(abs(x - y) for x, y in zip(w, before)) < 1e-9:
            break
    return np.sqrt(np.array(w[n:2 * n]))


PROFILED_LINE = Path(__file__).parent / 'data' / 'course_20261111' / 'raceline_profiled_orig_guard.csv'


def profiled_line():
    """The race-day line as the builder produced it with the original guard (est. 9.404 s), x, y only."""
    from pure_pursuit.racing_math import load_profiled_csv
    return load_profiled_csv(PROFILED_LINE)[0]


def profiled_physics(c):
    return CarPhysics(a_lat_left=3.2, a_lat_right=3.2, steering_delay=0.0, speed_delay=0.0, grip_coupling=c)


FIXTURES = [('stadium', stadium(), stadium_physics), ('race-day profiled', profiled_line(), profiled_physics)]


_REAL_COUPLED = min_time_line._accel_brake_coupled


def _with_rounds(monkeypatch, rounds):
    monkeypatch.setattr(min_time_line, '_accel_brake_coupled',
                        lambda cap, seg, p, kappa, grip: _REAL_COUPLED(cap, seg, p, kappa, grip, rounds=rounds))


@pytest.mark.parametrize('rounds', [0, 1, 2, 4, 6])
@pytest.mark.parametrize('name,xy,make', FIXTURES)
def test_projection_is_feasible_for_any_number_of_rounds(monkeypatch, name, xy, make, rounds):
    # Oracle: the ellipse invariant itself, recomputed here (5% for the discretisation of a station step).
    _with_rounds(monkeypatch, rounds)
    use, free = ellipse_use(xy, make(1.0))
    assert use[free].max() <= 1.0 + ELLIPSE_SLACK, (name, rounds, float(use[free].max()))


@pytest.mark.parametrize('name,xy,make', FIXTURES)
def test_more_rounds_converge(monkeypatch, name, xy, make):
    laps = {}
    for rounds in (0, 2, 4, 40):
        _with_rounds(monkeypatch, rounds)
        laps[rounds] = lap_time(xy, make(1.0))
    # oracle: the converged solution (40 rounds); 4 rounds is within 1.0% of it. Recorded 2026-11-13 on the
    # race-day profiled line, delays zero, c = 1: 0 rounds 11.108 s, 2 rounds 9.571, 4 rounds 9.465, 40 rounds 9.406.
    assert abs(laps[4] - laps[40]) <= 0.01 * laps[40], laps
    assert laps[0] >= laps[2] - 1e-6 and laps[2] >= laps[4] - 1e-6, laps       # s: each round is no slower than the last
    if name == 'race-day profiled':
        assert laps[0] == pytest.approx(11.108, abs=0.005) and laps[4] == pytest.approx(9.465, abs=0.005)    # s


@pytest.mark.parametrize('name,xy,make', FIXTURES)
def test_damped_projected_solution_is_close_to_the_sequential_solution(monkeypatch, name, xy, make):
    physics = make(1.0)
    fast = lap_time(xy, physics)
    monkeypatch.setattr(min_time_line, '_accel_brake_coupled', _sequential_reference)
    sequential = lap_time(xy, physics)
    # oracle: the sequential reference (the problem's own definition). The projected solution is feasible, so it
    # is no faster than the exact one beyond 0.2%, and within 1.0% of it.
    assert fast >= sequential * (1.0 - 0.002), (name, fast, sequential)
    assert abs(fast - sequential) <= 0.01 * sequential, (name, fast, sequential)


# --- acceleration against speed (2026-11-13) ------------------------------------------------
# A real car's acceleration falls as it gets faster. CarPhysics.accel_speeds / accel_values give the limit as a
# table (linear between points, flat beyond the ends); empty means the constant a_accel, exactly.

TABLE = ((0.0, 2.0, 5.0), (6.0, 6.0, 2.0))      # m/s, m/s^2: 6 up to 2 m/s, falling to 2 at 5


def with_table(physics, table=TABLE):
    return replace(physics, accel_speeds=table[0], accel_values=table[1])


def table_ellipse_use(xy, physics):
    """Acceleration steps only: (a_long / table(v at the station left))^2 + (lateral / grip)^2, plus the
    braking ratio a_long / a_brake, and a mask of unpinned steps. Recomputed here from the profile."""
    speed, kappa, seg = speed_profile(xy, physics)
    grip = np.where(kappa >= 0.0, physics.a_lat_left, physics.a_lat_right)
    w = speed ** 2
    a_long = (np.roll(w, -1) - w) / (2.0 * seg)
    limit = np.interp(speed, physics.accel_speeds, physics.accel_values)
    lateral = w * np.abs(kappa) / grip
    accelerating = a_long > 0.0
    use = np.where(accelerating, (a_long / limit) ** 2 + lateral ** 2, 0.0)
    braking = np.where(~accelerating, -a_long / physics.a_brake, 0.0)
    pinned = np.zeros(len(speed), dtype=bool)
    for edge in (physics.v_min, physics.v_max):
        pinned |= (np.abs(speed - edge) < 1e-9) | (np.abs(np.roll(speed, -1) - edge) < 1e-9)
    return use, braking, ~pinned, a_long, limit, accelerating


@pytest.mark.parametrize('name,xy,make', FIXTURES)
def test_an_empty_table_is_exactly_todays_profile(name, xy, make):
    # Oracle: at coupling 0 the unchanged closed form (_accel_brake_box) called directly; at 0.5 the profile built
    # with the table fields never mentioned. Spec: empty = constant a_accel, exactly (2026-11-13 brief).
    physics = make(0.0)
    assert physics.accel_speeds == () and physics.accel_values == ()
    kappa = signed_curvature(xy)
    seg = min_time_line._segments(xy)
    grip = np.where(kappa >= 0.0, physics.a_lat_left, physics.a_lat_right)
    cap = np.minimum(np.sqrt(grip / np.maximum(np.abs(kappa), 1e-9)), physics.v_max)
    assert np.array_equal(min_time_line._accel_brake(cap, seg, physics, kappa, grip),
                          min_time_line._accel_brake_box(cap, seg, physics))
    half = make(0.5)
    explicit = replace(half, accel_speeds=(), accel_values=())
    assert np.array_equal(speed_profile(xy, half)[0], speed_profile(xy, explicit)[0])


@pytest.mark.parametrize('name,xy,make', FIXTURES)
def test_the_table_limit_is_obeyed_at_every_station_and_binds_somewhere(name, xy, make):
    # Oracle: the invariant, recomputed from the profile (delays zero, c = 0): every acceleration step is at most
    # the table's value at the speed it starts from; braking is at most a_brake. It must also bind.
    physics = with_table(make(0.0))
    _, braking, free, a_long, limit, accelerating = table_ellipse_use(xy, physics)
    ratio = np.where(accelerating, a_long / limit, 0.0)
    assert ratio[free].max() <= 1.0 + 1e-6, float(ratio[free].max())
    assert braking[free].max() <= 1.0 + 1e-6, float(braking[free].max())
    assert ratio[free].max() >= 0.98, ('the limit never binds, this tests nothing', float(ratio[free].max()))


@pytest.mark.parametrize('name,xy,make', FIXTURES)
def test_the_table_with_coupling_stays_inside_the_ellipse(name, xy, make):
    # Oracle: the ellipse with the table's limit at the station being left, recomputed here (5% for the
    # discretisation of a step), plus the braking limit. c = 1.
    physics = with_table(make(1.0))
    use, braking, free, *_ = table_ellipse_use(xy, physics)
    assert use[free].max() <= 1.0 + ELLIPSE_SLACK, float(use[free].max())
    assert use[free].max() >= 0.8, 'no acceleration step near the ellipse, this tests nothing'


@pytest.mark.parametrize('c', [0.0, 0.5])
@pytest.mark.parametrize('name,xy,make', FIXTURES)
def test_a_constant_table_is_the_constant_limit(name, xy, make, c):
    # Oracle: closed form. A table that is constant at a_accel is the constant limit; at c = 0 the solver's
    # fixed point must reproduce the closed-form box, at c > 0 the same solver without a table.
    base = make(c)
    constant = replace(base, accel_speeds=(0.0, 10.0), accel_values=(base.a_accel, base.a_accel))
    got = speed_profile(xy, constant)[0]
    want = speed_profile(xy, base)[0]
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-9)         # m/s


@pytest.mark.parametrize('c', [0.0, 1.0])
@pytest.mark.parametrize('name,xy,make', FIXTURES)
def test_a_lower_table_never_shortens_the_lap(name, xy, make, c):
    base = make(c)
    full = with_table(base)
    low = with_table(base, (TABLE[0], tuple(0.8 * v for v in TABLE[1])))
    laps = [lap_time(xy, p) for p in (full, low)]
    assert laps[1] >= laps[0] - 1e-9, laps               # s
    assert laps[1] - laps[0] > 1e-3, laps                # and the limit is a real effect


@pytest.mark.parametrize('c', [0.0, 0.7])
@pytest.mark.parametrize('name,xy,make', FIXTURES)
def test_batch_matches_the_single_path_with_a_table(name, xy, make, c):
    physics = with_table(make(c))
    kappa = signed_curvature(xy)
    seg = min_time_line._segments(xy)
    grip = np.where(kappa >= 0.0, physics.a_lat_left, physics.a_lat_right)
    base = np.minimum(np.sqrt(grip / np.maximum(np.abs(kappa), 1e-9)), physics.v_max)
    rng = np.random.default_rng(3)
    caps = np.minimum(base * rng.uniform(0.6, 1.0, size=(4, len(base))), physics.v_max)
    batch = min_time_line._batch_accel_brake(caps, np.tile(seg, (len(caps), 1)), physics, kappa, grip)
    rows = np.stack([min_time_line._accel_brake(r, seg, physics, kappa, grip) for r in caps])
    np.testing.assert_allclose(batch, rows, rtol=0.0, atol=1e-9)       # m/s


@pytest.mark.parametrize('speeds,values,match', [
    ((0.0, 2.0), (6.0,), 'accel_speeds and accel_values'),                  # unequal lengths
    ((0.0,), (6.0,), 'at least 2'),
    ((0.0, 2.0), (6.0, float('nan')), 'finite'),
    ((0.0, float('inf')), (6.0, 2.0), 'finite'),
    ((0.0, 2.0, 2.0), (6.0, 5.0, 4.0), 'strictly increasing'),
    ((0.0, 3.0, 2.0), (6.0, 5.0, 4.0), 'strictly increasing'),
    ((0.5, 2.0), (6.0, 5.0), 'start at 0'),
    ((0.0, 2.0), (6.0, 0.0), 'positive'),
    ((0.0, 2.0), (6.0, -1.0), 'positive'),
    ((0.0, 2.0), (4.0, 6.0), 'non-increasing'),                          # a rising table is refused
])
def test_the_table_is_validated(speeds, values, match):
    with pytest.raises(ValueError, match=match):
        CarPhysics(accel_speeds=speeds, accel_values=values)


def test_a_flat_or_falling_table_is_accepted():
    p = CarPhysics(accel_speeds=(0.0, 2.0, 5.0), accel_values=(6.0, 6.0, 2.0))
    assert p.accel_at(0.0) == 6.0 and p.accel_at(5.0) == 2.0 and p.accel_at(9.0) == 2.0
    assert p.accel_at(3.5) == pytest.approx(4.0, abs=1e-12)               # m/s^2: halfway down 6 -> 2
    assert CarPhysics(a_accel=4.0).accel_at(2.0) == 4.0
