"""The safety wrapper (neural_racer/safety_wrapper.py). ROS-free.

Runner: python3 -m pytest src/neural_racer/test/test_safety_wrapper.py -v   (unsourced)

Oracles: gap_logic's own functions (swept_path_clearance, braking_speed_limit,
body_arc_free_distance), the curvature cap's closed form v = sqrt(a_lat * L / tan(delta)),
the steering-rate step max_rate * dt, the invariant that the wrapper never raises speed,
and a recorded measurement: car 1's wall-alongside latch, rb1 20261111-155542 (gap_follow's
fixture, src/gap_follow/test/data/flank_latch_rb1_20261111-155542.npz).
"""
import dataclasses
import math
from pathlib import Path

import nr_testutil
import numpy as np
import pytest

from gap_follow import gap_logic
from neural_racer.safety_wrapper import SafetyWrapper, WrapperConfig, config_problem, scan_angles

ANGLES = scan_angles(nr_testutil.UST_ANGLE_MIN, nr_testutil.UST_INCREMENT, 1081)
PATH = np.zeros((6, 2))
RMIN, RMAX = 0.02, 30.0


def _apply(w, steer=0.0, speed=1.0, ranges=None, prev=0.0, dt=0.025, path=PATH, map_hit=None):
    r = nr_testutil.clear_scan() if ranges is None else ranges
    return w.apply(steer, speed, path, r, ANGLES, RMIN, RMAX, prev, dt, map_hit)


@pytest.fixture
def w():
    return SafetyWrapper(WrapperConfig(max_speed=4.0))


def test_open_space_passes_the_network_command(w):
    res = _apply(w, steer=0.0, speed=1.5)
    assert not res.stopped and res.speed == 1.5 and res.binding == 'network' and not res.intervened


@pytest.mark.parametrize('field', ['steer', 'speed'])
@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf])
def test_non_finite_output_stops(w, field, bad):
    res = _apply(w, **{field: bad})
    assert res.stopped and res.speed == 0.0 and res.reason == 'nan_output'


def test_non_finite_forecast_stops(w):
    path = PATH.copy()
    path[3, 1] = math.nan
    res = _apply(w, path=path)
    assert res.stopped and res.reason == 'nan_output'


@pytest.mark.parametrize('n', [1080, 1082, 0])
def test_wrong_beam_count_stops(w, n):
    res = w.apply(0.0, 1.0, PATH, np.full(n, 5.0), np.zeros(n), RMIN, RMAX, 0.0, 0.025)
    assert res.stopped and res.reason == 'beam_count'


@pytest.mark.parametrize('fraction, stops', [(0.30, False), (0.31, True), (1.0, True)])
def test_invalid_fraction_boundary(w, fraction, stops):
    r = nr_testutil.clear_scan()
    r[:int(round(fraction * 1081))] = math.nan
    res = _apply(w, ranges=r)
    assert res.stopped == stops
    if stops:
        assert res.reason == 'invalid_beams' and res.speed == 0.0


def test_all_zero_scan_stops(w):
    res = _apply(w, ranges=np.zeros(1081))     # 0.0 is below range_min: every beam invalid
    assert res.stopped and res.reason == 'invalid_beams'


def test_braking_cap_matches_gap_logic(w):
    r = nr_testutil.clear_scan()
    r[530:551] = 1.2                             # a wall straight ahead
    res = _apply(w, speed=4.0, ranges=r)
    c = w.cfg
    valid = np.isfinite(r) & (r > RMIN) & (r < RMAX)
    ahead = np.abs(ANGLES) <= math.pi / 2.0 + 1e-9
    clearance = gap_logic.swept_path_clearance(r[ahead], valid[ahead], ANGLES[ahead], 0.0, c.car_width / 2.0 + 0.05,
                                               c.car_length, c.wheelbase, c.laser_offset_x, 0.0, 3.0)
    # the patch spans +/-2.5 deg: its nearest point is 1.2 cos(2.5 deg) ahead of the LiDAR
    assert clearance == pytest.approx(1.2 * math.cos(math.radians(2.5)) + c.laser_offset_x
                                      - (c.wheelbase + c.car_length) / 2.0, abs=1e-6)
    expected = gap_logic.braking_speed_limit(clearance, c.reserve_distance, c.max_braking_decel,
                                             c.max_speed, c.braking_latency_sec)
    assert res.binding == 'braking' and res.speed == pytest.approx(expected, abs=1e-12)
    assert res.speed < 4.0 and res.intervened and res.clearance == pytest.approx(clearance, abs=1e-12)


def _corridor(half_width):
    """A straight corridor along the car's axis, its walls half_width either side of the LiDAR."""
    with np.errstate(divide='ignore'):
        r = half_width / np.abs(np.sin(ANGLES))
    return np.minimum(r, 8.0)


@pytest.mark.parametrize('half_width', [0.30, 0.45, 0.75])
def test_corridor_side_walls_do_not_brake(w, half_width):
    """Driving straight down a corridor never reaches its side walls: the
    nearest return in any direction once held the student to ~0.3 m/s here."""
    res = _apply(w, steer=0.0, speed=4.0, ranges=_corridor(half_width))
    assert res.binding == 'network' and res.speed == 4.0 and not res.intervened


def test_turning_into_a_corridor_wall_brakes(w):
    """The same corridor, steering hard left: the arc runs into the left wall."""
    res = _apply(w, steer=0.33, speed=4.0, prev=0.33, ranges=_corridor(0.45))
    assert res.binding == 'braking' and res.speed < 1.5


def test_the_arc_asked_for_brakes_before_the_slew_reaches_it(w):
    """Straight ahead, the network asks for hard left into the wall: the slew lets only
    0.05 rad through this tick, and that arc is clear, but the arc asked for is not."""
    res = _apply(w, steer=0.33, speed=4.0, prev=0.0, ranges=_corridor(0.45))
    assert res.binding == 'braking' and res.speed < 1.5


def _flank_wall(r_wall, side=1.0):
    """A wall alongside the body on the left (side 1) or right (side -1): beams 80-100 deg
    that side at r_wall from the LiDAR, so its nearest point is r_wall sin(80 deg) off the axis."""
    r = nr_testutil.clear_scan()
    r[(side * ANGLES > math.radians(80)) & (side * ANGLES < math.radians(100))] = r_wall
    return r


def test_wall_alongside_the_body_stops(w):
    """A return inside the swept margin beside the body: the swept path reads no room at all.
    The flank creep is off here: this isolates the braking cap, and its baseline is the latch
    the creep removes (as gap_follow's test_arc_guard_backs_off_a_turn_whose_arc_is_blocked)."""
    w = SafetyWrapper(dataclasses.replace(w.cfg, flank_creep_length=0.0))
    res = _apply(w, speed=2.0, ranges=_flank_wall(0.18))   # 2.5 cm off the left flank
    assert res.speed == 0.0 and res.binding == 'braking' and res.clearance == 0.0


def test_wall_alongside_the_body_creeps(w):
    """The same wall with the shipped flank creep: a crawl at escape_creep_speed. Closed form:
    the wall's nearest point is 0.18 sin(80 deg) = 0.1773 m off the axis, outside the creep
    rectangle's 0.155 + 0.02 = 0.175 m, and driving straight never brings it closer."""
    c = w.cfg
    r = _flank_wall(0.18)
    assert 0.18 * math.sin(math.radians(80)) > c.car_width / 2.0 + c.flank_creep_margin
    valid = np.isfinite(r) & (r > RMIN) & (r < RMAX)
    ahead = np.abs(ANGLES) <= math.pi / 2.0 + 1e-9
    assert gap_logic.body_arc_free_distance(
        r[ahead], valid[ahead], ANGLES[ahead], 0.0, c.car_width / 2.0 + c.flank_creep_margin,
        c.car_length, c.wheelbase, c.laser_offset_x, 0.0, max_length=0.50) == math.inf
    res = _apply(w, speed=2.0, ranges=r)
    assert res.binding == 'flank_creep' and res.speed == pytest.approx(0.40, abs=1e-12)    # m/s
    assert not res.stopped and res.intervened and res.clearance == 0.0


def test_touching_the_flank_wall_stops(w):
    """A return 1 cm off the body (0.165 sin(80 deg) = 0.1625 m off the axis): inside the creep
    rectangle's 2 cm margin, so no crawl."""
    res = _apply(w, speed=2.0, ranges=_flank_wall(0.165))
    assert res.speed == 0.0 and res.binding == 'braking'


def test_steering_into_the_flank_wall_stops(w):
    """Wall on the left, the network steering hard left: the body's arc meets the wall within 0.5 m."""
    res = _apply(w, steer=0.33, prev=0.33, speed=2.0, ranges=_flank_wall(0.18))
    assert res.speed == 0.0 and res.binding == 'braking'


def test_the_arc_asked_for_into_the_flank_wall_stops(w):
    """Wall 0.19 m off the LiDAR on the left, the rack straight, the network asking hard left:
    the slew lets 0.05 rad through this tick and the body is free along that arc (oracle:
    gap_logic), but not along the arc asked for, so no crawl."""
    c = w.cfg
    r = _flank_wall(0.19)
    valid = np.isfinite(r) & (r > RMIN) & (r < RMAX)
    ahead = np.abs(ANGLES) <= math.pi / 2.0 + 1e-9

    def free(steer):
        return gap_logic.body_arc_free_distance(
            r[ahead], valid[ahead], ANGLES[ahead], math.tan(steer) / c.wheelbase,
            c.car_width / 2.0 + c.flank_creep_margin, c.car_length, c.wheelbase, c.laser_offset_x, 0.0,
            max_length=c.flank_creep_length)
    slew = c.max_steering_rate * 0.025
    assert free(slew) == math.inf and free(0.33) < c.flank_creep_length
    res = _apply(w, steer=0.33, prev=0.0, speed=2.0, ranges=r)
    assert res.steering == pytest.approx(slew, abs=1e-12)                              # rad
    assert res.speed == 0.0 and res.binding == 'braking'


def test_steering_away_from_the_flank_wall_creeps(w):
    res = _apply(w, steer=-0.33, prev=-0.33, speed=2.0, ranges=_flank_wall(0.18))
    assert res.binding == 'flank_creep' and res.speed == pytest.approx(0.40, abs=1e-12)    # m/s


def test_wall_ahead_inside_the_reserve_does_not_creep(w):
    """A wall alongside and one straight ahead 0.35 m from the LiDAR: the body meets the one
    ahead after 0.35 + 0.26 - 0.47 = 0.14 m, inside the creep length."""
    r = _flank_wall(0.18)
    r[530:551] = 0.35
    res = _apply(w, speed=2.0, ranges=r)
    assert res.speed == 0.0 and res.binding == 'braking'


def test_creep_only_when_the_braking_cap_is_zero(w):
    """A post just ahead and beside the path: inside the swept band (0.205 m a side), outside
    the creep rectangle (0.175 m), 0.22 m along. The braking cap is small but not zero, so it
    stands, below the crawl's 0.40 m/s."""
    c = w.cfg
    r = nr_testutil.clear_scan()
    post = (ANGLES > math.radians(23.0)) & (ANGLES < math.radians(24.5))
    r[post] = 0.47
    y = 0.47 * np.sin(ANGLES[post])
    assert np.all((y > c.car_width / 2.0 + c.flank_creep_margin) & (y < c.car_width / 2.0 + c.swept_clearance_margin))
    valid = np.isfinite(r) & (r > RMIN) & (r < RMAX)
    ahead = np.abs(ANGLES) <= math.pi / 2.0 + 1e-9
    clearance = gap_logic.swept_path_clearance(r[ahead], valid[ahead], ANGLES[ahead], 0.0,
                                               c.car_width / 2.0 + c.swept_clearance_margin, c.car_length,
                                               c.wheelbase, c.laser_offset_x, 0.0, 3.0)
    expected = gap_logic.braking_speed_limit(clearance, c.reserve_distance, c.max_braking_decel,
                                             c.max_speed, c.braking_latency_sec)
    assert 0.0 < expected < c.escape_creep_speed
    assert gap_logic.body_arc_free_distance(
        r[ahead], valid[ahead], ANGLES[ahead], 0.0, c.car_width / 2.0 + c.flank_creep_margin,
        c.car_length, c.wheelbase, c.laser_offset_x, 0.0, max_length=0.50) == math.inf
    res = _apply(w, speed=2.0, ranges=r)
    assert res.binding == 'braking' and res.speed == pytest.approx(expected, abs=1e-12)     # m/s


def test_flank_creep_is_a_ceiling():
    """The crawl raises the braking cap, never the command: a slower network or a lower
    max_speed still sets the speed."""
    w = SafetyWrapper(WrapperConfig(max_speed=4.0))
    res = _apply(w, speed=0.25, ranges=_flank_wall(0.18))
    assert res.binding == 'network' and res.speed == 0.25
    w = SafetyWrapper(WrapperConfig(max_speed=0.3))
    assert _apply(w, speed=2.0, ranges=_flank_wall(0.18)).speed == 0.3
    w = SafetyWrapper(WrapperConfig(max_speed=4.0, escape_creep_speed=0.30))
    assert _apply(w, speed=2.0, ranges=_flank_wall(0.18)).speed == 0.30


def test_never_raises_speed_beside_a_wall(w):
    """Invariant over random walls alongside either flank, steerings and commands; both
    outcomes occur, so the creep is exercised, not skipped."""
    rng = np.random.default_rng(1)
    bindings = set()
    for _ in range(300):
        r = _flank_wall(float(rng.uniform(0.16, 0.22)), side=float(rng.choice([-1.0, 1.0])))
        speed = float(rng.uniform(0.0, 4.0))
        res = _apply(w, steer=float(rng.uniform(-0.33, 0.33)), speed=speed, ranges=r,
                     prev=float(rng.uniform(-0.33, 0.33)))
        bindings.add(res.binding)
        assert res.speed <= speed + 1e-12
        if res.binding == 'flank_creep':
            assert res.speed == w.cfg.escape_creep_speed
    assert {'flank_creep', 'braking'} <= bindings


LATCH = Path(__file__).resolve().parents[2] / 'gap_follow' / 'test' / 'data' / 'flank_latch_rb1_20261111-155542.npz'


def _replay(w, steer):
    """The recorded scans through the wrapper, the network asking 1 m/s at `steer`."""
    data = np.load(LATCH)
    n = data['ranges'].shape[1]
    angles = scan_angles(float(data['angle_min']), float(data['angle_increment']), n)
    return [w.apply(steer, 1.0, PATH, r.astype(np.float64), angles, float(data['range_min']),
                    float(data['range_max']), steer, 0.025) for r in data['ranges']]


def test_the_recorded_latch_stops_without_the_creep(w):
    # recorded measurement: rb1 20261111-155542, 88.5-90.0 s, wall on the left, rack at -0.20 rad
    w = SafetyWrapper(dataclasses.replace(w.cfg, flank_creep_length=0.0))
    res = _replay(w, -0.20)
    assert len(res) == 60
    assert sum(r.speed == 0.0 and r.binding == 'braking' for r in res) >= 57      # 95 %


def test_the_recorded_latch_creeps(w):
    # recorded measurement: rb1 20261111-155542, 88.5-90.0 s
    res = _replay(w, -0.20)
    assert sum(r.binding == 'flank_creep' and r.speed == 0.40 for r in res) >= 57   # 95 %


def test_the_recorded_latch_steering_into_the_wall_stops(w):
    # recorded measurement: rb1 20261111-155542, 88.5-90.0 s (wall on the left)
    res = _replay(w, 0.33)
    assert sum(r.speed == 0.0 for r in res) >= 57                                    # 95 %


def test_shipped_config_has_no_problem():
    assert config_problem(WrapperConfig()) is None


@pytest.mark.parametrize('field, value, ok', [
    ('flank_creep_length', 0.0, True), ('flank_creep_length', 2.0, True),
    ('flank_creep_length', -0.01, False), ('flank_creep_length', 2.01, False),
    ('flank_creep_length', math.nan, False), ('flank_creep_length', math.inf, False),
    ('flank_creep_margin', 0.0, True), ('flank_creep_margin', 0.10, True),
    ('flank_creep_margin', -0.01, False), ('flank_creep_margin', 0.11, False),
    ('flank_creep_margin', math.nan, False),
    ('escape_creep_speed', 0.5, True), ('escape_creep_speed', 0.0, False),
    ('escape_creep_speed', 0.51, False), ('escape_creep_speed', -0.1, False),
    ('escape_creep_speed', math.nan, False)])
def test_config_problem_bounds(field, value, ok):
    """gap_follow_node's bounds: length [0, 2] m, margin [0, 0.10] m, speed (0, 0.5] m/s."""
    problem = config_problem(WrapperConfig(**{field: value}))
    assert (problem is None) == ok, problem
    if not ok:
        assert problem.startswith(field)


def test_config_problem_creep_must_stop_inside_its_length():
    """Closed form: stopping from v takes v * latency + v^2 / (2 decel), 0.1267 m at 0.4 m/s
    with 0.25 s and 3.0 m/s^2."""
    c = WrapperConfig()
    stopping = 0.40 * c.braking_latency_sec + 0.40 ** 2 / (2.0 * c.max_braking_decel)
    assert stopping == pytest.approx(0.12667, abs=1e-5)                              # m
    assert 'stop' in config_problem(WrapperConfig(flank_creep_length=stopping - 0.001))
    assert config_problem(WrapperConfig(flank_creep_length=stopping + 0.001)) is None


def test_returns_behind_the_forward_window_are_ignored(w):
    """gap_follow's 180 deg window: beams looking back (the car's own rear, a
    car behind) do not brake it."""
    r = nr_testutil.clear_scan()
    r[np.abs(ANGLES) > math.radians(95)] = 0.10
    res = _apply(w, speed=3.0, ranges=r)
    assert res.binding == 'network' and res.speed == 3.0


def test_obstacle_inside_reserve_gives_zero_speed(w):
    r = nr_testutil.clear_scan()
    r[530:551] = 0.35                            # 0.35 - 0.06 to the bumper: inside the 0.20 reserve
    res = _apply(w, speed=2.0, ranges=r)
    assert res.speed == 0.0 and not res.stopped and res.binding == 'braking'


def test_curvature_cap_closed_form(w):
    res = _apply(w, steer=0.33, speed=4.0, prev=0.33)
    expected = math.sqrt(2.0 * 0.36 / math.tan(0.33))
    assert res.binding == 'curvature' and res.speed == pytest.approx(expected, abs=1e-12)


def test_steering_rate_limited(w):
    res = _apply(w, steer=0.33, prev=0.0, dt=0.025)
    assert res.steering == pytest.approx(2.0 * 0.025, abs=1e-12)    # max_rate * dt
    assert res.intervened


def test_steering_rate_uses_capped_dt(w):
    res = _apply(w, steer=0.33, prev=0.0, dt=5.0)
    assert res.steering == pytest.approx(2.0 * 0.10, abs=1e-12)     # command_slew_max_dt


@pytest.mark.parametrize('steer', [0.5, -0.5, 10.0])
def test_steering_clipped(w, steer):
    res = _apply(w, steer=steer, prev=math.copysign(0.33, steer), dt=1.0)
    assert abs(res.steering) == pytest.approx(0.33, abs=1e-12)


def test_speed_clipped_to_max_speed():
    w = SafetyWrapper(WrapperConfig(max_speed=1.0))
    assert _apply(w, speed=3.0).speed == 1.0


def test_negative_speed_clipped_to_zero(w):
    assert _apply(w, speed=-1.0).speed == 0.0


def test_map_hit_caps_speed(w):
    res = _apply(w, speed=4.0, map_hit=0.5)
    c = w.cfg
    expected = gap_logic.braking_speed_limit(0.5, c.reserve_distance, c.max_braking_decel,
                                             c.max_speed, c.braking_latency_sec)
    assert res.binding == 'map_path' and res.speed == pytest.approx(expected, abs=1e-12)


def test_never_raises_speed(w):
    """Invariant over random scans and commands."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        r = rng.uniform(0.1, 12.0, 1081)
        speed = float(rng.uniform(0.0, 4.0))
        res = _apply(w, steer=float(rng.uniform(-0.33, 0.33)), speed=speed, ranges=r,
                     prev=float(rng.uniform(-0.33, 0.33)))
        assert res.speed <= speed + 1e-12


def test_bad_timing_stops(w):
    assert _apply(w, dt=math.nan).reason == 'bad_timing'
    assert _apply(w, prev=math.inf).reason == 'bad_timing'
