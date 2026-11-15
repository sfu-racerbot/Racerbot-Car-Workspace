"""gap_logic.escape_arc_clearance: the braking ceiling's clearance with a tighter arc toward the gap credited
(2026-11-14). ROS-free; run as `rb python3 -m pytest src/gap_follow/test/test_escape_arc_clearance.py`.

Free distance along an arc is gap_logic.swept_path_clearance (tested on its own in test_swept_path_clearance*.py);
what is new here is which arcs are credited and what each is charged, and those are what these tests pin, against
closed forms.
"""
import math

import numpy as np
import pytest

from gap_follow import gap_logic

N = 1081
ANGLES = np.linspace(-math.radians(135), math.radians(135), N)
GEOM = dict(half_width=0.205, car_length=0.58, wheelbase=0.36, laser_offset_x=0.26, laser_offset_y=0.0,
            max_length=5.0)


def pillar_scan(open_range=8.0, near=2.0, far=3.0):
    """A bending corridor: a wall square across the path ``near`` m ahead from 40 deg right to 10 deg left, and a
    farther one at ``far`` m from 10 to 70 deg left. Straight on is blocked at about near - 0.21 m; 0.1 rad more
    left steering curves past the near wall and is blocked by the far one, further on (2.78 m against 1.79 m)."""
    ranges = np.full(N, open_range)
    m = (ANGLES >= math.radians(-40)) & (ANGLES <= math.radians(10))
    ranges[m] = near / np.cos(ANGLES[m])
    m2 = (ANGLES > math.radians(10)) & (ANGLES <= math.radians(70))
    ranges[m2] = far
    return ranges, np.ones(N, dtype=bool)


def free(ranges, valid, steering):
    return gap_logic.swept_path_clearance(ranges, valid, ANGLES, math.tan(steering) / 0.36, 0.205, 0.58, 0.36,
                                          0.26, 0.0, 5.0)


def call(ranges, valid, arcs=(0.0, 0.0), reach=0.1, gap_side=1.0, speed=2.0, rate=4.0, latency=0.25,
         lateral=3.0, candidates=4):
    return gap_logic.escape_arc_clearance(ranges, valid, ANGLES, arcs, reach, gap_side, speed, rate, latency,
                                          lateral, GEOM['half_width'], GEOM['car_length'], GEOM['wheelbase'],
                                          GEOM['laser_offset_x'], GEOM['laser_offset_y'], GEOM['max_length'],
                                          0.42, candidates)


def expected(ranges, valid, arcs=(0.0, 0.0), reach=0.1, side=1.0, speed=2.0, rate=4.0, latency=0.25,
             lateral=3.0, candidates=4):
    # oracle: closed form of the spec (docstring of escape_arc_clearance)
    best = min(free(ranges, valid, s) for s in arcs)
    start = arcs[-1]
    for k in range(1, candidates + 1):
        s = start + side * reach * k / candidates
        if speed ** 2 * abs(math.tan(s)) / 0.36 > lateral:
            break
        best = max(best, free(ranges, valid, s) - speed * (latency + abs(s - start) / rate))
    return best


def test_reach_zero_is_the_shortest_current_arc():
    r, v = pillar_scan()
    assert call(r, v, arcs=(0.0, 0.05), reach=0.0) == min(free(r, v, 0.0), free(r, v, 0.05))


def test_a_tighter_arc_toward_the_gap_is_credited_minus_its_charge():
    r, v = pillar_scan()
    base = call(r, v, reach=0.0)
    got = call(r, v)
    assert got > base, 'precondition: a left arc has more room than straight ahead'
    assert got == pytest.approx(expected(r, v), abs=1e-9)    # m


def test_the_charge_is_the_distance_driven_before_the_rack_gets_there():
    # One candidate: 0.1 rad more steering at 4 rad/s is 0.025 s, plus 0.25 s latency, at 2 m/s: 0.55 m.
    r, v = pillar_scan()
    got = call(r, v, candidates=1)
    raw = free(r, v, 0.1)
    assert math.isfinite(raw), 'precondition: the candidate arc is blocked further on'
    assert got == pytest.approx(max(free(r, v, 0.0), raw - 2.0 * (0.25 + 0.1 / 4.0)), abs=1e-9)   # m


def test_the_gap_on_the_other_side_credits_nothing():
    # gap_side right: the right arcs turn into the near wall, so nothing beats the baseline; gap dead ahead: none tried.
    r, v = pillar_scan()
    base = call(r, v, reach=0.0)
    assert call(r, v, gap_side=-1.0) == base
    assert call(r, v, gap_side=0.0) == base


HOLD_SPEED = 2.5
HOLD_LIMIT = HOLD_SPEED ** 2 * math.tan(0.1) / 0.36     # m/s^2 the single 0.1 rad candidate asks at 2.5 m/s


def test_an_arc_the_car_cannot_hold_is_not_credited():
    r, v = pillar_scan()
    base = call(r, v, speed=HOLD_SPEED, reach=0.0)
    assert call(r, v, speed=HOLD_SPEED, lateral=HOLD_LIMIT - 1e-6, candidates=1) == base


def test_one_step_inside_the_hold_limit_is_credited():
    r, v = pillar_scan()
    base = call(r, v, speed=HOLD_SPEED, reach=0.0)
    assert call(r, v, speed=HOLD_SPEED, lateral=HOLD_LIMIT + 1e-6, candidates=1) > base


def test_more_latency_or_a_slower_rack_credits_less():
    r, v = pillar_scan()
    base = call(r, v)
    assert call(r, v, latency=0.5) < base
    assert call(r, v, rate=1.0) < base


def test_never_below_the_baseline():
    rng = np.random.default_rng(4)
    for _ in range(30):
        r = rng.uniform(0.3, 8.0, N)
        v = rng.random(N) > 0.05
        base = call(r, v, reach=0.0)
        assert call(r, v, reach=0.15, speed=float(rng.uniform(0, 5))) >= base


def test_a_clear_current_arc_stays_clear():
    r, v = pillar_scan(near=8.0, far=8.0)
    assert call(r, v) == math.inf


@pytest.mark.parametrize('kwargs, match', [
    (dict(reach=-0.1), 'reach'), (dict(rate=0.0), 'steering_rate'), (dict(latency=-0.1), 'latency'),
    (dict(lateral=0.0), 'lateral_limit'), (dict(speed=math.nan), 'speed'), (dict(reach=math.inf), 'reach'),
    (dict(candidates=0), 'candidates'), (dict(arcs=()), 'arc_steerings'),
])
def test_bad_inputs_are_refused(kwargs, match):
    r, v = pillar_scan()
    with pytest.raises(ValueError, match=match):
        call(r, v, **kwargs)
