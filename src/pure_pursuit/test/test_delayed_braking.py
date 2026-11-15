"""racing_math.delayed_braking_speed_limit: the arc braking cap's speed (ROS-free).

Run: `rb python3 -m pytest src/pure_pursuit/test/test_delayed_braking.py` (also collected by
`colcon test --packages-select pure_pursuit`).
"""
import math

import pytest

from pure_pursuit.racing_math import braking_speed_limit, delayed_braking_speed_limit


@pytest.mark.parametrize('clearance, reserve, decel, response', [
    (0.59, 0.14, 3.0, 0.25), (1.0, 0.14, 3.0, 0.25), (0.5, 0.0, 2.0, 0.4), (3.0, 0.3, 4.5, 0.1)])
def test_the_car_stops_exactly_at_the_reserve(clearance, reserve, decel, response):
    # oracle: closed form. Holding v for t then braking at a covers v t + v^2 / (2 a); at the returned v that is the
    # usable distance (clearance - reserve) exactly, and 1 cm/s faster overruns it.
    v = delayed_braking_speed_limit(clearance, reserve, decel, response)
    travelled = lambda s: s * response + s * s / (2.0 * decel)
    assert travelled(v) == pytest.approx(clearance - reserve, abs=1e-9)   # m
    assert travelled(v + 0.01) > clearance - reserve


def test_a_worked_value():
    # oracle: by hand, 0.59 m free, 0.14 m reserve, 3 m/s^2, 0.25 s: -0.75 + sqrt(0.5625 + 6 * 0.45) = 1.0563 m/s
    assert delayed_braking_speed_limit(0.59, 0.14, 3.0, 0.25) == pytest.approx(1.0563, abs=1e-4)  # m/s


def test_zero_response_is_the_plain_braking_limit():
    # invariant: with no response time it is v^2 <= 2 a (d - reserve), braking_speed_limit's law.
    for d in (0.2, 0.6, 1.0, 2.5):
        assert delayed_braking_speed_limit(d, 0.14, 3.0, 0.0) == pytest.approx(
            braking_speed_limit(d, 0.14, 3.0, 99.0), abs=1e-12)


def test_a_longer_response_or_weaker_brakes_allow_less_speed():
    base = delayed_braking_speed_limit(0.8, 0.14, 3.0, 0.25)
    assert delayed_braking_speed_limit(0.8, 0.14, 3.0, 0.35) < base
    assert delayed_braking_speed_limit(0.8, 0.14, 2.5, 0.25) < base


@pytest.mark.parametrize('clearance', [0.14, 0.10, 0.0, -0.05])
def test_inside_the_reserve_it_is_zero(clearance):
    # the boundary itself (0.14) and past it: nothing usable, so stop
    assert delayed_braking_speed_limit(clearance, 0.14, 3.0, 0.25) == 0.0


def test_one_step_past_the_reserve_is_positive():
    assert delayed_braking_speed_limit(0.15, 0.14, 3.0, 0.25) > 0.0


def test_nothing_seen_is_no_limit():
    assert delayed_braking_speed_limit(math.inf, 0.14, 3.0, 0.25) == math.inf


@pytest.mark.parametrize('args, match', [
    ((math.nan, 0.14, 3.0, 0.25), 'finite'),
    ((0.5, math.inf, 3.0, 0.25), 'finite'),
    ((0.5, 0.14, 0.0, 0.25), 'braking positive'),
    ((0.5, 0.14, -3.0, 0.25), 'braking positive'),
    ((0.5, -0.1, 3.0, 0.25), 'non-negative'),
    ((0.5, 0.14, 3.0, -0.1), 'non-negative'),
])
def test_bad_inputs_are_refused(args, match):
    with pytest.raises(ValueError, match=match):
        delayed_braking_speed_limit(*args)
