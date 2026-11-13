"""GripCurve, understeer_steering and delivered_curvature_speed_limit (Tasks 6 and 7 of
docs/superpowers/plans/2026-11-12-post-race-speed-program.md). Pure Python, no rclpy.

Every expected value is a closed form recomputed here from the plan's example curve
asked = (0, 2.5, 10), delivered = (0, 2.5, 5.5): kinematic lateral acceleration is
v^2 * tan(delta) / wheelbase, delivered is what the car achieves.
"""
import math
import random

import pytest

from pure_pursuit import racing_math as rm

WHEELBASE = 0.36   # m
MAX_STEER = 0.42   # rad, car 2's autonomy limit
SPEED = 3.0        # m/s


def example_curve():
    return rm.GripCurve(asked=(0.0, 2.5, 10.0), delivered=(0.0, 2.5, 5.5))


def steering_for_lateral(lateral, speed=SPEED):
    return math.atan(lateral * WHEELBASE / speed ** 2)


def lateral_of(steering, speed=SPEED):
    return speed ** 2 * math.tan(abs(steering)) / WHEELBASE


# --- GripCurve -----------------------------------------------------------------------

def test_deliver_is_piecewise_linear_and_flat_beyond_the_last_point():
    c = example_curve()
    assert c.deliver(1.0) == pytest.approx(1.0, abs=1e-12)                       # first segment, slope 1
    assert c.deliver(6.25) == pytest.approx(2.5 + (6.25 - 2.5) * 3.0 / 7.5, abs=1e-12)
    assert c.deliver(10.0) == pytest.approx(5.5, abs=1e-12)
    assert c.deliver(50.0) == pytest.approx(5.5, abs=1e-12)                      # flat past the last asked


def test_ask_for_inverts_deliver_over_the_curve():
    c = example_curve()
    for k in range(50):
        asked = 10.0 * k / 49.0
        assert c.ask_for(c.deliver(asked)) == pytest.approx(asked, abs=1e-9)  # lateral m/s^2


def test_ask_for_beyond_the_last_delivered_returns_the_last_asked():
    c = example_curve()
    assert c.ask_for(5.5) == pytest.approx(10.0, abs=1e-12)
    assert c.ask_for(9.0) == pytest.approx(10.0, abs=1e-12)   # never more than the last asked


def test_ask_for_swapped_arrays_would_differ():
    # oracle: closed form 2.5 + (4.0 - 2.5) * (10 - 2.5) / (5.5 - 2.5) = 6.25
    assert example_curve().ask_for(4.0) == pytest.approx(6.25, abs=1e-12)


# --- grip_curve_from_params ----------------------------------------------------------

def test_two_empty_lists_mean_compensation_off():
    assert rm.grip_curve_from_params([], []) is None


@pytest.mark.parametrize('asked, delivered, rule', [
    ([0.0, 2.5], [0.0], 'equal length'),
    ([0.0], [0.0], 'at least 2 points'),
    ([0.0, float('nan')], [0.0, 1.0], 'finite'),
    ([0.0, float('inf')], [0.0, 1.0], 'finite'),
    ([0.0, 2.5, 2.5], [0.0, 1.0, 2.0], 'strictly increasing'),
    ([0.0, 2.5, 5.0], [0.0, 2.0, 1.0], 'strictly increasing'),
    ([1.0, 2.5], [0.0, 1.0], 'start at'),
    ([0.0, 2.5], [0.5, 1.0], 'start at'),
    ([0.0, 2.5], [0.0, 3.0], 'delivered'),
])
def test_each_invalid_curve_raises_naming_its_rule(asked, delivered, rule):
    with pytest.raises(ValueError, match=rule):
        rm.grip_curve_from_params(asked, delivered)


def test_one_empty_list_and_one_not_is_refused():
    with pytest.raises(ValueError, match='equal length'):
        rm.grip_curve_from_params([], [0.0, 1.0])


# --- understeer_steering -------------------------------------------------------------

def test_identity_curve_returns_the_input_steering():
    c = rm.GripCurve(asked=(0.0, 10.0), delivered=(0.0, 10.0))
    for steering in (-0.3, -0.05, 0.0, 0.11, 0.3):
        assert rm.understeer_steering(steering, SPEED, WHEELBASE, c, MAX_STEER) == pytest.approx(
            steering, abs=1e-12)  # rad


def test_a_corner_above_the_knee_is_steered_harder_by_the_curve_inverse():
    # kinematic lateral 4.0 -> asked 2.5 + (4.0 - 2.5) * 7.5 / 3.0 = 6.25 (plan Task 6)
    out = rm.understeer_steering(steering_for_lateral(4.0), SPEED, WHEELBASE, example_curve(), MAX_STEER)
    assert lateral_of(out) == pytest.approx(6.25, abs=1e-6)  # m/s^2, the plan's tolerance in lateral terms


def test_below_the_knee_the_steering_is_unchanged():
    s = steering_for_lateral(2.0)
    assert rm.understeer_steering(s, SPEED, WHEELBASE, example_curve(), MAX_STEER) == pytest.approx(s, abs=1e-12)


def test_negative_steering_mirrors():
    s = steering_for_lateral(4.0)
    pos = rm.understeer_steering(s, SPEED, WHEELBASE, example_curve(), MAX_STEER)
    neg = rm.understeer_steering(-s, SPEED, WHEELBASE, example_curve(), MAX_STEER)
    assert neg == pytest.approx(-pos, abs=1e-12)  # rad


def test_the_result_clips_at_the_steering_limit():
    # the compensated steering for lateral 4.0 is atan(6.25 * 0.36 / 9) = 0.245 rad; a 0.20 rad limit clips it
    s = steering_for_lateral(4.0)
    assert math.atan(6.25 * WHEELBASE / SPEED ** 2) == pytest.approx(0.245, abs=1e-3)  # rad, closed form
    out = rm.understeer_steering(s, SPEED, WHEELBASE, example_curve(), 0.20)
    assert out == pytest.approx(0.20, abs=1e-12)   # rad


def test_below_min_speed_the_steering_is_returned_unchanged():
    # 0.2 rad at 3 m/s is a kinematic 9 * tan(0.2) / 0.36 = 5.07 m/s^2: above the 2.5 knee and below the
    # last delivered 5.5, so it is compensated -- unless the speed is under min_speed.
    c = example_curve()
    assert 2.5 < lateral_of(0.2) < 5.5
    assert rm.understeer_steering(0.2, SPEED, WHEELBASE, c, MAX_STEER) > 0.2 + 1e-3        # compensated
    assert rm.understeer_steering(0.2, SPEED, WHEELBASE, c, MAX_STEER, min_speed=3.5) == pytest.approx(
        0.2, abs=1e-12)                                                                      # rad, below min_speed


def test_no_curve_returns_the_clipped_steering():
    assert rm.understeer_steering(0.3, SPEED, WHEELBASE, None, MAX_STEER) == pytest.approx(0.3, abs=1e-12)
    assert rm.understeer_steering(0.9, SPEED, WHEELBASE, None, MAX_STEER) == pytest.approx(MAX_STEER, abs=1e-12)


def test_steering_is_continuous_across_the_grip_limit():
    # Oracle: the curve's last delivered is 5.5 m/s^2. Just below it the output is the steering for
    # asked[-1] = 10; just above it must not drop back to the kinematic steering (advisor correction:
    # the earlier "unchanged above the last delivered" rule was a discontinuity there).
    c = example_curve()
    below = rm.understeer_steering(steering_for_lateral(5.5 - 1e-6), SPEED, WHEELBASE, c, MAX_STEER)
    above = rm.understeer_steering(steering_for_lateral(5.5 + 1e-6), SPEED, WHEELBASE, c, MAX_STEER)
    assert abs(above - below) < 1e-6   # rad


def test_just_above_the_peak_the_output_holds_the_peak_ask_steering():
    # Oracle: closed form atan(asked[-1] * wheelbase / speed^2) = atan(10 * 0.36 / 9).
    c = example_curve()
    for wanted in (5.5 + 1e-6, 7.0, 9.9):
        out = rm.understeer_steering(steering_for_lateral(wanted), SPEED, WHEELBASE, c, MAX_STEER)
        assert out == pytest.approx(math.atan(10.0 * WHEELBASE / SPEED ** 2), abs=1e-9)   # rad


def test_wanted_at_or_above_the_last_asked_equals_the_clipped_input():
    c = example_curve()
    for wanted in (10.0, 12.0, 30.0):
        s = steering_for_lateral(wanted)
        out = rm.understeer_steering(s, SPEED, WHEELBASE, c, 1.4)
        assert out == pytest.approx(min(s, 1.4), abs=1e-12)   # rad
        assert rm.understeer_steering(-s, SPEED, WHEELBASE, c, 1.4) == pytest.approx(-min(s, 1.4), abs=1e-12)
    huge = rm.understeer_steering(1.5, SPEED, WHEELBASE, c, MAX_STEER)
    assert huge == pytest.approx(MAX_STEER, abs=1e-12)   # still clipped


def test_compensation_never_steers_less_than_the_clipped_input():
    # Invariant over 200 seeded inputs: the curve only ever asks for more steering.
    rng = random.Random(20261112)
    c = example_curve()
    for _ in range(200):
        s = rng.uniform(-0.6, 0.6)
        speed = rng.uniform(1.0, 6.0)
        out = rm.understeer_steering(s, speed, WHEELBASE, c, MAX_STEER)
        clipped = max(-MAX_STEER, min(MAX_STEER, s))
        assert abs(out) >= abs(clipped) - 1e-12   # rad
        assert out * clipped >= 0.0


def test_non_finite_input_never_returns_nan():
    c = example_curve()
    assert rm.understeer_steering(float('nan'), SPEED, WHEELBASE, c, MAX_STEER) == 0.0
    assert rm.understeer_steering(0.2, float('nan'), WHEELBASE, c, MAX_STEER) == pytest.approx(0.2, abs=1e-12)
    assert math.isfinite(rm.understeer_steering(float('inf'), SPEED, WHEELBASE, c, MAX_STEER))


# --- delivered_curvature_speed_limit (Task 7) ----------------------------------------

def test_no_curve_equals_curvature_speed_limit_exactly():
    rng = random.Random(7)
    for _ in range(100):
        k = rng.uniform(-2.0, 2.0)
        lat = rng.uniform(1.0, 5.0)
        vmax = rng.uniform(2.0, 6.0)
        assert rm.delivered_curvature_speed_limit(k, lat, vmax, None) == rm.curvature_speed_limit(k, lat, vmax)


def test_identity_curve_equals_curvature_speed_limit():
    c = rm.GripCurve(asked=(0.0, 10.0), delivered=(0.0, 10.0))
    rng = random.Random(8)
    for _ in range(100):
        k = rng.uniform(0.05, 2.0)
        lat = rng.uniform(1.0, 5.0)
        assert rm.delivered_curvature_speed_limit(k, lat, 6.0, c) == pytest.approx(
            rm.curvature_speed_limit(k, lat, 6.0), abs=1e-9)  # m/s


def test_the_speed_is_where_the_delivered_lateral_reaches_the_limit():
    # curvature 0.5/m, limit 3.1: asked = ask_for(3.1) = 2.5 + 0.6 * 2.5 = 4.0 -> v = sqrt(4.0 / 0.5)
    c = example_curve()
    expected = math.sqrt(c.ask_for(3.1) / 0.5)
    assert c.ask_for(3.1) == pytest.approx(4.0, abs=1e-12)
    assert rm.delivered_curvature_speed_limit(0.5, 3.1, 10.0, c) == pytest.approx(expected, abs=1e-6)  # m/s
    # and it really delivers the limit there
    v = rm.delivered_curvature_speed_limit(0.5, 3.1, 10.0, c)
    assert c.deliver(v * v * 0.5) == pytest.approx(3.1, abs=1e-9)  # m/s^2


def test_negative_curvature_uses_its_magnitude():
    c = example_curve()
    assert rm.delivered_curvature_speed_limit(-0.5, 3.1, 10.0, c) == pytest.approx(
        rm.delivered_curvature_speed_limit(0.5, 3.1, 10.0, c), abs=1e-12)


def test_zero_curvature_returns_max_speed():
    assert rm.delivered_curvature_speed_limit(0.0, 3.1, 4.5, example_curve()) == 4.5


def test_the_result_never_exceeds_max_speed():
    assert rm.delivered_curvature_speed_limit(0.01, 3.1, 4.5, example_curve()) == 4.5


def test_a_very_tight_curve_stays_on_the_first_segment():
    # curvature 20/m, limit 0.5: asked = ask_for(0.5) = 0.5 (slope 1 on the first segment)
    assert rm.delivered_curvature_speed_limit(20.0, 0.5, 10.0, example_curve()) == pytest.approx(
        math.sqrt(0.5 / 20.0), abs=1e-9)  # m/s


def test_a_curve_that_never_delivers_the_limit_caps_the_asked_lateral_at_its_last_asked():
    # delivered tops out at 2.5 < max_lateral_accel 3.1: the speed still has to be bounded, so the
    # ASKED lateral is limited to the curve's last asked value (advisor addition at checkpoint 2).
    c = rm.GripCurve(asked=(0.0, 2.0, 6.0), delivered=(0.0, 2.0, 2.5))
    assert rm.delivered_curvature_speed_limit(0.5, 3.1, 10.0, c) == pytest.approx(
        math.sqrt(6.0 / 0.5), abs=1e-6)  # m/s


def test_nan_curvature_returns_zero_speed():
    assert rm.delivered_curvature_speed_limit(float('nan'), 3.1, 4.5, example_curve()) == 0.0
    assert rm.delivered_curvature_speed_limit(float('nan'), 3.1, 4.5, None) == 0.0
