"""Unit tests for DelayLqrLaw and ServoLqrGainTable (pure_pursuit.tracking_laws).

ROS-free. Run with:

    python3 -m pytest src/pure_pursuit/test/test_delay_lqr_law.py -v

Every expected value is a closed form worked out in the test, an invariant, or an
independent recomputation: the LQR gain is checked against a Riccati recursion iterated
to its fixed point (not scipy's solver the code uses), and the servo against its own
geometric step response. Each test was checked against deliberate breaks of the code
it covers, and was seen to fail against at least one of them.
"""
import math
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pure_pursuit import racing_math, tracking_laws  # noqa: E402
from pure_pursuit.tracking_laws import (  # noqa: E402
    DelayLqrParams, PathContext, ServoLqrGainTable, TrackingInput,
    TrackingLawError, VehicleLimits)

PACKAGE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG = os.path.join(PACKAGE_DIR, 'config', 'pure_pursuit.yaml')

L = 0.36           # m, measured wheelbase (docs/hardware-reference.md)
DT = 0.025         # s, 40 Hz control tick
SERVO_GAIN = 15.0  # 1/s, the car 2 floor's steering_servo_gain (sim_fidelity/calibration.py)
LIMITS = VehicleLimits(wheelbase=L, max_steering_angle=0.42, max_steering_rate=1.0,
                       control_dt=DT)
# The same limits with the command slew out of the way, for tests that isolate the feedback.
UNSLEWED = VehicleLimits(wheelbase=L, max_steering_angle=0.42, max_steering_rate=100.0,
                         control_dt=DT)


def riccati_gain(v, q_lateral=1.0, q_heading=1.0, r_steer=1.0):
    """K(v) for the servo error model, by iterating the Riccati recursion to its fixed
    point. Independent of scipy's solve_discrete_are that ServoLqrGainTable uses."""
    a = SERVO_GAIN * DT
    A = np.array([[1.0, v * DT, 0.0], [0.0, 1.0, v * DT / L], [0.0, 0.0, 1.0 - a]])
    B = np.array([[0.0], [0.0], [a]])
    Q = np.diag([q_lateral, q_heading, 0.0])
    R = np.array([[r_steer]])
    P = Q.copy()
    for _ in range(50000):
        S = R + B.T @ P @ B
        P_next = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(S, B.T @ P @ A)
        if np.max(np.abs(P_next - P)) < 1e-13:
            P = P_next
            break
        P = P_next
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)[0]


def path_from_curvature(kappa_of_s, n, ds):
    """An open path, `ds` apart, whose curvature at arc length s is kappa_of_s(s).
    The curvature array is the exact value at each waypoint (no estimate)."""
    s = np.arange(n) * ds
    kappa = np.array([kappa_of_s(v) for v in s], dtype=float)
    psi = np.concatenate([[0.0], np.cumsum(kappa[:-1] * ds)])
    x = np.concatenate([[0.0], np.cumsum(np.cos(psi[:-1]) * ds)])
    y = np.concatenate([[0.0], np.cumsum(np.sin(psi[:-1]) * ds)])
    xy = np.column_stack([x, y])
    seg = racing_math.compute_segment_lengths(xy, closed=False)
    return PathContext(xy=xy, seg_len=seg,
                       cumulative=racing_math.compute_cumulative_arc_length(seg),
                       speed_profile=np.full(n, 2.0), curvature=kappa, closed=False)


def straight_path(n=400, ds=0.02):
    return path_from_curvature(lambda s: 0.0, n, ds)


def circle_path(radius, n=4000):
    """A closed circle, counter-clockwise (+ curvature), exact curvature 1/radius."""
    ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    xy = np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])
    seg = racing_math.compute_segment_lengths(xy, closed=True)
    return PathContext(xy=xy, seg_len=seg,
                       cumulative=racing_math.compute_cumulative_arc_length(seg),
                       speed_profile=np.full(n, 2.0),
                       curvature=np.full(n, 1.0 / radius), closed=True)


def nearest_index(ctx, x, y):
    """The index of the waypoint nearest the sensed pose, as the node passes it."""
    return int(racing_math.find_nearest_index(ctx.xy, (x, y), closed=ctx.closed)[0])


def make(params=None, limits=LIMITS):
    return tracking_laws.make_law('delay_lqr', limits, dict(params or {}))


def feedforward(curvature):
    return math.atan(L * curvature)


# ============================================================================
# Factory, parameters, and the car's delays
# ============================================================================

def test_factory_builds_the_named_law_with_the_dataclass_defaults():
    law = make()
    assert law.name == 'delay_lqr'
    assert 'delay_lqr' in tracking_laws.LAW_NAMES
    assert law.params == DelayLqrParams()


def test_the_factory_passes_every_parameter_through():
    # Non-default values for all eight parameters must reach the law's params object.
    given = {'delay_lqr.q_lateral': 2.0, 'delay_lqr.q_heading': 3.0,
             'delay_lqr.r_steer': 4.0, 'delay_lqr.v_grid_max': 5.0,
             'delay_lqr.sensing_delay': 0.04, 'delay_lqr.actuation_delay': 0.06,
             'delay_lqr.servo_gain': 12.0, 'delay_lqr.servo_rate': 2.0}
    assert make(given).params == DelayLqrParams(
        q_lateral=2.0, q_heading=3.0, r_steer=4.0, v_grid_max=5.0,
        sensing_delay=0.04, actuation_delay=0.06, servo_gain=12.0, servo_rate=2.0)


def test_the_delays_are_the_car_2_floor_ticks_at_40_hz():
    # tools/f1tenth_sim/sim_fidelity/calibration.py, sfu_course_2026_11_06 floor:
    # pose_delay_ticks 2 (default), steering_delay_ticks 3. Ticks at 40 Hz.
    law = make()
    assert law.sensing_ticks == 2
    assert law.actuation_ticks == 3
    assert law.horizon_ticks == 5


def test_yaml_mirrors_the_delay_lqr_dataclass_defaults():
    with open(CONFIG) as handle:
        cfg = yaml.safe_load(handle)['pure_pursuit_node']['ros__parameters']
    defaults = DelayLqrParams()
    assert cfg['delay_lqr.q_lateral'] == pytest.approx(defaults.q_lateral)
    assert cfg['delay_lqr.q_heading'] == pytest.approx(defaults.q_heading)
    assert cfg['delay_lqr.r_steer'] == pytest.approx(defaults.r_steer)
    assert cfg['delay_lqr.v_grid_max'] == pytest.approx(defaults.v_grid_max)
    assert cfg['delay_lqr.sensing_delay'] == pytest.approx(defaults.sensing_delay)
    assert cfg['delay_lqr.actuation_delay'] == pytest.approx(defaults.actuation_delay)
    assert cfg['delay_lqr.servo_gain'] == pytest.approx(defaults.servo_gain)
    assert cfg['delay_lqr.servo_rate'] == pytest.approx(defaults.servo_rate)


def test_a_servo_step_above_one_per_tick_is_refused():
    # Euler servo: gain * dt must be <= 1, or the discrete servo overshoots.
    with pytest.raises(ValueError, match='servo_gain'):
        ServoLqrGainTable(1.0, 1.0, 1.0, DT, L, servo_gain=50.0, v_grid_max=6.0)


def test_delay_outside_the_allowed_range_is_refused():
    with pytest.raises(ValueError, match='actuation_delay'):
        make({'delay_lqr.actuation_delay': 0.9})


# ============================================================================
# The gain schedule
# ============================================================================

@pytest.mark.parametrize('v', [0.5, 2.0, 5.0])
def test_gain_matches_an_independent_riccati_recursion(v):
    table = ServoLqrGainTable(1.0, 1.0, 1.0, DT, L, SERVO_GAIN, 6.0)
    np.testing.assert_allclose(table.gain_at(v), riccati_gain(v), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize('q_lateral', [1.0, 4.0])
def test_gain_follows_the_weights(q_lateral):
    # A heavier lateral weight must change the gain, and the oracle must track it.
    table = ServoLqrGainTable(q_lateral, 1.0, 1.0, DT, L, SERVO_GAIN, 6.0)
    np.testing.assert_allclose(table.gain_at(3.0),
                               riccati_gain(3.0, q_lateral=q_lateral), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize('v', [0.1, 0.5, 1.5, 3.0, 5.0, 6.0])
def test_the_closed_loop_is_stable_at_every_speed(v):
    # Invariant: the error model under the gain has all eigenvalues inside the unit circle.
    table = ServoLqrGainTable(1.0, 1.0, 1.0, DT, L, SERVO_GAIN, 6.0)
    A = np.array([[1.0, v * DT, 0.0], [0.0, 1.0, v * DT / L],
                  [0.0, 0.0, 1.0 - SERVO_GAIN * DT]])
    B = np.array([[0.0], [0.0], [SERVO_GAIN * DT]])
    A_cl = A - B @ table.gain_at(v).reshape(1, 3)
    assert np.max(np.abs(np.linalg.eigvals(A_cl))) < 1.0


# ============================================================================
# Steering on known geometry
# ============================================================================

def test_straight_on_the_line_is_zero_steering():
    ctx = straight_path()
    result = make().steer(TrackingInput(x=1.0, y=0.0, yaw=0.0, speed=2.0,
                                        nearest_idx=nearest_index(ctx, 1.0, 0.0),
                                        previous_steering=0.0), ctx)
    assert result.steering == pytest.approx(0.0, abs=1e-12)


def test_on_a_constant_circle_the_steering_is_the_feedforward():
    # Oracle: on a circle of radius R at steady steering, the wheel angle that keeps the car
    # on it is atan(L / R). The chord heading of the discrete circle biases the heading error
    # by kappa * ds / 2 (about 4e-4 rad at 4000 points), so the bound is 2e-3 rad.
    radius = 2.0
    ctx = circle_path(radius)
    law = make(limits=UNSLEWED)
    x, y, yaw = radius, 0.0, math.pi / 2.0
    steady = math.atan(L / radius)
    result = None
    for _ in range(60):
        result = law.steer(TrackingInput(x=x, y=y, yaw=yaw, speed=2.0,
                                         nearest_idx=nearest_index(ctx, x, y),
                                         previous_steering=steady), ctx)
    assert result.steering == pytest.approx(steady, abs=2e-3)


def test_lateral_error_is_corrected_by_the_gain_with_the_right_sign():
    # Car 0.1 m left of a straight line, heading along it, no steering: the feedback is
    # -K0 * e, and the line is to the right (steer negative). The slew is out of the way.
    v = 2.0
    ctx = straight_path()
    law = make(limits=UNSLEWED)
    result = law.steer(TrackingInput(x=1.0, y=0.1, yaw=0.0, speed=v,
                                     nearest_idx=nearest_index(ctx, 1.0, 0.1),
                                     previous_steering=0.0), ctx)
    expected = -riccati_gain(v)[0] * 0.1
    assert result.steering == pytest.approx(expected, abs=1e-9)
    assert result.steering < 0.0


def test_heading_error_steers_for_the_pose_at_the_action_not_the_pose_sensed():
    # The pose is 2 ticks old and the command acts 3 ticks later: 5 ticks of straight
    # driving (the wheels are at zero). At heading +0.1 the car drifts v * 5 * dt * sin(0.1)
    # to the left before the command acts, so the error the law corrects is that drift
    # plus the heading error. The oracle is closed form.
    v = 2.0
    theta = 0.1
    e_action = v * 5 * DT * math.sin(theta)
    ctx = straight_path()
    law = make(limits=UNSLEWED)
    result = law.steer(TrackingInput(x=1.0, y=0.0, yaw=theta, speed=v,
                                     nearest_idx=nearest_index(ctx, 1.0, 0.0),
                                     previous_steering=0.0), ctx)
    K = riccati_gain(v)
    expected = -(K[0] * e_action + K[1] * theta)
    assert result.steering == pytest.approx(expected, abs=1e-9)


def test_the_feedforward_is_read_one_servo_time_constant_past_the_action_point():
    # Straight to s* = 1.0 m, then a bend of curvature kappa1 (R1 = 2 m, so the expected
    # steering stays inside the 0.42 rad limit). The car starts at 0.7 m, 0.3 m before the
    # bend: the action point is 0.25 m on (still straight, arc 0.95) and the servo's time
    # constant (1 / 15 s) adds 2 * 0.0667 = 0.133 m, which lands on the bend (arc 1.08).
    # With the wheels at zero the feedback is -K2 * (0 - ff), so the output is ff * (1 + K2).
    v = 2.0
    kappa1 = 0.5
    ctx = path_from_curvature(lambda s: kappa1 if s >= 1.0 else 0.0, 400, 0.01)
    start = 0.7
    law = make(limits=UNSLEWED)
    result = law.steer(TrackingInput(x=start, y=0.0, yaw=0.0, speed=v,
                                     nearest_idx=nearest_index(ctx, start, 0.0),
                                     previous_steering=0.0), ctx)
    K = riccati_gain(v)
    expected = feedforward(kappa1) * (1.0 + K[2])
    assert result.steering == pytest.approx(expected, abs=1e-6)


def test_the_servo_state_is_the_first_order_response_to_the_command():
    # The law's own command u1 is applied one tick ago and only zeros before it. The servo
    # sees it 3 ticks after it was issued (the transport), so at the action point the wheel
    # has moved only by one Euler step of the servo: delta = g * dt * u1 (below the rate cap).
    # Then w = delta and the feedback on a straight line at zero error is -K2 * delta.
    # (2026-11-07: u1 is now the law's own output. A previous_steering the law did not
    # return means another path published in between, and the law re-seeds; see
    # test_a_command_the_law_did_not_return_reseeds_the_history.)
    v = 2.0
    ctx = straight_path()
    law = make(limits=UNSLEWED)
    idle = TrackingInput(x=1.0, y=0.0, yaw=0.0, speed=v,
                         nearest_idx=nearest_index(ctx, 1.0, 0.0), previous_steering=0.0)
    for _ in range(50):
        assert law.steer(idle, ctx).steering == pytest.approx(0.0, abs=1e-12)
    # one tick 2 cm to the right of the line: the law steers left, u1 > 0
    u1 = law.steer(TrackingInput(x=1.0, y=-0.02, yaw=0.0, speed=v,
                                 nearest_idx=nearest_index(ctx, 1.0, -0.02),
                                 previous_steering=0.0), ctx).steering
    assert 0.0 < u1 < 0.42 and SERVO_GAIN * u1 < 2.5   # off the clamp and under the rate cap
    result = law.steer(TrackingInput(x=1.0, y=0.0, yaw=0.0, speed=v,
                                     nearest_idx=nearest_index(ctx, 1.0, 0.0),
                                     previous_steering=u1), ctx)
    delta_at_action = SERVO_GAIN * DT * u1
    expected = -riccati_gain(v)[2] * delta_at_action
    assert result.steering == pytest.approx(expected, abs=1e-9)


def test_a_command_the_law_did_not_return_reseeds_the_history():
    # A stop, recovery or escape published in between: previous_steering is not what the
    # law returned last tick. The law must not replay its own stale history as recent; it
    # assumes the rack has held the applied command. Oracle (invariant): the result equals a
    # law that never ran, given the same input, and is NOT the stale replay, in which the
    # rack would only have taken one servo step towards the command (-K2 * g * dt * applied).
    v = 2.0
    ctx = straight_path()
    law = make(limits=UNSLEWED)
    idle = TrackingInput(x=1.0, y=0.0, yaw=0.0, speed=v,
                         nearest_idx=nearest_index(ctx, 1.0, 0.0), previous_steering=0.0)
    for _ in range(50):
        law.steer(idle, ctx)
    applied = 0.1
    result = law.steer(TrackingInput(x=1.0, y=0.0, yaw=0.0, speed=v,
                                     nearest_idx=nearest_index(ctx, 1.0, 0.0),
                                     previous_steering=applied), ctx)
    stale = -riccati_gain(v)[2] * SERVO_GAIN * DT * applied
    assert abs(result.steering - stale) > 0.05   # rad: far from the stale-history replay
    # ...and it is the same as a law that never ran before
    fresh = make(limits=UNSLEWED).steer(TrackingInput(
        x=1.0, y=0.0, yaw=0.0, speed=v, nearest_idx=nearest_index(ctx, 1.0, 0.0),
        previous_steering=applied), ctx)
    assert result.steering == pytest.approx(fresh.steering, abs=1e-12)


def test_a_servo_too_slow_for_the_estimate_window_is_refused():
    # Lock to lock at 0.1 rad/s is 0.84 rad / 0.1 = 8.4 s = 336 ticks, far past the 40-tick
    # window the wheel estimate is rebuilt over: refused at construction, not mispredicted.
    with pytest.raises(ValueError, match='needs .* ticks to settle'):
        make({'delay_lqr.servo_rate': 0.1})
    # the car's own servo (15/s, 2.5 rad/s): 0.84/2.5/0.025 = 13.4 + ln(1e-3)/ln(0.625) = 14.7 -> 28 ticks
    make({'delay_lqr.servo_rate': 2.5, 'delay_lqr.servo_gain': 15.0})


def test_sample_at_arc_wraps_at_the_true_seam_of_a_closed_line():
    # Three waypoints on a closed line, segments 1, 2 and 3 m (the last closes back to 0):
    # the lap is 6 m. Half way along the closing segment (s = 4.5) is half way from the last
    # waypoint's value to the first's; s = 6.5 wraps to 0.5.
    values = np.array([0.0, 10.0, 20.0])
    seg = np.array([1.0, 2.0, 3.0])
    cum = racing_math.compute_cumulative_arc_length(seg)
    assert tracking_laws.sample_at_arc(values, cum, 4.5, True, seg) == pytest.approx(10.0, abs=1e-12)
    assert tracking_laws.sample_at_arc(values, cum, 6.5, True, seg) == pytest.approx(5.0, abs=1e-12)
    assert tracking_laws.sample_at_arc(values, cum, 2.0, True, seg) == pytest.approx(15.0, abs=1e-12)


# ============================================================================
# Limits, edge inputs and state
# ============================================================================

def test_steering_is_clamped_to_the_angle_limit():
    # 1 m to the left: the feedback alone is far past the 0.42 rad limit.
    ctx = straight_path()
    law = make(limits=UNSLEWED)
    result = law.steer(TrackingInput(x=1.0, y=1.0, yaw=0.0, speed=2.0,
                                     nearest_idx=nearest_index(ctx, 1.0, 1.0),
                                     previous_steering=0.0), ctx)
    assert result.steering == pytest.approx(-0.42, abs=1e-12)
    assert result.unclipped < -0.42


def test_the_command_moves_no_faster_than_the_shaper_slew():
    # The output is slew-limited to max_steering_rate * dt from the last command, the same
    # limit the command shaper applies. A 0.2 m error asks for far more than that in one tick.
    ctx = straight_path()
    law = make(limits=LIMITS)
    result = law.steer(TrackingInput(x=1.0, y=0.2, yaw=0.0, speed=2.0,
                                     nearest_idx=nearest_index(ctx, 1.0, 0.2),
                                     previous_steering=0.0), ctx)
    assert abs(result.steering - 0.0) == pytest.approx(1.0 * DT, abs=1e-12)


@pytest.mark.parametrize('field', ['x', 'y', 'yaw', 'speed', 'previous_steering'])
def test_a_non_finite_input_raises_a_law_error(field):
    values = dict(x=1.0, y=0.0, yaw=0.0, speed=2.0, nearest_idx=40, previous_steering=0.0)
    values[field] = math.nan
    with pytest.raises(TrackingLawError, match='non-finite'):
        make().steer(TrackingInput(**values), straight_path())


def test_zero_speed_uses_the_slowest_gain_and_stays_finite():
    # At rest the car does not move in the prediction; the gain is the grid's floor.
    ctx = straight_path()
    law = make(limits=UNSLEWED)
    result = law.steer(TrackingInput(x=1.0, y=0.1, yaw=0.0, speed=0.0,
                                     nearest_idx=nearest_index(ctx, 1.0, 0.1),
                                     previous_steering=0.0), ctx)
    expected = -riccati_gain(ServoLqrGainTable.V_FLOOR)[0] * 0.1
    assert result.steering == pytest.approx(expected, abs=1e-9)


def test_reset_forgets_the_command_history():
    # After 50 ticks at a 0.2 rad command, a reset must make the law behave as a new one.
    ctx = straight_path()
    used = make(limits=UNSLEWED)
    loaded = TrackingInput(x=1.0, y=0.0, yaw=0.0, speed=2.0,
                           nearest_idx=nearest_index(ctx, 1.0, 0.0), previous_steering=0.2)
    for _ in range(50):
        used.steer(loaded, ctx)
    used.reset()
    probe = TrackingInput(x=1.0, y=0.0, yaw=0.0, speed=2.0,
                          nearest_idx=nearest_index(ctx, 1.0, 0.0), previous_steering=0.0)
    fresh = make(limits=UNSLEWED)
    assert used.steer(probe, ctx).steering == pytest.approx(
        fresh.steer(probe, ctx).steering, abs=1e-12)
