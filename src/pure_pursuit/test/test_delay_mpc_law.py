"""Delay MPC oracles: closed-form geometry, Euler servo and hard constraints.

Run by python3 -m pytest src/pure_pursuit/test/ with ROS sourced.
"""
from dataclasses import fields, replace
import math

import numpy as np
import pytest
import yaml

from pure_pursuit import tracking_laws as laws
from test_delay_lqr_law import (
    LIMITS, DT, L, straight_path, circle_path, CONFIG, riccati_gain)


def make(**params):
    # Math tests are independent of scheduler load; deadline is tested separately.
    return laws.make_law('delay_mpc', LIMITS, {
        'delay_mpc.deadline_sec': 10.,
        **{'delay_mpc.' + k: v for k, v in params.items()}})


def inp(**changes):
    return replace(laws.TrackingInput(1., 0., 0., 2., 50, 0.), **changes)


def test_straight_zero_and_signed_offset():
    # Symmetry gives zero at equilibrium; an offset requires opposite steering.
    law, path = make(), straight_path()
    assert law.steer(inp(), path).steering == pytest.approx(0., abs=1e-10)  # rad
    for y in (-0.1, 0.1):
        command = law.steer(inp(y=y), path).steering
        assert command * y < 0.


@pytest.mark.parametrize('radius', [2., -2.])
def test_circle_equilibrium(radius):
    path = circle_path(abs(radius), n=12000)
    if radius < 0:
        path = replace(path, xy=path.xy * [1., -1.], curvature=-path.curvature)
    ff = math.atan(L / radius)
    result = make().steer(inp(x=abs(radius), y=0., yaw=math.copysign(math.pi / 2, radius),
                              nearest_idx=0, previous_steering=ff), path)
    # Bicycle circle: delta = atan(L/R); discretized chord heading adds <0.001 rad.
    assert result.steering == pytest.approx(ff, abs=1e-3)  # rad


def test_step_demand_activates_rate_and_angle_constraints():
    law = make()
    result = law.steer(inp(y=2.), straight_path())
    plan = np.asarray(result.predicted)
    assert plan[0] == pytest.approx(-LIMITS.max_steering_rate * DT, abs=1e-8)  # rad
    tau = np.full(len(plan), law.params.dt)
    tau[0] = DT
    assert np.all(np.abs(np.diff(np.r_[0., plan])) <= LIMITS.max_steering_rate * tau + 1e-12)
    assert np.max(np.abs(plan)) <= LIMITS.max_steering_angle
    assert np.min(plan) == pytest.approx(-LIMITS.max_steering_angle, abs=1e-4)  # rad
    # Check the optimizer's raw moves too: output projection alone must not
    # disguise a QP that ignored command-rate constraints.
    raw = law._warm_start
    assert np.all(np.abs(np.diff(np.r_[0., raw])) <= LIMITS.max_steering_rate * tau + 1e-7)


def test_unconstrained_straight_matches_independent_servo_lqr():
    # The DARE terminal cost makes finite MPC equal infinite LQR on a constant
    # straight. Oracle: independent fixed-point Riccati iteration in the LQR tests.
    law = make()
    error = .001  # m; command is below the first rate bound
    result = law.steer(inp(y=error), straight_path())
    expected = -riccati_gain(2., q_lateral=law.params.q_lateral,
                            q_heading=law.params.q_heading,
                            r_steer=law.params.r_steer)[0] * error
    assert result.steering == pytest.approx(expected, abs=1e-8)  # rad


def test_action_initial_state_matches_closed_form_rollout():
    law = make()
    speed, angle, yaw = 2., 0.1, 0.12
    law.steer(inp(yaw=yaw, previous_steering=angle), straight_path())
    kappa = math.tan(angle) / L
    yaw_action = yaw + speed * 5 * DT * kappa
    e_action = (math.cos(yaw) - math.cos(yaw_action)) / kappa
    np.testing.assert_allclose(law.initial_state, [e_action, yaw_action, angle],
                               rtol=0., atol=1e-12)  # m, rad, rad


def test_servo_model_and_affine_curvature_match_euler_oracle():
    law = make()
    v, ff = 3., 0.2
    A, B = law._ab(v, ff)
    x, command = np.array([0.1, -0.03, 0.15]), 0.18
    actual = A @ x + B * command
    actual[1] -= A[1, 2] * ff
    dt = law.params.dt
    expected = [x[0] + v * dt * x[1],
                x[1] + v * dt / L / math.cos(ff)**2 * (x[2] - ff),
                x[2] + 15. * dt * (command - x[2])]
    np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-14)


def test_reference_advances_at_profile_speeds():
    path = straight_path()
    # v(s)=1+s/10 has an analytic Euler recurrence, distinct from measured v=2.
    path = replace(path, speed_profile=1 + path.cumulative / 10.,
                   curvature=path.cumulative / 100.)
    law = make()
    law.steer(inp(), path)
    s = 1. + 2. * 5 * DT
    for v, ff in zip(law.reference_speeds, law.reference_ff):
        assert v == pytest.approx(1 + s / 10., abs=1e-12)  # m/s
        assert ff == pytest.approx(math.atan(L * s / 100.), abs=1e-12)  # rad
        s += (1 + s / 10.) * law.params.dt


@pytest.mark.parametrize('field', ['x', 'y', 'yaw', 'speed', 'previous_steering'])
@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_nonfinite_input_is_a_law_error(field, value):
    with pytest.raises(laws.TrackingLawError, match='non-finite'):
        make().steer(inp(**{field: value}), straight_path())


@pytest.mark.parametrize('key, value', [
    ('dt', 0.), ('dt', .21), ('horizon_steps', 1), ('horizon_steps', 61),
    ('q_lateral', 0.), ('q_heading', -1.), ('r_steer', math.nan),
    ('r_rate', -1.), ('deadline_sec', 0.), ('max_iter', 0),
    ('max_consecutive_misses', 0), ('sensing_delay', -.01),
    ('actuation_delay', .51), ('servo_gain', 41.), ('servo_rate', 0.)])
def test_invalid_parameters_are_refused(key, value):
    with pytest.raises(ValueError, match=key):
        make(**{key: value})


def test_deadline_miss_discards_warm_start():
    law = make()
    law.steer(inp(y=.1), straight_path())
    assert law._warm_start is not None
    law.params = replace(law.params, deadline_sec=1e-9)
    with pytest.raises(laws.MpcMiss, match='deadline'):
        law.steer(inp(y=.1), straight_path())
    assert law._warm_start is None


def test_yaml_and_factory_cover_every_parameter():
    with open(CONFIG) as handle:
        cfg = yaml.safe_load(handle)['pure_pursuit_node']['ros__parameters']
    defaults = laws.DelayMpcParams()
    assert {f.name: cfg['delay_mpc.' + f.name] for f in fields(defaults)} == vars(defaults)
    given = dict(dt=.04, horizon_steps=23, q_lateral=2., q_heading=3., r_steer=4.,
                 r_rate=.002, deadline_sec=.007, max_iter=2300, max_consecutive_misses=4,
                 sensing_delay=.025, actuation_delay=.1, servo_gain=12., servo_rate=2.)
    law = make(**given)
    assert vars(law.params) == given


def test_reset_forgets_plan_and_history():
    law = make()
    result = law.steer(inp(y=.2), straight_path())
    assert result.steering < 0.
    law.reset()
    assert law._warm_start is None
    assert law.solves_since_reset == 0
    assert not law._delay._history
