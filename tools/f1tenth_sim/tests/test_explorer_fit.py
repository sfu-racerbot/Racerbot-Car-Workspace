"""explorer_fit: the fitter behind the explorer (docs/explorer.md).

The main test is the planted answer: drive the plant with explorer-like
commands using numbers we chose, hand the recording to the fitter starting
from the car's ordinary calibration, and require it to find those numbers
again. The oracle is the planted values themselves, which the fitter never
sees. Run with the tool suites:

    python3 -m pytest tools/f1tenth_sim/tests/test_explorer_fit.py -q
"""

from dataclasses import replace
from functools import lru_cache
import math

import numpy as np
import pytest

from sim_fidelity import explorer_fit as ef
from sim_fidelity.calibration import calibration_for_car

PLANTED_CAR = 'rb2'


def explorer_commands():
    """24 s of what the explorer asks for: three speed blocks, each ending in
    a hard brake, with full-ish steering swings throughout."""
    t = np.arange(0, 24, ef.DT)
    speed = np.select([t < 6, t < 12, t < 18], [2.0, 3.0, 4.0], 0.0)
    speed[(t % 6) > 4.5] = 0.5
    steer = 0.25 * np.sign(np.sin(2 * np.pi * t / 1.5))
    return t, np.column_stack([steer, speed])


@lru_cache(maxsize=1)
def planted_fit():
    """One fit, shared by both cars' parametrisations of this suite."""
    from run_analysis.fit_sim import simulate_commands
    base = calibration_for_car(PLANTED_CAR)
    truth = replace(base, surface_friction=0.42,
                    steering_delay_ticks=base.steering_delay_ticks + 2,
                    speed_loop_brake_max=3.0)
    t, cmd = explorer_commands()
    v, yaw = simulate_commands(cmd, truth, initial_speed=0.0)
    windows = ef.windows_from_series(t, cmd[:, 0], cmd[:, 1], v, yaw,
                                     np.ones(len(t), bool), horizon=2.0, spacing=2.0)
    result = ef.fit(base, windows, ['surface_friction', 'steering_delay_ticks',
                                    'speed_loop_brake_max'],
                    maxiter=12, popsize=6, workers=4)
    return base, truth, result


def test_the_fitter_finds_planted_numbers_it_never_saw():
    base, truth, result = planted_fit()
    # The start is far from the answer, so passing is not "it stayed put".
    assert abs(base.surface_friction - truth.surface_friction) > 0.1
    assert result.values['steering_delay_ticks'] == truth.steering_delay_ticks
    assert result.values['surface_friction'] == pytest.approx(truth.surface_friction, rel=0.10)  # 10% of grip
    assert result.values['speed_loop_brake_max'] == pytest.approx(
        truth.speed_loop_brake_max, rel=0.10)  # 10% of braking
    assert result.cost_after < 0.6 * result.cost_before
    assert result.at_bound == []


def test_every_planted_number_is_pinned_down_by_the_data():
    _, _, result = planted_fit()
    # A step of 10% (or one tick) must cost at least 2% of a typical error.
    for name, rise in result.sensitivity.items():
        assert rise > 0.02, name


@pytest.mark.parametrize('name', sorted(ef.FIXED))
def test_measured_numbers_are_refused(name):
    with pytest.raises(ValueError, match='measured, never fitted'):
        ef.check_names(['surface_friction', name])


def test_unknown_empty_and_repeated_names_are_refused():
    with pytest.raises(ValueError, match='not a fittable parameter'):
        ef.check_names(['grip'])
    with pytest.raises(ValueError, match='nothing to fit'):
        ef.check_names([])
    with pytest.raises(ValueError, match='named twice'):
        ef.check_names(['surface_friction', 'surface_friction'])


def test_no_parameter_is_both_fittable_and_measured():
    assert not set(ef.PARAMETERS) & ef.FIXED


def test_every_fittable_name_is_a_calibration_field():
    cal = calibration_for_car(PLANTED_CAR)
    for name in ef.PARAMETERS:
        assert hasattr(cal, name), name


def series(n=200, speed=2.0):
    t = np.arange(n) * ef.DT
    return dict(t=t, steer=np.zeros(n), speed_cmd=np.full(n, speed),
                speed=np.full(n, speed), yaw_rate=np.zeros(n), good=np.ones(n, bool))


def test_windows_tile_a_clean_run_at_the_requested_spacing():
    s = series(n=8 + 4 * 40)  # 8 prior ticks, then 4 s
    w = ef.windows_from_series(**s, horizon=1.0, spacing=1.0)
    assert len(w) == 4
    assert all(len(x.commands) == 40 for x in w)
    assert all(len(x.prior_speed_commands) == 8 for x in w)


def test_a_window_never_spans_an_untrusted_tick():
    s = series(n=8 + 4 * 40)
    s['speed'] = 2.0 + 0.001 * np.arange(len(s['t']))  # speed tags each tick's index
    bad = 8 + 60  # inside the second second
    s['good'][bad] = False
    w = ef.windows_from_series(**s, horizon=1.0, spacing=1.0)
    starts = [int(round((x.speed[0] - 2.0) / 0.001)) for x in w]
    # 8..48 is clean; the next clean start after the bad tick is 69, then 109.
    assert starts == [8, bad + 1, bad + 41]
    assert all(not (st <= bad < st + 40) for st in starts)


def test_a_window_needs_the_car_moving_at_its_start():
    s = series(speed=0.2)
    assert ef.windows_from_series(**s, horizon=1.0, min_speed=0.5) == []


def test_nan_in_a_window_drops_it():
    s = series(n=8 + 40)
    s['yaw_rate'][20] = math.nan
    assert ef.windows_from_series(**s, horizon=1.0) == []


def test_mismatched_lengths_and_too_short_a_horizon_are_refused():
    s = series()
    s['speed'] = s['speed'][:-1]
    with pytest.raises(ValueError, match='same length'):
        ef.windows_from_series(**s)
    with pytest.raises(ValueError, match='shorter than two ticks'):
        ef.windows_from_series(**series(), horizon=0.03)


def test_apply_clamps_to_the_search_range_and_rounds_delays():
    cal = calibration_for_car(PLANTED_CAR)
    _, values = ef.apply(cal, ['surface_friction', 'steering_delay_ticks'], [9.0, 2.6])
    assert values == {'surface_friction': ef.PARAMETERS['surface_friction'][1],
                      'steering_delay_ticks': 3}
    _, values = ef.apply(cal, ['speed_delay_ticks'], [-4.0])
    assert values == {'speed_delay_ticks': 0}


def test_cost_of_nothing_is_refused():
    with pytest.raises(ValueError, match='no windows'):
        ef.cost(calibration_for_car(PLANTED_CAR), [])
