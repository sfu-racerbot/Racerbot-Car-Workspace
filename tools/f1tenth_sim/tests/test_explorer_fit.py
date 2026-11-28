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
                    maxiter=12, popsize=6, workers=4, holdout_every=5, with_intervals=True)
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


def test_the_fit_holds_on_windows_it_never_saw():
    _, _, result = planted_fit()
    # Every 5th window was held back from the search; the fit must explain those too.
    assert result.holdout_after < 0.3 * result.holdout_before


def test_every_fitted_number_comes_with_a_range_around_it():
    _, _, result = planted_fit()
    for name, (low, high) in result.intervals.items():
        assert low <= result.values[name] + 1e-9 and result.values[name] - 1e-9 <= high, name
    # A delay's range is whole ticks.
    assert all(float(v).is_integer() for v in result.intervals['steering_delay_ticks'])


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
    w = ef.windows_from_series(**s, horizon=1.0, spacing=1.0, lead_in=0.0)
    assert len(w) == 4
    assert all(len(x.commands) == 40 for x in w)
    assert all(len(x.prior_speed_commands) == 8 for x in w)


def test_a_window_never_spans_an_untrusted_tick():
    s = series(n=8 + 4 * 40)
    s['speed'] = 2.0 + 0.001 * np.arange(len(s['t']))  # speed tags each tick's index
    bad = 8 + 60  # inside the second second
    s['good'][bad] = False
    w = ef.windows_from_series(**s, horizon=1.0, spacing=1.0, lead_in=0.0)
    starts = [int(round((x.speed[0] - 2.0) / 0.001)) for x in w]
    # 8..48 is clean; the next clean start after the bad tick is 69, then 109.
    assert starts == [8, bad + 1, bad + 41]
    assert all(not (st <= bad < st + 40) for st in starts)


def test_a_window_replays_a_lead_in_first_and_scores_only_after_it():
    s = series(n=8 + 20 + 40)
    s['speed'] = 2.0 + 0.001 * np.arange(len(s['t']))  # speed tags each tick's index
    s['speed_cmd'] = s['speed'].copy()
    (w,) = ef.windows_from_series(**s, horizon=1.0, spacing=5.0, lead_in=0.5)
    assert w.lead_in == 20 and len(w.commands) == 60 and len(w.speed) == 40
    # Scored from tick 28; the replay starts at the lead-in's first tick, 8, with the 8 before it.
    assert w.speed[0] == pytest.approx(2.0 + 0.001 * 28, abs=1e-12)  # change-detector of indexing
    assert w.initial_speed == pytest.approx(2.0 + 0.001 * 8, abs=1e-12)
    assert w.prior_speed_commands == pytest.approx(tuple(2.0 + 0.001 * i for i in range(8)))
    assert w.t0 == pytest.approx(28 * ef.DT)


def test_an_untrusted_tick_in_the_lead_in_drops_the_window_too():
    s = series(n=8 + 20 + 40)
    s['good'][8 + 5] = False
    assert ef.windows_from_series(**s, horizon=1.0, spacing=5.0, lead_in=0.5) == []


def test_the_lead_in_removes_the_start_up_error_no_number_can_fix():
    """Replaying a planted car's own recording with its own numbers must
    score near zero once the lead-in lets the wheels and turn rate catch up."""
    from run_analysis.fit_sim import simulate_commands
    base = calibration_for_car(PLANTED_CAR)
    t, cmd = explorer_commands()
    v, yaw = simulate_commands(cmd, base, initial_speed=0.0)
    args = (t, cmd[:, 0], cmd[:, 1], v, yaw, np.ones(len(t), bool))
    cold = ef.windows_from_series(*args, horizon=2.0, spacing=6.0, lead_in=0.0)
    warm = ef.windows_from_series(*args, horizon=2.0, spacing=6.0, lead_in=0.75)
    assert ef.cost(base, warm) < 0.25 * ef.cost(base, cold)


def test_interval_finds_a_parabolas_minimum_and_resampling_spread():
    grid = np.linspace(0.0, 2.0, 21)
    centres = np.array([0.9, 1.0, 1.1, 1.0, 0.95, 1.05])
    errors = (grid[:, None] - centres[None, :]) ** 2
    best, low, high = ef.interval(grid, errors)
    # Closed form: the mean of quadratics is a quadratic with its minimum at the mean centre.
    assert best == pytest.approx(centres.mean(), abs=1e-9)
    assert centres.min() <= low <= best <= high <= centres.max()
    same = (grid[:, None] - np.full((1, 6), 1.23)) ** 2
    assert ef.interval(grid, same)[1:] == pytest.approx((1.23, 1.23), abs=1e-9)


def test_interval_for_a_delay_stays_on_whole_ticks():
    grid = [2, 3, 4, 5, 6]
    errors = np.array([[4.0, 4.0], [1.0, 1.2], [0.1, 0.0], [1.0, 0.9], [4.0, 4.0]])
    assert ef.interval(grid, errors, whole=True) == (4.0, 4.0, 4.0)


def test_interval_refuses_a_mismatched_or_empty_table():
    with pytest.raises(ValueError, match='grid points, windows'):
        ef.interval([1.0, 2.0], np.zeros((3, 4)))
    with pytest.raises(ValueError, match='grid points, windows'):
        ef.interval([1.0, 2.0], np.zeros((2, 0)))


def test_holdout_takes_every_nth_window_by_time_and_keeps_the_rest():
    ws = [ef.Window(np.zeros((2, 2)), np.zeros(2), np.zeros(2), 1.0, (), 0, float(k))
          for k in (5, 1, 3, 0, 2, 4, 6, 9, 8, 7)]
    kept, held = ef.split_holdout(ws, every=5)
    assert [w.t0 for w in held] == [4.0, 9.0]
    assert sorted(w.t0 for w in kept) == [0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0]
    with pytest.raises(ValueError, match='2 or more'):
        ef.split_holdout(ws, every=1)


def drift_windows(start_friction, end_friction):
    """12 s of explorer-like driving on one floor, then 12 s on another."""
    from run_analysis.fit_sim import simulate_commands
    base = calibration_for_car(PLANTED_CAR)
    t, cmd = explorer_commands()
    half = len(t) // 2
    out = []
    for part, friction in ((slice(0, half), start_friction), (slice(half, None), end_friction)):
        truth = replace(base, surface_friction=friction)
        v, yaw = simulate_commands(cmd[part], truth, initial_speed=0.0)
        out += ef.windows_from_series(t[part], cmd[part, 0], cmd[part, 1], v, yaw,
                                      np.ones(len(v), bool), horizon=1.0, spacing=1.0)
    return base, out


@pytest.mark.parametrize('start, end, changed', [(0.42, 0.42, False), (0.42, 0.30, True)])
def test_drift_flags_a_floor_that_changed_mid_session_and_only_then(start, end, changed):
    base, windows = drift_windows(start, end)
    result = ef.drift(replace(base, surface_friction=0.42), windows, parts=2, workers=4)
    assert result['changed'] is changed, result


def test_drift_refuses_too_few_windows():
    base = calibration_for_car(PLANTED_CAR)
    w = ef.Window(np.zeros((2, 2)), np.zeros(2), np.zeros(2), 1.0, (), 0, 0.0)
    with pytest.raises(ValueError, match='too few windows'):
        ef.drift(base, [w] * 5, parts=2)


def test_a_window_needs_the_car_moving_at_its_start():
    s = series(speed=0.2)
    assert ef.windows_from_series(**s, horizon=1.0, min_speed=0.5, lead_in=0.0) == []


def test_nan_in_a_window_drops_it():
    s = series(n=8 + 40)
    s['yaw_rate'][20] = math.nan
    assert ef.windows_from_series(**s, horizon=1.0, lead_in=0.0) == []


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
