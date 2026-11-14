"""The speed loop's braking deadband (Calibration.speed_loop_brake_deadband, plant._measured_speed_loop) and rb1's fitted speed loop
(cars/rb1/sim/speed_loop.yaml via calibration.car_speed_loop, tools/run_analysis/fit_speed_loop.py). Run by the tool-suite command in CLAUDE.md.

Oracle: the loop's definition. The plant returns the gym speed command measured + accel / gym_gain, so accel = (result - measured) * gym_gain
is read back and compared to gain * max(0, excess - deadband) clamped at brake_max.
"""
import dataclasses
import json
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src', 'gap_follow'))
import eval_gap  # noqa: E402
from sim_fidelity import calibration_for_car  # noqa: E402


def plant_with(**fields):
    cal = dataclasses.replace(calibration_for_car(), **fields)
    return eval_gap.make_plant('indoor_oval', 1, calibration=cal)


def accel(plant, command, measured):
    gain = 10.0 * plant.params.a_max / (-plant.params.v_min if command < measured else plant.params.v_max)
    return (plant._measured_speed_loop(command, measured) - measured) * gain


def test_no_braking_at_or_inside_the_deadband():
    p = plant_with(speed_loop_brake_deadband=0.4, speed_loop_brake_gain=3.0, speed_loop_brake_max=4.0)
    assert accel(p, 2.0, 2.4) == pytest.approx(0.0, abs=1e-9)           # e = deadband exactly
    assert accel(p, 2.0, 2.2) == pytest.approx(0.0, abs=1e-9)           # e inside it: no braking, and above all no acceleration
    p.close()


def test_braking_is_the_gain_times_the_excess_past_the_deadband():
    p = plant_with(speed_loop_brake_deadband=0.4, speed_loop_brake_gain=3.0, speed_loop_brake_max=4.0)
    assert accel(p, 2.0, 2.9) == pytest.approx(-1.5, abs=1e-9)          # e = 0.9 = deadband + 0.5, gain 3
    p.close()


def test_large_excess_is_clamped_at_brake_max():
    p = plant_with(speed_loop_brake_deadband=0.4, speed_loop_brake_gain=3.0, speed_loop_brake_max=2.5)
    assert accel(p, 1.0, 4.0) == pytest.approx(-2.5, abs=1e-9)
    p.close()


def test_the_acceleration_side_ignores_the_deadband():
    on, off = plant_with(speed_loop_brake_deadband=0.4), plant_with(speed_loop_brake_deadband=0.0)
    for command, measured in ((2.0, 1.0), (3.0, 2.8), (0.5, 0.0)):
        assert on._measured_speed_loop(command, measured) == off._measured_speed_loop(command, measured)
    on.close(), off.close()


def test_deadband_zero_is_todays_loop_over_a_3_s_step_down():
    p = plant_with(speed_loop_brake_deadband=0.0, speed_loop_brake_gain=4.0, speed_loop_brake_max=4.0)
    gym_gain = 10.0 * p.params.a_max / -p.params.v_min
    v_new = v_old = 3.0
    new, old = [], []
    for _ in range(120):
        v_new = p._measured_speed_loop(1.0, v_new)                      # the plant's own loop, command 3.0 -> 1.0
        old_accel = max(4.0 * (1.0 - v_old), -4.0)                       # today's formula, written out
        v_old = v_old + old_accel / gym_gain
        new.append(v_new), old.append(v_old)
    assert max(abs(a - b) for a, b in zip(new, old)) == 0.0
    p.close()


@pytest.mark.parametrize('bad', [-0.1, math.nan, math.inf, -math.inf])
def test_a_negative_or_non_finite_deadband_is_refused_by_name(bad):
    with pytest.raises(ValueError, match='speed_loop_brake_deadband'):
        dataclasses.replace(calibration_for_car(), speed_loop_brake_deadband=bad)


# --- the pure loop function and the per-car wiring (2026-11-14) ------------------------------------------------------
from sim_fidelity.actuation import speed_loop_accel  # noqa: E402
from sim_fidelity.calibration import SHARED_SPEED_LOOP, car_speed_loop, describe  # noqa: E402

FIT = os.path.join(HERE, '..', 'sim_fidelity', 'sfu_course_2026_11_11_speed_loop_rb1.json')
LOOP = ('speed_loop_accel_gain', 'speed_loop_accel_max', 'speed_loop_brake_deadband', 'speed_loop_brake_gain',
        'speed_loop_brake_max', 'speed_delay_ticks')


def test_the_pure_loop_function_is_what_the_plant_uses():
    # Oracle: the plant's gym command is measured + accel / gym_gain with accel from speed_loop_accel (one formula, two callers).
    cal = dataclasses.replace(calibration_for_car(), speed_loop_brake_deadband=0.3, speed_loop_brake_gain=3.0)
    p = eval_gap.make_plant('indoor_oval', 1, calibration=cal)
    for command, measured in ((2.0, 2.2), (2.0, 2.9), (1.0, 4.0), (3.0, 1.0)):
        gain = 10.0 * p.params.a_max / (-p.params.v_min if command < measured else p.params.v_max)
        assert p._measured_speed_loop(command, measured) == measured + speed_loop_accel(cal, command, measured) / gain
    p.close()


@pytest.mark.parametrize('excess, expected', [
    (0.20, 0.0),            # exactly the deadband: no braking
    (0.21, -0.03),          # one step past it: gain 3 x 0.01
    (0.70, -1.5),           # 3 x 0.5
    (2.20, -4.0),           # 3 x 2.0 = 6, clamped at brake_max 4
    (-0.10, 2.0),           # accelerating: gain 20 x 0.1
    (-1.00, 5.5),           # 20 x 1.0 = 20, clamped at accel_max 5.5
])
def test_rb1_loop_in_closed_form(excess, expected):
    # Oracle: rb1's fitted loop written out: accelerating min(20 e, 5.5); braking max(-3 max(0, -e - 0.2), -4), e = cmd - v.
    cal = calibration_for_car('rb1')
    assert speed_loop_accel(cal, 2.0, 2.0 + excess) == pytest.approx(expected, abs=1e-9)


def test_rb1_has_its_fitted_loop_and_rb2_keeps_the_shared_values():
    rb1, rb2 = calibration_for_car('rb1'), calibration_for_car('rb2')
    # Oracle: the chosen set recorded by tools/run_analysis/fit_speed_loop.py.
    chosen = json.load(open(FIT))['chosen']
    assert {name: getattr(rb1, name) for name in LOOP} == chosen
    assert {name: getattr(rb2, name) for name in LOOP} == SHARED_SPEED_LOOP


def test_the_recorded_fit_beats_the_shared_values_it_replaces():
    # Recorded measurement (the fit file): the chosen set's replay error is below the shared values' on every subset.
    d = json.load(open(FIT))
    for part in ('all', 'accel', 'brake'):
        assert d['chosen_scores'][part] < d['shared'][part]
    assert d['runs'] == ['20261111-155542', '20261111-153721']


def test_describe_names_the_rb1_fit_as_the_source():
    rows = {name: src for _p, name, _v, src in describe(calibration_for_car('rb1'))}
    for name in LOOP:
        assert 'fit_speed_loop.py' in rows[name], name
    rows2 = {name: src for _p, name, _v, src in describe(calibration_for_car('rb2'))}
    assert all('fit_speed_loop.py' not in rows2[name] for name in LOOP)


def _profile_with(tmp_path, monkeypatch, text):
    """A copy of the real rb1 profile under tmp_path with sim/speed_loop.yaml replaced by ``text`` (None: no file)."""
    import shutil
    from racerbot_car import paths
    cars = tmp_path / 'cars'
    shutil.copytree(paths.cars_dir() / 'rb1', cars / 'rb1')
    loop = cars / 'rb1' / 'sim' / 'speed_loop.yaml'
    if text is None:
        loop.unlink()
    else:
        loop.write_text(text)
    monkeypatch.setenv('RACERBOT_CARS_DIR', str(cars))


def test_a_profile_without_the_file_gets_the_shared_loop(tmp_path, monkeypatch):
    _profile_with(tmp_path, monkeypatch, None)
    assert car_speed_loop('rb1') == (SHARED_SPEED_LOOP, '', frozenset())
    assert {name: getattr(calibration_for_car('rb1'), name) for name in LOOP} == SHARED_SPEED_LOOP


def test_the_file_sets_only_the_fields_it_names(tmp_path, monkeypatch):
    _profile_with(tmp_path, monkeypatch, 'source: a test\nspeed_loop_brake_deadband: 0.4\n')
    values, source, named = car_speed_loop('rb1')
    assert values == dict(SHARED_SPEED_LOOP, speed_loop_brake_deadband=0.4)
    assert named == {'speed_loop_brake_deadband'}
    assert source == 'cars/rb1/sim/speed_loop.yaml: a test'


@pytest.mark.parametrize('text, match', [
    ('speed_loop_brake_deadband: 0.4\n', 'source'),                                 # no source
    ('source: "  "\nspeed_loop_brake_deadband: 0.4\n', 'source'),                  # blank source
    ('source: x\nspeed_loop_brake_deadbnd: 0.4\n', 'unknown speed-loop fields'),   # misspelt field
    ('source: x\nspeed_delay_ticks: 2.5\n', 'speed_delay_ticks'),                  # not an integer
    ('source: x\nspeed_delay_ticks: -1\n', 'speed_delay_ticks'),
    ('source: x\nspeed_loop_brake_gain: .nan\n', 'speed_loop_brake_gain'),
    ('source: x\nspeed_loop_brake_gain: -3.0\n', 'speed_loop_brake_gain'),
    ('source: x\nspeed_loop_accel_max: fast\n', 'speed_loop_accel_max'),          # wrong type
    ('- a list\n', 'mapping'),
])
def test_a_malformed_file_is_refused_naming_the_problem(tmp_path, monkeypatch, text, match):
    _profile_with(tmp_path, monkeypatch, text)
    with pytest.raises(ValueError, match=match):
        car_speed_loop('rb1')


def test_omitted_fields_resolve_to_the_selected_cars_loop(monkeypatch):
    # The same contract as the VESC fields: CarCalibration(**overrides) fills what it omits from the selected car.
    from sim_fidelity.calibration import CarCalibration
    monkeypatch.setenv('RACERBOT_CAR', 'rb1')
    cal = CarCalibration(speed_loop_brake_gain=9.0)
    assert cal.speed_loop_brake_gain == 9.0
    assert cal.speed_loop_accel_gain == calibration_for_car('rb1').speed_loop_accel_gain == 20.0
