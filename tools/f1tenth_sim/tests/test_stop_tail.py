"""Forward zero-command stops on car 2, measured independently of the plant."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_analysis import validate_stop_tail as replay
from sim_fidelity.actuation import speed_loop_accel
from sim_fidelity.calibration import STOP_FLOOR, calibration_for_car, with_floor


def test_only_a_forward_stop_uses_the_new_gain():
    old = with_floor(replay.FLOOR, calibration_for_car('rb2'))
    new = with_floor(STOP_FLOOR, calibration_for_car('rb2'))
    assert old.stop_brake_gain is None
    # Exponential tail dv/dt = -k*v, never beyond the measured hard-brake limit.
    for v in (0.25, 1.0, 3.0):
        assert speed_loop_accel(new, 0, v) == pytest.approx(-min(3*v, 4.5), abs=1e-10)  # m/s²
    for cmd, v in [(0, 0), (0, -1), (1e-6, 0.7), (2, 0.5), (-0.5, 1)]:
        assert speed_loop_accel(new, cmd, v) == speed_loop_accel(old, cmd, v)


@pytest.mark.parametrize('bad', [0, -1, np.nan, np.inf])
def test_invalid_stop_gain_is_refused(bad):
    with pytest.raises(ValueError, match='stop_brake_gain'):
        replace(calibration_for_car('rb2'), stop_brake_gain=bad)


def test_stop_measurement_interpolates_time_and_distance():
    # v = 2 - t m/s; thresholds crossed between samples, integral is 2t - t²/2.
    t = np.array([0, 0.6, 1.2, 1.8, 2.0])
    got = replay.measures(t, 2 - t)
    assert got['tail_s'] == pytest.approx(0.7, abs=1e-10)   # s
    assert got['stop_s'] == pytest.approx(1.7, abs=1e-10)   # s
    assert got['wheel_m'] == pytest.approx(2*1.7 - 1.7**2/2, abs=1e-10)   # m


def test_stop_replay_agrees_with_recordings_outside_the_floor_fit():
    # Four stops from the additional Oct 30 bags (see validate_stop_tail.py).
    # Compare to the measured threshold times, not a saved simulator output.
    data = json.loads(replay.FIXTURE.read_text())
    stops = [s for run in replay.EXTRA for s in data[run]]
    cal = with_floor(STOP_FLOOR, calibration_for_car('rb2'))
    assert len(stops) == 4
    errors = [replay.predict(s, cal)['stop_s'] - s['stop_s'] for s in stops]
    assert np.sqrt(np.mean(np.square(errors))) < 0.08  # s; three 40 Hz ticks, rounded up
