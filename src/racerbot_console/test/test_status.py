"""status.py: what each reading is shown as.

Oracles: battery thresholds are racerbot_doctor/car.py's 3S per-cell limits
(3.4 V fail, 3.6 V warn, 4.25 V full; team, 2026-10-01), read back from that
module so the menu can never drift from the doctor; staleness is STALE_SEC;
rates are closed forms over evenly spaced arrivals.
"""
import math

import pytest

from racerbot_console import status
from racerbot_console.status import Level
from racerbot_doctor import car as dc


@pytest.mark.parametrize('volts', [None, math.nan, math.inf, -math.inf])
def test_battery_without_a_finite_reading_is_off(volts):
    assert status.battery(volts).level is Level.OFF


def test_battery_below_the_absent_threshold_is_off_not_failed():
    s = status.battery(dc.BATTERY_ABSENT_V - 0.01)
    assert s.level is Level.OFF and 'battery off' in s.text


@pytest.mark.parametrize('volts,level', [
    (dc.BATTERY_FAIL_V - 0.01, Level.FAIL),     # one step under 3.4 V/cell
    (dc.BATTERY_FAIL_V, Level.WARN),            # exactly 3.4 V/cell: not yet failed
    (dc.BATTERY_WARN_V - 0.01, Level.WARN),
    (dc.BATTERY_WARN_V, Level.OK),              # exactly 3.6 V/cell
    (dc.BATTERY_HIGH_V, Level.OK),              # exactly a full charge
    (dc.BATTERY_HIGH_V + 0.01, Level.WARN),
])
def test_battery_thresholds_match_the_doctor_at_each_boundary(volts, level):
    assert status.battery(volts).level is level


def test_battery_text_shows_volts_and_per_cell():
    # 11.70 V over 3 cells = 3.90 V/cell (closed form)
    assert status.battery(11.70).text.startswith('11.70 V 3.90/cell')


def test_is_fresh_edges():
    assert status.is_fresh(10.0, 10.0 + status.STALE_SEC)            # exactly at the limit
    assert not status.is_fresh(10.0, 10.0 + status.STALE_SEC + 1e-6)  # one step past
    assert not status.is_fresh(10.0, 9.0)                            # stamp from the future
    assert not status.is_fresh(None, 10.0)
    assert not status.is_fresh(math.nan, 10.0)
    assert not status.is_fresh(10.0, math.inf)


def _buttons(lb):
    b = [0] * 11
    b[dc.JOY_LB_BUTTON] = lb
    return tuple(b)


def test_lb_held_only_when_the_lb_button_is_1_and_joy_is_fresh():
    assert status.lb(_buttons(1), 10.0, 10.1).text == 'HELD'


@pytest.mark.parametrize('buttons,stamp,now,text', [
    (_buttons(0), 10.0, 10.1, 'released'),
    (None, None, 10.0, 'no controller'),                       # /joy never received
    (_buttons(1), 10.0, 10.0 + status.STALE_SEC + 0.01, 'controller silent'),  # stale
    ((1, 1, 1), 10.0, 10.1, 'no LB button in /joy'),           # too few buttons
    ((), 10.0, 10.1, 'no LB button in /joy'),
])
def test_lb_is_never_shown_held_otherwise(buttons, stamp, now, text):
    s = status.lb(buttons, stamp, now)
    assert s.text == text and s.level is not Level.OK


def test_lb_reads_the_doctor_button_not_another():
    other = [0] * 11
    other[dc.JOY_LB_BUTTON + 1] = 1
    assert status.lb(tuple(other), 10.0, 10.1).text == 'released'


@pytest.mark.parametrize('count,level', [
    (None, Level.OFF), (0, Level.OFF), (1, Level.OK), (2, Level.FAIL), (3, Level.FAIL)])
def test_joy_publishers(count, level):
    assert status.joy_publishers(count).level is level


@pytest.mark.parametrize('hz,level', [
    (None, Level.OFF), (math.nan, Level.OFF), (0.0, Level.OFF), (-5.0, Level.OFF),
    (dc.SCAN_MIN_HZ - 0.1, Level.WARN), (dc.SCAN_MIN_HZ, Level.OK)])
def test_rate_against_its_minimum(hz, level):
    assert status.rate(hz, dc.SCAN_MIN_HZ, 'LiDAR').level is level


def test_speed_is_hidden_when_stale_or_not_finite():
    assert status.speed(1.25, 10.0, 10.1).text == '+1.25 m/s'
    assert status.speed(1.25, 10.0, 12.0).level is Level.OFF
    assert status.speed(math.nan, 10.0, 10.1).level is Level.OFF
    assert status.speed(None, None, 10.1).level is Level.OFF


@pytest.mark.parametrize('state,level', [
    ('ok', Level.OK), ('waiting', Level.WARN), ('fault', Level.FAIL),
    (None, Level.OFF), ('weird', Level.WARN)])
def test_imu_state_from_imu_health(state, level):
    assert status.imu(state).level is level


def test_rate_meter_measures_an_even_40_hz_stream():
    meter = status.RateMeter(window=2.0)
    for i in range(81):                       # 2.0 s of arrivals every 25 ms
        meter.add(100.0 + i * 0.025)
    assert meter.hz(102.0) == pytest.approx(40.0, abs=0.01)  # 0.01 Hz


def test_rate_meter_goes_quiet_when_the_stream_stops():
    meter = status.RateMeter(window=2.0)
    for i in range(41):
        meter.add(100.0 + i * 0.025)          # last arrival at 101.0
    assert meter.hz(101.0 + status.STALE_SEC) is not None
    assert meter.hz(101.0 + status.STALE_SEC + 0.01) is None


def test_rate_meter_needs_two_arrivals_and_ignores_non_finite():
    meter = status.RateMeter()
    assert meter.hz(1.0) is None
    meter.add(1.0)
    meter.add(math.nan)
    assert meter.hz(1.0) is None
