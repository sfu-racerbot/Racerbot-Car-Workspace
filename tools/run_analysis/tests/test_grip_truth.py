"""Tests for grip_truth.py: sustained grip from accelerometer against speed times gyro.

Pure numpy, no bag and no ROS graph. Every expected value is a closed form of a
synthetic run the test built (a circle of known speed and yaw rate, a slide whose
gyro and accelerometer disagree by construction). Run by:
python3 -m pytest tools/f1tenth_sim/tests/ tools/racerbot_sim/tests/ tools/run_analysis/tests/ -q
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from run_analysis import grip_truth  # noqa: E402

WHEELBASE = 0.36   # m, docs/hardware-reference.md
HZ = 50.0


def circle(seconds=10.0, speed=3.0, yaw=1.0, lateral=3.0):
    t = np.arange(0.0, seconds, 1.0 / HZ)
    n = len(t)
    steering = np.full(n, math.atan(WHEELBASE * yaw / speed))   # kinematic: kappa = yaw / v
    return t, np.full(n, speed), np.full(n, yaw), np.full(n, lateral), steering


def test_a_steady_circle_is_all_steady_and_bins_to_its_lateral_acceleration():
    t, speed, yaw, lat, steer = circle()
    mask = grip_truth.steady_corners(t, speed, yaw, lat, steer)
    assert mask.all()
    asked = grip_truth.asked_lateral_accel(speed, steer, WHEELBASE)
    assert asked == pytest.approx(3.0, abs=1e-9)  # v^2 * kappa = 3^2 * (1/3)
    rows = grip_truth.curve(asked[mask], lat[mask], bins=((2.5, 3.5),))
    assert len(rows) == 1
    assert rows[0]['n'] == mask.sum()
    assert rows[0]['median'] == pytest.approx(3.0, abs=0.01)  # m/s^2
    assert rows[0]['p90'] == pytest.approx(3.0, abs=0.01)  # m/s^2


def test_a_steering_ramp_of_0_1_rad_in_0_3_s_is_rejected():
    t, speed, yaw, lat, steer = circle()
    steer = steer.copy()
    i = int(5.0 * HZ)
    steer[i:i + int(0.3 * HZ)] += np.linspace(0.0, 0.1, int(0.3 * HZ))
    steer[i + int(0.3 * HZ):] += 0.1
    mask = grip_truth.steady_corners(t, speed, yaw, lat, steer)
    mid_ramp = i + int(0.15 * HZ)
    assert not mask[mid_ramp]
    assert mask[int(2.0 * HZ)] and mask[int(8.0 * HZ)]  # steady before and after


def test_a_braking_segment_of_2_ms2_is_rejected():
    t, speed, yaw, lat, steer = circle(seconds=10.0, speed=4.0, yaw=1.0)
    speed = speed.copy()
    brake = (t >= 4.0) & (t < 6.0)
    speed[brake] = 4.0 - 2.0 * (t[brake] - 4.0)   # -2 m/s^2 from 4 m/s
    speed[t >= 6.0] = 2.0
    speed[brake] = np.maximum(speed[brake], 2.0)  # stays above min_speed, so only the slope rejects it
    mask = grip_truth.steady_corners(t, speed, yaw, lat, steer)
    assert not mask[int(4.5 * HZ)]
    assert mask[int(2.0 * HZ)]


def test_slow_samples_and_straights_are_not_cornering():
    t, speed, yaw, lat, steer = circle()
    assert not grip_truth.steady_corners(t, speed * 0.2, yaw, lat, steer).any()  # 0.6 m/s < 1.5
    assert not grip_truth.steady_corners(t, speed, yaw * 0.0, lat, steer * 0.0).any()  # a straight


def test_accelerometer_bias_is_removed_from_the_straights():
    rng = np.random.default_rng(1)
    n = 3000
    speed = np.full(n, 3.0)
    yaw = np.zeros(n)
    true_lat = np.zeros(n)
    yaw[1000:2000] = 1.0                 # a corner in the middle
    true_lat[1000:2000] = 3.0
    measured = true_lat + 0.4 + rng.normal(0.0, 0.05, n)   # 0.4 m/s^2 bias
    corrected, bias, count = grip_truth.remove_straight_bias(speed, yaw, measured)
    assert bias == pytest.approx(0.4, abs=0.05)  # m/s^2
    assert count == 2000                       # the straight samples it came from
    assert corrected[1000:2000].mean() == pytest.approx(3.0, abs=0.05)  # m/s^2


def test_no_straights_means_no_bias_estimate():
    n = 500
    corrected, bias, count = grip_truth.remove_straight_bias(
        np.full(n, 3.0), np.full(n, 1.0), np.full(n, 3.4))
    assert count == 0
    assert math.isnan(bias)
    assert np.array_equal(corrected, np.full(n, 3.4))   # untouched, not guessed


def _rows(delivered, n=100, centre=3.0):
    return [{'low': centre - 0.25, 'high': centre + 0.25, 'asked': centre, 'n': n,
             'median': delivered, 'p90': delivered}]


def test_sustained_limit_is_the_plateau_of_a_curve_that_rises_then_falls():
    # gyro-basis medians: rising to 3.0 at a 3.75 ask, then falling past the tyre's peak
    gyro = (_rows(2.2, centre=2.25) + _rows(2.7, centre=2.75) + _rows(2.9, centre=3.25)
            + _rows(3.0, centre=3.75) + _rows(2.6, centre=4.5) + _rows(2.4, centre=5.5))
    limit, peak = grip_truth.sustained_limit(gyro)
    assert limit == pytest.approx(3.0, abs=1e-12)  # m/s^2, the largest median
    assert peak == pytest.approx(3.75, abs=1e-12)  # m/s^2, that bin's centre (3.5-4.0)


def test_bins_under_fifty_samples_give_nan_for_both_numbers():
    gyro = _rows(3.0, n=49, centre=3.75)
    limit, peak = grip_truth.sustained_limit(gyro)
    assert math.isnan(limit) and math.isnan(peak)
    assert math.isnan(grip_truth.accel_over_gyro(_rows(3.5, n=49, centre=3.75), gyro))


def test_exactly_fifty_samples_qualify():
    limit, peak = grip_truth.sustained_limit(_rows(3.0, n=50, centre=3.75))
    assert limit == pytest.approx(3.0, abs=1e-12)  # m/s^2
    assert peak == pytest.approx(3.75, abs=1e-12)  # m/s^2


def test_an_accelerometer_25_percent_above_the_gyro_changes_nothing_but_the_roll_indicator():
    gyro = _rows(2.9, centre=3.25) + _rows(3.0, centre=3.75)
    acc = _rows(2.9 * 1.25, centre=3.25) + _rows(3.0 * 1.25, centre=3.75)
    limit, peak = grip_truth.sustained_limit(gyro)
    assert limit == pytest.approx(3.0, abs=1e-12)  # m/s^2, the accelerometer is not consulted
    assert grip_truth.accel_over_gyro(acc, gyro) == pytest.approx(1.25, abs=0.01)


def test_accel_over_gyro_is_the_median_ratio_over_qualifying_bins_only():
    gyro = _rows(3.0, centre=2.75) + _rows(3.0, centre=3.25) + _rows(3.0, centre=3.75) + _rows(3.0, n=10, centre=4.5)
    acc = _rows(3.3, centre=2.75) + _rows(3.6, centre=3.25) + _rows(3.9, centre=3.75) + _rows(30.0, n=10, centre=4.5)
    # ratios 1.1, 1.2, 1.3 (the n=10 bin with ratio 10 is excluded): median 1.2
    assert grip_truth.accel_over_gyro(acc, gyro) == pytest.approx(1.2, abs=1e-9)


def test_empty_input_returns_an_empty_curve_and_nan_limit():
    assert grip_truth.curve(np.array([]), np.array([]), bins=((0.0, 1.0),)) == []
    limit, peak = grip_truth.sustained_limit([])
    assert math.isnan(limit) and math.isnan(peak)
    assert math.isnan(grip_truth.accel_over_gyro([], []))


def test_nan_samples_are_dropped_not_propagated():
    asked = np.array([3.0, 3.0, np.nan, 3.0, 3.0])
    delivered = np.array([2.0, np.nan, 2.0, 2.0, 2.0])
    rows = grip_truth.curve(asked, delivered, bins=((2.5, 3.5),))
    assert rows[0]['n'] == 3
    assert rows[0]['median'] == pytest.approx(2.0, abs=1e-12)


def test_analyse_reports_both_signals_the_plateau_and_the_roll_indicator():
    t, speed, yaw, lat, steer = circle(seconds=20.0)
    lat = lat * 1.25                                           # accelerometer 25% above speed * gyro
    result = grip_truth.analyse(t, speed, yaw, lat, steer, WHEELBASE, bins=((2.5, 3.5),), min_bin_samples=10)
    row = result['rows'][0]
    assert row['acc_median'] == pytest.approx(3.75, abs=0.01)    # m/s^2
    assert row['gyro_median'] == pytest.approx(3.0, abs=0.01)    # speed * yaw = 3 * 1
    assert row['ratio'] == pytest.approx(1.25, abs=0.01)
    assert result['sustained_limit'] == pytest.approx(3.0, abs=0.01)   # gyro basis
    assert result['peak_asked'] == pytest.approx(3.0, abs=1e-9)        # bin 2.5-3.5 centre
    assert result['accel_over_gyro'] == pytest.approx(1.25, abs=0.01)
