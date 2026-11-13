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


def _rows(delivered, n=100, asked=3.0):
    return [{'low': asked - 0.5, 'high': asked + 0.5, 'asked': asked, 'n': n,
             'median': delivered, 'p90': delivered}]


def test_a_slide_is_excluded_from_the_sustained_limit():
    # gyro says 5, accelerometer says 3: the gyro keeps reading yaw while the path straightens.
    slide = grip_truth.sustained_limit(_rows(3.0, asked=6.0), _rows(5.0, asked=6.0))
    grip = grip_truth.sustained_limit(_rows(3.0, asked=3.0), _rows(3.1, asked=3.0))
    assert math.isnan(slide)
    assert grip == pytest.approx(3.0, abs=1e-9)  # accelerometer median of the agreeing bin


def test_sustained_limit_takes_the_highest_agreeing_bin_with_enough_samples():
    acc = _rows(2.8, asked=3.0) + _rows(3.2, asked=5.0) + _rows(3.3, asked=7.0, n=49)
    gyro = _rows(2.9, asked=3.0) + _rows(3.3, asked=5.0) + _rows(3.3, asked=7.0, n=49)
    # the 7.0 bin has 49 < 50 samples; the 5.0 bin agrees within 15% (3.2 against 3.3)
    assert grip_truth.sustained_limit(acc, gyro) == pytest.approx(3.2, abs=1e-9)  # m/s^2


def test_agreement_boundary_is_fifteen_percent_of_the_gyro_value():
    assert grip_truth.sustained_limit(_rows(3.0), _rows(3.0 / 0.85)) == pytest.approx(3.0)  # exactly 15%
    assert math.isnan(grip_truth.sustained_limit(_rows(3.0), _rows(3.0 / 0.85 * 1.01)))      # just past it


def test_empty_input_returns_an_empty_curve_and_nan_limit():
    assert grip_truth.curve(np.array([]), np.array([]), bins=((0.0, 1.0),)) == []
    assert math.isnan(grip_truth.sustained_limit([], []))


def test_nan_samples_are_dropped_not_propagated():
    asked = np.array([3.0, 3.0, np.nan, 3.0, 3.0])
    delivered = np.array([2.0, np.nan, 2.0, 2.0, 2.0])
    rows = grip_truth.curve(asked, delivered, bins=((2.5, 3.5),))
    assert rows[0]['n'] == 3
    assert rows[0]['median'] == pytest.approx(2.0, abs=1e-12)


def test_analyse_reports_both_signals_and_their_ratio():
    t, speed, yaw, lat, steer = circle(seconds=20.0)
    result = grip_truth.analyse(t, speed, yaw, lat, steer, WHEELBASE, bins=((2.5, 3.5),), min_bin_samples=10)
    row = result['rows'][0]
    assert row['acc_median'] == pytest.approx(3.0, abs=0.01)    # m/s^2
    assert row['gyro_median'] == pytest.approx(3.0, abs=0.01)   # speed * yaw = 3 * 1
    assert row['ratio'] == pytest.approx(1.0, abs=0.01)
    assert result['sustained_limit'] == pytest.approx(3.0, abs=0.01)
