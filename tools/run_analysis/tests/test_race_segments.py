"""race_segments.py: which parts of a recording were real racing.

Runner: python3 -m pytest tools/run_analysis/tests/ (no ROS needed).
Oracles: closed forms on built arrays (speed x time for distance, thresholds from the module's
documented constants). The real race-day result is recorded in data/race_days/2026-11-11-uvic/.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_analysis import race_segments as rs  # noqa: E402

STEP = rs.STEP_S


def labels_for(pattern):
    """Labels from a compact string: R racing, C coasting, S spinning, H handled, . parked."""
    names = {'R': 'racing', 'C': 'coasting', 'S': 'spinning', 'H': 'handled', '.': 'parked'}
    return [names[c] for c in pattern]


def grid(n):
    return np.arange(n) * STEP


# --- step_labels: each label and its boundary -------------------------------------------

def test_driving_with_lb_is_racing_and_without_lb_is_coasting():
    assert rs.step_labels([2.0, 2.0], [0.5, 0.5], [20.0, 20.0], [True, False]) == ['racing', 'coasting']


def test_wheels_turning_while_the_lidar_view_stays_put_is_spinning_not_racing():
    # A car held in the air with LB held: wheels at 3 m/s, scene unchanged.
    assert rs.step_labels([3.0], [0.01], [0.5], [True]) == ['spinning']


def test_moving_threshold_is_inclusive_at_half_a_metre_per_second():
    out = rs.step_labels([rs.MOVING_MPS, rs.MOVING_MPS - 0.01], [0.5, 0.5], [0.0, 0.0], [True, True])
    assert out == ['racing', 'parked']


def test_scan_still_threshold_separates_spinning_from_racing():
    out = rs.step_labels([2.0, 2.0], [rs.SCAN_STILL_MPS, rs.SCAN_STILL_MPS - 0.001], [0, 0], [True, True])
    assert out == ['racing', 'spinning']


def test_stopped_wheels_with_a_changing_view_or_a_tilt_is_handled():
    out = rs.step_labels([0.0, 0.0, 0.0, 0.0], [rs.SCAN_MOVED_MPS, 0.01, 0.01, rs.SCAN_MOVED_MPS - 0.01],
                         [0.5, rs.TILT_DPS, rs.TILT_DPS - 0.1, 0.5], [False] * 4)
    assert out == ['handled', 'handled', 'parked', 'parked']


def test_creeping_wheels_are_neither_racing_nor_handled():
    # Between STILL_MPS and MOVING_MPS: a creep, counted as a pause.
    assert rs.step_labels([0.3], [0.5], [20.0], [True]) == ['parked']


def test_reverse_wheel_speed_counts_like_forward():
    assert rs.step_labels([-2.0], [0.5], [0.0], [True]) == ['racing']


def test_missing_values_never_label_a_step_racing():
    out = rs.step_labels([float('nan'), 2.0, 0.0], [0.5, float('nan'), float('nan')],
                         [0.0, 0.0, float('nan')], [True, True, False])
    # NaN speed: parked. NaN scan change reads as no change: wheels turning, so spinning.
    assert out == ['parked', 'spinning', 'parked']


# --- stretches: pauses and handling ------------------------------------------------------

def test_a_pause_up_to_three_seconds_stays_inside_one_stretch():
    pause = int(rs.PAUSE_S / STEP)  # 6 parked steps = 3.0 s
    labels = labels_for('RRRR' + '.' * pause + 'RRRR')
    assert rs.stretches(grid(len(labels)), labels) == [(0, len(labels) - 1)]


def test_a_pause_one_step_past_three_seconds_splits_the_stretch():
    pause = int(rs.PAUSE_S / STEP) + 1
    labels = labels_for('RRRR' + '.' * pause + 'RRRR')
    assert rs.stretches(grid(len(labels)), labels) == [(0, 3), (4 + pause, len(labels) - 1)]


def test_any_handling_splits_even_a_short_gap():
    labels = labels_for('RRRHRRR')
    assert rs.stretches(grid(7), labels) == [(0, 2), (4, 6)]


def test_any_spinning_splits_even_a_short_gap():
    labels = labels_for('RRRSRRR')
    assert rs.stretches(grid(7), labels) == [(0, 2), (4, 6)]


def test_coasting_after_release_joins_its_stretch_but_coasting_alone_is_not_driving():
    assert rs.stretches(grid(5), labels_for('RRCC.')) == [(0, 3)]
    assert rs.stretches(grid(5), labels_for('.CCC.')) == []


def test_empty_and_all_parked_give_no_stretches():
    assert rs.stretches(grid(0), []) == []
    assert rs.stretches(grid(4), labels_for('....')) == []


# --- classify: race or short -------------------------------------------------------------

def run_of(seconds, speed):
    n = int(round(seconds / STEP))
    return grid(n), np.full(n, speed), ['racing'] * n


def test_metres_are_wheel_speed_times_time():
    t, v, labels = run_of(20.0, 2.5)
    (s,) = rs.classify(t, v, labels)
    assert s.metres == pytest.approx(50.0, abs=1e-9)  # m: 2.5 m/s x 20 s
    assert (s.start, s.end) == (0.0, 20.0)
    assert s.kind == 'race'


def test_shorter_than_ten_seconds_is_a_short_move_even_at_race_speed():
    t, v, labels = run_of(rs.RACE_MIN_S - STEP, 5.0)  # 47.5 m, one course length and more
    assert rs.classify(t, v, labels)[0].kind == 'short'
    t, v, labels = run_of(rs.RACE_MIN_S, 5.0)
    assert rs.classify(t, v, labels)[0].kind == 'race'


def test_less_than_one_course_length_is_a_short_move():
    # 12 s at 2.3 m/s = 27.6 m, under the 28.17 m course; 2.4 m/s = 28.8 m, over it.
    t, v, labels = run_of(12.0, 2.3)
    assert rs.classify(t, v, labels)[0].kind == 'short'
    t, v, labels = run_of(12.0, 2.4)
    assert rs.classify(t, v, labels)[0].kind == 'race'


def test_slow_average_is_a_short_move():
    # 40 s, 56 m: over one lap, but 1.4 m/s average is under 1.5.
    t, v, labels = run_of(40.0, 1.4)
    assert rs.classify(t, v, labels, course_m=28.17)[0].kind == 'short'
    t, v, labels = run_of(40.0, 1.5)
    assert rs.classify(t, v, labels, course_m=28.17)[0].kind == 'race'


def test_course_length_is_a_parameter():
    t, v, labels = run_of(12.0, 2.4)  # 28.8 m
    assert rs.classify(t, v, labels, course_m=30.0)[0].kind == 'short'


# --- events ------------------------------------------------------------------------------

def test_events_merge_runs_closer_than_two_seconds_and_keep_farther_ones_apart():
    labels = labels_for('HH...HH' + '.' * 5 + 'H')
    out = rs.events(grid(len(labels)), labels, 'handled')
    # 'HH' at 0-1 s and 'HH' at 2.5-3.5 s are 1.5 s apart: one event. The last is 3 s on: separate.
    assert out == [(0.0, 3.5), (6.0, 6.5)]


# --- signal helpers ----------------------------------------------------------------------

def test_lb_held_follows_the_last_joy_message_and_is_false_before_any():
    joy = np.array([[1.0, 1], [3.0, 0], [4.0, 1]])
    assert rs.lb_held(joy, np.array([0.5, 1.0, 2.9, 3.0, 5.0])).tolist() == [False, True, True, False, True]
    assert rs.lb_held(np.zeros((0, 2)), np.array([1.0])).tolist() == [False]


def test_scan_change_is_the_median_range_change_per_second():
    scan_t = np.array([0.0, 0.5, 1.0])
    base = np.linspace(1.0, 5.0, 40)
    scans = np.stack([base, base + 0.25, base + 0.25])
    out = rs.scan_change(scan_t, scans, np.array([0.0, 0.5, 1.0]))
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(0.5, abs=1e-6)  # m/s: 0.25 m in 0.5 s
    assert out[2] == pytest.approx(0.0, abs=1e-6)


def test_scan_change_ignores_missing_and_out_of_range_beams_and_needs_twenty_valid():
    scan_t = np.array([0.0, 0.5])
    a = np.full(40, 2.0)
    b = np.full(40, 2.5)
    b[:30] = np.nan  # only 10 valid beams left
    assert np.isnan(rs.scan_change(scan_t, np.stack([a, b]), np.array([0.0, 0.5]))[1])
    b = np.full(40, 2.5)
    b[:10] = np.inf
    b[10:15] = 0.0
    assert rs.scan_change(scan_t, np.stack([a, b]), np.array([0.0, 0.5]))[1] == pytest.approx(1.0, abs=1e-6)


def test_scan_change_of_an_empty_recording_is_all_missing():
    assert np.isnan(rs.scan_change(np.zeros(0), np.zeros((0, 4)), np.array([0.0, 0.5]))).all()


def test_tilt_rate_is_the_largest_roll_pitch_rate_in_each_step():
    imu = np.array([[0.1, 3.0, 4.0], [0.3, 0.0, 1.0], [0.6, -6.0, 8.0], [0.9, 0.0, 0.0]])
    out = rs.tilt_rate(imu, np.array([0.0, 0.5, 1.0]))
    assert out.tolist() == [0.0, 5.0, 10.0]  # hypot(3,4)=5 in the first step, hypot(-6,8)=10 in the second


def test_segment_finds_one_race_and_a_carry_on_built_arrays():
    # 40 Hz scans. 0-5 s parked; 5-25 s driving at 3 m/s with LB, the scene changing;
    # 25-30 s parked; 30-32 s carried (wheels still, scene changing); 32-40 s parked.
    scan_t = np.arange(0.0, 40.0, 0.025)
    beams = np.linspace(1.0, 6.0, 60)
    shift = np.where((scan_t >= 5) & (scan_t < 25), (scan_t - 5) * 0.5, 0.0)
    shift = np.where(scan_t >= 25, 10.0, shift)
    shift = np.where((scan_t >= 30) & (scan_t < 32), 10.0 + (scan_t - 30) * 0.5, shift)
    shift = np.where(scan_t >= 32, 11.0, shift)
    scans = beams[None, :] + (shift[:, None] % 3.0)
    odom = np.c_[scan_t, np.where((scan_t >= 5) & (scan_t < 25), 3.0, 0.0)]
    joy = np.array([[0.0, 0], [4.9, 1], [25.0, 0]])
    imu = np.c_[scan_t, np.zeros_like(scan_t), np.zeros_like(scan_t)]
    t, speed, labels, found = rs.segment(scan_t, scans, odom, joy, imu)
    assert [s.kind for s in found] == ['race']
    assert found[0].start == pytest.approx(5.0, abs=0.5)  # s
    assert found[0].end == pytest.approx(25.0, abs=0.5)  # s
    assert found[0].metres == pytest.approx(60.0, abs=1.5)  # m: 3 m/s x 20 s, half-second steps
    handled = rs.events(t, labels, 'handled')
    assert len(handled) == 1 and 29.5 <= handled[0][0] <= 30.5


def test_documented_thresholds_hold_at_their_written_values():
    # Spec: data/race_days/2026-11-11-uvic/README.md, "How a stretch is judged" (0.5 m/s, 3 s, 10 s, 1.5 m/s).
    assert rs.step_labels([0.5, 0.49], [0.5, 0.5], [0, 0], [True, True]) == ['racing', 'parked']
    assert rs.step_labels([0.0] * 4, [0.10, 0.09, 0.0, 0.0], [0.0, 0.0, 8.0, 7.9], [False] * 4) == [
        'handled', 'parked', 'handled', 'parked']
    labels = labels_for('R' + '.' * 6 + 'R' + '.' * 7 + 'R')
    assert rs.stretches(grid(len(labels)), labels) == [(0, 7), (15, 15)]


def test_driving_in_reverse_counts_its_distance():
    t, v, labels = run_of(20.0, -2.5)
    assert rs.classify(t, v, labels)[0].metres == pytest.approx(50.0, abs=1e-9)  # m


def test_events_exactly_two_seconds_apart_merge():
    labels = labels_for('H' + '.' * 3 + 'H')  # first ends at 0.5 s, second starts at 2.0 s: 1.5 s gap
    assert rs.events(grid(5), labels, 'handled') == [(0.0, 2.5)]
    labels = labels_for('H' + '.' * 4 + 'H')  # second starts at 2.5 s: exactly 2.0 s gap
    assert rs.events(grid(6), labels, 'handled') == [(0.0, 3.0)]
    labels = labels_for('H' + '.' * 5 + 'H')  # 2.5 s gap
    assert rs.events(grid(7), labels, 'handled') == [(0.0, 0.5), (3.0, 3.5)]
