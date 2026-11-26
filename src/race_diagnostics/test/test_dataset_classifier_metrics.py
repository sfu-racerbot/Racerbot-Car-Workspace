"""
race_diagnostics.dataset_classifier's measurements and rules, on arrays whose
answers are known. No ROS needed:

    python3 -m pytest src/race_diagnostics/test/test_dataset_classifier_metrics.py -v

(also collected by `colcon test --packages-select race_diagnostics`).

Oracles: closed forms recomputed here (a sine's turning points, a circle's
v * omega, an integral of a constant speed), invariants (the scan snapshots
give moved_between the same answer as every scan), and the threshold rule in
config/classification_rules.yaml's header (meet usable -> usable, meet only
conditional -> conditional, else unusable; boundaries inclusive).
"""
import copy
import math
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from race_diagnostics.dataset_classifier import metrics as mx  # noqa: E402
from race_diagnostics.dataset_classifier import load_rules  # noqa: E402
from race_diagnostics.dataset_classifier.bag_stats import CarryTracker  # noqa: E402
from race_diagnostics.dataset_classifier.classifier import grade_signal  # noqa: E402
from race_diagnostics.dataset_classifier.rules import RulesError, check_rules, level  # noqa: E402

RULES = load_rules()


# --------------------------------------------------------------------------- reversals

@pytest.mark.parametrize('periods', [1, 3, 7])
def test_a_sine_turns_twice_per_period(periods):
    t = np.linspace(0.0, periods, periods * 400 + 1)
    assert mx.reversals(0.2 * np.sin(2 * np.pi * t), swing=0.1) == 2 * periods


def test_swings_smaller_than_the_threshold_never_count():
    t = np.linspace(0.0, 5.0, 2001)
    assert mx.reversals(0.049 * np.sin(2 * np.pi * t), swing=0.1) == 0


def test_a_swing_exactly_at_the_threshold_counts_and_one_step_short_does_not():
    assert mx.reversals([0.0, 0.1, 0.0, 0.1], swing=0.1) == 2
    short = math.nextafter(0.1, 0.0)
    assert mx.reversals([0.0, short, 0.0, short], swing=0.1) == 0


def test_reversals_ignore_non_finite_values_and_need_two():
    assert mx.reversals([0.0, math.nan, 0.5, math.inf, 0.0], swing=0.1) == 1
    assert mx.reversals([], swing=0.1) == 0
    assert mx.reversals([1.0], swing=0.1) == 0


def test_a_monotonic_ramp_never_turns():
    assert mx.reversals(np.linspace(0, 10, 100), swing=0.1) == 0
    assert mx.reversals(np.linspace(10, 0, 100), swing=0.1) == 0


# --------------------------------------------------------------------------- timing

def test_timing_of_a_steady_topic():
    t = 100.0 + np.arange(0, 400) / 40.0  # 40 Hz for 10 s
    rate, gap, coverage = mx.timing(t, 100.0, 100.0 + 399 / 40.0, coverage_gap_s=1.0)
    assert rate == pytest.approx(40.0, abs=1e-6)  # Hz
    assert gap == pytest.approx(1 / 40.0, abs=1e-9)  # s
    assert coverage == 1.0


def test_a_three_second_silence_in_ten_seconds_leaves_seventy_percent_coverage():
    t = np.concatenate([np.arange(0, 400) / 100.0, 7.0 + np.arange(0, 301) / 100.0])
    rate, gap, coverage = mx.timing(t, 0.0, 10.0, coverage_gap_s=1.0)
    assert gap == pytest.approx(3.01, abs=1e-9)  # s, 3.99 -> 7.0
    assert coverage == pytest.approx(1 - 3.01 / 10.0, abs=1e-9)
    assert rate == pytest.approx(100.0, abs=1e-6)  # the median ignores the one pause


def test_silences_before_the_first_and_after_the_last_message_count():
    t = np.arange(200, 801) / 100.0  # 2 s .. 8 s of a 0..10 s bag
    assert mx.timing(t, 0.0, 10.0, 1.0)[2] == pytest.approx(0.6, abs=1e-9)


def test_a_pause_exactly_as_long_as_the_gap_allowance_is_still_covered():
    assert mx.timing([0.0, 1.0, 2.0], 0.0, 2.0, coverage_gap_s=1.0)[2] == 1.0
    assert mx.timing([0.0, 1.0 + 1e-9, 2.0], 0.0, 2.0, coverage_gap_s=1.0)[2] == pytest.approx(
        1 - (1.0 + 1e-9) / 2.0, abs=1e-9)


def test_timing_of_too_few_messages():
    assert mx.timing([], 0.0, 10.0, 1.0) == (None, None, 0.0)
    assert mx.timing([5.0], 0.0, 10.0, 1.0) == (None, None, 0.0)
    assert mx.timing([5.0, 5.0], 0.0, 10.0, 1.0)[0] is None  # zero median spacing: no rate


# --------------------------------------------------------------------------- motion

def test_constant_speed_integrates_to_speed_times_time():
    t = np.arange(0, 501) / 50.0  # 10 s at 50 Hz
    distance, moving, top = mx.motion(t, np.full(len(t), 2.0), 0.3, 0.5)
    assert distance == pytest.approx(20.0, abs=1e-9)  # m
    assert moving == pytest.approx(10.0, abs=1e-9)  # s
    assert top == 2.0


def test_reversing_counts_as_distance_and_moving():
    t = np.arange(0, 101) / 50.0
    distance, moving, top = mx.motion(t, np.full(len(t), -1.5), 0.3, 0.5)
    assert (distance, moving, top) == (pytest.approx(3.0, abs=1e-9), pytest.approx(2.0, abs=1e-9), 1.5)


def test_an_odometry_gap_is_not_integrated_across():
    t = np.concatenate([np.arange(0, 51) / 50.0, 3.0 + np.arange(0, 51) / 50.0])  # 2 s gap
    distance, moving, _ = mx.motion(t, np.full(len(t), 1.0), 0.3, 0.5)
    assert distance == pytest.approx(2.0, abs=1e-9)  # m: two 1 s stretches, not 4 s
    assert moving == pytest.approx(2.0, abs=1e-9)


def test_speed_exactly_at_the_moving_threshold_is_moving_and_just_below_is_not():
    t = np.arange(0, 11) / 10.0
    assert mx.motion(t, np.full(11, 0.3), 0.3, 0.5)[1] == pytest.approx(1.0, abs=1e-9)
    assert mx.motion(t, np.full(11, math.nextafter(0.3, 0)), 0.3, 0.5)[1] == 0.0


def test_motion_of_no_or_one_sample():
    assert mx.motion([], [], 0.3, 0.5) == (None, None, None)
    assert mx.motion([1.0], [2.0], 0.3, 0.5) == (0.0, 0.0, 2.0)


# --------------------------------------------------------------------------- yaw and lateral

def test_yaw_rate_in_car_axes_is_the_z_rate():
    gyro = np.array([[0.1, 0.2, 0.7], [0.0, 0.0, -0.3]])
    assert list(mx.yaw_rate(gyro, np.zeros((2, 3)), 'car')) == [0.7, -0.3]


@pytest.mark.parametrize('roll_deg, pitch_deg', [(0, 0), (90, 0), (30, 45), (180, 0)])
def test_yaw_rate_from_any_mounting_by_projecting_on_measured_gravity(roll_deg, pitch_deg):
    # A board rotated arbitrarily: express car-axis gyro and up in board axes.
    r, p = math.radians(roll_deg), math.radians(pitch_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    to_board = (ry @ rx).T
    yaw = np.linspace(-2.0, 2.0, 50)
    gyro_car = np.column_stack([np.full(50, 0.05), np.zeros(50), yaw])  # a little roll rate too
    gyro = gyro_car @ to_board.T
    accel = np.tile(to_board @ [0.0, 0.0, mx.GRAVITY], (50, 1))
    assert mx.yaw_rate(gyro, accel, 'board') == pytest.approx(yaw, abs=1e-9)  # rad/s


def test_no_yaw_rate_without_a_gravity_reading():
    assert mx.yaw_rate(np.ones((3, 3)), np.zeros((3, 3)), 'board') is None
    assert mx.yaw_rate(np.ones((3, 3)), np.full((3, 3), 0.49 * mx.GRAVITY / math.sqrt(3)),
                       'sensor') is None
    assert mx.yaw_rate(np.ones((0, 3)), np.zeros((0, 3)), 'board') is None


def test_a_steady_circle_gives_v_times_omega():
    imu_t = np.arange(0, 1001) / 100.0
    odom_t = np.arange(0, 501) / 50.0
    p95, cornering = mx.lateral_acceleration(imu_t, np.full(len(imu_t), -1.5), odom_t,
                                             np.full(len(odom_t), 3.0), 0.3, 1.0, 0.5)
    assert p95 == pytest.approx(4.5, abs=1e-9)  # m/s^2: 3 m/s * 1.5 rad/s, either direction
    assert cornering == pytest.approx(10.0, abs=1e-9)  # s


def test_the_95th_percentile_of_an_even_spread():
    # oracle: closed form, lateral acceleration evenly spread over 0..5 m/s^2 has its 95th percentile at 4.75.
    imu_t = np.arange(0, 100001) / 10000.0
    yaw = np.linspace(0.0, 5.0 / 2.0, len(imu_t))  # at 2 m/s
    p95, _ = mx.lateral_acceleration(imu_t, yaw, imu_t, np.full(len(imu_t), 2.0), 0.3, 1.0, 0.5)
    assert p95 == pytest.approx(4.75, abs=1e-3)  # m/s^2


def test_cornering_time_counts_only_above_the_threshold():
    imu_t = np.arange(0, 1001) / 100.0
    yaw = np.where(imu_t < 4.0, 0.5, 0.2)  # 1.5 m/s^2 for 4 s, then 0.6 m/s^2
    p95, cornering = mx.lateral_acceleration(imu_t, yaw, imu_t, np.full(len(imu_t), 3.0), 0.3, 1.0, 0.5)
    assert cornering == pytest.approx(4.0, abs=0.011)  # s


def test_lateral_acceleration_of_a_parked_car_is_zero_and_unknown_without_odometry():
    t = np.arange(0, 101) / 10.0
    assert mx.lateral_acceleration(t, np.ones(len(t)), t, np.zeros(len(t)), 0.3, 1.0, 0.5) == (0.0, 0.0)
    assert mx.lateral_acceleration(t, np.ones(len(t)), [], [], 0.3, 1.0, 0.5) == (None, None)
    assert mx.lateral_acceleration(t, None, t, np.ones(len(t)), 0.3, 1.0, 0.5) == (None, None)


def test_speed_outside_the_odometry_is_unknown_not_extrapolated():
    out = mx.speed_at([0.0, 1.5, 3.0], np.array([1.0, 2.0]), np.array([-1.0, -3.0]))
    assert math.isnan(out[0]) and math.isnan(out[2])
    assert out[1] == pytest.approx(2.0, abs=1e-12)  # |v| halfway between 1 and 3


# --------------------------------------------------------------------------- steering

def test_steering_excitation_counts_only_while_moving():
    t = np.arange(0, 2001) / 100.0
    steer = 0.2 * np.sin(2 * np.pi * t / 4.0)
    v = np.where(t < 10.0, 0.0, 2.0)  # parked for the first two periods
    std, turns = mx.steering_excitation(t, steer, t, v, 0.3, 0.1)
    moving = steer[t > 10.0]
    assert std == pytest.approx(float(np.std(moving)), abs=1e-3)  # rad
    assert turns == pytest.approx(2 * 10 / 4, abs=1)


def test_steering_excitation_without_commands_or_odometry_is_unknown():
    t = np.arange(0, 11) / 10.0
    assert mx.steering_excitation([], [], t, np.ones(11), 0.3, 0.1) == (None, None)
    assert mx.steering_excitation(t, np.ones(11), [], [], 0.3, 0.1) == (None, None)


def test_measured_steering_finite_share_only_counts_fast_samples():
    t = np.arange(0, 100) / 10.0
    v = np.where(t < 5.0, 0.2, 1.0)  # slow, where NaN is expected, then fast
    position = np.where(t < 5.0, np.nan, np.where(t < 7.5, 0.1, np.nan))
    assert mx.finite_fraction_while_fast(t, position, t, v, 0.5) == pytest.approx(0.5, abs=0.02)
    assert mx.finite_fraction_while_fast(t, position, t, np.zeros(100), 0.5) is None


# --------------------------------------------------------------------------- autonomy

def test_autonomous_time_needs_lb_a_recent_command_and_motion():
    t = np.arange(0, 3001) / 100.0  # 30 s
    v = np.full(len(t), 2.0)
    windows = [(0.0, 10.0), (20.0, 30.0)]  # LB released 10..20 s
    commands = t[t < 25.0]  # the algorithm stopped publishing at 25 s
    seconds, metres = mx.autonomous(t, v, windows, commands, 0.5, 0.3, 0.5)
    # oracle: 0..10 s, then 20..25.5 s (a command within 0.5 s counts).
    assert seconds == pytest.approx(10.0 + 5.5, abs=0.02)
    assert metres == pytest.approx(2.0 * seconds, abs=1e-9)
    assert mx.autonomous(t, v, [], commands, 0.5, 0.3, 0.5) == (0.0, 0.0)
    assert mx.autonomous(t, v, windows, [], 0.5, 0.3, 0.5) == (0.0, 0.0)


# --------------------------------------------------------------------------- LB and carrying

def test_lb_windows_close_at_the_last_message_and_drop_short_holds():
    t = [0.0, 1.0, 1.2, 1.4, 3.0, 4.0, 6.0]
    held = [0, 1, 0, 1, 1, 0, 1]
    # 1.0..1.2 is too short; 1.4..4.0; 6.0 is still held at the last message: zero long, dropped.
    assert mx.lb_windows(t, held, 0.5) == [(1.4, 4.0)]
    assert mx.lb_windows([0.0, 2.0], [1, 1], 0.5) == [(0.0, 2.0)]
    assert mx.lb_windows([], [], 0.5) == []


def test_driving_algorithms_leave_out_slam_and_the_gamepad():
    assert mx.driving_algorithms(['async_slam_toolbox_node', 'joy_teleop']) == []
    assert mx.driving_algorithms(['/gap_follow_node', 'joy_teleop']) == ['/gap_follow_node']


def _replay(scan_t, scans, joy_t, held):
    """Feed a CarryTracker in bag order, joy first on equal times."""
    tracker = CarryTracker()
    events = sorted([(t, 0, i) for i, t in enumerate(joy_t)] + [(t, 1, i) for i, t in enumerate(scan_t)])
    for t, kind, i in events:
        if kind == 0:
            tracker.joy(t, bool(held[i]))
        else:
            tracker.scan(t, scans[i])
    return tracker.finish()


@pytest.mark.parametrize('seed', range(12))
def test_scan_snapshots_give_moved_between_the_same_answer_as_every_scan(seed):
    # Invariant: moved_between only reads the last scan before a release and
    # the first after a press, which is all the tracker keeps.
    rng = np.random.default_rng(seed)
    scan_t = np.cumsum(rng.uniform(0.01, 0.06, 600))
    scans = [rng.uniform(1, 5, 30).astype(np.float32) + (0.5 if t > scan_t[300] else 0.0) for t in scan_t]
    joy_t = np.sort(rng.uniform(0, scan_t[-1] + 1.0, 120))
    held = (np.sin(joy_t * rng.uniform(0.2, 2.0)) > 0).astype(int)
    odom = np.column_stack([scan_t, np.where(rng.random(600) < 0.9, 0.0, 0.5)])
    windows = mx.lb_windows(joy_t, held, 0.5)
    expected = mx.moved_between(scan_t, scans, odom, windows)
    snap_t, snaps = _replay(scan_t, scans, joy_t, held)
    assert len(snaps) <= 2 * len(joy_t)
    assert mx.moved_between(snap_t, snaps, odom, windows) == expected


def test_a_gap_with_no_wheel_speed_after_the_car_settles_is_not_carried():
    # The scan jumps 0.5 m in a 0.4 s release while driving: no odometry after the 0.5 s settle, no evidence.
    scan_t = np.arange(0, 2000) / 100.0
    scans = [np.full(10, 2.0 + (0.5 if t > 10.9 else 0.0)) for t in scan_t]
    driving = np.column_stack([scan_t, np.full(len(scan_t), 2.7)])
    assert mx.moved_between(scan_t, scans, driving, [(2.0, 10.8), (11.2, 19.0)]) == [False]
    parked = np.column_stack([scan_t, np.where((scan_t > 10.0) & (scan_t < 13.0), 0.0, 2.7)])
    assert mx.moved_between(scan_t, scans, parked, [(2.0, 10.8), (12.0, 19.0)]) == [True]


def test_the_tracker_keeps_the_scans_at_the_release_and_the_next_press():
    # The view changes while driving inside each hold, and not in the gaps: never carried. A tracker that kept
    # the scan at the press instead of the release, or the release instead of the next press, would compare
    # scans 2 s of driving apart and call it carried. Then one real carry, in the second gap.
    scan_t = np.arange(0, 3000) / 100.0
    joy_t = np.arange(0, 600) / 20.0
    held = ((joy_t >= 1) & (joy_t < 5)) | ((joy_t >= 8) & (joy_t < 12)) | (joy_t >= 15)
    windows = mx.lb_windows(joy_t, held, 0.5)
    assert windows == [(1.0, 5.0), (8.0, 12.0), (15.0, joy_t[-1])]
    driving = (held[np.minimum((scan_t * 20).astype(int), 599)])
    odom = np.column_stack([scan_t, np.where(driving, 2.0, 0.0)])
    place = np.cumsum(np.where(driving, 0.01, 0.0))  # the view moves 1 m/s while driving
    for carry_at, expected in [(None, [False, False]), (13.5, [False, True])]:
        shift = place + (0.5 if carry_at is None else np.where(scan_t > carry_at, 0.5, 0.0))
        scans = [np.full(20, 2.0 + d, dtype=np.float32) for d in shift]
        assert mx.moved_between(scan_t, scans, odom, windows) == expected
        snap_t, snaps = _replay(scan_t, scans, joy_t, held)
        assert mx.moved_between(snap_t, snaps, odom, windows) == expected


def test_the_rules_carry_thresholds_are_run_triages():
    measurement = RULES['measurement']
    assert measurement['carried_still_mps'] == mx.STILL_MPS
    assert measurement['carried_moved_m'] == mx.MOVED_M


# --------------------------------------------------------------------------- levels

@pytest.mark.parametrize('value, expected', [
    (20.0, 'usable'), (math.nextafter(20.0, 0), 'conditional'), (5.0, 'conditional'),
    (math.nextafter(5.0, 0), 'unusable'), (0.0, 'unusable'), (-1.0, 'unusable')])
def test_at_least_levels_are_inclusive(value, expected):
    assert level(value, {'usable': 20.0, 'conditional': 5.0}, 'at_least') == expected


@pytest.mark.parametrize('value, expected', [
    (0.5, 'usable'), (math.nextafter(0.5, 1), 'conditional'), (2.0, 'conditional'),
    (math.nextafter(2.0, 3), 'unusable')])
def test_at_most_levels_are_inclusive(value, expected):
    assert level(value, {'usable': 0.5, 'conditional': 2.0}, 'at_most') == expected


def test_without_a_conditional_level_a_value_is_never_worse_than_conditional():
    assert level(7, {'usable': 0}, 'at_most') == 'conditional'
    assert level(0, {'usable': 1}, 'at_least') == 'conditional'


def test_a_signal_is_graded_by_its_worst_check():
    signal = RULES['signals']['scan']
    good = {'count': 400, 'rate_hz': 40.0, 'max_gap_s': 0.03, 'valid_fraction': 1.0,
            'coverage': 1.0, 'stamp_problem_fraction': 0.0}
    assert grade_signal('scan', signal, '/scan', good) == ('usable', [])
    at_limits = dict(good, rate_hz=30.0, max_gap_s=0.5, valid_fraction=0.95, coverage=0.95,
                     stamp_problem_fraction=0.01)
    assert grade_signal('scan', signal, '/scan', at_limits) == ('usable', [])
    status, reasons = grade_signal('scan', signal, '/scan', dict(good, rate_hz=29.9, coverage=0.69))
    assert status == 'unusable'
    assert reasons == ['scan (/scan) rate hz 29.9 Hz, usable needs at least 30 Hz',
                       'scan (/scan) coverage 0.69, usable needs at least 0.95']


def test_a_signal_with_one_message_has_no_rate_and_is_unusable():
    status, reasons = grade_signal('scan', RULES['signals']['scan'], '/scan',
                                   {'count': 1, 'rate_hz': None, 'max_gap_s': None,
                                    'valid_fraction': 1.0, 'coverage': 0.1,
                                    'stamp_problem_fraction': None})
    assert status == 'unusable'
    assert 'scan (/scan): no rate hz from 1 message(s)' in reasons


def test_a_missing_signal_names_every_topic_it_looked_for():
    status, reasons = grade_signal('map_pose', RULES['signals']['map_pose'], None, None)
    assert status == 'missing'
    assert reasons == ['map_pose: none of /pf/viz/inferred_pose, /slam_pose, /pose in the bag']


# --------------------------------------------------------------------------- rules file

def _rules_with(edit):
    data = copy.deepcopy(RULES.data)
    edit(data)
    return data


def test_the_shipped_rules_pass_their_own_checks():
    check_rules(yaml.safe_load(open(RULES.path)))


@pytest.mark.parametrize('edit, message', [
    (lambda d: d.update(extra=1), 'rules: unknown key'),
    (lambda d: d.pop('applications'), 'rules: missing applications'),
    (lambda d: d.update(schema=2), 'schema: this classifier reads schema 1'),
    (lambda d: d['measurement'].pop('moving_speed_mps'), 'measurement: missing moving_speed_mps'),
    (lambda d: d['measurement'].update(moving_speed_mps=-0.1), 'must not be negative'),
    (lambda d: d['measurement']['valid'].update(max_speed_mps=0), 'must be positive'),
    (lambda d: d['measurement'].update(moving_speed_mps='fast'), 'expected a number'),
    (lambda d: d['measurement'].update(moving_speed_mps=True), 'expected a number'),
    (lambda d: d['topics']['/sensors/imu/raw'].update(gyro_unit='rpm'), "'rpm' is not one of"),
    (lambda d: d['signals']['scan'].update(kind='lidar'), "'lidar' is not one of"),
    (lambda d: d['signals']['scan'].update(topics='/scan'), 'expected a list of topic names'),
    (lambda d: d['signals']['scan'].update(topics=['scan']), 'expected a list of topic names'),
    (lambda d: d['signals']['odom_tf'].pop('transform'), 'names a transform'),
    (lambda d: d['signals']['scan']['checks'].update(min_rate=1), 'unknown check min_rate'),
    (lambda d: d['signals']['scan']['checks'].update(min_rate_hz={'usable': 10, 'conditional': 30}),
     'stricter than the usable level'),
    (lambda d: d['signals']['scan']['checks'].update(max_gap_s={'usable': 2.0, 'conditional': 0.5}),
     'stricter than the usable level'),
    (lambda d: d['signals']['scan']['checks'].update(min_rate_hz={'conditional': 10}), 'needs a usable level'),
    (lambda d: d['recording_flags'].update(on_fire={'all': 'unusable'}), 'unknown flag on_fire'),
    (lambda d: d['recording_flags'].pop('superseded'), 'recording_flags: missing superseded'),
    (lambda d: d['recording_flags']['superseded'].update(all='broken'), "'broken' is not one of"),
    (lambda d: d['recording_flags']['started_late'].update(racing=1), 'no application racing'),
    (lambda d: d.update(read_min_fraction=1.5), 'between 0 and 1'),
    (lambda d: d['applications']['localization']['criteria'].update(laps={'at_least': {'usable': 1}}),
     'unknown metric laps'),
    (lambda d: d['applications']['localization']['criteria']['distance_m'].update(at_most={'usable': 1}),
     'needs exactly one of at_least, at_most'),
    (lambda d: d['applications']['localization']['criteria']['distance_m'].update(if_unknown='ignore'),
     "'ignore' is not one of"),
    (lambda d: d['applications']['localization']['required']['scan'].update(if_missing='usable'),
     "'usable' is not one of"),
    (lambda d: d['signals'].update(spare={'topics': ['/x'], 'kind': 'pose'}), 'spare not used by any'),
    (lambda d: (d.update(applications={}), d['recording_flags'].update(started_late={})),
     'applications: none defined'),
])
def test_a_wrong_rules_file_is_refused_with_the_key_named(edit, message):
    with pytest.raises(RulesError, match=message):
        check_rules(_rules_with(edit))


def test_a_rules_file_that_is_not_yaml_is_refused(tmp_path):
    path = tmp_path / 'rules.yaml'
    path.write_text('schema: [1\n')
    with pytest.raises(RulesError, match='not YAML'):
        load_rules(path)


def test_the_rules_hash_changes_with_the_file(tmp_path):
    path = tmp_path / 'rules.yaml'
    path.write_text(open(RULES.path).read())
    same = load_rules(path)
    path.write_text(open(RULES.path).read() + '\n# a comment\n')
    assert same.sha256 == RULES.sha256 and load_rules(path).sha256 != RULES.sha256
