"""
race_diagnostics.dataset_classifier on whole synthetic runs: missing topics,
a parked car, damaged and unfinished bags, invalid values, and that grading
never harms the recording. No ROS needed:

    python3 -m pytest src/race_diagnostics/test/test_dataset_classifier_runs.py -v

(also collected by `colcon test --packages-select race_diagnostics`).

Oracles: each run is built from closed forms (dataset_classifier_fake_bag.py)
so the expected grade follows from the shipped thresholds in
config/classification_rules.yaml, read here rather than copied, plus the
spec in docs/run-diagnostics.md#dataset-qualification (a missing reference
gives conditional, a missing required signal unusable, a damaged bag is never
deleted or changed). Real bags: tools/run_analysis/tests/test_dataset_classifier_bag.py.
"""
import hashlib
import math
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from race_diagnostics.dataset_classifier import classifier as cl  # noqa: E402
from race_diagnostics.dataset_classifier import outputs, load_rules  # noqa: E402
from race_diagnostics.dataset_classifier.rules import RulesError  # noqa: E402
from race_diagnostics.dataset_classifier.run_info import (  # noqa: E402
    read_run_info, write_run_info, UNKNOWN)
import dataset_classifier_fake_bag as fake  # noqa: E402

RULES = load_rules()
APPS = ['localization', 'velocity_estimation', 'steering_response', 'lateral_acceleration',
        'algorithm_accuracy']


def classify(tmp_path, messages, *, fail_after=None, error=None, **run):
    run_dir = fake.write_run(tmp_path, messages, **run)
    kwargs = {} if error is None else {'error': error}
    doc = cl.classify_run(run_dir, RULES, backend=fake.reader(messages, fail_after, **kwargs))
    return run_dir, doc


def files_and_hashes(run_dir, skip=('qualification.yaml',)):
    out = {}
    for path in sorted(run_dir.rglob('*')):
        if path.is_file() and path.name not in skip:
            out[str(path.relative_to(run_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def test_the_shipped_rules_name_the_five_uses_in_order():
    assert list(RULES['applications']) == APPS


@pytest.fixture(scope='module')
def driving():
    return fake.drive()


def test_a_run_meeting_every_threshold_is_usable_for_all_five(tmp_path, driving):
    _, doc = classify(tmp_path, driving)
    assert doc['error'] is None
    assert doc['summary'] == dict.fromkeys(APPS, 'usable'), doc['applications']
    for app in APPS:
        assert doc['applications'][app]['reasons'] == []
        assert doc['applications'][app]['explanation'].startswith('usable: ')


def test_the_metrics_of_a_built_run_match_their_closed_forms(tmp_path, driving):
    _, doc = classify(tmp_path, driving)
    m = doc['metrics']
    # oracle: closed form. Mean speed 3 m/s (the sine averages out over 39 s = 3.9 periods,
    # leaving at most one period's 1.5 m/s * 10 s / pi = 4.8 m) from 1 s to the end at 40 s.
    assert m['distance_m'] == pytest.approx(3.0 * 39.0, abs=5.0)  # m
    assert m['moving_s'] == pytest.approx(39.0, abs=0.1)  # s: always above 0.3 m/s once moving
    assert m['max_speed_mps'] == pytest.approx(4.5, abs=0.01)  # m/s: 3 + 1.5
    # oracle: closed form, steering 0.2 sin(...): std 0.2/sqrt(2); two turns per 4 s period.
    assert m['steering_std_rad'] == pytest.approx(0.2 / math.sqrt(2), abs=0.005)  # rad
    assert m['steering_reversals'] == pytest.approx(2 * 39 / 4, abs=2)
    # speed: two turns per 10 s period over 39 s.
    assert m['speed_swings'] == pytest.approx(2 * 39 / 10, abs=1)
    assert m['lb_holds'] == 1
    assert m['lb_held_s'] == pytest.approx(40.0 - 0.5, abs=0.06)  # s, last /joy at 39.951 s
    assert m['algorithm_known'] == 1
    assert doc['activity']['moved'] is True and doc['activity']['started_late'] is False


def test_speed_ripples_smaller_than_a_swing_are_not_swings(tmp_path):
    # 0.3 m/s ripples: above steering's 0.1 swing, below speed's 1.0 (the shipped measurement.speed_swing_mps).
    assert RULES['measurement']['speed_swing_mps'] == 1.0
    _, doc = classify(tmp_path, fake.drive(
        speed_of=lambda t: 0.0 if t - fake.T0 < 1.0 else 3.0 + 0.3 * math.sin(2 * math.pi * (t - fake.T0) / 2.0)))
    assert doc['metrics']['speed_swings'] == 0
    assert 'speed_swings 0, usable needs at least 4' in doc['applications']['velocity_estimation']['explanation']


def test_a_parked_car_is_unusable_for_every_use(tmp_path):
    messages = fake.drive(moving=False)
    _, doc = classify(tmp_path, messages)
    assert doc['summary'] == dict.fromkeys(APPS, 'unusable')
    m = doc['metrics']
    assert m['distance_m'] == 0.0 and m['moving_s'] == 0.0 and m['max_speed_mps'] == 0.0
    # Measured as zero, not unknown: the odometry says the car never moved.
    assert m['lateral_accel_p95_mps2'] == 0.0 and m['cornering_s'] == 0.0
    assert m['steering_std_rad'] == 0.0 and m['steering_reversals'] == 0
    assert m['autonomous_s'] == 0.0
    assert doc['activity']['moved'] is False
    assert 'distance_m 0, usable needs at least 20' in doc['applications']['localization']['explanation']
    # The recording itself is sound: only what it holds is not enough.
    assert doc['recording']['status'] == 'usable'


def test_without_scans_localization_and_algorithm_accuracy_are_unusable(tmp_path):
    _, doc = classify(tmp_path, fake.drive(drop={'/scan'}))
    assert doc['summary'] == {
        'localization': 'unusable', 'velocity_estimation': 'usable', 'steering_response': 'usable',
        'lateral_acceleration': 'usable', 'algorithm_accuracy': 'unusable'}
    assert 'required scan missing: scan: none of /scan in the bag' in \
        doc['applications']['localization']['explanation']
    assert doc['signals']['scan']['status'] == 'missing'
    assert doc['metrics']['carried_between_holds'] is None


def test_without_a_map_pose_the_three_uses_that_compare_with_one_are_conditional(tmp_path):
    _, doc = classify(tmp_path, fake.drive(drop={'/pf/viz/inferred_pose'}))
    assert doc['summary'] == {
        'localization': 'conditional', 'velocity_estimation': 'conditional',
        'steering_response': 'usable', 'lateral_acceleration': 'usable',
        'algorithm_accuracy': 'conditional'}
    assert doc['applications']['localization']['explanation'].startswith(
        'conditional: reference map_pose missing')


def test_without_any_imu_the_three_uses_that_need_one_are_unusable(tmp_path):
    _, doc = classify(tmp_path, fake.drive(drop={'/imu/data'}))
    assert doc['summary'] == {
        'localization': 'usable', 'velocity_estimation': 'unusable', 'steering_response': 'unusable',
        'lateral_acceleration': 'unusable', 'algorithm_accuracy': 'usable'}
    assert doc['metrics']['lateral_accel_p95_mps2'] is None


def test_without_measured_steering_steering_response_is_only_conditional(tmp_path):
    _, doc = classify(tmp_path, fake.drive(drop={'/steering/measured'}))
    assert doc['summary']['steering_response'] == 'conditional'
    assert doc['metrics']['measured_steering_finite_fraction'] is None  # if_unknown: skip
    assert [r for r in doc['applications']['steering_response']['reasons']] == [
        'conditional: reference measured_steering missing: measured_steering: none of '
        '/steering/measured in the bag']


def test_a_board_axis_imu_in_g_and_deg_per_s_gives_the_same_lateral_acceleration(tmp_path, driving):
    # The VESC's raw IMU, mounted tilted: rotate the car-axis gyro and gravity into the board
    # frame and convert to deg/s and g. Projecting on measured "up" must recover the yaw rate.
    angle = math.radians(30.0)
    to_board = np.array([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)],
                         [0, math.sin(angle), math.cos(angle)]]).T
    raw = []
    for topic, t, message in driving:
        if topic != '/imu/data':
            raw.append((topic, t, message))
            continue
        w = message.angular_velocity
        a = message.linear_acceleration
        gyro = to_board @ [w.x, w.y, w.z]
        accel = to_board @ [0.0, 0.0, fake.GRAVITY]  # gravity only: the projection's "up"
        raw.append(('/sensors/imu/raw', t, fake.imu(
            t, tuple(np.degrees(gyro)), tuple(accel / fake.GRAVITY))))
    _, reference = classify(tmp_path / 'car', driving)
    _, doc = classify(tmp_path / 'board', raw)
    assert doc['signals']['imu']['topic'] == '/sensors/imu/raw'
    assert doc['metrics']['lateral_accel_p95_mps2'] == pytest.approx(
        reference['metrics']['lateral_accel_p95_mps2'], abs=1e-3)  # m/s^2
    assert doc['metrics']['cornering_s'] == pytest.approx(reference['metrics']['cornering_s'], abs=0.02)
    # No calibrated car-axis IMU: lateral acceleration has no measured reference.
    assert doc['summary']['lateral_acceleration'] == 'conditional'


def test_an_all_zero_imu_is_invalid_and_the_next_topic_is_used(tmp_path, driving):
    zeroed = [(topic, t, fake.imu(t, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) if topic == '/imu/data' else m)
              for topic, t, m in driving]
    _, alone = classify(tmp_path / 'alone', zeroed)
    assert alone['streams']['/imu/data']['valid_fraction'] == 0.0
    assert alone['signals']['imu']['status'] == 'missing'
    assert alone['summary']['lateral_acceleration'] == 'unusable'
    vesc = [('/imu/vesc/data', t, m) for topic, t, m in driving if topic == '/imu/data']
    _, fallback = classify(tmp_path / 'fallback', sorted(zeroed + vesc, key=lambda m: m[1]))
    assert fallback['signals']['imu']['topic'] == '/imu/vesc/data'
    assert fallback['summary']['lateral_acceleration'] == 'usable'


@pytest.mark.parametrize('bad', [math.nan, math.inf])
def test_non_finite_imu_samples_are_invalid(tmp_path, driving, bad):
    imu_times = [t for topic, t, _ in driving if topic == '/imu/data']
    broken = set(imu_times[::10])  # every tenth sample
    messages = [(topic, t, fake.imu(t, (0.0, 0.0, bad), (0.0, 0.0, fake.GRAVITY)) if topic == '/imu/data' and t in broken else m)
                for topic, t, m in driving]
    _, doc = classify(tmp_path, messages)
    # oracle: one sample in ten replaced.
    assert doc['streams']['/imu/data']['valid_fraction'] == pytest.approx(
        1 - len(broken) / len(imu_times), abs=1e-4)
    assert doc['signals']['imu']['status'] == 'conditional'  # 0.9 <= valid < 0.99


def test_scans_with_no_return_in_range_are_invalid(tmp_path, driving):
    dead = np.full(fake.BEAMS, np.inf, dtype=np.float32)
    messages = [(topic, t, fake.scan(t, dead) if topic == '/scan' else m) for topic, t, m in driving]
    _, doc = classify(tmp_path, messages)
    assert doc['streams']['/scan']['valid_fraction'] == 0.0
    assert doc['signals']['scan']['status'] == 'missing'
    assert doc['summary']['localization'] == 'unusable'


def test_all_zero_and_nan_beams_are_out_of_range(tmp_path, driving):
    zero = np.zeros(fake.BEAMS, dtype=np.float32)
    nan = np.full(fake.BEAMS, np.nan, dtype=np.float32)
    scans = [t for topic, t, _ in driving if topic == '/scan']
    messages = [(topic, t, fake.scan(t, zero if i % 2 else nan) if t in scans[:20] and topic == '/scan'
                 else m) for i, (topic, t, m) in enumerate(driving)]
    _, doc = classify(tmp_path, messages)
    assert doc['streams']['/scan']['valid_fraction'] == pytest.approx(1 - 20 / len(scans), abs=1e-4)


@pytest.mark.parametrize('in_range, valid', [(270, True), (269, False)])
def test_a_scan_needs_a_quarter_of_its_beams_in_range(tmp_path, driving, in_range, valid):
    # oracle: spec, measurement.valid.scan_min_beam_fraction = 0.25 of 1080 beams is 270.
    assert RULES['measurement']['valid']['scan_min_beam_fraction'] == 0.25
    beams = np.full(1080, np.inf, dtype=np.float32)
    beams[:in_range] = 3.0
    messages = [(topic, t, fake.scan(t, beams) if topic == '/scan' else m) for topic, t, m in driving]
    _, doc = classify(tmp_path, messages)
    assert doc['streams']['/scan']['valid_fraction'] == (1.0 if valid else 0.0)


def test_scans_with_a_different_beam_count_are_invalid(tmp_path, driving):
    short = np.full(fake.BEAMS - 1, 5.0, dtype=np.float32)
    scans = [t for topic, t, _ in driving if topic == '/scan']
    odd = set(scans[100:110])
    messages = [(topic, t, fake.scan(t, short) if topic == '/scan' and t in odd else m) for topic, t, m in driving]
    _, doc = classify(tmp_path, messages)
    assert doc['streams']['/scan']['valid_fraction'] == pytest.approx(1 - 10 / len(scans), abs=1e-4)


def test_steering_commands_that_are_nan_or_beyond_any_servo_are_invalid(tmp_path, driving):
    commands = [t for topic, t, _ in driving if topic == '/ackermann_cmd']
    bad = {t: value for t, value in zip(commands[::20], [math.nan, 0.61, -1.5] * 1000)}
    messages = [(topic, t, fake.ackermann(t, bad[t], 2.0) if topic == '/ackermann_cmd' and t in bad else m)
                for topic, t, m in driving]
    _, doc = classify(tmp_path, messages)
    # oracle: spec, measurement.valid.max_steering_rad = 0.6.
    assert RULES['measurement']['valid']['max_steering_rad'] == 0.6
    assert doc['streams']['/ackermann_cmd']['valid_fraction'] == pytest.approx(
        1 - len(bad) / len(commands), abs=1e-4)
    assert doc['signals']['steering_command']['status'] == 'conditional'


def test_a_steering_command_exactly_at_the_bound_is_valid(tmp_path, driving):
    commands = [t for topic, t, _ in driving if topic == '/ackermann_cmd']
    at = set(commands[::20])
    messages = [(topic, t, fake.ackermann(t, 0.6, 2.0) if topic == '/ackermann_cmd' and t in at else m)
                for topic, t, m in driving]
    _, doc = classify(tmp_path, messages)
    assert doc['streams']['/ackermann_cmd']['valid_fraction'] == 1.0


def test_non_finite_wheel_speeds_are_invalid_and_never_integrated(tmp_path, driving):
    odom_times = [t for topic, t, _ in driving if topic == '/odom']
    broken = set(odom_times[::50])
    messages = [(topic, t, fake.odom(t, math.nan) if topic == '/odom' and t in broken else m) for topic, t, m in driving]
    _, doc = classify(tmp_path, messages)
    assert doc['streams']['/odom']['valid_fraction'] == pytest.approx(
        1 - len(broken) / len(odom_times), abs=1e-4)
    assert doc['metrics']['distance_m'] is not None and math.isfinite(doc['metrics']['distance_m'])


def test_a_wheel_speed_beyond_any_the_car_reaches_is_invalid(tmp_path, driving):
    # oracle: spec, measurement.valid.max_speed_mps = 15; exactly 15 is still valid.
    odom_times = [t for topic, t, _ in driving if topic == '/odom']
    fast = {t: v for t, v in zip(odom_times[::50], [15.0, 15.5] * 1000)}
    messages = [(topic, t, fake.odom(t, fast[t]) if topic == '/odom' and t in fast else m)
                for topic, t, m in driving]
    _, doc = classify(tmp_path, messages)
    over = sum(1 for v in fast.values() if v > 15.0)
    assert doc['streams']['/odom']['valid_fraction'] == pytest.approx(1 - over / len(odom_times), abs=1e-4)
    assert doc['metrics']['max_speed_mps'] == 15.0


def test_zero_header_stamps_are_stamp_problems(tmp_path, driving):
    scans = [t for topic, t, _ in driving if topic == '/scan']
    zeroed = set(scans[::4])

    def stamp_zero(t):
        message = fake.scan(t)
        message.header = fake.header(0.0)
        return message
    messages = [(topic, t, stamp_zero(t) if topic == '/scan' and t in zeroed else m) for topic, t, m in driving]
    _, doc = classify(tmp_path, messages)
    assert doc['streams']['/scan']['stamp_problem_fraction'] == pytest.approx(
        len(zeroed) / len(scans), abs=1e-4)
    # 0.25 is past the conditional limit of 0.1.
    assert doc['signals']['scan']['status'] == 'unusable'


def test_a_bag_that_breaks_early_is_unusable_but_left_alone(tmp_path, driving):
    run_dir, doc = classify(tmp_path, driving, fail_after=len(driving) // 2)
    assert doc['recording']['flags'] == ['read_failed']
    assert doc['summary'] == dict.fromkeys(APPS, 'unusable')
    assert doc['recording']['messages_read'] == len(driving) // 2
    assert "OSError: bag file truncated" in doc['applications']['localization']['explanation']
    assert (run_dir / 'bag' / 'bag_0.mcap').read_bytes() == b'not read by the fake reader'


def test_a_bag_that_breaks_near_its_end_caps_every_use_at_conditional(tmp_path, driving):
    # oracle: spec, read_min_fraction 0.9 in the shipped rules; exactly 90 % read is enough.
    assert RULES['read_min_fraction'] == 0.9
    messages = driving[:len(driving) // 10 * 10]
    stop = len(messages) * 9 // 10
    _, doc = classify(tmp_path, messages, fail_after=stop)
    assert doc['recording']['flags'] == ['read_incomplete']
    assert doc['summary'] == dict.fromkeys(APPS, 'conditional')
    _, below = classify(tmp_path / 'below', messages, fail_after=stop - 1)
    assert below['recording']['flags'] == ['read_failed']


def test_a_bag_never_closed_is_unusable_and_says_how_to_recover_it(tmp_path, driving):
    run_dir, doc = classify(tmp_path, driving, metadata=False)
    assert doc['recording']['flags'] == ['bag_not_closed']
    assert doc['summary'] == dict.fromkeys(APPS, 'unusable')
    assert 'ros2 bag reindex' in doc['applications']['velocity_estimation']['explanation']
    assert (run_dir / 'bag' / 'bag_0.mcap').exists()


def test_a_folder_with_no_bag_is_unusable(tmp_path):
    run_dir, doc = classify(tmp_path, [], bag=False)
    assert doc['recording']['flags'] == ['no_bag']
    assert doc['summary'] == dict.fromkeys(APPS, 'unusable')
    assert (run_dir / 'qualification.yaml').is_file()


@pytest.mark.parametrize('record, flag, expected', [
    ({'superseded': True}, 'superseded', dict.fromkeys(APPS, 'unusable')),
    ({'recorder_killed': True}, 'recorder_killed', dict.fromkeys(APPS, 'conditional')),
    ({'bag_exit_code': 2}, 'bag_recorder_failed', dict.fromkeys(APPS, 'conditional')),
    ({'recorder_failures': [{'process': 'ros2 bag record', 'exit_code': 1}]}, 'bag_recorder_failed',
     dict.fromkeys(APPS, 'conditional')),
    ({'recorder_failures': [{'process': 'race_diag_node', 'exit_code': 1}]}, 'other_recorder_failed',
     dict.fromkeys(APPS, 'usable')),
])
def test_what_the_recorder_reported_caps_the_grades(tmp_path, driving, record, flag, expected):
    _, doc = classify(tmp_path, driving, auto_record=record)
    assert doc['recording']['flags'] == [flag]
    assert doc['summary'] == expected


def test_a_late_start_only_costs_algorithm_accuracy(tmp_path):
    # The car is already at 3 m/s at the first /odom: the bag lost the start.
    messages = [(topic, t, fake.odom(t, 3.0) if topic == '/odom' and t - fake.T0 < 1.0 else m)
                for topic, t, m in fake.drive()]
    _, doc = classify(tmp_path, messages)
    assert doc['activity']['started_late'] is True
    assert doc['recording']['flags'] == ['started_late']
    assert doc['summary'] == dict(dict.fromkeys(APPS, 'usable'), algorithm_accuracy='conditional')


def test_a_car_carried_between_holds_is_only_conditional_for_localization(tmp_path):
    # Two holds; between them the wheels stay still and every range grows by 0.5 m.
    def lb(t):
        return 0.5 <= t - fake.T0 < 15.0 or t - fake.T0 >= 25.0
    moved = np.full(fake.BEAMS, 5.5, dtype=np.float32)
    messages = []
    for topic, t, m in fake.drive(lb=lb):
        rel = t - fake.T0
        if topic == '/scan' and rel >= 20.0:
            m = fake.scan(t, moved)
        elif topic == '/odom' and 15.0 <= rel < 25.0:
            m = fake.odom(t, 0.0)
        messages.append((topic, t, m))
    _, doc = classify(tmp_path, messages)
    assert doc['metrics']['lb_holds'] == 2
    assert doc['metrics']['carried_between_holds'] == 1
    assert doc['summary']['localization'] == 'conditional'
    assert 'carried_between_holds 1, usable needs at most 0' in \
        doc['applications']['localization']['explanation']


def test_the_algorithm_is_unknown_without_any_record_of_it(tmp_path, driving):
    _, doc = classify(tmp_path, driving, auto_record=False)
    assert doc['run_info']['algorithm_nodes'] == UNKNOWN
    assert doc['metrics']['algorithm_known'] == 0
    assert doc['summary']['algorithm_accuracy'] == 'conditional'
    assert 'algorithm_known 0, usable needs at least 1' in \
        doc['applications']['algorithm_accuracy']['explanation']


def test_a_hand_started_run_names_its_algorithm_from_the_parameter_snapshot(tmp_path, driving):
    # record_run.py writes no processes.txt, but snapshot_params saves each running node's parameters.
    run_dir = fake.write_run(tmp_path, driving, auto_record=False)
    params = run_dir / 'metadata' / 'params'
    params.mkdir()
    for node in ('ackermann_mux', 'auto_map_race_node', 'gap_follow_node', 'slam_toolbox', 'urg_node'):
        (params / f'{node}.yaml').write_text('{}\n')
    doc = cl.classify_run(run_dir, RULES, backend=fake.reader(driving))
    assert doc['run_info']['algorithm_nodes'] == ['auto_map_race_node', 'gap_follow_node']
    assert doc['run_info']['algorithm_source'] == 'metadata/params'
    assert doc['run_info']['recorder'] == UNKNOWN  # no auto_record.yaml and no events.jsonl
    assert doc['metrics']['algorithm_known'] == 1


def test_slam_and_the_gamepad_alone_are_not_an_algorithm(tmp_path, driving):
    processes = ('  1  1 Mon Dec  1 12:00:00 2026 /opt/ros/jazzy/lib/slam_toolbox/async_slam_toolbox_node\n'
                 '  2  1 Mon Dec  1 12:00:00 2026 /opt/ros/jazzy/lib/joy_teleop/joy_teleop\n')
    _, doc = classify(tmp_path, driving, auto_record=False, processes=processes)
    assert doc['run_info']['algorithm_nodes'] == ['async_slam_toolbox_node', 'joy_teleop']
    assert doc['metrics']['algorithm_known'] == 0


# --------------------------------------------------------------------------- safety

def test_a_classification_failure_is_recorded_and_the_run_is_untouched(tmp_path, driving):
    run_dir = fake.write_run(tmp_path, driving)
    before = files_and_hashes(run_dir)

    def explode(bag_dir):
        raise RuntimeError('reader crashed')
    doc = cl.classify_run(run_dir, RULES, backend=explode)
    assert doc['error'] == 'RuntimeError: reader crashed'
    assert doc['summary'] == dict.fromkeys(APPS, 'unusable')
    assert doc['applications']['localization']['explanation'] == \
        'unusable: classification failed: RuntimeError: reader crashed'
    assert files_and_hashes(run_dir) == before
    written = yaml.safe_load((run_dir / 'qualification.yaml').read_text())
    assert written['error'] == 'RuntimeError: reader crashed'


def test_grading_changes_nothing_in_the_run_but_adds_its_qualification(tmp_path, driving):
    run_dir = fake.write_run(tmp_path, driving)
    before = files_and_hashes(run_dir)
    cl.classify_run(run_dir, RULES, backend=fake.reader(driving))
    assert files_and_hashes(run_dir) == before
    assert sorted(p.name for p in run_dir.iterdir()) == sorted(
        ['auto_record.yaml', 'bag', 'metadata', 'qualification.yaml'])


def test_regrading_writes_nothing_new(tmp_path, driving):
    run_dir, first = classify(tmp_path, driving)
    path = run_dir / 'qualification.yaml'
    inventory = tmp_path / outputs.INVENTORY
    stamps = (path.stat().st_mtime_ns, inventory.stat().st_mtime_ns)
    text = (path.read_text(), inventory.read_text())
    second = cl.classify_run(run_dir, RULES, backend=fake.reader(driving))
    assert second == first
    assert (path.read_text(), inventory.read_text()) == text
    assert (path.stat().st_mtime_ns, inventory.stat().st_mtime_ns) == stamps
    assert [p.name for p in run_dir.iterdir() if p.name.startswith('.')] == []


def test_a_failed_write_keeps_the_old_file_and_leaves_no_temporary(tmp_path, driving, monkeypatch):
    run_dir, doc = classify(tmp_path, driving)
    path = run_dir / 'qualification.yaml'
    old = path.read_text()

    def refuse(*args):
        raise OSError('disk full')
    monkeypatch.setattr(outputs.os, 'replace', refuse)
    with pytest.raises(OSError, match='disk full'):
        outputs.write_qualification(run_dir, dict(doc, run_id='changed'))
    assert path.read_text() == old
    assert sorted(p.name for p in run_dir.iterdir()) == sorted(
        ['auto_record.yaml', 'bag', 'metadata', 'qualification.yaml'])


def test_a_dry_run_writes_nothing(tmp_path, driving):
    run_dir = fake.write_run(tmp_path, driving)
    cl.classify_run(run_dir, RULES, dry_run=True, backend=fake.reader(driving))
    assert not (run_dir / 'qualification.yaml').exists()
    assert not (tmp_path / outputs.INVENTORY).exists()


def test_the_stored_numbers_regrade_to_the_same_result(tmp_path, driving):
    # The regression check over committed runs depends on this.
    run_dir, doc = classify(tmp_path, fake.drive(drop={'/pf/viz/inferred_pose'}))
    stored = yaml.safe_load((run_dir / 'qualification.yaml').read_text())
    again = cl.regrade(stored, RULES)
    assert again['applications'] == stored['applications']
    assert again['signals'] == stored['signals']
    assert again['recording']['flags'] == stored['recording']['flags']


def test_a_threshold_that_could_not_be_measured_makes_the_use_conditional(tmp_path, driving):
    # Spec (classification_rules.yaml): an unmeasured metric gives if_unknown, conditional by default.
    run_dir, _ = classify(tmp_path, driving)
    stored = yaml.safe_load((run_dir / 'qualification.yaml').read_text())
    assert 'if_unknown' not in RULES['applications']['localization']['criteria']['distance_m']
    stored['metrics']['distance_m'] = None
    again = cl.regrade(stored, RULES)['applications']['localization']
    assert again['status'] == 'conditional'
    assert again['reasons'] == ['conditional: distance_m could not be measured']


def test_stricter_rules_regrade_stored_numbers_differently(tmp_path, driving):
    run_dir, doc = classify(tmp_path, driving)
    stored = yaml.safe_load((run_dir / 'qualification.yaml').read_text())
    data = yaml.safe_load(open(RULES.path))
    data['applications']['localization']['criteria']['distance_m']['at_least'] = {
        'usable': 1000.0, 'conditional': 500.0}
    strict = tmp_path / 'strict.yaml'
    strict.write_text(yaml.safe_dump(data))
    assert cl.regrade(stored, load_rules(strict))['applications']['localization']['status'] == 'unusable'


# --------------------------------------------------------------------------- inventory

def test_the_inventory_has_one_row_per_run_folder_sorted_and_never_duplicated(tmp_path, driving):
    runs = [fake.write_run(tmp_path, driving, car=car, run_id=run_id)
            for car, run_id in [('rb2', '20261201-120000'), ('rb1', '20261201-130000'),
                                ('rb1', '20261201-120000')]]
    (tmp_path / 'rb1' / 'racerbot_runs' / 'map_course-20261201-120000.log').write_text('')
    for _ in range(2):
        for run in runs[:2]:
            cl.classify_run(run, RULES, backend=fake.reader(driving))
    rows = list(__import__('csv').DictReader(open(tmp_path / outputs.INVENTORY)))
    assert [(r['car'], r['run_id']) for r in rows] == [
        ('rb1', '20261201-120000'), ('rb1', '20261201-130000'), ('rb2', '20261201-120000')]
    # The run never graded is listed, as unclassified, not left out.
    assert rows[0]['localization'] == 'unclassified'
    assert rows[1]['localization'] == 'usable' and rows[1]['algorithms'] == 'gap_follow_node'
    assert rows[1]['path'] == 'rb1/racerbot_runs/20261201-130000'


def test_a_car_refreshes_only_its_own_rows(tmp_path, driving):
    mine = fake.write_run(tmp_path, driving, car='rb1')
    other = fake.write_run(tmp_path, driving, car='rb2')
    cl.classify_run(other, RULES, backend=fake.reader(driving))
    cl.classify_run(mine, RULES, backend=fake.reader(driving))
    # The other car's newer grade has not been pulled yet: its committed row stays as it was.
    text = (tmp_path / outputs.INVENTORY).read_text()
    edited = text.replace('rb2,20261201-120000,rb2/racerbot_runs/20261201-120000,auto_record',
                          'rb2,20261201-120000,rb2/racerbot_runs/20261201-120000,EDITED')
    assert edited != text
    (tmp_path / outputs.INVENTORY).write_text(edited)
    outputs.update_inventory(tmp_path, RULES, cars={'rb1'})
    assert 'EDITED' in (tmp_path / outputs.INVENTORY).read_text()
    outputs.rebuild_inventory(tmp_path, RULES)
    assert (tmp_path / outputs.INVENTORY).read_text() == text


def test_an_inventory_with_other_columns_is_rebuilt_whole(tmp_path, driving):
    run = fake.write_run(tmp_path, driving, car='rb2')
    cl.classify_run(run, RULES, backend=fake.reader(driving))
    expected = (tmp_path / outputs.INVENTORY).read_text()
    (tmp_path / outputs.INVENTORY).write_text('car,run_id\nrb2,old\nrb2,old\n')
    outputs.update_inventory(tmp_path, RULES, cars={'rb1'})
    assert (tmp_path / outputs.INVENTORY).read_text() == expected


def test_a_run_outside_a_data_folder_gets_no_inventory(tmp_path, driving):
    run = tmp_path / 'elsewhere' / '20261201-120000'
    built = fake.write_run(tmp_path / 'scratch', driving)
    run.parent.mkdir()
    built.rename(run)
    doc = cl.classify_run(run, RULES, backend=fake.reader(driving))
    assert doc['car'] == UNKNOWN
    assert (run / 'qualification.yaml').is_file()
    assert not list(tmp_path.rglob(outputs.INVENTORY))


# --------------------------------------------------------------------------- run_info

def test_an_old_run_says_unknown_rather_than_guessing(tmp_path, driving):
    run_dir = fake.write_run(tmp_path, driving, auto_record=False)
    (run_dir / 'metadata' / 'car_id.txt').unlink()
    info = read_run_info(run_dir)
    assert info['source'].startswith('reconstructed')
    for field in ('recorder', 'started', 'car_id', 'host', 'workspace_commit', 'algorithms',
                  'topics', 'algorithm_nodes', 'algorithm_source', 'bag_start_utc'):
        assert info[field] == UNKNOWN, field
    assert not (run_dir / 'run_info.yaml').exists()


def test_an_old_run_keeps_what_its_files_do_say(tmp_path, driving):
    run_dir = fake.write_run(tmp_path, driving)
    git = run_dir / 'metadata' / 'git'
    git.mkdir()
    (git / 'racerbot-ws.txt').write_text('repository: /x\nHEAD: 2052bab3cccd8b1ee340b78147fd6d314cdae6bd\n')
    (run_dir.parent / 'map_course-20261201-115958.log').write_text('')
    info = read_run_info(run_dir, {'starting_time': {'nanoseconds_since_epoch': 1_800_000_000_000_000_000}})
    assert info['recorder'] == 'auto_record'
    assert info['car_id'] == 'rb1' and info['car_id_source'] == 'metadata/car_id.txt'
    assert info['workspace_commit'] == '2052bab3cccd8b1ee340b78147fd6d314cdae6bd'
    assert info['algorithm_nodes'] == ['gap_follow_node'] and info['algorithm_source'] == 'auto_record.yaml'
    assert info['session_log'] == 'map_course-20261201-115958.log'
    # oracle: 1.8e9 s since the epoch is 2027-01-15T08:00:00 UTC.
    assert info['bag_start_utc'] == '2027-01-15T08:00:00+00:00'


def test_a_written_run_info_is_what_the_classifier_reads(tmp_path, driving):
    run_dir = fake.write_run(tmp_path, driving, auto_record=False)
    write_run_info(run_dir, recorder='record_run', started='2026-12-01T12:00:00-08:00',
                   algorithms={'/drive': ['/pure_pursuit_node']}, car_id='rb2',
                   topics=['/scan', '/odom'], workspace_commit='abc1234', host='racerbot2')
    doc = cl.classify_run(run_dir, RULES, backend=fake.reader(driving))
    info = doc['run_info']
    assert info['source'] == 'run_info.yaml'
    assert (info['recorder'], info['car_id'], info['host']) == ('record_run', 'rb2', 'racerbot2')
    assert info['algorithm_nodes'] == ['pure_pursuit_node']
    assert doc['metrics']['algorithm_known'] == 1
    assert [p.name for p in run_dir.iterdir() if '.tmp' in p.name] == []


def test_run_info_fields_not_given_are_unknown(tmp_path):
    run_dir = tmp_path / '20261201-120000'
    run_dir.mkdir()
    write_run_info(run_dir, recorder='auto_record')
    info = yaml.safe_load((run_dir / 'run_info.yaml').read_text())
    assert info['car_id'] == UNKNOWN and info['algorithms'] == UNKNOWN
    assert info['workspace_commit'] == UNKNOWN and info['topics'] == UNKNOWN
    assert info['run_id'] == '20261201-120000' and info['recorder'] == 'auto_record'


def test_a_broken_rules_file_is_refused_before_any_run_is_graded(tmp_path):
    data = yaml.safe_load(open(RULES.path))
    data['applications']['localization']['required']['scna'] = {}
    path = tmp_path / 'rules.yaml'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(RulesError, match='applications.localization.required: no signal scna'):
        load_rules(path)
