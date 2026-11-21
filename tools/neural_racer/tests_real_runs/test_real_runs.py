"""Offline runner: python -m pytest tools/neural_racer/tests_real_runs

Synthetic CDR messages carry only the fields that the replay consumes.
"""
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from mcap_ros2.writer import Writer

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'src/neural_racer/test'))
import nr_testutil
import real_runs as rr
from real_run_dataset import MS, TOPICS, aligned_ticks, bag_info, extract, messages

SEP = '\n' + '=' * 80 + '\nMSG: '
HEADER = SEP + 'std_msgs/Header\nbuiltin_interfaces/Time stamp\n' + SEP + 'builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n'
SCHEMAS = {k: 'std_msgs/Header header\n' + s + HEADER for k, s in {
    '/scan': 'float32 angle_min\nfloat32 angle_increment\nfloat32 range_min\nfloat32 range_max\nfloat32[] ranges\n',
    '/joy': 'int32[] buttons\n',
    '/odom': ('geometry_msgs/TwistWithCovariance twist\n' + SEP + 'geometry_msgs/TwistWithCovariance\n'
              'geometry_msgs/Twist twist\n' + SEP + 'geometry_msgs/Twist\ngeometry_msgs/Vector3 linear\n'
              + SEP + 'geometry_msgs/Vector3\nfloat64 x\n'),
    '/ackermann_cmd': ('ackermann_msgs/AckermannDrive drive\n' + SEP + 'ackermann_msgs/AckermannDrive\n'
                       'float32 steering_angle\nfloat32 speed\n')
}.items()}


def event(topic, t, stamp=None, **kwargs):
    values = {'/scan': dict(ranges=[4.] * 1081, range_min=.02, range_max=30.,
                             angle_min=-2.3561945, angle_increment=.00436332),
              '/joy': dict(buttons=[0, 0, 0, 0, 1]),
              '/ackermann_cmd': dict(drive=dict(steering_angle=.2, speed=2.)),
              '/odom': dict(twist=dict(twist=dict(linear=dict(x=1.5))))}[topic]
    h = (t if stamp is None else stamp) * MS
    values.update(header=dict(stamp=dict(sec=h // 10**9, nanosec=h % 10**9)), **kwargs)
    return topic, t * MS, values


def obj(d):
    return SimpleNamespace(**{k: obj(v) if isinstance(v, dict) else v for k, v in d.items()})


def align(events):
    counts = Counter()
    ticks = list(aligned_ticks(((k, t, obj(v)) for k, t, v in events), counts, set()))
    return ticks, counts


def write_bag(data, car, run, chunks):
    bag = data / car / 'racerbot_runs' / run / 'bag'
    bag.mkdir(parents=True)
    for name, events in chunks.items():
        with Writer(str(bag / name)) as writer:
            schemas = {k: writer.register_msgdef(TOPICS[k], SCHEMAS[k]) for k in TOPICS}
            for topic, t, data in events:
                writer.write_message(topic, schemas[topic], data, log_time=t, publish_time=t)
    (bag / 'metadata.yaml').write_text(yaml.safe_dump({
        'rosbag2_bagfile_information': {'relative_file_paths': list(chunks)}}))
    return bag


def test_split_mcap_pairs_only_the_first_fresh_mux_command(tmp_path):
    data = tmp_path / 'data'
    # Causal oracle: the odom at 14 ms is too late for the scan at 12 ms.
    write_bag(data, 'rb1', '20261030-175701', {
        'bag_2.mcap': [event('/joy', 10), event('/odom', 11), event('/scan', 12)],
        'bag_10.mcap': [event('/odom', 14, twist=dict(twist=dict(linear=dict(x=9)))),
                        event('/ackermann_cmd', 17), event('/ackermann_cmd', 18),
                        event('/scan', 37), event('/ackermann_cmd', 40)]})
    out = tmp_path / 'aligned'
    extract(data, out, HERE / 'real_run_context.csv')
    r, = json.loads((out / 'manifest.json').read_text())['runs']
    assert (r['traffic'], r['counts'], r['missing_topics']) == ('two-car', {'accepted': 2, 'scans': 2}, [])
    with np.load(out / r['file']) as d:
        np.testing.assert_array_equal(d['time_ns'], np.array([12, 37]) * MS)
        np.testing.assert_array_equal(d['label_delay_ns'], np.array([5, 3]) * MS)
        np.testing.assert_array_equal(d['speed'], [1.5, 9.])
        np.testing.assert_array_equal(d['reset'], [True, False])


@pytest.mark.parametrize('events,missing', [
    ([], sorted(TOPICS)),
    ([event('/scan', 12), event('/ackermann_cmd', 13)], ['/joy', '/odom'])])
def test_unusable_run_is_counted_not_scored(tmp_path, events, missing):
    write_bag(tmp_path, 'rb2', 'example', {'bag_0.mcap': events})
    out = tmp_path / 'out'
    extract(tmp_path, out, HERE / 'real_run_context.csv')
    r, = json.loads((out / 'manifest.json').read_text())['runs']
    assert (r['file'], r['missing_topics'], r['traffic']) == (None, missing, 'unknown')


def test_refuses_missing_chunk(tmp_path):
    bag = write_bag(tmp_path, 'rb2', 'example', {'bag_0.mcap': []})
    (bag / 'bag_0.mcap').unlink()
    with pytest.raises(ValueError, match='chunks do not match'):
        list(messages(bag, bag_info(bag)))


@pytest.mark.parametrize('events,reason', [
    ([event('/joy', 100, buttons=[])], 'deadman'),
    ([event('/joy', 100, buttons=[0] * 5)], 'deadman'),
    ([event('/joy', 100), event('/joy', 109, stamp=99)], 'deadman'),
    ([event('/odom', 108, stamp=8)], 'odom'),  # exactly 100 ms old
    ([event('/odom', 108, twist=dict(twist=dict(linear=dict(x=float('nan')))))], 'odom'),
    ([], 'scan_time'),
    ([event('/scan', 128, ranges=[1.] * 1080)], 'scan_geometry'),
    ([event('/scan', 128, angle_min=-1.0)], 'scan_geometry'),
])
def test_untrustworthy_inputs_reject_scan(events, reason):
    sensor = events[-1:] if events and events[-1][0] == '/scan' else []
    inputs = [event('/joy', 98), event('/odom', 99)] + (events[:-1] if sensor else events)
    scan = sensor or [event('/scan', 128, stamp=52 if not events else 128)]
    ticks, counts = align(inputs + scan + [event('/ackermann_cmd', 135)])
    assert not ticks and counts[reason] == 1


def test_label_window_and_release_break_history():
    inputs = [event('/joy', 20), event('/odom', 21), event('/scan', 22),
              event('/ackermann_cmd', 47),  # exactly 25 ms
              event('/scan', 48), event('/ackermann_cmd', 74),  # too late
              event('/scan', 75), event('/joy', 76, buttons=[0] * 5),
              event('/joy', 77), event('/ackermann_cmd', 78),
              event('/scan', 79), event('/ackermann_cmd', 80, stamp=73),  # stale source
              event('/ackermann_cmd', 83)]
    ticks, counts = align(inputs)
    assert [t[0] for t in ticks] == [22 * MS, 79 * MS]
    assert [t[1] for t in ticks] == [25 * MS, 4 * MS]
    assert all(t[-1] for t in ticks)
    assert counts['released_before_label'] == counts['unmatched'] == 1


def test_backward_chunk_clock_refused():
    with pytest.raises(ValueError, match='log time went backwards'):
        align([event('/joy', 2), event('/scan', 1)])


def test_overlapping_scans_match_once_in_chronological_order():
    ticks, counts = align([
        event('/joy', 10), event('/odom', 11), event('/scan', 20), event('/scan', 40),
        event('/ackermann_cmd', 41, stamp=38), event('/ackermann_cmd', 42),
        event('/scan', 60), event('/scan', 80), event('/ackermann_cmd', 81)])
    # The first label predates scan 40. The last label fits both scans:
    # use the newest one, discard scan 60, and reset history across it.
    assert [t[0] for t in ticks] == [20 * MS, 40 * MS, 80 * MS]
    assert [t[-1] for t in ticks] == [True, False, True]
    assert counts['unmatched'] == 1


def test_error_sign_and_speed_bucket_oracle():
    metric = rr.Metrics()
    for target, pred in [((.12, 2), (.07, 1)), ((-.08, 1), (.02, 3)), ((0., 0), (.03, 0))]:
        metric.add(pred, np.array(target))
    report = metric.report()
    assert report['steer_mae_rad'] == pytest.approx((.05 + .10 + .03) / 3, abs=1e-9)  # rad
    assert report['speed_mae_mps'] == 1 and report['turn_sign_accuracy'] == .5
    assert [rr.speed_bucket(v) for v in [-.1, 0., .99, 1., 2.99, 3.]] == [
        'reverse', '[0,1)', '[0,1)', '[1,3)', '[1,3)', '[3,inf)']


def test_calibration_closed_form_bounds_and_unseen_speed():
    cal = rr.SteeringCalibration()
    for _ in range(100):
        cal.add(1.5, .1, .15)
        cal.add(4.0, .2, -.2)
        cal.add(.5, .1, .3)
    # Unit penalty at gain 1: (1 + 100*.1*.15) / (1 + 100*.1**2).
    assert rr.calibrated_steer(.1, 1.5, cal.gains()) == pytest.approx(.125, abs=1e-9)  # rad
    assert rr.calibrated_steer(.1, 4.0, cal.gains()) == .05 and rr.calibrated_steer(.1, -1., cal.gains()) == .1
    assert cal.gains()['[0,1)'] == 1.5 and rr.calibrated_steer(-.3, .5, cal.gains()) == -.33
    assert np.isinf(rr.calibrated_steer(np.inf, 4., cal.gains()))


def test_replay_history_is_causal_and_resets():
    policy = rr.PolicyNumpy(nr_testutil.random_weights(width=nr_testutil.HALF))
    data = dict(time_ns=np.arange(5) * 25 * MS, speed=np.ones(5), ranges=np.full((5, 1081), 4.),
                geometry=np.tile([-2.3561945, .00436332, .02, 30.], (5, 1)),
                action=np.array([[.1, 2.], [-.1, 3.], [.2, 4.], [.1, 1.], [0., 0.]]),
                reset=[True, False, True, False, False])
    data['ranges'][3] = np.inf
    ticks = list(rr.replay(data, policy, rr.wrapper_for('rb1', 1.)))
    # Identical scan after reset must give identical raw output despite different current target.
    np.testing.assert_array_equal(ticks[0].raw, ticks[2].raw)
    np.testing.assert_array_equal(ticks[0].raw, ticks[4].raw)
    assert ticks[3].result.stopped
    assert not np.array_equal(ticks[0].raw, ticks[1].raw)
    # Evaluation must be independent of ALL recorded labels, including previous ticks.
    data['action'] *= -1
    changed = list(rr.replay(data, policy, rr.wrapper_for('rb1', 1.)))
    np.testing.assert_array_equal([t.raw for t in ticks], [t.raw for t in changed])


def test_holdout_is_entire_days_across_cars(tmp_path):
    model, _ = nr_testutil.write_model(tmp_path, nr_testutil.random_weights(), nr_testutil.card())
    events = [event('/joy', 1), event('/odom', 2), event('/scan', 3), event('/ackermann_cmd', 5)]
    for car, run in [('rb1', '20261026-a'), ('rb2', '20261026-b'), ('rb2', '20261030-c')]:
        write_bag(tmp_path, car, run, {'bag_0.mcap': events})
    out = tmp_path / 'dataset'
    extract(tmp_path, out, HERE / 'real_run_context.csv')
    report = rr.evaluate(out, model, 1., ['20261026'])
    assert report['fit_ticks'] == 2 and report['evaluated_runs'] == ['rb2/20261030-c']
    assert report['groups']['all']['adapted']['ticks'] == 1
    assert report['groups']['all']['adapted']['speed_mae_mps'] >= 1.  # target 2, cap 1 m/s
    with pytest.raises(ValueError, match='no eligible evaluation ticks'):
        rr.evaluate(out, model, 1., ['20261026', '20261030'])
