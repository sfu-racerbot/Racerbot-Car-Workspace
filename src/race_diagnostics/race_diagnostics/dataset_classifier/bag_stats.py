"""
bag_stats.py -- read a run's bag once and keep what grading needs.

The bag is read message by message and never held in memory: each message is
reduced to a few numbers (its receive time, header stamp, whether it is
valid, and the values the metrics use). A scan is reduced to its beam count
and valid share; whole scans are kept only next to an LB press or release,
for the carried-between-holds check. Only topics some signal (or LB, or the
activity summary) needs are deserialized; every other topic is counted by
receive time alone.

It opens the bag read-only, with rosbag2_py when ROS is sourced (as on the
car) and with rosbags otherwise (the analysis environment in
tools/run_analysis/README.md). Nothing in the run directory is written.
"""

from array import array
from contextlib import contextmanager
import json
import math
from pathlib import Path

import numpy as np

from race_diagnostics.run_activity import LB_BUTTON, lb_state

from .metrics import GRAVITY

# Topics read for the LB and activity summary, whatever the rules say.
JOY = '/joy'
ODOM = '/odom'
SCAN = '/scan'
# The kind of these when no signal names them.
FIXED_KINDS = {JOY: 'joy', ODOM: 'odometry', SCAN: 'scan'}


def bag_metadata(bag_dir):
    """rosbag2_bagfile_information from bag/metadata.yaml, or None if there is none."""
    import yaml
    path = Path(bag_dir) / 'metadata.yaml'
    if not path.is_file():
        return None
    return (yaml.safe_load(path.read_text()) or {}).get('rosbag2_bagfile_information')


@contextmanager
def open_bag(bag_dir, backend='auto'):
    """(topic types, message iterator, deserialize) for a bag directory.

    The iterator yields (topic, receive time ns, raw bytes) in bag order;
    deserialize(topic, raw) turns one into a message. backend is 'rosbag2_py',
    'rosbags', 'auto' (rosbag2_py if importable), or a callable taking the bag
    directory and returning such a context manager (the tests' synthetic bags)."""
    if callable(backend):
        with backend(bag_dir) as opened:
            yield opened
        return
    if backend in ('auto', 'rosbag2_py'):
        try:
            import rosbag2_py
            from rclpy.serialization import deserialize_message
            from rosidl_runtime_py.utilities import get_message
        except ImportError:
            if backend == 'rosbag2_py':
                raise
        else:
            meta = bag_metadata(bag_dir) or {}
            storage = meta.get('storage_identifier') or (
                'mcap' if any(Path(bag_dir).glob('*.mcap')) else 'sqlite3')
            reader = rosbag2_py.SequentialReader()
            reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage),
                        rosbag2_py.ConverterOptions('cdr', 'cdr'))
            types = {t.name: t.type for t in reader.get_all_topics_and_types()}
            classes = {}

            def messages():
                while reader.has_next():
                    yield reader.read_next()

            def deserialize(topic, raw):
                if topic not in classes:
                    classes[topic] = get_message(types[topic])
                return deserialize_message(raw, classes[topic])

            try:
                yield types, messages(), deserialize
            finally:
                del reader
            return
    from rosbags.highlevel import AnyReader
    with AnyReader([Path(bag_dir)]) as reader:
        types = {c.topic: c.msgtype for c in reader.connections}
        msgtypes = dict(types)

        def messages():
            for connection, stamp_ns, raw in reader.messages():
                yield connection.topic, stamp_ns, raw

        def deserialize(topic, raw):
            return reader.deserialize(raw, msgtypes[topic])

        yield types, messages(), deserialize


class Stream:
    """One topic (or one transform on /tf) reduced to arrays."""

    def __init__(self, key, topic, msgtype, kind=None, fields=()):
        self.key, self.topic, self.msgtype, self.kind = key, topic, msgtype, kind
        self.t = array('d')
        self.stamp = array('d')     # NaN where the message has no stamp
        self.valid = bytearray()
        self.values = {name: array('d') for name in fields}
        self.stamps_checked = kind is not None

    def add(self, t, stamp, valid, values=None):
        self.t.append(t)
        self.stamp.append(math.nan if stamp is None else stamp)
        self.valid.append(1 if valid else 0)
        for name, value in (values or {}).items():
            self.values[name].append(value)

    def mask(self):
        """True for each valid message."""
        return np.frombuffer(bytes(self.valid), dtype=np.uint8).astype(bool)

    def times(self, valid_only=True):
        t = np.frombuffer(self.t, dtype=float)
        return t[self.mask()] if valid_only else t

    def column(self, name, valid_only=True):
        values = np.frombuffer(self.values[name], dtype=float)
        return values[self.mask()] if valid_only else values

    def invalidate(self, bad):
        """Mark the messages where bad is True invalid."""
        for index in np.flatnonzero(bad):
            self.valid[index] = 0


def _stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _finite(*values):
    return all(math.isfinite(v) for v in values)


# Each reducer: (message, rules, topic format) -> (stamp or None, valid, values).
def _scan(msg, rules, fmt):
    ranges = np.asarray(msg.ranges, dtype=np.float32)
    inside = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
    share = float(np.mean(inside)) if len(ranges) else 0.0
    valid = share >= rules['measurement']['valid']['scan_min_beam_fraction']
    return _stamp(msg.header), valid, {'beams': float(len(ranges)), 'beam_share': share}


def _odometry(msg, rules, fmt):
    v = msg.twist.twist.linear.x
    valid = _finite(v) and abs(v) <= rules['measurement']['valid']['max_speed_mps']
    return _stamp(msg.header), valid, {'v': v}


def _imu(msg, rules, fmt):
    gyro_scale = math.radians(1.0) if fmt['gyro_unit'] == 'deg/s' else 1.0
    accel_scale = GRAVITY if fmt['accel_unit'] == 'g' else 1.0
    w, a = msg.angular_velocity, msg.linear_acceleration
    gyro = (w.x * gyro_scale, w.y * gyro_scale, w.z * gyro_scale)
    accel = (a.x * accel_scale, a.y * accel_scale, a.z * accel_scale)
    limits = rules['measurement']['valid']
    valid = (_finite(*gyro, *accel)
             and any(v != 0.0 for v in gyro + accel)  # all six zero: IMU off or not fitted
             and math.hypot(*accel) <= limits['max_accel_mps2']
             and math.hypot(*gyro) <= limits['max_gyro_radps'])
    return _stamp(msg.header), valid, dict(zip(('wx', 'wy', 'wz', 'ax', 'ay', 'az'), gyro + accel))


def _vesc_state(msg, rules, fmt):
    state = msg.state
    valid = _finite(state.speed, state.voltage_input, state.duty_cycle)
    return _stamp(msg.header), valid, None


def _ackermann(msg, rules, fmt):
    steering, speed = msg.drive.steering_angle, msg.drive.speed
    limits = rules['measurement']['valid']
    valid = (_finite(steering, speed) and abs(steering) <= limits['max_steering_rad']
             and abs(speed) <= limits['max_speed_mps'])
    return _stamp(msg.header), valid, {'steering': steering}


def _joint_state(msg, rules, fmt):
    # NaN is a valid reading here: vehicle_state_node publishes NaN below
    # 0.3 m/s and when its gyros disagree (src/imu_bridge/README.md).
    names = list(msg.name)
    if 'steering' not in names or len(msg.position) <= names.index('steering'):
        return _stamp(msg.header), False, {'position': math.nan}
    return _stamp(msg.header), True, {'position': float(msg.position[names.index('steering')])}


def _joy(msg, rules, fmt):
    buttons = list(msg.buttons)
    return _stamp(msg.header), len(buttons) > LB_BUTTON, {'lb': float(lb_state(buttons))}


def _drive_intent(msg, rules, fmt):
    try:
        intent = json.loads(msg.data)
    except (ValueError, TypeError):
        return None, False, None
    if not isinstance(intent, dict):
        return None, False, None
    stamp = intent.get('stamp')
    stamp = float(stamp) if isinstance(stamp, (int, float)) and not isinstance(stamp, bool) else None
    return stamp, intent.get('v') == fmt.get('version', 1), None


def _pose(msg, rules, fmt):
    pose = msg.pose.pose if hasattr(msg.pose, 'pose') else msg.pose
    p, q = pose.position, pose.orientation
    return _stamp(msg.header), _finite(p.x, p.y, p.z, q.x, q.y, q.z, q.w), None


REDUCERS = {'scan': _scan, 'odometry': _odometry, 'imu': _imu, 'vesc_state': _vesc_state,
            'ackermann': _ackermann, 'joint_state': _joint_state, 'joy': _joy,
            'drive_intent': _drive_intent, 'pose': _pose}
FIELDS = {'scan': ('beams', 'beam_share'), 'odometry': ('v',),
          'imu': ('wx', 'wy', 'wz', 'ax', 'ay', 'az'), 'ackermann': ('steering',),
          'joint_state': ('position',), 'joy': ('lb',)}


def transform_key(topic, parent, child):
    return f'{topic} {parent}->{child}'


def _transform_valid(transform):
    t, r = transform.transform.translation, transform.transform.rotation
    if not _finite(t.x, t.y, t.z, r.x, r.y, r.z, r.w):
        return False
    return abs(math.sqrt(r.x ** 2 + r.y ** 2 + r.z ** 2 + r.w ** 2) - 1.0) < 0.01


class CarryTracker:
    """The scans next to each LB press and release, kept as the bag streams by.

    moved_between() looks only at the last scan at or before a release and the
    first scan at or after a press, so these snapshots give it the same
    answer the whole scan array would (tested in test_dataset_classifier_bag.py)."""

    def __init__(self):
        self.held = False
        self.last_scan = None          # (t, ranges)
        self.scan_at_last_joy = None
        self.waiting_for_press_scan = False
        self.snapshots = {}            # t -> ranges

    def scan(self, t, ranges):
        self.last_scan = (t, ranges)
        if self.waiting_for_press_scan:
            self.snapshots[t] = ranges
            self.waiting_for_press_scan = False

    def joy(self, t, held):
        if held and not self.held:
            if self.last_scan is not None and self.last_scan[0] == t:
                self.snapshots[t] = self.last_scan[1]
            else:
                self.waiting_for_press_scan = True
        elif not held and self.held and self.last_scan is not None:
            self.snapshots[self.last_scan[0]] = self.last_scan[1]
        self.held = held
        self.scan_at_last_joy = self.last_scan

    def finish(self):
        """(times, scans) sorted by time, closing a hold still open at the last /joy."""
        if self.held and self.scan_at_last_joy is not None:
            self.snapshots[self.scan_at_last_joy[0]] = self.scan_at_last_joy[1]
        times = sorted(self.snapshots)
        return np.array(times, dtype=float), [self.snapshots[t] for t in times]


def read_streams(bag_dir, rules, backend='auto'):
    """Read the bag once. Returns (streams by key, carry tracker, read report).

    The report is {'messages_read', 'read_error'}; a bag that fails part way
    returns what was read before the failure, with the error, never raises
    for it. Failing to open the bag at all does raise."""
    wanted = {}       # topic -> kind, for topics a signal or the summary reads
    transforms = {}   # topic -> [(parent, child)]
    for signal in rules['signals'].values():
        for topic in signal['topics']:
            if signal['kind'] == 'transform':
                transforms.setdefault(topic, []).append(tuple(signal['transform']))
            else:
                wanted[topic] = signal['kind']
    for topic, kind in FIXED_KINDS.items():
        wanted.setdefault(topic, kind)
    streams, carry = {}, CarryTracker()
    report = {'messages_read': 0, 'read_error': None}
    with open_bag(bag_dir, backend) as (types, messages, deserialize):
        for topic, msgtype in types.items():
            kind = wanted.get(topic)
            streams[topic] = Stream(topic, topic, msgtype, kind, FIELDS.get(kind, ()))
            for parent, child in transforms.get(topic, ()):
                key = transform_key(topic, parent, child)
                streams[key] = Stream(key, topic, msgtype, 'transform')
        try:
            for topic, stamp_ns, raw in messages:
                report['messages_read'] += 1
                t = stamp_ns * 1e-9
                stream = streams[topic]
                if topic in transforms:
                    message = deserialize(topic, raw)
                    stream.add(t, None, True)
                    for transform in message.transforms:
                        parent = transform.header.frame_id.lstrip('/')
                        child = transform.child_frame_id.lstrip('/')
                        key = transform_key(topic, parent, child)
                        if key in streams:
                            streams[key].add(t, _stamp(transform.header), _transform_valid(transform))
                    continue
                if stream.kind is None:
                    stream.add(t, None, True)
                    continue
                message = deserialize(topic, raw)
                stamp, valid, values = REDUCERS[stream.kind](message, rules, rules.topic_format(topic))
                stream.add(t, stamp, valid, values)
                if topic == SCAN:
                    carry.scan(t, np.asarray(message.ranges, dtype=np.float32))
                elif topic == JOY:
                    carry.joy(t, bool(values['lb']))
        except Exception as e:  # noqa: BLE001 -- a damaged bag is a finding, not a crash
            report['read_error'] = f'{type(e).__name__}: {e}'
    scans = streams.get(SCAN)
    if scans is not None and scans.kind == 'scan' and len(scans.t):
        # A scan with a different beam count from the rest is not the same
        # sensor configuration; replaying it would misplace every beam.
        beams = scans.column('beams', valid_only=False)
        counts, seen = np.unique(beams, return_counts=True)
        scans.invalidate(beams != counts[np.argmax(seen)])
    return streams, carry, report


def stream_stats(stream, start, end, rules, latched=False):
    """The numbers a signal check compares, for one stream."""
    from .metrics import timing
    m = rules['measurement']
    t = stream.times(valid_only=False)
    count = len(t)
    valid = stream.mask()
    rate, max_gap, coverage = timing(t, start, end, m['coverage_gap_s'])
    stamps = np.frombuffer(stream.stamp, dtype=float)
    has = np.isfinite(stamps)
    problem_fraction = None
    if stream.kind is not None and not latched and has.any():
        s, rx = stamps[has], t[has]
        backwards = np.zeros(len(s), dtype=bool)
        backwards[1:] = np.diff(s) < 0
        problems = (s == 0) | (np.abs(rx - s) > m['stamp_tolerance_s']) | backwards
        problem_fraction = float(np.mean(problems))
    return {
        'type': stream.msgtype, 'count': count,
        'valid_fraction': float(np.mean(valid)) if count and stream.kind is not None else None,
        'rate_hz': rate, 'max_gap_s': max_gap, 'coverage': coverage,
        'stamp_problem_fraction': problem_fraction,
    }
