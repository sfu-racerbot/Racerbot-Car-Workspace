"""
Synthetic runs for the dataset classifier's tests: a run folder on disk with
a bag/metadata.yaml, and a stand-in for the bag reader that yields
SimpleNamespace messages shaped like the real ones. No ROS, no rosbags.

The reader is the one thing replaced (the bag file is a process boundary,
CLAUDE.md A1); everything above it -- reducers, metrics, grading, writing --
is the real code. The real reader is exercised on recorded bags in
tools/run_analysis/tests/test_dataset_classifier_bag.py.

A drive() run is built so every value is known in closed form:
  * odometry speed v(t) = 3 + 1.5 sin(2 pi t / 10) m/s after a 1 s standstill
  * steering delta(t) = 0.2 sin(2 pi t / 4) rad, yaw rate v tan(delta) / L
  * LB held from 0.5 s to the end, one hold
"""

from contextlib import contextmanager
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import yaml

T0 = 1_800_000_000.0
WHEELBASE = 0.36   # src/f1tenth_system/f1tenth_stack/config/vesc.yaml
BEAMS = 1081
GRAVITY = 9.80665
TYPES = {
    '/scan': 'sensor_msgs/msg/LaserScan', '/odom': 'nav_msgs/msg/Odometry',
    '/tf': 'tf2_msgs/msg/TFMessage', '/tf_static': 'tf2_msgs/msg/TFMessage',
    '/imu/data': 'sensor_msgs/msg/Imu', '/imu/vesc/data': 'sensor_msgs/msg/Imu',
    '/sensors/imu/raw': 'sensor_msgs/msg/Imu', '/sensors/core': 'vesc_msgs/msg/VescStateStamped',
    '/ackermann_cmd': 'ackermann_msgs/msg/AckermannDriveStamped',
    '/drive': 'ackermann_msgs/msg/AckermannDriveStamped',
    '/steering/measured': 'sensor_msgs/msg/JointState', '/joy': 'sensor_msgs/msg/Joy',
    '/drive_intent': 'std_msgs/msg/String', '/pf/viz/inferred_pose': 'geometry_msgs/msg/PoseStamped',
    '/rosout': 'rcl_interfaces/msg/Log',
}
RATES = {'/scan': 40, '/odom': 50, '/tf': 50, '/imu/data': 100, '/sensors/core': 50,
         '/ackermann_cmd': 40, '/drive': 40, '/steering/measured': 50, '/joy': 20,
         '/drive_intent': 10, '/pf/viz/inferred_pose': 30, '/rosout': 1}


def header(t):
    sec = math.floor(t)
    return NS(stamp=NS(sec=int(sec), nanosec=int(round((t - sec) * 1e9)) % 1_000_000_000),
              frame_id='')


def vec(x, y, z):
    return NS(x=x, y=y, z=z)


def speed(t, moving=True):
    rel = t - T0
    return 0.0 if (not moving or rel < 1.0) else 3.0 + 1.5 * math.sin(2 * math.pi * rel / 10.0)


def steering(t):
    return 0.2 * math.sin(2 * math.pi * (t - T0) / 4.0)


def yaw(t, moving=True):
    return speed(t, moving) * math.tan(steering(t)) / WHEELBASE


SCAN_RANGES = np.full(BEAMS, 5.0, dtype=np.float32)


def scan(t, ranges=None):
    return NS(header=header(t), ranges=SCAN_RANGES if ranges is None else ranges,
              range_min=0.02, range_max=30.0)


def odom(t, v):
    return NS(header=header(t), twist=NS(twist=NS(linear=vec(v, 0.0, 0.0), angular=vec(0, 0, 0))))


def transform(t, parent, child, x=0.0):
    return NS(header=NS(stamp=header(t).stamp, frame_id=parent), child_frame_id=child,
              transform=NS(translation=vec(x, 0.0, 0.0), rotation=NS(x=0.0, y=0.0, z=0.0, w=1.0)))


def tf(t, parent='odom', child='base_link'):
    return NS(transforms=[transform(t, parent, child)])


def imu(t, gyro, accel):
    return NS(header=header(t), angular_velocity=vec(*gyro), linear_acceleration=vec(*accel))


def car_imu(t, moving=True):
    w = yaw(t, moving)
    return imu(t, (0.0, 0.0, w), (0.0, speed(t, moving) * w, GRAVITY))


def vesc(t):
    return NS(header=header(t), state=NS(speed=0.0, voltage_input=11.1, duty_cycle=0.1))


def ackermann(t, steer, v):
    return NS(header=header(t), drive=NS(steering_angle=steer, speed=v))


def joint_state(t, position):
    return NS(header=header(t), name=['steering'], position=[position])


def joy(t, lb):
    buttons = [0] * 11
    buttons[4] = 1 if lb else 0
    return NS(header=header(t), buttons=buttons, axes=[0.0] * 8)


def intent(t):
    return NS(data=json.dumps({'v': 1, 'stamp': t, 'node': 'gap_follow_node'}))


def pose(t):
    return NS(header=header(t), pose=NS(position=vec(0.0, 0.0, 0.0),
                                        orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0)))


def ticks(rate, duration, offset=0.0):
    n = int(round(duration * rate))
    return [T0 + offset + i / rate for i in range(n)]


def drive(duration=40.0, moving=True, drop=(), lb=None, speed_of=None):
    """[(topic, t, message)] for a run in which the car drives, or stands
    still with moving=False. drop: topics left out. lb: f(t) -> held.
    speed_of: f(t) -> wheel speed, in place of the default profile."""
    lb = lb or (lambda t: t - T0 >= 0.5)
    if speed_of is not None:
        global speed
        default, speed = speed, (lambda t, moving=True: speed_of(t) if moving else 0.0)
        try:
            return drive(duration, moving, drop, lb)
        finally:
            speed = default
    out = [('/tf_static', T0, NS(transforms=[transform(T0, 'base_link', 'laser', 0.26)]))]
    makers = {
        '/scan': lambda t: scan(t),
        '/odom': lambda t: odom(t, speed(t, moving)),
        '/tf': lambda t: tf(t),
        '/imu/data': lambda t: car_imu(t, moving),
        '/sensors/core': vesc,
        '/ackermann_cmd': lambda t: ackermann(t, steering(t), speed(t, moving)),
        '/drive': lambda t: ackermann(t, steering(t), speed(t, moving)),
        '/steering/measured': lambda t: joint_state(
            t, steering(t) if speed(t, moving) >= 0.3 else math.nan),
        '/joy': lambda t: joy(t, lb(t)),
        '/drive_intent': intent,
        '/pf/viz/inferred_pose': pose,
        '/rosout': lambda t: NS(),
    }
    for topic, rate in RATES.items():
        for t in ticks(rate, duration, offset=0.001):
            out.append((topic, t, makers[topic](t)))
    out = [m for m in out if m[0] not in drop]
    out.sort(key=lambda m: m[1])
    return out


def write_run(root, messages, *, car='rb1', run_id='20261201-120000', auto_record=None,
              metadata=True, bag=True, processes=None):
    """A run folder data/<car>/racerbot_runs/<run_id>/ holding these messages."""
    run = Path(root) / car / 'racerbot_runs' / run_id
    (run / 'metadata').mkdir(parents=True)
    if bag:
        (run / 'bag').mkdir()
        (run / 'bag' / 'bag_0.mcap').write_bytes(b'not read by the fake reader')
    if bag and metadata:
        times = [m[1] for m in messages] or [T0]
        counts = {}
        for topic, _, _ in messages:
            counts[topic] = counts.get(topic, 0) + 1
        info = {'rosbag2_bagfile_information': {
            'version': 9, 'storage_identifier': 'mcap',
            'duration': {'nanoseconds': int(round((max(times) - min(times)) * 1e9))},
            'starting_time': {'nanoseconds_since_epoch': int(round(min(times) * 1e9))},
            'message_count': len(messages), 'relative_file_paths': ['bag_0.mcap'],
            'topics_with_message_count': [
                {'topic_metadata': {'name': t, 'type': TYPES[t]}, 'message_count': n}
                for t, n in sorted(counts.items())]}}
        (run / 'bag' / 'metadata.yaml').write_text(yaml.safe_dump(info))
    if auto_record is not False:
        record = {'started_by': 'auto_record (bringup); turn off with record:=false',
                  'started': '2026-12-01T12:00:00-08:00',
                  'algorithms': {'/drive': ['/gap_follow_node']},
                  'stop_reason': 'no algorithm publishing for 3.5 s', 'duration_sec': 40.0,
                  'recorder_exit_code': 0, 'bag_exit_code': 0, 'recorder_failures': [],
                  'recorder_killed': False, 'bag_bytes': 1000}
        record.update(auto_record or {})
        (run / 'auto_record.yaml').write_text(yaml.safe_dump(record, sort_keys=False))
    if processes is not None:
        (run / 'metadata' / 'processes.txt').write_text(processes)
    (run / 'metadata' / 'car_id.txt').write_text(car + '\n')
    return run


def reader(messages, fail_after=None, error=OSError('bag file truncated')):
    """A stand-in for bag_stats.open_bag over these messages. fail_after:
    raise `error` after that many messages, as a damaged bag does."""
    topics = {topic: TYPES[topic] for topic, _, _ in messages}

    @contextmanager
    def open_bag(bag_dir):
        def stream():
            for index, (topic, t, message) in enumerate(messages):
                if fail_after is not None and index >= fail_after:
                    raise error
                yield topic, int(round(t * 1e9)), message
        yield topics, stream(), lambda topic, raw: raw
    return open_bag
