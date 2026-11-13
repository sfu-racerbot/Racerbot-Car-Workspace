"""Reverse escape on the recorded nose-in stop of rb2 20261111-142254 (docs/gap-follow.md#reverse-escape).

Fixture nose_in_rb2_20261111-142254.npz (tools/run_analysis/scan_window_fixture.py): 260 scans of 100.95-107.45 s on the
fixture clock (first /scan or /joy; 101.5-108.0 on intent_report's), with the wheel speed the node had. The car arrives
at 3.6 m/s, brakes and sits 0.22 m from the bumper from 104.7 s (nothing else moves). The scans and speeds are fed
to the real node at their recorded 25 ms cadence with LB held. The recorded scans cannot react to a reverse, so this
checks the decision to start one, and the timing of it. Needs ROS sourced.
"""
import os

import numpy as np
import pytest
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy, LaserScan

from gap_follow.gap_follow_node import GapFollowNode
from racerbot_car.layers import layered

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, 'data', 'nose_in_rb2_20261111-142254.npz')
CONFIG = os.path.join(HERE, '..', 'config', 'gap_follow.yaml')
DT = 0.025


def replay(monkeypatch, extra):
    monkeypatch.setenv('RACERBOT_CAR', 'rb2')
    args = ['--ros-args']
    for file in layered(CONFIG, 'gap_follow.yaml', 'rb2'):
        args += ['--params-file', file]
    rclpy.init(args=[*args, *extra, '-p', 'drive_topic:=/test_only/drive'])
    node = GapFollowNode()
    data = np.load(FIXTURE)
    sent = []
    original = node.drive_pub.publish
    node.drive_pub.publish = lambda m: (sent.append(m.drive.speed), original(m))
    try:
        for i, ranges in enumerate(data['ranges']):
            joy = Joy()
            joy.buttons = [0] * 5
            joy.buttons[4] = 1
            node.joy_callback(joy)
            odom = Odometry()
            odom.twist.twist.linear.x = float(data['odom_speeds'][i])
            node.odom_callback(odom)
            scan = LaserScan()
            scan.angle_min, scan.angle_increment = float(data['angle_min']), float(data['angle_increment'])
            scan.angle_max = scan.angle_min + (len(ranges) - 1) * scan.angle_increment
            scan.range_min, scan.range_max = float(data['range_min']), float(data['range_max'])
            scan.ranges = [float(r) for r in ranges]
            stamp = 100.0 + float(data['times'][i])
            scan.header.stamp.sec, scan.header.stamp.nanosec = int(stamp), int(round((stamp - int(stamp)) * 1e9))
            node.scan_callback(scan)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return data, sent


def stop_index(data):
    """First scan with the recorded wheel speed under 0.1 m/s after the car had been driving above 1 m/s."""
    v = np.abs(data['odom_speeds'])
    moving = np.flatnonzero(v > 1.0)
    return int(moving[-1] + 1 + np.flatnonzero(v[moving[-1] + 1:] < 0.1)[0])


def test_with_the_feature_on_the_first_reverse_comes_one_second_after_the_stop(monkeypatch):
    data, sent = replay(monkeypatch, ['-p', 'enable_reverse_escape:=true'])
    stop = stop_index(data)
    first = next(i for i, s in enumerate(sent) if s < 0.0)
    # recorded measurement: stopped from the 104.7 s mark (fixture 104.1) with 0.43 m of LiDAR range (0.22 m
    # from the bumper); the machine waits wait_sec 1.0 s after the first blocked, stopped tick.
    assert 1.0 <= (first - stop) * DT <= 1.2, (first - stop) * DT                   # s
    assert max(s for s in sent[first:first + 10]) <= 0.0
    assert min(sent[first:first + 10]) >= -0.30 - 1e-12                              # m/s
    assert all(s >= 0.0 for s in sent[:first])


def test_with_the_feature_off_the_same_replay_never_goes_negative(monkeypatch):
    data, sent = replay(monkeypatch, [])
    assert len(sent) == len(data['ranges'])
    assert min(sent) >= 0.0
    assert stop_index(data) > 0


def test_the_recorded_car_was_stopped_for_over_two_seconds_with_zero_commands(monkeypatch):
    # precondition for the two tests above: the stop is real in the replayed (feature off) commands
    data, sent = replay(monkeypatch, [])
    stop = stop_index(data)
    assert all(s == 0.0 for s in sent[stop + 5:stop + 80])                           # 2 s of zero command
