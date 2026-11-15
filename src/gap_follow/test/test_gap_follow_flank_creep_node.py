"""Flank creep in the real GapFollowNode (Task 2 of
docs/superpowers/plans/2026-11-12-post-race-speed-program.md).

A wall 5 cm off the body, parallel, with open road ahead used to latch the
car at speed 0 (rb1 20261111-155542, 87.6-95.9 s). Needs ROS sourced; run as
`rb python3 -m pytest src/gap_follow/test/test_gap_follow_flank_creep_node.py`.

Every node publishes on /test_only/drive (never the real /drive).
"""
import math
import os

import numpy as np
import pytest
import rclpy
from conftest import car_config_args
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from sensor_msgs.msg import Joy, LaserScan

from drive_intent import schema
from gap_follow.gap_follow_node import GapFollowNode

TEST_ONLY_DRIVE = ['-p', 'drive_topic:=/test_only/drive']

PACKAGED_CONFIG = os.path.join(
    os.path.dirname(__file__), '..', 'config', 'gap_follow.yaml')
DEADMAN_BUTTON = 4
OPEN = 3.0  # m, the open road


def _scan(wall_left=None, block=None, n=1081, span=math.radians(270)):
    """LiDAR scan: open to 3 m, a straight wall `wall_left` m to the left of
    the LiDAR running parallel to the car (y = +wall_left), and optionally
    `block=(lo_deg, hi_deg, range)` returned closer."""
    msg = LaserScan()
    msg.angle_min = -span / 2.0
    msg.angle_increment = span / (n - 1)
    msg.angle_max = msg.angle_min + (n - 1) * msg.angle_increment
    msg.range_min, msg.range_max = 0.05, 12.0
    angles = msg.angle_min + np.arange(n) * msg.angle_increment
    ranges = np.full(n, OPEN)
    if wall_left is not None:
        s = np.sin(angles)
        wall = np.where(s > 1e-6, wall_left / np.maximum(s, 1e-6), OPEN)
        ranges = np.minimum(ranges, wall)
    if block is not None:
        lo, hi, r = block
        sel = (angles >= math.radians(lo)) & (angles <= math.radians(hi))
        ranges[sel] = np.minimum(ranges[sel], r)
    msg.ranges = [float(r) for r in ranges]
    return msg


def _make(extra=(), use_config=True, intent=False):
    args = ['--ros-args']
    if use_config:
        args += car_config_args(PACKAGED_CONFIG)
    # -p after --params-file, and the drive remap last, so neither is
    # overridden by the YAML (see test_gap_follow_node.py).
    args += list(extra)
    if intent:
        args += ['-p', 'intent_rate_hz:=0.0']
    rclpy.init(args=[*args, *TEST_ONLY_DRIVE])
    node = GapFollowNode()
    assert node.drive_pub.topic_name == '/test_only/drive'
    return node


@pytest.fixture
def make():
    nodes = []

    def build(*a, **kw):
        node = _make(*a, **kw)
        nodes.append(node)
        return node

    yield build
    for n in nodes:
        n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def _capture(node):
    published = []
    original = node.drive_pub.publish
    node.drive_pub.publish = lambda msg: (published.append(msg), original(msg))
    return published


def _joy(node, held=True):
    msg = Joy()
    msg.buttons = [0] * (DEADMAN_BUTTON + 1)
    msg.buttons[DEADMAN_BUTTON] = 1 if held else 0
    node.joy_callback(msg)


def _odom(node, speed=0.0):
    msg = Odometry()
    msg.twist.twist.linear.x = float(speed)
    node.odom_callback(msg)


def _tick(node, scan, held=True, speed=0.0):
    if held is not None:
        _joy(node, held)
    _odom(node, speed)
    node.scan_callback(scan)


def _run(node, scan, ticks):
    published = _capture(node)
    for _ in range(ticks):
        _tick(node, scan)
    return [m.drive.speed for m in published]


CREEP = ['-p', 'flank_creep_length:=0.5']


def test_wall_5cm_off_the_flank_with_open_road_creeps(make):
    # wall 0.20 m from the LiDAR = 0.045 m off the 0.31 m body
    node = make(CREEP)
    speeds = _run(node, _scan(wall_left=0.20), 10)
    assert any(0.0 < s <= 0.40 for s in speeds), speeds
    assert max(speeds) <= node.escape_creep_speed + 1e-9, speeds


def test_the_latch_is_real_with_flank_creep_off(make):
    """Precondition for the creep test: same scan, shipped config, creep off
    -> the recorded stop (swept clearance 0)."""
    node = make(['-p', 'flank_creep_length:=0.0'])
    assert _run(node, _scan(wall_left=0.20), 20) == [0.0] * 20


def test_flank_creep_off_by_default_keeps_the_stop(make):
    # Code defaults only, no flank_creep_length given. The code's default
    # forward reserve (0.20 m) would let the ordinary creep move this car, so
    # the shipped 0.12 m is set to reproduce the recorded latch.
    node = make(['-p', 'forward_stop_clearance:=0.12'], use_config=False)
    assert node.flank_creep_length == 0.0
    assert _run(node, _scan(wall_left=0.20), 20) == [0.0] * 20


def test_shipped_config_turns_the_creep_on_at_half_a_metre(make):
    node = make()
    assert node.flank_creep_length == pytest.approx(0.50, abs=1e-12)  # plan Task 2
    assert node.flank_creep_margin == pytest.approx(0.02, abs=1e-12)


def test_flank_creep_does_not_start_the_launch_boost(make):
    """The shipped config arms the launch boost. A crawl away from a wall must
    not begin it (full speed, no ramp, no caps)."""
    node = make(CREEP)
    assert node.launch_boost.armed, 'precondition: shipped config arms it'
    _run(node, _scan(wall_left=0.20), 10)
    assert not node.launch_boost.active
    assert node.last_decision_state == 'gap_follow'


def test_wall_inside_the_body_still_stops(make):
    # 0.15 m from the LiDAR is inside the 0.155 m half width: contact
    node = make(CREEP)
    assert _run(node, _scan(wall_left=0.15), 20) == [0.0] * 20


def test_steering_toward_the_flank_wall_still_stops(make):
    # wall 0.20 m to the left; everything ahead and right is blocked at
    # 0.6 m except a gap between +25 and +45 deg, so the gap steers left,
    # into the wall's side.
    node = make(CREEP)
    scan = _scan(wall_left=0.20, block=(-135.0, 24.0, 0.6))
    speeds = _run(node, scan, 20)
    assert speeds == [0.0] * 20, speeds


def test_slew_limited_rack_pointing_at_the_wall_still_stops(make):
    # The rack the car will drive this tick is the slew-limited angle, not
    # the desired one. Desired steering here is away from the wall, but the
    # rack was left at +0.30 rad (toward the wall) a moment ago and cannot
    # swing across in one tick.
    node = make(CREEP)
    published = _capture(node)
    node.steering_basis = 0.30
    _tick(node, _scan(wall_left=0.20))
    assert published[-1].drive.speed == 0.0
    assert published[-1].drive.steering_angle > 0.0


def test_flank_creep_reports_itself_in_drive_intent(make):
    node = make(CREEP, intent=True)
    seen = []
    original = node.intent_pub.publish

    def spy(msg):
        seen.append(schema.decode(msg.data))
        return original(msg)

    node.intent_pub.publish = spy
    _run(node, _scan(wall_left=0.20), 10)
    last = seen[-1]
    names = [f['name'] for f in last['factors']]
    assert 'flank creep' in names
    assert 'clearance cap' not in names and 'creep cap' not in names
    assert schema.binding_factor(last) == 'flank creep'


def test_lb_released_during_flank_creep_is_zero(make):
    node = make(CREEP)
    speeds = _run(node, _scan(wall_left=0.20), 10)
    assert speeds[-1] > 0.0, 'precondition: creeping'
    published = _capture(node)
    _tick(node, _scan(wall_left=0.20), held=False)
    assert published[-1].drive.speed == 0.0


def test_joy_never_received_during_flank_creep_is_zero(make):
    node = make(CREEP)
    published = _capture(node)
    for _ in range(10):
        _tick(node, _scan(wall_left=0.20), held=None)  # no /joy at all
    assert [m.drive.speed for m in published] == [0.0] * 10


def test_stale_joy_during_flank_creep_is_zero(make):
    node = make(CREEP)
    speeds = _run(node, _scan(wall_left=0.20), 10)
    assert speeds[-1] > 0.0, 'precondition: creeping'
    published = _capture(node)
    node.last_joy_time = node.get_clock().now() - Duration(
        seconds=node.joy_timeout_sec + 0.1)
    _odom(node)
    node.scan_callback(_scan(wall_left=0.20))
    assert published[-1].drive.speed == 0.0


@pytest.mark.parametrize('param, value', [
    ('flank_creep_length', -1.0), ('flank_creep_length', 3.0),
    ('flank_creep_margin', 0.2)])
def test_bad_flank_creep_parameters_refuse_to_start(param, value):
    rclpy.init(args=['--ros-args', '-p', f'{param}:={value}', *TEST_ONLY_DRIVE])
    try:
        with pytest.raises(ValueError, match='flank_creep'):
            GapFollowNode()
    finally:
        rclpy.shutdown()

