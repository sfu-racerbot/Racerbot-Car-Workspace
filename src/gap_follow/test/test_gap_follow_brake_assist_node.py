"""Brake assist in the real GapFollowNode (docs/gap-follow.md#brake-assist). Needs ROS sourced; run as
`rb python3 -m pytest src/gap_follow/test/test_gap_follow_brake_assist_node.py`.

Scenario: open to 3 m with a wall 1.5 m ahead, so the node's curvature cap (1.62 m/s) holds its planned speed below what the wheel
reports (in open space the plan is ramp-bound and the ramp follows the wheel, so there is never an overspeed to judge).
The plan (last_commanded_speed, the ramp basis) is read back each tick: the ramp runs on the wall clock. Oracle: max(0, planned - gain * (wheel - planned)) when wheel - planned exceeds the
threshold (gap_logic.brake_assist_speed's definition), else the planned speed. Every node publishes on
/test_only/drive and keeps the deadman ENABLED; the four deadman deny paths are asserted with the assist on.
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

from gap_follow.gap_follow_node import GapFollowNode

TEST_ONLY_DRIVE = ['-p', 'drive_topic:=/test_only/drive']
PACKAGED_CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config', 'gap_follow.yaml')
DEADMAN_BUTTON = 4
DT = 0.025
BASE = ['-p', 'enable_launch_boost:=false', '-p', 'enable_opponent_awareness:=false']
ON = ['-p', 'enable_brake_assist:=true']


def open_scan(stamp, wall=1.5, n=1081, span=math.radians(270)):
    msg = LaserScan()
    msg.angle_min = -span / 2.0
    msg.angle_increment = span / (n - 1)
    msg.angle_max = msg.angle_min + (n - 1) * msg.angle_increment
    msg.range_min, msg.range_max = 0.05, 12.0
    a = msg.angle_min + np.arange(n) * msg.angle_increment
    r = np.full(n, 3.0)
    if wall is not None:
        r = np.minimum(r, np.where(np.abs(a) < math.radians(85), wall / np.maximum(np.cos(a), 1e-6), 3.0))
    msg.ranges = [float(x) for x in r]
    msg.header.stamp.sec = int(stamp)
    msg.header.stamp.nanosec = int(round((stamp - int(stamp)) * 1e9))
    return msg


class Rig:
    def __init__(self, extra=()):
        rclpy.init(args=['--ros-args', *car_config_args(PACKAGED_CONFIG), *BASE, *extra, '-p', 'intent_rate_hz:=0.0',
                         *TEST_ONLY_DRIVE])
        try:
            self.node = GapFollowNode()
        except Exception:
            rclpy.shutdown()
            raise
        assert self.node.drive_pub.topic_name == '/test_only/drive'
        self.sent = []
        original = self.node.drive_pub.publish
        self.node.drive_pub.publish = lambda m: (self.sent.append(m.drive.speed), original(m))
        self.t = 100.0

    def close(self):
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def tick(self, wheel, held=True, odom=True):
        """One scan tick: LB (None = no /joy this tick), odometry at ``wheel`` m/s, a scan 25 ms later.
        Returns (published speed, the speed the node planned)."""
        if held is not None:
            msg = Joy()
            msg.buttons = [0] * (DEADMAN_BUTTON + 1)
            msg.buttons[DEADMAN_BUTTON] = 1 if held else 0
            self.node.joy_callback(msg)
        if odom:
            o = Odometry()
            o.twist.twist.linear.x = float(wheel)
            self.node.odom_callback(o)
        self.t += DT
        n = len(self.sent)
        self.node.scan_callback(open_scan(self.t))
        assert len(self.sent) == n + 1, 'exactly one drive command per scan tick'
        return self.sent[-1], self.node.last_commanded_speed


@pytest.fixture
def rig():
    rigs = []

    def build(*a, **kw):
        r = Rig(*a, **kw)
        rigs.append(r)
        return r

    yield build
    for r in rigs:
        r.close()


def ramp_to_plan(r):
    """Bring the plan to the wall's clearance cap. The ramp's step uses the wall clock between commands (microseconds
    here), but its basis is max(last plan, wheel speed): a few ticks with the wheel far above any cap put the plan on
    the cap at once (after a few slow ticks: the first ticks start from rest). Returns the cap, read back twice to
    show it holds."""
    for _ in range(4):
        r.tick(wheel=0.5)
    for _ in range(3):
        _published, planned = r.tick(wheel=6.0)
    _published, again = r.tick(wheel=6.0)
    assert 0.3 < planned == again < 6.0
    return planned


def test_a_wheel_over_the_plan_publishes_the_assisted_speed(rig):
    r = rig(ON)
    cap = ramp_to_plan(r)
    wheel = cap + 0.5
    published, planned = r.tick(wheel=wheel)
    assert planned == pytest.approx(cap, abs=0.02)          # the wall's cap, not the wheel, sets the plan
    assert wheel - planned > 0.15
    assert published == pytest.approx(max(0.0, planned - 1.0 * (wheel - planned)), abs=1e-9)
    assert r.node.brake_assist_active


def test_within_the_threshold_or_below_the_plan_it_publishes_the_plan(rig):
    r = rig(ON)
    ramp_to_plan(r)
    for offset in (0.15, 0.0, -0.6):                     # at most the 0.15 threshold over, equal, under
        wheel = r.node.last_commanded_speed + offset
        published, planned = r.tick(wheel=wheel)
        assert wheel - planned <= 0.15 + 1e-9           # precondition: never past the threshold
        assert published == planned
        assert not r.node.brake_assist_active


def test_off_publishes_the_plan(rig):
    r = rig()
    cap = ramp_to_plan(r)
    published, planned = r.tick(wheel=cap + 0.8)
    assert published == planned


def test_the_planned_speed_stays_the_ramp_basis_when_the_assist_acts(rig):
    # What is published drops; the planned speed handed to _publish_drive must stay last_commanded_speed (the next
    # tick's ramp basis), or re-acceleration after every assisted tick would start from the lowered command.
    r = rig(ON)
    handed = []
    publish = r.node._publish_drive
    r.node._publish_drive = lambda steering, speed, now=None: (handed.append(speed), publish(steering, speed, now=now))
    ramp_to_plan(r)
    acted = 0
    for w in (5.0, 5.0, 4.0, 3.0):
        published, planned = r.tick(wheel=w)
        assert planned == handed[-1]                       # last_commanded_speed is the plan, not the published value
        acted += published < planned - 0.3
    assert acted >= 2


def test_it_never_publishes_a_negative_speed(rig):
    r = rig(ON)
    cap = ramp_to_plan(r)
    published, planned = r.tick(wheel=3.0 * cap + 1.0)
    assert planned - (3.0 * cap + 1.0 - planned) < 0.0  # the formula goes negative here (precondition)
    assert published == 0.0                              # clamped at zero, never a reverse command


def test_stale_odometry_gives_no_assist(rig):
    # TTC off so the node's own stale-odometry stop cannot produce the answer; the assist must not judge an old speed.
    r = rig([*ON, '-p', 'enable_ttc:=false'])
    cap = ramp_to_plan(r)
    r.tick(wheel=cap + 0.8)                              # the last odometry: 0.8 m/s over the plan
    r.node.last_odom_time = r.node.get_clock().now() - Duration(seconds=r.node.odom_timeout_sec + 1.0)
    assert not r.node._odom_fresh()
    published, planned = r.tick(wheel=0.0, odom=False)
    assert published == planned > 0.0


# --- the deadman with the assist on: the four deny paths (CLAUDE.md, standing requirement) ---------------------------
def test_lb_held_drives(rig):
    r = rig(ON)
    published, _ = r.tick(wheel=0.0)
    for _ in range(10):
        published, _ = r.tick(wheel=0.0)
    assert published > 0.0


def test_lb_released_publishes_zero(rig):
    r = rig(ON)
    ramp_to_plan(r)
    published, _ = r.tick(wheel=r.node.last_commanded_speed + 0.5, held=False)
    assert published == 0.0


def test_no_joy_ever_publishes_zero(rig):
    r = rig(ON)
    for _ in range(5):
        published, _ = r.tick(wheel=r.node.last_commanded_speed + 0.5, held=None)
        assert published == 0.0


def test_stale_joy_publishes_zero(rig):
    r = rig(ON)
    ramp_to_plan(r)
    r.node.last_joy_time = r.node.get_clock().now() - Duration(seconds=r.node.joy_timeout_sec + 1.0)
    published, _ = r.tick(wheel=r.node.last_commanded_speed + 0.5, held=None)
    assert published == 0.0


@pytest.mark.parametrize('name, value', [('brake_assist_threshold', -0.1), ('brake_assist_threshold', 1.5),
                                         ('brake_assist_gain', -1.0), ('brake_assist_gain', 3.5)])
def test_out_of_range_parameters_are_refused_by_name(name, value):
    rclpy.init(args=['--ros-args', *car_config_args(PACKAGED_CONFIG), *ON, '-p', f'{name}:={value}', *TEST_ONLY_DRIVE])
    try:
        with pytest.raises(ValueError, match=name):
            GapFollowNode()
    finally:
        rclpy.shutdown()
