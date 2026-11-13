"""Reverse escape in the real GapFollowNode (docs/gap-follow.md#reverse-escape). Needs ROS sourced; run as
`rb python3 -m pytest src/gap_follow/test/test_gap_follow_reverse_node.py`.

Scenario (rb2 20261111-142254, 104.6-118.2 s, simplified): the car drives 1 m at 2 m/s, then stops with a wall
0.43 m ahead of the LiDAR (0.22 m ahead of the bumper), open sides and rear quarters. Scans are stamped 25 ms apart
(the machine's clock is the scan stamp). Every node publishes on /test_only/drive and keeps the deadman ENABLED.
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
from gap_follow import gap_logic
from gap_follow.gap_follow_node import GapFollowNode

TEST_ONLY_DRIVE = ['-p', 'drive_topic:=/test_only/drive']
PACKAGED_CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config', 'gap_follow.yaml')
DEADMAN_BUTTON = 4
DT = 0.025
WALL = 0.43      # m, LiDAR to the wall ahead (0.22 m from the bumper)
ON = ['-p', 'enable_reverse_escape:=true']


def boundary(deg):
    """LiDAR-to-body distance along the beam, from the footprint check's own geometry (rb1/rb2 body)."""
    return float(gap_logic.vehicle_boundary_distances(np.array([math.radians(deg)]), 0.31, 0.58, 0.36, 0.26, 0.0)[0])


def make_scan(stamp, wall=WALL, rear_block=None, flank=None, cage=None, n=1081, span=math.radians(270)):
    """Open to 3 m, a wall ``wall`` m ahead (a plane across the front 170 degrees), optionally a return at
    ``rear_block=(deg, range)`` or ``flank=(deg, range)``."""
    msg = LaserScan()
    msg.angle_min = -span / 2.0
    msg.angle_increment = span / (n - 1)
    msg.angle_max = msg.angle_min + (n - 1) * msg.angle_increment
    msg.range_min, msg.range_max = 0.05, 12.0
    a = msg.angle_min + np.arange(n) * msg.angle_increment
    r = np.full(n, 3.0)
    if wall is not None:
        w = np.where(np.abs(a) < math.radians(85), wall / np.maximum(np.cos(a), 1e-6), 3.0)
        r = np.minimum(r, w)
    if cage is not None:
        r = np.where(np.abs(a) < math.radians(95), np.minimum(r, cage), r)
    for spec in (rear_block, flank):
        if spec is not None:
            deg, rng = spec
            r[int(round((math.radians(deg) - msg.angle_min) / msg.angle_increment))] = rng
    msg.ranges = [float(x) for x in r]
    msg.header.stamp.sec = int(stamp)
    msg.header.stamp.nanosec = int(round((stamp - int(stamp)) * 1e9))
    return msg


class Rig:
    def __init__(self, extra=(), use_config=True):
        args = ['--ros-args']
        if use_config:
            args += car_config_args(PACKAGED_CONFIG)
        rclpy.init(args=[*args, *extra, '-p', 'intent_rate_hz:=0.0', *TEST_ONLY_DRIVE])
        self.node = GapFollowNode()
        assert self.node.drive_pub.topic_name == '/test_only/drive'
        self.sent, self.states, self.intents = [], [], []
        original = self.node.drive_pub.publish
        self.node.drive_pub.publish = lambda m: (self.sent.append((m.drive.steering_angle, m.drive.speed)), original(m))
        log = self.node._log_decision
        self.node._log_decision = lambda state, *a, **k: (self.states.append(state), log(state, *a, **k))
        intent = self.node.intent_pub.publish
        self.node.intent_pub.publish = lambda m: (self.intents.append(schema.decode(m.data)), intent(m))
        self.t = 100.0

    def close(self):
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def joy(self, held=True):
        msg = Joy()
        msg.buttons = [0] * (DEADMAN_BUTTON + 1)
        msg.buttons[DEADMAN_BUTTON] = 1 if held else 0
        self.node.joy_callback(msg)

    def odom(self, speed):
        msg = Odometry()
        msg.twist.twist.linear.x = float(speed)
        self.node.odom_callback(msg)

    def tick(self, speed=0.0, held=True, odom=True, **scan):
        """One scan tick: LB (None = no /joy message this tick), odometry at ``speed``, a scan 25 ms later."""
        if held is not None:
            self.joy(held)
        if odom:
            self.odom(speed)
        self.t += DT
        n = len(self.sent)
        self.node.scan_callback(make_scan(self.t, **scan))
        assert len(self.sent) == n + 1, 'exactly one drive command per scan tick'
        return self.sent[-1]

    def drive_up(self, metres=1.0, speed=2.0):
        for _ in range(int(round(metres / speed / DT))):
            self.tick(speed=speed, wall=None)

    def wait_for_reverse(self, limit=120, **kw):
        """Ticks at the wall, stopped, until the first negative command; returns its index (from the first tick)."""
        for i in range(limit):
            if self.tick(speed=0.0, **kw)[1] < 0.0:
                return i
        return None


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


def test_blocked_ahead_waits_one_second_then_backs_straight_at_most_030(rig):
    r = rig(ON)
    r.drive_up()
    first = r.wait_for_reverse()
    assert first is not None
    before = [s for _, s in r.sent[-first - 1:-1]]
    assert min(before) >= 0.0 and all(s == 0.0 for s in before[5:])       # the normal controller's stop, never negative
    assert 1.0 <= first * DT <= 1.15, first                                 # s, wait_sec 1.0 after the first blocked tick
    steering, speed = r.sent[-1]
    assert steering == 0.0 and -0.30 - 1e-12 <= speed < 0.0


def test_the_reverse_ends_after_the_target_distance_then_settles_at_zero(rig):
    r = rig(ON)
    r.drive_up()
    r.wait_for_reverse()
    reversing = 1
    while True:
        steering, speed = r.tick(speed=-0.30)                               # odometry reads the car backing at 0.30 m/s
        if speed >= 0.0:
            break
        assert steering == 0.0 and speed >= -0.30 - 1e-12
        reversing += 1
        assert reversing < 100
    # closed form: min(0.35 m, 1.0 m driven) at 0.30 m/s = 1.167 s = 46.7 ticks; the machine ends on the 47th
    assert 46 <= reversing <= 48, reversing
    assert speed == 0.0
    settle = [r.tick(speed=0.0)[1] for _ in range(10)]
    assert all(s == 0.0 for s in settle)
    assert any(i['state'] == 'reverse_escape' for i in r.intents)


def test_the_intent_names_the_reverse_and_its_remaining_distance(rig):
    r = rig(ON)
    r.drive_up()
    r.wait_for_reverse()
    for _ in range(5):
        r.tick(speed=-0.30)
    seen = [i for i in r.intents if i['state'] == 'reverse_escape']
    assert seen and seen[-1]['commanded_speed'] == pytest.approx(-0.30, abs=1e-3)       # m/s
    assert seen[-1]['commanded_steering'] == 0.0
    assert 'left at 0.30m/s' in seen[0]['reason'] or any('m of 0.35m' in (i.get('reason') or '') for i in seen)


# --- the deadman, during the reverse ---------------------------------------------------------------

def test_lb_released_during_the_reverse_commands_zero_on_the_next_tick(rig):
    r = rig(ON)
    r.drive_up()
    assert r.wait_for_reverse() is not None
    for _ in range(3):
        assert r.tick(speed=-0.30)[1] < 0.0
    assert r.tick(speed=-0.30, held=False) == (0.0, 0.0) or r.sent[-1][1] == 0.0
    for _ in range(20):
        assert r.tick(speed=0.0, held=False)[1] == 0.0
    assert r.node.reverse.state == 'idle'


def test_no_joy_ever_never_reverses(rig):
    r = rig(ON)
    for _ in range(20):
        r.tick(speed=2.0, held=None, wall=None)
    speeds = [r.tick(speed=0.0, held=None)[1] for _ in range(150)]
    assert all(s == 0.0 for s in speeds)


def test_stale_joy_during_the_reverse_commands_zero(rig):
    r = rig(ON)
    r.drive_up()
    assert r.wait_for_reverse() is not None
    assert r.tick(speed=-0.30)[1] < 0.0
    r.node.last_joy_time = r.node.get_clock().now() - Duration(seconds=r.node.joy_timeout_sec + 1.0)
    assert r.tick(speed=-0.30, held=None)[1] == 0.0
    assert r.tick(speed=-0.30, held=None)[1] == 0.0


# --- refusals ---------------------------------------------------------------------------------------

@pytest.mark.parametrize('deg', [110.0, -120.0, 100.0, -129.0])
def test_a_blocked_rear_quarter_never_reverses(rig, deg):
    # a return 5 cm from the body (the default clearance is 10 cm)
    r = rig(ON)
    r.drive_up()
    assert r.wait_for_reverse(limit=160, rear_block=(deg, boundary(deg) + 0.05)) is None
    assert r.node.reverse.state == 'waiting'                                # it is held, not idle


def test_a_rear_quarter_that_closes_mid_reverse_stops_it(rig):
    r = rig(ON)
    r.drive_up()
    assert r.wait_for_reverse() is not None
    assert r.tick(speed=-0.30)[1] < 0.0
    assert r.tick(speed=-0.30, rear_block=(110.0, boundary(110.0) + 0.05))[1] == 0.0


def test_stale_odometry_never_reverses(rig):
    r = rig(ON)
    r.drive_up()
    old = r.node.get_clock().now() - Duration(seconds=10.0)
    speeds = []
    for _ in range(150):
        r.node.last_odom_time = old
        speeds.append(r.tick(odom=False)[1])
    assert all(s == 0.0 for s in speeds)


def test_stale_odometry_during_the_reverse_ends_it(rig):
    r = rig(ON)
    r.drive_up()
    assert r.wait_for_reverse() is not None
    r.node.last_odom_time = r.node.get_clock().now() - Duration(seconds=10.0)
    assert r.tick(odom=False)[1] == 0.0


def test_nothing_driven_just_before_the_stop_never_reverses(rig):
    r = rig(ON)
    speeds = [r.tick(speed=0.0)[1] for _ in range(200)]                   # never drove: no ground to back over
    assert all(s >= 0.0 for s in speeds)


def test_a_flank_emergency_stop_does_not_arm_it(rig):
    r = rig(ON)
    r.drive_up()
    # a return 0.149 m to the side at 85 degrees is inside the 0.155 m half width: a flank contact
    speeds = [r.tick(speed=0.0, wall=None, flank=(85.0, 0.15))[1] for _ in range(160)]
    assert 'emergency_clearance' in r.states
    assert all(s >= 0.0 for s in speeds)
    assert r.node._rev_hint is False


def test_a_ttc_stop_does_not_arm_it(rig):
    # caged in at 0.4 m all round the front while rolling at 2 m/s: TTC trips and there is no gap to escape into
    r = rig(ON)
    r.drive_up()
    speeds = []
    for _ in range(80):
        speeds.append(r.tick(speed=2.0, wall=None, cage=0.4)[1])
        if r.states and r.states[-1] == 'ttc_brake':
            assert r.node._rev_hint is False
            break
    assert 'ttc_brake' in r.states
    assert all(s >= 0.0 for s in speeds)


def test_no_safe_gap_stopped_is_blocked_ahead(rig):
    # the same cage once the car has stopped: no_safe_gap, which does arm it
    r = rig(ON)
    r.drive_up()
    for _ in range(100):
        r.tick(speed=0.0, wall=None, cage=0.4)
    assert r.node.reverse.state in ('waiting', 'reversing', 'settling', 'cooldown')
    assert 'no_safe_gap' in r.states


# --- off by default protects every car -----------------------------------------------------------------

@pytest.mark.parametrize('use_config', [True, False])
def test_feature_off_never_goes_negative_over_the_whole_scenario(rig, use_config):
    r = rig((), use_config=use_config)
    assert r.node.enable_reverse_escape is False
    r.drive_up()
    speeds = [r.tick(speed=0.0)[1] for _ in range(400)]
    speeds += [r.tick(speed=-0.30)[1] for _ in range(100)]
    assert min(speeds) >= 0.0
    assert not any(i['state'] == 'reverse_escape' for i in r.intents)


@pytest.mark.parametrize('name,value,match', [
    ('reverse_escape_speed', '0.6', 'speed'), ('reverse_escape_max_distance', '0.7', 'max_distance'),
    ('reverse_escape_wait_sec', '0.2', 'wait_sec'), ('reverse_escape_max_attempts', '4', 'max_attempts'),
    ('reverse_escape_rear_band_hi_deg', '140.0', 'rear_band')])
def test_bad_parameters_refuse_to_start(name, value, match):
    rclpy.init(args=['--ros-args', *car_config_args(PACKAGED_CONFIG), '-p', f'{name}:={value}', *TEST_ONLY_DRIVE])
    try:
        with pytest.raises(ValueError, match=match):
            GapFollowNode()
    finally:
        rclpy.shutdown()
