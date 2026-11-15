"""Understeer compensation (Task 6) and the delivered-curvature speed cap (Task 7) in the real
PurePursuitNode. Both are OFF by default. Needs ROS sourced; run as
`rb python3 -m pytest src/pure_pursuit/test/test_understeer_node.py`.

The deadman is ENABLED in every node here (LB faked through the node's own joy_callback) and every
node publishes on /test_only/drive, never /drive.
"""
import math
import os

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from sensor_msgs.msg import Joy, LaserScan

from drive_intent import schema
from pure_pursuit import racing_math
from pure_pursuit.pure_pursuit_node import PurePursuitNode

DEADMAN_BUTTON = 4
WHEELBASE = 0.36   # m
# lateral 3.6 m/s^2 in the stadium's bend is above the 2.5 knee and below the last delivered 12
CURVE = ['-p', 'understeer_curve_asked:=[0.0,2.5,20.0]', '-p', 'understeer_curve_delivered:=[0.0,2.5,12.0]']
BEND = (3.09, -0.51, 0.0)   # on the stadium's right-hand bend; yaw set from the line below
STRAIGHT_IDX, BEND_IDX = 12, 36


@pytest.fixture(scope='module')
def stadium(tmp_path_factory):
    src = os.path.join(os.path.dirname(__file__), '..', 'waypoints', 'example_stadium_raw.csv')
    xy = racing_math.load_xy_csv(src)
    seg = racing_math.compute_segment_lengths(xy, closed=True)
    curvature = racing_math.estimate_path_curvature(xy, closed=True)
    speed = racing_math.compute_velocity_profile(
        seg, curvature, v_max=6.0, v_min=0.5, a_lat_max=8.0, a_accel_max=3.0, a_brake_max=8.0, closed=True)
    out = str(tmp_path_factory.mktemp('wp') / 'profiled.csv')
    racing_math.save_profiled_csv(out, xy, speed)
    return out, xy


@pytest.fixture
def make(stadium):
    csv, _ = stadium
    nodes = []

    def build(*extra, intent=False):
        # One rclpy context at a time: building a node tears the previous one down, so a test reads
        # what it needs from a node (a number, a captured list) before it builds the next.
        while nodes:
            nodes.pop().destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        args = ['--ros-args', '-p', f'waypoints_file:={csv}', '-p', 'max_speed:=6.0',
                '-p', 'max_lateral_accel:=3.1', *extra]
        if intent:
            args += ['-p', 'intent_rate_hz:=0.0']
        rclpy.init(args=[*args, '-p', 'drive_topic:=/test_only/drive'])
        node = PurePursuitNode()
        assert node.drive_pub.topic_name == '/test_only/drive'
        assert node.enable_deadman is True
        node.max_steering_rate = 100.0   # the slew is tested elsewhere; here it would hide the steering decision
        nodes.append(node)
        return node

    yield build
    for n in nodes:
        n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def _scan(n=361):
    scan = LaserScan()
    scan.angle_min = -math.pi / 2
    scan.angle_increment = math.pi / n
    scan.angle_max = scan.angle_min + (n - 1) * scan.angle_increment
    scan.range_min, scan.range_max = 0.05, 12.0
    scan.ranges = [10.0] * n
    return scan


def _joy(node, held=True):
    msg = Joy()
    msg.buttons = [0] * (DEADMAN_BUTTON + 1)
    msg.buttons[DEADMAN_BUTTON] = 1 if held else 0
    node.joy_callback(msg)


def _tick(node, xy, index, held=True):
    if held is not None:
        _joy(node, held)
    x, y = xy[index]
    nx, ny = xy[(index + 1) % len(xy)]
    yaw = math.atan2(ny - y, nx - x)
    pose = PoseStamped()
    pose.pose.position.x, pose.pose.position.y = float(x), float(y)
    pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    node.pose_callback(pose)
    node.scan_callback(_scan())
    node.control_loop()


def _capture(node):
    published = []
    original = node.drive_pub.publish
    node.drive_pub.publish = lambda msg: (published.append(msg), original(msg))
    return published


def _steer(node, xy, index):
    published = _capture(node)
    _tick(node, xy, index)
    return published[-1].drive.steering_angle


# --- Task 6 --------------------------------------------------------------------------

def test_no_curve_by_default_and_an_identity_curve_changes_nothing(make, stadium):
    _, xy = stadium
    plain = make()
    assert plain.grip_curve is None
    reference = _steer(plain, xy, BEND_IDX)
    identity = make('-p', 'understeer_curve_asked:=[0.0,20.0]', '-p', 'understeer_curve_delivered:=[0.0,20.0]')
    assert _steer(identity, xy, BEND_IDX) == pytest.approx(reference, abs=1e-9)  # rad
    assert abs(reference) > 0.05, 'precondition: the bend really steers'


def test_a_curve_steers_harder_in_a_bend_and_identically_on_a_straight(make, stadium):
    _, xy = stadium
    base = _steer(make(), xy, BEND_IDX)
    straight_plain = _steer(make(), xy, STRAIGHT_IDX)
    more = _steer(make(*CURVE), xy, BEND_IDX)
    assert math.copysign(1.0, more) == math.copysign(1.0, base)
    assert abs(more) > abs(base) * 1.1
    # closed-form ceiling: the curve's steepest inverse slope is (20 - 2.5) / (12 - 2.5) = 1.84 of the lateral
    assert abs(math.tan(more)) <= abs(math.tan(base)) * 1.85
    straight_curve = _steer(make(*CURVE), xy, STRAIGHT_IDX)
    assert straight_curve == pytest.approx(straight_plain, abs=1e-12)  # rad


def test_the_intent_reports_the_uncompensated_steering_and_says_so(make, stadium):
    _, xy = stadium
    base = _steer(make(), xy, BEND_IDX)
    node = make(*CURVE, intent=True)
    seen, published = [], _capture(node)
    original = node.intent_pub.publish
    node.intent_pub.publish = lambda msg: (seen.append(schema.decode(msg.data)), original(msg))
    _tick(node, xy, BEND_IDX)
    assert published[-1].drive.steering_angle != pytest.approx(base, abs=1e-6)
    assert seen[-1]['desired_steering'] == pytest.approx(base, abs=1e-3)  # rad, the schema rounds to 4 places
    assert seen[-1]['commanded_steering'] == pytest.approx(published[-1].drive.steering_angle, abs=1e-3)
    assert 'understeer +' in seen[-1]['reason']


@pytest.mark.parametrize('asked, delivered', [
    ('[0.0,2.5]', '[0.0]'), ('[0.0,2.5,2.5]', '[0.0,1.0,2.0]'), ('[0.5,2.5]', '[0.0,1.0]'),
    ('[0.0,2.5]', '[0.0,3.0]')])
def test_an_invalid_curve_refuses_to_start(stadium, asked, delivered):
    csv, _ = stadium
    rclpy.init(args=['--ros-args', '-p', f'waypoints_file:={csv}', '-p', f'understeer_curve_asked:={asked}',
                     '-p', f'understeer_curve_delivered:={delivered}', '-p', 'drive_topic:=/test_only/drive'])
    try:
        with pytest.raises(RuntimeError, match='understeer_curve'):
            PurePursuitNode()
    finally:
        rclpy.shutdown()


def test_lb_held_with_a_curve_publishes_a_nonzero_command(make, stadium):
    _, xy = stadium
    node = make(*CURVE)
    published = _capture(node)
    for _ in range(5):
        _tick(node, xy, STRAIGHT_IDX)
    assert published[-1].drive.speed > 0.0


def test_lb_released_with_a_curve_is_zero(make, stadium):
    _, xy = stadium
    node = make(*CURVE)
    published = _capture(node)
    for _ in range(5):
        _tick(node, xy, STRAIGHT_IDX)
    _tick(node, xy, STRAIGHT_IDX, held=False)
    assert published[-1].drive.speed == 0.0


def test_joy_never_received_with_a_curve_is_zero(make, stadium):
    _, xy = stadium
    node = make(*CURVE)
    published = _capture(node)
    for _ in range(5):
        _tick(node, xy, STRAIGHT_IDX, held=None)
    assert [m.drive.speed for m in published] == [0.0] * 5


def test_stale_joy_with_a_curve_is_zero(make, stadium):
    _, xy = stadium
    node = make(*CURVE)
    published = _capture(node)
    for _ in range(5):
        _tick(node, xy, STRAIGHT_IDX)
    node.last_joy_time = node.get_clock().now() - Duration(seconds=node.joy_timeout_sec + 0.1)
    _tick(node, xy, STRAIGHT_IDX, held=None)
    assert published[-1].drive.speed == 0.0


# --- Task 7 --------------------------------------------------------------------------

def _curve_speed(node, steering):
    node.steering_basis = steering
    _now, _steer_out, _speed, online_curvature, curve_speed, _dt = node._shape_normal_command(
        steering, 10.0, 10.0)
    return online_curvature, curve_speed


def test_without_a_curve_the_online_cap_is_the_commanded_curvature_cap(make):
    node = make()
    kappa, cap = _curve_speed(node, math.atan(0.5 * WHEELBASE))
    assert kappa == pytest.approx(0.5, abs=1e-9)  # 1/m
    assert cap == pytest.approx(math.sqrt(3.1 / 0.5), abs=1e-9)  # m/s, the old cap


def test_with_a_curve_the_online_cap_uses_the_delivered_curvature(make):
    # curve asked (0, 2.5, 10), delivered (0, 2.5, 5.5); limit 3.1; curvature 0.5/m:
    # ask_for(3.1) = 2.5 + 0.6 * 7.5 / 3.0 = 4.0  ->  v = sqrt(4.0 / 0.5)
    node = make('-p', 'understeer_curve_asked:=[0.0,2.5,10.0]', '-p', 'understeer_curve_delivered:=[0.0,2.5,5.5]')
    _kappa, cap = _curve_speed(node, math.atan(0.5 * WHEELBASE))
    assert cap == pytest.approx(math.sqrt(4.0 / 0.5), abs=1e-6)  # m/s


def test_a_curve_that_never_delivers_the_limit_still_bounds_the_speed(make):
    # delivered tops out at 2.5 < 3.1: asked lateral is capped at the last asked value 6.0
    node = make('-p', 'understeer_curve_asked:=[0.0,2.0,6.0]', '-p', 'understeer_curve_delivered:=[0.0,2.0,2.5]')
    _kappa, cap = _curve_speed(node, math.atan(0.5 * WHEELBASE))
    assert cap == pytest.approx(math.sqrt(6.0 / 0.5), abs=1e-6)  # m/s
