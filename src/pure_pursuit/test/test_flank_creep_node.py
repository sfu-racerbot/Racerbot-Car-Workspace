"""Flank creep in the real PurePursuitNode (Task 3 of
docs/superpowers/plans/2026-11-12-post-race-speed-program.md).

rb2 20261111-152132, 19.0-49.5 s: the car stopped for 30.5 s against a wall
0.18 m from the LiDAR on its right (2.5 cm off the body) with 0.63 m free
ahead, because Tier 0's `body_contact` refuses a flank escape. Needs ROS
sourced; run as `rb python3 -m pytest src/pure_pursuit/test/test_flank_creep_node.py`.

The deadman is ENABLED in every node here (LB faked through the node's own
joy_callback), and every node publishes on /test_only/drive, never /drive.
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

FIXTURE = os.path.join(os.path.dirname(__file__), 'data',
                       'flank_stop_rb2_20261111-152132.npz')
OPEN = 6.0
DEADMAN_BUTTON = 4


@pytest.fixture(scope='module')
def profiled_csv(tmp_path_factory):
    src = os.path.join(os.path.dirname(__file__), '..', 'waypoints', 'example_stadium_raw.csv')
    xy = racing_math.load_xy_csv(src)
    seg_len = racing_math.compute_segment_lengths(xy, closed=True)
    curvature = racing_math.estimate_path_curvature(xy, closed=True)
    speed = racing_math.compute_velocity_profile(
        seg_len, curvature, v_max=6.0, v_min=0.5, a_lat_max=8.0, a_accel_max=3.0,
        a_brake_max=8.0, closed=True)
    out = str(tmp_path_factory.mktemp('waypoints') / 'profiled.csv')
    racing_math.save_profiled_csv(out, xy, speed)
    return out


def _scan(wall_right=None, block=None, n=1081, span=math.radians(270)):
    """Open to 6 m with a straight wall `wall_right` m to the RIGHT of the
    LiDAR (y = -wall_right), parallel to the car; `block=(lo_deg, hi_deg, r)`
    returns closer in that bearing range."""
    msg = LaserScan()
    msg.angle_min = -span / 2.0
    msg.angle_increment = span / (n - 1)
    msg.angle_max = msg.angle_min + (n - 1) * msg.angle_increment
    msg.range_min, msg.range_max = 0.05, 12.0
    angles = msg.angle_min + np.arange(n) * msg.angle_increment
    ranges = np.full(n, OPEN)
    if wall_right is not None:
        s = -np.sin(angles)
        ranges = np.minimum(ranges, np.where(s > 1e-6, wall_right / np.maximum(s, 1e-6), OPEN))
    if block is not None:
        lo, hi, r = block
        sel = (angles >= math.radians(lo)) & (angles <= math.radians(hi))
        ranges[sel] = np.minimum(ranges[sel], r)
    msg.ranges = [float(r) for r in ranges]
    return msg


@pytest.fixture
def make(profiled_csv):
    nodes = []

    def build(*extra, intent=False):
        args = ['--ros-args', '-p', f'waypoints_file:={profiled_csv}',
                *extra]
        if intent:
            args += ['-p', 'intent_rate_hz:=0.0']
        rclpy.init(args=[*args, '-p', 'drive_topic:=/test_only/drive'])
        node = PurePursuitNode()
        assert node.drive_pub.topic_name == '/test_only/drive'
        assert node.enable_deadman is True
        nodes.append(node)
        return node

    yield build
    for n in nodes:
        n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def _pose(node, x=-1.5, y=-1.2, yaw=0.0):
    msg = PoseStamped()
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.z = math.sin(yaw / 2.0)
    msg.pose.orientation.w = math.cos(yaw / 2.0)
    node.pose_callback(msg)


def _joy(node, held=True):
    msg = Joy()
    msg.buttons = [0] * (DEADMAN_BUTTON + 1)
    msg.buttons[DEADMAN_BUTTON] = 1 if held else 0
    node.joy_callback(msg)


def _capture(node):
    published = []
    original = node.drive_pub.publish
    node.drive_pub.publish = lambda msg: (published.append(msg), original(msg))
    return published


def _tick(node, scan, held=True, yaw=0.0):
    if held is not None:
        _joy(node, held)
    _pose(node, yaw=yaw)
    node.scan_callback(scan)
    node.control_loop()


def _run(node, scan, ticks, **kw):
    published = _capture(node)
    for _ in range(ticks):
        _tick(node, scan, **kw)
    return [m.drive.speed for m in published]


CREEP = ('-p', 'flank_creep_length:=0.5')
WALL = 0.18  # m from the LiDAR = 0.025 m off the 0.31 m body (the recorded stop)


def test_wall_alongside_with_open_road_creeps(make):
    node = make(*CREEP)
    speeds = _run(node, _scan(wall_right=WALL), 10)
    assert node.last_decision_state == 'flank_creep'
    # the creep speed is the one _emergency_escape uses, not a second knob
    assert speeds[-1] == pytest.approx(node.emergency_escape_speed, abs=1e-9)  # m/s


def test_the_latch_is_real_with_flank_creep_off(make):
    """Precondition for the creep test: same scan, creep off -> body_contact."""
    node = make('-p', 'flank_creep_length:=0.0')
    assert _run(node, _scan(wall_right=WALL), 10) == [0.0] * 10
    assert node.last_decision_state == 'body_contact'


def test_flank_creep_off_by_default_keeps_the_stop(make):
    node = make()
    assert node.flank_creep_length == 0.0
    assert _run(node, _scan(wall_right=WALL), 10) == [0.0] * 10
    assert node.last_decision_state == 'body_contact'


def test_wall_touching_the_body_still_stops(make):
    # 0.15 m from the LiDAR is inside the 0.155 m half width: body_clearance < 0
    node = make(*CREEP)
    assert _run(node, _scan(wall_right=0.15), 10) == [0.0] * 10
    assert node.last_decision_state == 'body_contact'


def test_steering_toward_the_flank_wall_still_stops(make):
    # Heading 0.6 rad left of the line: the tracking law steers hard right,
    # into the wall that is 2.5 cm off the right flank.
    node = make(*CREEP)
    published = _capture(node)
    for _ in range(10):
        _tick(node, _scan(wall_right=WALL), yaw=0.6)
    assert published[-1].drive.steering_angle < -0.05, 'precondition: steering toward the wall'
    assert [m.drive.speed for m in published] == [0.0] * 10
    assert node.last_decision_state == 'body_contact'


def test_flank_creep_reports_itself_in_drive_intent(make):
    node = make(*CREEP, intent=True)
    seen = []
    original = node.intent_pub.publish

    def spy(msg):
        seen.append(schema.decode(msg.data))
        return original(msg)

    node.intent_pub.publish = spy
    _run(node, _scan(wall_right=WALL), 5)
    assert seen[-1]['state'] == 'flank_creep'
    assert seen[-1]['commanded_speed'] == pytest.approx(node.emergency_escape_speed, abs=1e-3)


def test_lb_released_during_flank_creep_is_zero(make):
    node = make(*CREEP)
    assert _run(node, _scan(wall_right=WALL), 5)[-1] > 0.0, 'precondition: creeping'
    published = _capture(node)
    _tick(node, _scan(wall_right=WALL), held=False)
    assert published[-1].drive.speed == 0.0


def test_joy_never_received_during_flank_creep_is_zero(make):
    node = make(*CREEP)
    assert _run(node, _scan(wall_right=WALL), 5, held=None) == [0.0] * 5


def test_stale_joy_during_flank_creep_is_zero(make):
    node = make(*CREEP)
    assert _run(node, _scan(wall_right=WALL), 5)[-1] > 0.0, 'precondition: creeping'
    published = _capture(node)
    node.last_joy_time = node.get_clock().now() - Duration(seconds=node.joy_timeout_sec + 0.1)
    _pose(node)
    node.scan_callback(_scan(wall_right=WALL))
    node.control_loop()
    assert published[-1].drive.speed == 0.0


@pytest.mark.parametrize('param, value', [
    ('flank_creep_length', -1.0), ('flank_creep_length', 3.0),
    ('flank_creep_margin', 0.2)])
def test_bad_flank_creep_parameters_refuse_to_start(profiled_csv, param, value):
    rclpy.init(args=['--ros-args', '-p', f'waypoints_file:={profiled_csv}',
                     '-p', f'{param}:={value}', '-p', 'drive_topic:=/test_only/drive'])
    try:
        with pytest.raises(RuntimeError, match='flank_creep'):
            PurePursuitNode()
    finally:
        rclpy.shutdown()


def test_recorded_flank_stop_scans_are_free_along_the_arc():
    # recorded measurement: rb2 20261111-152132, 20.0-21.5 s (wall 0.18 m on the right)
    data = np.load(FIXTURE)
    angles = float(data['angle_min']) + np.arange(data['ranges'].shape[1]) * float(data['angle_increment'])
    window = np.abs(angles) <= math.radians(90.0)  # body_clearance_fov_deg 180
    free = 0
    for ranges in data['ranges']:
        r = ranges.astype(float)
        valid = np.isfinite(r) & (r >= float(data['range_min'])) & (r <= 10.0)
        result = racing_math.body_arc_free_distance(
            np.where(valid, r, 0.0)[window], valid[window], angles[window],
            math.tan(0.025) / 0.36, 0.165, 0.58, 0.36, 0.26, 0.0, 0.5)
        free += result == math.inf
    assert free >= 0.95 * len(data['ranges'])


def test_shipped_config_turns_the_creep_on():
    import yaml
    path = os.path.join(os.path.dirname(__file__), '..', 'config', 'pure_pursuit.yaml')
    with open(path) as f:
        params = yaml.safe_load(f)['pure_pursuit_node']['ros__parameters']
    assert params['flank_creep_length'] == pytest.approx(0.50, abs=1e-12)  # plan Task 3
    assert params['flank_creep_margin'] == pytest.approx(0.01, abs=1e-12)
