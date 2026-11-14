"""The arc braking cap in the real PurePursuitNode (2026-11-14).

Simulator finding (tools/run_analysis/results/2026-11-14-map-racer-stuck-case.md): after a flank creep the
tracking law steered full lock back toward the line, reached 1.4 m/s toward a wall inside the 1 m arc horizon
and arrived at tier 1's fixed 0.14 m stop too fast to make it. The cap holds the speed to what can still stop
with emergency_arc_clearance free, assuming arc_braking_response_sec at that speed then arc_braking_decel.

Oracle: a flat wall square across the path D m ahead of the LiDAR, steering near zero, so the body's swept
path meets it D - 0.21 m on (front edge = wheelbase/2 + car_length/2 - laser_offset_x = 0.18 + 0.29 - 0.26),
and the expected speed is the closed form -a t + sqrt((a t)^2 + 2 a (D - 0.21 - 0.14)). Each tick starts
from a 3.0 m/s command (last_commanded_speed), so an uncapped command stays near 3.

Needs ROS sourced; `rb python3 -m pytest src/pure_pursuit/test/test_arc_braking_cap_node.py`. The deadman is
ENABLED (LB through the node's own joy_callback) and every node publishes on /test_only/drive.
"""
import math
import os

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy, LaserScan

from pure_pursuit import racing_math
from pure_pursuit.pure_pursuit_node import PurePursuitNode

DEADMAN_BUTTON = 4
OPEN = 6.0
FRONT_EDGE = 0.36 / 2 + 0.58 / 2 - 0.26     # m ahead of the LiDAR (docs/hardware-reference.md dimensions)
CAP_ON = ('-p', 'enable_arc_braking_cap:=true', '-p', 'arc_braking_decel:=3.0',
          '-p', 'arc_braking_response_sec:=0.25')


def expected_cap(distance, reserve=0.14, decel=3.0, response=0.25):
    usable = distance - FRONT_EDGE - reserve
    at = decel * response
    return -at + math.sqrt(at * at + 2.0 * decel * usable)


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


def _scan(wall_ahead=None, n=1081, span=math.radians(270)):
    """Open to 6 m; ``wall_ahead`` m: a flat wall square across the path at x = wall_ahead from the LiDAR."""
    msg = LaserScan()
    msg.angle_min = -span / 2.0
    msg.angle_increment = span / (n - 1)
    msg.angle_max = msg.angle_min + (n - 1) * msg.angle_increment
    msg.range_min, msg.range_max = 0.05, 12.0
    angles = msg.angle_min + np.arange(n) * msg.angle_increment
    ranges = np.full(n, OPEN)
    if wall_ahead is not None:
        c = np.cos(angles)
        ranges = np.minimum(ranges, np.where(c > 1e-6, wall_ahead / np.maximum(c, 1e-6), OPEN))
    msg.ranges = [float(r) for r in ranges]
    return msg


@pytest.fixture
def make(profiled_csv):
    nodes = []

    def build(*extra):
        rclpy.init(args=['--ros-args', '-p', f'waypoints_file:={profiled_csv}', '-p', 'intent_rate_hz:=0.0',
                         *extra, '-p', 'drive_topic:=/test_only/drive'])
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


def _tick(node, scan, held=True):
    if held is not None:
        joy = Joy()
        joy.buttons = [0] * (DEADMAN_BUTTON + 1)
        joy.buttons[DEADMAN_BUTTON] = 1 if held else 0
        node.joy_callback(joy)
    pose = PoseStamped()
    pose.pose.position.x, pose.pose.position.y = -1.5, -1.2
    pose.pose.orientation.w = 1.0
    node.pose_callback(pose)
    node.scan_callback(scan)
    node.last_commanded_speed = 3.0          # as if driving at 3 m/s: the slew starts there
    node.control_loop()


def _speeds(node, scan, ticks=8, **kw):
    published = []
    original = node.drive_pub.publish
    node.drive_pub.publish = lambda msg: (published.append(msg), original(msg))
    for _ in range(ticks):
        _tick(node, scan, **kw)
    return published


@pytest.mark.parametrize('wall', [0.8, 0.95, 1.1])
def test_a_wall_inside_the_arc_horizon_caps_the_speed_to_a_stop_short_of_it(make, wall):
    node = make(*CAP_ON)
    published = _speeds(node, _scan(wall_ahead=wall))
    assert abs(published[-1].drive.steering_angle) < 0.05, 'precondition: driving at the wall, not past it'
    assert published[-1].drive.speed == pytest.approx(expected_cap(wall), abs=0.03)   # m/s, ~1 cm of path


def test_the_same_wall_with_the_cap_off_is_not_capped(make):
    node = make('-p', 'enable_arc_braking_cap:=false')
    published = _speeds(node, _scan(wall_ahead=0.8))
    assert published[-1].drive.speed > 2.5


def test_the_cap_is_off_unless_configured(make):
    # The node's own default is off (pure_pursuit.yaml ships it on), as flank_creep_length's is.
    node = make()
    assert node.enable_arc_braking_cap is False
    assert _speeds(node, _scan(wall_ahead=0.8))[-1].drive.speed > 2.5


def test_a_wall_beyond_the_arc_horizon_is_not_capped(make):
    # 1.0 m horizon: a wall 1.5 m ahead of the LiDAR is 1.29 m from the body, so the arc reads inf.
    node = make(*CAP_ON)
    assert _speeds(node, _scan(wall_ahead=1.5))[-1].drive.speed > 2.5
    assert node._arc_braking_cap is None          # the swept path read clear over the whole horizon


def test_weaker_brakes_cap_lower(make):
    node = make('-p', 'enable_arc_braking_cap:=true', '-p', 'arc_braking_decel:=2.0',
                '-p', 'arc_braking_response_sec:=0.25')
    published = _speeds(node, _scan(wall_ahead=0.8))
    assert published[-1].drive.speed == pytest.approx(expected_cap(0.8, decel=2.0), abs=0.03)   # m/s


def test_inside_tier_ones_clearance_it_still_stops(make):
    # 0.30 m ahead of the LiDAR: 0.09 m from the body, inside emergency_arc_clearance (0.14): tier 1 stops at zero.
    node = make(*CAP_ON)
    assert _speeds(node, _scan(wall_ahead=0.30))[-1].drive.speed == 0.0


def test_lb_released_with_the_cap_binding_is_zero(make):
    node = make(*CAP_ON)
    assert _speeds(node, _scan(wall_ahead=0.8))[-1].drive.speed > 0.0, 'precondition: capped but moving'
    assert _speeds(node, _scan(wall_ahead=0.8), ticks=1, held=False)[-1].drive.speed == 0.0


@pytest.mark.parametrize('name, value, match', [
    ('arc_braking_decel', '0.0', 'arc_braking_decel'),
    ('arc_braking_response_sec', '-0.1', 'arc_braking_response_sec'),
    ('arc_braking_response_sec', '2.5', 'arc_braking_response_sec'),
])
def test_bad_values_refuse_to_start(make, name, value, match):
    with pytest.raises(RuntimeError, match=match):
        make('-p', f'{name}:={value}')


def test_the_cap_lifts_once_the_path_clears(make):
    # Each tick computes its own cap: a wall inside the horizon caps, the same node with the wall gone does not.
    node = make(*CAP_ON)
    assert _speeds(node, _scan(wall_ahead=0.8))[-1].drive.speed < 1.5, 'precondition: capped'
    assert _speeds(node, _scan())[-1].drive.speed > 2.5
