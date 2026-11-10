"""speed_lead_time in the real node: refusals, the published command with and without a lead, and the
LB deadman still stopping a car with a lead configured.

Every node here publishes on /test_only/drive. Run via `python3 -m pytest src/pure_pursuit/test/` on an
isolated ROS domain (CLAUDE.md). The lead's arithmetic is tested in test_racing_math.py (lead_speed).
"""
import math

import numpy as np
import pytest
import rclpy
from nav_msgs.msg import Odometry

from pure_pursuit import racing_math
from pure_pursuit.pure_pursuit_node import PurePursuitNode
from test_tracking_law_integration import (
    _capture_published, _clear_scan, _hold_lb, _one_command_interval, _release_lb, _set_pose,
)

RADIUS = 4.0          # m, a circle: curvature 0.25/m, so the default 2.5 m/s^2 ceiling allows 3.16 m/s
SPACING = 0.1         # m between waypoints


@pytest.fixture(scope='module')
def step_csv(tmp_path_factory):
    """Counter-clockwise circle: 1 m/s for theta in [0, pi), 3 m/s for [pi, 2 pi)."""
    n = int(round(2 * math.pi * RADIUS / SPACING))
    theta = np.arange(n) * (2 * math.pi / n)
    xy = np.column_stack((RADIUS * np.cos(theta), RADIUS * np.sin(theta)))
    speed = np.where(theta < math.pi, 1.0, 3.0)
    path = str(tmp_path_factory.mktemp('lead') / 'step.csv')
    racing_math.save_profiled_csv(path, xy, speed)
    return path


def _node(step_csv, *params):
    rclpy.init(args=['--ros-args', '-p', f'waypoints_file:={step_csv}', '-p', 'steering_law:=pure_pursuit',
                     '-p', 'enable_opponent_awareness:=false', *params,
                     '-p', 'drive_topic:=/test_only/drive'])   # remap after every other param
    try:
        return PurePursuitNode()
    except Exception:
        rclpy.shutdown()
        raise


def _command_at(node, theta, measured_speed):
    """One control tick with LB held, the car on the circle at theta heading along it."""
    _hold_lb(node)
    _set_pose(node, RADIUS * math.cos(theta), RADIUS * math.sin(theta), theta + math.pi / 2)
    node.scan_callback(_clear_scan())
    odom = Odometry()
    odom.twist.twist.linear.x = float(measured_speed)
    node.odom_callback(odom)
    _one_command_interval(node)
    published = _capture_published(node)
    node.control_loop()
    return published[-1].drive.speed


@pytest.mark.parametrize('value', ['-0.1', '1.5', 'nan', 'inf'])
def test_the_node_refuses_a_speed_lead_outside_zero_to_one_second(step_csv, value):
    try:
        with pytest.raises(ValueError, match='speed_lead_time must be finite and in'):
            node = _node(step_csv, '-p', f'speed_lead_time:={value}')
            node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_without_a_lead_the_command_is_the_profile_here(step_csv):
    # oracle: spec (speed_lead_time 0 = the profile speed at the car). 0.4 m before the 1 -> 3 m/s
    # step at theta = pi, measured 1 m/s: the slew from 1.0 towards 1.0 is 1.0.
    node = _node(step_csv)
    try:
        assert node.speed_lead_time == 0.0
        assert _command_at(node, math.pi - 0.4 / RADIUS, 1.0) == pytest.approx(1.0, abs=1e-6)  # m/s
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_a_lead_starts_the_climb_before_the_profile_rises(step_csv):
    # oracle: closed form. Lead 1.0 s at 1 m/s reaches 1.0 m, past the step 0.4 m ahead (with room for
    # the node's nearest waypoint lagging the pose by a few cm-dm), so the target
    # is 3 m/s and the command is one acceleration slew step above the measured 1 m/s:
    # 1.0 + max_acceleration * command_slew_max_dt (6.0 * 0.1 = 0.6 with the shipped values).
    node = _node(step_csv, '-p', 'speed_lead_time:=1.0')
    try:
        expected = min(3.0, 1.0 + node.max_acceleration * node.command_slew_max_dt)
        assert expected > 1.0
        assert _command_at(node, math.pi - 0.4 / RADIUS, 1.0) == pytest.approx(expected, abs=1e-6)  # m/s
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_a_lead_never_brakes_early(step_csv):
    # oracle: spec, braking is never brought forward. 0.4 m before the 3 -> 1 m/s step at theta = 0
    # (2 pi), measured 3 m/s, lead 1.0 s reaches 2.6 m past it: the command stays 3 m/s.
    node = _node(step_csv, '-p', 'speed_lead_time:=1.0')
    try:
        assert _command_at(node, 2 * math.pi - 0.4 / RADIUS, 3.0) == pytest.approx(3.0, abs=1e-6)  # m/s
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_lb_released_still_stops_a_car_with_a_lead(step_csv):
    # oracle: workspace policy (CLAUDE.md, LB deadman). Same pose as the climbing case, LB released.
    node = _node(step_csv, '-p', 'speed_lead_time:=1.0')
    try:
        assert _command_at(node, math.pi - 0.4 / RADIUS, 1.0) > 0.0
        published = _capture_published(node)
        _release_lb(node)
        node.control_loop()
        assert published[-1].drive.speed == 0.0
    finally:
        node.destroy_node()
        rclpy.shutdown()
