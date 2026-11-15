"""The escape-arc credit (swept_clearance_reach, 2026-11-14) in the real GapFollowNode and in the simulator's
mirror (tools/f1tenth_sim/gap_mirror.py) must command the same, tick by tick, as test_gap_mirror_parity.py
requires of the rest of the pipeline. The node is built from the shipped YAML with reach 0.1; the scenes approach
corners at speed so the credit binds. Needs ROS sourced:

    rb python3 -m pytest src/gap_follow/test/test_gap_mirror_escape_arc_parity.py
"""
import os
import sys

import numpy as np
import pytest
import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from sensor_msgs.msg import Joy

from gap_follow.gap_follow_node import GapFollowNode
from conftest import car_config_args

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'tools', 'f1tenth_sim'))
import gap_mirror  # noqa: E402

sys.path.insert(0, HERE)
from test_gap_follow_corners_node import _left_turn, _mirror, _raycast  # noqa: E402
from test_gap_mirror_parity import DT, PACKAGED_CONFIG, TEST_ONLY_DRIVE, TOLERANCE, _Lidar  # noqa: E402

REACH = 0.1


def _scenes():
    """(scan, speed, rack steering) per tick: a 0.7 m left turn approached at 2.0-2.5 m/s with the rack already
    turning 0.1-0.2 rad, where a slightly tighter arc the car can hold clears the outer wall (found by sweeping the
    mirror; reach 0 and 0.1 command different speeds there), and its mirror image (gap on the right)."""
    out = []
    for cx in (2.0, 1.8, 1.6):
        for speed, basis in ((2.0, 0.2), (2.5, 0.1), (2.5, 0.2)):
            out.append((_raycast(_left_turn(0.7, 0.7, cx)), speed, basis))
            out.append((_raycast(_mirror(_left_turn(0.7, 0.7, cx))), speed, -basis))
    # A narrower 0.6 m entry, where the credited arc is itself blocked further on, so its charge (latency plus the
    # rack's travel) sets the speed: dropping the latency changes these commands (found the same way).
    for cx, speed, basis in ((2.4, 2.0, 0.2), (2.4, 2.5, 0.2), (1.4, 2.5, 0.1), (1.4, 2.5, 0.2)):
        out.append((_raycast(_left_turn(0.6, 0.7, cx)), speed, basis))
    return out


def test_mirror_and_node_agree_with_the_escape_arc_on():
    rclpy.init(args=['--ros-args', *car_config_args(PACKAGED_CONFIG), '-p', f'swept_clearance_reach:={REACH}',
                     '-p', 'enable_launch_boost:=false', *TEST_ONLY_DRIVE])
    node = GapFollowNode()
    try:
        assert node.drive_pub.topic_name == '/test_only/drive'
        assert node.swept_clearance_reach == REACH
        published = []
        node.drive_pub.publish = lambda msg: published.append(msg)
        node.command_slew_max_dt = DT
        params = gap_mirror.node_params()
        params.update(swept_clearance_reach=REACH, enable_launch_boost=False, command_slew_max_dt=DT)
        off = dict(params, swept_clearance_reach=0.0)
        scenes = _scenes()
        mirror = gap_mirror.GapController(params, _Lidar(scenes[0][0]))
        mirror_off = gap_mirror.GapController(off, _Lidar(scenes[0][0]))
        joy = Joy()
        joy.buttons = [0] * 5
        joy.buttons[4] = 1
        differs = 0
        for step, (scan, speed, basis) in enumerate(scenes):
            node.last_commanded_speed = speed       # each tick from its own speed and rack angle
            node.steering_basis = basis
            node.joy_callback(joy)
            odom = Odometry()
            odom.twist.twist.linear.x = speed
            node.odom_callback(odom)
            node.last_command_time = node.get_clock().now() - Duration(seconds=DT)
            for m in (mirror, mirror_off):
                m.steering_basis = node.steering_basis
                m.last_commanded_speed = node.last_commanded_speed
                m.previous_target_idx = node.previous_target_idx
                m.last_command_time = step - 1
            node.scan_callback(scan)
            expected = mirror.command(np.asarray(scan.ranges), speed, step)
            drive = published[-1].drive
            assert drive.steering_angle == pytest.approx(expected[0], abs=TOLERANCE), f'tick {step}: steering'
            assert drive.speed == pytest.approx(expected[1], abs=TOLERANCE), f'tick {step}: speed'
            # the same tick with the credit off, from the same carried state
            off_speed = mirror_off.command(np.asarray(scan.ranges), speed, step)[1]
            differs += abs(off_speed - expected[1]) > 1e-3
        assert differs >= 1, 'the scenes must make the credit change the command, or this proves nothing'
    finally:
        node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize('value', ['-0.05', '0.31', 'nan'])
def test_the_node_refuses_a_reach_out_of_range(value):
    rclpy.init(args=['--ros-args', *car_config_args(PACKAGED_CONFIG), '-p', f'swept_clearance_reach:={value}',
                     *TEST_ONLY_DRIVE])
    try:
        with pytest.raises(ValueError, match='swept_clearance_reach'):
            GapFollowNode()
    finally:
        rclpy.shutdown()
