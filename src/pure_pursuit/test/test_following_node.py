"""Published-command regressions for stage 1, with real ROS-free logic.

Run via python3 -m pytest src/pure_pursuit/test/ on an isolated ROS domain.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy

from pure_pursuit.pure_pursuit_node import PurePursuitNode
from test_opponent_integration import (
    profiled_csv, _capture_published, _set_pose, _car_ahead_scan,
    _clear_scan, _one_command_interval,
)


@pytest.fixture
def following_node(profiled_csv):
    rclpy.init(args=['--ros-args', '-p', f'waypoints_file:={profiled_csv}',
                     '-p', 'drive_topic:=/test_only/drive',
                     '-p', 'enable_deadman:=false',
                     '-p', 'enable_opponent_following:=true',
                     '-p', 'enable_opponent_overtake:=false',
                     '-p', 'steering_law:=delay_lqr'])
    node = PurePursuitNode()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def test_following_publishes_braking_cap_without_changing_law(following_node):
    node = following_node
    published = _capture_published(node)
    clock = [0.0]
    node._now_sec = lambda: clock[0]  # clock boundary, not driving logic
    _set_pose(node, -1.5, -1.2)
    scan, _ = _car_ahead_scan(1.8)
    for step in range(9):
        clock[0] = step * .025
        node.scan_callback(scan)
        _one_command_interval(node)
        node.control_loop()
    # oracle: invariant; six confirmations spanning .15s still required,
    # following command capped below free-road speed, same delay-aware law.
    assert node.follow_speed_cap is not None
    assert 0 < published[-1].drive.speed <= node.follow_speed_cap < 2.0
    assert node.last_decision_state == 'delay_lqr'
    assert not node.overtake_active
    scan, _ = _car_ahead_scan(.8)
    for step in range(9, 30):
        clock[0] = step * .025
        node.scan_callback(scan)
        _one_command_interval(node)
        node.control_loop()
    assert published[-1].drive.speed == 0.0
    assert not node.overtake_active


def test_reactive_pass_ceiling_cannot_raise_a_slow_profile(following_node):
    node = following_node
    published = _capture_published(node)
    _set_pose(node, -1.5, -1.2)
    node.scan_callback(_clear_scan())
    # A retained confirmed memory suppresses Tier 2, without a pass target.
    node.speed_profile[:] = .6
    node.opponent.arc_length = 5.0
    node.opponent.last_update_time = node._now_sec()
    # Map I/O boundary, as in race_two's source adapter: exact ray ranges
    # for the synthetic vertical wall at x=-0.5. All control math is real.
    def wall_ranges(laser_x, laser_y, yaw, angles):
        cosine = np.cos(yaw + angles)
        return np.where(cosine > .01, (-.5 - laser_x) / np.maximum(cosine, .01), 12.0)
    node._profile_active_since = None
    node._live_map_ray_caster = SimpleNamespace(expected_ranges=wall_ranges)
    for _ in range(12):
        _one_command_interval(node)
        node.control_loop()
    # oracle: invariant; a clear mapped cone is a ceiling, not acceleration
    # permission above the .6 m/s profile. Prior bug assigned that ceiling.
    assert published[-1].drive.speed == pytest.approx(.6, abs=1e-9)  # m/s
    assert not node.overtake_active
