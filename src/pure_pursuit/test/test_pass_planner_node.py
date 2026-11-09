"""Stage 2 (passing a stopped car) in the real node: parameter refusals, the LB
deadman's four deny paths with the planner on, and solo driving unchanged.

Every node here publishes on /test_only/drive. Run via
`python3 -m pytest src/pure_pursuit/test/` on an isolated ROS domain
(CLAUDE.md). The state machine itself is exercised through the node's own
method bodies in tools/f1tenth_sim/tests/test_race_two.py.
"""
import math

import numpy as np
import pytest
import rclpy
from rclpy.duration import Duration

from pure_pursuit import pass_planner
from pure_pursuit.pure_pursuit_node import PurePursuitNode
from test_tracking_law_integration import (  # noqa: F401 (fixture)
    profiled_csv, _capture_published, _clear_scan, _hold_lb, _one_command_interval,
    _pose_slightly_left, _release_lb, _set_pose,
)

PLANNER = ('-p', 'enable_local_pass_planner:=true', '-p', 'enable_opponent_following:=true',
           '-p', 'enable_opponent_overtake:=false')


def _node(profiled_csv, *params):
    rclpy.init(args=['--ros-args', '-p', f'waypoints_file:={profiled_csv}',
                     '-p', 'steering_law:=delay_lqr', *params,
                     '-p', 'drive_topic:=/test_only/drive'])   # remap after every other param
    try:
        return PurePursuitNode()
    except Exception:
        rclpy.shutdown()
        raise


@pytest.fixture
def planner_node(profiled_csv):
    node = _node(profiled_csv, *PLANNER)
    yield node
    node.destroy_node()
    rclpy.shutdown()


@pytest.mark.parametrize('params, message', [
    (('-p', 'enable_local_pass_planner:=true', '-p', 'enable_opponent_following:=true',
      '-p', 'enable_opponent_overtake:=true'), 'replaces enable_opponent_overtake'),
    (('-p', 'enable_local_pass_planner:=true', '-p', 'enable_opponent_following:=false',
      '-p', 'enable_opponent_overtake:=false'), 'needs enable_opponent_following'),
    ((*PLANNER, '-p', 'pass_standoff:=0.5'), 'pass_standoff must be between'),
    ((*PLANNER, '-p', 'pass_speed_uncertainty:=0.1'), 'pass_speed_uncertainty must be in'),
    ((*PLANNER, '-p', 'pass_clear_margin:=0.1'), 'pass_clear_margin in'),
])
def test_the_node_refuses_an_unsafe_planner_configuration(profiled_csv, params, message):
    try:
        with pytest.raises(ValueError, match=message):
            node = _node(profiled_csv, *params)
            node.destroy_node()
            rclpy.shutdown()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


# --- LB deadman: the four deny paths, planner on (CLAUDE.md standing rule) ---

def test_planner_node_lb_held_drives(planner_node):
    _hold_lb(planner_node)
    _pose_slightly_left(planner_node)
    _one_command_interval(planner_node)
    published = _capture_published(planner_node)
    planner_node.control_loop()
    assert published[-1].drive.speed > 0.0


def test_planner_node_lb_released_stops(planner_node):
    _hold_lb(planner_node)
    _pose_slightly_left(planner_node)
    planner_node.control_loop()
    published = _capture_published(planner_node)
    _release_lb(planner_node)
    planner_node.control_loop()
    assert published[-1].drive.speed == 0.0


def test_planner_node_joy_never_received_stops(planner_node):
    _pose_slightly_left(planner_node)
    _one_command_interval(planner_node)   # so a bypassed gate would publish > 0
    published = _capture_published(planner_node)
    planner_node.control_loop()
    assert published[-1].drive.speed == 0.0


def test_planner_node_joy_stale_stops(planner_node):
    _hold_lb(planner_node)
    _pose_slightly_left(planner_node)
    planner_node.last_joy_time = planner_node.get_clock().now() - Duration(
        seconds=planner_node.joy_timeout_sec + 1.0)
    _one_command_interval(planner_node)   # so a bypassed gate would publish > 0
    published = _capture_published(planner_node)
    planner_node.control_loop()
    assert published[-1].drive.speed == 0.0


def test_planner_node_lb_released_mid_pass_stops(planner_node):
    # An active pass (state forced; its geometry is tested elsewhere) never
    # outranks the deadman, which is checked before anything else.
    _hold_lb(planner_node)
    _pose_slightly_left(planner_node)
    planner_node.control_loop()
    planner_node._pass_state, planner_node.overtake_active = 'pass', True
    published = _capture_published(planner_node)
    _release_lb(planner_node)
    planner_node.control_loop()
    assert published[-1].drive.speed == 0.0


def test_an_active_pass_plan_is_what_the_law_tracks(planner_node):
    # On the stadium's bottom straight (y = -1.2, car heading +x), on the
    # line. A plan shifting 0.6 m off the line steers towards the side its
    # target lies on in the car's own frame; the line alone steers (near)
    # straight. Same node, same pose, one tick each.
    node = planner_node
    _hold_lb(node)
    _set_pose(node, -1.5, -1.2, 0.0)
    node.scan_callback(_clear_scan())
    _one_command_interval(node)
    published = _capture_published(node)
    node.control_loop()
    straight = published[-1].drive.steering_angle
    line = node._pass_line_frame()
    s0, _ = line.frenet(-1.5, -1.2)
    # A short plan straight off the shift (build() directly: the stadium's
    # bends are too tight for a full-length pass plan, which is refused).
    shift = pass_planner.Manoeuvre(s0, 0.0, -0.6, 2.5, math.inf, 2.5, -1)
    open_floor = lambda xs, ys: np.full(len(xs), 5.0)
    s = s0 + np.arange(0.0, 2.0, 0.1)
    node._pass_plan = pass_planner.build(line, s0, shift.lateral(s, line), 'pass', node._pass_limits(),
                                         open_floor)
    assert node._pass_plan is not None
    side = math.copysign(1.0, float(line.point(np.array(s0 + 2.5), np.array(-0.6))[1]) + 1.2)
    _set_pose(node, -1.5, -1.2, 0.0)
    _one_command_interval(node)
    node.control_loop()
    # Oracle: body-frame side of the plan's target (car heads +x, so +y is left).
    assert side * (published[-1].drive.steering_angle - straight) > 0.02   # rad


# --- Solo driving is unchanged by turning the planner on ----------------------

def _solo_commands(profiled_csv, *params):
    node = _node(profiled_csv, '-p', 'enable_opponent_following:=true',
                 '-p', 'enable_opponent_overtake:=false', *params)
    try:
        _hold_lb(node)
        published = _capture_published(node)
        for _ in range(12):
            _pose_slightly_left(node)
            _one_command_interval(node)
            node.control_loop()
        return [(m.drive.steering_angle, m.drive.speed) for m in published]
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_with_no_car_in_sight_the_planner_changes_no_command(profiled_csv):
    # Invariant: planner on publishes exactly what planner off does when no
    # opponent is confirmed (the law runs once per tick on the racing line).
    off = _solo_commands(profiled_csv, '-p', 'enable_local_pass_planner:=false')
    on = _solo_commands(profiled_csv, '-p', 'enable_local_pass_planner:=true')
    assert len(on) == len(off) == 12
    for (s_on, v_on), (s_off, v_off) in zip(on, off):
        assert s_on == pytest.approx(s_off, abs=1e-6)   # rad
        assert v_on == pytest.approx(v_off, abs=1e-6)   # m/s
    assert any(abs(v) > 0 for _, v in on)
    assert all(math.isfinite(s) for s, _ in on)
