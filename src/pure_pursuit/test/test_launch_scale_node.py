"""The launch-lap speed scale in the real node: refusals, the published command with and without it, the LB
deadman still gating a node that has it armed, and the /drive_intent factor. The arithmetic of arming and
counting down is tested without ROS in test_launch_scale.py.

Every node here publishes on /test_only/drive and runs with the deadman ENABLED (the default). Run via
`python3 -m pytest src/pure_pursuit/test/` on an isolated ROS domain (CLAUDE.md).
"""
import math

import numpy as np
import pytest
import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration

from drive_intent import schema
from pure_pursuit import racing_math
from pure_pursuit.pure_pursuit_node import PurePursuitNode
from test_tracking_law_integration import (
    _capture_published, _clear_scan, _hold_lb, _one_command_interval, _release_lb, _set_pose,
)

RADIUS = 4.0          # m, a circle: curvature 0.25/m, so the default 2.5 m/s^2 ceiling allows 3.16 m/s
SPACING = 0.1
PROFILE = 2.0         # m/s everywhere on the circle
THETA = 1.0           # rad, where the car sits


@pytest.fixture(scope='module')
def flat_csv(tmp_path_factory):
    n = int(round(2 * math.pi * RADIUS / SPACING))
    theta = np.arange(n) * (2 * math.pi / n)
    xy = np.column_stack((RADIUS * np.cos(theta), RADIUS * np.sin(theta)))
    path = str(tmp_path_factory.mktemp('launch') / 'flat.csv')
    racing_math.save_profiled_csv(path, xy, np.full(n, PROFILE))
    return path


def _node(csv, *params):
    rclpy.init(args=['--ros-args', '-p', f'waypoints_file:={csv}', '-p', 'steering_law:=pure_pursuit',
                     '-p', 'enable_opponent_awareness:=false', '-p', 'intent_rate_hz:=0.0', *params,
                     '-p', 'drive_topic:=/test_only/drive', '-p', 'intent_topic:=/test_only/drive_intent'])
    try:
        return PurePursuitNode()
    except Exception:
        rclpy.shutdown()
        raise


def _tick(node, measured=PROFILE, *, dt=None, theta=THETA, radius=RADIUS):
    """One control tick, car on the circle heading along it, fresh odometry at ``measured``. The deadman state
    is whatever the caller set. ``dt`` is the time since the previous tick as the launch clock sees it."""
    _set_pose(node, radius * math.cos(theta), radius * math.sin(theta), theta + math.pi / 2)
    node.scan_callback(_clear_scan())
    odom = Odometry()
    odom.twist.twist.linear.x = float(measured)
    node.odom_callback(odom)
    _one_command_interval(node)
    if dt is not None:
        node._launch_last_tick = node.get_clock().now().nanoseconds / 1e9 - dt
    published = _capture_published(node)
    node.control_loop()
    return published[-1].drive.speed


def _run(csv, params, scenario):
    node = _node(csv, *params)
    try:
        return scenario(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _launch_sequence(node, ticks=4, dt=0.1):
    """LB pressed at standstill (no odometry yet: stale counts as standing), then ``ticks`` ticks rolling at the
    profile speed, ``dt`` apart on the launch clock. Returns the published speeds."""
    _hold_lb(node)
    out = [_tick(node)]                                   # the first tick has no previous tick: counts nothing
    out += [_tick(node, dt=dt) for _ in range(ticks - 1)]
    return out


SCALED = ['-p', 'launch_speed_scale:=0.85', '-p', 'launch_scale_distance:=0.5']


@pytest.mark.parametrize('name,value', [('launch_speed_scale', '0.5'), ('launch_speed_scale', '0.0'),
                                        ('launch_speed_scale', '1.5'), ('launch_speed_scale', 'nan'),
                                        ('launch_speed_scale', 'inf'),
                                        ('launch_scale_distance', '-1.0'), ('launch_scale_distance', 'nan'),
                                        ('launch_scale_distance', 'inf')])
def test_the_node_refuses_a_bad_launch_parameter(flat_csv, name, value):
    try:
        with pytest.raises(RuntimeError, match=name):
            node = _node(flat_csv, '-p', f'{name}:={value}')
            node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_the_default_is_off_and_publishes_the_profile(flat_csv):
    # oracle: spec (default 1.0 = off): the command is the profile speed, as on main. Measured = profile,
    # so the slew leaves it unchanged.
    def scenario(node):
        assert node.get_parameter('launch_speed_scale').value == 1.0
        _hold_lb(node)
        return _tick(node), _tick(node, dt=0.1)
    first, second = _run(flat_csv, [], scenario)
    assert first == pytest.approx(PROFILE, abs=1e-6) and second == pytest.approx(PROFILE, abs=1e-6)   # m/s


def test_the_shipped_config_leaves_it_off():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).parents[1] / 'config' / 'pure_pursuit.yaml').read_text())
    params = cfg['pure_pursuit_node']['ros__parameters']
    assert params['launch_speed_scale'] == 1.0 and params['launch_scale_distance'] == 0.0


def test_a_press_from_standstill_scales_the_profile_for_the_distance_then_releases_it(flat_csv):
    # oracle: closed form. 0.85 x 2.0 m/s = 1.7 m/s while metres remain; distance 0.5 m at 2 m/s x 0.1 s =
    # 0.2 m a tick: the first tick counts nothing (remaining 0.5 -> scale), ticks 2 and 3 count to 0.1 (scale),
    # tick 4 counts to 0 and is back to 2.0 (a 0.3 m/s step up, inside the node's slew limit).
    speeds = _run(flat_csv, SCALED, lambda node: _launch_sequence(node, ticks=4))
    assert speeds[:3] == pytest.approx([PROFILE * 0.85] * 3, abs=1e-6)      # m/s
    assert speeds[3] == pytest.approx(PROFILE, abs=1e-6)


def test_two_nodes_differ_only_while_the_launch_distance_lasts(flat_csv):
    plain = _run(flat_csv, [], lambda node: _launch_sequence(node, ticks=5))
    scaled = _run(flat_csv, SCALED, lambda node: _launch_sequence(node, ticks=5))
    assert all(s < p - 0.1 for s, p in zip(scaled[:3], plain[:3])), (scaled, plain)
    assert scaled[3:] == pytest.approx(plain[3:], abs=1e-6)                  # m/s


def test_a_press_while_rolling_does_not_scale(flat_csv):
    # oracle: spec, arming needs a standing start. Fresh odometry at 2 m/s before the press.
    def scenario(node):
        _tick(node)                                       # odometry now fresh at 2 m/s, LB not held: stops
        _hold_lb(node)
        return _tick(node), _tick(node, dt=0.1)
    first, second = _run(flat_csv, SCALED, scenario)
    assert first == pytest.approx(PROFILE, abs=1e-6) and second == pytest.approx(PROFILE, abs=1e-6)


def test_the_scale_is_applied_before_the_speed_cap_not_after_it(flat_csv):
    # oracle: closed form. max_speed 1.5: the profile 2.0 x 0.85 = 1.7 is still above the cap, so the command is
    # the cap, 1.5. Scaling AFTER the cap would give 1.275. The scale can only lower a command.
    speeds = _run(flat_csv, SCALED + ['-p', 'max_speed:=1.5'], lambda node: _launch_sequence(node, ticks=2))
    assert speeds[0] == pytest.approx(1.5, abs=1e-6) and speeds[1] == pytest.approx(1.5, abs=1e-6)   # m/s


def test_a_stale_odometry_tick_does_not_use_up_the_distance(flat_csv):
    # oracle: spec, stale odometry does not count. The tick runs with odometry made stale after arming.
    def scenario(node):
        _hold_lb(node)
        _tick(node)
        for _ in range(5):
            node.last_odom_time = node.get_clock().now() - Duration(seconds=5.0)
            _one_command_interval(node)
            node._launch_last_tick = node.get_clock().now().nanoseconds / 1e9 - 0.1
            node.control_loop()
        return node._launch.remaining
    assert _run(flat_csv, SCALED, scenario) == 0.5            # m, untouched


def test_the_decision_detail_and_intent_name_the_launch_scale(flat_csv):
    def scenario(node):
        seen = []
        original = node.intent_pub.publish
        node.intent_pub.publish = lambda msg: (seen.append(schema.decode(msg.data)), original(msg))
        _launch_sequence(node, ticks=4)
        return seen
    seen = _run(flat_csv, SCALED, scenario)
    assert len(seen) == 4
    for intent in seen[:3]:
        names = {f['name']: f for f in intent['factors']}
        assert names['launch scale']['value'] == pytest.approx(PROFILE * 0.85, abs=1e-3)       # m/s
        assert names['launch scale']['binding'] is True and names['profile speed']['binding'] is False
    # the reason is throttled to state changes: the first tick carries it, with the metres left
    assert 'launch lap x0.85, 0.5 m left' in seen[0]['reason']
    assert 'launch scale' not in {f['name'] for f in seen[3]['factors']}


# --- the LB deadman, for a node with the scale armed (each node needs its own four) -------------------

def test_deadman_held_publishes_a_nonzero_scaled_command(flat_csv):
    speeds = _run(flat_csv, SCALED, lambda node: _launch_sequence(node, ticks=2))
    assert speeds[0] > 0.0 and speeds[1] > 0.0


def test_deadman_released_publishes_zero(flat_csv):
    def scenario(node):
        _launch_sequence(node, ticks=2)
        _release_lb(node)
        return _tick(node, dt=0.1)
    assert _run(flat_csv, SCALED, scenario) == 0.0


def test_no_joy_ever_publishes_zero(flat_csv):
    assert _run(flat_csv, SCALED, lambda node: _tick(node)) == 0.0


def test_stale_joy_publishes_zero(flat_csv):
    def scenario(node):
        _launch_sequence(node, ticks=2)
        node.last_joy_time = node.get_clock().now() - Duration(seconds=node.joy_timeout_sec + 1.0)
        return _tick(node, dt=0.1)
    assert _run(flat_csv, SCALED, scenario) == 0.0


def test_a_new_press_from_standstill_rearms_after_a_release(flat_csv):
    def scenario(node):
        _launch_sequence(node, ticks=5)                    # used up
        assert node._launch.remaining == 0.0
        _release_lb(node)
        _tick(node, measured=0.0)                          # released: stops, the clock restarts
        _hold_lb(node)                                     # odometry fresh at 0 m/s: a standing start
        return node._launch.remaining
    assert _run(flat_csv, SCALED, scenario) == 0.5         # m
