"""Published-command tests. Run with the sourced pure_pursuit pytest suite.

Only map/clock/message I/O is synthetic; planner, tracker, completion gate,
delay_lqr, shaping and publisher path are the real implementations.
"""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from sensor_msgs.msg import Joy, LaserScan

from pure_pursuit import pass_completion, racing_math, tracking_laws
from pure_pursuit.pure_pursuit_node import PurePursuitNode


@pytest.fixture
def node(tmp_path):
    xy = np.vstack((np.column_stack((np.arange(0., 100., .1), np.zeros(1000))),
                    [[100., 100.], [0., 100.]]))
    csv = str(tmp_path / 'line.csv')
    racing_math.save_profiled_csv(csv, xy, np.full(len(xy), 4.))
    rclpy.init(args=['--ros-args', '-p', f'waypoints_file:={csv}',
                     '-p', 'drive_topic:=/test_only/drive',
                     '-p', 'steering_law:=delay_lqr', '-p', 'enable_opponent_following:=true',
                     '-p', 'enable_opponent_overtake:=true', '-p', 'enable_local_pass_planner:=true'])
    n = PurePursuitNode()
    yield n
    n.destroy_node()
    rclpy.shutdown()


def ready(node, *, x=10., y=0., speed=2.):
    pose = PoseStamped()
    pose.pose.position.x, pose.pose.position.y = x, y
    pose.pose.orientation.w = 1.
    node.pose_callback(pose)
    scan = LaserScan()
    scan.angle_min, scan.angle_increment = -math.pi / 2, math.pi / 720
    scan.angle_max = math.pi / 2
    scan.range_min, scan.range_max = .05, 10.
    scan.ranges = [10.] * 721
    node.scan_callback(scan)
    node.current_speed = speed
    node.last_odom_time = node.get_clock().now()
    joy = Joy()
    joy.buttons = [0] * 8
    joy.buttons[node.deadman_button] = 1
    node.joy_callback(joy)
    node.last_command_time = node.get_clock().now() - Duration(seconds=.025)
    node._live_wall_distance = SimpleNamespace(at=lambda x, y: 9.)
    now = node._now_sec()
    node._now_sec = lambda: now  # clock I/O, not a planner substitute
    messages = []
    original = node.drive_pub.publish
    node.drive_pub.publish = lambda msg: (messages.append(msg), original(msg))
    return messages


def active(node, arc=20.):
    now = node._now_sec()
    node.opponent.update(arc, now, node.total_track_length)
    node._local_opponent = (0., 0., now)
    node._following_last_seen_ahead = True
    node.overtake_active = True
    node.overtake_side = 1
    node._pass_completion = pass_completion.CompletionGate()
    node._overtake_started_sec = now - 9.  # specified 8s pass timeout -> abort
    node._overtake_best_lead = -10.
    node._overtake_best_lead_sec = now


@pytest.mark.parametrize('branch', ['follow_brake', 'checked_rejoin', 'zero_stop'])
def test_abort_branches_publish_checked_steering_and_preserve_frozen_track(node, branch):
    published = ready(node, y=.9, speed=2.)
    active(node)
    if branch == 'checked_rejoin':
        # Analytic upper-right occupied quadrant: holding offset clips the
        # corner; a gradual checked return has room during residual braking.
        node._live_wall_distance = SimpleNamespace(at=lambda x, y: max(11.8 - x, 1.50 - y))
    elif branch == 'zero_stop':
        node._live_wall_distance = SimpleNamespace(at=lambda x, y: 0.)
    previous_steering = .08 if branch == 'zero_stop' else 0.
    node.steering_basis = previous_steering
    tracker = node.opponent
    expected_path = (node._local_pass_candidate(100, braking=True,
                                               rejoin=branch == 'checked_rejoin')
                     if branch != 'zero_stop' else None)
    if branch != 'zero_stop':
        assert expected_path is not None  # scenario precondition
        law = copy.deepcopy(node.tracking_law)
        nearest, _ = racing_math.find_nearest_index(expected_path.path.xy, (10., .9), closed=False)
        expected = law.steer(tracking_laws.TrackingInput(10., .9, 0., 2., nearest, previous_steering),
                             expected_path.path).steering
    else:
        expected = .08
    node.control_loop()
    # oracle: approved abort contract -- command zero while braking, use
    # the checked temporary path (or current steering when none exists).
    assert published[-1].drive.speed == 0.
    assert published[-1].drive.steering_angle == pytest.approx(expected, abs=1e-9)  # rad
    assert node._local_abort_event == branch
    assert node.opponent is tracker and node.opponent.arc_length == 20.
    assert node._local_aborting and not node.overtake_active
    assert node.follow_speed_cap == 0.


def test_swerve_into_corridor_aborts_before_rejoining_and_keeps_track(node):
    published = ready(node, y=.8, speed=1.)
    active(node, arc=12.)
    node._overtake_started_sec = node._now_sec()
    # A confirmed surface has swerved alongside; its lateral rate expands
    # the tube across the corridor. Do not manufacture a completion.
    node._local_opponent = (.8, 2., node._now_sec())
    tracker = node.opponent
    node.steering_basis = -.07
    node.control_loop()
    # oracle: invariant -- occupied corridor cannot retain a passing speed.
    assert published[-1].drive.speed == 0.
    assert node._local_abort_event == 'zero_stop'
    assert published[-1].drive.steering_angle == -.07
    assert node.opponent is tracker and tracker.arc_length == 12.
    assert not node.overtake_active


def test_checked_temporary_path_uses_delay_lqr_and_loss_never_completes(node):
    published = ready(node, speed=2.)
    active(node, arc=150.)
    node._overtake_started_sec = node._now_sec()
    node._pass_completion.since = node._now_sec() - 1.
    expected = node._local_pass_candidate(100)
    assert expected is not None  # open arena with a distant opponent tube
    law = copy.deepcopy(node.tracking_law)
    steering = law.steer(tracking_laws.TrackingInput(10., 0., 0., 2., 0, 0.), expected.path).steering
    node.control_loop()
    # oracle: invariant -- chosen open path, rather than legacy offset-point
    # steering; a stale armed completion hold is invalidated by loss of sight.
    assert published[-1].drive.steering_angle == pytest.approx(steering, abs=1e-9)  # rad
    assert 0 < published[-1].drive.speed <= expected.speed_cap
    assert node._pass_completion.since is None
    assert node.overtake_active and not node._local_rejoining
    assert node.follow_speed_cap == pytest.approx(expected.speed_cap, abs=1e-9)  # m/s


@pytest.mark.parametrize('joy_state', ['held', 'released', 'never', 'stale', 'empty'])
def test_planner_cannot_bypass_lb_on_published_command(node, joy_state):
    published = ready(node)
    if joy_state == 'never':
        node.last_joy_time = None
    elif joy_state == 'stale':
        node.last_joy_time = node.get_clock().now() - Duration(seconds=node.joy_timeout_sec + .01)
    elif joy_state in ('released', 'empty'):
        joy = Joy()
        joy.buttons = [] if joy_state == 'empty' else [0] * 8
        node.joy_callback(joy)
    node.control_loop()
    # oracle: spec CLAUDE.md LB contract; deny paths inspect the command.
    assert bool(published[-1].drive.speed > 0.) == (joy_state == 'held')
