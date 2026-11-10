"""Node tests for pure_pursuit_tech_node: the built C++ binary, on /test_only/* topics.

Run (ROS sourced, workspace built):
    export ROS_DOMAIN_ID=79 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
    python3 -m pytest src/pure_pursuit_tech/test/ -v
or `colcon test --packages-select pure_pursuit_tech`. Every topic the node
touches is remapped under /test_only/, and this module forces domain 79,
localhost only, so nothing here can reach a car's /drive or /joy.

What is asserted is always the published AckermannDriveStamped, never an
internal flag (CLAUDE.md, "Standing requirement -- the LB deadman").
"""

import math
import os
import subprocess
import time

os.environ['ROS_DOMAIN_ID'] = '79'
os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'

import pytest  # noqa: E402
import rclpy  # noqa: E402
from ackermann_msgs.msg import AckermannDriveStamped  # noqa: E402
from ament_index_python.packages import get_package_prefix  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from sensor_msgs.msg import Joy  # noqa: E402
from std_msgs.msg import Bool  # noqa: E402

LB = 4
HELD = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
RELEASED = [0] * 11
SHORT = [1, 1, 1, 1]            # no index 4 at all, every button pressed

# Node settings the oracles below use (all passed explicitly).
WHEELBASE = 0.36
MAX_STEERING = 0.30
MAX_SPEED = 1.0
MIN_SPEED = 0.5
LOOKAHEAD_BASE = 1.0
LOOKAHEAD_MIN = 0.8
LOOKAHEAD_MAX = 3.0
LOOKAHEAD_GAIN = 0.1
JOY_TIMEOUT = 0.5
SAFETY_TIMEOUT = 0.25

EXE = os.path.join(get_package_prefix('pure_pursuit_tech'), 'lib',
                   'pure_pursuit_tech', 'pure_pursuit_tech_node')


def line_waypoints(y, n=201, step=0.1):
    """A straight path parallel to x at lateral offset y."""
    return [(step * k, y) for k in range(n)]


def expected_command(waypoints, car=(0.0, 0.0), yaw=0.0, speed=0.0):
    """Oracle: pure_pursuit_a's control law (racerbot_a
    src/pure_pursuit_a/src/pure_pursuit_a_node.cpp), except the goal: the first
    waypoint from the nearest one onward that is ahead and at least L away, moved
    to where the segment from the waypoint before it (if inside L) crosses the
    circle of radius L. k = 2y/d^2; steering = atan(wheelbase * k) clamped;
    speed linear from MAX_SPEED (straight) to MIN_SPEED (full lock).
    """
    L = min(max(LOOKAHEAD_BASE + LOOKAHEAD_GAIN * abs(speed), LOOKAHEAD_MIN), LOOKAHEAD_MAX)
    start = min(range(len(waypoints)),
                key=lambda i: math.hypot(waypoints[i][0] - car[0], waypoints[i][1] - car[1]))
    prev = None
    for step in range(len(waypoints)):
        wx, wy = waypoints[(start + step) % len(waypoints)]
        dx, dy = wx - car[0], wy - car[1]
        lx = math.cos(yaw) * dx + math.sin(yaw) * dy
        ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
        if lx > 0.0 and math.hypot(dx, dy) >= L:
            if prev is not None and math.hypot(*prev) < L:
                # bisect |prev + t (this - prev)| = L on t in [0, 1]
                lo, hi = 0.0, 1.0
                for _ in range(60):
                    t = (lo + hi) / 2.0
                    if math.hypot(prev[0] + t * (lx - prev[0]), prev[1] + t * (ly - prev[1])) < L:
                        lo = t
                    else:
                        hi = t
                gx, gy = prev[0] + lo * (lx - prev[0]), prev[1] + lo * (ly - prev[1])
                if gx > 0.0:
                    lx, ly = gx, gy
            steer = math.atan(WHEELBASE * 2.0 * ly / (lx * lx + ly * ly))
            steer = max(-MAX_STEERING, min(MAX_STEERING, steer))
            return steer, MAX_SPEED - (MAX_SPEED - MIN_SPEED) * abs(steer) / MAX_STEERING
        prev = (lx, ly)
    raise AssertionError('oracle found no goal')


@pytest.fixture(scope='module', autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


class Harness:
    """Starts the node binary and plays /joy, /safety/brake and pose at it."""

    def __init__(self, tmp_path, waypoints, **params):
        path = tmp_path / 'track.csv'
        path.write_text('x,y,speed\n' + ''.join(f'{x},{y},1.0\n' for x, y in waypoints))
        settings = {
            'use_pf': False,
            'odom_topic': '/test_only/odom',
            'pose_topic': '/test_only/pose',   # use_pf:=true reads the pose here
            'drive_topic': '/test_only/drive',
            'joy_topic': '/test_only/joy',
            'safety_topic': '/test_only/safety/brake',
            'obstacle_topic': '/test_only/obstacle_ahead',
            'waypoints_file': str(path),
            'wheelbase': WHEELBASE, 'max_steering': MAX_STEERING,
            'max_speed': MAX_SPEED, 'min_speed': MIN_SPEED,
            'lookahead_base': LOOKAHEAD_BASE, 'lookahead_min': LOOKAHEAD_MIN,
            'lookahead_max': LOOKAHEAD_MAX, 'lookahead_speed_gain': LOOKAHEAD_GAIN,
            'joy_timeout_sec': JOY_TIMEOUT, 'safety_timeout_sec': SAFETY_TIMEOUT,
            'enable_safety': True,
        }
        settings.update(params)
        args = [EXE, '--ros-args']
        for key, value in settings.items():
            value = str(value).lower() if isinstance(value, bool) else value
            args += ['-p', f'{key}:={value}']
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.node = rclpy.create_node('pure_pursuit_tech_test_driver')
        self.joy_pub = self.node.create_publisher(Joy, '/test_only/joy', 10)
        self.safety_pub = self.node.create_publisher(Bool, '/test_only/safety/brake', 10)
        self.odom_pub = self.node.create_publisher(Odometry, '/test_only/odom', 10)
        self.pose_pub = self.node.create_publisher(Odometry, '/test_only/pose', 10)
        self.step = 0   # play() ticks so far, for a moving pose
        self.commands = []   # (receive time, speed, steering)
        self.node.create_subscription(
            AckermannDriveStamped, '/test_only/drive',
            lambda m: self.commands.append(
                (time.monotonic(), m.drive.speed, m.drive.steering_angle)), 10)
        deadline = time.monotonic() + 15.0
        pubs = (self.joy_pub, self.safety_pub, self.odom_pub) + (
            (self.pose_pub,) if settings['use_pf'] else ())
        while not all(p.get_subscription_count() > 0 for p in pubs):
            assert self.proc.poll() is None, self.proc.stderr.read().decode()
            assert time.monotonic() < deadline, 'node never subscribed'
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def play(self, seconds, joy=None, safety=None, speed=0.0, pose=(0.0, 0.0, 0.0)):
        """Publish at 20 Hz for `seconds`. joy/safety None = don't publish it.

        `pose` is (x, y, yaw), or a function of the tick count giving one, to
        move the car. It goes out on odom_topic and on pose_topic, so it is
        the pose whether use_pf is false or true.
        """
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if joy is not None:
                msg = Joy()
                msg.header.stamp = self.node.get_clock().now().to_msg()
                msg.axes = [0.0] * 8
                msg.buttons = joy
                self.joy_pub.publish(msg)
            if safety is not None:
                self.safety_pub.publish(Bool(data=safety))
            x, y, yaw = pose(self.step) if callable(pose) else pose
            self.step += 1
            odom = Odometry()
            odom.header.frame_id = 'map'
            odom.pose.pose.position.x, odom.pose.pose.position.y = float(x), float(y)
            odom.pose.pose.orientation.z = math.sin(yaw / 2.0)
            odom.pose.pose.orientation.w = math.cos(yaw / 2.0)
            odom.twist.twist.linear.x = speed
            self.odom_pub.publish(odom)
            self.pose_pub.publish(odom)
            stop = time.monotonic() + 0.05
            while time.monotonic() < stop:
                rclpy.spin_once(self.node, timeout_sec=0.01)
        return time.monotonic()

    def since(self, t):
        return [(s, a) for when, s, a in self.commands if when >= t]

    def close(self):
        self.node.destroy_node()
        self.proc.terminate()
        self.proc.wait(timeout=10)


@pytest.fixture
def harness(tmp_path):
    made = []

    def make(waypoints=None, **params):
        made.append(Harness(tmp_path, waypoints or line_waypoints(0.2), **params))
        return made[-1]
    yield make
    for h in made:
        h.close()


def assert_all_zero(commands):
    # >= 10 at 20 Hz: the node really was answering, so "all zero" isn't vacuous
    assert len(commands) >= 10, f'only {len(commands)} commands received'
    assert all(speed == 0.0 and steer == 0.0 for speed, steer in commands), commands


# --- the four LB deny-path states (CLAUDE.md standing requirement) ---------

def test_lb_held_publishes_the_pure_pursuit_command(harness):
    h = harness(line_waypoints(0.2))
    t = h.play(0.5, joy=HELD, safety=False)
    h.play(1.0, joy=HELD, safety=False)
    steer, speed = expected_command(line_waypoints(0.2))
    commands = h.since(t)
    assert len(commands) >= 10
    for s, a in commands:
        assert s == pytest.approx(speed, abs=1e-4)   # 0.1 mm/s
        assert a == pytest.approx(steer, abs=1e-4)   # 0.1 mrad
    assert speed > 0.5 and steer > 0.1   # the oracle itself: a left turn, moving


def test_lb_released_publishes_zero(harness):
    h = harness()
    t = h.play(0.5, joy=RELEASED, safety=False)
    h.play(1.0, joy=RELEASED, safety=False)
    assert_all_zero(h.since(t))


def test_no_joy_ever_publishes_zero(harness):
    h = harness()
    t = h.play(0.5, safety=False)
    h.play(1.0, safety=False)
    assert_all_zero(h.since(t))


def test_stale_joy_publishes_zero(harness):
    h = harness()
    t = h.play(0.5, joy=HELD, safety=False)
    last_joy = h.play(1.0, joy=HELD, safety=False)
    assert any(s > 0.0 for s, _ in h.since(t)), 'LB never authorised motion first'
    h.play(JOY_TIMEOUT + 1.0, safety=False)          # /joy goes silent
    assert_all_zero(h.since(last_joy + JOY_TIMEOUT + 0.15))  # 150 ms transport budget


def test_joy_without_the_lb_index_publishes_zero(harness):
    h = harness()
    t = h.play(0.5, joy=SHORT, safety=False)
    h.play(1.0, joy=SHORT, safety=False)
    assert_all_zero(h.since(t))


# --- racerbot_a's safety_node gate (/safety/brake) -------------------------

def test_safety_brake_true_publishes_zero_with_lb_held(harness):
    h = harness()
    t = h.play(0.5, joy=HELD, safety=True)
    h.play(1.0, joy=HELD, safety=True)
    assert_all_zero(h.since(t))


def test_no_safety_node_publishes_zero_with_lb_held(harness):
    h = harness()
    t = h.play(0.5, joy=HELD)
    h.play(1.0, joy=HELD)
    assert_all_zero(h.since(t))


def test_stale_safety_publishes_zero_with_lb_held(harness):
    h = harness()
    t = h.play(0.5, joy=HELD, safety=False)
    last = h.play(1.0, joy=HELD, safety=False)
    assert any(s > 0.0 for s, _ in h.since(t)), 'safety never authorised motion first'
    h.play(SAFETY_TIMEOUT + 1.0, joy=HELD)           # safety_node goes silent
    assert_all_zero(h.since(last + SAFETY_TIMEOUT + 0.15))


def test_enable_safety_false_still_needs_lb(harness):
    # enable_safety:=false drops only racerbot_a's gate, never the node's own LB check.
    h = harness(enable_safety=False)
    t = h.play(0.5, joy=RELEASED)
    h.play(1.0, joy=RELEASED)
    assert_all_zero(h.since(t))
    t = h.play(0.5, joy=HELD)
    h.play(1.0, joy=HELD)
    _, speed = expected_command(line_waypoints(0.2))
    held = h.since(t)
    assert len(held) >= 10 and all(s == pytest.approx(speed, abs=1e-4) for s, _ in held)


# --- control law edges -------------------------------------------------------

def test_steering_is_clamped_to_max_steering(harness):
    # y=0.5: atan(0.36 * 2*0.5/(0.9^2+0.5^2)) = 0.3275 rad > 0.30, so it clamps
    # and the speed drops to MIN_SPEED (full lock).
    h = harness(line_waypoints(0.5))
    t = h.play(0.5, joy=HELD, safety=False)
    h.play(1.0, joy=HELD, safety=False)
    raw = math.atan(WHEELBASE * 2.0 * 0.5 / (0.9 ** 2 + 0.5 ** 2))
    assert raw > MAX_STEERING
    commands = h.since(t)
    assert len(commands) >= 10
    for s, a in commands:
        # float32 message fields: 1e-6 rad / 1e-6 m/s, ~30x their resolution here
        assert a == pytest.approx(MAX_STEERING, abs=1e-6)
        assert s == pytest.approx(MIN_SPEED, abs=1e-6)


def test_right_offset_path_steers_right(harness):
    # Mirror of the held test: the sign of the steering follows the path.
    h = harness(line_waypoints(-0.2))
    t = h.play(0.5, joy=HELD, safety=False)
    h.play(1.0, joy=HELD, safety=False)
    steer, _ = expected_command(line_waypoints(-0.2))
    commands = h.since(t)
    assert steer < -0.1 and len(commands) >= 10
    assert all(a == pytest.approx(steer, abs=1e-4) for _, a in commands)


@pytest.mark.parametrize('params, message', [
    ({'min_speed': 2.0}, 'need 0 <= min_speed <= max_speed'),
    ({'medium_angle_speed': 5.0}, 'need min_speed <= medium_angle_speed <= max_speed'),
    ({'medium_angle_speed': 0.1}, 'need min_speed <= medium_angle_speed <= max_speed'),
    ({'medium_angle_speed': -0.5}, 'need min_speed <= medium_angle_speed <= max_speed'),
    ({'lookahead_min': 0.0}, 'need 0 < lookahead_min <= lookahead_max'),
    ({'max_steering': 0.0}, 'wheelbase and max_steering must be > 0'),
    ({'joy_timeout_sec': 0.0}, 'joy_timeout_sec and safety_timeout_sec must be > 0'),
    ({'deadman_button': -1}, 'deadman_button must be >= 0'),
    ({'waypoints_file': '', 'track': ''}, 'set track'),
    ({'waypoints_file': '/nonexistent/track.csv'}, 'Cannot open waypoint file'),
    ({'speed_mode': 'fast'}, "speed_mode must be 'steering' or 'profile'"),
    ({'max_accel': 0.0}, 'max_lateral_accel, max_accel and max_brake must be > 0'),
    ({'max_brake': -1.0}, 'max_lateral_accel, max_accel and max_brake must be > 0'),
    ({'laps': -1}, 'need laps >= 0, goal_tolerance > 0 and max_path_error > 0'),
    ({'goal_tolerance': 0.0}, 'need laps >= 0, goal_tolerance > 0 and max_path_error > 0'),
    ({'max_path_error': 0.0}, 'need laps >= 0, goal_tolerance > 0 and max_path_error > 0'),
    ({'loop_closure_distance': -1.0}, 'loop_closure_distance must be >= 0'),
])
def test_bad_settings_refuse_to_start(tmp_path, params, message):
    settings = {'drive_topic': '/test_only/drive', 'joy_topic': '/test_only/joy',
                'waypoints_file': str(tmp_path / 'track.csv')}
    (tmp_path / 'track.csv').write_text('0,0\n1,0\n')
    settings.update(params)
    args = [EXE, '--ros-args']
    for key, value in settings.items():
        if value != '':          # '' = leave the node's default ('') in place
            args += ['-p', f'{key}:={value}']
    done = subprocess.run(args, capture_output=True, timeout=20)
    assert done.returncode != 0
    assert message in done.stderr.decode()


# --- path tracking and speed control (speed_mode, goals, laps, laser offset) ---

def straight_path(n=201, step=0.1):
    return [(step * k, 0.0) for k in range(n)]


def circle_path(r=2.0, n=200):
    return [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)) for k in range(n)]


def on_circle(k, r=2.0, n=200):
    """Pose at waypoint k of circle_path, facing along it (counter-clockwise)."""
    a = 2 * math.pi * (k % n) / n
    return (r * math.cos(a), r * math.sin(a), a + math.pi / 2)


def test_laser_offset_moves_the_pose_back_to_the_rear_axle(harness):
    # The PF pose is the LiDAR's, 0.26 m ahead of the rear axle. A LiDAR at
    # (0.26, 0) means a rear axle at the origin: the command must be the one
    # for a car at the origin (oracle: expected_command, the closed-form law).
    # A slanted line: on a parallel one the interpolated goal is the same from both.
    path = [(0.1 * k, 0.2 + 0.01 * k) for k in range(201)]
    h = harness(path, use_pf=True, laser_offset_x=0.26)
    t = h.play(0.5, joy=HELD, safety=False, pose=(0.26, 0.0, 0.0))
    h.play(1.0, joy=HELD, safety=False, pose=(0.26, 0.0, 0.0))
    steer, speed = expected_command(path, car=(0.0, 0.0))
    unshifted, _ = expected_command(path, car=(0.26, 0.0))
    assert abs(steer - unshifted) > 5e-3   # the oracle itself: the two cases differ
    commands = h.since(t)
    assert len(commands) >= 10
    for s, a in commands:
        assert a == pytest.approx(steer, abs=1e-4)   # 0.1 mrad
        assert s == pytest.approx(speed, abs=1e-4)   # 0.1 mm/s


def test_profile_mode_ramps_up_to_the_cornering_speed(harness):
    # On a 2 m circle the planned speed is sqrt(a_lat * r) = sqrt(2.5 * 2)
    # (closed form); the steering rule would allow 4 - 2.5 * atan(0.36/2)/0.3
    # = 2.52 m/s, so the plan is what binds. max_accel 3 m/s^2 means the
    # command may rise at most 3 * dt per tick: a ramp, not a step.
    h = harness(circle_path(), speed_mode='profile', max_speed=4.0, min_speed=1.5,
                max_lateral_accel=2.5, max_accel=3.0)
    t = h.play(0.3, joy=HELD, safety=False, pose=on_circle(0))
    h.play(2.2, joy=HELD, safety=False, pose=on_circle(0))
    speeds = [s for s, _ in h.since(t) if s > 0.0]
    assert len(speeds) >= 30
    assert speeds[0] <= 3.0 * 0.2 + 1e-6                        # first step: at most a * 0.2 s
    assert all(b >= a - 1e-6 for a, b in zip(speeds, speeds[1:]))   # never falls while ramping
    assert any(0.8 < s < 2.0 for s in speeds)                  # passes through the middle
    assert speeds[-1] == pytest.approx(math.sqrt(2.5 * 2.0), abs=1e-3)   # 1 mm/s


def test_steering_mode_has_no_ramp(harness):
    # The default stays pure_pursuit_a's: the steering rule straight away.
    h = harness(circle_path(), max_speed=4.0, min_speed=1.5)
    t = h.play(0.3, joy=HELD, safety=False, pose=on_circle(0))
    h.play(1.0, joy=HELD, safety=False, pose=on_circle(0))
    speeds = [s for s, _ in h.since(t) if s > 0.0]
    assert len(speeds) >= 10
    # closed form: the pursuit arc of a car on the circle is the circle, k = 1/2
    rule = 4.0 - 2.5 * math.atan(0.36 * 0.5) / 0.30
    assert speeds[0] == pytest.approx(rule, abs=2e-3)


def test_medium_angle_speed_bends_the_steering_rule(harness):
    # Same circle as above, steering atan(0.36 * 0.5) = 0.178 rad of 0.30: past
    # half lock, so on the line from medium_angle_speed (3.5) to min_speed (1.5).
    h = harness(circle_path(), max_speed=4.0, min_speed=1.5, medium_angle_speed=3.5)
    t = h.play(0.3, joy=HELD, safety=False, pose=on_circle(0))
    h.play(1.0, joy=HELD, safety=False, pose=on_circle(0))
    speeds = [s for s, _ in h.since(t) if s > 0.0]
    assert len(speeds) >= 10
    a = math.atan(0.36 * 0.5) / 0.30                 # fraction of full lock
    rule = 3.5 + (1.5 - 3.5) * (a - 0.5) / 0.5       # closed form, m/s
    assert speeds[0] == pytest.approx(rule, abs=2e-3)   # 2 mm/s
    assert rule > 4.0 - 2.5 * a + 0.5                # clearly not the old straight line


def test_open_path_slows_to_its_end_then_stays_stopped(harness):
    h = harness(straight_path(), max_speed=4.0, min_speed=1.5, max_brake=3.0, goal_tolerance=0.3)
    # 2 m before the end: v = sqrt(2 * 3 * (2.0 - 0.3)) (closed form), below max_speed
    t = h.play(0.3, joy=HELD, safety=False, pose=(18.0, 0.0, 0.0))
    h.play(0.7, joy=HELD, safety=False, pose=(18.0, 0.0, 0.0))
    near = [s for s, _ in h.since(t)]
    assert len(near) >= 10
    assert near[-1] == pytest.approx(math.sqrt(2 * 3.0 * 1.7), abs=1e-3)
    # inside the tolerance: zero ...
    t = h.play(0.2, joy=HELD, safety=False, pose=(19.8, 0.0, 0.0))
    h.play(0.8, joy=HELD, safety=False, pose=(19.8, 0.0, 0.0))
    assert_all_zero(h.since(t + 0.1))
    # ... and still zero back at the start with LB held all along: it latched
    t = h.play(0.2, joy=HELD, safety=False, pose=(0.0, 0.0, 0.0))
    h.play(0.8, joy=HELD, safety=False, pose=(0.0, 0.0, 0.0))
    assert_all_zero(h.since(t))
    # releasing LB and pressing it again starts a new run
    h.play(0.5, joy=RELEASED, safety=False, pose=(0.0, 0.0, 0.0))
    t = h.play(0.3, joy=HELD, safety=False, pose=(0.0, 0.0, 0.0))
    h.play(0.7, joy=HELD, safety=False, pose=(0.0, 0.0, 0.0))
    assert any(s > 0.0 for s, _ in h.since(t))


def test_loop_stops_at_the_line_after_its_laps(harness):
    # 4 waypoints a tick round a 200-waypoint circle: one lap in 50 ticks (2.5 s).
    h = harness(circle_path(), laps=1, goal_tolerance=0.3)
    t0 = h.play(0.05, joy=HELD, safety=False, pose=lambda k: on_circle(4 * k))
    h.play(4.5, joy=HELD, safety=False, pose=lambda k: on_circle(4 * k))
    first_half = [s for when, s, _ in h.commands if t0 + 0.2 <= when < t0 + 1.2]   # after LB is seen
    assert first_half and all(s > 0.0 for s in first_half)
    assert_all_zero(h.since(t0 + 3.2))   # one lap and margin later: stopped, car still moving


def test_loop_without_laps_keeps_going(harness):
    # Same drive as above, laps:=0: the stop above is the laps setting, not the loop.
    h = harness(circle_path(), laps=0)
    t0 = h.play(0.05, joy=HELD, safety=False, pose=lambda k: on_circle(4 * k))
    h.play(4.5, joy=HELD, safety=False, pose=lambda k: on_circle(4 * k))
    late = h.since(t0 + 3.2)
    assert len(late) >= 10 and all(s > 0.0 for s, _ in late)


def test_far_from_the_path_publishes_zero(harness):
    h = harness(line_waypoints(0.2), max_path_error=1.5)
    t = h.play(0.3, joy=HELD, safety=False, pose=(5.0, 2.0, 0.0))   # 1.8 m off: over the limit
    h.play(0.8, joy=HELD, safety=False, pose=(5.0, 2.0, 0.0))
    assert_all_zero(h.since(t))
    t = h.play(0.3, joy=HELD, safety=False, pose=(5.0, 1.2, 0.0))   # 1.0 m off: drives
    h.play(0.8, joy=HELD, safety=False, pose=(5.0, 1.2, 0.0))
    assert any(s > 0.0 for s, _ in h.since(t))


def test_facing_away_from_an_open_path_publishes_zero(harness):
    # At the end of an open path, facing back toward its start: the path
    # must not wrap round to its first waypoint, so nothing is ahead.
    h = harness(straight_path(), goal_tolerance=0.05)
    t = h.play(0.3, joy=HELD, safety=False, pose=(19.5, 0.0, math.pi))
    h.play(0.8, joy=HELD, safety=False, pose=(19.5, 0.0, math.pi))
    assert_all_zero(h.since(t))
