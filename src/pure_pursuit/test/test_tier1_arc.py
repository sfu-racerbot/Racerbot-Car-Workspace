"""
Tier 1 of pure_pursuit_node._reactive_override: the swept-path arc check and the raw cone
backstop (2026-11-07). The real rclpy node is driven scan by scan. Each test asserts the override
the node returns, or the command control_loop() publishes.

Every test runs for both cars. The node is built from each car's effective pure_pursuit_node
parameters (src/pure_pursuit/config/pure_pursuit.yaml layered with
src/racerbot_car/cars/<id>/calibration/pure_pursuit.yaml, the same merge the launch makes), so the
full lock is the car's own: rb1 0.33 rad, rb2 0.42 rad.

Oracles, named per test:
  closed form  -- the expected value is written out from the geometry in the test
  brute force  -- _brute_contact_arc: drives the car's rectangle along the arc and asks when it
                  first covers the point. It shares no code with racing_math or the node.
  spec         -- a value in the car's calibration or pure_pursuit.yaml, or a message the node
                  is specified to raise

Needs ROS2 sourced and pure_pursuit built, and an isolated ROS domain:
    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    export ROS_DOMAIN_ID=79 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
    python3 -m pytest src/pure_pursuit/test/test_tier1_arc.py -v
"""
import contextlib
import math
import os
import re

import numpy as np
import pytest
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from sensor_msgs.msg import LaserScan

from pure_pursuit import racing_math
from pure_pursuit.pure_pursuit_node import PurePursuitNode
from racerbot_car.layers import effective_config

CARS = ('rb1', 'rb2')
# spec: src/racerbot_car/cars/<id>/calibration/pure_pursuit.yaml, max_steering_angle
FULL_LOCK = {'rb1': 0.33, 'rb2': 0.42}
TIER1_STATES = ('emergency_obstacle', 'emergency_escape')
TIER0_OR_TIER1 = ('body_contact', 'body_contact_escape') + TIER1_STATES
TRIP_M = 0.14       # spec: emergency_arc_clearance, pure_pursuit.yaml
BACKSTOP_M = 0.25   # spec: emergency_hard_stop_distance, pure_pursuit.yaml
BODY_CONTACT_M = 0.03  # spec: emergency_stop_clearance, pure_pursuit.yaml
# Inside the 60 deg cone, 10 deg to the right, and past the 3 cm contact band (closed form:
# 0.246 - 0.21 / cos(10 deg) = 0.0328 m), so Tier 0 does not claim it, yet inside the backstop.
SLIVER = (-10.0, 0.246)


@pytest.fixture(scope='module')
def profiled_csv(tmp_path_factory):
    """The checked-in example track, profiled once for the module."""
    src = os.path.join(os.path.dirname(__file__), '..', 'waypoints', 'example_stadium_raw.csv')
    xy = racing_math.load_xy_csv(src)
    seg_len = racing_math.compute_segment_lengths(xy, closed=True)
    curvature = racing_math.estimate_path_curvature(xy, closed=True)
    speed = racing_math.compute_velocity_profile(
        seg_len, curvature, v_max=6.0, v_min=0.5, a_lat_max=8.0, a_accel_max=3.0,
        a_brake_max=8.0, closed=True)
    out_path = str(tmp_path_factory.mktemp('waypoints') / 'profiled.csv')
    racing_math.save_profiled_csv(out_path, xy, speed)
    return out_path


@pytest.fixture(scope='module')
def car_configs(tmp_path_factory):
    """Each car's effective pure_pursuit_node parameters, written as a --params-file."""
    configs = {}
    for car in CARS:
        values = effective_config('pure_pursuit.yaml', 'pure_pursuit_node', car)
        path = tmp_path_factory.mktemp(car) / 'pure_pursuit.yaml'
        path.write_text(yaml.safe_dump({'pure_pursuit_node': {'ros__parameters': values}}))
        configs[car] = str(path)
    return configs


@pytest.fixture(params=CARS)
def car(request):
    return request.param


@pytest.fixture
def build(profiled_csv, car_configs, car):
    """Context manager building this test's car's node, with extra -p overrides last.

    The deadman is off and the drive topic is a test topic, as in test_opponent_integration.py.
    Each node gets its own rclpy context.
    """
    @contextlib.contextmanager
    def make(*extra):
        rclpy.init(args=['--ros-args',
                         '--params-file', car_configs[car],
                         '-p', f'waypoints_file:={profiled_csv}',
                         '-p', 'enable_deadman:=false',
                         '-p', 'drive_topic:=/test_only/drive',
                         *extra])
        try:
            node = PurePursuitNode()
        except Exception:
            rclpy.shutdown()
            raise
        try:
            # Preconditions the scenes depend on: this car's steering limit is in effect, and
            # commands go to the test topic, never /drive.
            assert node.max_steering_angle == pytest.approx(FULL_LOCK[car], abs=1e-9)
            assert node.drive_pub.topic_name == '/test_only/drive'
            yield node
        finally:
            node.destroy_node()
            rclpy.shutdown()
    return make


def _scan(returns=(), flat_wall=None):
    """A 270 deg, 1081-beam scan, the Hokuyo's geometry, open (10 m) except for the returns.

    returns    -- (bearing_deg, range_m) pairs, each set on its nearest beam
    flat_wall  -- a range d: every beam within +/-30 deg reads d / cos(bearing), a wall d metres
                  ahead across the whole avoidance cone
    """
    n = 1081
    scan = LaserScan()
    scan.angle_min = math.radians(-135.0)
    scan.angle_increment = math.radians(270.0) / (n - 1)
    scan.angle_max = scan.angle_min + (n - 1) * scan.angle_increment
    scan.range_min, scan.range_max = 0.05, 30.0
    ranges = [10.0] * n
    if flat_wall is not None:
        for index in range(n):
            bearing = scan.angle_min + index * scan.angle_increment
            if abs(bearing) <= math.radians(30.0):
                ranges[index] = flat_wall / math.cos(bearing)
    for bearing_deg, rng in returns:
        ranges[_beam(scan, bearing_deg)] = rng
    scan.ranges = ranges
    return scan


def _beam(scan, bearing_deg):
    return int(round((math.radians(bearing_deg) - scan.angle_min) / scan.angle_increment))


def _beam_point(node, scan, index):
    """The base_link point a beam's return lands on (the LiDAR sits at laser_offset_x/y)."""
    angle = scan.angle_min + index * scan.angle_increment
    rng = float(scan.ranges[index])
    return (node.laser_offset_x + rng * math.cos(angle),
            node.laser_offset_y + rng * math.sin(angle))


def _brute_contact_arc(node, point, curvature, horizon=1.0, step=0.001):
    """Brute-force oracle: the rear-axle arc length at which the car's rectangle first covers the
    base_link point while driving the constant-curvature arc, or inf within the horizon. The
    geometry comes from the node's own parameters. Left is positive."""
    half = node.car_width / 2.0
    front = node.wheelbase / 2.0 + node.car_length / 2.0
    rear = node.wheelbase / 2.0 - node.car_length / 2.0
    x0, y0 = point
    if abs(curvature) < 1e-9:
        if abs(y0) > half or x0 < rear:
            return math.inf
        return max(x0 - front, 0.0)
    radius = 1.0 / curvature
    sign = 1.0 if radius > 0.0 else -1.0
    span = abs(radius)
    for phi in np.arange(0.0, horizon / span + 1e-12, step / span):
        axle_x = span * math.sin(phi)
        axle_y = sign * span * (1.0 - math.cos(phi))
        heading = sign * phi
        dx, dy = x0 - axle_x, y0 - axle_y
        body_x = dx * math.cos(heading) + dy * math.sin(heading)
        body_y = -dx * math.sin(heading) + dy * math.cos(heading)
        if rear <= body_x <= front and abs(body_y) <= half:
            return span * phi
    return math.inf


def _curvature(node, steering):
    return math.tan(steering) / node.wheelbase


def _tier1(node, scan, desired, basis=0.0):
    """Feed one scan and return the override for this tick's desired steering."""
    node.scan_callback(scan)
    node.steering_basis = basis
    return node._reactive_override(desired_steering=desired)


def _set_pose(node, x, y, yaw=0.0):
    msg = PoseStamped()
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.z = math.sin(yaw / 2.0)
    msg.pose.orientation.w = math.cos(yaw / 2.0)
    node.pose_callback(msg)


def _capture_published(node):
    published = []
    original = node.drive_pub.publish
    node.drive_pub.publish = lambda msg: (published.append(msg), original(msg))
    return published


# ============================================================================
# The regression this fixes, and the full-lock cases the arc check exists for
# ============================================================================

def test_an_outside_wall_on_a_full_lock_turn_does_not_stop_the_car(build, car):
    """The regression this change fixes. Full lock left, a wall 25 deg to the right at 0.33 m.
    The old 60 deg cone rule read 0.33 m and stopped, but the car turns away from that wall.
    Oracle: brute force. The car never reaches that point on the full-lock arc, so Tier 1 must
    not fire."""
    with build() as node:
        full = node.max_steering_angle
        scan = _scan([(-25.0, 0.33)])
        point = _beam_point(node, scan, _beam(scan, -25.0))
        assert _brute_contact_arc(node, point, _curvature(node, full)) == math.inf  # oracle
        steering, speed, state, detail = _tier1(node, scan, desired=full, basis=full)
        assert state not in TIER0_OR_TIER1, detail
        assert 'swept body path' not in detail


def test_a_wall_across_the_cone_on_a_full_lock_turn_stops_the_car(build, car):
    """Full lock left, a flat wall 0.33 m ahead across the whole cone, with no opening to crawl
    through: the car stops, its speed is zero, and the detail names the swept path.
    Oracle: brute force. The body reaches the wall's centre after about 0.11 m of arc, inside the
    0.14 m threshold."""
    with build() as node:
        full = node.max_steering_angle
        scan = _scan(flat_wall=0.33)
        point = _beam_point(node, scan, _beam(scan, 0.0))
        assert _brute_contact_arc(node, point, _curvature(node, full)) < TRIP_M  # oracle
        steering, speed, state, detail = _tier1(node, scan, desired=full, basis=full)
        assert state == 'emergency_obstacle', detail
        assert speed == 0.0
        assert 'swept body path' in detail


def test_a_single_dead_ahead_return_on_a_full_lock_turn_crawls_out_not_drives(build, car):
    """The same full-lock wall as one return: the cone is open beside it, so the existing escape
    crawls at emergency_escape_speed toward the opening rather than driving on at the plan's
    speed. Oracle: brute force (as above)."""
    with build() as node:
        full = node.max_steering_angle
        scan = _scan([(0.0, 0.33)])
        point = _beam_point(node, scan, _beam(scan, 0.0))
        assert _brute_contact_arc(node, point, _curvature(node, full)) < TRIP_M  # oracle
        steering, speed, state, detail = _tier1(node, scan, desired=full, basis=full)
        assert state == 'emergency_escape', detail
        assert speed == pytest.approx(node.emergency_escape_speed, abs=1e-12)  # spec: 0.25 m/s
        assert speed <= 0.3


def test_the_arc_being_steered_onto_counts_even_when_the_rack_is_straight(build, car):
    """A return at +40 deg, 0.312 m, inside a full-left turn. The rack is still straight, but the
    car is being steered onto the arc that reaches it, so Tier 1 must trip. With the rack and the
    command both straight the same return is clear. Oracles: brute force for both (the body
    reaches it on the full-lock arc, and never on the straight one)."""
    with build() as node:
        full = node.max_steering_angle
        scan = _scan([(40.0, 0.312)])
        point = _beam_point(node, scan, _beam(scan, 40.0))
        assert _brute_contact_arc(node, point, _curvature(node, full)) < TRIP_M    # oracle
        assert _brute_contact_arc(node, point, 0.0) == math.inf                    # oracle
        steering, speed, state, detail = _tier1(node, scan, desired=full, basis=0.0)
        assert state in TIER1_STATES, detail
        assert 'swept body path' in detail
        steering, speed, state, detail = _tier1(node, scan, desired=0.0, basis=0.0)
        assert state not in TIER0_OR_TIER1, detail


def test_the_arc_the_rack_is_on_counts_when_the_command_has_straightened(build, car):
    """A return the front-left corner of the body reaches on a full-left arc and never reaches
    straight: base_link (0.52, 0.17), 0.17 m left of the centreline, outside the 0.155 m half width.
    The rack still at full lock and the command straight must trip on the rack's own arc; the mirror
    (rack straight, command full lock) trips on the commanded arc; both straight is clear. Oracles:
    closed form for the lateral offset (0.17 m > 0.155 m), brute force (contact 0.059 m on the
    full-left arc, inf straight). Without the rack's arc, the first case would pass unseen."""
    with build() as node:
        full = node.max_steering_angle
        forward = 0.52 - node.laser_offset_x
        bearing = math.degrees(math.atan2(0.17, forward))
        scan = _scan([(bearing, math.hypot(forward, 0.17))])
        point = _beam_point(node, scan, _beam(scan, bearing))
        assert point[1] > node.car_width / 2.0                                      # closed form
        assert _brute_contact_arc(node, point, _curvature(node, full)) < TRIP_M      # oracle
        assert _brute_contact_arc(node, point, 0.0) == math.inf                      # oracle
        steering, speed, state, detail = _tier1(node, scan, desired=0.0, basis=full)
        assert state in TIER1_STATES, detail
        assert 'swept body path' in detail
        steering, speed, state, detail = _tier1(node, scan, desired=0.0, basis=0.0)
        assert state not in TIER0_OR_TIER1, detail
        steering, speed, state, detail = _tier1(node, scan, desired=full, basis=0.0)
        assert state in TIER1_STATES, detail


def test_a_return_on_the_full_lock_arc_trips_inside_the_threshold_only(build, car):
    """A return on the car's own full-lock path at 0.10 m of arc beyond the front edge trips; one
    at 0.20 m does not. Oracle: closed form. The return sits on the rear axle's path, so its free
    distance is the arc length from the front edge: 0.10 m and 0.20 m. The beam's bearing is
    rounded to the nearest beam (0.25 deg), which moves the point by mm only."""
    with build() as node:
        full = node.max_steering_angle
        radius = node.wheelbase / math.tan(full)
        front = node.wheelbase / 2.0 + node.car_length / 2.0

        def on_path(arc_from_front):
            phi = (front + arc_from_front) / radius
            x = radius * math.sin(phi)
            y = radius * (1.0 - math.cos(phi))
            lx = x - node.laser_offset_x
            return math.degrees(math.atan2(y, lx)), math.hypot(lx, y)

        near = _scan([on_path(0.10)])
        steering, speed, state, detail = _tier1(node, near, desired=full, basis=full)
        assert state in TIER1_STATES, detail
        assert 'swept body path' in detail

        far = _scan([on_path(0.20)])
        steering, speed, state, detail = _tier1(node, far, desired=full, basis=full)
        assert state not in TIER0_OR_TIER1, detail


# ============================================================================
# Straight-line equivalence, the threshold itself, and the cone backstop
# ============================================================================

def test_a_wall_straight_ahead_stops_at_the_threshold_and_not_beyond(build, car):
    """Straight ahead the arc check is the old 0.35 m cone check shifted by the 0.21 m from the
    LiDAR to the body's front edge: a raw 0.33 m wall (0.12 m free) and 0.34 m (0.13 m free) stop,
    0.36 m (0.15 m free) and 0.37 m (0.16 m free) do not. Oracle: closed form, 0.35 - 0.21 =
    0.14 m."""
    with build() as node:
        front = node.wheelbase / 2.0 + node.car_length / 2.0
        assert front - node.laser_offset_x == pytest.approx(0.21, abs=1e-12)  # m, closed form

        steering, speed, state, detail = _tier1(node, _scan(flat_wall=0.33), desired=0.0)
        assert state == 'emergency_obstacle', detail
        assert speed == 0.0
        steering, speed, state, detail = _tier1(node, _scan(flat_wall=0.34), desired=0.0)
        assert state in TIER1_STATES, detail
        steering, speed, state, detail = _tier1(node, _scan(flat_wall=0.36), desired=0.0)
        assert state not in TIER0_OR_TIER1, detail
        steering, speed, state, detail = _tier1(node, _scan(flat_wall=0.37), desired=0.0)
        assert state not in TIER0_OR_TIER1, detail


def test_the_threshold_is_strict_at_its_own_value(build, car):
    """A return whose free distance equals emergency_arc_clearance does not trip: the check is
    'free < threshold'. The scan is built, then read back as the node reads it (float32 ranges),
    and the threshold is set to the free distance of that exact beam, so the equality is exact in
    floating point and a '<=' mutant trips. Oracle: closed form for the value (0.35 + 0.26 - 0.47
    = 0.14 m), spec for the strictness."""
    scan = _scan([(0.0, 0.35)])
    index = _beam(scan, 0.0)
    free = racing_math.swept_path_clearance(
        np.array([float(scan.ranges[index])]), np.array([True]),
        np.array([scan.angle_min + index * scan.angle_increment]), 0.0,
        0.155, 0.58, 0.36, 0.26, 0.0, 1.0)
    assert free == pytest.approx(0.14, abs=1e-6)  # m, closed form (float32 range rounding)
    with build('-p', f'emergency_arc_clearance:={float(free)!r}') as node:
        steering, speed, state, detail = _tier1(node, scan, desired=0.0)
        assert state not in TIER0_OR_TIER1, detail


def test_a_return_in_the_cone_beside_the_body_path_does_not_stop_the_car(build, car):
    """Regression on the other side of the change. 28 deg to the left at 0.34 m, straight ahead:
    inside the 60 deg cone, where the old 0.35 m rule stopped, but its lateral offset
    0.34 sin 28 = 0.160 m is past the 0.155 m half width, so the body never reaches it. Oracle:
    closed form (the lateral offset) and brute force."""
    with build() as node:
        scan = _scan([(28.0, 0.34)])
        point = _beam_point(node, scan, _beam(scan, 28.0))
        assert 0.34 < 0.35                                             # the old rule's threshold
        assert abs(point[1]) > node.car_width / 2.0                    # closed form: 0.160 m
        assert _brute_contact_arc(node, point, 0.0) == math.inf        # oracle
        steering, speed, state, detail = _tier1(node, scan, desired=0.0)
        assert state not in TIER0_OR_TIER1, detail


def test_a_return_inside_the_backstop_is_also_inside_the_arc_at_full_lock(build, car):
    """Measured, and it changed the test. The sliver return (10 deg to the right, 0.246 m, 3 cm
    clear of the body) is inside the backstop AND inside the swept path, even on a full-lock left
    turn, where the body reaches it after about 3 cm of arc. So the cone backstop adds no stop
    here: the arc already makes it. (Measured over a grid, see the backstop redundancy note in
    docs/racing-autonomy.md.) Oracles: brute force (contact 0.031 m, under the 0.14 m trip), closed
    form for the clearance (0.246 - 0.21 / cos 10 deg = 0.033 m, above Tier 0's 0.03 m), spec for
    the backstop's own value (0.25 m)."""
    with build() as node:
        full = node.max_steering_angle
        scan = _scan([SLIVER])
        point = _beam_point(node, scan, _beam(scan, SLIVER[0]))
        assert _brute_contact_arc(node, point, _curvature(node, full)) < TRIP_M       # oracle
        assert SLIVER[1] - 0.21 / math.cos(math.radians(-SLIVER[0])) > BODY_CONTACT_M  # closed form
        assert SLIVER[1] < BACKSTOP_M                                                  # spec
        steering, speed, state, detail = _tier1(node, scan, desired=full, basis=full)
        assert state in TIER1_STATES, detail
        assert 'swept body path' in detail


def test_the_backstop_stops_the_sliver_at_every_steering_in_the_envelope(build, car):
    """The sliver at straight, full left and full right. Each must stop the car, and each detail
    names the backstop (the cone clause is in every detail). Which check fires first is not the
    point here: the test that isolates the backstop is the threshold test below. Oracle: spec
    (inside the backstop, at every steering)."""
    with build() as node:
        full = node.max_steering_angle
        scan = _scan([SLIVER])
        for steering_rad in (0.0, full, -full):
            steering, speed, state, detail = _tier1(
                node, scan, desired=steering_rad, basis=steering_rad)
            assert state in TIER1_STATES, (steering_rad, detail)
            assert 'hard-stop backstop' in detail, (steering_rad, detail)


@pytest.mark.parametrize('bearing_range, expected', [
    ((-10.0, 0.245), 'trips'),   # inside the backstop: 0.245 < 0.25
    ((-10.0, 0.255), 'clear'),   # past it:            0.255 > 0.25
])
def test_the_backstop_threshold_is_its_own_value(build, car, bearing_range, expected):
    """With the arc check switched off (emergency_arc_clearance 0.0, which no free distance
    reaches), only the backstop can stop the car. It must trip inside 0.25 m and not past it.
    Oracle: spec (emergency_hard_stop_distance 0.25 m, pure_pursuit.yaml). A mutant that deletes
    the backstop, or moves it to 0.35 m, fails one of the two cases."""
    with build('-p', 'emergency_arc_clearance:=0.0') as node:
        scan = _scan([bearing_range])
        steering, speed, state, detail = _tier1(node, scan, desired=0.0)
        if expected == 'trips':
            assert state in TIER1_STATES, detail
            assert 'hard-stop backstop' in detail
        else:
            assert state not in TIER0_OR_TIER1, detail


def test_a_return_behind_the_sensor_inside_the_car_does_not_trip(build, car):
    """A return at -120 deg, 0.15 m, is the car's own rear structure: a chassis echo that lies
    inside the body. The body window is +/-90 deg, so it is never read; had it been read, the
    swept path would report 0.0 on every scan. Oracle: closed form (the return sits at
    (0.185, -0.130) m in base_link, inside the rectangle x in [-0.11, 0.47], |y| <= 0.155).
    Measured on car 2's 2026-10-30 run, 62.7% of close rear returns are inside the footprint
    (docs/racing-autonomy.md, reactive safety)."""
    with build() as node:
        scan = _scan([(-120.0, 0.15)])
        x, y = _beam_point(node, scan, _beam(scan, -120.0))
        front = node.wheelbase / 2.0 + node.car_length / 2.0
        rear = node.wheelbase / 2.0 - node.car_length / 2.0
        assert rear <= x <= front and abs(y) <= node.car_width / 2.0  # oracle: closed form
        steering, speed, state, detail = _tier1(node, scan, desired=0.0)
        assert state not in TIER0_OR_TIER1, detail


def test_a_reading_below_range_min_is_the_sensors_invalid_code_not_an_obstacle(build, car):
    """A dead-ahead reading of 0.01 m, below the scan's range_min (0.05 m), is the sensor's invalid
    code, not a wall at the bumper. Counting it would stop the car on every scan the sensor marks
    that way. Oracle: spec (range_min is the sensor's own floor; _closest_in_cone and
    _footprint_clearance already treat readings below it as invalid)."""
    with build() as node:
        steering, speed, state, detail = _tier1(node, _scan([(0.0, 0.01)]), desired=0.0)
        assert state not in TIER0_OR_TIER1, detail


@pytest.mark.parametrize('kind', ['inf', 'nan', 'zero', 'negative'])
def test_unknown_beams_are_never_obstacles(build, car, kind):
    """An all-inf, all-NaN, all-zero or all-negative scan is unknown, not an obstacle: the node
    overrides nothing. Oracle: spec (invalid beams are unknown; the node's own validity test)."""
    value = {'inf': math.inf, 'nan': math.nan, 'zero': 0.0, 'negative': -1.0}[kind]
    with build() as node:
        scan = _scan()
        scan.ranges = [value] * len(scan.ranges)
        steering, speed, state, detail = _tier1(
            node, scan, desired=node.max_steering_angle, basis=node.max_steering_angle)
        assert state is None, detail
        assert steering is None and speed is None


def test_a_non_finite_steering_is_blocked_not_driven(build, car):
    """A steering that cannot be evaluated reads as blocked, so the car stops or crawls rather than
    publishing a NaN command. Oracle: spec (fail-safe, racing_math.tier1_arc_clearance)."""
    with build() as node:
        steering, speed, state, detail = _tier1(node, _scan(), desired=math.nan)
        assert state in TIER1_STATES, detail
        assert 'swept body path' in detail


def test_the_detail_names_the_check_that_fired_and_its_value(build, car):
    """The override's detail says which check fired and by how much, for the dashboard and the run
    logs. Oracle: spec (the message format in _reactive_override)."""
    with build() as node:
        steering, speed, state, detail = _tier1(node, _scan(flat_wall=0.33), desired=0.0)
        assert re.search(r'free distance along the arc is \d\.\d{3}m, inside the 0\.14m arc '
                         r'clearance', detail), detail
        steering, speed, state, detail = _tier1(node, _scan([SLIVER]), desired=0.0)
        assert re.search(r'is \d\.\d\dm, inside the 0\.25m hard-stop backstop', detail), detail


@pytest.mark.parametrize('extra, message', [
    (['-p', 'emergency_arc_length:=0.0'], 'emergency_arc_length must be finite and > 0'),
    (['-p', 'emergency_arc_length:=0.10'], 'emergency_arc_length must be at least '
                                           'emergency_arc_clearance'),
    (['-p', 'emergency_arc_clearance:=-0.01'], 'must be finite and non-negative'),
    (['-p', 'emergency_hard_stop_distance:=-0.01'], 'must be finite and non-negative'),
])
def test_a_bad_arc_configuration_refuses_to_start(build, car, extra, message):
    """The horizon must be positive, because swept_path_clearance raises for a non-positive horizon
    inside the control loop. It must also reach the trip distance, or the band between them is
    never measured. Negative thresholds are refused too. Oracle: spec (the startup validation in
    pure_pursuit_node's __init__)."""
    with pytest.raises(RuntimeError, match=message):
        with build(*extra):
            pass


# ============================================================================
# The command that reaches the drive topic
# ============================================================================

def test_the_published_command_stops_for_a_wall_inside_the_arc_threshold(build, car):
    """Through control_loop(), the stop reaches the drive topic: speed 0.0 on the published
    command, and the decision is emergency_obstacle. The same flat 0.33 m wall as the closed-form
    test above (free distance 0.12 m, under 0.14 m). Asserts the published command, the standing
    requirement for driving-node tests."""
    with build() as node:
        published = _capture_published(node)
        _set_pose(node, -1.5, -1.2, 0.0)
        node.scan_callback(_scan(flat_wall=0.33))
        node.control_loop()
        assert node.last_decision_state == 'emergency_obstacle'
        assert published, 'control_loop published nothing'
        assert published[-1].drive.speed == 0.0


def test_the_published_command_is_not_stopped_by_tier1_on_a_clear_track(build, car):
    """The other side: an open scan at the same pose does not trip Tier 0 or Tier 1, so the
    command is the plan's own. Oracle: spec (an open scan is clear)."""
    with build() as node:
        published = _capture_published(node)
        _set_pose(node, -1.5, -1.2, 0.0)
        node.scan_callback(_scan())
        node.control_loop()
        assert node.last_decision_state not in TIER0_OR_TIER1, node.last_decision_state
        assert published, 'control_loop published nothing'
