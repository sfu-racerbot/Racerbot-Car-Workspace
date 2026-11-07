"""pure_pursuit_node with OpponentGate's shipped defaults: what it takes to start a pass, and to give one up.

Every node here publishes on /test_only/drive with the deadman off (CLAUDE.md: allowed only with that remap);
the deadman's own four deny-path tests are in test_deadman*.py and do not change. The opponent clock
(_now_sec) is stepped in ticks of 1/32 s: near the Hokuyo's 40 Hz, and a binary fraction, so every timer's
boundary falls exactly on a tick with no floating-point drift (at 25 ms ticks "more than 1.5 s" came true on
the 60th tick). Everything else reads the real clock as usual. The gate's 40 Hz closed forms are in
test_opponent_gate.py. The ego is parked at one pose and the "car" is a car-sized
cluster in the scan, moved by changing its range, so track geometry plays no part.
Oracles: closed forms from the shipped defaults (6 sightings, 0.15 s, 0.30 m from a wall, stall 1.5 s,
max pass 8 s, retry 3 s; config/pure_pursuit.yaml) and recorded run 20261106-175300.
Runner: python3 -m pytest src/pure_pursuit/test/ (sourced, own ROS domain) and colcon test.
"""

import math
import os

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
from nav_msgs.msg import OccupancyGrid  # noqa: E402
from sensor_msgs.msg import LaserScan  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402

from pure_pursuit import racing_math  # noqa: E402
from pure_pursuit.pure_pursuit_node import PurePursuitNode  # noqa: E402

TICK = 1.0 / 32.0           # s, see the module docstring
EGO = (-1.5, -1.2, 0.0)     # a pose on the example stadium's straight, as test_opponent_integration.py uses
LASER_X = 0.26              # m, laser_offset_x in pure_pursuit.yaml


@pytest.fixture(scope='module')
def profiled_csv(tmp_path_factory):
    src = os.path.join(os.path.dirname(__file__), '..', 'waypoints', 'example_stadium_raw.csv')
    xy = racing_math.load_xy_csv(src)
    seg_len = racing_math.compute_segment_lengths(xy, closed=True)
    curvature = racing_math.estimate_path_curvature(xy, closed=True)
    speed = racing_math.compute_velocity_profile(
        seg_len, curvature, v_max=6.0, v_min=0.5, a_lat_max=8.0, a_accel_max=3.0, a_brake_max=8.0, closed=True)
    out_path = str(tmp_path_factory.mktemp('waypoints') / 'profiled.csv')
    racing_math.save_profiled_csv(out_path, xy, speed)
    return out_path


@pytest.fixture
def node(profiled_csv):
    rclpy.init(args=['--ros-args',
                     '-p', f'waypoints_file:={profiled_csv}',
                     '-p', 'enable_deadman:=false',
                     '-p', 'opponent_detection_mode:=heuristic',
                     '-p', 'drive_topic:=/test_only/drive'])
    n = PurePursuitNode()
    clock = [64.0]
    n._now_sec = lambda: clock[0]
    n.test_clock = clock
    yield n
    n.destroy_node()
    rclpy.shutdown()


def _pose(node):
    msg = PoseStamped()
    msg.pose.position.x, msg.pose.position.y = EGO[0], EGO[1]
    msg.pose.orientation.z, msg.pose.orientation.w = math.sin(EGO[2] / 2), math.cos(EGO[2] / 2)
    node.pose_callback(msg)


def _car_ahead(car_range):
    """A car-sized cluster dead ahead at car_range, open track either side (left a little more open)."""
    scan = LaserScan()
    n = 361
    scan.angle_min, scan.angle_increment = -math.pi / 2, math.pi / n
    scan.angle_max = scan.angle_min + (n - 1) * scan.angle_increment
    scan.range_min, scan.range_max = 0.05, 12.0
    ranges = [10.0] * n
    c = n // 2
    half = max(2, int(round(math.atan2(0.15, car_range) / scan.angle_increment)))  # a 0.30 m wide car
    for i in range(c - half, c + half):
        ranges[i] = car_range
    for i in range(c + half + 6, c + half + 56):
        ranges[i] = 9.9
    for i in range(c - half - 56, c - half - 6):
        ranges[i] = 9.6
    scan.ranges = ranges
    return scan


def _tick(node, car_range):
    node.test_clock[0] += TICK
    node.scan_callback(_car_ahead(car_range))
    _pose(node)
    node.control_loop()
    return node.overtake_active


def _map_with_block(cx, cy, half=0.15, size_px=400, resolution=0.05, origin=-10.0):
    """A 20 m free arena with border walls and one square block of wall centred on (cx, cy)."""
    grid = np.zeros((size_px, size_px), dtype=np.int8)
    grid[:2, :] = grid[-2:, :] = 100
    grid[:, :2] = grid[:, -2:] = 100
    c0, c1 = int((cx - half - origin) / resolution), int((cx + half - origin) / resolution)
    r0, r1 = int((cy - half - origin) / resolution), int((cy + half - origin) / resolution)
    grid[r0:r1 + 1, c0:c1 + 1] = 100
    msg = OccupancyGrid()
    msg.info.resolution, msg.info.width, msg.info.height = resolution, size_px, size_px
    msg.info.origin.position.x = msg.info.origin.position.y = origin
    msg.info.origin.orientation.w = 1.0
    msg.data = grid.flatten().tolist()
    return msg


def test_a_car_is_passed_only_from_the_sixth_sighting(node):
    # Closed form: the 6th sighting is the 6 needed, spanning 5/32 = 0.156 s >= 0.15 s. Before that nothing
    # is tracked, so no pass.
    active = [_tick(node, 2.0) for _ in range(6)]
    assert active == [False] * 5 + [True]
    assert node.last_decision_state in ('overtake_left', 'overtake_right')


def test_an_unconfirmed_detection_says_why_and_tracks_nothing(node):
    _tick(node, 2.0)
    assert 'not confirmed: seen 1 of 6 needed' in node.last_opponent_status
    assert node.opponent.arc_length is None


def test_a_detection_on_a_mapped_wall_is_never_passed(node):
    # The 175300 phantom, reduced to its cause: the "car" is where the map has a wall. The cluster sits
    # LASER_X + 2.0 m ahead of the ego; put a 0.3 m block of wall there.
    node.map_callback(_map_with_block(EGO[0] + LASER_X + 2.0, EGO[1]))
    active = [_tick(node, 2.0) for _ in range(32)]   # 1 s, far past the 0.15 s confirmation
    assert not any(active)
    assert 'from a mapped wall' in node.last_opponent_status
    assert node.last_decision_state not in ('overtake_left', 'overtake_right')


def test_the_same_car_clear_of_the_walls_is_passed_with_that_map_loaded(node):
    # The permissive half of the test above: same map machinery, block 3 m to the side of the car.
    node.map_callback(_map_with_block(EGO[0] + LASER_X + 2.0, EGO[1] + 3.0))
    active = [_tick(node, 2.0) for _ in range(6)]
    assert active == [False] * 5 + [True]


def _commit(node, car_range=2.0):
    for _ in range(6):
        _tick(node, car_range)
    assert node.overtake_active is True


def test_a_pass_that_gains_nothing_is_given_up_after_the_stall_time(node):
    # The 175300 failure mode: committed, and the lead never grows. Here the "car" holds 2.0 m ahead of a
    # parked ego. Closed form: stall 1.5 s = 48 ticks after the commit; given up on the first tick past it.
    _commit(node)
    held = [_tick(node, 2.0) for _ in range(48)]
    assert all(held)
    assert _tick(node, 2.0) is False
    assert node.last_opponent_status.startswith('overtake abandoned: no gain on the opponent for 1.5')


def test_after_giving_up_no_new_pass_for_the_retry_time(node):
    _commit(node)
    for _ in range(49):
        _tick(node, 2.0)
    assert node.overtake_active is False
    # Closed form: retry 3.0 s = 96 ticks from the abandon. The car is still there and is re-confirmed
    # after 6 ticks, but no pass starts until the retry time is up.
    waiting = [_tick(node, 2.0) for _ in range(95)]
    assert not any(waiting)
    assert 'no new pass for another' in node.last_opponent_status
    assert _tick(node, 2.0) is True


def test_a_pass_that_keeps_gaining_slowly_is_given_up_at_the_time_limit(node):
    # The car comes back towards the parked ego at 0.1 m/s (range falls 0.1/32 m a tick), so the lead gains
    # 0.1 m a second: enough to never stall (gain > 0.10 m inside 1.5 s), never enough to finish.
    # Closed form: max pass 8.0 s = 256 ticks; given up on the first tick past it.
    car = 2.0
    for _ in range(6):
        _tick(node, car)
    assert node.overtake_active is True
    held = []
    for _ in range(256):
        car -= 0.1 / 32
        held.append(_tick(node, car))
    assert all(held)
    car -= 0.1 / 32
    assert _tick(node, car) is False
    assert node.last_opponent_status.startswith('overtake abandoned: pass still unfinished after 8.0')


@pytest.mark.parametrize('name,value', [
    ('overtake_stall_sec', 0.0), ('overtake_stall_sec', float('nan')),
    ('overtake_max_duration_sec', -1.0), ('overtake_min_lead_gain', -0.1),
    ('overtake_retry_sec', float('inf')), ('opponent_min_wall_distance', -0.1),
    ('opponent_confirm_count', 0)])
def test_bad_pass_and_gate_parameters_refuse_to_start(profiled_csv, name, value):
    rclpy.init(args=['--ros-args',
                     '-p', f'waypoints_file:={profiled_csv}',
                     '-p', 'enable_deadman:=false',
                     '-p', 'drive_topic:=/test_only/drive',
                     '-p', f'{name}:={value}'])
    try:
        stem = name.replace('opponent_', '').replace('confirm_', '')
        with pytest.raises(ValueError, match=stem):
            PurePursuitNode()
    finally:
        rclpy.shutdown()
