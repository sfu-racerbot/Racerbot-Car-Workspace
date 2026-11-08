"""apply_map_safety's escape tier, checked against pure_pursuit_node._reactive_override.

pure_pursuit_node.py imports rclpy and the ROS message packages at module level, so the
node's decision logic cannot be imported without ROS. Each expectation below is a closed
form written from the node's effective parameters (racerbot_car.layers.effective_config,
the layered config the node loads) and from the body footprint. None is read back from
apply_map_safety.

The scans sit on the harness's 1081-beam grid: beam i points at -135 + 0.25 * i deg, so
0 deg is beam 540 and +90 deg is beam 900. A flat wall distance d ahead of the LiDAR
returns d / cos(t) at beam angle t. The body's front edge is
front = wheelbase/2 + car_length/2 - laser_offset_x ahead of the LiDAR (0.21 m on both
cars), so such a wall has body clearance (d - front) / cos(t), smallest dead ahead.

Runner: RB_WS=<worktree> rb python3 -m pytest tools/f1tenth_sim/tests/ (the ROS wrapper
supplies numpy and the sim). conftest.py runs every test once per calibrated car.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import run_validation as rv  # noqa: E402
from racerbot_car.layers import effective_config  # noqa: E402
from sim_fidelity.calibration import selected_car  # noqa: E402

# Input command passed through when the safety tiers do not act. Chosen so that a
# pass-through cannot be confused with the 0.25 m/s crawl or with a stop.
INPUT_STEERING = 0.12
INPUT_SPEED = 2.4
# The open run in the scenes is (0, 30] deg; its centre beam is 15 deg, to the left.
OPEN_RUN_CENTRE_DEG = 15.0
# How far a wall sits beyond the escape floor in the "body clear" scenes, in metres.
BODY_MARGIN = 0.02


def _setup():
    car = selected_car(note=False)
    rv._ensure_config(car)
    lidar = rv._lidar()
    node = effective_config('pure_pursuit.yaml', 'pure_pursuit_node', car)
    return car, lidar, node


def _beam(angle_deg, lidar):
    """Index of the beam at angle_deg on the harness grid (whole and quarter degrees)."""
    index = round((math.radians(angle_deg) - lidar.angle_min) / lidar.angle_increment)
    assert math.isclose(lidar.angle_min + index * lidar.angle_increment,
                        math.radians(angle_deg), abs_tol=1e-9)  # rad, the grid is exact
    return index


def _front_edge(node):
    """Distance from the LiDAR to the body's front edge, along the centreline (m)."""
    return node['wheelbase'] / 2.0 + node['car_length'] / 2.0 - node['laser_offset_x']


def _blank(lidar):
    return np.full(lidar.num_beams, np.inf)


def _flat_wall(scan, lidar, distance, lo_deg, hi_deg):
    for index in range(_beam(lo_deg, lidar), _beam(hi_deg, lidar) + 1):
        scan[index] = distance / math.cos(lidar.angle_min + index * lidar.angle_increment)
    return scan


def _escape_scene(lidar, distance):
    """A wall across the right half of the forward cone; the left half, 0 to 30 deg, is open."""
    return _flat_wall(_blank(lidar), lidar, distance, -30.0, 0.0)


def _body_clear_wall(node):
    """Wall distance whose body clearance clears emergency_escape_clearance by BODY_MARGIN."""
    return _front_edge(node) + node['emergency_escape_clearance'] + BODY_MARGIN


def test_a_wall_ahead_with_room_and_an_open_side_escapes_toward_the_opening():
    car, lidar, node = _setup()
    distance = _body_clear_wall(node)  # body clearance 0.12 m
    # Inside emergency_stop_distance, so the emergency tier is the one that acts.
    assert distance < node['emergency_stop_distance']
    steering, speed, result = rv.apply_map_safety(
        _escape_scene(lidar, distance), INPUT_STEERING, INPUT_SPEED, car)
    assert result == "escape"
    assert speed == node['emergency_escape_speed']  # m/s, the node's crawl speed
    # The open run's centre is 15 deg to the left; left is positive steering.
    assert steering == pytest.approx(math.radians(OPEN_RUN_CENTRE_DEG), abs=1e-9)  # rad


def test_b_same_scene_with_something_inside_the_escape_clearance_stops():
    car, lidar, node = _setup()
    distance = _body_clear_wall(node)
    # Premise: without the flank return this scene escapes, so the flank alone decides it.
    assert rv.apply_map_safety(
        _escape_scene(lidar, distance), INPUT_STEERING, INPUT_SPEED, car)[2] == "escape"
    scan = _escape_scene(lidar, distance)
    # A return 0.05 m outside the left flank: body clearance 0.05 m. That is above contact
    # (0.03 m) and below the 0.10 m escape floor, so the node stops rather than crawls.
    scan[_beam(90.0, lidar)] = node['car_width'] / 2.0 + 0.05
    assert rv.apply_map_safety(scan, INPUT_STEERING, INPUT_SPEED, car) == (
        INPUT_STEERING, 0.0, "stop")


def test_c_wall_ahead_with_no_opening_stops():
    car, lidar, node = _setup()
    scan = _flat_wall(_blank(lidar), lidar, _body_clear_wall(node), -60.0, 60.0)
    cone = slice(_beam(-30.0, lidar), _beam(30.0, lidar) + 1)
    # Premise: every beam in the avoidance cone is at or below the minimum gap depth.
    assert np.all(scan[cone] <= node['emergency_escape_min_gap'])
    assert rv.apply_map_safety(scan, INPUT_STEERING, INPUT_SPEED, car) == (
        INPUT_STEERING, 0.0, "stop")


def test_d_nothing_inside_the_stop_distance_passes_the_command_through():
    car, lidar, node = _setup()
    distance = node['emergency_stop_distance'] + 0.15  # 0.50 m
    assert distance > node['emergency_stop_distance']
    scan = _flat_wall(_blank(lidar), lidar, distance, -60.0, 60.0)
    assert rv.apply_map_safety(scan, INPUT_STEERING, INPUT_SPEED, car) == (
        INPUT_STEERING, INPUT_SPEED, "none")


def test_escape_floor_just_below_refuses_the_crawl():
    car, lidar, node = _setup()
    # Body clearance 0.095 m, 5 mm under the 0.10 m escape floor.
    distance = _front_edge(node) + node['emergency_escape_clearance'] - 0.005
    assert rv.apply_map_safety(
        _escape_scene(lidar, distance), INPUT_STEERING, INPUT_SPEED, car) == (
            INPUT_STEERING, 0.0, "stop")


def test_escape_floor_just_above_allows_the_crawl():
    car, lidar, node = _setup()
    # Body clearance 0.105 m, 5 mm over the 0.10 m escape floor.
    distance = _front_edge(node) + node['emergency_escape_clearance'] + 0.005
    steering, speed, result = rv.apply_map_safety(
        _escape_scene(lidar, distance), INPUT_STEERING, INPUT_SPEED, car)
    assert result == "escape"
    assert speed == node['emergency_escape_speed']  # m/s
    assert steering == pytest.approx(math.radians(OPEN_RUN_CENTRE_DEG), abs=1e-9)  # rad


def test_forward_contact_with_an_opening_crawls_out_under_the_contact_floor():
    car, lidar, node = _setup()
    # 0.02 m from the body dead ahead: inside emergency_stop_clearance (0.03 m) and above
    # body_contact_escape_clearance (0.0 m). That contact floor is not the 0.10 m escape floor.
    distance = _front_edge(node) + 0.02
    assert 0.02 <= node['emergency_stop_clearance']
    assert 0.02 > node['body_contact_escape_clearance']
    steering, speed, result = rv.apply_map_safety(
        _escape_scene(lidar, distance), INPUT_STEERING, INPUT_SPEED, car)
    assert result == "escape"
    assert speed == node['emergency_escape_speed']  # m/s
    assert steering == pytest.approx(math.radians(OPEN_RUN_CENTRE_DEG), abs=1e-9)  # rad


def test_flank_contact_stops_even_with_an_open_forward_path():
    car, lidar, node = _setup()
    scan = _blank(lidar)  # the forward cone is clear, so a gap is there to steer into
    scan[_beam(90.0, lidar)] = node['car_width'] / 2.0 + 0.01  # 0.01 m off the left flank
    assert rv.apply_map_safety(scan, INPUT_STEERING, INPUT_SPEED, car) == (
        INPUT_STEERING, 0.0, "stop")


def test_forward_contact_without_an_opening_stops():
    car, lidar, node = _setup()
    scan = _flat_wall(_blank(lidar), lidar, _front_edge(node) + 0.02, -60.0, 60.0)
    assert rv.apply_map_safety(scan, INPUT_STEERING, INPUT_SPEED, car) == (
        INPUT_STEERING, 0.0, "stop")


@pytest.mark.parametrize("key, value, open_run, expected", [
    # The body (0.12 m) no longer clears a raised escape floor.
    ("emergency_escape_clearance", 0.13, None, "stop"),
    # 0.0 is the documented way back to the old latching stop.
    ("emergency_escape_speed", 0.0, None, "stop"),
    # The crawl speed is the parameter, not a constant.
    ("emergency_escape_speed", 0.40, None, "escape"),
    # The 0.33 m wall's swept arc is 0.088 m at the input steering. test_the_arc_threshold_
    # is_bracketed_by_the_brute_force_contact checks that against an independent oracle, so
    # the two rows below sit either side of it. emergency_stop_distance no longer decides
    # Tier 1: the swept arc does.
    ("emergency_arc_clearance", 0.08, None, "none"),
    ("emergency_arc_clearance", 0.09, None, "escape"),
    # 0.12 m is contact once the contact threshold is raised, and contact crawls at its own
    # 0.0 m floor.
    ("emergency_stop_clearance", 0.13, None, "escape"),
    # A 0.9 m open run is a gap at the node's 0.8 m minimum...
    ("emergency_escape_min_gap", 0.8, 0.9, "escape"),
    # ...and is not one at 1.0 m.
    ("emergency_escape_min_gap", 1.0, 0.9, "stop"),
])
def test_e_the_tier_thresholds_are_the_nodes_parameters(key, value, open_run, expected,
                                                        monkeypatch):
    car, lidar, node = _setup()
    monkeypatch.setitem(rv.PURE_PURSUIT_PARAMS, key, value)
    scan = _escape_scene(lidar, _body_clear_wall(node))
    if open_run is not None:
        scan[_beam(0.25, lidar):_beam(30.0, lidar) + 1] = open_run
    steering, speed, result = rv.apply_map_safety(scan, INPUT_STEERING, INPUT_SPEED, car)
    want_steering, want_speed = {
        "stop": (INPUT_STEERING, 0.0),
        "none": (INPUT_STEERING, INPUT_SPEED),
        "escape": (math.radians(OPEN_RUN_CENTRE_DEG),
                   rv.PURE_PURSUIT_PARAMS['emergency_escape_speed']),
    }[expected]
    assert result == expected
    assert steering == pytest.approx(want_steering, abs=1e-9)  # rad
    assert speed == pytest.approx(want_speed, abs=1e-12)  # m/s


def _brute_contact(node, point, curvature, horizon=1.0, step=0.001):
    """Brute-force oracle: the rear-axle arc length at which the car's rectangle first covers the
    base_link point driving the constant-curvature arc, or inf within the horizon. It shares no
    code with racing_math or apply_map_safety. Left is positive."""
    half = node['car_width'] / 2.0
    front = node['wheelbase'] / 2.0 + node['car_length'] / 2.0
    rear = node['wheelbase'] / 2.0 - node['car_length'] / 2.0
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


def _return_points(scan, lidar, node):
    """base_link points of every finite beam (the LiDAR sits at laser_offset_x/y)."""
    angles = lidar.angle_min + np.arange(lidar.num_beams) * lidar.angle_increment
    points = []
    for index in np.flatnonzero(np.isfinite(scan)):
        rng = float(scan[index])
        points.append((node['laser_offset_x'] + rng * math.cos(angles[index]),
                       node['laser_offset_y'] + rng * math.sin(angles[index])))
    return points


def test_the_arc_threshold_is_bracketed_by_the_brute_force_contact():
    """The table's arc rows (0.08 m: no stop, 0.09 m: crawl) decide this wall only if the model's
    free distance lies between them. The safety oracle is the real contact: the model must never
    be later than the car's brute-force contact with the wall, or the car would touch it before
    Tier 1 acted. Oracle: brute force over the wall's own returns (the real contact), and the
    invariant model <= brute. The 0.08 to 0.09 m bracket is a change-detector (not an oracle),
    kept so the table's rows keep meaning what they say."""
    car, lidar, node = _setup()
    scan = _escape_scene(lidar, _body_clear_wall(node))
    curvature = math.tan(INPUT_STEERING) / node['wheelbase']
    brute = min(_brute_contact(node, point, curvature) for point in _return_points(scan, lidar, node))
    model = rv._arc_clearance(scan, INPUT_STEERING, INPUT_STEERING, rv.PURE_PURSUIT_PARAMS)
    assert model <= brute + 1e-9, (model, brute)  # invariant: never later than the real contact
    assert 0.08 < model < 0.09, model  # change-detector (not an oracle): the rows straddle it


def test_the_rack_arc_counts_in_the_simulator_too():
    """The node's rack-only scene (tests/test_tier1_arc.py): one return at base_link (0.52, 0.17),
    0.17 m left of the centreline, outside the body. Rack at full lock with the command straight
    must stop the car, and with the rack straight it must pass through. Oracle: brute force
    (contact inside the 0.14 m threshold on the full-lock arc, inf straight)."""
    car, lidar, node = _setup()
    full = node['max_steering_angle']
    forward = 0.52 - node['laser_offset_x']
    index = round((math.atan2(0.17, forward) - lidar.angle_min) / lidar.angle_increment)
    beam_angle = lidar.angle_min + index * lidar.angle_increment
    scan = _blank(lidar)
    scan[index] = math.hypot(forward, 0.17)  # range from the LiDAR to the return
    point = (node['laser_offset_x'] + scan[index] * math.cos(beam_angle),
             node['laser_offset_y'] + scan[index] * math.sin(beam_angle))
    assert point[1] > node['car_width'] / 2.0  # closed form: the return is outside the body
    assert _brute_contact(node, point, math.tan(full) / node['wheelbase']) < 0.14  # oracle
    assert _brute_contact(node, point, 0.0) == math.inf                             # oracle
    assert rv.apply_map_safety(scan, 0.0, INPUT_SPEED, car, applied_steering=full) == (
        0.0, 0.0, "stop")
    assert rv.apply_map_safety(scan, 0.0, INPUT_SPEED, car, applied_steering=0.0) == (
        0.0, INPUT_SPEED, "none")


@pytest.mark.parametrize("distance, expected", [(0.245, "stop"), (0.255, "none")])
def test_the_backstop_is_its_own_value_in_the_simulator(distance, expected, monkeypatch):
    """With the arc check off (no free distance is below zero), only the raw cone backstop can stop
    the car. A return 10 deg to the right at 0.245 m is inside the 0.25 m backstop and past the
    3 cm contact band (closed form: 0.245 - 0.21 / cos 10 deg = 0.032 m), so it stops; at 0.255 m
    it does not. Oracle: spec (emergency_hard_stop_distance, pure_pursuit.yaml)."""
    car, lidar, node = _setup()
    monkeypatch.setitem(rv.PURE_PURSUIT_PARAMS, "emergency_arc_clearance", 0.0)
    scan = _blank(lidar)
    scan[_beam(-10.0, lidar)] = distance
    assert rv.apply_map_safety(scan, INPUT_STEERING, INPUT_SPEED, car)[2] == expected


def test_a_return_behind_the_sensor_is_outside_the_arc_window_in_the_simulator():
    """A return at -120 deg, 0.15 m, is the car's own rear structure, inside the body. The body
    window is +/-90 deg, so it is never read; read, it would put the swept path at 0.0 on every scan.
    Oracle: closed form (the return is at (0.185, -0.130) m in base_link, inside the rectangle
    x in [-0.11, 0.47] m, |y| <= 0.155 m)."""
    car, lidar, node = _setup()
    scan = _blank(lidar)
    scan[_beam(-120.0, lidar)] = 0.15
    x = node['laser_offset_x'] + 0.15 * math.cos(math.radians(-120.0))
    y = 0.15 * math.sin(math.radians(-120.0))
    front = node['wheelbase'] / 2.0 + node['car_length'] / 2.0
    rear = node['wheelbase'] / 2.0 - node['car_length'] / 2.0
    assert rear <= x <= front and abs(y) <= node['car_width'] / 2.0  # oracle: closed form
    assert rv.apply_map_safety(scan, INPUT_STEERING, INPUT_SPEED, car) == (
        INPUT_STEERING, INPUT_SPEED, "none")


def test_escape_steering_is_clamped_to_the_steering_limit():
    car, lidar, node = _setup()
    # Open run (20, 30] deg, centre 25 deg: past this car's steering limit.
    scan = _flat_wall(_blank(lidar), lidar, _body_clear_wall(node), -30.0, 19.75)
    limit = node['max_steering_angle']
    assert math.radians(25.0) > limit  # premise: the clamp is what binds
    steering, speed, result = rv.apply_map_safety(scan, INPUT_STEERING, INPUT_SPEED, car)
    assert result == "escape"
    assert steering == pytest.approx(limit, abs=1e-9)  # rad, the car's steering limit
    assert speed == node['emergency_escape_speed']  # m/s


@pytest.mark.parametrize("value", [np.nan, 0.0, -1.0])
def test_an_unreadable_open_side_is_not_an_opening(value):
    car, lidar, node = _setup()
    scan = _escape_scene(lidar, _body_clear_wall(node))
    scan[_beam(0.25, lidar):_beam(30.0, lidar) + 1] = value
    assert rv.apply_map_safety(scan, INPUT_STEERING, INPUT_SPEED, car) == (
        INPUT_STEERING, 0.0, "stop")


@pytest.mark.parametrize("value", [np.inf, np.nan, 0.0, -1.0])
def test_a_scan_with_no_valid_return_passes_through(value):
    car, lidar, _node = _setup()
    scan = np.full(lidar.num_beams, value)
    assert rv.apply_map_safety(scan, INPUT_STEERING, INPUT_SPEED, car) == (
        INPUT_STEERING, INPUT_SPEED, "none")


def test_run_pure_solo_counts_escape_steps_apart_from_stop_steps(monkeypatch):
    import eval_gap

    car = selected_car(note=False)
    calls = []

    def always_escape(scan, steering, speed, car_id=None, *, applied_steering=None):
        calls.append(1)
        return steering, speed, "escape"

    monkeypatch.setattr(rv, "apply_map_safety", always_escape)
    result = rv.run_pure_solo(eval_gap.track_path("indoor_wide"), 12345, 2.0,
                              car_id=car, safety="map")
    # Every control step is an escape, and none of them is a stop.
    assert len(calls) == result["steps"]
    assert result["escape_steps"] == result["steps"]
    assert result["stop_steps"] == 0
