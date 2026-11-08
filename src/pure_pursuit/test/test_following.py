"""ROS-free following checks; collected by the complete pure_pursuit suite."""
import math
from collections import deque

import numpy as np
import pytest

from pure_pursuit import following


@pytest.mark.parametrize('speed', [0.0, 0.5, 1.0, 2.0, 4.0])
def test_envelope_inverts_delayed_physical_stop(speed):
    # oracle: closed form, coasting delay plus constant 3 m/s² braking;
    # docs/racing-autonomy.md identifies measured physical vs command limits.
    gap = 0.65 + speed * 0.45 + speed ** 2 / (2 * 3.0)
    actual = following.speed_ceiling(gap, 2.0, 0.0)
    assert actual == pytest.approx(speed, abs=1e-12)  # m/s
    if speed > 0:
        assert following.speed_ceiling(gap - 0.01, 2.0, 0.0) < speed
        assert following.speed_ceiling(gap + 0.01, 2.0, 0.0) > speed


def test_lost_or_braking_opponent_never_creates_space():
    # oracle: invariant; even an instantly stopped leader must be safe.
    fresh = following.speed_ceiling(3, 0, 0)
    assert fresh > 0
    assert following.speed_ceiling(3, 5, 0) == fresh
    lost = following.speed_ceiling(3, 2, .4)
    assert 0 < lost < fresh
    assert following.speed_ceiling(3, 3, .4) < lost
    assert following.speed_ceiling(3, 2, 2) == 0
    assert following.speed_ceiling(.65, 0, 0) == 0
    assert following.speed_ceiling(-1, 0, 0) == 0


def test_standstill_threshold_boundary():
    # oracle: closed form with b=1, delay=reserve=0: v=sqrt(2*gap).
    options = dict(brake=1, delay=0, reserve=0, stop_speed=1)
    assert following.speed_ceiling(.5, 0, 0, **options) == 1
    assert following.speed_ceiling(.5 - 1e-9, 0, 0, **options) == 0
    assert following.speed_ceiling(.5 + 1e-9, 0, 0, **options) > 1


@pytest.mark.parametrize('leader_speed', [0.0, 1.0, 2.0, 4.0])
def test_confident_speed_credit_is_only_the_leaders_braking_distance(leader_speed):
    ego_speed, brake, delay, reserve = leader_speed + 1, 3, .45, .65
    # oracle: closed form; ego delayed stop minus leader immediate stop.
    gap = reserve + ego_speed * delay + ego_speed ** 2 / (2 * brake) - leader_speed ** 2 / (2 * brake)
    assert following.speed_ceiling(gap, leader_speed, 0, speed_credit=leader_speed) == pytest.approx(
        ego_speed, abs=1e-12)  # m/s
    assert following.speed_ceiling(gap, leader_speed, 0) <= ego_speed
    assert following.speed_ceiling(gap, leader_speed, 0, speed_credit=-1) == 0
    assert following.speed_ceiling(gap, leader_speed, 0, speed_credit=math.nan) == 0
    assert following.speed_ceiling(gap, leader_speed, 0, opponent_brake=0) == 0


def test_delayed_speed_loop_stops_before_hard_braking_leader():
    # oracle: invariant against an independently integrated measured plant:
    # 0.10s transport, first-order gain 7/s, +4/-3 m/s² (6 Nov floor).
    # The leader brakes at 8 m/s², harder than the fitted car, then stops.
    gap, ego_speed, leader_speed = 6.0, 4.0, 2.0
    commands = deque([4.0] * 4)
    minimum = gap
    for tick in range(400):
        cap = following.speed_ceiling(gap, leader_speed, 0.0)
        commands.append(min(5.0, cap))
        target = commands.popleft()
        for substep in range(5):
            now = tick * .025 + substep * .005
            ego_speed += .005 * max(-3, min(4, 7 * (target - ego_speed)))
            if now >= 1.0:
                leader_speed = max(0.0, leader_speed - 8 * .005)
            gap += .005 * (leader_speed - ego_speed)
            minimum = min(minimum, gap)
    assert minimum >= .65 - .001  # 1 mm integration tolerance, body gap
    assert ego_speed < .001  # m/s; must actually stop rather than just avoid collision briefly


@pytest.mark.parametrize('key,value', [
    ('gap', math.nan), ('gap', math.inf), ('opponent_speed', math.nan),
    ('age', math.inf), ('age', -1), ('brake', 0), ('brake', -1),
    ('delay', -1), ('reserve', -1), ('speed_uncertainty', math.nan), ('stop_speed', -1)])
def test_invalid_envelope_inputs_fail_closed(key, value):
    args = dict(gap=4, opponent_speed=1, age=0)
    args[key] = value
    # oracle: invariant; invalid safety inputs cannot authorise motion.
    assert following.speed_ceiling(**args) == 0


def straight():
    xy = np.column_stack((np.arange(0, 10, .1), np.zeros(100)))
    return xy, np.arange(100) * .1, np.full(100, .1)


def test_continuous_projection_and_frame_invariance():
    xy, cumulative, _ = straight()
    # oracle: closed form; x=.237 is .237 m along a horizontal line.
    assert following.project_arc(xy, cumulative, [.237, .4], 2) == pytest.approx(.237, abs=1e-12)
    theta = .71
    rotation = np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])
    moved = xy @ rotation.T + [3, -5]
    point = np.array([.237, .4]) @ rotation.T + [3, -5]
    assert following.project_arc(moved, cumulative, point, 2) == pytest.approx(.237, abs=1e-12)
    assert following.project_arc(xy, cumulative, [-1, 0], 0) == 0
    assert math.isnan(following.project_arc(xy, cumulative, [math.nan, 0], 0))
    assert math.isnan(following.project_arc(np.empty((0, 2)), [], [0, 0], 0))
    assert math.isnan(following.project_arc(np.zeros((1, 2)), [0], [0, 0], 0))
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    assert following.project_arc(square, [0, 1, 2, 3], [0, .25], 0) == pytest.approx(3.75, abs=1e-12)


def test_visible_corridor_stops_before_obstacle_and_is_frame_invariant():
    xy, _, seg = straight()
    angle_min, increment = -3 * math.pi / 4, math.pi / 720
    angles = angle_min + np.arange(1081) * increment
    clear = np.full(1081, 10.0)
    args = (angle_min, increment, .26, 0, .255)
    free = following.visible_arc(xy, seg, 0, [0, 0, 0], clear, *args)
    # oracle: invariant; empty road permits >4 m, a transverse car at x=2
    # cannot permit even 2 m. A plane intersection supplies actual ranges.
    assert free > 4.5
    blocked = clear.copy()
    mask = np.abs(angles) < math.atan(.4 / (2 - .26))
    blocked[mask] = (2 - .26) / np.cos(angles[mask])
    bound = following.visible_arc(xy, seg, 0, [0, 0, 0], blocked, *args)
    assert 1.5 < bound < 2
    theta = -.93
    rotation = np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])
    transformed = xy @ rotation.T + [7, -2]
    assert following.visible_arc(transformed, seg, 0, [7, -2, theta], blocked, *args) == pytest.approx(bound, abs=1e-12)


@pytest.mark.parametrize('side', [-1, 1])
def test_near_front_flank_return_cannot_be_skipped_as_ego_body(side):
    # The shipped line spacing is .15m. The first checked slice is x=.75;
    # place a return just before it, inside the inflated corridor but beyond
    # ego's .47m front extent (docs/hardware-reference.md body geometry).
    stations = np.arange(0, 10, .15)
    xy = np.column_stack((stations, np.zeros(len(stations))))
    seg = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    angle_min, increment = -3 * math.pi / 4, math.pi / 720
    angles = angle_min + np.arange(1081) * increment
    laser_x, half_width = .26, .255
    obstacle_x, obstacle_y = .735, side * .245
    bearing = math.atan2(obstacle_y, obstacle_x - laser_x)
    beam = int(np.argmin(np.abs(angles - bearing)))
    distance = (obstacle_x - laser_x) / math.cos(angles[beam])
    return_y = distance * math.sin(angles[beam])
    # oracle: geometry. This is forward of the body and inside the corridor,
    # not a self-return. It blocks the x=.75 slice before any free arc can
    # be certified. The x=.90 slice's narrower angular interval misses it:
    # about 21.7 degrees versus the return's 27.25 degrees. Thus checking
    # only later slices cannot substitute for inspecting the nearer one.
    assert (.58 + .36) / 2 < obstacle_x < stations[5]
    assert abs(return_y) < half_width
    assert abs(angles[beam]) < math.atan2(half_width, stations[5] - laser_x)
    assert abs(angles[beam]) > math.atan2(half_width, stations[6] - laser_x) + increment
    scan = np.full(1081, 10.0)
    args = (angle_min, increment, laser_x, 0, half_width)
    assert following.visible_arc(xy, seg, 0, [0, 0, 0], scan, *args) > 4.5
    scan[beam] = distance
    assert following.visible_arc(xy, seg, 0, [0, 0, 0], scan, *args) == 0.0


@pytest.mark.parametrize('kind', ['empty', 'single', 'nan', 'inf', 'zero', 'negative', 'short'])
def test_unknown_corridor_is_not_observed_free_space(kind):
    xy, _, seg = straight()
    ranges = np.full(1081, 10.0)
    if kind == 'empty':
        xy = np.empty((0, 2))
    elif kind == 'single':
        xy = np.zeros((1, 2))
    elif kind == 'short':
        ranges = [10]
    else:
        ranges[:] = {'nan': math.nan, 'inf': math.inf, 'zero': 0, 'negative': -1}[kind]
    # oracle: invariant; unknown/invalid beams and absent paths cannot clear a memory.
    assert following.visible_arc(xy, seg, 0, [0, 0, 0], ranges,
                                 -3 * math.pi / 4, math.pi / 720, .26, 0, .255) == 0
