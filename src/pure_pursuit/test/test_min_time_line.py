"""ROS-free minimum-time line tests, collected by the pure_pursuit suite."""

import numpy as np

from pure_pursuit.min_time_line import CarPhysics, cone_clearance, speed_profile


def test_speed_delay_respects_future_planned_speeds():
    # Stadium: 6 m straights and 2 m radius semicircles, counter-clockwise.
    radius, straight, spacing = 2.0, 6.0, 0.1
    x = np.arange(0.0, straight, spacing)
    angle = np.linspace(-np.pi / 2, np.pi / 2, 64, endpoint=False)
    xy = np.vstack([
        np.column_stack([x, np.full_like(x, -radius)]),
        np.column_stack([straight + radius * np.cos(angle), radius * np.sin(angle)]),
        np.column_stack([straight - x, np.full_like(x, radius)]),
        np.column_stack([-radius * np.cos(angle), -radius * np.sin(angle)]),
    ])
    physics = CarPhysics(v_max=8.0, steering_delay=0.0, speed_delay=0.1)
    speed, _, seg = speed_profile(xy, physics)
    positions = np.concatenate(([0.0], np.cumsum(seg)))
    total = positions[-1]
    # oracle: invariant, min_time_line._respond_early docstring: every station
    # must be no faster than any station within speed * delay metres ahead.
    for i in range(len(speed)):
        distance = (positions[:-1] - positions[i]) % total
        future = speed[distance <= speed[i] * physics.speed_delay]
        assert speed[i] <= future.min() + 1e-10  # m/s, floating-point tolerance


def test_cone_clearance_parallel_wall():
    xy = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [3, -2], [0, -2]])
    wall_y = 0.5  # m
    step_floor = 0.01  # m
    field = lambda x, y: np.maximum(wall_y - y, 0.0)
    actual = cone_clearance(field, xy, laser_x=0.0, step_floor=step_floor)[1]
    # oracle: closed form; the +30 degree ray intersects y=wall_y at this range.
    expected = wall_y / np.sin(np.deg2rad(30.0))
    assert abs(actual - expected) <= step_floor  # m, documented ray-march accuracy
