"""ROS-free minimum-time line tests, collected by the pure_pursuit suite."""

import numpy as np

from pure_pursuit.min_time_line import CarPhysics, speed_profile


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
