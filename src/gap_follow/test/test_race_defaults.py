"""The race defaults of 2026-10-27, checked against the table that justifies them.

Spec: docs/gap-follow.md#race-defaults-2026-10-27 -- the simulator table and the
list of what changed. The values are read from the YAML the car loads
(config/gap_follow.yaml). Pure YAML, no rclpy, so it runs unsourced:

    python3 -m pytest src/gap_follow/test/test_race_defaults.py -v

It also runs in the sourced gap_follow suite, unchanged.
"""
import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'racerbot_car'))
from racerbot_car.identity import known_car_ids  # noqa: E402
from racerbot_car.layers import effective_config  # noqa: E402
from racerbot_car.paths import cars_dir  # noqa: E402

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config',
                      'gap_follow.yaml')
with open(CONFIG, encoding='utf-8') as handle:
    P = yaml.safe_load(handle)['gap_follow_node']['ros__parameters']

OLD_TOP_SPEED = 3.5  # m/s, the top speed before the 2026-10-27 race defaults
OLD_SPEEDS = {'min_speed': 1.0, 'corner_speed': 1.5, 'corner_speed_wide': 1.9,
              'narrow_corridor_speed': 2.5}  # m/s, the same before


def test_top_speed_is_the_launch_ceiling():
    # oracle: spec -- the table's max_speed row (3.5 -> 5.0), and the launch's
    # ceiling (gap_follow/launch_speed.py MAX_SPEED_CEILING, requested 2026-10-26).
    assert P['max_speed'] == pytest.approx(5.0, abs=1e-9)  # m/s


@pytest.mark.parametrize('key', sorted(OLD_SPEEDS))
def test_speeds_tied_to_the_top_speed_scale_with_it(key):
    # oracle: closed form -- the old value x 5.0/3.5, shipped to 1 mm.
    assert P[key] == pytest.approx(OLD_SPEEDS[key] * 5.0 / OLD_TOP_SPEED, abs=5e-4)  # m/s


def test_the_race_switches_are_set_as_the_table_says():
    # oracle: spec -- the table in docs/gap-follow.md#race-defaults-2026-10-27.
    assert P['enable_steering_damping'] is True
    assert P['steering_damping_gain'] == pytest.approx(0.10, abs=1e-9)  # s
    assert P['enable_corner_setup'] is False
    assert P['max_acceleration'] == pytest.approx(5.0, abs=1e-9)  # m/s^2
    # Ships off: the simulator put it slower on car 1's course.
    assert P['enable_adaptive_grip'] is False


@pytest.mark.parametrize('car', sorted(car for car in known_car_ids(cars_dir())
                                      if (cars_dir() / car / 'calibration/gap_follow.yaml').is_file()))
def test_the_safety_values_the_race_defaults_left_alone(car):
    # oracle: spec -- docs/gap-follow.md#race-defaults-2026-10-27 lists none of
    # these as changed. The deadman is the workspace policy in
    # docs/architecture.md, not a tuning knob.
    assert P['enable_deadman'] is True
    assert P['enable_ttc'] is True
    assert P['ttc_threshold_sec'] == pytest.approx(0.30, abs=1e-9)  # s
    assert P['max_braking_decel'] == pytest.approx(3.0, abs=1e-9)  # m/s^2
    assert P['emergency_stop_clearance'] == pytest.approx(0.0, abs=1e-9)  # m
    assert P['forward_stop_clearance'] == pytest.approx(0.20, abs=1e-9)  # m
    # docs/gap-follow.md#race-defaults-2026-10-27, car 1.
    # Car 2: 0.42 since 2026-11-07 (0.349 before), inside its 2026-10-16 measured
    # circles: docs/hardware-reference.md, car 2's steering.
    expected = {'rb1': 0.33, 'rb2': 0.42}
    assert effective_config('gap_follow.yaml', 'gap_follow_node', car_id=car)['max_steering_angle'] == \
        pytest.approx(expected[car], abs=1e-9)  # rad


def test_the_course_defaults_of_2026_10_30():
    # oracle: spec -- docs/gap-follow.md#speed-on-the-real-course-2026-10-30.
    assert P['aim_depth_bias'] == pytest.approx(0.0, abs=1e-9)
    assert P['aim_depth_bias_fallback'] == pytest.approx(1.0, abs=1e-9)
    assert P['max_lateral_accel'] == pytest.approx(2.5, abs=1e-9)  # m/s^2
    assert P['swept_clearance_max_length'] == pytest.approx(5.0, abs=1e-9)  # m
    assert P['opponent_follow_floor'] == pytest.approx(0.5, abs=1e-9)  # m/s


def test_the_braking_lookahead_covers_a_measured_stop_from_top_speed():
    # oracle: recorded measurement, run 20261030-175124: LB released at 5.41 m/s on
    # a straight, 3.39 m by the wall ahead to 0.29 m/s. A shorter lookahead than the
    # real stop means the braking ceiling cannot act in time at top speed.
    assert P['max_speed'] <= 5.41
    assert P['swept_clearance_max_length'] >= 3.39 + P['forward_stop_clearance']
