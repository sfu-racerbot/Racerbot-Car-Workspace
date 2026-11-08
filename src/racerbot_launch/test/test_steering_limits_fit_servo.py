"""
Every steering limit the car ships with must be one the servo can deliver.

A steering command becomes a servo value in ackermann_to_vesc.cpp,
``servo = steering_angle_to_servo_gain * angle + steering_angle_to_servo_offset``,
and vesc_driver then clips that to [servo_min, servo_max] without a warning.
A limit past the clip is steering the car silently never gets, on one side
only. That happened: from 2026-10-10 to 2026-10-16 full stick (0.34 rad) and
the autonomy limit (0.26 rad) both asked for more right steering than
servo_max 0.85 allowed (docs/odom-calibration.md, docs/hardware-reference.md).

Oracle: the closed form above, recomputed from vesc.yaml, applied to the
limits in each config that sets one. Also an invariant: the race planner must
plan below the controller's limit, so the controller has steering left to
correct with on the tightest planned corner (auto_map_race.yaml).

The servo mapping is linear but this car's linkage is not
(docs/hardware-reference.md), so this proves only that the servo is not
clipped, not that the wheels reach the angle.

No ROS needed:

    python3 -m pytest src/racerbot_launch/test/ -v
"""
import os
import sys

import pytest
import yaml

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# optimize_raceline is pure Python (no rclpy), so its real argument parser is
# importable here without ROS.
sys.path.insert(0, os.path.join(_SRC, 'pure_pursuit'))
sys.path.insert(0, os.path.join(_SRC, 'racerbot_car'))
from pure_pursuit import optimize_raceline  # noqa: E402
from racerbot_car.identity import known_car_ids  # noqa: E402
from racerbot_car.layers import effective_config  # noqa: E402
from racerbot_car.paths import cars_dir  # noqa: E402
_STACK = os.path.join(_SRC, 'f1tenth_system', 'f1tenth_stack', 'config')
CARS = sorted(car for car in known_car_ids(cars_dir())
              if (cars_dir() / car / 'calibration' / 'vesc.yaml').is_file())


def _params(path, node):
    with open(path, encoding='utf-8') as handle:
        return yaml.safe_load(handle)[node]['ros__parameters']


def _servo_mapping(car):
    vesc = effective_config('vesc.yaml', '/**', car_id=car)
    return (vesc['steering_angle_to_servo_gain'],
            vesc['steering_angle_to_servo_offset'],
            vesc['servo_min'], vesc['servo_max'])


def clipped_sides(angle, gain, offset, servo_min, servo_max):
    """The sides ('left', 'right') on which +/-angle rad is clipped."""
    clipped = []
    for side, signed in (('left', angle), ('right', -angle)):
        servo = gain * signed + offset
        if not servo_min <= servo <= servo_max:
            clipped.append(side)
    return clipped


def _limits(car):
    joy = effective_config('joy_teleop.yaml', 'joy_teleop', car_id=car)
    assert joy['human_control.axis_mappings.drive-steering_angle.offset'] == 0.0
    # oracle: spec, the hand-run CLI defaults to the current car's planner.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('RACERBOT_CAR', car)
        optimizer_limit = optimize_raceline.build_arg_parser().get_default('max_steering_angle')
    return {
        # Full stick: the F710 axis reads +/-1.0.
        'joy_teleop.yaml full stick': abs(joy['human_control.axis_mappings.drive-steering_angle.scale']),
        'gap_follow.yaml max_steering_angle': effective_config(
            'gap_follow.yaml', 'gap_follow_node', car_id=car)['max_steering_angle'],
        'pure_pursuit.yaml max_steering_angle': effective_config(
            'pure_pursuit.yaml', 'pure_pursuit_node', car_id=car)['max_steering_angle'],
        'auto_map_race.yaml profile_max_steering_angle': effective_config(
            'auto_map_race.yaml', 'auto_map_race_node', car_id=car)['profile_max_steering_angle'],
        # Hand-run optimizations check feasibility against this unless the
        # flag is passed (docs/racing-autonomy.md).
        'optimize_raceline --max-steering-angle default':
            optimizer_limit,
    }


@pytest.mark.parametrize('car', CARS)
@pytest.mark.parametrize('name', ['joy_teleop.yaml full stick', 'gap_follow.yaml max_steering_angle',
                                  'pure_pursuit.yaml max_steering_angle',
                                  'auto_map_race.yaml profile_max_steering_angle',
                                  'optimize_raceline --max-steering-angle default'])
def test_shipped_limit_is_not_clipped_by_the_servo(name, car):
    angle = _limits(car)[name]
    assert angle > 0.0
    assert clipped_sides(angle, *_servo_mapping(car)) == [], (
        f'{name} = {angle} rad is clipped by servo_min/servo_max in vesc.yaml')


@pytest.mark.parametrize('car', CARS)
def test_planner_plans_below_the_controller_limit(car):
    limits = _limits(car)
    assert (limits['auto_map_race.yaml profile_max_steering_angle']
            < limits['pure_pursuit.yaml max_steering_angle'])


@pytest.mark.parametrize('car', CARS)
def test_hand_run_optimizer_checks_against_the_planners_limit(car):
    # The optimizer and auto_map_race both decide what "the rack cannot steer
    # this" means for a racing line; they must agree, or a line accepted by
    # one is rejected (or worse, accepted too tight) by the other. Until
    # 2026-10-24 the optimizer still defaulted to the pre-October 0.26 rad.
    limits = _limits(car)
    assert (limits['optimize_raceline --max-steering-angle default']
            == limits['auto_map_race.yaml profile_max_steering_angle'])


# The check itself, on known cases, so it is not only ever seen passing.
# Recorded state 2026-10-10 (docs/odom-calibration.md): gain -1.2135, offset
# 0.5904, clip [0.15, 0.85]; full right at 0.34 rad needs 0.5904 + 0.4126 =
# 1.003, and 0.26 rad needs 0.9059. Both past 0.85, on the right only.
@pytest.mark.parametrize('angle', [0.34, 0.26])
def test_check_catches_the_2026_10_10_right_side_clip(angle):
    assert clipped_sides(angle, -1.2135, 0.5904, 0.15, 0.85) == ['right']


def test_check_boundary_is_inclusive_and_one_step_past_fails():
    # gain -1, offset 0.5: angle 0.35 lands exactly on servo_max 0.85 (and
    # 0.15 on the left). 0.3501 is past both.
    assert clipped_sides(0.35, -1.0, 0.5, 0.15, 0.85) == []
    assert clipped_sides(0.3501, -1.0, 0.5, 0.15, 0.85) == ['left', 'right']


def test_check_catches_a_left_side_clip():
    # Offset low enough that the left end runs out first.
    assert clipped_sides(0.3, -1.0, 0.4, 0.15, 0.85) == ['left']
