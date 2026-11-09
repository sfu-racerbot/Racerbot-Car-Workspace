"""Launch-file checks for pure_pursuit_tech_launch.py.

Run with ROS sourced: python3 -m pytest src/pure_pursuit_tech/test/ -v
"""

import importlib.util
import math
from pathlib import Path

import pytest

LAUNCH = Path(__file__).resolve().parents[1] / 'launch' / 'pure_pursuit_tech_launch.py'


@pytest.fixture
def launch_module():
    spec = importlib.util.spec_from_file_location('pure_pursuit_tech_launch', LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('car, cap', [
    # cited spec: max_steering_angle in each car's
    # src/racerbot_car/cars/<car>/calibration/pure_pursuit.yaml
    ('rb1', 0.33),
    ('rb2', 0.349),
])
def test_car_limits_are_this_cars_calibration(launch_module, car, cap):
    # cited spec: wheelbase 0.36 m, shared src/pure_pursuit/config/pure_pursuit.yaml
    # (measured 2026-08-24, docs/hardware-reference.md)
    assert launch_module.car_limits(car) == (car, 0.36, cap)


@pytest.mark.parametrize('value', [0.33, 0.20, 1e-6])
def test_steering_at_or_below_the_cap_is_accepted(launch_module, value):
    launch_module.check_steering(value, 0.33, 'rb1')


@pytest.mark.parametrize('value', [0.3301, 0.4189, 0.0, -0.1, math.nan, math.inf])
def test_steering_above_the_cap_or_not_positive_is_refused(launch_module, value):
    # 0.4189 is pure_pursuit_a's default, above both cars' caps.
    with pytest.raises(RuntimeError, match='outside \\(0, 0.33\\] rad'):
        launch_module.check_steering(value, 0.33, 'rb1')


def test_safety_node_reads_this_cars_odometry(launch_module, monkeypatch):
    # racerbot_a's safety_node subscribes to the simulator's /ego_racecar/odom;
    # on the car its iTTC needs /odom or its speed stays 0 and it never brakes.
    monkeypatch.setattr(launch_module, 'car_limits', lambda: ('rb1', 0.36, 0.33))
    from launch import LaunchContext
    from launch.utilities import perform_substitutions
    from launch_ros.actions import Node

    description = launch_module.generate_launch_description()
    context = LaunchContext()
    for entity in description.entities:
        if not isinstance(entity, Node):
            entity.visit(context)       # declare arguments with their defaults
    nodes = [e for e in description.entities if isinstance(e, Node)]
    safety = [n for n in nodes if n.node_package == 'safety_node']
    assert len(safety) == 1
    remaps = {perform_substitutions(context, list(src)): perform_substitutions(context, list(dst))
              for src, dst in safety[0]._Node__remappings}
    assert remaps == {'/ego_racecar/odom': '/odom'}


def tech_node_parameters(launch_module, monkeypatch, **arguments):
    """Run the launch's node-building step with `arguments` and return the
    node's parameters, evaluated. car_limits/laser_offset are replaced with
    fixed values: the car profile files are tested in racerbot_launch."""
    monkeypatch.setattr(launch_module, 'car_limits', lambda car_id=None: ('rb1', 0.36, 0.33))
    monkeypatch.setattr(launch_module, 'laser_offset', lambda car_id: (0.26, 0.01))
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument, OpaqueFunction
    from launch_ros.actions import Node
    from launch_ros.utilities import evaluate_parameters

    description = launch_module.generate_launch_description()
    context = LaunchContext()
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.visit(context)
    context.launch_configurations.update({'waypoints_file': '/tmp/line.csv', **arguments})
    opaque = [e for e in description.entities if isinstance(e, OpaqueFunction)][0]
    nodes = [a for a in opaque.visit(context) if isinstance(a, Node)]
    assert len(nodes) == 1
    params = {}
    for entry in evaluate_parameters(context, nodes[0]._Node__parameters):
        params.update(entry)
    return params


def test_laser_offset_is_applied_only_with_the_particle_filter(launch_module, monkeypatch):
    # The PF reports the LiDAR's pose (src/pure_pursuit/config/pure_pursuit.yaml
    # pose_frame: laser); odom_topic's pose is already the rear axle's.
    with_pf = tech_node_parameters(launch_module, monkeypatch, use_pf='true')
    assert (with_pf['laser_offset_x'], with_pf['laser_offset_y']) == (0.26, 0.01)
    without = tech_node_parameters(launch_module, monkeypatch, use_pf='false')
    assert (without['laser_offset_x'], without['laser_offset_y']) == (0.0, 0.0)


def test_speed_and_goal_arguments_reach_the_node(launch_module, monkeypatch):
    params = tech_node_parameters(
        launch_module, monkeypatch, speed_mode='profile', max_lateral_accel='3.5', max_accel='2.0',
        max_brake='2.5', laps='3', goal_tolerance='0.4', max_path_error='1.2',
        medium_angle_speed='0.8')
    assert params['medium_angle_speed'] == 0.8
    assert params['speed_mode'] == 'profile'
    assert params['laps'] == 3 and isinstance(params['laps'], int)
    assert (params['max_lateral_accel'], params['max_accel'], params['max_brake'],
            params['goal_tolerance'], params['max_path_error']) == (3.5, 2.0, 2.5, 0.4, 1.2)


def test_speed_mode_defaults_to_pure_pursuit_a_rule(launch_module, monkeypatch):
    params = tech_node_parameters(launch_module, monkeypatch)
    assert params['speed_mode'] == 'steering'
    assert params['medium_angle_speed'] == -1.0   # halfway: the old straight line
    assert params['laps'] == 0
