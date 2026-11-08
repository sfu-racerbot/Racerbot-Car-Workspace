"""The commands printed at the end of a mapping-only run (run_commands.py).

Pure Python -- runs unsourced:

    python3 -m pytest src/pure_pursuit/test/test_run_commands.py -v

Oracles: the launch arguments the commands use are checked against what
race_launch.py actually declares, and the steering laws against
tracking_laws.LAW_NAMES, so a renamed argument or law fails here instead of
on the car.
"""
import ast
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pure_pursuit import run_commands  # noqa: E402
from pure_pursuit.tracking_laws import LAW_NAMES  # noqa: E402

RUN = Path('/home/someone/.ros/racerbot_auto/20261022-101500')
RACE_LAUNCH = (Path(__file__).resolve().parents[2]
               / 'racerbot_launch' / 'launch' / 'race_launch.py')


def _declared_launch_arguments(path):
    """Names passed to DeclareLaunchArgument(...) in a launch file."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'DeclareLaunchArgument'
                and node.args and isinstance(node.args[0], ast.Constant)):
            names.add(node.args[0].value)
    return names


def test_every_steering_law_is_offered_and_every_offered_law_exists():
    assert [law for law, _ in run_commands.STEERING_LAWS] == list(LAW_NAMES)


def test_race_command_uses_only_arguments_race_launch_declares():
    declared = _declared_launch_arguments(RACE_LAUNCH)
    for law in LAW_NAMES:
        command = run_commands.race_command(RUN, law)
        assert command.startswith('ros2 launch racerbot_launch race_launch.py ')
        args = command.split()[4:]
        names = {a.split(':=')[0] for a in args}
        assert all(':=' in a for a in args), command
        assert names <= declared, f'{names - declared} not declared in race_launch.py'


def test_race_command_names_the_run_and_the_law():
    assert run_commands.race_command(RUN, 'pure_pursuit') == (
        f'ros2 launch racerbot_launch race_launch.py run_dir:={RUN}')
    assert run_commands.race_command(RUN, 'mpc') == (
        f'ros2 launch racerbot_launch race_launch.py run_dir:={RUN} steering_law:=mpc')


def test_text_has_one_command_per_law_bringup_and_the_start_anywhere_rule():
    text = run_commands.commands_text(RUN)
    for law in LAW_NAMES:
        assert f'  {run_commands.race_command(RUN, law)}\n' in text
    assert '  ros2 launch f1tenth_stack bringup_launch.py\n' in text
    # tech lead, 2026-11-07: put the car anywhere on the course
    assert 'anywhere on the course' in text
    assert 'facing the racing direction' in text
    assert 'start:=start_pose' in text
    assert str(RUN / 'commands.txt') in text
    assert '!!' not in text


def test_missing_raceline_is_called_out_only_when_missing():
    text = run_commands.commands_text(RUN, raceline_ok=False)
    assert 'No racing line was saved' in text
    assert 'raceline_profiled.csv' not in text
    assert 'map did NOT save' not in text


def test_missing_map_is_called_out_only_when_missing():
    text = run_commands.commands_text(RUN, map_ok=False)
    assert 'The map did NOT save' in text
    assert 'map.yaml' not in text
    assert 'No racing line' not in text


def test_start_pose_round_trips(tmp_path):
    run_commands.write_start_pose(tmp_path, (1.23456, -7.5, 3.0))
    assert run_commands.read_start_pose(tmp_path) == (1.2346, -7.5, 3.0)


def test_missing_or_damaged_start_pose_reads_as_none(tmp_path):
    assert run_commands.read_start_pose(tmp_path) is None
    path = tmp_path / run_commands.START_POSE_FILE
    for text in ('x: 1.0\ny: 2.0\n', 'x: 1\ny: two\nyaw: 0\n', 'x: nan\ny: 0\nyaw: 0\n',
                 'x: 1\ny: 2\nyaw: inf\n', ''):
        path.write_text(text)
        assert run_commands.read_start_pose(tmp_path) is None, text


def test_missing_start_pose_is_called_out_only_when_missing():
    assert 'start pose was not recorded' in run_commands.commands_text(RUN, start_ok=False)
    text = run_commands.commands_text(RUN)
    assert 'start_pose.yaml' in text and 'not recorded' not in text


def test_seeder_parameters_anywhere_uses_run_paths_and_offsets(tmp_path):
    # Interface spec: the run files, supplied laser offsets and 0.30 m radius.
    assert run_commands.seeder_parameters(tmp_path, 'anywhere', 0.26, -0.02) == {
        'mode': 'anywhere', 'map_yaml': str(tmp_path / 'map.yaml'),
        'waypoints_file': str(tmp_path / 'raceline_profiled.csv'),
        'laser_offset_x': 0.26, 'laser_offset_y': -0.02, 'confirm_radius': 0.30}


def test_seeder_parameters_start_pose_reads_saved_pose(tmp_path):
    # Interface spec: use the recorded pose and the supplied laser offsets.
    run_commands.write_start_pose(tmp_path, (1.25, -2.5, 0.75))
    # Spec: start_pose keeps the node's default confirmation radius.
    assert 'confirm_radius' not in run_commands.seeder_parameters(tmp_path, 'start_pose', 0.26, 0.0)
    assert run_commands.seeder_parameters(tmp_path, 'start_pose', 0.26, 0.0) == {
        'mode': 'start_pose', 'x': 1.25, 'y': -2.5, 'yaw': 0.75,
        'laser_offset_x': 0.26, 'laser_offset_y': 0.0}


def test_seeder_parameters_start_pose_without_file_uses_origin(tmp_path):
    # Interface spec: absent start pose falls back to (0, 0, 0).
    assert run_commands.seeder_parameters(tmp_path, 'start_pose', 0.26, 0.0) == {
        'mode': 'start_pose', 'x': 0.0, 'y': 0.0, 'yaw': 0.0,
        'laser_offset_x': 0.26, 'laser_offset_y': 0.0}


def test_seeder_parameters_rejects_unknown_start(tmp_path):
    # Interface spec: other start modes raise ValueError.
    with pytest.raises(ValueError, match='start must be one of'):
        run_commands.seeder_parameters(tmp_path, 'elsewhere', 0.26, 0.0)
def test_floor_test_precedes_default_speed_commands(tmp_path):
    # oracle: spec, a capped floor test is the first race command for either car.
    from pure_pursuit.run_commands import commands_text, race_command
    text = commands_text(tmp_path)
    floor = race_command(tmp_path) + ' profile:=raceline_floor_test.csv'
    commands = [line.strip() for line in text.splitlines() if line.strip().startswith('ros2 launch racerbot_launch race_launch.py')]
    assert commands[0] == floor
    assert commands[1] == race_command(tmp_path)
    assert 'wheels off the ground' in text
