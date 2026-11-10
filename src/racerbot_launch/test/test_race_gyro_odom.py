"""Inspect real launch actions without starting nodes.

Oracle: docs/operations.md saved-map race odometry contract. Collected by
python3 -m pytest src/racerbot_launch/test/ (through rb in a worktree).
"""
import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
import pytest
import yaml


def _load(name):
    path = Path(__file__).resolve().parents[1] / 'launch' / f'{name}_launch.py'
    spec = importlib.util.spec_from_file_location(f'gyro_test_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def race(tmp_path, monkeypatch):
    module = _load('race')
    # Filesystem boundary: CPU PF config, independent of host CUDA support.
    pf = tmp_path / 'particle_filter'
    (pf / 'config').mkdir(parents=True)
    (pf / 'config/localize.yaml').write_text(
        'particle_filter:\n  ros__parameters:\n    range_method: rm\n'
        '    odometry_topic: /odom\n')
    real_share = module.get_package_share_directory
    monkeypatch.setattr(module, 'get_package_share_directory',
                        lambda name: str(pf) if name == 'particle_filter' else real_share(name))
    (tmp_path / 'map.yaml').touch()
    (tmp_path / 'raceline_profiled.csv').touch()
    return module, tmp_path, pf


def _expand(module, **overrides):
    context = LaunchContext()
    context.launch_configurations.update(overrides)
    actions = []
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
        elif isinstance(action, OpaqueFunction):
            actions.extend(action.execute(context))
        else:
            actions.append(action)
    return actions, context


def _includes(actions, context):
    return {Path(perform_substitutions(
                context, a.launch_description_source._LaunchDescriptionSource__location)).name:
            dict(a.launch_arguments)
            for a in actions if isinstance(a, IncludeLaunchDescription)}


def _pf_topic(race, includes):
    _, run_dir, pf_share = race
    localize = _load('localize_run')
    args = includes['localize_run_launch.py']
    assert args['map_yaml'] == str(run_dir / 'map.yaml')
    assert args['standalone'] == 'true'
    actions, context = _expand(
        localize, localize_config=str(pf_share / 'config/localize.yaml'), **args)
    timer, = [a for a in actions if isinstance(a, TimerAction)]
    node, = [a for a in timer.actions if isinstance(a, Node)]
    # oracle: spec -- override must follow the config, so the file cannot win.
    config, override = node._Node__parameters
    assert str(config.evaluate(context)) == str(pf_share / 'config/localize.yaml')
    key, value = next(iter(override.items()))
    assert perform_substitutions(context, key) == 'odometry_topic'
    return yaml.safe_load(perform_substitutions(context, value))


@pytest.mark.parametrize('seed_start', ['true', 'false'])
def test_saved_race_defaults_to_gyro_without_parenting_odom(race, seed_start):
    module, run_dir, _ = race
    actions, context = _expand(module, run_dir=str(run_dir), seed_start=seed_start,
                               imu_hardware='false')
    includes = _includes(actions, context)
    # oracle: spec docs/operations.md -- gyro PF and IMU stack, no racing odom TF.
    assert _pf_topic(race, includes) == '/odom/imu'
    assert includes['vehicle_state_launch.py'] == {
        'hardware': 'false', 'publish_odom_tf': 'false'}


def test_imu_false_restores_steering_odometry_and_no_imu_stack(race):
    module, run_dir, _ = race
    actions, context = _expand(module, run_dir=str(run_dir), imu='false')
    includes = _includes(actions, context)
    assert _pf_topic(race, includes) == '/odom'
    assert 'vehicle_state_launch.py' not in includes


def test_pf_false_keeps_imu_but_restores_steering_odometry(race):
    module, run_dir, _ = race
    actions, context = _expand(module, run_dir=str(run_dir), pf_gyro_odom='false',
                               imu_hardware='true')
    includes = _includes(actions, context)
    assert _pf_topic(race, includes) == '/odom'
    assert includes['vehicle_state_launch.py'] == {
        'hardware': 'true', 'publish_odom_tf': 'false'}


@pytest.mark.parametrize('saved_run', [True, False])
def test_gyro_without_imu_refuses_and_names_both_flags(race, saved_run):
    module, run_dir, _ = race
    with pytest.raises(RuntimeError, match='pf_gyro_odom:=true needs imu:=true'):
        _expand(module, run_dir=str(run_dir) if saved_run else '',
                pf_gyro_odom='true', imu='false')


@pytest.mark.parametrize('imu', ['true', 'false'])
def test_no_run_defaults_to_upstream_steering_odometry(race, imu):
    module, _, _ = race
    actions, context = _expand(module, imu=imu, imu_hardware='false')
    includes = _includes(actions, context)
    assert includes['localize_launch.py'] == {}
    assert 'localize_run_launch.py' not in includes
    assert ('vehicle_state_launch.py' in includes) == (imu == 'true')
    if imu == 'true':
        assert includes['vehicle_state_launch.py']['publish_odom_tf'] == 'false'


def test_no_run_explicit_gyro_refuses_with_recovery_arguments(race):
    module, _, _ = race
    with pytest.raises(RuntimeError, match=r'pf_gyro_odom:=true needs run_dir:=.*'
                       r'Pass run_dir:=.*or pf_gyro_odom:=false'):
        _expand(module, pf_gyro_odom='true')


@pytest.mark.parametrize('imu,gyro,run_dir,topic', [
    ('true', '', '/saved', '/odom/imu'),
    ('false', '', '/saved', '/odom'),
    ('true', 'false', '/saved', '/odom'),
    ('true', '', '', '/odom'),
    (' 1 ', ' TRUE ', '/saved', '/odom/imu'),
    ('0', '0', '', '/odom'),
])
def test_pure_odometry_decision(imu, gyro, run_dir, topic):
    # oracle: spec docs/operations.md; pure helper needs no ROS context or node.
    assert _load('race').pf_odometry_topic(imu, gyro, run_dir) == topic


@pytest.mark.parametrize('flag,value', [
    ('imu', ''), ('imu', 'yes'), ('imu', '2'),
    ('pf_gyro_odom', 'yes'), ('pf_gyro_odom', 'NaN'), ('pf_gyro_odom', 'false!'),
])
def test_invalid_boolean_refuses_at_launch(race, flag, value):
    module, run_dir, _ = race
    with pytest.raises(RuntimeError, match=rf"{flag}:=.*expected 'true' or 'false'"):
        _expand(module, run_dir=str(run_dir), **{flag: value})


@pytest.mark.parametrize('simulated', [True, False])
def test_auto_hardware_selection_reaches_vehicle_state(race, monkeypatch, simulated):
    import subprocess
    from types import SimpleNamespace

    def probe(command, **kwargs):
        # Process boundary only: exercise the real select_imu_hardware logic.
        assert command[:2] == ['pgrep', '-f']
        return SimpleNamespace(returncode=0 if simulated else 1)

    monkeypatch.setattr(subprocess, 'run', probe)
    module, run_dir, _ = race
    actions, context = _expand(module, run_dir=str(run_dir))
    includes = _includes(actions, context)
    assert includes['vehicle_state_launch.py'] == {
        'hardware': 'false' if simulated else 'true', 'publish_odom_tf': 'false'}
    assert _pf_topic(race, includes) == '/odom/imu'


def test_localize_run_defaults_to_steering_odom_for_mid_run_localization(race):
    _, run_dir, pf_share = race
    actions, context = _expand(
        _load('localize_run'), map_yaml=str(run_dir / 'map.yaml'),
        localize_config=str(pf_share / 'config/localize.yaml'))
    node, = [a for a in actions if isinstance(a, Node) and a.node_package == 'particle_filter']
    _, override = node._Node__parameters
    key, value = next(iter(override.items()))
    # oracle: spec -- auto_map_race's mid-run localizer retains /odom by default.
    assert perform_substitutions(context, key) == 'odometry_topic'
    assert yaml.safe_load(perform_substitutions(context, value)) == '/odom'
