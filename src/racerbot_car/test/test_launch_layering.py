"""Launch captures exercise car selection and parameter precedence without starting nodes."""
import ast
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip('launch', reason='ROS launch is needed; source the workspace to run wiring tests')

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/racerbot_car'))
sys.path.insert(0, str(ROOT / 'src/gap_follow'))
from racerbot_car import hardware, identity
from gap_follow.launch_speed import speed_cap_overrides, yaml_parameters
from gap_follow.speed_overrides import mapping_speed_overrides

spec = importlib.util.spec_from_file_location('launch_params', ROOT / 'tools/dev/launch_params.py')
launch_params = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch_params)

CARS = ROOT / 'src/racerbot_car/cars'
SHARED = ROOT / 'src/gap_follow/config/gap_follow.yaml'
SPLIT = {f'{name}.yaml' for name in ('vesc', 'joy_teleop', 'sensors', 'gap_follow',
                                      'pure_pursuit', 'neural_racer', 'odom_calibration')}
EXCLUDED = {'racerbot_a', 'racerbot_b', 'web_dashboards', 'realsense-ros', 'particle_filter'}


@pytest.fixture
def car(monkeypatch, tmp_path):
    monkeypatch.setenv('RACERBOT_CAR', 'rb1')
    monkeypatch.setenv('RACERBOT_CARS_DIR', str(CARS))
    monkeypatch.setattr(identity, 'SERIAL_PATH', str(tmp_path / 'missing-serial'))
    monkeypatch.setattr(hardware, 'USB_DEVICES_ROOT', tmp_path / 'usb')


def capture(package, filename, args=()):
    path = ROOT / 'src' / package / 'launch' / filename
    return launch_params.capture(str(path), list(args), str(ROOT))


def node(result, name):
    return next(item for item in result['nodes'] if item['name'] == name)


def test_static_every_launch_naming_a_split_config_calls_layered_or_car_params():
    paths = list((ROOT / 'src').glob('*/launch/*.py'))
    paths += list((ROOT / 'src').glob('*/*/launch/*.py'))
    missing = []
    for path in paths:
        if path.relative_to(ROOT / 'src').parts[0] in EXCLUDED:
            continue
        source = path.read_text()
        strings = {n.value for n in ast.walk(ast.parse(source))
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        # effective_config() is racerbot_car.layers' layered read (it calls layered() itself).
        if strings & SPLIT and not any(c in source for c in ('layered(', 'car_params(', 'effective_config(')):
            missing.append(str(path.relative_to(ROOT)))
    assert not missing, missing


def test_gap_follow_node_gets_rb1_overlay(car):
    record = node(capture('gap_follow', 'gap_follow_launch.py'), '/gap_follow_node')
    files = record['param_files']
    shared = next(i for i, value in enumerate(files) if 'gap_follow/config/gap_follow.yaml' in value)
    overlay = next(i for i, value in enumerate(files) if 'rb1/calibration/gap_follow.yaml' in value)
    assert shared < overlay
    # Oracle: car 1's 2026-10-21 circles, docs/hardware-reference.md.
    assert record['params']['max_steering_angle'] == 0.33


def test_gap_follow_launch_speed_reads_effective_values(car, monkeypatch, tmp_path):
    cars = tmp_path / 'cars'
    shutil.copytree(CARS, cars)
    tuning = cars / 'rb1/tuning/gap_follow.yaml'
    tuning.parent.mkdir(exist_ok=True)
    tuning.write_text('gap_follow_node:\n  ros__parameters:\n    # why: fixture: car-specific corner speed\n    corner_speed: 2.6\n')
    monkeypatch.setenv('RACERBOT_CARS_DIR', str(cars))
    shared = yaml_parameters(SHARED)
    effective = {**shared, 'corner_speed': 2.6}
    # Oracle: launch_speed.speed_cap_overrides applied to effective YAML values.
    overrides = speed_cap_overrides(effective, '2.5')
    assert speed_cap_overrides(shared, '2.5') != overrides
    expected = {**effective, **overrides}
    params = node(capture('gap_follow', 'gap_follow_launch.py', ['max_speed:=2.5']),
                  '/gap_follow_node')['params']
    for key in ('max_speed', 'corner_speed', 'corner_speed_wide', 'min_speed'):
        assert params[key] == pytest.approx(expected[key], abs=1e-9)  # m/s
    assert [expected[key] for key in ('max_speed', 'corner_speed', 'corner_speed_wide', 'min_speed')] == [2.5, 2.5, 2.5, 1.429]


@pytest.mark.parametrize('package,filename,args', [
    ('gap_follow', 'gap_follow_launch.py', []),
    ('pure_pursuit', 'pure_pursuit_launch.py', []),
    ('neural_racer', 'neural_racer_launch.py', ['model_hash:=capture-placeholder']),
    ('racerbot_launch', 'race_launch.py', [
        f'run_dir:={ROOT / "tools/dev/tests/fixtures/launch_params/run_dir"}']),
])
def test_driving_launch_refuses_unknown_machine(car, monkeypatch, package, filename, args):
    monkeypatch.delenv('RACERBOT_CAR')
    with pytest.raises(launch_params.CaptureError, match='RACERBOT_CAR'):
        capture(package, filename, args)


def test_driving_launch_refuses_missing_calibration_key(car, monkeypatch, tmp_path):
    cars = tmp_path / 'cars'
    shutil.copytree(CARS, cars)
    overlay = cars / 'rb1/calibration/gap_follow.yaml'
    overlay.write_text(overlay.read_text().replace('    max_steering_angle: 0.33\n', ''))
    monkeypatch.setenv('RACERBOT_CARS_DIR', str(cars))
    with pytest.raises(launch_params.CaptureError, match='max_steering_angle'):
        capture('gap_follow', 'gap_follow_launch.py')


def test_inline_overrides_still_last(car):
    params = node(capture('racerbot_launch', 'map_course_launch.py'), '/gap_follow_node')['params']
    # Mapping defaults opponent awareness off, despite the racing YAML.
    assert params['enable_opponent_awareness'] is False


def test_map_course_speed_overrides_read_effective_values(car, monkeypatch, tmp_path):
    cars = tmp_path / 'cars'
    shutil.copytree(CARS, cars)
    tuning = cars / 'rb1/tuning/gap_follow.yaml'
    tuning.parent.mkdir(exist_ok=True)
    tuning.write_text('gap_follow_node:\n  ros__parameters:\n    # why: fixture: car-specific corner speed\n    corner_speed: 2.6\n')
    monkeypatch.setenv('RACERBOT_CARS_DIR', str(cars))
    defaults = {**yaml_parameters(SHARED), 'corner_speed': 2.6}
    expected = mapping_speed_overrides(SHARED, '4.0', '', defaults=defaults)
    assert expected != mapping_speed_overrides(SHARED, '4.0', '')
    params = node(capture('racerbot_launch', 'map_course_launch.py',
                          ['mapping_max_speed:=4.0']), '/gap_follow_node')['params']
    for key, value in expected.items():
        assert params[key] == value


def test_pure_pursuit_launch_defaults_read_effective_values(car, monkeypatch, tmp_path):
    cars = tmp_path / 'cars'
    shutil.copytree(CARS, cars)
    tuning = cars / 'rb1/tuning/pure_pursuit.yaml'
    tuning.parent.mkdir(exist_ok=True)
    tuning.write_text('pure_pursuit_node:\n  ros__parameters:\n    # why: fixture: car-specific steering\n    steering_law: stanley\n')
    monkeypatch.setenv('RACERBOT_CARS_DIR', str(cars))
    params = node(capture('pure_pursuit', 'pure_pursuit_launch.py'), '/pure_pursuit_node')['params']
    assert params['steering_law'] == 'stanley'
