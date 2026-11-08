"""Resolve car-specific ROS parameter files and their effective values."""

import os
from pathlib import Path

from . import guards, hardware, identity, paths, profile

SHARED_CONFIGS = {
    'auto_map_race.yaml': 'src/pure_pursuit/config/auto_map_race.yaml',
    'vesc.yaml': 'src/f1tenth_system/f1tenth_stack/config/vesc.yaml',
    'sensors.yaml': 'src/f1tenth_system/f1tenth_stack/config/sensors.yaml',
    'joy_teleop.yaml': 'src/f1tenth_system/f1tenth_stack/config/joy_teleop.yaml',
    'gap_follow.yaml': 'src/gap_follow/config/gap_follow.yaml',
    'pure_pursuit.yaml': 'src/pure_pursuit/config/pure_pursuit.yaml',
    'neural_racer.yaml': 'src/neural_racer/config/neural_racer.yaml',
    'odom_calibration.yaml': 'src/odom_calibration/config/odom_calibration.yaml',
}


def repo_root() -> Path:
    """Find the config source tree, distinct from paths.workspace_root()'s data root."""
    override = os.environ.get('RACERBOT_WS')
    source = Path(__file__).resolve().parents[3]
    if override and (Path(override) / 'src/racerbot_car').is_dir():
        return Path(override).resolve()
    if (source / 'src/racerbot_car').is_dir():
        return source
    raise FileNotFoundError(f'No src/racerbot_car in RACERBOT_WS={override!r} or {source}')


def shared_config_path(config_name: str) -> Path:
    """Locate a shared config in the source tree, not the workspace data root."""
    if config_name not in SHARED_CONFIGS:
        raise KeyError(f'Unknown config {config_name!r}; known configs: {", ".join(SHARED_CONFIGS)}')
    return repo_root() / SHARED_CONFIGS[config_name]


def effective_config(config_name: str, node_name: str, car_id: str | None = None) -> dict:
    """Read effective parameters using shared configs from the source tree, not the data root."""
    return effective_values(layered(str(shared_config_path(config_name)), config_name, car_id), node_name)


class MissingCalibrationError(RuntimeError):
    """A required car calibration is absent."""


def launch_car_id(cars_dir: Path | None = None) -> str:
    car_id = identity.current_car_id()
    message = hardware.contradiction(
        profile.load_profile(car_id, cars_dir), profile.all_profiles(cars_dir),
        hardware.usb_ids_present())
    if message:
        raise hardware.CarHardwareError(f'{message} Refusing to start.')
    return car_id


def car_params(launch_config_name: str, shared_default: str, config_name: str, car_id: str) -> list:
    from launch.substitutions import LaunchConfiguration

    layered(shared_default, config_name, car_id)
    return [LaunchConfiguration(launch_config_name), *overlay_files(config_name, car_id)]


def overlay_files(config_name: str, car_id: str | None = None,
                  cars_dir: Path | None = None) -> list[str]:
    car_id = car_id if car_id is not None else identity.current_car_id()
    directory = Path(cars_dir if cars_dir is not None else paths.cars_dir()) / car_id
    candidates = [directory / kind / config_name for kind in ('calibration', 'tuning')]
    return [str(path.resolve()) for path in candidates if path.is_file()]


def car_overlays(car_id: str) -> list[Path]:
    """Return all existing calibration and tuning YAML files for a car."""
    directory = paths.cars_dir() / car_id
    return sorted(path.resolve() for kind in ('calibration', 'tuning')
                  for path in (directory / kind).glob('*.yaml') if path.is_file())


def layered(shared_path: str, config_name: str, car_id: str | None = None,
            cars_dir: Path | None = None) -> list[str]:
    car_id = car_id if car_id is not None else identity.current_car_id()
    directory = Path(cars_dir if cars_dir is not None else paths.cars_dir())
    calibration = directory / car_id / 'calibration' / config_name
    if config_name in guards.CALIBRATION_KEYS:
        blocks = guards.overlay_blocks(calibration) if calibration.is_file() else {}
        missing = [f'{block}.{key}' for block, keys in guards.CALIBRATION_KEYS[config_name].items()
                   for key, requirement in keys.items()
                   if requirement == 'required' and key not in blocks.get(block, {})]
        if missing:
            raise MissingCalibrationError(
                f'Car {car_id}: {calibration} is missing required calibration keys: '
                f'{", ".join(missing)}. Configure src/racerbot_car/cars/{car_id}/calibration/{config_name}.')
    tuning = directory / car_id / 'tuning' / config_name
    if tuning.is_file():
        problems = guards.check_tuning_overlay(tuning, Path(shared_path), config_name)
        if problems:
            raise guards.TuningOverlayError('\n'.join(problems))
    return [shared_path, *overlay_files(config_name, car_id, directory)]


def effective_values(paths: list[str], node_name: str | None) -> dict:
    entries = {}
    for path in paths:
        for block, params in guards.overlay_blocks(Path(path)).items():
            entries.setdefault(block, {}).update(params)
    keys = {'/**'}
    if node_name:
        if not node_name.startswith('/'):
            node_name = '/' + node_name
        basename = node_name.rsplit('/', 1)[-1]
        keys.update((node_name, basename, '/**/' + basename))
    result = {}
    for block, params in entries.items():
        if block in keys:
            result.update(params)
    return result


def check_block_keys_match(shared_path: str, overlay_path: str) -> list[str]:
    shared = guards.overlay_blocks(Path(shared_path))
    overlay = guards.overlay_blocks(Path(overlay_path))
    problems = []
    for block, params in overlay.items():
        for key in params:
            shared_blocks = [name for name, values in shared.items() if key in values]
            if shared_blocks and block not in shared_blocks:
                problems.append(f'{overlay_path}: {block}.{key} uses a different block than {shared_path}: '
                                f'{", ".join(shared_blocks)}.')
    return problems
