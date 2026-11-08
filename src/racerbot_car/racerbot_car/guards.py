"""Validate per-car calibration and tuning overlays."""

import math
from pathlib import Path
import re

import yaml


CALIBRATION_KEYS = {
    'vesc.yaml': {
        '/**': {key: 'required' for key in (
            'speed_to_erpm_gain', 'steering_angle_to_servo_gain',
            'steering_angle_to_servo_offset', 'servo_min', 'servo_max')},
        'vesc_to_odom_node': {'speed_to_erpm_gain': 'required',
                              'speed_to_erpm_offset': 'optional'},
    },
    'sensors.yaml': {'urg_node': {key: 'optional' for key in (
        'serial_port', 'ip_address', 'ip_port')}},
    'joy_teleop.yaml': {'joy_teleop': {
        'human_control.axis_mappings.drive-steering_angle.scale': 'required'}},
    'gap_follow.yaml': {'gap_follow_node': {
        'max_steering_angle': 'required', 'arc_guard_max_correction': 'required'}},
    'pure_pursuit.yaml': {'pure_pursuit_node': {
        'max_steering_angle': 'required', 'max_lateral_accel': 'optional'}},
    'auto_map_race.yaml': {'auto_map_race_node': {key: 'optional' for key in (
        'profile_max_steering_angle', 'profile_wheelbase',
        'profile_max_lateral_accel', 'profile_max_lateral_accel_right',
        'profile_max_accel', 'profile_max_brake', 'profile_steering_linkage_gain',
        'profile_steering_rate', 'profile_steering_delay', 'profile_speed_delay')}},
    'neural_racer.yaml': {'neural_racer_node': {'max_steering_angle': 'required'}},
    'odom_calibration.yaml': {'odom_calibration_node': {key: 'required' for key in (
        'vehicle_model', 'planner_max_steering_rad', 'speed_to_erpm_gain',
        'steering_angle_to_servo_gain', 'steering_angle_to_servo_offset',
        'servo_min', 'servo_max')}},
}


class CalibrationOverlayError(ValueError):
    """An overlay is not a ROS parameter mapping."""


class TuningOverlayError(RuntimeError):
    """A tuning overlay cannot safely be loaded."""


SAFETY_TRUE = {'enable_deadman', 'enable_ttc', 'enable_swept_clearance'}
SAFETY_FIXED = {'deadman_button'}
SAFETY_CEILING = {'max_speed', 'profile_max_speed', 'profile_max_brake',
                  'ttc_min_closing_speed'}
SAFETY_FLOOR = {
    'ttc_threshold_sec', 'ttc_path_margin', 'car_width', 'car_length',
    'optimize_car_width', 'emergency_stop_clearance', 'forward_stop_clearance',
    'swept_clearance_margin', 'swept_clearance_max_length', 'emergency_stop_distance',
    'profile_wall_clearance', 'narrow_corridor_clearance', 'open_corridor_clearance',
    'overtake_min_side_clearance', 'body_clearance_fov_deg',
}


def _flatten(data, prefix=''):
    result = {}
    for key, value in data.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, name))
        else:
            result[name] = value
    return result


def overlay_blocks(path: Path) -> dict[str, dict]:
    try:
        with open(path, encoding='utf-8') as stream:
            data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise CalibrationOverlayError(f'{path}: invalid YAML: {exc}') from exc
    if not isinstance(data, dict):
        raise CalibrationOverlayError(f'{path}: expected a mapping')
    result = {}
    for block, value in data.items():
        if not isinstance(value, dict) or not isinstance(value.get('ros__parameters'), dict):
            raise CalibrationOverlayError(f'{path}: {block} needs a ros__parameters mapping')
        result[block] = _flatten(value['ros__parameters'])
    return result


def check_calibration_overlay(path: Path, config_name: str) -> list[str]:
    if config_name not in CALIBRATION_KEYS:
        return [f'{path}: {config_name} has no calibration allow-list.']
    allowed = CALIBRATION_KEYS[config_name]
    blocks = overlay_blocks(path)
    problems = []
    for block, params in blocks.items():
        if block not in allowed:
            problems.append(f'{path}: {block} is not a calibration block for {config_name}.')
            continue
        for key in params:
            if key not in allowed[block]:
                problems.append(f'{path}: {block}.{key} is not a calibration key for {config_name}.')
    for block, keys in allowed.items():
        for key, requirement in keys.items():
            if requirement == 'required' and key not in blocks.get(block, {}):
                problems.append(f'{path}: {block}.{key} is a missing required calibration key for {config_name}.')
    return problems


def _tuning_locations(document):
    # YAML's parsed nodes retain source line numbers, including nested keys.
    locations = {}

    def leaves(node, prefix, block):
        if not isinstance(node, yaml.MappingNode):
            return
        for key_node, value_node in node.value:
            key = key_node.value
            name = f'{prefix}.{key}' if prefix else key
            if isinstance(value_node, yaml.MappingNode):
                leaves(value_node, name, block)
            else:
                locations[(block, name)] = key_node.start_mark.line

    for block_node, body in document.value:
        for key_node, params in body.value:
            if key_node.value == 'ros__parameters':
                leaves(params, '', block_node.value)
    return locations


def tuning_overrides(path: Path) -> list[tuple[str, str, object, str]]:
    """Return flattened tuning leaves and the reason immediately above each."""
    try:
        text = path.read_text(encoding='utf-8')
        document = yaml.compose(text, Loader=yaml.SafeLoader)
        if document is None:
            return []
        blocks = overlay_blocks(path)
        locations = _tuning_locations(document)
    except (CalibrationOverlayError, yaml.YAMLError) as exc:
        raise TuningOverlayError(f'{path}: {exc}') from exc
    lines = text.splitlines()
    result = []
    for block, params in blocks.items():
        for key, value in params.items():
            line = locations.get((block, key), 0)
            match = re.fullmatch(r'# why:\s*(\S.*)', lines[line - 1].strip()) if line else None
            result.append((block, key, value, match.group(1).strip() if match else ''))
    return result


def check_tuning_overlay(path: Path, shared_path: Path, config_name: str) -> list[str]:
    """Check each tuning leaf against its shared value and its source comment."""
    overrides = tuning_overrides(path)
    if not overrides:
        return []
    try:
        shared = overlay_blocks(shared_path)
    except (CalibrationOverlayError, yaml.YAMLError) as exc:
        raise TuningOverlayError(f'{path}: {exc}') from exc

    calibration = {key for keys in CALIBRATION_KEYS.get(config_name, {}).values() for key in keys}
    problems = []
    for block, key, value, why in overrides:
        label = f'{path}: {block}.{key}'
        if key not in shared.get(block, {}):
            problems.append(f'{label}: key is absent from the shared node block.')
        if key in calibration:
            problems.append(f'{label}: calibration keys are not tuning keys.')
        if not why:
            problems.append(f'{label}: add a non-empty # why: comment directly above the key.')
        name = key.rsplit('.', 1)[-1]
        if name in SAFETY_TRUE and value is not True:
            problems.append(f'{label}: must be true.')
        if name in SAFETY_FIXED:
            problems.append(f'{label}: cannot be overridden.')
        ceiling = name in SAFETY_CEILING or name.endswith('_timeout_sec')
        floor = name in SAFETY_FLOOR
        if ceiling or floor:
            original = shared.get(block, {}).get(key)
            numeric = lambda x: type(x) in (int, float) and not math.isnan(x)
            if not numeric(value) or not numeric(original):
                problems.append(f'{label}: shared and overlay values must be numeric and not NaN.')
            elif ceiling and not value <= original:
                problems.append(f'{label}: exceeds the shared safety ceiling {original}.')
            elif floor and not value >= original:
                problems.append(f'{label}: falls below the shared safety floor {original}.')
    return problems
