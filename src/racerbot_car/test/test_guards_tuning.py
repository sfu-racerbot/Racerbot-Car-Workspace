from pathlib import Path
import shutil
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from racerbot_car import guards, layers

ROOT = Path(__file__).resolve().parents[3]
CARS = ROOT / 'src/racerbot_car/cars'
SHARED = ROOT / 'src/gap_follow/config/gap_follow.yaml'
OVERLAYS = sorted(CARS.glob('*/tuning/*.yaml'))


def test_tuning_overrides_lists_each_key_with_its_why(tmp_path):
    overlay = tmp_path / 'tuning.yaml'
    overlay.write_text('node:\n  ros__parameters:\n    # why: grip\n    speed: 0.5\n'
                       '    group:\n      # why: surface\n      gain: 2\n'
                       '    missing: true\n')
    assert guards.tuning_overrides(overlay) == [
        ('node', 'speed', 0.5, 'grip'), ('node', 'group.gain', 2, 'surface'),
        ('node', 'missing', True, '')]


def check(tmp_path, key, value, why='# why: physical difference', shared=None):
    source = tmp_path / 'shared.yaml'
    source.write_text(yaml.safe_dump({'node': {'ros__parameters': shared or {
        key: 1.0}}}), encoding='utf-8')
    overlay = tmp_path / 'overlay.yaml'
    overlay.write_text(f'node:\n  ros__parameters:\n    {why}\n    {key}: {value}\n', encoding='utf-8')
    return guards.check_tuning_overlay(overlay, source, 'gap_follow.yaml')


def rejected(problems, key):
    assert any('overlay.yaml' in p and 'node' in p and key in p for p in problems)


def test_tuning_rejects_enable_deadman_false(tmp_path):
    rejected(check(tmp_path, 'enable_deadman', 'false'), 'enable_deadman')


def test_tuning_allows_enable_deadman_true(tmp_path):
    assert check(tmp_path, 'enable_deadman', 'true') == []


def test_tuning_rejects_deadman_button(tmp_path):
    rejected(check(tmp_path, 'deadman_button', '1'), 'deadman_button')


def test_tuning_rejects_max_speed_above_shared(tmp_path):
    rejected(check(tmp_path, 'max_speed', '1.1'), 'max_speed')


def test_tuning_allows_max_speed_below_shared(tmp_path):
    assert check(tmp_path, 'max_speed', '0.9') == []


def test_tuning_allows_max_speed_equal_to_shared(tmp_path):
    assert check(tmp_path, 'max_speed', '1.0') == []


def test_tuning_rejects_timeout_longer_than_shared(tmp_path):
    rejected(check(tmp_path, 'joy_timeout_sec', '1.1'), 'joy_timeout_sec')


def test_tuning_rejects_ttc_threshold_below_shared(tmp_path):
    rejected(check(tmp_path, 'ttc_threshold_sec', '0.9'), 'ttc_threshold_sec')


def test_tuning_allows_ttc_threshold_above_shared(tmp_path):
    assert check(tmp_path, 'ttc_threshold_sec', '1.1') == []


@pytest.mark.parametrize('key', ['max_speed', 'ttc_threshold_sec'])
@pytest.mark.parametrize('value', ['.nan', 'wrong', 'true'])
def test_tuning_rejects_nan_and_non_numeric_for_ceiling_and_floor_keys(tmp_path, key, value):
    rejected(check(tmp_path, key, value), key)


def test_tuning_rejects_unknown_key(tmp_path):
    rejected(check(tmp_path, 'max_speeed', '0.9', shared={'max_speed': 1.0}), 'max_speeed')


def test_tuning_rejects_calibration_key(tmp_path):
    rejected(check(tmp_path, 'max_steering_angle', '0.2'), 'max_steering_angle')


def test_tuning_rejects_key_without_why(tmp_path):
    rejected(check(tmp_path, 'max_speed', '0.9', why='# explanation'), 'max_speed')


def test_tuning_rejects_empty_why(tmp_path):
    rejected(check(tmp_path, 'max_speed', '0.9', why='# why:   '), 'max_speed')


def test_tuning_reports_all_violations(tmp_path):
    shared = tmp_path / 'shared.yaml'
    shared.write_text('node:\n  ros__parameters:\n    enable_deadman: true\n    max_speed: 1.0\n')
    overlay = tmp_path / 'overlay.yaml'
    overlay.write_text('node:\n  ros__parameters:\n    enable_deadman: false\n    max_speed: 2.0\n')
    problems = guards.check_tuning_overlay(overlay, shared, 'gap_follow.yaml')
    assert len(problems) == 4
    assert sum('enable_deadman' in p for p in problems) == 2
    assert sum('max_speed' in p for p in problems) == 2
    assert all(str(overlay) in p and 'node.' in p for p in problems)


def test_tuning_accepts_a_why_above_a_nested_leaf(tmp_path):
    shared = tmp_path / 'shared.yaml'
    shared.write_text('node:\n  ros__parameters:\n    group:\n      max_speed: 1.0\n')
    overlay = tmp_path / 'overlay.yaml'
    overlay.write_text('node:\n  ros__parameters:\n    group:\n      # why: different tyres\n      max_speed: 0.9\n')
    assert guards.check_tuning_overlay(overlay, shared, 'gap_follow.yaml') == []


@pytest.mark.parametrize('content', ['', '# no overrides\n', '{}\n',
                                   'node:\n  ros__parameters: {}\n'])
def test_empty_tuning_does_not_read_shared_file(tmp_path, content):
    overlay = tmp_path / 'overlay.yaml'
    overlay.write_text(content)
    assert guards.check_tuning_overlay(overlay, tmp_path / 'missing.yaml', 'gap_follow.yaml') == []


@pytest.mark.parametrize('path', OVERLAYS if OVERLAYS else [None])
def test_every_car_tuning_overlay_passes_guards(path):
    assert OVERLAYS == sorted(CARS.glob('*/tuning/*.yaml'))
    if path is None:
        assert not OVERLAYS
    else:
        assert guards.check_tuning_overlay(path, layers.shared_config_path(path.name), path.name) == []


def setup_car(tmp_path, key, value):
    car = tmp_path / 'testcar'
    shutil.copytree(CARS / 'rb1/calibration', car / 'calibration')
    tuning = car / 'tuning'
    tuning.mkdir()
    overlay = tuning / 'gap_follow.yaml'
    overlay.write_text(f'gap_follow_node:\n  ros__parameters:\n    # why: different tyres\n    {key}: {value}\n')
    return overlay


def test_layered_refuses_a_bad_tuning_overlay(tmp_path):
    setup_car(tmp_path, 'enable_deadman', 'false')
    with pytest.raises(guards.TuningOverlayError, match='enable_deadman'):
        layers.layered(str(SHARED), 'gap_follow.yaml', 'testcar', tmp_path)


def test_layered_accepts_a_good_tuning_overlay(tmp_path):
    shared = yaml.safe_load(SHARED.read_text())['gap_follow_node']['ros__parameters']
    overlay = setup_car(tmp_path, 'max_speed', shared['max_speed'] / 2)
    result = layers.layered(str(SHARED), 'gap_follow.yaml', 'testcar', tmp_path)
    assert result == [str(SHARED), str((tmp_path / 'testcar/calibration/gap_follow.yaml').resolve()), str(overlay.resolve())]


@pytest.mark.parametrize('content', ['node: [', 'node: {wrong: 1}'])
def test_layered_refuses_invalid_tuning_yaml(tmp_path, content):
    overlay = setup_car(tmp_path, 'max_speed', '0.1')
    overlay.write_text(content)
    with pytest.raises(guards.TuningOverlayError, match='gap_follow.yaml'):
        layers.layered(str(SHARED), 'gap_follow.yaml', 'testcar', tmp_path)


# Tier 1 arc check (pure_pursuit.yaml, 2026-11-07). Each shared value is the spec, copied from
# that file. A car's tuning overlay may only raise these: a lower arc trip distance, horizon or
# backstop lets the car drive closer to a return before Tier 1 acts.
PURE_PURSUIT_FLOOR_KEYS = {
    'emergency_arc_clearance': 0.14,
    'emergency_arc_length': 1.0,
    'emergency_hard_stop_distance': 0.25,
}


def check_pure_pursuit(tmp_path, key, value, shared_value):
    source = tmp_path / 'shared.yaml'
    source.write_text(yaml.safe_dump({'pure_pursuit_node': {'ros__parameters': {key: shared_value}}}),
                      encoding='utf-8')
    overlay = tmp_path / 'overlay.yaml'
    overlay.write_text(f'pure_pursuit_node:\n  ros__parameters:\n    # why: physical difference\n'
                       f'    {key}: {value}\n', encoding='utf-8')
    return guards.check_tuning_overlay(overlay, source, 'pure_pursuit.yaml')


@pytest.mark.parametrize('key, shared_value', sorted(PURE_PURSUIT_FLOOR_KEYS.items()))
def test_tuning_rejects_a_lower_tier1_arc_value(tmp_path, key, shared_value):
    """Oracle: spec (SAFETY_FLOOR in guards.py: a floor key may not fall below its shared value).
    Ten per cent under the shared value is refused under the floor's own message."""
    problems = check_pure_pursuit(tmp_path, key, shared_value * 0.9, shared_value)
    assert any('overlay.yaml' in p and key in p and 'falls below the shared safety floor' in p
               for p in problems), problems


@pytest.mark.parametrize('key, shared_value', sorted(PURE_PURSUIT_FLOOR_KEYS.items()))
def test_tuning_allows_a_higher_or_equal_tier1_arc_value(tmp_path, key, shared_value):
    """Oracle: spec (the same floor rule: at or above the shared value passes)."""
    assert check_pure_pursuit(tmp_path, key, shared_value * 1.1, shared_value) == []
    assert check_pure_pursuit(tmp_path, key, shared_value, shared_value) == []
