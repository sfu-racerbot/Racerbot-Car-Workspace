"""Oracle: 2026-11-06 car_measure runs 163633, 170803, 171030 and wiring brief."""
import pytest

from racerbot_car.guards import check_calibration_overlay
from racerbot_car.layers import effective_config


@pytest.mark.parametrize('car,angle,grip,accel,gain', [
    ('rb1', 0.30, 2.5, 6.0, 1.0), ('rb2', 0.38, 2.9, 4.0, 0.99)])
def test_effective_line_physics(car, angle, grip, accel, gain):
    values = effective_config('auto_map_race.yaml', 'auto_map_race_node', car)
    # oracle: measured runs above; rb1 retains shared defaults by specification.
    assert [values[k] for k in ('profile_max_steering_angle', 'profile_max_lateral_accel',
                               'profile_max_lateral_accel_right', 'profile_max_accel',
                               'profile_steering_linkage_gain')] == [angle, grip, grip, accel, gain]
    assert values['race_line_mode'] == 'fastest'
    assert values['profile_max_speed'] == 5.0
    pp = effective_config('pure_pursuit.yaml', 'pure_pursuit_node', car)
    assert pp['max_speed'] == 5.0
    if car == 'rb2':
        assert pp['max_lateral_accel'] == grip


def test_line_calibration_allows_physics_but_not_safety_or_mode(tmp_path):
    path = tmp_path / 'auto_map_race.yaml'
    # oracle: spec, calibration admits physical measurements, never policy keys.
    path.write_text('auto_map_race_node:\n  ros__parameters:\n    profile_max_accel: 4.0\n')
    assert check_calibration_overlay(path, path.name) == []
    for key in ('enable_deadman', 'emergency_stop_distance', 'race_line_mode', 'profile_max_speed'):
        path.write_text(f'auto_map_race_node:\n  ros__parameters:\n    {key}: 1\n')
        problems = check_calibration_overlay(path, path.name)
        assert len(problems) == 1 and key in problems[0]
