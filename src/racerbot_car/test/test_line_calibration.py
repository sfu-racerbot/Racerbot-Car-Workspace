"""Oracle: 2026-11-06 car_measure runs 163633, 170803, 171030 and wiring brief."""
import pytest

from racerbot_car.guards import check_calibration_overlay
from racerbot_car.layers import effective_config


# 2026-11-07: rb2's planning grip 2.9 -> 3.2 and speed delay 0.10 -> 0.22, and its online
# ceiling (pure_pursuit max_lateral_accel) 2.9 -> 2.7, deliberately BELOW the planning grip:
# the ceiling binds only near full lock, where car 2 achieved 2.42 when asked 2.5-3
# (cars/rb2/calibration/*.yaml comments give the measurements and simulator evidence).
# 2026-11-09: online ceiling 2.7 -> 3.1 with the multi-start lines and speed lead: no contact
# on 06a/06b (6 simulator seeds) or the 50-course suite; still under the ~3.4 the floor
# achieves at hard asks (cars/rb2/calibration/pure_pursuit.yaml).
@pytest.mark.parametrize('car,angle,grip,accel,gain,delay,online', [
    ('rb1', 0.30, 2.5, 6.0, 1.0, 0.10, None), ('rb2', 0.38, 3.2, 4.0, 0.99, 0.22, 3.1)])
def test_effective_line_physics(car, angle, grip, accel, gain, delay, online):
    values = effective_config('auto_map_race.yaml', 'auto_map_race_node', car)
    # oracle: measured runs above; rb1 retains shared defaults by specification.
    assert [values[k] for k in ('profile_max_steering_angle', 'profile_max_lateral_accel',
                               'profile_max_lateral_accel_right', 'profile_max_accel',
                               'profile_steering_linkage_gain', 'profile_speed_delay')] == [
        angle, grip, grip, accel, gain, delay]
    assert values['race_line_mode'] == 'fastest'
    assert values['profile_max_speed'] == 5.0
    pp = effective_config('pure_pursuit.yaml', 'pure_pursuit_node', car)
    assert pp['max_speed'] == 5.0
    if online is not None:
        assert pp['max_lateral_accel'] == online


def test_line_calibration_allows_physics_but_not_safety_or_mode(tmp_path):
    path = tmp_path / 'auto_map_race.yaml'
    # oracle: spec, calibration admits physical measurements, never policy keys.
    path.write_text('auto_map_race_node:\n  ros__parameters:\n    profile_max_accel: 4.0\n')
    assert check_calibration_overlay(path, path.name) == []
    for key in ('enable_deadman', 'emergency_stop_distance', 'race_line_mode', 'profile_max_speed'):
        path.write_text(f'auto_map_race_node:\n  ros__parameters:\n    {key}: 1\n')
        problems = check_calibration_overlay(path, path.name)
        assert len(problems) == 1 and key in problems[0]


@pytest.mark.parametrize('car,expected', [('rb1', 1.0), ('rb2', 0.88)])
def test_launch_speed_scale_effective_per_car(car, expected):
    # oracle: spec (2026-11-13 brief): rb2's tuning overlay sets 0.88 (run 20261111-152132), rb1 keeps the shared
    # 1.0 (off); launch_scale_distance stays the shared 0.0 (one lap) on both.
    config = effective_config('pure_pursuit.yaml', 'pure_pursuit_node', car)
    assert config['launch_speed_scale'] == expected
    assert config['launch_scale_distance'] == 0.0
