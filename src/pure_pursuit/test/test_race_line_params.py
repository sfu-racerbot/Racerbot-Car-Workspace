"""Parameter wiring oracles: interface contract and measured per-car physics."""
from dataclasses import asdict

import pytest

from pure_pursuit.race_line import RaceLineConfig, load_params


# 2026-11-07: rb2 grip 2.9 -> 3.2 and speed delay 0.10 -> 0.22 (its calibration overlay gives
# the measurements); shared body floor 0.08 -> 0.22 on straights, 0.30 near the steering limit
# (auto_map_race.yaml comment: the line must cover the controller's tracking error).
@pytest.mark.parametrize('car,angle,grip,accel,gain,delay', [
    ('rb1', .30, 2.5, 6., 1., .10), ('rb2', .38, 3.2, 4., .99, .22)])
def test_car_physics_and_controller_stop(car, angle, grip, accel, gain, delay, monkeypatch):
    monkeypatch.setenv('RACERBOT_CAR', car)
    values = load_params()
    cfg = RaceLineConfig.from_params(values)
    # oracle: spec docs/hardware-reference.md; 20261106 car_measure runs in overlay.
    assert asdict(cfg.physics) == dict(
        a_lat_left=grip, a_lat_right=grip, a_accel=accel, a_brake=3.,
        v_max=5., v_min=.5, wheelbase=.36, max_steering_angle=angle,
        linkage_gain=gain, steering_rate=1., steering_delay=.1, speed_delay=delay)
    from racerbot_car.layers import effective_config
    assert cfg.emergency_stop_distance == effective_config(
        'pure_pursuit.yaml', 'pure_pursuit_node', car)['emergency_stop_distance']
    assert cfg.mode == 'fastest'
    assert cfg.spacing == .15
    assert (cfg.wall_margin, cfg.cone_margin, cfg.body_floor, cfg.body_floor_tight) == (
        .15, .10, .22, .30)


def test_parameter_mapping_preserves_handedness_and_all_builder_knobs():
    values = load_params('rb1')
    values.update(profile_max_lateral_accel=2.1, profile_max_lateral_accel_right=1.8,
                  profile_steering_rate=.8, profile_steering_delay=.12, profile_speed_delay=.07,
                  race_line_mode='recorded', waypoint_spacing=.2, race_line_wall_margin=.18,
                  race_line_cone_margin=.13, race_line_body_floor=.09,
                  profile_min_feature_wavelength=1.7, profile_curvature_margin=.9,
                  profile_max_deviation=.25, emergency_stop_distance=.41)
    cfg = RaceLineConfig.from_params(values)
    # oracle: spec, each named parameter maps to its documented field, not its neighbour.
    assert (cfg.physics.a_lat_left, cfg.physics.a_lat_right) == (2.1, 1.8)
    assert (cfg.physics.steering_rate, cfg.physics.steering_delay, cfg.physics.speed_delay) == (.8, .12, .07)
    assert (cfg.mode, cfg.spacing, cfg.wall_margin, cfg.cone_margin, cfg.body_floor) == (
        'recorded', .2, .18, .13, .09)
    assert (cfg.min_feature_wavelength, cfg.curvature_margin, cfg.max_deviation,
            cfg.emergency_stop_distance) == (1.7, .9, .25, .41)


@pytest.mark.parametrize('key,value,error', [
    ('profile_max_lateral_accel', 0., 'a_lat_left'),
    ('profile_max_accel', -1., 'a_accel'),
    ('profile_steering_delay', float('nan'), 'steering_delay'),
    ('profile_speed_delay', float('inf'), 'speed_delay'),
    ('race_line_mode', 'unknown', 'unknown mode'),
    ('waypoint_spacing', 0., 'spacing'),
])
def test_invalid_parameters_are_refused(key, value, error):
    values = load_params('rb1')
    values[key] = value
    # oracle: invariant, invalid physical values never reach the builder.
    with pytest.raises(ValueError, match=error):
        RaceLineConfig.from_params(values)


def test_missing_physics_is_not_silently_replaced_with_another_cars_default():
    values = load_params('rb1')
    del values['profile_max_lateral_accel_right']
    # oracle: spec, no implicit CarPhysics (car 2) default for an incomplete config.
    with pytest.raises(KeyError, match='profile_max_lateral_accel_right'):
        RaceLineConfig.from_params(values)
