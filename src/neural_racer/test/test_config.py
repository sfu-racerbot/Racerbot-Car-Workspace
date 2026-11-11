"""The shipped config (config/neural_racer.yaml). ROS-free.

Runner: python3 -m pytest src/neural_racer/test/test_config.py -v   (unsourced)
Oracles: the workspace safety policy (CLAUDE.md), and gap_follow.yaml -- the
wrapper's braking and footprint values are gap_follow's, so the two stay in step.
"""
import os
from pathlib import Path
import sys

import nr_testutil
import pytest
import yaml

from neural_racer.safety_wrapper import WrapperConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'racerbot_car'))
from racerbot_car.guards import CALIBRATION_KEYS  # noqa: E402
from racerbot_car.identity import known_car_ids  # noqa: E402
from racerbot_car.layers import effective_config  # noqa: E402
from racerbot_car.paths import cars_dir  # noqa: E402

CARS = sorted(car for car in known_car_ids(cars_dir())
              if (cars_dir() / car / 'calibration/neural_racer.yaml').is_file())

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, '..', 'config', 'neural_racer.yaml')
GAP_FOLLOW = os.path.join(HERE, '..', '..', 'gap_follow', 'config', 'gap_follow.yaml')


def _params(path, node):
    with open(path) as f:
        return yaml.safe_load(f)[node]['ros__parameters']


def test_deadman_on_and_shadow_by_default():
    p = _params(CONFIG, 'neural_racer_node')
    assert p['enable_deadman'] is True and p['deadman_button'] == 4
    assert p['shadow'] is True
    assert p['max_speed'] == 1.0 and p['mode'] == 'reactive' and p['require_imu'] is True
    assert p['camera_enabled'] is False


@pytest.mark.parametrize('car', CARS)
def test_wrapper_values_match_gap_follow(car):
    p = effective_config('neural_racer.yaml', 'neural_racer_node', car_id=car)
    g = effective_config('gap_follow.yaml', 'gap_follow_node', car_id=car)
    # max_lateral_accel is deliberately NOT paired here, since 2026-10-27:
    # gap_follow raised its cornering limit to 2.5 m/s^2 from car 1's grip
    # data, while neural_racer keeps 2.0 in training and deployment. The cap
    # is never a training-time change (see the 2026-10-23 neural-racer plan,
    # docs/superpowers/specs/2026-10-23-neural-racer-plan.md). Every other
    # pair below is checked exactly, except the DELIBERATE ones, each pinned
    # to the value the policy was trained with.
    pairs = {'max_steering_angle': 'max_steering_angle', 'car_width': 'car_width',
             'car_length': 'car_length', 'max_braking_decel': 'max_braking_decel',
             'braking_latency_sec': 'braking_latency_sec',
             'max_steering_rate': 'max_steering_rate', 'command_slew_max_dt': 'command_slew_max_dt',
             'reserve_distance': 'forward_stop_clearance', 'wheelbase': 'wheelbase',
             'laser_offset_x': 'laser_offset_x', 'forward_fov_deg': 'forward_fov_deg',
             'swept_clearance_margin': 'swept_clearance_margin',
             'swept_clearance_max_length': 'swept_clearance_max_length'}
    pinned = {k: DELIBERATE[k](car) for k in DELIBERATE if DELIBERATE[k](car) is not None}
    mismatched = {k: (p[k], g[v]) for k, v in pairs.items() if k not in pinned and p[k] != g[v]}
    assert mismatched == {}
    assert {k: p[k] for k in pinned} == pinned


# Wrapper values that deliberately differ from gap_follow's, each pinned to its own value
# (None: paired with gap_follow on that car as usual).
DELIBERATE = {
    # gap_follow's forward_stop_clearance went 0.20 -> 0.12 (2026-11-07). The policies were
    # trained with 0.20, and a sim what-if (2026-11-08: two policies, 0.20 vs 0.12) changed nothing.
    'reserve_distance': lambda car: 0.20,
    # gap_follow reads 5.0 m of swept path; the training sim's wrapper reads 3.0.
    'swept_clearance_max_length': lambda car: 3.0,
    # Car 2's gap_follow went to 0.42 rad (2026-11-07); its neural_racer overlay keeps 0.349,
    # since the policy never asks for more than observation.STEER_LIMIT (0.33).
    'max_steering_angle': lambda car: 0.349 if car == 'rb2' else None,
}


@pytest.mark.parametrize('car', CARS)
def test_yaml_matches_wrapper_defaults(car):
    # Per-car calibration keys (racerbot_car.guards.CALIBRATION_KEYS, team lead's decision 13:
    # physical characteristics are per car) differ between cars by definition, so no code
    # default can equal every car's; test_wrapper_calibration_defaults_are_the_strictest_cars
    # pins those. Every other key is shared and must equal the code default for every car.
    p = effective_config('neural_racer.yaml', 'neural_racer_node', car_id=car)
    d = WrapperConfig()
    shared = [k for k in vars(d) if k != 'max_speed' and k not in _CALIBRATION]
    assert shared, 'no shared keys left to compare: the test lost its subject'
    assert {k: p[k] for k in shared} == {k: getattr(d, k) for k in shared}


_CALIBRATION = set(CALIBRATION_KEYS['neural_racer.yaml']['neural_racer_node'])


def test_wrapper_calibration_defaults_are_the_strictest_cars():
    # A node whose car overlay did not load must clamp no wider than any car allows: each
    # calibration default is the smallest calibrated value (oracle: the cars' own overlays).
    assert _CALIBRATION == {'max_steering_angle'}, 'a new calibration key needs its rule here'
    assert len(CARS) >= 2
    values = [effective_config('neural_racer.yaml', 'neural_racer_node', car_id=car)['max_steering_angle']
              for car in CARS]
    assert WrapperConfig().max_steering_angle == min(values)
    assert WrapperConfig().max_steering_angle == pytest.approx(0.33, abs=1e-12)   # rad, car 1's
