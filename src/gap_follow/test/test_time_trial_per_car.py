"""time_trial:=true's per-car additions (gap_follow/launch_speed.py TIME_TRIAL_BY_CAR, 2026-11-14).

Spec: car 2's time trial adds steering_gain 0.8 and max_steering_rate 4.0 to the shared TIME_TRIAL; car 1 keeps the
shared values (tools/run_analysis/results/2026-11-14-gap-follow-pace.md: the same change made contact on three
simulator courses with car 1). The launch builds the node's overrides from the running car. Needs ROS sourced for
the launch tests; run as `rb python3 -m pytest src/gap_follow/test/test_time_trial_per_car.py`.
"""
import os
import sys

import pytest

from gap_follow.launch_speed import TIME_TRIAL, TIME_TRIAL_BY_CAR, time_trial_overrides

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def test_car_2_adds_the_softer_gain_and_faster_slew():
    # oracle: spec (module docstring)
    assert time_trial_overrides('rb2') == {**TIME_TRIAL, 'steering_gain': 0.8, 'max_steering_rate': 4.0}


def test_car_1_gets_exactly_the_shared_time_trial():
    assert time_trial_overrides('rb1') == TIME_TRIAL


def test_an_unknown_car_gets_the_shared_time_trial():
    assert time_trial_overrides('rb9') == TIME_TRIAL
    assert time_trial_overrides(None) == TIME_TRIAL


def test_the_shared_values_are_not_mutated():
    before = dict(TIME_TRIAL)
    out = time_trial_overrides('rb2')
    out['aim_depth_bias'] = -1.0
    assert TIME_TRIAL == before and TIME_TRIAL_BY_CAR['rb2'] == {'steering_gain': 0.8, 'max_steering_rate': 4.0}


def test_a_per_car_value_never_switches_opponent_awareness_back_on():
    # The time trial's refusal (awareness off) must survive any car's additions.
    for car in TIME_TRIAL_BY_CAR:
        assert time_trial_overrides(car)['enable_opponent_awareness'] is False


@pytest.mark.parametrize('car, gain, rate', [('rb2', 0.8, 4.0), ('rb1', None, None)])
def test_the_launch_passes_the_running_cars_time_trial(monkeypatch, car, gain, rate):
    monkeypatch.setenv('RACERBOT_CAR', car)
    from test_gap_follow_launch_steering_damping import _gap_node, _launch, _node_params
    nodes, _includes, ctx = _launch(time_trial='true')
    params = _node_params(_gap_node(nodes), ctx)
    assert params.get('steering_gain') == gain
    assert params.get('max_steering_rate') == rate
    assert params['aim_depth_bias'] == 1.0 and params['enable_opponent_awareness'] is False


def test_without_time_trial_car_2_gets_none_of_it(monkeypatch):
    monkeypatch.setenv('RACERBOT_CAR', 'rb2')
    from test_gap_follow_launch_steering_damping import _gap_node, _launch, _node_params
    nodes, _includes, ctx = _launch()
    params = _node_params(_gap_node(nodes), ctx)
    assert 'steering_gain' not in params and 'max_steering_rate' not in params
