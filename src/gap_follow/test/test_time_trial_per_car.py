"""time_trial:=true's per-car additions (gap_follow/launch_speed.py TIME_TRIAL_BY_CAR, 2026-11-14).

Spec: no car adds to the shared TIME_TRIAL today. Car 2's steering_gain 0.8 / max_steering_rate 4.0 (#71) was
reverted on 2026-11-15 after the floor matrix (tools/run_analysis/results/2026-11-14-gap-follow-pace.md, update
2026-11-15). The mechanism stays: the launch builds the node's overrides from the running car, and the tests of
the mechanism use a fixture entry. Needs ROS sourced for
the launch tests; run as `rb python3 -m pytest src/gap_follow/test/test_time_trial_per_car.py`.
"""
import os
import sys

import pytest

from gap_follow.launch_speed import TIME_TRIAL, time_trial_overrides

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


FIXTURE = {'steering_gain': 0.8, 'max_steering_rate': 4.0}


@pytest.fixture
def car_entry(monkeypatch):
    """A per-car entry for the mechanism tests, since none ships."""
    from gap_follow import launch_speed
    monkeypatch.setitem(launch_speed.TIME_TRIAL_BY_CAR, 'rb2', dict(FIXTURE))
    return launch_speed.TIME_TRIAL_BY_CAR


def test_car_2_gets_exactly_the_shared_time_trial():
    # oracle: spec (module docstring) -- #71's values reverted
    assert time_trial_overrides('rb2') == TIME_TRIAL


def test_a_per_car_entry_is_added_to_the_shared_values(car_entry):
    assert time_trial_overrides('rb2') == {**TIME_TRIAL, **FIXTURE}
    assert time_trial_overrides('rb1') == TIME_TRIAL


def test_car_1_gets_exactly_the_shared_time_trial():
    assert time_trial_overrides('rb1') == TIME_TRIAL


def test_an_unknown_car_gets_the_shared_time_trial():
    assert time_trial_overrides('rb9') == TIME_TRIAL
    assert time_trial_overrides(None) == TIME_TRIAL


def test_the_shared_values_are_not_mutated(car_entry):
    before = dict(TIME_TRIAL)
    out = time_trial_overrides('rb2')
    out['aim_depth_bias'] = -1.0
    out['steering_gain'] = -1.0
    assert TIME_TRIAL == before and car_entry['rb2'] == FIXTURE


def test_a_per_car_value_never_switches_opponent_awareness_back_on(monkeypatch):
    # The time trial's refusal (awareness off) must survive any car's additions, even one that names it.
    from gap_follow import launch_speed
    monkeypatch.setitem(launch_speed.TIME_TRIAL_BY_CAR, 'rb2', {'enable_opponent_awareness': True})
    assert time_trial_overrides('rb2')['enable_opponent_awareness'] is False


@pytest.mark.parametrize('car, gain, rate', [('rb2', 0.8, 4.0), ('rb1', None, None)])
def test_the_launch_passes_the_running_cars_time_trial(monkeypatch, car_entry, car, gain, rate):
    monkeypatch.setenv('RACERBOT_CAR', car)
    from test_gap_follow_launch_steering_damping import _gap_node, _launch, _node_params
    nodes, _includes, ctx = _launch(time_trial='true')
    params = _node_params(_gap_node(nodes), ctx)
    assert params.get('steering_gain') == gain
    assert params.get('max_steering_rate') == rate
    assert params['aim_depth_bias'] == 1.0 and params['enable_opponent_awareness'] is False


def test_without_time_trial_car_2_gets_none_of_it(monkeypatch, car_entry):
    monkeypatch.setenv('RACERBOT_CAR', 'rb2')
    from test_gap_follow_launch_steering_damping import _gap_node, _launch, _node_params
    nodes, _includes, ctx = _launch()
    params = _node_params(_gap_node(nodes), ctx)
    assert 'steering_gain' not in params and 'max_steering_rate' not in params
