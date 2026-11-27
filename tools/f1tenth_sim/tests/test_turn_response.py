"""A held rack angle requests curvature; yaw and sideslip relax in SI units."""
import csv
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fit_turn_response as study
from f1tenth_gym.envs.dynamic_models.single_track import vehicle_dynamics_st
from sim_fidelity.turn_response import TurnResponse, make_dynamics, steady_rate


def test_rejects_unusable_models():
    for model in [(1, 0, 4, math.nan), (1, .1, 4, .05), (0, 0, 4, .05),
                  (1, 0, 0, .05), (1, 0, 4, .024)]:
        with pytest.raises(ValueError):
            TurnResponse(*model)


def test_steering_sign_low_angle_slope_and_acceleration_ceiling():
    speed, wheelbase, gain, limit = 4., .36, .85, 4.7
    def rate(steer):
        return steady_rate(speed, steer, wheelbase, gain, .01, limit)
    assert rate(-.01) == 0
    assert rate(-.01 + 1e-6) / 1e-6 == pytest.approx(speed * gain / wheelbase)
    assert speed * rate(.4) == pytest.approx(limit, rel=.005)
    assert rate(-.42) == pytest.approx(-rate(.4))


def test_dynamics_relaxes_yaw_and_slip_but_preserves_gym_kinematics():
    model = TurnResponse(1., 0., 4.5, .10)
    dynamics = make_dynamics(model)
    params = np.array([.55, 4.718, 5.4562, .18, .18, .074, 3.74, .04712,
                       -.42, .42, -3.2, 3.2, 7.319, 9.51, -5., 20.])
    u = np.array([.12, -.6])
    x = np.array([0., 0., 0., 3., .3, 1.2, -.04])
    stock = vehicle_dynamics_st(x, u, params)
    actual = dynamics(x, u, params)
    np.testing.assert_array_equal(actual[:5], stock[:5])
    assert actual[5] == pytest.approx(-1.2 / .10)  # straight wheels: decay toward zero
    assert actual[6] == pytest.approx((math.asin(.18 * 1.2 / 3.) + .04) / .10)
    for x[3] in [-.6, 0., .49]:
        np.testing.assert_array_equal(dynamics(x, u, params), vehicle_dynamics_st(x, u, params))


def test_fit_excludes_eval_rejected_and_unknown_dev(tmp_path, monkeypatch):
    dev = '20261111-153721'
    rows = [('rb1', dev, '1', 'development', 'accepted', '2.5'),
            ('rb1', '20261116-095242', '2', 'evaluation', 'accepted', '3'),
            ('rb1', dev, '3', 'development', 'rejected', ''),
            ('rb1', 'unknown', '4', 'development', 'accepted', '4')]
    for name, columns, data in [
            ('blocks.csv', ('car', 'run', 'b_ns', 'role', 'status', 'initial_mps'), rows),
            ('ticks.csv', ('car', 'run', 'b_ns'), [r[:3] for r in rows])]:
        with (tmp_path / name).open('w') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(data)
    monkeypatch.setattr(study, 'INPUT_HASH', {})
    assert [(k, v) for k, v, _ in study.blocks(tmp_path, 'rb1', True)] == [(('rb1', dev, '1'), 2.5)]


def test_plant_replay_uses_commands_only_and_changes_turning(tmp_path):
    cal = study.study_calibration('rb2')
    rows = [dict(step=i, steer_rad='.25', demand_mps='3.5') for i in range(480)]
    model = study.frozen_model(Path(study.__file__).parent / 'turn_response_study/rb2.json', 'rb2')
    track = study.flat_map(tmp_path)
    baseline, candidate = [study.replay(track, cal, m, 3.5, rows) for m in (None, model)]
    assert abs(candidate[-1, 5] - baseline[-1, 5]) > .10
    np.testing.assert_allclose(candidate[-1, 5], steady_rate(candidate[-1, 3], candidate[-1, 2],
                               cal.physical_wheelbase, model.gain, model.trim_rad, model.lateral_limit_mps2), atol=.002)
