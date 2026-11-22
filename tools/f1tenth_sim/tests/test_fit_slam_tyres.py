"""Fixed numerical/profile oracles and development/evaluation protocol isolation."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fit_slam_tyres as ft  # noqa: E402
from sim_fidelity.calibration import FLOORS, calibration_for_car, with_floor  # noqa: E402
from sim_fidelity.grip import ProgressiveGrip  # noqa: E402

PINNED = {'rb1': [5.153504850577635, 4.71326245124088, .3981510161398307,
                  .6890943799648725, 1.5371303171196806],
          'rb2': [4.333612688384001, 5.882536607979519, .5296822924799998,
                  .5836831052800002, 1.6724281753599997]}


def payload(car='rb1'):
    return dict(car=car, fit_runs=list(ft.DEV[car]), parameters=[3, 4, .55, .65, 1.5],
                acceptance=ft.ACCEPTANCE, objective_scales=ft.SCALES, clamp=False, frozen=True)


def metric_row(loss=1, horizon=2):
    return dict(horizon_s=horizon, start_t=0, yaw_rms_rad_s=.3 * loss,
                slip_cg_rms_rad=.1 * loss, position_error_m=.5 * loss,
                speed_rms_mps=.2 * loss, heading_error_rad=.2 * loss)


@pytest.mark.parametrize('car', sorted(PINNED))
def test_frozen_tyre_baseline_and_candidate_keep_pre_speed_scales(car):
    # SLAM_TYRE_FIT.md and slam_tyres_2026_11_fit.json predate the speed fit.
    current = calibration_for_car(car)
    if car == 'rb2':
        assert current.odom_speed_scale == 1.0261486728316354
        assert current.speed_command_scale == 1.0108904807240708
    base = ft.base_calibration(car)
    assert base == replace(current, cornering_stiffness_front=4.718,
                           cornering_stiffness_rear=5.4562, tyre_model=None, grip_curve=None,
                           odom_speed_scale=1.0, speed_command_scale=1.0, floor='')
    candidate = ft.candidate_calibration(car, PINNED[car])
    assert base.odom_speed_scale == candidate.odom_speed_scale == 1.0
    assert base.speed_command_scale == candidate.speed_command_scale == 1.0


def test_objective_and_candidate_keep_fixed_physics_and_enforce_bounds():
    rows = [dict(windows=[metric_row(1), metric_row(2)])]
    assert ft.objective(rows) == pytest.approx(1.5)
    with pytest.raises(ValueError, match='nonempty'):
        ft.objective([])
    with pytest.raises(ValueError, match='finite'):
        ft.objective([dict(windows=[metric_row(np.nan)])])
    cal = ft.candidate_calibration('rb1', [3, 4, .55, .65, 1.5])
    assert (cal.cornering_stiffness_front, cal.cornering_stiffness_rear) == (3, 4)
    assert (cal.tyre_model.peak_front, cal.tyre_model.peak_rear, cal.tyre_model.shape) == (.55, .65, 1.5)
    base = ft.base_calibration('rb1')
    for key in ('mass', 'lr', 'yaw_inertia', 'surface_friction', 'steering_delay_ticks', 'speed_delay_ticks'):
        assert getattr(cal, key) == getattr(base, key)
    for bad in ([.1, 4, .55, .65, 1.5], [3, 4, .55, .65], [3, 4, .55, .65, 2]):
        with pytest.raises(ValueError, match='bounds'):
            ft.candidate_calibration('rb1', bad)


def test_acceptance_requires_both_horizons_all_metrics_and_speed_tolerance():
    default = ft.sr.aggregate([metric_row(1, h) for h in (1, 2, 4)])
    candidate = ft.sr.aggregate([metric_row(.8, h) for h in (1, 2, 4)])
    assert ft.acceptance(default, candidate)['accepted']
    candidate['4']['metrics']['yaw_rms_rad_s']['median'] = .301
    assert not ft.acceptance(default, candidate)['accepted']
    candidate['4']['metrics']['yaw_rms_rad_s']['median'] = .24
    candidate['2']['metrics']['speed_rms_mps']['mean'] = .22
    assert ft.acceptance(default, candidate)['accepted']
    candidate['2']['metrics']['speed_rms_mps']['mean'] = .221
    assert not ft.acceptance(default, candidate)['accepted']
    candidate['2']['metrics']['speed_rms_mps']['mean'] = .16
    candidate['4']['metrics']['position_error_m']['mean'] = .5
    assert not ft.acceptance(default, candidate)['accepted']
    candidate['4']['windows'] = 0
    assert 'missing or mismatched' in ft.acceptance(default, candidate)['failures'][-1]


def test_frozen_hash_and_development_manifest_refuse_tampering(tmp_path):
    path = tmp_path / 'frozen.json'
    p = payload()
    record = dict(sha256=ft.digest(p), payload=p)
    path.write_text(json.dumps(record))
    assert ft.load_frozen(path)['payload']['parameters'] == p['parameters']
    p['parameters'][0] += .1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='fingerprint'):
        ft.load_frozen(path)
    p['fit_runs'] = ['20261116-095242']
    record['sha256'] = ft.digest(p)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='development-only'):
        ft.load_frozen(path)


def test_subset_scoring_is_stratified_and_uses_only_h2(monkeypatch):
    seen = []
    def replay(motion, drive, index, ids, horizon, cal, **kwargs):
        seen.append(index)
        return metric_row(1, horizon)
    monkeypatch.setattr(ft.sr, 'replay_window', replay)
    track = dict(run='dev', motion={}, drive=[], topic='/drive',
                 eligible=[(h, i, []) for h in (1, 2, 4) for i in range(10)])
    result = ft.score_tracks([track], ft.base_calibration('rb1'), subset=3)
    assert seen == [0, 4, 9]
    assert result[0]['summary']['2']['windows'] == 3


def test_fit_reads_only_development_and_finally_ranks_full_h2(tmp_path, monkeypatch):
    calls = []
    def load(car, names, loc, data):
        calls.append(tuple(names))
        assert tuple(names) == ft.DEV[car]
        return ['development']
    def score(tracks, cal, horizons=(2,), subset=None, clamp=False):
        assert tracks == ['development']
        front = cal.cornering_stiffness_front
        loss = front if subset else abs(front - 3) + .1
        return [dict(windows=[metric_row(loss)])]
    monkeypatch.setattr(ft, 'load_tracks', load)
    monkeypatch.setattr(ft, 'score_tracks', score)
    path = tmp_path / 'frozen.json'
    result = ft.fit('rb1', 'unused', 'unused', path, coarse=4, maxfev=1)
    assert calls == [ft.DEV['rb1']]
    assert result['payload']['parameters'] == [3, 2, .55, .55, 1.4]
    assert ft.load_frozen(path)['sha256'] == result['sha256']
    with pytest.raises(ValueError, match='positive'):
        ft.fit('rb1', 'unused', 'unused', path, coarse=0)


def test_evaluation_checks_freeze_before_opening_any_drives(tmp_path, monkeypatch):
    calls = []
    p = payload()
    record = dict(sha256=ft.digest(p), payload=p)
    path = tmp_path / 'frozen.json'
    path.write_text(json.dumps(record))
    def load(car, names, loc, data):
        calls.append(tuple(names))
        return []
    def score(tracks, cal, horizons):
        loss = .8 if cal.tyre_model else 1
        return [dict(windows=[metric_row(loss, h) for h in horizons])]
    monkeypatch.setattr(ft, 'load_tracks', load)
    monkeypatch.setattr(ft, 'score_tracks', score)
    result = ft.evaluate(path, 'unused', 'unused', tmp_path / 'eval.json')
    assert result['accepted']
    assert calls == [ft.DEV['rb1'], ft.EVAL['rb1']['nov-eval']]
    calls.clear()
    record['sha256'] = 'bad'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='fingerprint'):
        ft.evaluate(path, 'unused', 'unused', tmp_path / 'bad.json')
    assert calls == []


def test_progressive_ablation_curve_uses_only_trusted_development_motion():
    asked = np.repeat([1., 2., 3.], 12)
    motion = dict(trusted=np.ones(36, dtype=bool), ground_speed=np.full(36, 2.),
                  steer_cmd=np.arctan(asked * .36 / 4), lateral_accel=np.repeat([.8, 1.7, 2.5], 12))
    curve = ft.progressive_curve([dict(motion=motion)], .36)
    assert np.array(curve) == pytest.approx(np.array([[0, 0], [1, .8], [2, 1.7], [3, 2.5]]))
    motion['trusted'][:] = False
    with pytest.raises(ValueError, match='insufficient'):
        ft.progressive_curve([dict(motion=motion)], .36)


def test_ablation_refreezes_unchanged_parameters_on_development_only(tmp_path, monkeypatch):
    p = payload('rb2')
    p['clamp_ablation'] = dict(off_loss=1., on_loss=1.2)
    path = tmp_path / 'frozen.json'
    path.write_text(json.dumps(dict(sha256=ft.digest(p), payload=p)))
    def load(car, names, loc, data):
        assert car == 'rb2' and tuple(names) == ft.DEV['rb2']
        return ['dev']
    def score(tracks, cal, clamp):
        assert tracks == ['dev'] and clamp
        assert cal.grip_curve == ((0, 0), (2, 1)) and cal.grip_curve_falls
        return [dict(windows=[metric_row(.75)])]
    monkeypatch.setattr(ft, 'load_tracks', load)
    monkeypatch.setattr(ft, 'progressive_curve', lambda tracks, wheelbase: ((0, 0), (2, 1)))
    monkeypatch.setattr(ft, 'score_tracks', score)
    out = tmp_path / 'refrozen.json'
    result = ft.ablate(path, 'unused', 'unused', out)
    assert result['payload']['parameters'] == p['parameters']
    assert result['payload']['clamp_ablation']['progressive_on_loss'] == pytest.approx(.75)
    assert ft.load_frozen(out)['sha256'] == result['sha256'] != ft.digest(p)


def test_cli_dispatches_fit_ablation_and_frozen_evaluation(tmp_path, monkeypatch):
    calls = []
    for name in ('fit', 'ablate', 'evaluate'):
        monkeypatch.setattr(ft, name, lambda *args, name=name: calls.append(name))
        extra = ['--car', 'rb1'] if name == 'fit' else ['--frozen', str(tmp_path / 'frozen.json')]
        assert ft.main([name, '--out', str(tmp_path / 'out.json'), *extra]) == 0
    assert calls == ['fit', 'ablate', 'evaluate']
    with pytest.raises(SystemExit) as error:
        ft.main(['evaluate', '--out', str(tmp_path / 'bad.json')])
    assert error.value.code == 2


@pytest.mark.parametrize('car', sorted(PINNED))
def test_candidate_profile_pins_values_sources_and_preserves_default(car):
    default = calibration_for_car(car)
    name = f'slam_tyres_2026_11_{car}'
    floor = FLOORS[name]
    base = ft.base_calibration(car)
    cal = with_floor(name, base)
    values = [cal.cornering_stiffness_front, cal.cornering_stiffness_rear,
              cal.tyre_model.peak_front, cal.tyre_model.peak_rear, cal.tyre_model.shape]
    assert values == PINNED[car]
    assert cal.grip_curve is None and cal.tyre_model.coupling == 0
    assert set(floor.fit_runs) == set(ft.DEV[car])
    assert set(floor.held_out_runs) == {run for runs in ft.EVAL[car].values() for run in runs}
    assert not set(floor.fit_runs) & set(floor.held_out_runs)
    assert set(floor.overrides) - {'floor'} <= set(floor.sources)
    assert calibration_for_car(car) == default


@pytest.mark.parametrize('car', sorted(PINNED))
def test_candidate_loads_saturating_gym_and_progressive_ablation_really_clamps(car):
    cal = with_floor(f'slam_tyres_2026_11_{car}', ft.base_calibration(car))
    original = ft.tyres.make_dynamics
    plant = ft.plant_factory(cal)
    try:
        assert ft.tyres.make_dynamics is original
        assert plant._tyres == cal.tyre_model
        assert plant.env.unwrapped.sim.dynamics_fn is ft._compile(cal.tyre_model)
        plant.reset(np.array([[20., 20., 0.]]), speeds=[2.])
        for _ in range(15):
            plant.step([[.25, 2.]])
        assert np.isfinite(plant.truth(0)).all() and plant.truth(0)[5] > .01
        assert plant.grip[0].total_steps == 0
    finally:
        plant.close()
    cal = replace(cal, grip_curve=((0., 0.), (2., .5), (10., 1.)))
    plant = ft.plant_factory(cal, clamp=True)
    try:
        assert isinstance(plant.grip[0], ProgressiveGrip)
        assert plant.env.unwrapped.sim.dynamics_fn is ft._compile(cal.tyre_model)
        plant.reset(np.array([[20., 20., 0.]]), speeds=[2.])
        for _ in range(15):
            plant.step([[.25, 2.]])
        assert plant.grip[0].limited_steps > 0
    finally:
        plant.close()
