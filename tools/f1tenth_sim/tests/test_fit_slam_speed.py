"""Analytic wheel ratios, frozen protocol, and recorded profile regressions."""

from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fit_slam_speed as fs  # noqa: E402
from sim_fidelity.calibration import FLOORS, calibration_for_car, describe, with_floor  # noqa: E402
from sim_fidelity.sensing import OdomSensor  # noqa: E402


def tracks(tmp_path, ratio=1.05):
    loc = tmp_path / 'loc'
    loc.mkdir()
    data = tmp_path / 'data'
    for run, wheel_ratio in zip(fs.ft.DEV['rb1'], (ratio - .04, ratio, ratio + .04)):
        t = np.arange(121) * .05
        a = np.zeros((len(t), 11))
        a[:, 0], a[:, 1], a[:, 4], a[:, 5], a[:, 6] = t, 2 * t, 1, 1, 2 * wheel_ratio
        a[:10, 4], a[:10, 6] = 0, 200
        np.savez(loc / f'{run}.npz', a=a)
        params = data / 'rb1/racerbot_runs' / run / 'metadata/params'
        params.mkdir(parents=True)
        for node, gain in [('ackermann_to_vesc_node', 4600), ('vesc_to_odom_node', -4040)]:
            values = dict(speed_to_erpm_gain=gain, speed_to_erpm_offset=0)
            (params / f'{node}.yaml').write_text(yaml.safe_dump({node: {'ros__parameters': values}}))
    return loc, data


def summary():
    metrics = {key: dict(mean=1., median=1.) for key in ('speed_rms_mps', *fs.GUARDS)}
    return {h: dict(windows=2, metrics=deepcopy(metrics)) for h in ('2', '4')}


def test_ratio_freeze_is_trusted_pooled_development_only_and_cli(tmp_path):
    loc, data = tracks(tmp_path)
    out = tmp_path / 'frozen.json'
    fs.main(['freeze', '--car', 'rb1', '--loc', str(loc), '--data', str(data), '--out', str(out)])
    p = fs.load_frozen(out)['payload']
    assert p['fit_runs'] == list(fs.ft.DEV['rb1'])
    assert p['odom_speed_scale'] == pytest.approx(1.05, abs=1e-12)
    assert p['speed_command_scale'] == pytest.approx(4600 / 4040 / 1.05, abs=1e-12)
    assert p['pooled_wheel_over_ground']['count'] > 100
    assert [e['wheel_over_ground']['median'] for e in p['evidence']] == pytest.approx(
        [1.01, 1.05, 1.09], abs=1e-12)
    for run in fs.ft.DEV['rb1']:
        with np.load(loc / f'{run}.npz') as saved:
            a = saved['a'].copy()
        a[:, 4] = 0
        np.savez(loc / f'{run}.npz', a=a)
    with pytest.raises(ValueError, match='no moving'):
        fs.freeze('rb1', loc, data, out)


def test_gains_validate_recorded_offsets_and_consistency(tmp_path):
    loc, data = tracks(tmp_path)
    run = fs.ft.DEV['rb1'][0]
    path = data / 'rb1/racerbot_runs' / run / 'metadata/params/ackermann_to_vesc_node.yaml'
    for gain, offset in [(0, 0), (float('nan'), 0), (4600, 1)]:
        path.write_text(yaml.safe_dump({'node': {'ros__parameters': dict(
            speed_to_erpm_gain=gain, speed_to_erpm_offset=offset)}}))
        with pytest.raises(ValueError, match='zero offset'):
            fs.recorded_gains(data, 'rb1', run)
    path.write_text(yaml.safe_dump({'node': {'ros__parameters': dict(
        speed_to_erpm_gain=4700, speed_to_erpm_offset=0)}}))
    with pytest.raises(ValueError, match='inconsistent'):
        fs.freeze('rb1', loc, data, tmp_path / 'bad.json')


def test_candidate_changes_only_two_scales_and_refuses_invalid():
    for car in fs.ft.DEV:
        base = calibration_for_car(car)
        proposed = fs.candidate(car, 1.1, 1.2)
        assert proposed.odom_speed_scale == 1.1
        assert proposed.speed_command_scale == pytest.approx(1.2 / 1.1, abs=1e-12)
        for field in fields(base):
            if field.name not in ('odom_speed_scale', 'speed_command_scale'):
                assert getattr(proposed, field.name) == getattr(base, field.name)
    for bad in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match='positive finite'):
            fs.candidate('rb1', bad, 1)
        with pytest.raises(ValueError, match='positive finite'):
            fs.candidate('rb1', 1, bad)


def test_acceptance_strict_speed_both_horizons_and_two_percent_guard():
    base, prop = summary(), summary()
    assert not fs.acceptance(base, prop)['accepted']
    for h in ('2', '4'):
        prop[h]['metrics']['speed_rms_mps'] = dict(mean=.9, median=.9)
    assert fs.acceptance(base, prop)['accepted']
    for h in ('2', '4'):
        for key in ('mean', 'median'):
            bad = deepcopy(prop)
            bad[h]['metrics']['speed_rms_mps'][key] = 1.
            assert not fs.acceptance(base, bad)['accepted']
        for metric in fs.GUARDS:
            bad = deepcopy(prop)
            bad[h]['metrics'][metric]['mean'] = 1.02
            assert fs.acceptance(base, bad)['accepted']
            bad[h]['metrics'][metric]['mean'] = 1.020001
            assert not fs.acceptance(base, bad)['accepted']
            bad[h]['metrics'][metric]['mean'] = np.nan
            assert not fs.acceptance(base, bad)['accepted']
        bad = deepcopy(prop)
        bad[h]['windows'] = 0
        assert not fs.acceptance(base, bad)['accepted']


def test_frozen_protocol_and_evaluation_gate_reject_tampering(tmp_path):
    loc, data = tracks(tmp_path)
    out = tmp_path / 'frozen.json'
    frozen = fs.freeze('rb1', loc, data, out)
    for key, value, message in [('odom_speed_scale', 2, 'fingerprint'),
                                ('fit_runs', ['20261116-095242'], 'manifest'),
                                ('acceptance', {}, 'protocol'),
                                ('speed_command_scale', 2, 'command scale')]:
        record = deepcopy(frozen)
        record['payload'][key] = value
        if key != 'odom_speed_scale':
            record['sha256'] = fs.ft.digest(record['payload'])
        out.write_text(json.dumps(record))
        with pytest.raises(ValueError, match=message):
            fs.main(['evaluate', '--frozen', str(out), '--loc', '/does-not-exist',
                     '--data', '/does-not-exist', '--out', str(tmp_path / 'eval.json')])


@pytest.mark.parametrize('car,ratio,gain_ratio', [
    ('rb1', 1.0746581411238885, 4600 / 4040),
    ('rb2', 1.0261486728316354, 4614 / 4447.983786),
])
def test_recorded_profile_pins_scales_default_tyres_and_steps(car, ratio, gain_ratio):
    # Recorded measurements: November dev manifests in slam_speed_2026_11_fit.json.
    name = f'slam_speed_2026_11_{car}'
    cal = with_floor(name, calibration_for_car(car))
    assert cal.odom_speed_scale == ratio
    assert cal.speed_command_scale == pytest.approx(gain_ratio / ratio, abs=1e-15)
    assert cal.tyre_model is None and cal.grip_curve is None
    assert (cal.cornering_stiffness_front, cal.cornering_stiffness_rear) == (4.718, 5.4562)
    assert FLOORS[name].fit_runs == fs.ft.DEV[car]
    assert set(FLOORS[name].held_out_runs) == {r for runs in fs.ft.EVAL[car].values() for r in runs}
    assert set(FLOORS[name].sources) == {'odom_speed_scale', 'speed_command_scale'}
    assert OdomSensor(0, np.random.default_rng(1), ratio).update(2) == 2 * ratio
    speeds = []
    for profile in (fs.baseline(car), cal):
        plant = fs.ft.sr.make_plant(profile)
        try:
            plant.reset(np.array([[20., 20., 0.]]), speeds=[1.5])
            for _ in range(120):
                plant.step([[0., 2.]])
            speeds.append(plant.truth(0)[3])
        finally:
            plant.close()
    assert speeds[1] > speeds[0] + .001


def test_report_verdicts_follow_fixed_rule_and_freeze_sources():
    report = json.loads((Path(fs.__file__).parent / 'slam_speed_2026_11_fit.json').read_text())
    for car in ('rb1', 'rb2'):
        result = report['cars'][car]
        frozen = result['frozen']
        assert frozen['sha256'] == fs.ft.digest(frozen['payload'])
        assert frozen['sha256'] in FLOORS[f'slam_speed_2026_11_{car}'].sources['odom_speed_scale']
        assert result['frozen_before_evaluation']
        pair = result['reports']['nov-eval']
        expected = True
        for h in ('2', '4'):
            a, b = (pair[label]['pooled'][h]['metrics'] for label in ('default', 'candidate'))
            expected &= all(b['speed_rms_mps'][key] < a['speed_rms_mps'][key] for key in ('mean', 'median'))
            expected &= all(b[key]['mean'] <= 1.02 * a[key]['mean'] for key in fs.GUARDS)
        assert result['accepted'] == expected
        assert pair['verdict']['accepted'] == expected
        if not expected:
            assert 'REJECTED' in FLOORS[f'slam_speed_2026_11_{car}'].description
            assert calibration_for_car(car) == fs.baseline(car)
        else:
            default = calibration_for_car(car)
            assert default.odom_speed_scale == frozen['payload']['odom_speed_scale']
            assert default.speed_command_scale == frozen['payload']['speed_command_scale']
            assert with_floor('pre_slam_speed_2026_11_rb2', default).speed_command_scale == 1.
            assert with_floor('pre_slam_speed_2026_11_rb2', default).odom_speed_scale == 1.
            sources = {name: source for _, name, _, source in describe(default)}
            assert frozen['sha256'] in sources['odom_speed_scale']
