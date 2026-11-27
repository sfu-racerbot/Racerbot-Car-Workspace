"""Analytic trajectories and deliberate discontinuities; no ROS or recorded bags."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_analysis import slam_motion as sm  # noqa: E402
from run_analysis.loc import COL, COLUMNS  # noqa: E402


def track(t, x, y, yaw, wheel=2.0, gyro=0.0, steer=0.0):
    a = np.zeros((len(t), len(COLUMNS)))
    for name, values in dict(t=t, x=x, y=y, yaw=yaw, wheel_v=wheel,
                             gyro_z=gyro, steer_cmd=steer, h10=1, h20=1, lb=1).items():
        a[:, COL[name]] = values
    return a


def straight(duration=5.0):
    t = np.arange(0, duration, 0.05)
    yaw = 0.6
    return track(t, 2 * t * np.cos(yaw), 2 * t * np.sin(yaw), yaw)


def circle(speed=2.5, rate=0.7, beta=-0.12, duration=12.0):
    t = np.arange(0, duration, 0.05)
    t[1:-1] += 0.004 * np.sin(np.arange(len(t) - 2))
    yaw = rate * t + 2.8
    a = track(t, speed / rate * np.sin(yaw + beta),
              -speed / rate * np.cos(yaw + beta), np.arctan2(np.sin(yaw), np.cos(yaw)),
              wheel=1.04 * speed, gyro=rate, steer=np.arctan(0.36 * rate / speed))
    return a


def test_rotated_straight_has_correct_body_velocity_and_zero_slip():
    motion = sm.extract_motion(straight(), lr=0.18)
    good = motion['trusted']
    assert good.sum() > 85
    assert motion['ground_speed'][good] == pytest.approx(2.0, abs=1e-10)
    assert motion['longitudinal'][good] == pytest.approx(2.0, abs=1e-10)
    for key in ('lateral_rear', 'lateral_cg', 'yaw_rate', 'slip_rear', 'slip_cg', 'curvature'):
        assert motion[key][good] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize('rate,beta', [(0.7, -0.12), (-0.7, 0.12)])
def test_irregular_wrapped_circle_recovers_drift_at_rear_and_cg(rate, beta):
    motion = sm.extract_motion(circle(rate=rate, beta=beta), lr=0.18)
    good = motion['trusted']
    assert good.sum() > 200
    assert motion['ground_speed'][good] == pytest.approx(2.5, abs=0.001)
    assert motion['longitudinal'][good] == pytest.approx(2.5 * np.cos(beta), abs=0.001)
    assert motion['lateral_rear'][good] == pytest.approx(2.5 * np.sin(beta), abs=0.001)
    assert motion['yaw_rate'][good] == pytest.approx(rate, abs=1e-10)
    assert motion['slip_rear'][good] == pytest.approx(beta, abs=0.0005)
    cg_lateral = 2.5 * np.sin(beta) + 0.18 * rate
    assert motion['lateral_cg'][good] == pytest.approx(cg_lateral, abs=0.001)
    assert motion['slip_cg'][good] == pytest.approx(np.arctan2(cg_lateral, 2.5 * np.cos(beta)), abs=0.0005)
    assert motion['lateral_accel'][good] == pytest.approx(2.5 * rate, abs=0.001)
    assert motion['curvature'][good] == pytest.approx(rate / 2.5, abs=0.0002)


@pytest.mark.parametrize('column,offset', [('x', 5.0), ('yaw', 0.8)])
def test_pose_jump_has_a_margin_and_does_not_poison_either_side(column, offset):
    a = straight()
    a[50:, COL[column]] += offset
    motion = sm.extract_motion(a, lr=0.18)
    t, good = motion['t'], motion['trusted']
    assert not good[(t >= 2.25) & (t <= 2.70)].any()
    assert good[t < 2].sum() > 25 and good[t > 3].sum() > 25
    assert np.unique(motion['segment'][good]).size == 2
    assert motion['ground_speed'][good] == pytest.approx(2.0, abs=1e-10)
    assert motion['yaw_rate'][good] == pytest.approx(0.0, abs=1e-10)
    assert np.isnan(motion['ground_speed'][~good]).all()


def test_quality_mask_break_is_never_bridged_and_boundary_is_inclusive():
    a = straight()
    a[:, COL['h10']] = 0.75
    a[48:53, COL['h10']] = 0.749
    # Two locally correct paths, separated by a small offset hidden inside the bad interval.
    a[50:, COL['x']] += 0.20
    motion = sm.extract_motion(a, lr=0.18)
    good = motion['trusted']
    assert good[:40].sum() > 25 and good[60:].sum() > 25
    assert not good[45:56].any()
    assert np.unique(motion['segment'][good]).size == 2
    assert motion['ground_speed'][good] == pytest.approx(2.0, abs=1e-10)


def test_time_gap_is_not_interpolated_across():
    a = straight()
    a[50:, COL['t']] += 2.0
    motion = sm.extract_motion(a, lr=0.18)
    assert np.unique(motion['segment'][motion['trusted']]).size == 2
    assert not motion['trusted'][48:53].any()
    assert motion['ground_speed'][motion['trusted']] == pytest.approx(2.0, abs=1e-10)


def test_empty_short_parked_and_rejected_tracks_have_no_invented_motion():
    for a in (straight()[:0], straight()[:6]):
        motion = sm.extract_motion(a, lr=0.18)
        assert not motion['trusted'].any()
        summary = sm.summarize(motion, 0.36)
        assert summary['trusted_fraction'] == 0
        assert summary['wheel_over_ground']['median'] is None
        assert summary['gyro_vs_slam']['scale'] is None
        assert summary['sideslip_by_abs_lateral_accel'][0]['rear_deg']['count'] == 0
        json.dumps(summary, allow_nan=False)
    a = straight()
    a[:, [COL['x'], COL['y'], COL['yaw']]] = 0
    motion = sm.extract_motion(a, lr=0.18)
    assert motion['trusted'].sum() > 85
    assert np.isnan(motion['slip_rear']).all()
    assert sm.summarize(motion, 0.36)['moving_samples'] == 0
    a[:, COL['h10']] = 0
    assert not sm.extract_motion(a, lr=0.18)['trusted'].any()


@pytest.mark.parametrize('fault', ['shape', 'duplicate', 'reverse', 'nan_time', 'lr'])
def test_bad_inputs_are_refused(fault):
    a, lr = straight(), 0.18
    if fault == 'shape':
        a = a[:, :-1]
    elif fault == 'duplicate':
        a[1, COL['t']] = a[0, COL['t']]
    elif fault == 'reverse':
        a = a[::-1]
    elif fault == 'nan_time':
        a[1, COL['t']] = np.nan
    else:
        lr = -0.18
    with pytest.raises(ValueError):
        sm.extract_motion(a, lr)


def test_summary_recovers_sensor_scale_bias_rms_bins_and_curvature():
    # World position and body yaw are polynomials, so the cubic filter is exact.
    t = np.arange(0, 10, 0.05)
    speed = np.hypot(2.0, 0.04 * t)
    rate = 0.06 * t
    a = track(t, 2 * t, 0.02 * t ** 2, 0.03 * t ** 2,
              wheel=1.08 * speed, gyro=0.85 * rate + 0.04)
    motion = sm.extract_motion(a, 0.18)
    summary = sm.summarize(motion, 0.36)
    moving = motion['trusted'] & (motion['ground_speed'] >= .5) & (motion['longitudinal'] > 0)
    assert summary['wheel_over_ground']['median'] == pytest.approx(1.08, abs=1e-10)
    assert summary['wheel_over_ground']['speed_rms_mps'] == pytest.approx(
        np.sqrt(np.mean((0.08 * speed[moving]) ** 2)), abs=1e-10)
    gyro = summary['gyro_vs_slam']
    assert gyro['scale'] == pytest.approx(0.85, abs=1e-10)
    assert gyro['bias_rad_s'] == pytest.approx(0.04, abs=1e-10)
    assert gyro['raw_rms_rad_s'] == pytest.approx(np.sqrt(np.mean((-0.15 * rate[moving] + .04) ** 2)), abs=1e-10)
    assert gyro['fitted_rms_rad_s'] < 1e-10
    drift = sm.extract_motion(circle(speed=3.0, rate=1.2, beta=-0.1), 0.18)
    result = sm.summarize(drift, .36)
    row = result['sideslip_by_abs_lateral_accel'][3]
    assert row['rear_abs_deg']['count'] > 200
    assert row['rear_abs_deg']['median'] == pytest.approx(np.degrees(.1), abs=.03)
    assert row['rear_abs_deg']['iqr'] < .02
    assert row['cg_abs_deg']['median'] < row['rear_abs_deg']['median']
    bins = [row for row in result['steady_curvature_by_abs_command'] if row['count']]
    assert len(bins) == 1
    assert bins[0]['achieved_curvature_per_m']['median'] == pytest.approx(.4, abs=.0002)
    assert bins[0]['achieved_over_commanded']['median'] == pytest.approx(1.0, abs=.001)


def test_steady_selection_rejects_changing_commands_and_pool_keeps_run_boundaries():
    a = circle(duration=5)
    first = sm.extract_motion(a, .18)
    assert first['steady'].sum() > 60
    transient = sm.extract_motion(a, .18)
    transient['yaw_rate'] = 2 * transient['t']
    assert not sm.steady_mask(transient).any()
    transient['yaw_rate'][:] = .7
    transient['ground_speed'] = 2 * transient['t'] + 1
    assert not sm.steady_mask(transient).any()
    a[:, COL['steer_cmd']] = np.where(np.arange(len(a)) % 2, 0.25, -0.25)
    second = sm.extract_motion(a, .18)
    assert not second['steady'].any()
    combined = sm.pool_motions([first, second])
    assert np.unique(combined['segment'][combined['trusted']]).size == 2
    pooled = sm.summarize(combined, .36)
    assert pooled['samples'] == len(a) * 2
    assert pooled['trusted_samples'] == first['trusted'].sum() + second['trusted'].sum()
    assert sum(b['count'] for b in pooled['steady_curvature_by_abs_command']) == first['steady'].sum()
    with pytest.raises(ValueError, match='at least one'):
        sm.pool_motions([])


def test_cli_writes_strict_json_and_cache_samples_and_refuses_outside_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    source, report = tmp_path / 'track.npz', tmp_path / 'summary.json'
    np.savez(source, a=circle())
    assert sm.main([str(source), '--car', 'rb2', '--json', str(report), '--samples']) == 0
    result = json.loads(report.read_text())
    assert result['car'] == 'rb2'
    # Car 2's CG from its axle weights (statics: lr = L * front / total), measured 2026-11-23.
    assert result['geometry']['lr_m'] == pytest.approx(0.325 * 1.673 / (1.673 + 2.283), abs=1e-9)  # m
    assert result['runs'][0]['run'] == 'track'
    assert result['runs'][0]['wheel_over_ground']['median'] == pytest.approx(1.04, abs=.001)
    with np.load(tmp_path / '.cache/racerbot/slam_motion/rb2_track.npz') as saved:
        assert saved['trusted'].sum() > 200
        assert np.nanmedian(saved['slip_rear']) == pytest.approx(-.12, abs=.0005)
    with pytest.raises(SystemExit) as error:
        sm.main([str(source), '--car', 'rb2', '--json', str(report), '--samples', str(tmp_path / 'data')])
    assert error.value.code == 2
