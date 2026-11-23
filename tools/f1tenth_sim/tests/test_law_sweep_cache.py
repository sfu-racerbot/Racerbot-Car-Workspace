"""A cached baseline is usable only for its car and effective calibration."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _plan(cache, car):
    import law_sweep

    return law_sweep.prepare_jobs(
        ['pure_pursuit', 'stanley'], ['test_track'], [7], {}, 60.0, car, cache)


@pytest.mark.parametrize('change', ['car', 'calibration', 'legacy'])
def test_incompatible_cache_schedules_a_new_baseline(car_profiles, tmp_path, change):
    import law_sweep

    path = tmp_path / 'baseline.json'
    _, jobs, key = _plan(path, 'rb1')
    assert [job[0] for job in jobs] == ['pure_pursuit', 'stanley']
    rows = [{'track': 'test_track', 'seed': 7, 'steering_law': 'pure_pursuit'}]
    law_sweep.write_baseline_cache(path, key, rows)
    car = 'rb1'
    if change == 'car':
        car_profiles.clone('rb1', 'rb2')
        car = 'rb2'
    elif change == 'calibration':
        car_profiles.set_calibration('rb1', steering_angle_to_servo_offset=0.4)
    else:
        path.write_text(json.dumps(rows))
    cached, jobs, new_key = _plan(path, car)
    if change == 'car':
        # Cloned hardware still uses rb2's measured scales (SLAM_SPEED_FIT.md).
        # ...and rb2's measured body and mass (calibration.RB2_MEASURED, 2026-11-23).
        from sim_fidelity.calibration import rb2_measured
        rb1 = law_sweep.calibration_for_car('rb1')
        expected = replace(rb1, odom_speed_scale=1.0261486728316354,
                           speed_command_scale=1.0108904807240708, **rb2_measured(rb1))
        assert law_sweep.calibration_for_car('rb2') == expected
        values = {'plant': asdict(expected),
                  'controller': law_sweep.effective_config(
                      'pure_pursuit.yaml', 'pure_pursuit_node', 'rb2')}
        digest = hashlib.sha256(json.dumps(
            law_sweep._json_safe(values), sort_keys=True, separators=(',', ':'),
            allow_nan=False).encode('utf-8')).hexdigest()
        assert new_key['calibration'] == digest
        assert key['calibration'] != new_key['calibration']
    assert cached == []
    # This is the actual work list main submits to the worker pool: rejecting
    # a cache must recompute pure pursuit, not merely omit the old rows.
    assert [job[0] for job in jobs] == ['pure_pursuit', 'stanley']
    assert all(job[-1] == car for job in jobs)
    if change == 'car':
        # Even a matching real calibration digest cannot replace car identity.
        law_sweep.write_baseline_cache(path, dict(new_key, car='rb1'), rows)
        cached, jobs, _ = _plan(path, 'rb2')
        assert cached == []
        assert [job[0] for job in jobs] == ['pure_pursuit', 'stanley']
        assert all(job[-1] == 'rb2' for job in jobs)


def test_matching_cache_reuses_baseline(car_profiles, tmp_path):
    import law_sweep

    path = tmp_path / 'baseline.json'
    _, _, key = _plan(path, 'rb1')
    rows = [{'track': 'test_track', 'seed': 7, 'steering_law': 'pure_pursuit'}]
    law_sweep.write_baseline_cache(path, key, rows)
    cached, jobs, same_key = _plan(path, 'rb1')
    assert same_key == key
    assert cached == rows
    assert [job[0] for job in jobs] == ['stanley']
