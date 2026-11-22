"""A cached baseline is usable only for its car and effective calibration."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _plan(cache, car):
    import law_sweep

    return law_sweep.prepare_jobs(
        ['pure_pursuit', 'stanley'], ['test_track'], [7], {}, 60.0, car, cache)


@pytest.mark.parametrize('change', ['car', 'calibration', 'legacy'])
def test_incompatible_cache_schedules_a_new_baseline(car_profiles, tmp_path, change, monkeypatch):
    import law_sweep
    from dataclasses import replace

    calibration = law_sweep.calibration_for_car
    monkeypatch.setattr(law_sweep, 'calibration_for_car', lambda car: replace(
        calibration(car), odom_speed_scale=1.0, speed_command_scale=1.0))

    path = tmp_path / 'baseline.json'
    _, jobs, key = _plan(path, 'rb1')
    assert [job[0] for job in jobs] == ['pure_pursuit', 'stanley']
    rows = [{'track': 'test_track', 'seed': 7, 'steering_law': 'pure_pursuit'}]
    law_sweep.write_baseline_cache(path, key, rows)
    car = 'rb1'
    if change == 'car':
        # Same calibration, different identity: comparing hashes alone is
        # insufficient. Oracle: explicit car identity is part of the key.
        car_profiles.clone('rb1', 'rb2')
        car = 'rb2'
    elif change == 'calibration':
        car_profiles.set_calibration('rb1', steering_angle_to_servo_offset=0.4)
    else:
        import json
        path.write_text(json.dumps(rows))
    cached, jobs, new_key = _plan(path, car)
    if change == 'car':
        assert key['calibration'] == new_key['calibration']
    assert cached == []
    # This is the actual work list main submits to the worker pool: rejecting
    # a cache must recompute pure pursuit, not merely omit the old rows.
    assert [job[0] for job in jobs] == ['pure_pursuit', 'stanley']
    assert all(job[-1] == car for job in jobs)


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
