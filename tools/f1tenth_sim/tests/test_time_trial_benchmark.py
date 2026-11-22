"""Paired three-lap comparison: a contact or unfinished run cannot be a win."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time_trial_benchmark as bench  # noqa: E402


@pytest.mark.parametrize('laps,collision,status', [(3, False, 0), (2, False, 1), (3, True, 1)])
def test_paired_runs_retain_failures_and_change_only_requested_parameter(
        tmp_path, monkeypatch, laps, collision, status):
    def simulate(job):
        car, arm, track, _, seed, _ = job
        return dict(car=car, arm=arm, track=track, seed=seed, laps=laps, collision=collision,
                    lap_times_s=[8.0] * laps, contact_steps=int(collision),
                    min_chassis_map_clearance_m=0.1, stop_steps=0, max_stuck_steps=0)

    monkeypatch.setattr(bench, 'COURSES', [('indoor_oval', None)])
    monkeypatch.setattr(bench, 'ProcessPoolExecutor', ThreadPoolExecutor)
    monkeypatch.setattr(bench, 'run', simulate)
    out = tmp_path / 'result.json'
    assert bench.main(['--arm', 'before:open_aim_min_width=0', '--arm', 'after:open_aim_min_width=1.24',
                       '--car', 'rb2', '--first-seed', '15', '--output', str(out)]) == status
    report = json.loads(out.read_text())
    assert [(r['arm'], r['seed'], r['laps'], r['collision']) for r in report['runs']] == [
        (arm, seed, laps, collision) for arm in ('after', 'before') for seed in (15, 16, 17)]
    params = report['params']['rb2']
    assert params['before']['open_aim_min_width'] == 0
    assert params['after'] == params['before'] | {'open_aim_min_width': 1.24}


def test_unknown_override_is_not_silently_benchmarked(tmp_path):
    with pytest.raises(SystemExit) as error:
        bench.main(['--arm', 'typo:steerng_gain=0.9', '--output', str(tmp_path / 'bad.json')])
    assert error.value.code == 2
