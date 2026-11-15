"""eval_gap.run_track's max_laps and lap_times_s (2026-11-14): standing and flying laps reported apart.

Oracle: invariants. The laps' own times sum to the run's simulated time (one control tick, 0.025 s), there is one
per completed lap, and a flying lap (no standing start) is faster than the first. Run by the tool-suite command.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import eval_gap  # noqa: E402
import gap_mirror  # noqa: E402


@pytest.fixture(scope='module')
def two_laps():
    return eval_gap.run_track('indoor_oval', gap_mirror.node_params(car_id='rb2'), 12345, 60.0, car_id='rb2', max_laps=2)


def test_one_time_per_completed_lap(two_laps):
    assert two_laps['laps'] == 2.0
    assert len(two_laps['lap_times_s']) == 2


def test_the_laps_add_up_to_the_run(two_laps):
    assert sum(two_laps['lap_times_s']) == pytest.approx(two_laps['sim_time_s'], abs=0.03)   # s, one tick + rounding


def test_the_flying_lap_is_faster_than_the_standing_one(two_laps):
    standing, flying = two_laps['lap_times_s']
    assert flying < standing


def test_the_default_is_still_one_lap():
    r = eval_gap.run_track('indoor_oval', gap_mirror.node_params(car_id='rb2'), 12345, 60.0, car_id='rb2')
    assert r['laps'] == 1.0 and len(r['lap_times_s']) == 1
    assert r['lap_times_s'][0] == pytest.approx(r['sim_time_s'], abs=0.03)   # s


@pytest.mark.parametrize('kwargs, match', [(dict(max_laps=0), 'max_laps'), (dict(max_laps=2, reverse=True), 'reverse')])
def test_bad_lap_counts_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        eval_gap.run_track('indoor_oval', gap_mirror.node_params(car_id='rb2'), 12345, 60.0, car_id='rb2', **kwargs)
