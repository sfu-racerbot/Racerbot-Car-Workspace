"""The explorer against gap_follow, closed loop in the simulator (explore_sim.py).

The verdicts come from the plant, never from the explorer or gap_follow
(TEST_QUALITY_STANDARDS A7): contact is FidelityPlant's own chassis-against-
map check, and the grip the explorer must run into is planted in the plant
(``surface_friction`` 0.32, about 3.1 m/s^2), which the explorer never reads.
Run with the tool suites:

    python3 -m pytest tools/f1tenth_sim/tests/test_explore_sim.py -q
"""

from dataclasses import replace
from functools import lru_cache

import explore_sim
from car_explorer.explorer import GAP_FOLLOW_MAX_LATERAL, LATERAL, Rules
from sim_fidelity.calibration import calibration_for_car

TRACK = 'car2_course_2026_11_11'


@lru_cache(maxsize=None)
def session(friction):
    explore_sim.eval_gap.track_path(TRACK)
    truth = replace(calibration_for_car('rb2'), surface_friction=friction)
    return explore_sim.run(TRACK, truth, 'rb2', minutes=3.0, rules=Rules(segment_sec=10.0))


def test_on_a_slippery_floor_the_explorer_finds_the_slide_and_never_touches_a_wall():
    report, _ = session(0.32)
    assert report['contact_steps'] == 0
    assert any(e['kind'] == 'slide' for e in report['events'])
    # It stopped raising cornering well short of gap_follow's own top.
    assert report['limits'][LATERAL] < GAP_FOLLOW_MAX_LATERAL[1]
    assert report['setting'][LATERAL] < report['limits'][LATERAL]


def test_on_a_grippy_floor_it_keeps_raising_with_no_false_limit():
    report, _ = session(0.55)  # the plain calibration's 5.4 m/s^2: no limit in reach
    assert report['contact_steps'] == 0
    assert not any(e['kind'] == 'slide' for e in report['events'])
    assert report['raises'] >= 8


def test_the_recording_holds_what_the_fitter_needs():
    _, rec = session(0.32)
    assert set(rec) >= {'t', 'steer', 'speed_cmd', 'speed', 'yaw_rate'}
    n = len(rec['t'])
    assert n > 1000 and all(len(rec[k]) == n for k in rec)
    assert rec['speed'].max() > 3.0
