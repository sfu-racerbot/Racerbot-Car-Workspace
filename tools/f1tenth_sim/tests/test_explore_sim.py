"""The explorer against gap_follow, closed loop in the simulator (explore_sim.py).

The verdicts come from the plant, never from the explorer or gap_follow
(TEST_QUALITY_STANDARDS A7): contact is FidelityPlant's own chassis-against-
map check, and the numbers the fit must find are planted in the plant
(``surface_friction``, ``speed_loop_brake_max``, ``steering_delay_ticks``),
which the explorer never reads. Run with the tool suites:

    python3 -m pytest tools/f1tenth_sim/tests/test_explore_sim.py -q
"""

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

import explore_sim
from car_explorer.explorer import (GAP_FOLLOW_MAX_LATERAL, GAP_FOLLOW_MAX_SPEED, LATERAL, SPEED,
                                   Fence, Rules)
from sim_fidelity import explorer_fit as ef
from sim_fidelity.calibration import calibration_for_car

TRACK = 'car2_course_2026_11_11'


def planted(friction):
    """The car's own calibration with grip, braking and steering delay moved
    away from it: the session must show the fit the moved values."""
    base = calibration_for_car('rb2')
    return base, replace(base, surface_friction=friction, speed_loop_brake_max=3.0,
                         steering_delay_ticks=base.steering_delay_ticks + 2)


@lru_cache(maxsize=None)
def session(friction, minutes=3.0, plant_all=False):
    explore_sim.eval_gap.track_path(TRACK)
    truth = (planted(friction)[1] if plant_all
             else replace(calibration_for_car('rb2'), surface_friction=friction))
    return explore_sim.run(TRACK, truth, 'rb2', minutes=minutes, rules=Rules(segment_sec=10.0))


def test_on_a_slippery_floor_the_explorer_finds_the_slide_and_never_touches_a_wall():
    report, _ = session(0.32)
    assert report['contact_steps'] == 0
    slides = [e for e in report['events'] if e['kind'] == 'slide']
    assert slides
    # It stopped raising cornering well short of gap_follow's own top, and
    # stepped it down from where it slid (never below the fence's start: with
    # the mapping lap first, the first slide can come at the starting notch).
    fence = Fence()
    assert report['limits'][LATERAL] < GAP_FOLLOW_MAX_LATERAL[1]
    assert report['setting'][LATERAL] == pytest.approx(  # m/s^2
        max(fence.lateral_start,
            report['limits'][LATERAL] - fence.back_off_steps * fence.lateral_step), abs=1e-9)
    # Once cornering was held, the next slide lowered speed: speed stopped well
    # short of gap_follow's top instead of climbing on through slide after slide
    # (blaming cornering every time took it to 5 m/s through 14 slides).
    assert report['limits'][SPEED] < GAP_FOLLOW_MAX_SPEED[1]
    assert report['setting'][SPEED] < report['limits'][SPEED]
    assert len(slides) <= 3


def test_on_a_grippy_floor_it_keeps_raising_with_no_false_limit():
    report, _ = session(0.55)  # the plain calibration's 5.4 m/s^2: no limit in reach
    assert report['contact_steps'] == 0
    assert not any(e['kind'] == 'slide' for e in report['events'])
    assert report['raises'] >= 8


def test_it_maps_the_course_first_then_stop_tests_where_there_is_room_and_restores_gap_follow():
    report, rec = session(0.55)
    assert report['course']['learned'] and report['brake_tests'] >= 1
    assert report['contact_steps'] == 0
    # Every stop test lowered min_speed and put gap_follow's own back afterwards.
    shipped = rec['min_speed'][0]
    assert rec['min_speed'].min() < shipped and rec['min_speed'][-1] == shipped
    # A stop test really braked: the car fell from above 2.5 m/s to under 1 m/s.
    assert rec['speed'].max() > 2.5 and (rec['speed'][rec['t'] > 30.0] < 1.0).any()


def test_the_recording_holds_what_the_fitter_needs():
    # The grippy floor: on the slippery one the explorer now (rightly) holds
    # speed at 3.0 m/s once it slides, so the fast windows come from here.
    _, rec = session(0.55)
    assert set(rec) >= {'t', 'steer', 'speed_cmd', 'speed', 'yaw_rate'}
    n = len(rec['t'])
    assert n > 1000 and all(len(rec[k]) == n for k in rec)
    assert rec['speed'].max() > 3.0


@pytest.mark.parametrize('name', ['surface_friction', 'steering_delay_ticks',
                                  'speed_loop_brake_max'])
def test_the_explorers_own_session_pins_down_each_planted_number(name):
    """The oracle is the planted value. Profile each number around it with the
    others at their planted values: the session's windows must put the lowest
    replay error on it (a delay exactly, the others within one 5% step)."""
    _, truth = planted(0.50)
    _, rec = session(0.50, minutes=4.0, plant_all=True)
    windows = ef.windows_from_series(rec['t'], rec['steer'], rec['speed_cmd'], rec['speed'],
                                     rec['yaw_rate'], np.ones(len(rec['t']), bool),
                                     horizon=2.0, spacing=6.0, min_speed=1.0)
    value = getattr(truth, name)
    whole = ef.PARAMETERS[name][2]
    grid = (ef.profile_grid(truth, name) if whole
            else [value * f for f in (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2)])
    best, _, _ = ef.interval(grid, ef.profile(truth, windows, name, grid, workers=4),
                             whole=whole, resamples=50)
    if whole:
        assert best == value
    else:
        assert best == pytest.approx(value, rel=0.05)  # one 5% grid step
