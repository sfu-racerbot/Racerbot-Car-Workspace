"""Plan Task 11: the simulator agrees with race day (2026-11-11) on car 1's gap follow.

Recorded measurement: rb1 gap follow, default config, run 20261111-153721 lapped the race-day course in 10.8-12.2 s,
about 11.5 s typical (docs/superpowers/plans/2026-11-12-post-race-speed-program.md section 1.1). Oracle: the simulated
median lap of car 1's shipped gap follow on car2_course_2026_11_11 (car 1's calibration, with its fitted speed loop,
cars/rb1/sim/speed_loop.yaml) is within 4% of 11.5 s, with no contact, which is the plan's agreement band. On
2026-11-14 it was 11.74 s mean over these 6 seeds. Car 2's arms (map racer 10.05 s, time trial 11.9 s) are not pinned
here yet. Run by the tool-suite command in CLAUDE.md.
"""
import os
import sys
import statistics

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src', 'gap_follow'))
import eval_gap  # noqa: E402
import gap_mirror  # noqa: E402
from sim_fidelity import calibration_for_car  # noqa: E402

REAL_LAP_S = 11.5        # rb1 20261111-153721, typical lap (plan section 1.1)
BAND = 0.04              # the plan's agreement band
SEEDS = range(12345, 12351)


@pytest.fixture(scope='module')
def rb1_laps():
    params = gap_mirror.node_params(car_id='rb1')
    calibration = calibration_for_car('rb1')
    return [eval_gap.run_track('car2_course_2026_11_11', params, seed, 60.0, calibration=calibration, car_id='rb1')
            for seed in SEEDS]


def test_car_1_gap_follow_laps_the_race_day_course_within_4_percent_of_race_day(rb1_laps):
    assert all(r['passed'] for r in rb1_laps)
    assert sum(r['contact_steps'] for r in rb1_laps) == 0
    median = statistics.median(r['sim_time_s'] for r in rb1_laps)
    assert abs(median - REAL_LAP_S) <= BAND * REAL_LAP_S, median
