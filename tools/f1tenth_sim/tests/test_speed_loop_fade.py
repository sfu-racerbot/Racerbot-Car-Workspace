"""The speed loop's acceleration ceiling falling with speed above the measured range (2026-11-09).

Run with the tool suites: python3 -m pytest tools/f1tenth_sim/tests/ -q (CLAUDE.md).
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_fidelity import FLOORS, calibration_for_car, with_floor  # noqa: E402
from sim_fidelity.actuation import speed_loop_accel_ceiling  # noqa: E402


@pytest.mark.parametrize('speed', [0.0, 3.0, 5.4])
def test_up_to_the_fade_speed_the_ceiling_is_the_measured_one(speed):
    # oracle: spec, accel_max unchanged up to fade_from (the measured, traction-limited range).
    assert speed_loop_accel_ceiling(4.0, 5.4, 16.0, speed) == 4.0


@pytest.mark.parametrize('speed, expected', [
    (7.0, 4.0 * (16.0 - 7.0) / (16.0 - 5.4)),     # 3.396 m/s^2
    (10.7, 4.0 * (16.0 - 10.7) / (16.0 - 5.4)),   # 2.0 m/s^2, half way
    (16.0, 0.0),
    (20.0, 0.0),
])
def test_above_it_the_ceiling_falls_linearly_to_zero_at_no_load(speed, expected):
    # oracle: closed form, accel_max * (no_load - v) / (no_load - fade_from), never negative.
    assert speed_loop_accel_ceiling(4.0, 5.4, 16.0, speed) == pytest.approx(expected, abs=1e-12)  # m/s^2


def test_infinite_fade_speed_never_changes_the_ceiling():
    # oracle: spec, the default calibration (inf) is the unchanged plant.
    for speed in (0.0, 5.4, 7.0, 50.0):
        assert speed_loop_accel_ceiling(4.0, math.inf, math.inf, speed) == 4.0


@pytest.mark.parametrize('no_load', [5.0, 5.4, math.inf, float('nan')])
def test_a_no_load_speed_at_or_below_the_fade_speed_is_refused(no_load):
    with pytest.raises(ValueError, match='no_load must be finite and above fade_from'):
        speed_loop_accel_ceiling(4.0, 5.4, no_load, 7.0)


def test_the_plant_speed_loop_accelerates_at_the_faded_ceiling():
    # oracle: closed form through the real plant. _measured_speed_loop turns a wanted acceleration into
    # gym's speed command: measured + accel / gym_gain, gym_gain = 10 * a_max / v_max (gym's pid_accl).
    # A command far above the measured speed asks for the ceiling: 4.0 at 3 m/s, 3.396 at 7 m/s.
    import eval_gap
    from dataclasses import replace
    cal = replace(with_floor('sfu_course_2026_11_06', calibration_for_car('rb2')), speed_max=9.0, speed_min=-9.0)
    plant = eval_gap.make_plant('indoor_oval', 12345, cal)
    try:
        gain = 10.0 * plant.params.a_max / plant.params.v_max
        for measured, accel in ((3.0, 4.0), (7.0, 4.0 * (16.0 - 7.0) / (16.0 - 5.4))):
            got = plant._measured_speed_loop(20.0, measured)
            assert (got - measured) * gain == pytest.approx(accel, abs=1e-9)  # m/s^2
    finally:
        plant.close()


def test_the_6_nov_floor_fades_above_its_measured_range_and_says_it_is_estimated():
    # oracle: sim-calibration.md, the 2026-11-06 car reached 5.4 m/s on its wheels with no fall-off seen.
    cal = with_floor('sfu_course_2026_11_06', calibration_for_car('rb2'))
    assert cal.speed_loop_accel_fade_from == 5.4
    assert cal.speed_loop_accel_no_load == 16.0
    sources = FLOORS['sfu_course_2026_11_06'].sources
    assert sources['speed_loop_accel_fade_from'].startswith('ESTIMATED')
    assert sources['speed_loop_accel_no_load'].startswith('ESTIMATED')
    # the base calibration keeps no fall-off
    base = calibration_for_car('rb2')
    assert math.isinf(base.speed_loop_accel_fade_from) and math.isinf(base.speed_loop_accel_no_load)
