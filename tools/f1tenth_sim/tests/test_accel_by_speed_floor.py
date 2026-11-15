"""A floor's acceleration ceiling as a table against wheel speed (accel_by_speed), opt-in.

Oracles: linear interpolation closed form; the speed-loop formula written out in the test from the calibration's
own gain and ceiling; the absence of a table must leave every existing floor's speed loop exactly as before.
Run by: python3 -m pytest tools/f1tenth_sim/tests/ -q -p no:cacheprovider
"""
import sys
from dataclasses import replace
from pathlib import Path

import pytest

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from sim_fidelity import FLOORS, calibration_for_car, with_floor  # noqa: E402
from sim_fidelity.actuation import accel_table_ceiling, speed_loop_accel_ceiling  # noqa: E402

TABLE = ((0.0, 6.0), (2.0, 6.0), (5.0, 2.0))


def test_table_ceiling_is_linear_between_points_and_flat_beyond_the_ends():
    assert accel_table_ceiling(TABLE, 1.0) == pytest.approx(6.0, abs=1e-12)           # m/s^2
    assert accel_table_ceiling(TABLE, 3.5) == pytest.approx(4.0, abs=1e-12)           # halfway down the fall
    assert accel_table_ceiling(TABLE, 5.0) == pytest.approx(2.0, abs=1e-12)
    assert accel_table_ceiling(TABLE, 9.0) == pytest.approx(2.0, abs=1e-12)           # flat beyond the last
    assert accel_table_ceiling(TABLE, -1.0) == pytest.approx(6.0, abs=1e-12)          # and below the first


@pytest.mark.parametrize('bad,rule', [
    (((0.0, 4.0),), 'at least two'),
    (((0.0, 4.0), (0.0, 3.0)), 'ascending'),
    (((1.0, 4.0), (0.5, 3.0)), 'ascending'),
    (((0.0, 4.0), (2.0, 0.0)), 'positive'),
    (((0.0, 4.0), (2.0, -1.0)), 'positive'),
    (((0.0, 4.0), (2.0, float('nan'))), 'finite'),
])
def test_a_bad_table_is_refused_naming_its_rule(bad, rule):
    with pytest.raises(ValueError, match=rule):
        accel_table_ceiling(bad, 1.0)


def _plant(cal):
    import eval_gap
    return eval_gap.make_plant('car2_course_2026_11_06a_scans', 12345, cal)


@pytest.mark.parametrize('floor', sorted(FLOORS))
def test_without_a_table_every_existing_floors_speed_loop_is_what_it_was(floor):
    # Oracle: the speed-loop formula as it stood before the table existed, from the floor's own numbers.
    cal = with_floor(floor, calibration_for_car('rb2'))
    assert cal.accel_by_speed == ()
    plant = _plant(cal)
    try:
        for measured in (0.3, 1.0, 2.5, 4.0, 5.2, 6.0):
            for command in (measured + 0.4, measured + 3.0, measured - 0.5, measured - 4.0):
                diff = command - measured
                if diff >= 0.0:
                    accel = min(cal.speed_loop_accel_gain * diff, speed_loop_accel_ceiling(
                        cal.speed_loop_accel_max, cal.speed_loop_accel_fade_from, cal.speed_loop_accel_no_load, measured))
                    gain = 10.0 * plant.params.a_max / plant.params.v_max
                else:
                    accel = max(cal.speed_loop_brake_gain * diff, -cal.speed_loop_brake_max)
                    gain = 10.0 * plant.params.a_max / -plant.params.v_min
                assert plant._measured_speed_loop(command, measured) == measured + accel / gain      # bit-identical
    finally:
        plant.close()


def test_with_a_table_the_loops_ceiling_is_the_tables_value_at_the_wheel_speed():
    base = with_floor('sfu_course_2026_11_11_rb2', calibration_for_car('rb2'))
    cal = replace(base, accel_by_speed=TABLE, speed_loop_accel_gain=50.0)         # a gain high enough to saturate
    plant = _plant(cal)
    try:
        gain = 10.0 * plant.params.a_max / plant.params.v_max
        for true_speed in (0.5, 1.9, 3.0, 4.5):
            wheel = true_speed * cal.odom_speed_scale
            got = (plant._measured_speed_loop(true_speed + 4.0, true_speed) - true_speed) * gain
            assert got == pytest.approx(accel_table_ceiling(TABLE, wheel), abs=1e-9)         # m/s^2
        # braking is untouched by the table
        assert plant._measured_speed_loop(0.0, 3.0) == 3.0 + max(cal.speed_loop_brake_gain * -3.0,
                                                                  -cal.speed_loop_brake_max) / (
            10.0 * plant.params.a_max / -plant.params.v_min)
    finally:
        plant.close()
