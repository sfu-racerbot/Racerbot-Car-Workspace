"""start_replay's geometry helpers against closed forms (a circle of known radius).

Oracles: a counter-clockwise circle of radius 10 m centred on the origin, sampled every 0.5 degree, has arc
length 2*pi*10 and a point at radius r on the left of the direction of travel (inside) is r_line - r to the
left. Run by: python3 -m pytest tools/f1tenth_sim/tests/ -q -p no:cacheprovider
"""
import math
import sys
from pathlib import Path

import pytest

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM.parents[1] / 'tools'))
sys.path.insert(0, str(SIM.parents[1] / 'src' / 'pure_pursuit'))
import numpy as np  # noqa: E402
import start_replay as sr  # noqa: E402

R = 10.0


def circle():
    angle = np.radians(np.arange(0.0, 360.0, 0.5))
    return sr.line_geometry(np.column_stack([R * np.cos(angle), R * np.sin(angle)]))


def test_arc_length_and_projection_follow_the_circle():
    xy, seg, ds, arc = circle()
    assert ds.sum() == pytest.approx(2 * math.pi * R, abs=0.005)       # m, chord error of 0.5 deg steps
    s, off = sr.project(xy, seg, ds, arc, R * math.cos(math.radians(90)), R * math.sin(math.radians(90)))
    assert s == pytest.approx(R * math.pi / 2, abs=0.05)               # m, a quarter turn
    assert off == pytest.approx(0.0, abs=0.01)


def test_offset_is_positive_to_the_left_of_travel():
    xy, seg, ds, arc = circle()
    inside = sr.project(xy, seg, ds, arc, 9.0, 0.0)[1]               # nearer the centre: left of a CCW line
    outside = sr.project(xy, seg, ds, arc, 10.7, 0.0)[1]
    assert inside == pytest.approx(1.0, abs=0.01)    # m
    assert outside == pytest.approx(-0.7, abs=0.01)


def test_lap_times_are_the_gaps_between_start_finish_crossings():
    length = 100.0
    t = np.arange(0.0, 40.0, 0.025)
    arcs = (t * 10.0) % length                       # 10 m/s: a lap every 10 s
    laps = sr.lap_times(t, arcs, length)
    assert laps == pytest.approx([10.0, 10.0], abs=0.03)       # s; crossings at 10, 20, 30 s are two laps
    assert sr.lap_times(t[:300], arcs[:300], length) == []           # under one crossing: no lap


# --- option plumbing (2026-11-12) --------------------------------------------------------------
from dataclasses import replace  # noqa: E402
from sim_fidelity import calibration_for_car, with_floor  # noqa: E402
from sim_fidelity.calibration import sfu_2026_11_11_rb2_grip_curve  # noqa: E402
from sim_fidelity.grip import ProgressiveGrip  # noqa: E402

FLOOR = 'sfu_course_2026_11_11_rb2'


def test_defaults_leave_the_floor_untouched():
    assert sr.floor_calibration(FLOOR) == with_floor(FLOOR)
    assert sr.floor_calibration(FLOOR, None, None) == with_floor(FLOOR)


def test_grip_scale_replaces_the_floors_curve_with_the_measured_one_at_that_scale():
    cal = sr.floor_calibration(FLOOR, grip_scale=1.0)
    assert cal.grip_curve == sfu_2026_11_11_rb2_grip_curve(1.0)
    # Oracle: at scale 1.0 the delivered peak is the measured gyro-basis plateau, 2.94 m/s^2 (grip_truth.py).
    peak = max(d for _, d in cal.grip_curve)
    assert peak == pytest.approx(2.94, abs=0.005)       # m/s^2
    assert max(d for _, d in with_floor(FLOOR).grip_curve) == pytest.approx(2.94 * 1.25, abs=0.01)
    assert cal.grip_curve != with_floor(FLOOR).grip_curve
    assert replace(cal, grip_curve=with_floor(FLOOR).grip_curve) == with_floor(FLOOR)    # nothing else changed


def test_grip_scale_is_refused_on_a_floor_that_has_no_scalable_curve():
    with pytest.raises(ValueError, match='grip-scale'):
        sr.floor_calibration('sfu_course_2026_11_06', grip_scale=1.1)


def test_grip_coupling_reaches_the_plants_grip_object():
    import eval_gap
    cal = sr.floor_calibration(FLOOR, grip_coupling=0.5)
    assert cal.longitudinal_grip_coupling == 0.5
    plant = eval_gap.make_plant('car2_course_2026_11_11', 12345, cal)
    try:
        assert isinstance(plant.grip[0], ProgressiveGrip)
        assert plant.grip[0].longitudinal_coupling == 0.5
        assert plant.grip[0]._delivered == [d for _, d in with_floor(FLOOR).grip_curve]   # curve untouched
    finally:
        plant.close()


@pytest.mark.parametrize('bad', [-0.1, 1.5])
def test_grip_coupling_outside_zero_to_one_is_refused(bad):
    with pytest.raises(ValueError, match='grip-coupling'):
        sr.floor_calibration(FLOOR, grip_coupling=bad)


def test_online_lateral_reaches_the_controllers_shaper_limit():
    args = sr.parse_args(['--track', 't', '--floor', FLOOR, '--line', 'l', '--start=0,0,0', '--out', 'o'])
    assert args.lateral == 3.1                    # the race-day config's value, unchanged by default
    args = sr.parse_args(['--track', 't', '--floor', FLOOR, '--line', 'l', '--start=0,0,0', '--out', 'o',
                          '--online-lateral', '99'])
    kwargs = sr.solo_arguments(args.law, args.speed, args.lateral, 3, 0.42)
    assert kwargs['max_lateral_accel'] == 99.0
    import run_validation as rv
    shaper = rv.CommandShaper('rb2', kwargs['max_speed'], kwargs['max_lateral_accel'])
    assert shaper.max_lateral_accel == 99.0       # the object the run drives with


def test_summary_reports_speed_offset_and_where_the_run_ended():
    # A synthetic run round the 10 m circle at 3 m/s (wheel scale 1.0), 0.5 m outside the line, ending a
    # quarter turn round: arc length of the end = R * pi / 2, offset = -0.5 (to the right of a CCW line).
    from types import SimpleNamespace
    angle = np.radians(np.arange(0.0, 360.0, 0.5))
    plan = SimpleNamespace(xy=np.column_stack([R * np.cos(angle), R * np.sin(angle)]), speed=np.full(len(angle), 3.0))
    steps = 400
    theta = np.linspace(0.0, math.pi / 2, steps)
    trace = np.column_stack([np.arange(steps) * 0.025, (R + 0.5) * np.cos(theta), (R + 0.5) * np.sin(theta),
                             np.full(steps, 3.0), np.zeros(steps), np.full(steps, 3.1)])
    cal = SimpleNamespace(odom_speed_scale=1.0)
    result = dict(contact_steps=7, collision=True, passed=False, max_cross_track_m=0.5, laps=0.2)
    out = sr.summarise(cal, plan, trace, result, [5.0])
    assert out['seed_result']['end_arc'] == pytest.approx(R * math.pi / 2, abs=0.05)     # m
    assert out['seed_result']['end_offset_left'] == pytest.approx(-0.5, abs=0.01)        # m
    assert out['at'][5.0]['wheel_speed'] == pytest.approx(3.0, abs=1e-9)
    assert out['at'][5.0]['offset_left'] == pytest.approx(-0.5, abs=0.01)
    assert out['seed_result']['contact'] == 7
