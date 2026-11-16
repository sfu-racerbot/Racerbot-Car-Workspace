"""The floor grip scale (GripEnvelope.scale, ProgressiveGrip.scale, FidelityPlant.set_grip_scale, run_pure_solo's
grip_schedule; 2026-11-14): a floor some fraction as grippy as calibrated, from the start or mid-run.

Oracles: closed forms of the two grip models with the scale applied, and in closed loop the invariant that a slicker
floor delivers less lateral acceleration (yaw rate times speed) on the same hard turn. Run by the tool-suite command.
"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import eval_gap  # noqa: E402
from sim_fidelity import calibration_for_car, with_floor  # noqa: E402
from sim_fidelity.grip import GripEnvelope, ProgressiveGrip  # noqa: E402

CURVE = ((0.0, 0.0), (2.0, 2.0), (3.0, 2.8), (4.0, 3.2))


def test_progressive_grip_delivers_scale_times_the_curve():
    grip = ProgressiveGrip(CURVE, 0.36, scale=0.85)
    for asked, delivered in CURVE:
        assert grip.deliver(asked) == pytest.approx(0.85 * delivered, abs=1e-12)   # m/s^2
    assert grip.deliver(10.0) == pytest.approx(0.85 * 3.2, abs=1e-12)


def test_progressive_grip_apply_holds_the_scaled_delivery():
    # At 3 m/s, ask 4.0 m/s^2: the steering returned turns at exactly the scaled delivery, v^2 tan(d) / L.
    grip = ProgressiveGrip(CURVE, 0.36, scale=0.85)
    asked_steer = math.atan(4.0 * 0.36 / 9.0)
    out = grip.apply(asked_steer, 3.0)
    assert 9.0 * math.tan(out) / 0.36 == pytest.approx(0.85 * 3.2, abs=1e-9)


def test_the_envelope_shrinks_the_friction_circle():
    env = GripEnvelope(friction_accel_limit=5.0, wheelbase=0.36, longitudinal_coupling=0.0, scale=0.8)
    assert env.lateral_budget(0.0) == pytest.approx(4.0, abs=1e-12)          # m/s^2
    out = env.apply(0.4, 3.0, 0.0)
    assert 9.0 * math.tan(out) / 0.36 == pytest.approx(4.0, abs=1e-9)


def _hard_turn_delivery(scale_at, seconds=1.5):
    """Car 2 on its race-day floor, 2.6 m/s, wheels at 0.35 rad: median of yaw rate x speed over the last 0.5 s."""
    cal = with_floor('sfu_course_2026_11_11_rb2', calibration_for_car('rb2'))
    plant = eval_gap.make_plant('car2_course_2026_11_11', 12345, calibration=cal, car_id='rb2')
    try:
        plant.reset(np.array([[-10.0, 8.0, 0.0]]), speeds=[2.6])
        out = []
        for k in range(int(seconds / eval_gap.CONTROL_DT)):
            if scale_at is not None and k == scale_at[0]:
                plant.set_grip_scale(scale_at[1])
            plant.step([(0.35, 2.6)])
            st = plant.truth(0)
            out.append(abs(st[5]) * st[3])
        return float(np.median(out[-int(0.5 / eval_gap.CONTROL_DT):]))
    finally:
        plant.close()


def test_a_slicker_floor_delivers_less_in_the_same_turn():
    full = _hard_turn_delivery(None)
    slick = _hard_turn_delivery((0, 0.85))
    assert slick < 0.92 * full, (slick, full)          # within a few % of 0.85x: the turn is grip-limited


def test_a_mid_run_cut_takes_effect_from_then_on():
    full = _hard_turn_delivery(None)
    late = _hard_turn_delivery((30, 0.85))             # cut at 0.75 s; judged over the last 0.5 s
    assert late < 0.92 * full, (late, full)


@pytest.mark.parametrize('bad', [0.0, -0.5, math.nan, math.inf])
def test_bad_scales_are_refused(bad):
    plant = eval_gap.make_plant('car2_course_2026_11_11', 12345, car_id='rb2')
    try:
        with pytest.raises(ValueError, match='grip scale'):
            plant.set_grip_scale(bad)
    finally:
        plant.close()


def test_run_pure_solo_applies_the_schedule():
    # invariant: the map racer on a floor at half its grip from the start is slower, or does not finish cleanly.
    import run_validation as rv
    track = eval_gap.track_path('room_2026_07_27')
    base = rv.run_pure_solo(track, 12345, 40.0, car_id='rb2', safety='map')
    half = rv.run_pure_solo(track, 12345, 40.0, car_id='rb2', safety='map', grip_schedule=((0.0, 0.5),))
    assert base['lap_time_s'] is not None, 'precondition: the full-grip lap finishes'
    assert half['lap_time_s'] is None or half['lap_time_s'] > base['lap_time_s'] + 0.2 or half['max_cross_track_m'] > base['max_cross_track_m'] + 0.05
