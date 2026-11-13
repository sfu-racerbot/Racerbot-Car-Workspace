"""The measured floor profile and the models it switches on (sim_fidelity, docs/sim-calibration.md).

The plain calibration must stay exactly as it was: every model here is off unless a floor
profile turns it on. Oracles are named on each test: a closed form, an invariant, a
cited measurement, or the plant's own geometry.
Run by: python3 -m pytest tools/f1tenth_sim/tests/ tools/racerbot_sim/tests/ tools/run_analysis/tests/ tools/dev/tests/ -q
"""
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
from sim_fidelity import FLOORS, calibration_for_car, with_floor, with_friction  # noqa: E402
from sim_fidelity.calibration import STEERING_MAX_LEFT_MEASURED, STEERING_MAX_RIGHT_MEASURED  # noqa: E402
from sim_fidelity.grip import KINEMATIC_SPEED, ProgressiveGrip  # noqa: E402
from sim_fidelity.sensing import OdomSensor, ScanSensor  # noqa: E402

FLOOR = "sfu_course_2026_10_30"
L = 0.36


# --- the progressive grip model --------------------------------------------------------

CURVE = ((0.0, 0.0), (2.0, 2.0), (4.0, 3.0), (8.0, 4.0))


def test_progressive_grip_interpolates_and_holds_past_the_table():
    g = ProgressiveGrip(CURVE, L)
    assert g.deliver(1.0) == pytest.approx(1.0, abs=1e-12)     # m/s^2
    assert g.deliver(3.0) == pytest.approx(2.5, abs=1e-12)
    assert g.deliver(50.0) == pytest.approx(4.0, abs=1e-12)


def test_progressive_grip_turns_the_car_as_hard_as_the_curve_says():
    # Oracle: a_lat = v^2 tan(delta) / L before and after; after = curve(before).
    g = ProgressiveGrip(CURVE, L)
    v, steering = 3.0, 0.2
    asked = v * v * math.tan(steering) / L
    out = g.apply(steering, v)
    assert v * v * math.tan(out) / L == pytest.approx(g.deliver(asked), abs=1e-9)
    assert g.apply(-steering, v) == pytest.approx(-out, abs=1e-12)      # symmetric
    assert g.limited_steps == 2


def test_progressive_grip_passes_through_below_the_curve_knee():
    g = ProgressiveGrip(CURVE, L)
    steering = math.atan(1.5 * L / 9.0)          # asks 1.5 m/s^2 at 3 m/s, on the identity part
    assert g.apply(steering, 3.0) == pytest.approx(steering, abs=1e-12)
    assert g.limited_steps == 0


def test_progressive_grip_leaves_slow_steering_alone():
    g = ProgressiveGrip(CURVE, L)
    assert g.apply(0.4, KINEMATIC_SPEED - 0.01) == 0.4


@pytest.mark.parametrize("curve", [((0.0, 0.0),), ((0.0, 0.0), (2.0, 2.0), (1.0, 3.0)),
                                   ((0.0, 0.0), (2.0, 2.0), (3.0, 1.0))])
def test_progressive_grip_refuses_a_curve_that_is_not_ascending(curve):
    with pytest.raises(ValueError, match="ascending"):
        ProgressiveGrip(curve, L)


FALLING = ((0.0, 0.0), (2.0, 2.0), (4.0, 3.0), (6.0, 2.0), (8.0, 2.4))


def test_progressive_grip_takes_a_falling_curve_only_when_told_to():
    # Oracle: docs/sim-calibration.md, 2026-11-11 rb2 grip: delivered lateral acceleration
    # peaks and then FALLS as more is asked. The default refusal (test above) stays.
    with pytest.raises(ValueError, match="ascending"):
        ProgressiveGrip(FALLING, L)
    g = ProgressiveGrip(FALLING, L, falling=True)
    assert g.deliver(4.0) == pytest.approx(3.0, abs=1e-12)     # m/s^2, the peak
    assert g.deliver(5.0) == pytest.approx(2.5, abs=1e-12)     # halfway down the fall
    assert g.deliver(6.0) == pytest.approx(2.0, abs=1e-12)
    assert g.deliver(50.0) == pytest.approx(2.4, abs=1e-12)    # holds the last value


def test_progressive_grip_falling_curve_turns_the_car_less_when_asked_for_more():
    # Oracle: a_lat = v^2 tan(delta) / L; asking 5 m/s^2 must deliver 2.5, below the 3.0 peak.
    g = ProgressiveGrip(FALLING, L, falling=True)
    v = 3.0
    at_peak = g.apply(math.atan(4.0 * L / (v * v)), v)
    past_peak = g.apply(math.atan(5.0 * L / (v * v)), v)
    assert v * v * math.tan(at_peak) / L == pytest.approx(3.0, abs=1e-9)
    assert v * v * math.tan(past_peak) / L == pytest.approx(2.5, abs=1e-9)
    assert past_peak < at_peak


def test_progressive_grip_falling_still_refuses_asked_that_does_not_ascend():
    with pytest.raises(ValueError, match="ascending"):
        ProgressiveGrip(((0.0, 0.0), (3.0, 2.0), (2.0, 2.5)), L, falling=True)


# --- the floor profile -----------------------------------------------------------------

def test_plain_calibration_keeps_every_floor_model_off():
    # Pins the behaviour every other test and the checked-in results were made with.
    # (The selected car's calibration is resolved per call; conftest selects each car in turn.)
    c = calibration_for_car()
    assert c.grip_curve is None and c.steering_linkage is None
    assert c.speed_command_scale == 1.0 and c.odom_speed_scale == 1.0
    assert c.lidar_noise_per_metre == 0.0 and c.lidar_self_hits == () and c.floor == ""


def test_with_floor_none_is_the_plain_calibration():
    base = calibration_for_car()
    assert with_floor(None, base) is base
    assert with_floor(None) == calibration_for_car()


def test_with_floor_refuses_an_unknown_floor():
    with pytest.raises(ValueError, match="unknown floor"):
        with_floor("ice_rink")


def test_with_floor_applies_its_overrides_and_friction_composes():
    cal = with_friction(0.3, with_floor(FLOOR))
    assert cal.floor == FLOOR and cal.surface_friction == 0.3
    for name, value in FLOORS[FLOOR].overrides.items():
        assert getattr(cal, name) == value


def test_floor_fit_and_held_out_runs_do_not_overlap():
    f = FLOORS[FLOOR]
    assert f.fit_runs and f.held_out_runs and not set(f.fit_runs) & set(f.held_out_runs)
    # The two runs the validation leans on are held out (docs/sim-calibration.md).
    assert {"20261030-184826", "20261030-175124"} <= set(f.held_out_runs)


def test_floor_records_a_source_for_every_value_it_sets():
    f = FLOORS[FLOOR]
    assert set(f.overrides) - {"floor"} <= set(f.sources)


def test_floor_grip_never_delivers_more_than_was_asked():
    # Invariant: a tyre cannot turn the car harder than the rack's geometry asks.
    curve = FLOORS[FLOOR].overrides["grip_curve"]
    assert all(delivered <= asked + 1e-12 for asked, delivered in curve)
    ProgressiveGrip(curve, L)                     # and it is a usable curve


def test_floor_grip_rolls_off_like_the_measured_car():
    # Cited measurement: car_measure.py on the fit runs, 2026-10-30. Asked 3.25 got 2.62,
    # asked 7 got 3.7. The floor curve is that times 1.15 (fit-plant), so it must sit
    # 1.15 above those points, well under what was asked.
    g = ProgressiveGrip(FLOORS[FLOOR].overrides["grip_curve"], L)
    assert g.deliver(3.25) == pytest.approx(1.15 * 2.62, abs=0.01)
    assert g.deliver(7.0) == pytest.approx(1.15 * 3.7, abs=0.01)
    assert g.deliver(1.0) == pytest.approx(1.0, abs=1e-12)


def test_floor_linkage_meets_the_measured_circles():
    # Cited measurement: docs/hardware-reference.md, 2026-10-16. Full stick 0.436 rad drove
    # 1.33 m inner circles both sides, 24.2 deg; the servo limits reach the calibration's
    # measured full-lock angles.
    commands, angles = zip(*FLOORS[FLOOR].overrides["steering_linkage"])
    angle = lambda c: float(np.interp(c, commands, angles))
    assert angle(0.436) == pytest.approx(math.radians(24.2), abs=0.002)
    assert angle(-0.436) == pytest.approx(-math.radians(24.2), abs=0.002)
    assert angle(10.0) == pytest.approx(STEERING_MAX_LEFT_MEASURED, abs=0.001)
    assert angle(-10.0) == pytest.approx(STEERING_MAX_RIGHT_MEASURED, abs=0.001)
    assert angle(0.0) == pytest.approx(0.0105, abs=1e-12)        # the measured trim


def test_floor_command_scale_follows_from_the_vesc_gains():
    # Closed form: vesc.yaml commands at 4614 eRPM per m/s, /odom reads at 4447.98, and the
    # wheels over-read true speed by 1.033 (car_measure.py, fit and held-out runs alike).
    cal = with_floor(FLOOR)
    assert cal.speed_command_scale == pytest.approx(4614.0 / 4447.983786 / 1.033, abs=5e-4)
    assert cal.odom_speed_scale == pytest.approx(1.033, abs=1e-12)


# --- sensors ----------------------------------------------------------------------------

def test_odom_sensor_reads_the_over_read():
    s = OdomSensor(0.0, np.random.default_rng(0), scale=1.033)
    assert s.update(3.0) == pytest.approx(3.099, abs=1e-12)     # m/s
    assert OdomSensor(0.0, np.random.default_rng(0)).update(3.0) == 3.0


def test_scan_sensor_writes_the_self_hits_exactly():
    s = ScanSensor(0, 0.0, np.random.default_rng(0), self_hits=((0, 0.15), (2, 0.2), (99, 0.1)))
    out = s.update(np.full(5, 4.0))
    assert out.tolist() == [0.15, 4.0, 0.2, 4.0, 4.0]          # beam 99 does not exist; ignored


def test_scan_sensor_noise_grows_with_range():
    s = ScanSensor(0, 0.0, np.random.default_rng(1), noise_per_metre=0.01)
    near = np.std([s.update(np.full(200, 1.0)) for _ in range(50)])
    far = np.std([s.update(np.full(200, 8.0)) for _ in range(50)])
    assert near == pytest.approx(0.01, rel=0.05)                # 1 cm at 1 m
    assert far == pytest.approx(0.08, rel=0.05)                 # 8 cm at 8 m


def test_scan_sensor_without_the_new_options_is_unchanged():
    s = ScanSensor(0, 0.0, np.random.default_rng(1))
    assert s.update(np.full(4, 2.0)).tolist() == [2.0] * 4


# --- in the plant (needs the gym checkout, .sim/) -----------------------------------------

@pytest.fixture(scope="module")
def eval_gap():
    import eval_gap as module
    return module


def test_plant_steers_through_the_floor_linkage(eval_gap):
    # Oracle: the linkage table. A held zero command settles the wheels at the trim, at a
    # crawl where grip does not act; the plain calibration settles them at zero.
    for cal, expected in ((with_floor(FLOOR), 0.0105), (calibration_for_car(), 0.0)):
        plant = eval_gap.make_plant("course_2026_10_30", 1, cal)
        try:
            line = plant.env.unwrapped.track.raceline
            plant.reset(np.array([[line.xs[0], line.ys[0], line.yaws[0]]]))
            for _ in range(40):
                plant.step([(0.0, 0.3)])
            assert plant.truth(0)[2] == pytest.approx(expected, abs=1e-4)     # rad
        finally:
            plant.close()


def test_plant_judges_contact_on_the_contact_walls(eval_gap):
    from PIL import Image
    plant = eval_gap.make_plant("course_2026_10_30_scans", 1, with_floor(FLOOR))
    try:
        field = plant._contact_dt
        walls = np.array(Image.open(eval_gap.track_path("course_2026_10_30_scans") + "_contact.png").convert("L")) < 128
        # Laid out as gym lays out its own map, bottom row first: zero exactly on wall cells.
        assert field is not None and np.array_equal(field == 0.0, walls[::-1])
    finally:
        plant.close()
    plain = eval_gap.make_plant("course_2026_10_30", 1, calibration_for_car())
    try:
        assert plain._contact_dt is None
    finally:
        plain.close()


def test_plant_top_speed_is_the_vesc_clamp(eval_gap):
    # Closed form: the selected car's vesc.yaml speed_max / speed_to_erpm_gain commanded (car 2:
    # 23250 / 4614 = 5.039 m/s; car 1: 23250 / 4600 = 5.054 m/s), times the floor's command
    # scale; /odom reads that times 1.033.
    from racerbot_car.layers import effective_config
    from run_analysis.fit_sim import simulate_commands
    vesc = effective_config('vesc.yaml', '/**', os.environ['RACERBOT_CAR'])
    cal = with_floor(FLOOR)
    speed, _ = simulate_commands(np.column_stack([np.zeros(400), np.full(400, 8.0)]), cal)
    top = vesc['speed_max'] / abs(vesc['speed_to_erpm_gain'])
    assert speed[-1] == pytest.approx(top * cal.speed_command_scale, abs=0.01)   # m/s
