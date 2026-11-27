"""Car 2's measured body in the simulator (docs/sim-calibration.md, 2026-11-23).

Runner: python3 -m pytest tools/f1tenth_sim/tests/ (the tool suites line in CLAUDE.md).
"""

import math
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_fidelity.calibration import calibration_for_car  # noqa: E402


def test_cg_follows_the_axle_weights():
    # Oracle: statics. Each axle carries the weight share of the CG's distance from the *other*
    # axle, so front/total = lr/L. Scale readings 1.673 kg front, 2.283 kg rear (2026-11-23).
    cal = calibration_for_car('rb2')
    assert cal.physical_wheelbase == pytest.approx(0.325, abs=1e-9)  # m, tape hub to hub
    assert cal.lf + cal.lr == pytest.approx(0.325, abs=1e-9)  # m
    assert cal.lr / cal.physical_wheelbase == pytest.approx(1.673 / (1.673 + 2.283), abs=1e-9)
    assert cal.mass == pytest.approx(1.673 + 2.283, abs=1e-9)  # kg


@pytest.mark.parametrize('angle', [0.0, 0.2, -0.3, 0.42])
def test_a_command_keeps_its_circle_on_the_real_axles(angle):
    # Oracle: the circle definition the October steering calibration used,
    # radius = 0.36 / tan(command) (docs/hardware-reference.md). On 0.325 m axles the wheels
    # must turn to the angle with that same radius, and so must be smaller unless straight.
    cal = calibration_for_car('rb2')
    physical = cal.physical_steering(angle)
    assert math.tan(physical) / 0.325 == pytest.approx(math.tan(angle) / 0.36, abs=1e-12)  # 1/m
    if angle:
        assert abs(physical) < abs(angle)


def test_a_car_without_a_measured_axle_distance_steers_unchanged():
    cal = calibration_for_car('rb1')
    assert cal.physical_wheelbase == cal.wheelbase
    for angle in (-0.3, 0.0, 0.25):
        assert cal.physical_steering(angle) == angle


def test_physical_wheelbase_must_be_positive_and_finite():
    cal = calibration_for_car('rb2')
    for bad in (0.0, -0.325, math.nan, math.inf):
        with pytest.raises(ValueError, match='physical_wheelbase must be finite and positive'):
            replace(cal, physical_wheelbase=bad)


def test_the_plant_turns_each_command_on_the_calibrated_circle():
    # Oracle: kinematic yaw rate u * tan(command) / 0.36 at a crawl (0.5 m/s, about
    # 0.15 m/s^2 lateral, where tyre slip is negligible). Driving the raw command on
    # 0.325 m axles instead would turn about 11% faster (0.36 / 0.325).
    pytest.importorskip('f1tenth_gym')
    import slam_rollout as sr
    cal = replace(calibration_for_car('rb2'), steering_delay_ticks=0, speed_delay_ticks=0)
    command, speed = 0.2, 0.5
    plant = sr.make_plant(cal)
    try:
        state = np.array([sr.ANCHOR[0] + cal.lr, sr.ANCHOR[1], cal.physical_steering(command),
                          speed, 0.0, 0.0, 0.0])
        plant.reset(state[None, [0, 1, 4]], speeds=[speed])
        obs, _ = plant.env.reset(options={'states': state[None, :]})
        plant._observe(obs)
        for _ in range(120):  # 3 s
            plant.step([(command, speed / cal.speed_command_scale)])
        truth = plant.truth(0)
    finally:
        plant.close()
    u = truth[3] * math.cos(truth[6])
    expected = u * math.tan(command) / 0.36
    assert truth[5] == pytest.approx(expected, rel=0.02)  # rad/s, 2%
