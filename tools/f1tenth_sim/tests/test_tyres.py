"""Saturating tyres (sim_fidelity.tyres): gym's model below the limit, loss of control past it.

Runner: ``rb python3 -m pytest tools/f1tenth_sim/tests/test_tyres.py`` (the tool-suites line in CLAUDE.md).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sim_fidelity.bootstrap import bootstrap  # noqa: E402

bootstrap()
from f1tenth_gym.envs.dynamic_models.single_track import vehicle_dynamics_st  # noqa: E402
from sim_fidelity.tyres import TyreModel, axle_force_ratio, make_dynamics  # noqa: E402

# Gym's VehicleParameters order with rb1's calibrated values (calibration.CarCalibration defaults, 2026-11-14):
# mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max.
PARAMS = np.array([0.55, 4.718, 5.4562, 0.18, 0.18, 0.074, 3.74, 0.04712,
                   -0.4189, 0.4189, -3.2, 3.2, 7.319, 9.51, -5.0, 20.0])


def test_below_the_limit_it_is_gyms_model():
    # oracle: gym's own vehicle_dynamics_st; with peaks far above any force reached and small angles the magic formula
    # is its linear tyre and the exact slip angles its small-angle ones, so the two agree to the small-angle error.
    f = make_dynamics(TyreModel(50.0, 50.0, 1.5))
    rng = np.random.default_rng(3)
    for _ in range(100):
        x = np.array([0.0, 0.0, rng.uniform(-0.05, 0.05), rng.uniform(1.0, 4.0), 0.3,
                      rng.uniform(-0.3, 0.3), rng.uniform(-0.02, 0.02)])
        u = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-2.0, 2.0)])
        np.testing.assert_allclose(f(x, u, PARAMS), vehicle_dynamics_st(x, u, PARAMS), rtol=0.01, atol=0.01)


def test_the_force_peaks_at_the_peak_and_slides_at_sin_c_pi_over_2():
    # oracle: closed form of F/Fz = D sin(C atan(B alpha)) with B C D = slope: peak D where C atan(B alpha) = pi/2,
    # slope at zero slip, and the sliding limit D sin(C pi / 2) as alpha -> pi/2 with large B.
    slope, peak, shape = 0.55 * 5.4562, 0.48, 1.6
    b = slope / (shape * peak)
    at_peak = math.tan(math.pi / (2.0 * shape)) / b
    assert axle_force_ratio(at_peak, slope, peak, shape) == pytest.approx(peak, abs=1e-9)
    assert axle_force_ratio(1e-6, slope, peak, shape) / 1e-6 == pytest.approx(slope, rel=1e-6)
    assert axle_force_ratio(-at_peak, slope, peak, shape) == pytest.approx(-peak, abs=1e-9)
    slid = axle_force_ratio(1.5, 1000.0, peak, shape)
    assert slid == pytest.approx(peak * math.sin(shape * math.pi / 2.0), abs=0.01)
    assert TyreModel(peak, peak, shape).slide_ratio == pytest.approx(math.sin(shape * math.pi / 2.0), abs=1e-12)


def _drive(model, speed, steering, seconds=1.5, dt=0.001):
    """Euler-integrate from a straight line at ``speed`` with the wheels held at ``steering``; return the states."""
    f = make_dynamics(model)
    x = np.array([0.0, 0.0, steering, speed, 0.0, 0.0, 0.0])
    out = []
    for _ in range(int(seconds / dt)):
        x = x + dt * f(x, np.array([0.0, 0.0]), PARAMS)
        out.append(x.copy())
    return np.array(out)


def test_a_weak_rear_spins_and_a_weak_front_runs_wide():
    # invariant: steady cornering has yaw rate = v * curvature with slip near zero. A rear that saturates first
    # (oversteer) lets yaw rate grow past what the speed and the tyres can hold, with large body slip; a front that
    # saturates first (understeer) holds the yaw rate under the kinematic one (v tan(delta) / L).
    speed, steering = 3.0, 0.30
    kinematic = speed * math.tan(steering) / 0.36
    spin = _drive(TyreModel(0.60, 0.35, 1.6), speed, steering)
    wide = _drive(TyreModel(0.35, 0.60, 1.6), speed, steering)
    assert np.max(np.abs(spin[:, 6])) > 0.5                    # rad of body slip: sliding sideways
    assert np.max(spin[:, 5]) > kinematic                      # rotating faster than the path it was steered on
    assert np.max(np.abs(wide[:, 6])) < 0.2
    assert wide[-1, 5] < 0.8 * kinematic
    # linear tyres (peaks out of reach) neither spin nor run wide at this ask
    linear = _drive(TyreModel(50.0, 50.0, 1.6), speed, steering)
    assert np.max(np.abs(linear[:, 6])) < 0.2


@pytest.mark.parametrize("kwargs, match", [
    (dict(peak_front=0.0, peak_rear=0.5), "peak_front"),
    (dict(peak_front=0.5, peak_rear=math.nan), "peak_rear"),
    (dict(peak_front=0.5, peak_rear=0.5, shape=2.0), "shape"),
    (dict(peak_front=0.5, peak_rear=0.5, shape=1.0), "shape"),
    (dict(peak_front=0.5, peak_rear=0.5, coupling=1.1), "coupling"),
])
def test_bad_models_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        TyreModel(**kwargs)


def test_the_plant_swaps_gyms_tyres_only_when_asked():
    # invariant: the default calibration keeps gym's dynamics function; tyre_model set replaces it and turns the steering clamp off.
    from dataclasses import replace
    import eval_gap
    from sim_fidelity import calibration_for_car
    base = eval_gap.make_plant("car2_course_2026_11_11", 1, car_id="rb1")
    try:
        assert base.env.unwrapped.sim.dynamics_fn is vehicle_dynamics_st
    finally:
        base.close()
    cal = replace(calibration_for_car("rb1"), tyre_model=TyreModel(0.48, 0.48, 1.6))
    plant = eval_gap.make_plant("car2_course_2026_11_11", 1, calibration=cal, car_id="rb1")
    try:
        assert plant.env.unwrapped.sim.dynamics_fn is not vehicle_dynamics_st
        plant.reset(np.array([[-10.49, 11.28, 0.0]]), speeds=[3.0])
        plant.step([(0.3, 3.0)])
        assert plant.grip[0].total_steps == 0                  # the clamp never ran
    finally:
        plant.close()
