"""Pin car 2's 2026-11-11 floor to its recorded measurements, and its speed loop to the braking lag.

Oracles: adjacent sfu_course_2026_11_11_rb2_{measurement,grip}*.json (recorded measurements of runs
20261111-152132, -135934, -142254), the braking fixture (sfu_course_2026_11_11_rb2_braking.npz: run
20261111-152132's /drive speed, steering and wheel speed over its two flying-lap LB holds), and the
recorded numbers below. Run by: python3 -m pytest tools/f1tenth_sim/tests/ -q -p no:cacheprovider
"""
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM.parents[1] / 'tools'))
from sim_fidelity import FLOORS, calibration_for_car, with_floor  # noqa: E402
from sim_fidelity.grip import ProgressiveGrip  # noqa: E402

NAME = 'sfu_course_2026_11_11_rb2'
RUN = '20261111-152132'


def evidence(kind):
    files = {'grip_rb2_vesc': 'sfu_course_2026_11_11_grip_rb2_vesc.json'}
    return json.loads((SIM / 'sim_fidelity' / files.get(kind, f'{NAME}_{kind}.json')).read_text())


def test_the_floor_names_its_runs_and_every_override_has_a_source():
    floor = FLOORS[NAME]
    assert floor.fit_runs == tuple(evidence('measurement')['runs'])
    assert RUN in floor.fit_runs
    assert set(floor.sources) == set(floor.overrides)
    assert all(isinstance(s, str) and s.strip() for s in floor.sources.values())
    assert floor.track == 'car2_course_2026_11_11'


def test_grip_curve_is_the_measured_gyro_curve_times_the_replay_scale_and_it_falls():
    # Oracle: grip_truth.py's recorded gyro-basis medians (bins of 10+ samples), scaled by the
    # replay-selected 1.25 and never above the request, as the 11-06 floor was built.
    rows = [(0.0, 0.0)] + [(r['asked'], min(r['asked'], r['delivered'])) for r in evidence('grip_rb2_vesc')['gyro_curve']
                           if r['n'] >= 10]
    expected = [(a, min(a, d * 1.25)) for a, d in rows]
    cal = with_floor(NAME, calibration_for_car('rb2'))
    np.testing.assert_allclose(cal.grip_curve, expected, rtol=0., atol=1e-12)    # m/s^2
    assert cal.grip_curve_falls
    grip = ProgressiveGrip(cal.grip_curve, 0.36, falling=cal.grip_curve_falls)
    peak_asked = max(cal.grip_curve, key=lambda row: row[1])[0]
    peak = grip.deliver(peak_asked)
    assert 3.5 < peak_asked < 4.0
    assert grip.deliver(4.3847966706921095) < peak - 0.2       # it really falls past the peak (m/s^2)
    assert evidence('grip_rb2_vesc')['sustained_limit'] == pytest.approx(2.94, abs=0.005)


def test_speed_loop_odometry_and_linkage_follow_the_measurement():
    measured = evidence('measurement')
    cal = with_floor(NAME, calibration_for_car('rb2'))
    loop = measured['speed_loop']
    for suffix in ('accel_gain', 'accel_max', 'brake_gain', 'brake_max'):
        assert getattr(cal, 'speed_loop_' + suffix) == loop[suffix]
    assert cal.speed_delay_ticks == loop['delay']
    assert cal.odom_speed_scale == pytest.approx(measured['wheel_over_true'], abs=1e-12)
    assert cal.speed_command_scale == pytest.approx(4614.0 / 4447.983786 / measured['wheel_over_true'], abs=1e-12)
    commands, angles = np.array(cal.steering_linkage).T
    for command in (-.349, -.1, 0., .1, .349):       # central slope 1.0 plus the measured trim
        assert np.interp(command, commands, angles) == pytest.approx(command + measured['steering_trim_rad'], abs=1e-12)


# --- Acceptance B: braking lag, recorded on 152132 ----------------------------------------
# Recorded measurement (this fixture, braking_lag.py): the delay that best aligns /drive speed with
# wheel speed is 0.2375 s on rising commands and 0.2375 s on falling; the three hardest braking
# zones of the flying laps decelerate at 2.976, 2.904 and 2.793 m/s^2 on the wheels.
REAL_RISE_S, REAL_FALL_S = 0.2375, 0.2375
REAL_ZONE_DECEL = (2.976, 2.904, 2.793)


def _fixture():
    d = np.load(SIM / 'sim_fidelity' / f'{NAME}_braking.npz')
    return [(d[f't{i}'].astype(float), d[f'cmd{i}'].astype(float), d[f'steer{i}'].astype(float),
             d[f'wheel{i}'].astype(float)) for i in (1, 2)]


def _plant_wheel(cal, windows):
    from run_analysis.fit_sim import simulate_commands
    out = []
    for t, cmd, steer, wheel in windows:
        speed, _ = simulate_commands(np.column_stack([steer, cmd]), cal)
        out.append(speed * cal.odom_speed_scale)
    return out


def test_braking_lag_metric_recovers_a_known_delay_and_the_recorded_real_numbers():
    from run_analysis import braking_lag as bl
    # Oracle 1: a wheel that is the command delayed by exactly 0.2 s has lag 0.2 on both signs.
    t = np.arange(0, 20, 0.025)
    cmd = 3.0 + 2.0 * np.sin(t * 1.3)
    wheel = np.interp(t - 0.2, t, cmd)
    assert bl.command_lag([(t, cmd, wheel)], +1)[0] == pytest.approx(0.2, abs=0.0125 / 2 + 1e-9)    # s
    assert bl.command_lag([(t, cmd, wheel)], -1)[0] == pytest.approx(0.2, abs=0.0125 / 2 + 1e-9)
    # Oracle 2: the recorded numbers of run 152132, recomputed from its fixture.
    windows = _fixture()
    segs = [(t, cmd, wheel) for t, cmd, _, wheel in windows]
    assert bl.command_lag(segs, +1)[0] == pytest.approx(REAL_RISE_S, abs=1e-9)
    assert bl.command_lag(segs, -1)[0] == pytest.approx(REAL_FALL_S, abs=1e-9)
    zones = bl.hardest_zones([(t, wheel) for t, _, _, wheel in windows], n=3)
    assert [z['decel'] for z in zones] == pytest.approx(list(REAL_ZONE_DECEL), abs=0.0015)   # m/s^2


def _compare(cal):
    from run_analysis import braking_lag as bl
    windows = _fixture()
    sim = _plant_wheel(cal, windows)
    sim_segs = [(t, cmd, w) for (t, cmd, _, _), w in zip(windows, sim)]
    zones = bl.hardest_zones([(t, wheel) for t, _, _, wheel in windows], n=3)
    sim_decel = bl.zone_decels([(t, w) for (t, _, _, _), w in zip(windows, sim)], zones)
    return (bl.command_lag(sim_segs, +1)[0], bl.command_lag(sim_segs, -1)[0], sim_decel)


def test_floor_speed_loop_reproduces_the_real_braking_lag_and_deceleration():
    # Acceptance B (advisor, 2026-11-12): within 0.05 s on both signs and 0.3 m/s^2 in each zone.
    rise, fall, decel = _compare(with_floor(NAME, calibration_for_car('rb2')))
    assert rise == pytest.approx(REAL_RISE_S, abs=0.05)     # s
    assert fall == pytest.approx(REAL_FALL_S, abs=0.05)
    assert decel == pytest.approx(list(REAL_ZONE_DECEL), abs=0.3)   # m/s^2


def test_a_slower_speed_loop_fails_the_braking_acceptance():
    # The check can fail: doubling the transport delay (4 -> 8 ticks) moves the lag out of 0.05 s.
    cal = replace(with_floor(NAME, calibration_for_car('rb2')), speed_delay_ticks=8)
    rise, fall, _ = _compare(cal)
    assert abs(rise - REAL_RISE_S) > 0.05 or abs(fall - REAL_FALL_S) > 0.05


def test_floor_matches_the_selection_recorded_in_the_replay_file():
    # Oracle: the recorded fit (sfu_course_2026_11_11_rb2_replay.json): best of its own top-10 table by yaw RMS.
    replay = evidence('replay')
    best = min(replay['top10'], key=lambda row: row['yaw'])
    assert (best['scale'], best['slope'], best['delay']) == (
        replay['selected']['grip_scale'], replay['selected']['linkage_slope'], replay['selected']['steering_delay_ticks'])
    cal = with_floor(NAME, calibration_for_car('rb2'))
    assert cal.steering_delay_ticks == best['delay']
    commands, angles = np.array(cal.steering_linkage).T
    trim = evidence('measurement')['steering_trim_rad']
    assert np.interp(0.3, commands, angles) == pytest.approx(best['slope'] * 0.3 + trim, abs=1e-12)    # rad
    assert replay['fit_run'] == RUN
