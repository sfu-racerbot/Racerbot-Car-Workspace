"""Pin car 2's November 6 floor to fit-only recorded measurements and gyro replay.

Oracles: adjacent sfu_course_2026_11_06_{measurement,replay,split,stable_lidar}.json;
the 2026-10-16 circles in docs/hardware-reference.md; the VESC conversion formula.
Run by: python3 -m pytest tools/f1tenth_sim/tests/ -q -p no:cacheprovider
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt
from PIL import Image

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from sim_fidelity import FLOORS, calibration_for_car, with_floor  # noqa: E402
from sim_fidelity.grip import ProgressiveGrip  # noqa: E402

NAME = 'sfu_course_2026_11_06'


def evidence(kind):
    return json.loads((SIM / 'sim_fidelity' / f'{NAME}_{kind}.json').read_text())


def test_november_split_and_sources_exclude_both_validation_runs():
    # Oracle: the approved run-level split, recorded before fitting.
    split = evidence('split')
    floor = FLOORS[NAME]
    assert floor.fit_runs == tuple(split['fit_runs'])
    assert floor.held_out_runs == tuple(split['held_out_runs'])
    assert not set(floor.fit_runs) & set(floor.held_out_runs)
    assert set(floor.fit_runs) == set(evidence('measurement')['runs'])
    assert set(floor.fit_runs) == set(evidence('replay')['fit_runs'])
    assert split['courses']['06b']['fit'] == []
    assert split['courses']['06b']['held_out'] == ['20261106-184753']
    assert set(floor.sources) == set(floor.overrides)
    assert all(isinstance(s, str) and s.strip() for s in floor.sources.values())
    assert floor.track == 'car2_course_2026_11_06a_scans'


def test_november_grip_reconstructs_the_measured_bins_and_replay_scale():
    # Oracle: the measured medians, not today's profile values. Replay corrects
    # the stock tyres' additional understeer; a tyre never exceeds the request.
    measured = evidence('measurement')
    scale = evidence('replay')['selected_grip_scale']
    expected = [(0., 0.), (2., 2.)]
    for row in measured['grip']:
        if row['low'] >= 2.:
            ask = (row['low'] + row['high']) / 2
            expected.append((ask, min(ask, row['both'] * scale)))
    curve = with_floor(NAME, calibration_for_car('rb2')).grip_curve
    np.testing.assert_allclose(curve, expected, rtol=0., atol=1e-12)  # m/s²
    grip = ProgressiveGrip(curve, 0.36)
    for asked, achieved in expected:
        assert grip.deliver(asked) == pytest.approx(achieved, abs=1e-12)
        assert achieved <= asked
    assert grip.deliver(100.) == pytest.approx(expected[-1][1], abs=1e-12)


def test_november_linkage_and_delay_follow_gyro_replay_not_binned_rack():
    measured, replay = evidence('measurement'), evidence('replay')
    cal = with_floor(NAME, calibration_for_car('rb2'))
    commands, angles = np.array(cal.steering_linkage).T
    # Closed form of the replay-selected central linkage; edges exercise both signs.
    for command in (-.349, -.1, 0., .1, .349):
        expected = replay['selected_linkage_slope'] * command + measured['steering_trim_rad']
        assert np.interp(command, commands, angles) == pytest.approx(expected, abs=1e-12)  # rad
    # Cited circles: outer points retained from 2026-10-16, not extrapolated slope.
    for command, angle in ((-10., -.448), (-.436, -.423), (.436, .423), (10., .509)):
        assert np.interp(command, commands, angles) == pytest.approx(angle, abs=1e-12)
    assert cal.steering_delay_ticks == replay['selected']['setting']['steering_delay_ticks']
    assert cal.steering_servo_gain == measured['servo']['gain']
    assert cal.steering_rate_max == measured['servo']['rate']


def test_november_speed_loop_and_odometry_follow_true_speed_measurement():
    measured, replay = evidence('measurement'), evidence('replay')
    cal = with_floor(NAME, calibration_for_car('rb2'))
    loop = measured['speed_loop']
    for suffix in ('accel_gain', 'accel_max', 'brake_gain', 'brake_max'):
        assert getattr(cal, 'speed_loop_' + suffix) == loop[suffix]
    assert cal.speed_delay_ticks == loop['delay']
    assert cal.odom_speed_scale == pytest.approx(measured['wheel_over_true'], abs=1e-12)
    gains = replay['metadata_gains']
    scale = gains['command_erpm_per_m_s'] / gains['odom_erpm_per_m_s'] / measured['wheel_over_true']
    assert cal.speed_command_scale == pytest.approx(scale, abs=1e-12)


def test_november_sensing_and_braking_coupling_have_measured_sources():
    stable = evidence('stable_lidar')
    cal = with_floor(NAME, calibration_for_car('rb2'))
    assert cal.lidar_noise_std == stable['profile_noise_std_m']
    assert cal.lidar_noise_per_metre == stable['profile_noise_per_metre']
    assert cal.lidar_dropout_rate == stable['moving_no_echo']
    assert cal.lidar_self_hits == tuple(map(tuple, stable['self_hits']))
    # Cited measurement: braking did not diminish lateral acceleration.
    bands = evidence('measurement')['grip_by_long_accel']
    braking = next(b['achieved'] for b in bands if b['high'] == -3)
    steady = next(b['achieved'] for b in bands if b['low'] == -.5)
    assert braking >= steady
    assert cal.longitudinal_grip_coupling == 0.


def test_november_floor_reaches_the_plant_and_new_track_contact_layer():
    # Oracle: steady crawl has no tyre slip, so the fitted command/rack mapping
    # must reach the plant. Contact geometry is independently read from the PNG.
    import eval_gap
    cal = with_floor(NAME, calibration_for_car('rb2'))
    track = 'car2_course_2026_11_06b_scans'
    plant = eval_gap.make_plant(track, 12345, cal)
    try:
        plant.reset(np.array([[.543807016147227, .213606818673232, np.radians(349.25150384970954)]]))
        for command in (.1, -.1):
            for _ in range(40):
                plant.step([(command, .3)])
            expected = evidence('replay')['selected_linkage_slope'] * command + evidence('measurement')['steering_trim_rad']
            assert plant.truth(0)[2] == pytest.approx(expected, abs=1e-4)  # rad
        image = np.array(Image.open(SIM / 'maps' / f'{track}_contact.png').convert('L'))[::-1]
        import yaml
        resolution = yaml.safe_load((SIM / 'maps' / f'{track}_map.yaml').read_text())['resolution']
        expected_field = distance_transform_edt(image >= 128) * resolution
        np.testing.assert_allclose(plant._contact_dt, expected_field, rtol=0., atol=1e-12)  # m
        assert (SIM / 'maps' / f'{track}_centerline.csv').read_bytes() == (
            SIM / 'maps' / 'car2_course_2026_11_06b_centerline.csv').read_bytes()
    finally:
        plant.close()


def test_november_9_floor_is_november_6_plus_braking_and_a_grip_circle():
    # Oracle: the 2026-11-09 hypothesis in calibration.py -- run 20261109-190202 braked at
    # ~4.4 m/s^2 and cornered 2.6-2.8; everything else is the 2026-11-06 fit, unchanged.
    old, new = FLOORS[NAME].overrides, FLOORS['sfu_course_2026_11_09'].overrides
    changed = {k for k in set(old) | set(new) if old.get(k) != new.get(k)}
    assert changed == {'floor', 'speed_loop_brake_max', 'grip_circle_radius'}
    assert new['speed_loop_brake_max'] == 4.4 and new['grip_circle_radius'] == 4.9  # m/s^2
    assert set(FLOORS['sfu_course_2026_11_09'].sources) >= set(new)


def test_november_9_grip_circle_reaches_the_plant():
    import eval_gap
    plant = eval_gap.make_plant('car2_course_2026_11_09', 12345,
                                with_floor('sfu_course_2026_11_09', calibration_for_car('rb2')))
    try:
        assert isinstance(plant.grip[0], ProgressiveGrip)
        assert plant.grip[0].circle_radius == 4.9
    finally:
        plant.close()
    plant = eval_gap.make_plant('car2_course_2026_11_09', 12345, with_floor(NAME, calibration_for_car('rb2')))
    try:
        assert plant.grip[0].circle_radius is None
    finally:
        plant.close()
