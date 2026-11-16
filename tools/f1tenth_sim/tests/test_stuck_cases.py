"""Plan Task 12: the simulator reproduces the race-day wall-alongside stop, and flank creep gets out of it.

Recorded measurement: rb1 run 20261111-155542 sat stopped beside the U-turn's inner barrier (fixture
src/gap_follow/test/data/flank_latch_rb1_20261111-155542.npz, its finite poses at (-10.49, 11.28)), the wall alongside
and the road open ahead. Oracle: with flank creep off (flank_creep_length 0) gap follow must stay stopped there, the
latch; with the shipped value it must leave within 3 s with no chassis contact (the plant's own body-against-map check,
never gym's flag). This keeps Phase 1's fix fixed in closed loop. Run by the tool-suite command in CLAUDE.md.
"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src', 'gap_follow'))
import eval_gap  # noqa: E402
import gap_mirror  # noqa: E402
from sim_fidelity import calibration_for_car  # noqa: E402

FIXTURE = os.path.join(HERE, '..', '..', '..', 'src', 'gap_follow', 'test', 'data', 'flank_latch_rb1_20261111-155542.npz')
TRACK = 'car2_course_2026_11_11'
TICKS = int(round(3.0 / eval_gap.CONTROL_DT))


def recorded_pose():
    poses = np.load(FIXTURE)['poses']
    finite = poses[np.isfinite(poses[:, 0])]
    x, y, yaw = np.median(finite, axis=0)
    return np.array([[x, y, math.remainder(yaw, 2 * math.pi)]])


def drive(car, creep, seed=12345):
    """Gap follow from rest at the recorded pose for 3 s; (metres moved, minimum chassis clearance, contact steps)."""
    plant = eval_gap.make_plant(TRACK, seed, calibration=calibration_for_car(car), car_id=car)
    try:
        pose = recorded_pose()
        obs, _ = plant.reset(pose, speeds=[0.0])
        lidar = type('Lidar', (), {})()
        lidar.angle_min = -eval_gap.components_half_fov(car)
        lidar.angle_increment = eval_gap.components_increment(car)
        lidar.range_min = eval_gap.components_range_min(car)
        params = gap_mirror.node_params(car_id=car)
        params.update(flank_creep_length=creep, enable_opponent_awareness=False, enable_launch_boost=False)
        ctrl = gap_mirror.GapController(params, lidar)
        x0, y0 = plant.truth(0)[0], plant.truth(0)[1]
        clearance = math.inf
        for k in range(TICKS):
            state = np.asarray(obs['agent_0']['std_state'], float)
            steering, speed = ctrl.command(obs['agent_0']['scan'], float(state[3]), k, yaw_rate=float(state[5]))
            obs, *_ = plant.step([(steering, speed)])
            c = plant.chassis_map_clearance(0)
            if math.isfinite(c):
                clearance = min(clearance, c)
        moved = math.hypot(plant.truth(0)[0] - x0, plant.truth(0)[1] - y0)
        return moved, clearance, plant.report().get('contact_steps', 0)
    finally:
        plant.close()


def test_the_start_is_the_recorded_stuck_pose_beside_the_wall():
    pose = recorded_pose()[0]
    assert pose[0] == pytest.approx(-10.49, abs=0.03) and pose[1] == pytest.approx(11.28, abs=0.03)   # m, the fixture
    plant = eval_gap.make_plant(TRACK, 12345, calibration=calibration_for_car('rb1'), car_id='rb1')
    try:
        plant.reset(recorded_pose(), speeds=[0.0])
        plant.step([(0.0, 0.0)])
        assert 0.0 < plant.chassis_map_clearance(0) < 0.2      # beside the wall, not in it and not in open space
    finally:
        plant.close()


@pytest.mark.parametrize('car', ['rb1', 'rb2'])
def test_without_flank_creep_gap_follow_latches_beside_the_wall(car):
    moved, _clearance, contact = drive(car, 0.0)
    assert moved < 0.05 and contact == 0                   # m: the race-day stop, reproduced


@pytest.mark.parametrize('car', ['rb1', 'rb2'])
def test_with_the_shipped_flank_creep_it_leaves_within_3_s_without_contact(car):
    shipped = gap_mirror.node_params(car_id=car)['flank_creep_length']
    assert shipped > 0.0
    moved, clearance, contact = drive(car, shipped)
    assert moved > 1.0                                     # m in 3 s: out of the latch and away
    assert contact == 0 and clearance > 0.0
