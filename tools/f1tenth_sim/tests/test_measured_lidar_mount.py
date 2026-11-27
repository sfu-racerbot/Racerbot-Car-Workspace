"""A wall with a known position checks actual scans, not two copies of the transform."""
from dataclasses import replace
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_gap
from racerbot_sim import tracks
from racerbot_sim.sim_bridge import SimBridge, SimConfig


@pytest.fixture
def room(tmp_path):
    # East and north walls both start at +2.00 m. No obstacles in between.
    pixels = np.zeros((401, 401), dtype=np.uint8)
    pixels[1:-1, 1:-1] = 255
    Image.fromarray(pixels).save(tmp_path / 'room.png')
    (tmp_path / 'room_map.yaml').write_text(
        'image: room.png\nresolution: 0.01\norigin: [-2, -2, 0]\n'
        'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
    line = tracks.rounded_rectangle_centerline(1.0, 1.0, 0.5)
    np.savetxt(tmp_path / 'room_centerline.csv', line, delimiter=',', header='x_m,y_m', comments='')
    return str(tmp_path / 'room'), line


@pytest.mark.parametrize('backend', ['levine', 'bridge'])
@pytest.mark.parametrize('options,expected', [
    ({}, 1.56),                               # legacy: 2.00 - 0.18 - 0.26
    ({'measured_lidar_mount': True}, 1.74),    # measured: 2.00 - (0.36 - 0.10)
])
def test_forward_beam_from_rear_axle_at_room_origin(room, backend, options, expected):
    path, line = room
    if backend == 'levine':
        car = eval_gap.make_plant(path, 12347, car_id='rb2', **options)
    else:
        car = SimBridge(SimConfig(path, line, seed=12347, car='rb2', **options))
    try:
        for yaw in (0.0, math.pi / 2):
            # Put the rear axle at (0, 0), independently of the caster: the levine plant's
            # pose is the CG, lr ahead of the axle (car 2's measured lr); the bridge keeps 0.18.
            lr = car.calibration.lr if backend == 'levine' else 0.18
            if backend == 'levine':
                # Closed form for the plant's own geometry: wall at 2.00 m, scanner at the
                # calibration's lidar_offset_x ahead of the rear axle (measured) or of the CG (legacy).
                expected = 2.00 - car.calibration.lidar_offset_x - (0.0 if options else lr)
            car.reset(np.array([[lr * math.cos(yaw), lr * math.sin(yaw), yaw]]))
            if backend == 'levine':
                car.step([(0., 0.)])  # allow the delayed scan to arrive
                obs, *_ = car.step([(0., 0.)])
                scan = obs['agent_0']['scan']
            else:
                scan = car.ego_scan()
                # ROS keeps +0.26 m from odom's base_link to laser in both modes.
                odom = car.odometry(0)
                xy = np.array([odom.x, odom.y])
                expected_base = 0.0 if options else 0.18
                assert xy == pytest.approx(expected_base * np.array([math.cos(yaw), math.sin(yaw)]),
                                           abs=1e-6)
            mid = len(scan) // 2
            # Five near-normal beams, 1 cm cells and 1 cm range noise.
            assert np.median(scan[mid-2:mid+3]) == pytest.approx(expected, abs=0.035)
    finally:
        car.close()


def test_measured_origin_uses_active_rear_axle_distance_not_half_wheelbase(room):
    path, _ = room
    cal = replace(eval_gap.calibration_for_car('rb2'), car_id='rb2', cog_front_fraction=0.25)
    plant = eval_gap.make_plant(path, 12347, calibration=cal, measured_lidar_mount=True)
    try:
        # Rear axle at zero, CoG 0.27 m forward; scanner is now *behind* the CoG.
        plant.reset(np.array([[0.27, 0., 0.]]))
        plant.step([(0., 0.)])
        obs, *_ = plant.step([(0., 0.)])
        scan = obs['agent_0']['scan']
        assert scan[len(scan) // 2] == pytest.approx(1.74, abs=0.035)
    finally:
        plant.close()


@pytest.mark.parametrize('backend', ['levine', 'bridge'])
def test_unmeasured_car_cannot_select_the_measured_mount(room, backend):
    path, line = room
    with pytest.raises(ValueError, match='only verified for rb2'):
        if backend == 'levine':
            eval_gap.make_plant(path, 1, car_id='rb1', measured_lidar_mount=True)
        else:
            SimBridge(SimConfig(path, line, car='rb1', measured_lidar_mount=True))


def test_other_state_conventions_are_not_silently_translated():
    from f1tenth_gym.envs.dynamic_models import DynamicModel
    from f1tenth_gym.envs.lidar import LiDARConfig
    from racerbot_sim.lidar_geometry import measured_st_lidar

    with pytest.raises(ValueError, match='requires the ST state frame'):
        measured_st_lidar(LiDARConfig(), car_id='rb2', model=DynamicModel.KS, lr=0.18)


def test_explicit_profile_must_match_the_supplied_calibration(room):
    path, _ = room
    with pytest.raises(ValueError, match="must match the LiDAR calibration"):
        eval_gap.make_plant(path, 1, calibration=eval_gap.calibration_for_car('rb2'),
                            car_id='rb1', measured_lidar_mount=True)
