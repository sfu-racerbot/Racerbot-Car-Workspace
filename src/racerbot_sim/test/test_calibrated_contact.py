"""Calibrated physics reports wall contact from the plant's own contact walls."""
import math
from pathlib import Path
import shutil

import numpy as np
import pytest

from racerbot_sim import sim_bridge, tracks

ROOT = Path(__file__).resolve().parents[3]
TRACK = 'car2_course_2026_11_06a_scans'


def _require_gym():
    sim_bridge.add_sim_python_path()
    try:
        import f1tenth_gym  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != 'f1tenth_gym':
            raise
        pytest.skip('needs the gym checkout at .sim/ (docs/ros-simulator.md)')


def _padded_body_on_beam_map(bridge, x, y, yaw):
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    samples = bridge._body_samples
    xs = x + samples[:, 0] * cos_yaw - samples[:, 1] * sin_yaw
    ys = y + samples[:, 0] * sin_yaw + samples[:, 1] * cos_yaw
    return bool(bridge.occupied(xs, ys).any())


def test_calibrated_bridge_reports_a_wall_only_in_the_contact_map(tmp_path):
    _require_gym()
    from PIL import Image
    for path in (ROOT / 'tools/f1tenth_sim/maps').glob(TRACK + '*'):
        shutil.copy(path, tmp_path)
    bridge = sim_bridge.SimBridge(sim_bridge.SimConfig(
        track_path=str(tracks.resolve(TRACK, tmp_path)),
        centerline=tracks.load_centerline(tmp_path, TRACK),
        car='rb2', physics='calibrated', odom_speed_noise_std=0.0))
    try:
        assert not bridge.body_contact(0)  # Starts on the centreline, clear of every wall.
        beams = np.array(Image.open(tmp_path / f'{TRACK}.png').convert('L'))[::-1] <= 128
        assert np.array_equal(beams, bridge._occupancy <= 128)  # Gym lays the map bottom row first.
        walls = np.array(Image.open(tmp_path / f'{TRACK}_contact.png').convert('L'))[::-1] <= 128
        resolution, (origin_x, origin_y) = bridge._map_resolution, bridge._map_origin
        pose = None
        for row, col in zip(*np.nonzero(walls & ~beams)):
            x, y = origin_x + (col + 0.5) * resolution, origin_y + (row + 0.5) * resolution
            if not _padded_body_on_beam_map(bridge, x, y, 0.0):
                pose = (x, y, 0.0)
                break
        assert pose is not None, 'the contact map closes no gap wider than the padded body'
        bridge.reset([list(pose)])
        assert bridge.body_contact(0), pose
    finally:
        bridge.close()
