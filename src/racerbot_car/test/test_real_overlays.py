import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from racerbot_car import guards, layers

ROOT = Path(__file__).resolve().parents[3]
CARS = ROOT / 'src/racerbot_car/cars'
CONFIGS = {name: ROOT / 'src/f1tenth_system/f1tenth_stack/config' / name
           for name in ('vesc.yaml', 'sensors.yaml', 'joy_teleop.yaml')}
CONFIGS.update({f'{name}.yaml': ROOT / 'src' / name / 'config' / f'{name}.yaml'
                for name in ('gap_follow', 'pure_pursuit', 'neural_racer', 'odom_calibration')})
CONFIGS['auto_map_race.yaml'] = ROOT / 'src/pure_pursuit/config/auto_map_race.yaml'
OVERLAYS = sorted(CARS.glob('*/calibration/*.yaml'))


@pytest.mark.parametrize('overlay', OVERLAYS)
def test_calibration_overlay_valid(overlay):
    assert guards.check_calibration_overlay(overlay, overlay.name) == []
    assert layers.check_block_keys_match(str(CONFIGS[overlay.name]), str(overlay)) == []
