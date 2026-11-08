from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from racerbot_car import guards, layers

# oracle: spec, shared line physics remain rb1's defaults; rb2 overlays them.
# All other calibration keys still must be absent, including future additions.
SHARED_PHYSICS = {
    'auto_map_race.yaml': {'auto_map_race_node': {
        'profile_max_steering_angle': 0.30, 'profile_wheelbase': 0.36,
        'profile_max_lateral_accel': 2.5, 'profile_max_lateral_accel_right': 2.5,
        'profile_max_accel': 6.0, 'profile_max_brake': 3.0,
        'profile_steering_linkage_gain': 1.0, 'profile_steering_rate': 1.0,
        'profile_steering_delay': 0.10, 'profile_speed_delay': 0.10}},
    'pure_pursuit.yaml': {'pure_pursuit_node': {'max_lateral_accel': 2.5}},
}

@pytest.mark.parametrize('config', layers.SHARED_CONFIGS)
def test_shared_config_holds_no_calibration(config):
    blocks = guards.overlay_blocks(layers.shared_config_path(config))
    for block, keys in guards.CALIBRATION_KEYS[config].items():
        for key in keys:
            defaults = SHARED_PHYSICS.get(config, {}).get(block, {})
            if key in defaults:
                assert blocks[block][key] == defaults[key], f'{config}: {block}.{key}'
            else:
                assert key not in blocks.get(block, {}), f'{config}: {block}.{key}'
