"""following_response_sec is a safety floor (2026-11-16): a car's tuning overlay may raise the response time following
assumes (brakes for a car ahead earlier), never lower it. ROS-free: python3 -m pytest src/racerbot_car/test/."""
from pathlib import Path

from racerbot_car import guards, layers


def check(tmp_path, value):
    overlay = tmp_path / 'pure_pursuit.yaml'
    overlay.write_text('pure_pursuit_node:\n  ros__parameters:\n    # why: physical difference\n'
                       f'    following_response_sec: {value}\n', encoding='utf-8')
    return guards.check_tuning_overlay(overlay, layers.shared_config_path('pure_pursuit.yaml'), 'pure_pursuit.yaml')


SHARED = 0.45   # oracle: src/pure_pursuit/config/pure_pursuit.yaml following_response_sec


def test_the_shared_value_is_the_one_this_test_assumes():
    import yaml
    shared = yaml.safe_load(Path(layers.shared_config_path('pure_pursuit.yaml')).read_text())
    assert shared['pure_pursuit_node']['ros__parameters']['following_response_sec'] == SHARED


def test_an_overlay_may_not_lower_it():
    problems = check(Path(__import__('tempfile').mkdtemp()), SHARED - 0.01)
    assert any('following_response_sec' in p and 'falls below the shared safety floor' in p for p in problems), problems


def test_an_overlay_may_raise_it_or_keep_it(tmp_path):
    assert check(tmp_path, SHARED + 0.07) == []
    assert check(tmp_path, SHARED) == []


def test_car_1s_shipped_overlay_passes_the_guard():
    path = Path(layers.repo_root()) / 'src/racerbot_car/cars/rb1/tuning/pure_pursuit.yaml'
    assert guards.check_tuning_overlay(path, layers.shared_config_path('pure_pursuit.yaml'), 'pure_pursuit.yaml') == []
