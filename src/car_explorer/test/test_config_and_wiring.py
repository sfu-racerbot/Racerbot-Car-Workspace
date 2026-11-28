"""The shipped config builds a valid explorer, and the node cannot drive.

No ROS needed: the config is read as YAML and the node's source is read as
text. Run: python3 -m pytest src/car_explorer/test/ -q
"""

import ast
from pathlib import Path

import pytest
import yaml

from car_explorer.explorer import (GAP_FOLLOW_MAX_LATERAL, GAP_FOLLOW_MAX_SPEED, ConfigError,
                                   Fence, Rules, check_status_topic)

PKG = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((PKG / 'config/car_explorer.yaml').read_text())['car_explorer']['ros__parameters']
NODE_SOURCE = (PKG / 'car_explorer/explorer_node.py').read_text()


def test_shipped_config_builds_a_valid_fence_and_rules():
    Fence(**{k: CONFIG[k] for k in Fence.__dataclass_fields__})
    Rules(**{k: CONFIG[k] for k in Rules.__dataclass_fields__})


def test_shipped_config_adds_no_cap_below_gap_follows_own():
    # Club decision 2026-11-28: the explorer looks for the limit, so its tops
    # are gap_follow's slider maxima, not something lower.
    assert CONFIG['speed_top'] == GAP_FOLLOW_MAX_SPEED[1]
    assert CONFIG['lateral_top'] == GAP_FOLLOW_MAX_LATERAL[1]


def test_shipped_wall_margin_is_the_small_one_the_club_asked_for():
    assert 0 < CONFIG['wall_margin'] <= 0.05  # m


def test_every_shipped_key_is_one_the_node_declares():
    declared = {node.args[0].value for node in ast.walk(ast.parse(NODE_SOURCE))
                if isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'declare_parameter'
                and node.args and isinstance(node.args[0], ast.Constant)}
    declared |= set(Fence.__dataclass_fields__) | set(Rules.__dataclass_fields__)
    assert set(CONFIG) <= declared, set(CONFIG) - declared


def publishers():
    calls = [node for node in ast.walk(ast.parse(NODE_SOURCE))
             if isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'create_publisher']
    return [ast.unparse(c.args[0]) for c in calls]


def test_the_node_publishes_one_status_string_and_nothing_else():
    assert publishers() == ['String']


def test_the_node_checks_its_status_topic_before_publishing():
    assert NODE_SOURCE.index("check_status_topic(p('status_topic'))") < NODE_SOURCE.index('create_publisher(')


def test_the_node_has_no_drive_command_message_publisher():
    for forbidden in ('Float64', 'AckermannDrive,', 'Twist'):
        assert f'create_publisher({forbidden}' not in NODE_SOURCE


@pytest.mark.parametrize('topic', ['/drive', '/car_explorer/drive', '/teleop', '/ackermann_cmd',
                                   '/commands/motor/speed', 'relative', '/', ''])
def test_drive_like_or_malformed_status_topics_are_refused(topic):
    with pytest.raises(ConfigError):
        check_status_topic(topic)


def test_the_shipped_status_topic_passes():
    check_status_topic(CONFIG['status_topic'])
