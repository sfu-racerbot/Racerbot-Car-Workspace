"""Run with pytest tools/racerbot_sim/tests/; no ROS nodes are required."""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

VIEW = Path(__file__).resolve().parents[1] / 'lichtblick'


def _bridge_params():
    root = ET.parse(VIEW / 'sim_view.launch.xml').getroot()
    bridges = [n for n in root.iter('node') if n.get('pkg') == 'foxglove_bridge']
    assert len(bridges) == 1
    return {p.get('name'): p.get('value') for p in bridges[0].iter('param')}


def _layout_topics():
    config = json.loads((VIEW / 'sim-layout.json').read_text())['configById']
    topics = set()
    for panel in config.values():
        topics.update(panel.get('topics', {}))
        topics.update(p['value'].split('.')[0] for p in panel.get('paths', []))
        if 'topicPath' in panel:
            topics.add(panel['topicPath'])
    return topics


def test_bridge_is_loopback_only_and_accepts_no_client_writes():
    params = _bridge_params()
    assert params['address'] == '127.0.0.1'
    assert params['remote_access'] == 'false'
    assert params['capabilities'] == "['connectionGraph']"
    # '(?!)' matches no string, so clients may publish, set or call nothing.
    for key in ('client_topic_whitelist', 'param_whitelist', 'service_whitelist'):
        assert params[key] == "['(?!)']"
        assert re.search(params[key][2:-2], '/drive') is None


def test_view_stays_on_the_isolated_simulator_domain():
    root = ET.parse(VIEW / 'sim_view.launch.xml').getroot()
    env = {e.get('name'): e.get('value') for e in root.iter('set_env')}
    assert env == {'ROS_DOMAIN_ID': '79', 'ROS_LOCALHOST_ONLY': '1',
                   'ROS_AUTOMATIC_DISCOVERY_RANGE': 'LOCALHOST'}
    assert {n.get('pkg') for n in root.iter('node')} == {
        'nav2_map_server', 'nav2_lifecycle_manager', 'foxglove_bridge'}


def test_every_layout_topic_is_bridged_and_no_real_drive_topic_is():
    pattern = re.compile(_bridge_params()['topic_whitelist'][2:-2])
    topics = _layout_topics()
    # oracle: the four stock panels in sim-layout.json
    assert topics == {'/sim/track_map', '/sim/ground_truth_pose', '/sim/car2/ground_truth_pose',
                      '/scan', '/odom', '/sim_only/ackermann_cmd', '/sim/status'}
    assert all(pattern.fullmatch(t) for t in topics)
    for real in ('/drive', '/teleop', '/ackermann_cmd', '/joy', '/sim_only/drive'):
        assert pattern.fullmatch(real) is None


def test_fast_dds_profile_uses_loopback_udp_only():
    text = (VIEW / 'loopback-udp.xml').read_text()
    root = ET.fromstring(text)
    ns = {'f': 'http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles'}
    assert [a.text for a in root.iterfind('.//f:interfaceWhiteList/f:address', ns)] == ['127.0.0.1']
    assert root.find('.//f:useBuiltinTransports', ns).text == 'false'
