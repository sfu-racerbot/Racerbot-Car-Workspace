"""layers.py: which layers run, from /proc argv as launch starts them.
Oracle: the executables and paths ros2 launch uses (install/<pkg>/lib/<pkg>/<exe>),
racerbot_doctor car.BRINGUP_EXES, and Team A's launch files (racerbot_a src/*/launch)."""
from dataclasses import dataclass

from racerbot_console import layers
from racerbot_doctor import car as dc

WS = '/home/racerbot/racerbot-ws/install'


@dataclass
class P:
    argv: list


def cpp(pkg, exe):
    return P([f'{WS}/{pkg}/lib/{pkg}/{exe}', '--ros-args'])


def py(pkg, exe):
    return P(['/usr/bin/python3', f'{WS}/{pkg}/lib/{pkg}/{exe}', '--ros-args'])


BRINGUP = [cpp('vesc_driver', 'vesc_driver_node'), cpp('urg_node', 'urg_node_driver'),
           cpp('joy', 'joy_node'), cpp('ackermann_mux', 'ackermann_mux'),
           cpp('vesc_ackermann', 'ackermann_to_vesc_node'),
           cpp('vesc_ackermann', 'vesc_to_odom_node')]


def test_bringup_list_covers_every_doctor_bringup_exe():
    exes = {layers.package_and_exe(p.argv)[1] for p in BRINGUP}
    assert exes == set(dc.BRINGUP_EXES)


def test_package_and_exe():
    assert layers.package_and_exe(py('gap_follow', 'gap_follow_node').argv) == \
        ('gap_follow', 'gap_follow_node')
    assert layers.package_and_exe(['/usr/bin/bash', '-c', 'x']) == (None, 'bash')
    assert layers.package_and_exe([]) == (None, None)


def test_hardware_up_off_and_partial_names_what_is_missing():
    assert layers.classify(BRINGUP).hardware == 'up'
    assert layers.classify([]).hardware == 'off'
    part = layers.classify(BRINGUP[1:])
    assert part.hardware == 'partial' and part.missing == ('vesc_driver_node',)


def test_each_driving_layer_is_named():
    got = layers.classify(BRINGUP + [py('gap_follow', 'gap_follow_node')])
    assert got.driving == ('gap follow',)
    assert layers.classify(BRINGUP).driving == ()


def test_team_a_reactive_is_not_mistaken_for_our_gap_follow():
    got = layers.classify([cpp('reactive', 'gap_follow_node')])
    assert got.driving == ('Team A reactive',)


def test_two_driving_layers_are_both_reported():
    got = layers.classify([py('joy_teleop', 'joy_teleop'), py('gap_follow', 'gap_follow_node')])
    assert got.driving == ('gap follow', 'manual')


def test_unknown_package_with_a_known_exe_name_is_not_a_driving_layer():
    assert layers.classify([cpp('some_tool', 'gap_follow_node')]).driving == ()


def test_simulator_and_dashboard_are_seen():
    got = layers.classify([py('racerbot_sim', 'gym_bridge_node'),
                           py('web_dashboard', 'web_dashboard_node')])
    assert got.simulator and got.dashboard
    assert not layers.classify(BRINGUP).simulator
