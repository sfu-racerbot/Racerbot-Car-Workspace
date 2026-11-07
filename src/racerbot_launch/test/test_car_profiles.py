"""Both cars retain their hardware facts in the unified workspace (Task 12d).

Run with the launch/car/doctor suites via ``rb env ... python3 -m pytest``.
Hardware oracles: docs/hardware-reference.md (rb1 before the rb2 section,
rb2 within it), docs/car1-vs-car2.md, and each pre-merge repo's configs.
The approved Task 12d rulings replace tree-wide absence with per-car routing.

Old guard -> replacement (all names below are test functions).

| Old test | New test(s) |
|---|---|
| test_urg_node_uses_the_usb_hokuyo_not_ethernet | test_lidar_profile_and_effective_config |
| test_urg_node_uses_the_ethernet_hokuyo_not_usb | test_lidar_profile_and_effective_config |
| test_hokuyo_usb_id_agrees_between_doctor_and_hardware_doc | test_doctor_lidar_matches_hardware_doc |
| test_doctor_lidar_symlink_is_the_one_urg_node_opens | test_doctor_lidar_matches_hardware_doc |
| test_doctor_checks_the_ethernet_lidar_the_doc_describes | test_doctor_lidar_matches_hardware_doc |
| test_doctor_has_no_ethernet_lidar_settings | test_doctor_lidar_has_no_other_cars_settings |
| test_doctor_has_no_usb_lidar_settings | test_doctor_lidar_has_no_other_cars_settings |
| test_doctor_camera_device_is_the_dashboard_webcam_device | test_doctor_camera_matches_dashboard |
| test_doctor_checks_the_realsense_the_dashboard_streams | test_doctor_camera_matches_dashboard |
| test_no_realsense_anywhere_in_car_1_launch_or_doctor | test_camera_launch_selects_only_this_cars_camera, test_doctor_camera_matches_dashboard, test_rb1_boot_launches_have_no_realsense, test_rb1_setup_does_not_build_realsense |
| test_no_car_1_webcam_launch | test_camera_launch_selects_only_this_cars_camera |
| test_realsense_launch_and_driver_are_still_here | test_realsense_launch_and_driver_are_still_here |
| test_dashboard_launch_loads_car_1s_settings | test_dashboard_profile_matches_yaml |
| test_dashboard_launch_loads_car_2s_settings | test_dashboard_profile_matches_yaml |
| test_web_dashboard_service_starts_car_1s_dashboard_launch | test_web_dashboard_service_starts_car_1s_dashboard_launch |
| test_x11vnc_follows_the_live_display_not_car_2s_fixed_one | test_x11vnc_follows_the_live_display_not_car_2s_fixed_one |
| test_x11vnc_runs_a_root_owned_copy_not_the_workspace_script | test_x11vnc_runs_a_root_owned_copy_not_the_workspace_script |
| test_vesc_steering_calibration_is_car_1s_measured_one | test_vesc_steering_matches_hardware_doc |
| test_vesc_steering_calibration_is_car_2s_measured_one | test_vesc_steering_matches_hardware_doc |
| test_full_stick_and_autonomy_limits_are_car_1s | test_steering_limits_match_hardware_doc |
| test_full_stick_and_autonomy_limits_are_car_2s | test_steering_limits_match_hardware_doc |
| test_odometry_gain_is_car_2s | test_rb2_odometry_gain_matches_hardware_doc |
| test_bno_is_read_through_the_jetson_i2c_bridge | test_bno_is_read_through_the_jetson_i2c_bridge |
| test_imu_bridge_restarts_the_bno_driver | test_imu_bridge_restarts_the_bno_driver |
| test_map_course_runs_car_1s_imu_stack | test_map_course_runs_the_imu_stack_by_default |
| test_map_course_runs_the_imu_stack_by_default | test_map_course_runs_the_imu_stack_by_default |
| test_map_course_maps_on_the_gyro_by_default_on_car_2 | test_map_course_maps_on_the_gyro_by_default_on_car_2 |
| test_vehicle_state_launch_uses_the_bno_by_default | test_vehicle_state_launch_uses_the_bno_by_default |
| test_vehicle_state_geometry_is_car_1s | test_vehicle_state_geometry_matches_calibration |
| test_every_bringup_starts_car_1s_automatic_recorder | test_every_bringup_starts_the_automatic_recorder |
| test_gap_follow_steering_damping_is_on_by_default | test_gap_follow_steering_damping_is_on_by_default |
| test_gap_follow_keeps_car_2s_course_settings | test_gap_follow_keeps_car_2s_course_settings |
| test_gap_follow_imu_features_ship_off_on_car2 | test_gap_follow_imu_features_ship_off_on_car2 |
| test_car_1_imu_values_stay_labelled_as_car_1s | test_car_1_imu_values_stay_labelled_as_car_1s |
| test_grip_replay_finds_runs_in_car_2s_data_folder | test_grip_replay_finds_runs_in_each_cars_data_folder |

Additional independent historical oracles: test_rb1_calibration_matches_car1_repo
and test_rb2_calibration_matches_car2_repo. Historical YAML is read from git,
never used as a live calibration file.

Mutation evidence (Task 12d, ROS_DOMAIN_ID=69). Every row was applied alone,
the selected tests failed, the break was reverted, and the same selection
passed. Counts exclude deselected tests; rb1/rb2 cases are counted separately.
Paths below abbreviate package subdirectories. Car files are under
racerbot_car/cars; launch filenames are in their owning package's launch/.

| Break | Selection in this file unless noted | Failed | Passed after revert |
|---|---|---|---|
| Swap cars/rb1 and cars/rb2 car.yaml lidar blocks | -k lidar | 6 | 6 |
| Swap cars/rb1 and cars/rb2 calibration/vesc.yaml files | -k 'vesc_steering or calibration_matches or odometry_gain' | 5 | 5 |
| camera_launch.py: RealSense branch selects webcam | -k camera_launch_selects | 1 (rb2) | 2 |
| racerbot_doctor/car.py: spec_for always returns RB1 | -k doctor | 3 (rb2) | 6 |
| docs/hardware-reference.md rb2 IP .10 -> .11 | -k 'lidar_profile or doctor_lidar_matches' | 2 (rb2) | 4 |
| Swap car.yaml camera blocks | -k 'doctor_camera or camera_launch_selects' | 4 | 4 |
| dashboard_launch.py: include RealSense instead of dashboard | -k rb1_boot | 1 | 1 |
| tools/dev/setup_clone.sh: add RealSense build | -k rb1_setup | 1 | 1 |
| .gitmodules: replace RealSense submodule path | -k realsense_launch_and_driver | 1 | 1 |
| Swap car.yaml dashboard_site values | -k dashboard_profile | 2 | 2 |
| tools/systemd/web-dashboard.service: start bare dashboard | -k web_dashboard_service | 1 | 1 |
| tools/systemd/x11vnc-follow.sh: fixed display :1 | -k x11vnc_follows | 1 | 1 |
| tools/systemd/x11vnc.service: run workspace script | -k x11vnc_runs | 1 | 1 |
| Swap calibration/joy_teleop.yaml full-stick scales | -k steering_limits_match | 2 | 2 |
| imu_bridge/launch/vehicle_state_launch.py: remove --no-dyncal argument | -k bno_is_read | 1 | 1 |
| imu_bridge/imu_bridge_node.py: remove BNO restart timer | -k imu_bridge_restarts | 1 | 1 |
| map_course_launch.py: imu defaults false | -k map_course_runs | 2 | 2 |
| map_course_launch.py: gyro_odom defaults false | -k maps_on_the_gyro | 1 | 1 |
| vehicle_state_launch.py: bno defaults false | -k vehicle_state_launch_uses | 2 | 2 |
| imu_bridge/config/vehicle_state.yaml: bno_x_offset 0.36 -> 0.324 | -k vehicle_state_geometry | 2 | 2 |
| no_lidar_bringup_launch.py: replace auto_record include | -k every_bringup | 1 | 1 |
| gap_follow/config/gap_follow.yaml: damping false | -k gap_follow_steering_damping | 2 | 2 |
| gap_follow/config/gap_follow.yaml: swept length 5 -> 3 | -k keeps_car_2s_course | 1 | 1 |
| gap_follow/config/gap_follow.yaml: impact stop true | -k imu_features_ship | 1 | 1 |
| gap_follow/config/gap_follow.yaml: remove CAR 1'S VALUES label | -k stay_labelled | 1 | 1 |
| racerbot_sim/sim_imu.py: remove CAR 1'S NOISE label | -k stay_labelled | 1 | 1 |
| tools/run_analysis/grip_replay.py: omit car directory from run lookup | -k grip_replay | 2 | 2 |
| rb2 calibration/vesc.yaml: command gain +10% (4614 -> 5075.4) | imu_bridge/test/test_imu_calibration_per_car.py::test_rb2_imu_car_yaml_agrees_with_rb2_calibration | 1 | 1 |
| racerbot_doctor/car.py: exchange both cars' complete specs | -k doctor | 6 | 6 |

Final verification: launch/car/doctor suites: 682 passed, only the explicitly
expected domain-69-vs-79 assertion failed. IMU calibration plus gap_follow:
1344 passed. No mutation remains in the working tree.
"""

import importlib.util
from pathlib import Path
import re
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'src'
STACK = SRC / 'f1tenth_system/f1tenth_stack'
LAUNCH = SRC / 'racerbot_launch/launch'
SYSTEMD = ROOT / 'tools/systemd'
IMU = SRC / 'imu_bridge'
for package in ('racerbot_car', 'racerbot_doctor'):
    sys.path.insert(0, str(SRC / package))
sys.path.insert(0, str(ROOT / 'tools/run_analysis'))

from racerbot_car import hardware  # noqa: E402
from racerbot_car.layers import effective_config, SHARED_CONFIGS  # noqa: E402
from racerbot_car.profile import (  # noqa: E402
    CameraRealSense, CameraUVC, LidarEthernet, LidarSerial, load_profile)
from racerbot_doctor import car as doctor  # noqa: E402

CARS = pytest.mark.parametrize('car', ['rb1', 'rb2'])


def read_yaml(path):
    return yaml.safe_load(path.read_text())


def dashboard(car):
    return read_yaml(SRC / 'racerbot_launch/config' / f'web_dashboard_{car}.yaml')


def doc(car):
    sections = (ROOT / 'docs/hardware-reference.md').read_text().split('## Car 2 (rb2)')
    assert len(sections) == 2
    return sections[0 if car == 'rb1' else 1]


def doc_match(car, pattern):
    match = re.search(pattern, doc(car))
    assert match, f'{car}: hardware-reference.md no longer matches {pattern!r}'
    return match


def doc_number(car, pattern):
    return float(doc_match(car, pattern).group(1))


def lidar_address(car):
    ip, port = doc_match(car, r"Sensor's IP:port \| `([0-9.]+):([0-9]+)`").groups()
    return ip, int(port)


@CARS
def test_lidar_profile_and_effective_config(car):
    profile = load_profile(car)
    params = effective_config('sensors.yaml', 'urg_node', car_id=car)
    if car == 'rb1':
        # docs/car1-vs-car2.md and hardware-reference.md's USB Hokuyo rule.
        assert isinstance(profile.lidar, LidarSerial)
        assert profile.lidar.port == params.get('serial_port') == '/dev/sensors/hokuyo'
        assert 'ip_address' not in params and 'ip_port' not in params
    else:
        ip, port = lidar_address(car)
        assert isinstance(profile.lidar, LidarEthernet)
        assert profile.lidar.ip == params.get('ip_address') == ip
        assert profile.lidar.port == params.get('ip_port') == port
        assert 'serial_port' not in params


@CARS
def test_doctor_lidar_matches_hardware_doc(car):
    spec = doctor.spec_for(load_profile(car)).lidar
    params = effective_config('sensors.yaml', 'urg_node', car_id=car)
    if car == 'rb1':
        usb = doc_match(car, r'99-hokuyo\.rules` .{0,40}idVendor `([0-9a-f]{4})`, '
                        r'idProduct `([0-9a-f]{4})`').groups()
        assert isinstance(spec, doctor.SerialLidarSpec)
        assert spec.usb == usb
        assert '/dev/' + spec.symlink == params['serial_port']
    else:
        ip, port = lidar_address(car)
        host = doc_match(car, r'NetworkManager profile \| `hokuyo`, static IPv4 `([0-9./]+)`').group(1)
        assert isinstance(spec, doctor.EthernetLidarSpec)
        assert (spec.ip, spec.port, spec.host_ip) == (ip, port, host)
        assert spec.ip == params['ip_address']


@CARS
def test_doctor_lidar_has_no_other_cars_settings(car):
    spec = doctor.spec_for(load_profile(car)).lidar
    if car == 'rb1':
        assert isinstance(spec, doctor.SerialLidarSpec)
        absent = ('iface', 'ip', 'port', 'host_ip')
    else:
        assert isinstance(spec, doctor.EthernetLidarSpec)
        absent = ('usb', 'symlink', 'kernel_driver')
    for name in absent:
        assert not hasattr(spec, name), name


@CARS
def test_doctor_camera_matches_dashboard(car):
    profile = load_profile(car)
    spec = doctor.spec_for(profile).camera
    params = dashboard(car)['usb_cam_stream_node']['ros__parameters']
    if car == 'rb1':
        assert isinstance(profile.camera, CameraUVC)
        assert isinstance(spec, doctor.UvcCameraSpec)
        assert '/dev/' + spec.device == params['device']
        assert not hasattr(spec, 'usb') and not hasattr(spec, 'topic')
        assert spec.exe != 'realsense2_camera_node'
    else:
        assert isinstance(profile.camera, CameraRealSense)
        assert isinstance(spec, doctor.RealSenseSpec)
        assert spec.topic == params['image_topic']
        assert spec.exe == 'realsense2_camera_node'
        assert not hasattr(spec, 'device')


def load_launch(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@CARS
def test_camera_launch_selects_only_this_cars_camera(monkeypatch, car):
    from launch import LaunchContext
    from launch.actions import IncludeLaunchDescription
    from launch.utilities import perform_substitutions

    # Only mock the USB discovery boundary, as in the existing launch tests.
    monkeypatch.setattr(hardware, 'usb_ids_present', lambda: set())
    monkeypatch.setenv('RACERBOT_CAR', car)
    monkeypatch.delenv('RACERBOT_CARS_DIR', raising=False)
    camera = load_profile(car).camera
    assert isinstance(camera, CameraUVC if car == 'rb1' else CameraRealSense)
    description = load_launch(LAUNCH / 'camera_launch.py').generate_launch_description()
    includes = [e for e in description.entities if isinstance(e, IncludeLaunchDescription)]
    assert len(includes) == 1
    substitutions = includes[0].launch_description_source._LaunchDescriptionSource__location
    actual = Path(perform_substitutions(LaunchContext(), substitutions)).name
    assert actual == ('webcam_launch.py' if car == 'rb1' else 'realsense_camera_launch.py')


def test_rb1_boot_launches_have_no_realsense():
    # Follow literal launch includes transitively, including installed submodule
    # launch files. Profile-dispatched camera_launch is evaluated separately above.
    from ament_index_python.packages import get_package_share_directory

    units = sorted(SYSTEMD.glob('*.service'))
    assert units
    pending = []
    for unit in units:
        source = unit.read_text()
        assert not re.search(r'realsense2_camera|realsense_camera_launch|realsense-ros', source)
        for package, filename in re.findall(r'ros2 launch ([\w_]+) ([\w_]+\.py)', source):
            pending.append(Path(get_package_share_directory(package)) / 'launch' / filename)
    assert pending, 'no boot launch roots found'
    visited = set()
    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        visited.add(path)
        source = path.read_text()
        assert not re.search(r'realsense2_camera|realsense_camera_launch|realsense-ros', source), path
        filenames = re.findall(r"['\"]([\w_]+_launch\.py)['\"]", source)
        for filename in filenames:
            candidates = list(SRC.glob(f'*/launch/{filename}'))
            candidates += list(SRC.glob(f'*/*/launch/{filename}'))
            for package in re.findall(r"get_package_share_directory\(['\"]([\w_]+)['\"]\)", source):
                candidate = Path(get_package_share_directory(package)) / 'launch' / filename
                if candidate.is_file():
                    candidates.append(candidate)
            assert candidates, f'cannot follow {filename} from {path}'
            pending.extend(candidates)


def test_rb1_setup_does_not_build_realsense():
    # Task 12d: check setup/build notes if present; no skip when setup_clone exists.
    notes = [p for p in (ROOT / 'tools/dev').rglob('*') if p.is_file()
             and p.suffix in ('.sh', '.md', '.txt')
             and re.search(r'setup|build|install|readme', p.name, re.I)]
    if not notes:
        pytest.skip('Task 12d: no tools/dev setup or build notes exist')
    for path in notes:
        assert 'realsense-ros' not in path.read_text(), path


def test_realsense_launch_and_driver_are_still_here():
    assert 'realsense2_camera' in (LAUNCH / 'realsense_camera_launch.py').read_text()
    # A registered submodule is the contract; initialization is machine-local.
    import configparser
    modules = configparser.ConfigParser()
    modules.read(ROOT / '.gitmodules')
    assert any(modules[s].get('path') == 'src/realsense-ros' for s in modules.sections())


@CARS
def test_dashboard_profile_matches_yaml(car):
    site = load_profile(car).dashboard_site
    assert site == car
    configs = SRC / 'racerbot_launch/config'
    path = configs / f'web_dashboard_{site}.yaml'
    assert path.is_file()
    assert path.stem == f'web_dashboard_{car}'
    assert sorted(configs.glob('web_dashboard_rb*.yaml')) == [
        configs / 'web_dashboard_rb1.yaml', configs / 'web_dashboard_rb2.yaml']


def exec_start(unit):
    lines = [line for line in unit.splitlines() if line.startswith('ExecStart=')]
    assert len(lines) == 1, lines
    return lines[0]


def test_web_dashboard_service_starts_car_1s_dashboard_launch():
    cmd = exec_start((SYSTEMD / 'web-dashboard.service').read_text())
    assert 'ros2 launch racerbot_launch dashboard_launch.py' in cmd
    assert 'web_dashboard_launch.py' not in cmd


def test_x11vnc_follows_the_live_display_not_car_2s_fixed_one():
    unit = (SYSTEMD / 'x11vnc.service').read_text()
    script = (SYSTEMD / 'x11vnc-follow.sh').read_text()
    for text in (unit, script):
        assert not re.search(r'-display\s+:\d', text)
    assert re.search(r'-display\s+":\$\{target', script)


def test_x11vnc_runs_a_root_owned_copy_not_the_workspace_script():
    unit = (SYSTEMD / 'x11vnc.service').read_text()
    assert not re.search(r'^User=', unit, re.M)
    assert exec_start(unit) == 'ExecStart=/usr/local/sbin/racerbot-x11vnc-follow'
    assert re.search(r'install -m 0755 -o root -g root "\$here/x11vnc-follow\.sh" '
                     r'/usr/local/sbin/racerbot-x11vnc-follow',
                     (SYSTEMD / 'install.sh').read_text())


@CARS
def test_vesc_steering_matches_hardware_doc(car):
    params = effective_config('vesc.yaml', '/**', car_id=car)
    for key in ('steering_angle_to_servo_gain', 'steering_angle_to_servo_offset',
                'servo_min', 'servo_max'):
        assert params[key] == doc_number(car, r'(?m)^' + key + r':\s*(-?[0-9.]+)'), key


@CARS
def test_steering_limits_match_hardware_doc(car):
    joy = effective_config('joy_teleop.yaml', 'joy_teleop', car_id=car)
    assert joy['human_control.axis_mappings.drive-steering_angle.scale'] == doc_number(
        car, r'Full stick \(`joy_teleop\.yaml` scale\) is \*\*([0-9.]+) rad')
    controller = doc_number(car, r'`max_steering_angle` \*\*([0-9.]+) rad')
    for config, node in (('gap_follow.yaml', 'gap_follow_node'),
                         ('pure_pursuit.yaml', 'pure_pursuit_node')):
        assert effective_config(config, node, car_id=car)['max_steering_angle'] == controller
    planner = read_yaml(SRC / 'pure_pursuit/config/auto_map_race.yaml')
    assert planner['auto_map_race_node']['ros__parameters']['profile_max_steering_angle'] == doc_number(
        car, r'`profile_max_steering_angle` is \*\*([0-9.]+) rad')


def test_rb2_odometry_gain_matches_hardware_doc():
    gain = effective_config('vesc.yaml', 'vesc_to_odom_node', car_id='rb2')['speed_to_erpm_gain']
    assert gain == doc_number('rb2', r'vesc_to_odom_node speed_to_erpm_gain\s+# must be NEGATIVE '
                             r'\(currently (-[0-9.]+)\)')
    assert gain < 0


def test_rb1_command_gain_matches_hardware_doc():
    # The motor-command gain is measured calibration (2026-11-02, docs/hardware-reference.md,
    # car 1's VESC section); the doc's excerpt and the overlay must say the same number.
    gain = effective_config('vesc.yaml', '/**', car_id='rb1')['speed_to_erpm_gain']
    assert gain == doc_number('rb1', r'(?m)^speed_to_erpm_gain: ([0-9.]+)\s+# /\*\* block: sets the motor command')
    assert gain > 0


def historical_params(ref, config, node):
    # Independent frozen oracle, not a read of live shared calibration files.
    text = subprocess.check_output(['git', 'show', f'{ref}:{SHARED_CONFIGS[config]}'],
                                   cwd=ROOT, text=True)
    blocks = yaml.safe_load(text)
    params = dict(blocks.get('/**', {}).get('ros__parameters', {}))
    params.update(blocks.get(node, {}).get('ros__parameters', {}))

    def flatten(values, prefix=''):
        result = {}
        for key, value in values.items():
            name = prefix + key
            if isinstance(value, dict):
                result.update(flatten(value, name + '.'))
            else:
                result[name] = value
        return result

    return flatten(params)


# Values deliberately recalibrated after the merge: (value at the frozen ref, value now, where the
# measurement is written down). Both are exact: the ref must still say the old value, and the car
# must carry exactly the new one, so this cannot hide a drift in either. Any other key still has to
# match the ref.
RECALIBRATED = {
    ('rb1', 'vesc.yaml', '/**', 'speed_to_erpm_gain'): (
        3650.0, 4600.0, "docs/hardware-reference.md, car 1's VESC section, 2026-11-02"),
    ('rb2', 'gap_follow.yaml', 'gap_follow_node', 'max_steering_angle'): (
        0.349, 0.42, "docs/hardware-reference.md, car 2's steering, 2026-11-07"),
    ('rb2', 'pure_pursuit.yaml', 'pure_pursuit_node', 'max_steering_angle'): (
        0.349, 0.42, "docs/hardware-reference.md, car 2's steering, 2026-11-07"),
}


def assert_historical_calibration(car, ref):
    checks = [
        ('vesc.yaml', '/**', ('speed_to_erpm_gain', 'speed_to_erpm_offset',
                             'steering_angle_to_servo_gain', 'steering_angle_to_servo_offset',
                             'servo_min', 'servo_max')),
        ('vesc.yaml', 'vesc_to_odom_node', ('speed_to_erpm_gain', 'speed_to_erpm_offset',
                                         'steering_angle_to_servo_gain', 'steering_angle_to_servo_offset',
                                         'servo_min', 'servo_max')),
        ('joy_teleop.yaml', 'joy_teleop', ('human_control.axis_mappings.drive-steering_angle.scale',)),
        ('sensors.yaml', 'urg_node', ('serial_port', 'ip_address', 'ip_port')),
    ]
    checks += [(f'{pkg}.yaml', f'{pkg}_node', ('max_steering_angle',))
               for pkg in ('gap_follow', 'pure_pursuit', 'neural_racer')]
    for config, node, keys in checks:
        expected = historical_params(ref, config, node)
        actual = effective_config(config, node, car_id=car)
        for key in keys:
            assert (key in actual) == (key in expected), (car, config, node, key)
            if (car, config, node, key) in RECALIBRATED:
                old, new, _source = RECALIBRATED[(car, config, node, key)]
                assert expected[key] == old, (car, config, node, key, 'frozen ref changed')
                assert actual[key] == new, (car, config, node, key)
                continue
            if key in expected:
                assert actual[key] == expected[key], (car, config, node, key)


def test_rb1_calibration_matches_car1_repo():
    assert_historical_calibration('rb1', 'ae7ceeb')


def test_rb2_calibration_matches_car2_repo():
    # 9c2c2ed is car 2's main before the merge: the merge commit a205f29's second
    # parent, so every clone has it (refs/car2/main existed only on the dev box).
    assert_historical_calibration('rb2', '9c2c2ed')


def test_bno_is_read_through_the_jetson_i2c_bridge():
    launch = (IMU / 'launch/vehicle_state_launch.py').read_text()
    assert "BRIDGE_SCRIPT = 'bno_i2c_bridge.py'" in launch
    assert "'--no-dyncal'" in launch
    assert "'bno_pty', default_value='/tmp/bno085'" in launch
    assert "'bno_port': pty if bno else ''" in launch


def test_imu_bridge_restarts_the_bno_driver():
    node = (IMU / 'imu_bridge/imu_bridge_node.py').read_text()
    assert 'BNO_RESTART_AFTER_S' in node
    assert 'self.create_timer(1.0, self._watch_bno)' in node


@CARS
def test_map_course_runs_the_imu_stack_by_default(car):
    # Shared launch facts, required by both original guards.
    launch = (LAUNCH / 'map_course_launch.py').read_text()
    assert "'launch', 'vehicle_state_launch.py'" in launch
    assert re.findall(r"'imu', default_value='(\w+)'", launch) == ['true']


def test_map_course_maps_on_the_gyro_by_default_on_car_2():
    launch = (LAUNCH / 'map_course_launch.py').read_text()
    assert re.findall(r"'gyro_odom', default_value=LaunchConfiguration\('imu'\)", launch)
    assert re.findall(r"'imu_lap_turns', default_value=LaunchConfiguration\('gyro_odom'\)", launch)


@CARS
def test_vehicle_state_launch_uses_the_bno_by_default(car):
    launch = (IMU / 'launch/vehicle_state_launch.py').read_text()
    assert re.findall(r"'bno', default_value='(\w+)'", launch) == ['true']


@CARS
def test_vehicle_state_geometry_matches_calibration(car):
    wheelbase = effective_config('vesc.yaml', 'vesc_to_odom_node', car_id=car)['wheelbase']
    params = read_yaml(IMU / 'config/vehicle_state.yaml')['vehicle_state_node']['ros__parameters']
    assert params['wheelbase'] == wheelbase
    assert params['bno_x_offset'] == wheelbase


def test_every_bringup_starts_the_automatic_recorder():
    for name in ('bringup_launch.py', 'no_lidar_bringup_launch.py', 'sick_bringup_launch.py'):
        source = (STACK / 'launch' / name).read_text()
        assert "'launch', 'auto_record_launch.py')" in source, name
        assert "condition=IfCondition(LaunchConfiguration('record'))" in source, name


@CARS
def test_gap_follow_steering_damping_is_on_by_default(car):
    params = effective_config('gap_follow.yaml', 'gap_follow_node', car_id=car)
    assert params['enable_steering_damping'] is True
    launch = (SRC / 'gap_follow/launch/gap_follow_launch.py').read_text()
    assert re.findall(r"'steering_damping', default_value=LaunchConfiguration\('imu', "
                      r"default='true'\)", launch)


def test_gap_follow_keeps_car_2s_course_settings():
    params = effective_config('gap_follow.yaml', 'gap_follow_node', car_id='rb2')
    for key, expected in (('swept_clearance_max_length', 5.0), ('aim_depth_bias', 0.0),
                          ('aim_depth_bias_fallback', 1.0), ('opponent_follow_floor', 0.5)):
        assert params[key] == expected, key  # docs/car1-vs-car2.md#gap_follow-course-settings
    launch = (SRC / 'gap_follow/launch/gap_follow_launch.py').read_text()
    assert re.findall(r"'time_trial', default_value='(\w+)'", launch) == ['false']


def test_gap_follow_imu_features_ship_off_on_car2():
    params = effective_config('gap_follow.yaml', 'gap_follow_node', car_id='rb2')
    for name in ('enable_imu_understeer', 'enable_imu_traction', 'enable_imu_impact_stop'):
        assert params[name] is False, name
    for path in (SRC / 'gap_follow/launch/gap_follow_launch.py', LAUNCH / 'map_course_launch.py'):
        launch = path.read_text()
        assert re.findall(r"'imu_all', default_value='(\w+)'", launch) == ['false'], path
        flags = re.findall(r"'((?:fused_speed|imu_\w+))', default_value=", launch)
        assert flags
        for flag in flags:
            if flag in ('imu_all', 'imu_hardware', 'imu_lap_turns'):
                continue
            assert re.search(r"'%s', default_value=(LaunchConfiguration\('imu_all', "
                             r"default='false'\)|'false')" % flag, launch), (path, flag)


def test_car_1_imu_values_stay_labelled_as_car_1s():
    yaml_text = (SRC / 'gap_follow/config/gap_follow.yaml').read_text()
    if '33.7257297468959' in yaml_text or 'imu_traction_slip_threshold: 0.246' in yaml_text:
        assert "CAR 1'S VALUES" in yaml_text
    sim_imu = (SRC / 'racerbot_sim/racerbot_sim/sim_imu.py').read_text()
    if '0.000711560678439245' in sim_imu:
        assert "CAR 1'S NOISE" in sim_imu


@CARS
def test_grip_replay_finds_runs_in_each_cars_data_folder(tmp_path, monkeypatch, car):
    import grip_replay
    monkeypatch.setenv('RACERBOT_WS', str(tmp_path))
    run = tmp_path / 'data' / car / 'racerbot_runs' / 'example'
    run.mkdir(parents=True)
    assert grip_replay.resolve_run('example', car) == run
