"""
Tests for cli.py and report.py, plus the package-wide read-only guarantee.

Pure Python, no rclpy (the live stage is injected):
    python3 -m pytest src/racerbot_doctor/test/ -v

Oracle for exit codes: the plan's contract (0 = no FAIL, 1 = any FAIL,
2 = the doctor itself crashed), docs/superpowers/plans/2026-10-01-racerbot-doctor.md.
"""
import io
import os
import pathlib
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from racerbot_doctor import car  # noqa: E402
from racerbot_doctor.cli import main  # noqa: E402
from racerbot_doctor.report import render  # noqa: E402
from racerbot_doctor.snapshot import (  # noqa: E402
    DeviceReport, Layer, Status, TopicStats)
from test_checks import healthy_live, healthy_snap, without  # noqa: E402

PKG = pathlib.Path(__file__).resolve().parents[1] / 'racerbot_doctor'


def test_doctor_lists_overrides_with_reasons(tmp_path, monkeypatch):
    import shutil
    from racerbot_car import identity, guards, layers
    root = pathlib.Path(__file__).resolve().parents[3]
    shutil.copytree(root / 'src/racerbot_car/cars', tmp_path / 'cars')
    tuning = tmp_path / 'cars/rb1/tuning/gap_follow.yaml'
    tuning.parent.mkdir(exist_ok=True)
    tuning.write_text('gap_follow_node:\n  ros__parameters:\n'
                      '    # why: different tyres\n    max_speed: 0.1\n')
    assert guards.check_tuning_overlay(tuning, layers.shared_config_path(tuning.name), tuning.name) == []
    monkeypatch.setenv('RACERBOT_CARS_DIR', str(tmp_path / 'cars'))
    monkeypatch.setattr(identity, 'SERIAL_PATH', str(tmp_path / 'missing'))
    code, text, _ = _run(['--no-live'], healthy_snap())
    assert code == 0
    assert text.splitlines()[:2] == [
        'Car rb1 (serial none), tuning overrides:',
        '  gap_follow.yaml: gap_follow_node.max_speed = 0.1  # different tyres']


def test_doctor_says_no_overrides_when_none(monkeypatch, tmp_path):
    # A copy of the profiles with car 1's tuning overlay removed: the shipped car 1 has one since 2026-11-14
    # (pure_pursuit.yaml, PR #63), and this test is about the no-overlay case, not about the shipped car.
    import shutil
    from racerbot_car import identity
    root = pathlib.Path(__file__).resolve().parents[3]
    shutil.copytree(root / 'src/racerbot_car/cars', tmp_path / 'cars')
    shutil.rmtree(tmp_path / 'cars/rb1/tuning', ignore_errors=True)
    monkeypatch.setenv('RACERBOT_CARS_DIR', str(tmp_path / 'cars'))
    monkeypatch.setattr(identity, 'SERIAL_PATH', str(tmp_path / 'missing'))
    code, text, _ = _run(['--no-live'], healthy_snap())
    assert code == 0
    assert text.splitlines()[0] == 'Car rb1 (serial none), tuning overrides: none'


def _run(argv, snap, live=None, sample_exc=None):
    out = io.StringIO()
    calls = {'gather': [], 'sample': []}

    def gather(**kw):
        calls['gather'].append(kw)
        return snap

    def sample(window_s, **kw):
        calls['sample'].append(window_s)
        if sample_exc:
            raise sample_exc
        return live

    argv = argv if '--car' in argv else ['--car', 'rb1', *argv]
    code = main(argv, gather_fn=gather, sample_fn=sample, out=out)
    return code, out.getvalue(), calls


def test_healthy_car_exits_zero_and_says_so():
    code, text, calls = _run(['--no-color'], healthy_snap(), healthy_live())
    assert code == 0
    assert re.search(r'^\s+FAIL ', text, re.M) is None
    assert '0 FAIL, 0 WARN' in text
    assert calls['sample'] == [3.0]           # car.py default window


def test_any_fail_exits_one_and_prints_the_fix():
    snap = healthy_snap()
    snap.vesc_symlink_rw = False
    code, text, _ = _run(['--no-color'], snap, healthy_live())
    assert code == 1
    assert re.search(r'^\s+FAIL ', text, re.M)
    assert 'usermod -aG dialout' in text


def test_camera_only_problem_does_not_fail_the_run():
    snap = healthy_snap()
    snap.usb = [d for d in snap.usb if 'uvcvideo' not in d.drivers]
    snap.camera_device = None
    code, text, _ = _run(['--no-color'], snap, healthy_live())
    assert code == 0
    assert 'WARN' in text


def test_window_flag_is_passed_to_the_sampler():
    _, _, calls = _run(['--window', '1.5'], healthy_snap(), healthy_live())
    assert calls['sample'] == [1.5]


def test_no_live_flag_never_samples():
    code, text, calls = _run(['--no-live', '--no-color'], healthy_snap())
    assert calls['sample'] == []
    assert 'SKIP' in text and code == 0


def test_bringup_down_does_not_sample_and_is_not_a_failure():
    snap = healthy_snap()
    for exe in car.BRINGUP_EXES + (car.EXE_JOY_TELEOP,):
        without(snap, exe)
    snap.tty_holders = {}
    code, text, calls = _run(['--no-color'], snap)
    assert calls['sample'] == []
    assert code == 0
    assert 'bringup_launch.py' in text


def test_without_rclpy_still_reports_hardware_and_says_source_ros():
    code, text, _ = _run(['--no-color'], healthy_snap(),
                         sample_exc=ImportError("No module named 'rclpy'"))
    assert code == 0                           # not 2: the doctor coped
    assert 'source /opt/ros/jazzy/setup.bash' in text
    assert re.search(r'SKIP .*/sensors/core rate', text)
    assert re.search(r'PASS .*USB device', text)


def test_sampler_crash_is_reported_not_raised():
    code, text, _ = _run(['--no-color'], healthy_snap(),
                         sample_exc=RuntimeError('rmw exploded'))
    assert code != 2
    assert 'rmw exploded' in text


def test_gather_crash_exits_two():
    out = io.StringIO()

    def gather(**kw):
        raise PermissionError('/proc')
    assert main(['--car', 'rb1'], gather_fn=gather, sample_fn=None, out=out) == 2
    assert 'PermissionError' in out.getvalue()


def test_gather_takes_no_lidar_probe_option():
    # Car 1's LiDAR is a USB tty: there is no TCP probe to opt into, and
    # the doctor never opens a tty (system.py).
    _, _, calls = _run([], healthy_snap(), healthy_live())
    from racerbot_car.profile import load_profile
    assert calls['gather'] == [{'spec': car.spec_for(load_profile('rb1'))}]


def test_teleop_alone_still_samples_the_graph():
    # joy_teleop counts as "something is up": its drive path is the point.
    snap = healthy_snap()
    for exe in car.BRINGUP_EXES:
        without(snap, exe)
    _, _, calls = _run(['--no-color'], snap, healthy_live())
    assert calls['sample'] == [3.0]


def test_render_no_color_has_no_escape_codes_and_color_does():
    reports = [DeviceReport('VESC', [Layer('x', Status.FAIL, 'bad', 'do y')])]
    assert '\x1b[' not in render(reports, color=False)
    assert '\x1b[' in render(reports, color=True)


def test_render_shows_device_detail_fix_and_summary_counts():
    reports = [
        DeviceReport('VESC', [Layer('USB device', Status.PASS, 'ok'),
                              Layer('kernel driver', Status.FAIL, 'no tty',
                                    'replug'),
                              Layer('rate', Status.SKIP, 'blocked by: kernel')]),
        DeviceReport('Camera', [Layer('USB speed', Status.WARN, 'USB 2')])]
    text = render(reports, color=False)
    assert 'VESC' in text and 'Camera' in text
    assert re.search(r'FAIL +kernel driver +no tty', text)
    assert 'fix: replug' in text
    assert '1 FAIL' in text and '1 WARN' in text
    assert 'VESC: kernel driver' in text       # summary names the broken layer


def test_render_collapses_a_run_of_blocked_layers_into_one_line():
    blocked = 'blocked by: USB device'
    reports = [DeviceReport('VESC', [
        Layer('USB device', Status.FAIL, 'not plugged in'),
        Layer('kernel driver', Status.SKIP, blocked),
        Layer('/dev/sensors/vesc', Status.SKIP, blocked),
        Layer('battery (3S)', Status.SKIP, blocked)])]
    lines = [l for l in render(reports, color=False).splitlines()
             if 'SKIP' in l]
    assert len(lines) == 1
    assert 'kernel driver' in lines[0] and 'battery (3S)' in lines[0]
    assert '3 layers' in lines[0] and blocked in lines[0]


def test_render_does_not_collapse_a_single_or_unrelated_skip():
    reports = [DeviceReport('VESC', [
        Layer('a', Status.SKIP, 'blocked by: x'),
        Layer('a2', Status.SKIP, 'blocked by: y'),     # different blocker
        Layer('b', Status.SKIP, 'live stage not run'),
        Layer('c', Status.SKIP, 'live stage not run')])]
    lines = [l for l in render(reports, color=False).splitlines()
             if 'SKIP' in l]
    assert len(lines) == 4


def test_package_never_publishes_or_opens_device_nodes():
    """The doctor is read-only by contract (system.py, live.py docstrings).
    Any of these in the package source means someone broke that."""
    banned = [r'create_publisher', r'create_client', r'call_async',
              r'set_parameters', r'\.publish\(', r'open\([^)]*/dev']
    hits = []
    for path in PKG.glob('*.py'):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            for pat in banned:
                if re.search(pat, line):
                    hits.append(f'{path.name}:{n}: {line.strip()}')
    assert hits == []
    assert len(list(PKG.glob('*.py'))) >= 7    # the scan saw the package


def test_car_override_selects_rb2_ladders(monkeypatch):
    monkeypatch.setenv('RACERBOT_CAR', 'rb1')
    from test_rb2_checks import healthy_snap as rb2_snap, live
    code, text, _ = _run(['--car', 'rb2', '--no-color'], rb2_snap(), live())
    assert code == 0
    assert 'Ethernet link' in text and 'realsense2_camera_node' in text
    assert '/dev/sensors/hokuyo' not in text


def test_identity_resolution_and_refusal(monkeypatch):
    from racerbot_car import identity
    monkeypatch.setattr(identity, 'SERIAL_PATH', '/nonexistent/doctor-test-serial')
    out = io.StringIO()
    monkeypatch.setenv('RACERBOT_CAR', 'rb1')
    assert main(['--no-live'], gather_fn=lambda **kw: healthy_snap(), out=out) == 0
    assert '/dev/sensors/hokuyo' in out.getvalue()
    monkeypatch.delenv('RACERBOT_CAR')
    out = io.StringIO()
    assert main([], out=out) == 2
    assert 'RACERBOT_CAR' in out.getvalue() and '--car' in out.getvalue()
    out = io.StringIO()
    assert main(['--car', 'unknown'], out=out) == 2
    assert 'rb1' in out.getvalue() and 'rb2' in out.getvalue()
