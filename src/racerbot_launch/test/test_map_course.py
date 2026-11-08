"""The map_course wrapper (scripts/map_course). No ROS needed:

    python3 -m pytest src/racerbot_launch/test/test_map_course.py -v

The one process boundary -- the ``ros2`` executable -- is replaced by a fake
script on PATH that behaves like a finished (or abandoned) mapping run, so
main() is exercised end to end.
"""
import importlib.machinery
import importlib.util
import json
import os
import stat
import sys
import time
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'map_course'


def _load():
    loader = importlib.machinery.SourceFileLoader('map_course', str(SCRIPT))
    spec = importlib.util.spec_from_loader('map_course', loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


mc = _load()


def _run_dir(parent, name, text, mtime=None):
    d = parent / name
    d.mkdir(parents=True)
    f = d / 'commands.txt'
    f.write_text(text)
    if mtime is not None:
        os.utime(f, (mtime, mtime))
    return d


def test_new_run_picks_the_newest_commands_written_since_start(tmp_path):
    now = time.time()
    _run_dir(tmp_path, 'old', 'old', mtime=now - 100)
    _run_dir(tmp_path, 'a', 'a', mtime=now + 1)
    newest = _run_dir(tmp_path, 'b', 'b', mtime=now + 2)
    (tmp_path / 'no_commands').mkdir()
    assert mc.new_run(tmp_path, now) == newest
    assert mc.new_run(tmp_path, now + 3) is None        # nothing that new
    assert mc.new_run(tmp_path / 'missing', now) is None


def test_output_directory_argument_is_honoured(tmp_path, monkeypatch):
    monkeypatch.setenv('RACERBOT_WS', str(tmp_path))
    assert mc.output_directory(['laps:=4', 'output_directory:=/x/y']) == Path('/x/y')
    assert mc.output_directory(['laps:=4']) == tmp_path / 'data/rb1/racerbot_auto'


def test_new_run_is_the_newest_whatever_order_the_directory_lists(tmp_path):
    # Ten runs written since start; the newest is neither first nor last by
    # name, so picking by listing order instead of by time is caught.
    now = time.time()
    stamps = [now + k for k in (3, 7, 1, 9, 5, 10, 2, 8, 4, 6)]
    dirs = [_run_dir(tmp_path, f'run{k:02d}', str(k), mtime=t)
            for k, t in enumerate(stamps)]
    assert mc.new_run(tmp_path, now) == dirs[stamps.index(max(stamps))]
    assert sorted(p.name for p in tmp_path.iterdir())[-1] != dirs[5].name


def test_report_without_a_new_run_says_nothing_was_saved(tmp_path):
    assert 'No map was saved' in mc.report(tmp_path, time.time())


def _fake_ros2(tmp_path, body):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    fake = bin_dir / 'ros2'
    fake.write_text('#!/bin/sh\n' + body)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _drivers(monkeypatch, *running):
    """Pretend exactly these driver processes are running (pgrep is the
    process boundary)."""
    monkeypatch.setattr(mc, '_running', lambda pattern: pattern in running)


def test_hardware_counts_as_up_only_with_every_real_driver_or_the_sim(monkeypatch):
    _drivers(monkeypatch, 'vesc_driver_node', 'urg_node_driver')
    assert mc.missing_drivers() == []
    _drivers(monkeypatch, 'vesc_driver_node')
    assert mc.missing_drivers() == ['urg_node_driver']
    _drivers(monkeypatch)
    assert mc.missing_drivers() == ['vesc_driver_node', 'urg_node_driver']
    _drivers(monkeypatch, 'gym_bridge_node')
    assert mc.missing_drivers() == []


def test_refuses_without_bringup_and_never_launches_anything(tmp_path, monkeypatch, capsys):
    called = tmp_path / 'called'
    bin_dir = _fake_ros2(tmp_path, f'touch {called}\nexit 0\n')
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    _drivers(monkeypatch, 'vesc_driver_node')          # LiDAR missing
    assert mc.main([f'output_directory:={tmp_path / "runs"}']) == 2
    assert not called.exists()
    err = capsys.readouterr().err
    assert 'ros2 launch f1tenth_stack bringup_launch.py' in err
    assert 'urg_node_driver' in err


def test_the_launch_file_starts_no_drivers():
    # Static oracle: map_course_launch.py must not include bringup in any
    # form, so a second copy of the drivers can never start beside it.
    text = (SCRIPT.parents[1] / 'launch' / 'map_course_launch.py').read_text()
    code = '\n'.join(line.split('#', 1)[0] for line in text.splitlines())
    for forbidden in ("'bringup_launch.py')", 'include_bringup', 'vesc_driver', 'urg_node'):
        assert forbidden not in code, forbidden


def test_main_passes_arguments_through_and_prints_the_commands_last(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('RACERBOT_WS', str(tmp_path))
    out = tmp_path / 'runs'
    args_file = tmp_path / 'args'
    bin_dir = _fake_ros2(tmp_path, (
        f'echo "$@" > {args_file}\n'
        f'mkdir -p {out}/20261022-120000\n'
        f'echo "RACE COMMANDS HERE" > {out}/20261022-120000/commands.txt\n'
        'echo "[INFO] nodes shutting down"\n'
        'exit 0\n'))
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    _drivers(monkeypatch, 'vesc_driver_node', 'urg_node_driver')
    code = mc.main([f'output_directory:={out}', 'laps:=4'])
    assert code == 0
    assert args_file.read_text().split() == [
        'launch', 'racerbot_launch', 'map_course_launch.py',
        f'output_directory:={out}', 'laps:=4']
    printed = capsys.readouterr().out
    # The commands are the last thing that matters on screen; only the log
    # path follows them.
    body, _, last = printed.rstrip().rpartition('\n')
    assert body.rstrip().endswith('RACE COMMANDS HERE')
    assert last.startswith('Full log of this run: ')
    log = Path(last.split(': ', 1)[1])
    saved = log.read_text()
    assert '[INFO] nodes shutting down' in saved and 'RACE COMMANDS HERE' in saved


def test_main_reports_no_map_and_the_launch_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('RACERBOT_WS', str(tmp_path))
    out = tmp_path / 'runs'
    bin_dir = _fake_ros2(tmp_path, 'exit 3\n')
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    _drivers(monkeypatch, 'vesc_driver_node', 'urg_node_driver')
    assert mc.main([f'output_directory:={out}']) == 3
    assert 'No map was saved' in capsys.readouterr().out


def test_main_refuses_arguments_that_are_not_launch_arguments(capsys):
    assert mc.main(['--laps', '3']) == 2
    assert 'name:=value' in capsys.readouterr().err


def test_the_recorder_gets_its_own_folder_not_the_map_run_folder():
    # Launch arguments are shared with included files: without an explicit
    # output_directory the recorder wrote into ~/.ros/racerbot_auto/ (found
    # on the car, 2026-10-22).
    for name in ('map_course_launch.py', 'auto_map_race_launch.py'):
        text = (SCRIPT.parents[1] / 'launch' / name).read_text()
        assert "'output_directory': str(data_dir('racerbot_runs', car_id))" in text, name


def test_simulator_runs_get_imu_hardware_false_unless_imu_hardware_was_given():
    # oracle: spec -- map_course.launch_arguments docstring; approved 2026-10-29
    # (I1 part b): simulated IMU stays on, only its hardware readers stay off.
    assert mc.launch_arguments(['laps:=4'], simulated=True) == ['laps:=4', 'imu_hardware:=false']
    assert mc.launch_arguments(['imu_hardware:=true'], simulated=True) == ['imu_hardware:=true']
    assert mc.launch_arguments(['imu:=false'], simulated=True) == ['imu:=false', 'imu_hardware:=false']
    assert mc.launch_arguments(['laps:=4'], simulated=False) == ['laps:=4']
    assert mc.launch_arguments([], simulated=False) == []


def test_main_in_the_simulator_launches_without_imu_hardware(tmp_path, monkeypatch, capsys):
    # oracle: spec -- map_course.launch_arguments docstring; approved 2026-10-29
    # (I1 part b): the real launch receives imu_hardware:=false, keeping the IMU.
    monkeypatch.setenv('RACERBOT_WS', str(tmp_path))
    out = tmp_path / 'runs'
    args_file = tmp_path / 'args'
    bin_dir = _fake_ros2(tmp_path, f'echo "$@" > {args_file}\nexit 0\n')
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    _drivers(monkeypatch, 'gym_bridge_node')
    assert mc.main([f'output_directory:={out}']) == 0
    assert args_file.read_text().split() == [
        'launch', 'racerbot_launch', 'map_course_launch.py',
        f'output_directory:={out}', 'imu_hardware:=false']


def _analysis_fixture(tmp_path, monkeypatch, recorder=True, fail=False):
    run = tmp_path / 'recorded run'
    run.mkdir()
    maps = tmp_path / 'maps'
    events = tmp_path / 'events'
    args = tmp_path / 'args'
    root = tmp_path / 'workspace'
    scripts = root / 'tools' / 'run_analysis'
    scripts.mkdir(parents=True)
    for name in ('imu_report', 'slam_report'):
        (scripts / f'{name}.py').write_text(
            'import sys\n'
            f'open({str(events)!r}, "a").write({name!r} + " " + repr(sys.argv[1:]) + "\\n")\n'
            f'print({name!r})\n'
            'print("\\n".join("line " + str(i) for i in range(20)))\n')
    bin_dir = _fake_ros2(tmp_path, (
        'if [ "$1" = run ]; then\n'
        f'  echo "summarize_run $4" >> "{events}"\n'
        '  echo summary-headline\n'
        f'  exit {7 if fail else 0}\n'
        'fi\n'
        f'printf "%s\\n" "$@" > "{args}"\n'
        f'mkdir -p "{maps}/latest"\n'
        f'echo RACE-COMMANDS > "{maps}/latest/commands.txt"\n'
        + (f'echo "Recording run to: {run}"\n' if recorder else '') +
        'exit 3\n'))
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    monkeypatch.setenv('RACERBOT_WS', str(root))
    _drivers(monkeypatch, 'gym_bridge_node')
    return run, maps, events, args, root


def test_analysis_reports_order_arguments_headlines_and_full_output(tmp_path, monkeypatch, capsys):
    # oracle: Task J1a specifies order, 15-line headlines, and commands last.
    run, maps, events, args, root = _analysis_fixture(tmp_path, monkeypatch)
    assert mc.main([f'output_directory:={maps}', 'analysis:=true']) == 3
    printed = capsys.readouterr().out
    log = Path(printed.strip().split('Full log of this run: ')[-1])
    lines = events.read_text().splitlines()
    assert lines == [f'summarize_run {run}', f'imu_report {[str(run)]!r}',
                     f'slam_report {[str(run), "--log", str(log)]!r}']
    for name in ('imu_report', 'slam_report'):
        target = run / 'analysis' / 'map_course' / f'{name}.txt'
        assert target.read_text().splitlines() == [name] + [f'line {i}' for i in range(20)]
        assert '\n'.join(target.read_text().splitlines()[:15]) in printed
        assert 'line 14' not in printed
        assert str(target) in printed
        assert printed.index(str(target)) < printed.index('RACE-COMMANDS')
    assert 'summary-headline' in printed
    assert printed.index('summary-headline') < printed.index('RACE-COMMANDS')
    assert 'RACE-COMMANDS' in log.read_text()


def test_analysis_disabled_is_stripped_and_runs_no_reports(tmp_path, monkeypatch, capsys):
    # oracle: Task J1a: wrapper-only option and no subprocess reports.
    run, maps, events, args, root = _analysis_fixture(tmp_path, monkeypatch)
    assert mc.main([f'output_directory:={maps}', 'analysis:=false']) == 3
    assert 'analysis:=false' not in args.read_text()
    assert not events.exists()
    assert not (run / 'analysis').exists()


def test_analysis_invalid_values_never_launch(tmp_path, monkeypatch, capsys):
    # oracle: Task J1a: only true/false; empty is invalid too.
    run, maps, events, args, root = _analysis_fixture(tmp_path, monkeypatch)
    for value in ('maybe', ''):
        assert mc.main([f'analysis:={value}']) == 2
        assert 'analysis' in capsys.readouterr().err
        assert not args.exists()


def test_analysis_without_recording_reports_skip_and_keeps_exit(tmp_path, monkeypatch, capsys):
    # oracle: Task J1a: absent recorder banner skips analysis and preserves launch status.
    run, maps, events, args, root = _analysis_fixture(tmp_path, monkeypatch, recorder=False)
    assert mc.main([f'output_directory:={maps}', 'analysis:=true']) == 3
    assert 'Analysis skipped' in capsys.readouterr().out
    assert not events.exists()


def test_analysis_failure_reports_command_without_changing_launch_status(tmp_path, monkeypatch, capsys):
    # oracle: Task J1a: report failures are advisory.
    run, maps, events, args, root = _analysis_fixture(tmp_path, monkeypatch, fail=True)
    assert mc.main([f'output_directory:={maps}', 'analysis:=true']) == 3
    output = capsys.readouterr().out
    assert 'exit 7' in output
    assert f"ros2 run race_diagnostics summarize_run '{run}'" in output
    assert 'slam_report' in output


def _interrupt_analysis_at(boundary, tmp_path, monkeypatch, capsys):
    import builtins
    import pytest
    run, maps, events, _, _ = _analysis_fixture(tmp_path, monkeypatch)
    write_text, print_text = Path.write_text, builtins.print
    interrupted = []

    def write(path, *args, **kwargs):
        if boundary == 'save' and path.name == 'summarize_run.txt':
            interrupted.append('save')
            raise KeyboardInterrupt
        return write_text(path, *args, **kwargs)

    def emit(*args, **kwargs):
        if boundary == 'print' and args and str(args[0]).startswith('summarize_run:'):
            interrupted.append('print')
            raise KeyboardInterrupt
        return print_text(*args, **kwargs)

    monkeypatch.setattr(Path, 'write_text', write)
    monkeypatch.setattr(builtins, 'print', emit)
    try:
        code = mc.main([f'output_directory:={maps}', 'analysis:=true'])
    except KeyboardInterrupt:
        pytest.fail(f'analysis {boundary} interrupt escaped before final commands')
    output = capsys.readouterr().out
    # oracle: spec R5a: keep launch status and final commands; skip all
    # remaining analyses after an interrupt anywhere in run/save/print.
    assert code == 3
    assert interrupted == [boundary]
    assert events.read_text().splitlines() == [f'summarize_run {run}']
    assert 'RACE-COMMANDS' in output
    body, _, last = output.rstrip().rpartition('\n')
    assert body.rstrip().endswith('RACE-COMMANDS')
    assert last.startswith('Full log of this run: ')
    log = Path(output.strip().split('Full log of this run: ')[-1])
    assert 'RACE-COMMANDS' in log.read_text()


def test_analysis_save_interrupt_preserves_commands_and_launch_status(tmp_path, monkeypatch, capsys):
    _interrupt_analysis_at('save', tmp_path, monkeypatch, capsys)


def test_analysis_print_interrupt_preserves_commands_and_launch_status(tmp_path, monkeypatch, capsys):
    _interrupt_analysis_at('print', tmp_path, monkeypatch, capsys)


def _rebuild_fixture(tmp_path, monkeypatch, body=None, **kwargs):
    """_analysis_fixture plus a stand-in rebuild script, recording how it was called."""
    run, maps, events, args, root = _analysis_fixture(tmp_path, monkeypatch, **kwargs)
    body = body or (
        'from pathlib import Path\n'
        'folder = Path(sys.argv[3]) / "rebuilt"\n'
        'folder.mkdir(parents=True, exist_ok=True)\n'
        '(folder / "commands.txt").write_text("REBUILT-COMMANDS")\n'
        'print("GO: rebuilt")\n')
    (root / 'tools' / 'run_analysis' / 'rebuild_run.py').write_text(
        'import sys, time\n'
        f'open({str(events)!r}, "a").write("rebuild_run " + repr(sys.argv[1:]) + "\\n")\n'
        'time.sleep(0.02)\n' + body)
    return run, maps, events, args, root


def test_every_recorded_run_is_rebuilt_before_the_reports(tmp_path, monkeypatch, capsys):
    # oracle: invariant. A run stopped by Ctrl+C has the same recording, so it gets the same step, and the
    # verdict comes first because the operator may not wait for a report.
    run, maps, events, _, _ = _rebuild_fixture(tmp_path, monkeypatch)
    assert mc.main([f'output_directory:={maps}']) == 3
    assert events.read_text().splitlines()[0] == f'rebuild_run {[str(run), "--out-root", str(maps)]!r}'
    assert 'GO: rebuilt' in capsys.readouterr().out


def test_the_rebuilt_folder_is_what_the_operator_is_told_to_race(tmp_path, monkeypatch, capsys):
    # oracle: invariant. The node's own folder is older than the rebuild's, and its line came from the live
    # pose, so the last thing printed is the rebuild's commands.
    _, maps, _, _, _ = _rebuild_fixture(tmp_path, monkeypatch)
    mc.main([f'output_directory:={maps}'])
    output = capsys.readouterr().out
    assert 'REBUILT-COMMANDS' in output
    assert 'RACE-COMMANDS' not in output


def test_a_run_with_no_recording_is_not_rebuilt_and_keeps_its_exit(tmp_path, monkeypatch, capsys):
    _, maps, events, _, _ = _rebuild_fixture(tmp_path, monkeypatch, recorder=False)
    assert mc.main([f'output_directory:={maps}']) == 3
    assert 'Map check skipped' in capsys.readouterr().out
    assert not events.exists()


def test_rebuild_off_is_stripped_and_never_runs(tmp_path, monkeypatch, capsys):
    run, maps, events, args, _ = _rebuild_fixture(tmp_path, monkeypatch)
    assert mc.main([f'output_directory:={maps}', 'rebuild:=false', 'analysis:=true']) == 3
    assert 'rebuild:=false' not in args.read_text()
    assert 'rebuild_run' not in events.read_text()
    assert 'summarize_run' in events.read_text()  # the reports are a separate switch


def test_rebuild_invalid_value_never_launches(tmp_path, monkeypatch, capsys):
    _, _, _, args, _ = _rebuild_fixture(tmp_path, monkeypatch)
    assert mc.main(['rebuild:=maybe']) == 2
    assert 'rebuild' in capsys.readouterr().err
    assert not args.exists()


def test_a_failed_rebuild_names_the_command_and_changes_nothing_else(tmp_path, monkeypatch, capsys):
    # oracle: invariant. Advisory like the reports: the launch's exit stands and the reports still run.
    run, maps, events, _, _ = _rebuild_fixture(tmp_path, monkeypatch, body='sys.exit(7)\n')
    assert mc.main([f'output_directory:={maps}', 'analysis:=true']) == 3
    output = capsys.readouterr().out
    assert 'exit 7' in output
    assert 'rebuild_run.py' in output
    assert 'summarize_run' in events.read_text()


def test_a_refusal_is_not_a_failure(tmp_path, monkeypatch, capsys):
    # Exit 1 is the tool's own answer, "this map is no good", and it says so itself.
    _, maps, _, _, _ = _rebuild_fixture(tmp_path, monkeypatch, body='print("NO-GO: doubled")\nsys.exit(1)\n')
    mc.main([f'output_directory:={maps}'])
    output = capsys.readouterr().out
    assert 'NO-GO: doubled' in output
    assert 'Map check failed' not in output


def test_ctrl_c_during_the_rebuild_still_reports_and_keeps_the_exit(tmp_path, monkeypatch, capsys):
    # oracle: review focus 5. A second Ctrl+C means stop now, so the reports are skipped too, but the launch's
    # status and the last-thing-on-screen report survive, with the command to finish the job later.
    run, maps, events, _, _ = _rebuild_fixture(tmp_path, monkeypatch)
    real_run = mc.subprocess.run

    def interrupted(command, *args, **kwargs):
        if any('rebuild_run' in str(part) for part in command):
            raise KeyboardInterrupt
        return real_run(command, *args, **kwargs)
    monkeypatch.setattr(mc.subprocess, 'run', interrupted)
    assert mc.main([f'output_directory:={maps}', 'analysis:=true']) == 3
    output = capsys.readouterr().out
    assert 're-run:' in output and 'rebuild_run.py' in output
    assert not events.exists()
    assert 'RACE-COMMANDS' in output


def test_the_rebuild_runs_in_its_own_session(tmp_path, monkeypatch, capsys):
    # A second Ctrl+C at the terminal goes to the whole foreground group: it must reach only the wrapper, which
    # decides what stops. Asserted on the call, since that is the process boundary.
    _, maps, _, _, _ = _rebuild_fixture(tmp_path, monkeypatch)
    real_run, seen = mc.subprocess.run, {}

    def spy(command, *args, **kwargs):
        if any('rebuild_run' in str(part) for part in command):
            seen.update(kwargs)
        return real_run(command, *args, **kwargs)
    monkeypatch.setattr(mc.subprocess, 'run', spy)
    mc.main([f'output_directory:={maps}'])
    assert seen['start_new_session'] is True


class _FakeLaunch:
    """A process boundary: `ros2 launch`, whose output stream can be interrupted by Ctrl+C partway."""

    def __init__(self, lines, interrupt_after=None, code=0, interrupts=1):
        self.lines, self.interrupt_after, self.code, self.interrupts = lines, interrupt_after, code, interrupts
        self.position = 0
        self.stdout = self

    def __iter__(self):
        return self

    def __next__(self):
        if self.position == self.interrupt_after and self.interrupts:
            self.interrupts -= 1
            raise KeyboardInterrupt
        if self.position >= len(self.lines):
            raise StopIteration
        self.position += 1
        return self.lines[self.position - 1] + '\n'

    def wait(self):
        return self.code


@pytest.mark.parametrize('outcome', ['success', 'failed', 'cannot_start'])
def test_builder_runs_after_shutdown_and_preserves_node_profile_on_failure(
        tmp_path, monkeypatch, capsys, outcome):
    # oracle: spec, offline process starts only after wait(), before commands print;
    # a non-zero builder must preserve the exact node profile, even after a partial write.
    maps = tmp_path / 'maps'
    run = _run_dir(maps, 'node-run', 'RACE-COMMANDS')
    profile = run / 'raceline_profiled.csv'
    original = b'x,y,speed\n0,0,0.8\n'
    profile.write_bytes(original)
    shutdown = tmp_path / 'shutdown'

    class Launch(_FakeLaunch):
        def wait(self):
            shutdown.touch()
            return 0

    launch = Launch([])
    real_popen = mc.subprocess.Popen
    seen = []

    def popen(command, *args, **kwargs):
        if command[:2] == ['ros2', 'launch']:
            return launch
        assert shutdown.exists(), 'builder started while mapping still ran'
        assert command[1:3] == ['-m', 'pure_pursuit.race_line']
        assert command[3:5] == ['--run-dir', str(run)]
        assert command[5:] == ['--config', str(tmp_path / 'custom.yaml')]
        seen.append(command)
        if outcome == 'cannot_start':
            raise OSError('builder unavailable')
        script = (f'from pathlib import Path; Path({str(profile)!r}).write_text("built"); '
                  f'Path({str(run / "raceline_floor_test.csv")!r}).write_text("partial floor"); '
                  'print("choice: recorded; chosen recorded: 12.000 s"); '
                  f'raise SystemExit({1 if outcome == "failed" else 0})')
        return real_popen([sys.executable, '-c', script], *args, **kwargs)

    monkeypatch.setattr(mc.subprocess, 'Popen', popen)
    monkeypatch.setattr(mc, 'missing_drivers', lambda: [])
    monkeypatch.setattr(mc, '_running', lambda _: False)
    monkeypatch.setenv('RACERBOT_WS', str(tmp_path))
    assert mc.main([f'output_directory:={maps}', 'rebuild:=false',
                    f'supervisor_config:={tmp_path / "custom.yaml"}']) == 0
    text = capsys.readouterr().out
    assert len(seen) == 1
    assert profile.read_bytes() == (b'built' if outcome == 'success' else original)
    assert (run / 'raceline_floor_test.csv').exists() is (outcome == 'success')
    summary = 'Race line: choice:' if outcome == 'success' else "kept the node's profile"
    assert summary in text
    assert text.index(summary) < text.index('RACE-COMMANDS')


def _only_the_launch_is_fake(monkeypatch, launch):
    """The rebuild and the reports stay real child processes: only `ros2 launch` is replaced."""
    real = mc.subprocess.Popen

    def popen(command, *args, **kwargs):
        return launch if command[:2] == ['ros2', 'launch'] else real(command, *args, **kwargs)
    monkeypatch.setattr(mc.subprocess, 'Popen', popen)


LAP_LINE = ('[auto_map_race_node-9] [INFO] [1.0] [auto_map_race_node]: Closed mapping lap {n}/3 detected '
            '({m}m, 365deg of turning, 1.01 revolutions, 219 samples, closest approach to the start 0.1m)')


def _reports_fixture(tmp_path, monkeypatch, laps=2, interrupt=True, verdict=None, code=0):
    """A run through main() with a stand-in launch and rebuild, and every report script present."""
    run, maps, events, _, root = _rebuild_fixture(tmp_path, monkeypatch)
    lines = [f'Recording run to: {run}'] + [LAP_LINE.format(n=n + 1, m=50.5) for n in range(laps)]
    launch = _FakeLaunch(lines, interrupt_after=len(lines) if interrupt else None, code=code)
    _only_the_launch_is_fake(monkeypatch, launch)
    if verdict is not None:
        folder = maps / f'{run.name}-rebuilt'
        folder.mkdir(parents=True)
        (folder / 'rebuild.json').write_text(json.dumps(verdict))
    return run, maps, events


def test_a_run_that_finishes_by_itself_just_gives_the_commands(tmp_path, monkeypatch, capsys):
    # oracle: user's request, 2026-10-30. Three laps done: the map check that the commands depend on, then the
    # commands. No reports and no summary.
    _, maps, events = _reports_fixture(tmp_path, monkeypatch, laps=3, interrupt=False)
    assert mc.main([f'output_directory:={maps}']) == 0
    output = capsys.readouterr().out
    assert 'REBUILT-COMMANDS' in output
    assert 'RUN STOPPED' not in output
    assert 'summarize_run' not in events.read_text()
    assert 'imu_report' not in events.read_text()
    assert 'slam_report' not in events.read_text()


def test_ctrl_c_finishes_processing_and_gives_a_quick_report(tmp_path, monkeypatch, capsys):
    # oracle: user's request, 2026-10-30. The map check still runs, no long reports do, and the summary comes
    # before the commands, which stay the last thing on screen.
    run, maps, events = _reports_fixture(tmp_path, monkeypatch, laps=2)
    assert mc.main([f'output_directory:={maps}']) == 0
    output = capsys.readouterr().out
    assert events.read_text().splitlines()[0].startswith('rebuild_run')
    assert 'summarize_run' not in events.read_text() and 'slam_report' not in events.read_text()
    assert 'RUN STOPPED' in output
    assert 'Laps closed: 2' in output
    assert output.index('RUN STOPPED') < output.index('REBUILT-COMMANDS')


def test_the_quick_report_says_what_the_map_check_found(tmp_path, monkeypatch, capsys):
    run, maps, _ = _reports_fixture(tmp_path, monkeypatch, verdict={
        'verdict': {'ok': True, 'mean_fit': 0.978, 'see_through': 0.05}, 'raceline_ok': True, 'laps': 5,
        'tracked_path_m': 261.4, 'lap_lengths_m': [50.4, 50.8], 'message': 'map OK', 'odom_source': '/odom/imu'})
    # the stand-in rebuild would refuse an existing folder in the real tool; this one just writes commands
    mc.main([f'output_directory:={maps}'])
    output = capsys.readouterr().out
    assert 'GO' in output
    assert '261' in output         # metres, the distance the scans place the car over
    assert '/odom/imu' in output
    assert '5 whole lap' in output


def test_the_quick_report_says_no_go_and_why(tmp_path, monkeypatch, capsys):
    run, maps, _ = _reports_fixture(tmp_path, monkeypatch, verdict={
        'verdict': {'ok': False, 'mean_fit': 0.97, 'see_through': 0.19}, 'raceline_ok': False, 'laps': 0,
        'message': 'see-through 19% is over 15%', 'odom_source': '/odom/imu'})
    mc.main([f'output_directory:={maps}'])
    output = capsys.readouterr().out
    assert 'NO-GO' in output
    assert 'see-through 19%' in output


def test_the_quick_report_says_how_to_get_the_full_reports(tmp_path, monkeypatch, capsys):
    run, maps, events = _reports_fixture(tmp_path, monkeypatch)
    mc.main([f'output_directory:={maps}'])
    output = capsys.readouterr().out
    assert str(run) in output
    assert 'slam_report.py' in output
    assert 'slam_report' not in events.read_text()  # named, not run


def test_the_quick_report_survives_no_map_check(tmp_path, monkeypatch, capsys):
    # A stopped run with rebuild:=false still gets its laps, and is told the map was not checked.
    _, maps, _ = _reports_fixture(tmp_path, monkeypatch, laps=1)
    mc.main([f'output_directory:={maps}', 'rebuild:=false'])
    output = capsys.readouterr().out
    assert 'Laps closed: 1' in output
    assert 'not checked' in output


def test_the_quick_report_survives_a_damaged_map_check_file(tmp_path, monkeypatch, capsys):
    # oracle: invariant. The last thing on screen is the commands, whatever the summary could not read.
    run, maps, _ = _reports_fixture(tmp_path, monkeypatch)
    folder = maps / f'{run.name}-rebuilt'
    folder.mkdir(parents=True)
    (folder / 'rebuild.json').write_text('{not json')
    assert mc.main([f'output_directory:={maps}']) == 0
    output = capsys.readouterr().out
    assert 'not checked' in output
    assert 'REBUILT-COMMANDS' in output


def test_a_launch_that_fails_by_itself_also_gets_the_quick_report(tmp_path, monkeypatch, capsys):
    # A run that dies with no Ctrl+C is the one most in need of an explanation.
    _, maps, _ = _reports_fixture(tmp_path, monkeypatch, laps=1, interrupt=False, code=1)
    assert mc.main([f'output_directory:={maps}']) == 1
    assert 'Laps closed: 1' in capsys.readouterr().out


def test_ctrl_c_says_at_once_that_it_is_stopping(tmp_path, monkeypatch, capsys):
    # What it is doing after Ctrl+C was invisible, which looked like it was still running.
    _, maps, _ = _reports_fixture(tmp_path, monkeypatch)
    mc.main([f'output_directory:={maps}'])
    output = capsys.readouterr().out
    assert 'Stopping' in output
    assert output.index('Stopping') < output.index('Checking the map')


def test_a_second_ctrl_c_while_stopping_does_not_repeat_itself(tmp_path, monkeypatch, capsys):
    run, maps, _, _, _ = _rebuild_fixture(tmp_path, monkeypatch)
    _only_the_launch_is_fake(monkeypatch, _FakeLaunch([f'Recording run to: {run}'], interrupt_after=1, interrupts=2))
    mc.main([f'output_directory:={maps}'])
    assert capsys.readouterr().out.count('Stopping') == 1


def test_simulator_probe_matches_executable_not_shell_text(monkeypatch):
    import re
    from types import SimpleNamespace

    for command, expected in [
        ('/ws/install/racerbot_sim/lib/racerbot_sim/gym_bridge_node --ros-args', True),
        ('/usr/bin/python3 /ws/install/racerbot_sim/lib/racerbot_sim/gym_bridge_node', True),
        ('/bin/bash -c wait-for gym_bridge_node', False),
        ('/bin/bash -c wait-for /ws/install/racerbot_sim/lib/racerbot_sim/gym_bridge_node', False),
    ]:
        def probe(argv, **kwargs):
            assert argv[:2] == ['pgrep', '-f']
            return SimpleNamespace(returncode=0 if re.search(argv[2], command) else 1)
        monkeypatch.setattr(mc.subprocess, 'run', probe)
        # oracle: spec R5b: only an executable/script process argument counts.
        assert mc._running(mc.SIM_DRIVER) is expected, command
