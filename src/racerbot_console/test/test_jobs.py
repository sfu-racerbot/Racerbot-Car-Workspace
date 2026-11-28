"""jobs.py: only listed, non-driving jobs start; output is kept; stop
stops the whole process group."""
import subprocess
import sys
import time

import pytest

from racerbot_console import jobs


def fake_popen(script):
    """A Popen that runs `script` with this Python instead of the real argv."""
    def popen(argv, **kw):
        return subprocess.Popen([sys.executable, '-c', script], **kw)
    return popen


def wait_done(job, timeout=10.0):
    end = time.monotonic() + timeout
    while job.running and time.monotonic() < end:
        time.sleep(0.02)
    time.sleep(0.1)


def test_unknown_job_is_refused(tmp_path):
    with pytest.raises(jobs.JobRefused, match='unknown job'):
        jobs.JobRunner(tmp_path, tmp_path).start('gap_follow')


def test_a_job_that_can_move_the_car_is_refused():
    spec = jobs.JobSpec('x', 'Gap follow', ('ros2', 'launch', 'gap_follow', 'gap_follow_launch.py'),
                        moves_car=True)
    with pytest.raises(jobs.JobRefused, match='can move the car'):
        jobs.build_argv(spec)


def test_a_run_report_needs_a_run_and_fills_it_in(tmp_path):
    spec = jobs.JOBS['summary']
    with pytest.raises(jobs.JobRefused, match='needs a run folder'):
        jobs.build_argv(spec)
    assert jobs.build_argv(spec, tmp_path)[-1] == str(tmp_path)


def test_unfilled_placeholder_is_refused():
    spec = jobs.JobSpec('x', 'X', ('echo', '{run}'))
    with pytest.raises(jobs.JobRefused, match='unfilled'):
        jobs.build_argv(spec)


def test_output_is_kept_and_logged(tmp_path):
    runner = jobs.JobRunner(tmp_path, tmp_path / 'logs',
                            popen=fake_popen('print("  PASS  usb"); print("0 FAIL, 0 WARN")'))
    job = runner.start('doctor')
    wait_done(job)
    assert job.returncode == 0
    assert job.snapshot() == ['  PASS  usb', '0 FAIL, 0 WARN', '[exit 0]']
    log = job.log_path.read_text()
    assert log.startswith('$ ros2 run racerbot_doctor doctor --no-color\n')
    assert '0 FAIL, 0 WARN' in log


def test_a_program_that_is_not_installed_is_a_finished_job(tmp_path):
    def missing(argv, **kw):
        raise FileNotFoundError(2, 'No such file', argv[0])
    job = jobs.JobRunner(tmp_path, tmp_path, popen=missing).start('doctor')
    assert not job.running and job.returncode == 127
    assert 'could not start ros2' in job.snapshot()[0]


def test_stop_reaches_children_that_ignore_ctrl_c(tmp_path):
    script = ('import signal, subprocess, sys, time\n'
              'signal.signal(signal.SIGINT, signal.SIG_IGN)\n'
              'subprocess.Popen([sys.executable, "-c", '
              '"import signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); '
              'signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])\n'
              'print("up", flush=True); time.sleep(60)\n')
    job = jobs.JobRunner(tmp_path, tmp_path, popen=fake_popen(script)).start('doctor')
    end = time.monotonic() + 10
    while 'up' not in job.snapshot() and time.monotonic() < end:
        time.sleep(0.02)
    pgid = job._proc.pid
    t0 = time.monotonic()
    job.stop(grace=0.3)
    assert not job.running
    assert time.monotonic() - t0 < 5.0
    time.sleep(0.2)
    out = subprocess.run(['ps', '-o', 'stat=', '-g', str(pgid)], capture_output=True, text=True)
    alive = [st for st in out.stdout.split() if not st.startswith('Z')]   # Z = exited, unreaped
    assert alive == []                       # nothing left running in the group


def test_a_missing_workspace_is_named_not_mistaken_for_a_missing_program(tmp_path):
    started = []
    job = jobs.JobRunner(tmp_path / 'no-ws', tmp_path,
                         popen=lambda *a, **k: started.append(a)).start('doctor')
    assert started == [] and job.returncode == 127
    assert 'does not exist' in job.snapshot()[0]
