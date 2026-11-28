"""Programs the menu runs for you, with their output kept for the Logs screen.

Step 1 runs only jobs that cannot move the car: the hardware check and the
run reports. `JOBS` lists every one, and `start` refuses anything not in it,
so a typo or a later edit can't turn this into a way to start a driving
layer before the Get ready checks exist (design doc, build plan step 2).

Each job runs in its own process group with its output saved to a log file,
so stopping it stops its children too and nothing is lost when the menu
closes.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JobSpec:
    key: str
    title: str
    argv: tuple[str, ...]          # `{run}` is replaced by a run folder
    moves_car: bool = False
    needs_run: bool = False


JOBS = {
    spec.key: spec for spec in (
        JobSpec('doctor', 'Hardware check',
                ('ros2', 'run', 'racerbot_doctor', 'doctor', '--no-color')),
        JobSpec('summary', 'Run summary',
                ('ros2', 'run', 'race_diagnostics', 'summarize_run', '{run}'), needs_run=True),
        JobSpec('slam', 'SLAM report',
                ('python3', 'tools/run_analysis/slam_report.py', '{run}'), needs_run=True),
        JobSpec('imu', 'IMU report',
                ('python3', 'tools/run_analysis/imu_report.py', '{run}'), needs_run=True),
    )
}


class JobRefused(Exception):
    pass


def build_argv(spec: JobSpec, run: Path | None = None) -> list[str]:
    if spec.moves_car:
        raise JobRefused(f'{spec.title} can move the car; the menu cannot start it yet')
    if spec.needs_run:
        if run is None:
            raise JobRefused(f'{spec.title} needs a run folder')
        return [str(run) if a == '{run}' else a for a in spec.argv]
    if any('{run}' in a for a in spec.argv):
        raise JobRefused(f'{spec.title} has an unfilled run folder')
    return list(spec.argv)


class Job:
    """One running (or finished) program."""

    MAX_LINES = 5000

    def __init__(self, spec: JobSpec, argv: list[str], *, cwd: Path, log_path: Path,
                 popen=subprocess.Popen):
        self.spec = spec
        self.argv = argv
        self.cwd = cwd
        self.log_path = log_path
        self.lines: deque[str] = deque(maxlen=self.MAX_LINES)
        self.started = time.time()
        self.returncode: int | None = None
        self._lock = threading.Lock()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, 'a', buffering=1)
        self._log.write(f'$ {" ".join(argv)}\n')
        if not Path(cwd).is_dir():
            self._proc = None
            self.returncode = 127
            self._add(f'could not start {argv[0]}: the workspace {cwd} does not exist '
                      '(set RACERBOT_WS)')
            self._log.close()
            return
        try:
            self._proc = popen(argv, cwd=str(cwd), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               text=True, bufsize=1, start_new_session=True)
        except OSError as exc:
            self._proc = None
            self.returncode = 127
            self._add(f'could not start {argv[0]}: {exc}')
            self._log.close()
            return
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _add(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)
        try:
            self._log.write(line + '\n')
        except ValueError:          # log already closed
            pass

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for raw in self._proc.stdout:
            self._add(raw.rstrip('\n'))
        self.returncode = self._proc.wait()
        self._add(f'[exit {self.returncode}]')
        self._log.close()

    @property
    def running(self) -> bool:
        return self._proc is not None and self.returncode is None and self._proc.poll() is None

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self.lines)

    def stop(self, *, grace: float = 5.0) -> None:
        """Ctrl+C the whole group, then SIGTERM, then SIGKILL."""
        if not self.running:
            return
        pgid = self._proc.pid
        for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 2.0), (signal.SIGKILL, 2.0)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            try:
                self._proc.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
        # The leader is gone; a child that ignored the signals may not be.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class JobRunner:
    def __init__(self, workspace: Path, log_dir: Path, *, popen=subprocess.Popen):
        self.workspace = workspace
        self.log_dir = log_dir
        self.popen = popen
        self.jobs: list[Job] = []

    def start(self, key: str, run: Path | None = None) -> Job:
        spec = JOBS.get(key)
        if spec is None:
            raise JobRefused(f'unknown job {key!r}')
        argv = build_argv(spec, run)
        stamp = time.strftime('%Y%m%d-%H%M%S')
        job = Job(spec, argv, cwd=self.workspace,
                  log_path=self.log_dir / f'{stamp}-{key}.log', popen=self.popen)
        self.jobs.append(job)
        return job

    def stop_all(self) -> None:
        for job in self.jobs:
            job.stop()
