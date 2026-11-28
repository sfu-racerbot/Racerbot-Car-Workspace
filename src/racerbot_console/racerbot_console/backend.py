"""Gathers everything the screens show into one Snapshot.

The app asks the backend for a snapshot a few times a second and never
touches /proc, /sys or ROS itself. Tests give the app a FakeBackend with
chosen readings instead, so every screen can be checked without a car.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from . import layers as layers_mod
from . import sysinfo
from .ros_reader import Live, RosReader


@dataclass(frozen=True)
class Snapshot:
    car_id: str | None
    car_error: str | None
    live: Live
    layers: layers_mod.Layers
    jetson: sysinfo.Jetson


@dataclass
class Backend:
    workspace: Path
    proc_root: str = sysinfo.PROC_ROOT
    sys_root: str = sysinfo.SYS_ROOT
    reader: RosReader | None = None
    _cpu: sysinfo.CpuMeter = field(init=False)
    _slow: tuple | None = field(default=None, init=False)
    _slow_t: float = field(default=-1e9, init=False)
    car_id: str | None = field(default=None, init=False)
    car_error: str | None = field(default=None, init=False)

    SLOW_PERIOD = 1.0        # seconds between /proc scans and /sys reads

    def __post_init__(self):
        self._cpu = sysinfo.CpuMeter(self.proc_root)
        try:
            from racerbot_car import identity
            self.car_id = identity.current_car_id()
        except Exception as exc:  # noqa: BLE001 -- unknown car is shown, not fatal
            self.car_error = str(exc)

    def start(self) -> None:
        if self.reader is None:
            self.reader = RosReader()
        self.reader.start()

    def stop(self) -> None:
        if self.reader is not None:
            self.reader.stop()

    def _slow_parts(self):
        from racerbot_doctor.system import read_processes
        procs = read_processes(self.proc_root)
        jetson = sysinfo.Jetson(
            cpu=self._cpu.percent(), gpu=sysinfo.gpu_percent(self.sys_root),
            mem=sysinfo.memory(self.proc_root), temp=sysinfo.hottest_c(self.sys_root),
            disk_free=sysinfo.disk_free_gb(str(self.workspace)))
        return layers_mod.classify(procs), jetson

    def snapshot(self) -> Snapshot:
        t = time.monotonic()
        if self._slow is None or t - self._slow_t >= self.SLOW_PERIOD:
            self._slow = self._slow_parts()
            self._slow_t = t
        live = self.reader.snapshot() if self.reader else Live(now=t, error='ROS reader not started')
        layers, jetson = self._slow
        return Snapshot(self.car_id, self.car_error, live, layers, jetson)


class FakeBackend:
    """Returns whatever snapshot a test sets. `workspace` is used by the Runs
    screen and the job runner."""

    def __init__(self, workspace: Path, snapshot: Snapshot):
        self.workspace = workspace
        self.value = snapshot

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def snapshot(self) -> Snapshot:
        return self.value
