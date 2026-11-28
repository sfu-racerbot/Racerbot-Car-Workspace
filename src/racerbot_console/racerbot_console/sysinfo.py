"""Jetson load from /proc and /sys: CPU, memory, temperature, GPU, disk.

Plain file reads, no tegrastats subprocess, so the menu costs the car almost
nothing. Every root is a module-level path or an argument, so tests point
them at a temporary folder.
"""
from __future__ import annotations

import glob
import os
import shutil
from dataclasses import dataclass

PROC_ROOT = '/proc'
SYS_ROOT = '/sys'

# Orin Nano's GPU load, in tenths of a percent (0..1000). The first path
# that exists is used; on JetPack 6/7 it is the 17000000.gpu one.
GPU_LOAD_PATHS = (
    'devices/platform/17000000.gpu/load',
    'devices/platform/gpu.0/load',
    'devices/gpu.0/load',
)

# Thermal zones that are offline on a Jetson read -256000 (millidegrees);
# anything outside this range is not a temperature.
TEMP_MIN_C = -40.0
TEMP_MAX_C = 150.0


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def parse_cpu_times(stat_text: str | None) -> tuple[int, int] | None:
    """(busy, total) jiffies from the first line of /proc/stat."""
    if not stat_text:
        return None
    first = stat_text.splitlines()[0].split()
    if not first or first[0] != 'cpu' or len(first) < 5:
        return None
    try:
        values = [int(v) for v in first[1:]]
    except ValueError:
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)   # idle + iowait
    total = sum(values[:8])          # guest time is already in user/nice
    return total - idle, total


class CpuMeter:
    """CPU use since the previous call, in percent of all cores."""

    def __init__(self, proc_root: str | None = None):
        self.proc_root = proc_root or PROC_ROOT
        self._last = None

    def percent(self) -> float | None:
        now = parse_cpu_times(_read(os.path.join(self.proc_root, 'stat')))
        if now is None:
            return None
        last, self._last = self._last, now
        if last is None:
            return None
        busy, total = now[0] - last[0], now[1] - last[1]
        if total <= 0 or busy < 0:
            return None
        return 100.0 * busy / total


def parse_meminfo(text: str | None) -> tuple[float, float] | None:
    """(used, total) in GB: total minus MemAvailable, the figure free(1) uses."""
    if not text:
        return None
    fields = {}
    for line in text.splitlines():
        name, _, rest = line.partition(':')
        parts = rest.split()
        if parts and parts[0].isdigit():
            fields[name.strip()] = int(parts[0])       # kB
    if 'MemTotal' not in fields or 'MemAvailable' not in fields:
        return None
    total = fields['MemTotal'] / 1e6
    used = (fields['MemTotal'] - fields['MemAvailable']) / 1e6
    if total <= 0 or used < 0:
        return None
    return used, total


def memory(proc_root: str | None = None) -> tuple[float, float] | None:
    return parse_meminfo(_read(os.path.join(proc_root or PROC_ROOT, 'meminfo')))


def hottest_c(sys_root: str | None = None) -> float | None:
    """Highest valid thermal zone, in degrees C."""
    pattern = os.path.join(sys_root or SYS_ROOT, 'class/thermal/thermal_zone*/temp')
    best = None
    for path in glob.glob(pattern):
        raw = (_read(path) or '').strip()
        try:
            c = int(raw) / 1000.0
        except ValueError:
            continue
        if TEMP_MIN_C <= c <= TEMP_MAX_C and (best is None or c > best):
            best = c
    return best


def gpu_percent(sys_root: str | None = None) -> float | None:
    for rel in GPU_LOAD_PATHS:
        raw = _read(os.path.join(sys_root or SYS_ROOT, rel))
        if raw is None:
            continue
        try:
            value = int(raw.strip())
        except ValueError:
            return None
        if 0 <= value <= 1000:
            return value / 10.0
        return None
    return None


def disk_free_gb(path: str) -> float | None:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return None


@dataclass(frozen=True)
class Jetson:
    cpu: float | None
    gpu: float | None
    mem: tuple[float, float] | None
    temp: float | None
    disk_free: float | None
