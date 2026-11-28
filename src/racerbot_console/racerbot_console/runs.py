"""The car's recorded runs, newest first, for the Runs screen.

Reads what the recorders already write (race_diagnostics auto_record.yaml:
algorithms, duration_sec, bag_bytes, activity.moved) and never writes or
deletes anything. Folder layout: data/<car>/racerbot_runs/<ts>/,
data/<car>/racerbot_auto/<ts>/ and data/<car>/racerbot_sim/auto/<ts>/
(data/README.md).
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

KINDS = (('racerbot_runs', 'run'), ('racerbot_auto', 'map'), ('racerbot_sim/auto', 'sim'))

_STAMP = re.compile(r'(\d{8}-\d{6})')


@dataclass(frozen=True)
class Run:
    path: Path
    kind: str                   # 'run', 'map' or 'sim'
    started: datetime | None
    what: str                   # algorithm, or the folder's suffix
    moved: bool | None
    duration_sec: float | None
    size_bytes: int | None
    in_git: bool | None


def started_from_name(name: str) -> datetime | None:
    m = _STAMP.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), '%Y%m%d-%H%M%S')
    except ValueError:
        return None


def _what(name: str, record: dict) -> str:
    algorithms = record.get('algorithms')
    if isinstance(algorithms, dict):
        for nodes in algorithms.values():
            if isinstance(nodes, list) and nodes and isinstance(nodes[0], str):
                return nodes[0].lstrip('/').removesuffix('_node')
    m = _STAMP.search(name)
    if m:
        rest = (name[:m.start()] + name[m.end():]).strip('-_')
        if rest:
            return rest
    return '-'


def read_record(folder: Path) -> dict:
    """auto_record.yaml as a dict; {} if missing, unreadable or not a mapping."""
    try:
        data = yaml.safe_load((folder / 'auto_record.yaml').read_text())
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value >= 0 else None


def folder_bytes(folder: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def tracked_folders(workspace: Path, car_id: str) -> set[str] | None:
    """Run folders git already tracks, as paths relative to the workspace.
    None when git can't answer (not a repo, git missing)."""
    try:
        out = subprocess.run(
            ['git', '-C', str(workspace), 'ls-files', '--', f'data/{car_id}'],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    folders = set()
    for line in out.splitlines():
        parts = line.split('/')
        for depth in (4, 5):            # data/<id>/<kind>/<ts> and .../racerbot_sim/auto/<ts>
            if len(parts) > depth:
                folders.add('/'.join(parts[:depth]))
    return folders


def list_runs(workspace: Path, car_id: str, *, tracked: set[str] | None = None,
              limit: int = 200) -> list[Run]:
    runs = []
    for sub, kind in KINDS:
        base = workspace / 'data' / car_id / sub
        try:
            entries = [e for e in base.iterdir() if e.is_dir()]
        except OSError:
            continue
        for folder in entries:
            record = read_record(folder)
            activity = record.get('activity')
            moved = activity.get('moved') if isinstance(activity, dict) else None
            size = _number(record.get('bag_bytes'))
            rel = str(folder.relative_to(workspace))
            runs.append(Run(
                path=folder, kind=kind, started=started_from_name(folder.name),
                what=_what(folder.name, record),
                moved=moved if isinstance(moved, bool) else None,
                duration_sec=_number(record.get('duration_sec')),
                size_bytes=int(size) if size is not None else None,
                in_git=None if tracked is None else rel in tracked))
    runs.sort(key=lambda r: (r.started or datetime.min, r.path.name), reverse=True)
    return runs[:limit]


def human_bytes(n: int | None) -> str:
    if n is None:
        return '-'
    for unit, size in (('GB', 1e9), ('MB', 1e6), ('kB', 1e3)):
        if n >= size:
            return f'{n / size:.0f} {unit}' if n >= 10 * size else f'{n / size:.1f} {unit}'
    return f'{n} B'


def human_duration(sec: float | None) -> str:
    if sec is None:
        return '-'
    sec = int(round(sec))
    return f'{sec // 60}:{sec % 60:02d}'
