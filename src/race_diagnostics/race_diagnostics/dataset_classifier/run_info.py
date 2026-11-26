"""
run_info.py -- what a run is, written once when its recording starts.

<run>/run_info.yaml is written by the recorder (auto_record_node.py, or
record_run.py for a recording started by hand) and never changed afterwards.
A run recorded before run_info.yaml existed has none: read_run_info() then
reconstructs the same fields from the files the run does have, and writes
`unknown` for anything those files do not say. It never guesses, and never
writes a run_info.yaml for an old run.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import socket

import yaml

from .metrics import algorithms as drivers_in

RUN_INFO = 'run_info.yaml'
UNKNOWN = 'unknown'
SCHEMA = 1


def _atomic_write(path, text):
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    temporary.write_text(text)
    os.replace(temporary, path)


def write_run_info(run_dir, *, recorder, started=None, algorithms=None, car_id=None,
                   topics=None, workspace_commit=None, host=None):
    """Write <run_dir>/run_info.yaml. Fields not given are `unknown`.

    algorithms: {topic: [node, ...]} as auto_record found them, or a list."""
    run_dir = Path(run_dir)
    info = {
        'schema': SCHEMA,
        'run_id': run_dir.name,
        'recorder': recorder,
        'started': started or datetime.now().astimezone().isoformat(timespec='seconds'),
        'car_id': car_id or UNKNOWN,
        'host': host or socket.gethostname(),
        'workspace_commit': workspace_commit or workspace_commit_of(run_dir),
        'algorithms': algorithms if algorithms else UNKNOWN,
        'topics': list(topics) if topics else UNKNOWN,
    }
    path = run_dir / RUN_INFO
    _atomic_write(path, yaml.safe_dump(info, sort_keys=False))
    return path


def workspace_commit_of(run_dir):
    """HEAD of racerbot-ws when the run was recorded, from metadata/git/, else unknown."""
    path = Path(run_dir) / 'metadata' / 'git' / 'racerbot-ws.txt'
    try:
        match = re.search(r'^HEAD: ([0-9a-f]{7,40})$', path.read_text(errors='replace'), re.M)
    except OSError:
        return UNKNOWN
    return match.group(1) if match else UNKNOWN


def _load_yaml(path):
    try:
        return yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return None


def _car_id(run_dir):
    for path, read in ((run_dir / 'metadata' / 'car_id.txt', lambda p: p.read_text().strip()),
                       (run_dir / 'metadata' / 'car.yaml',
                        lambda p: str((_load_yaml(p) or {}).get('car_id') or ''))):
        try:
            value = read(path) if path.is_file() else ''
        except OSError:
            value = ''
        if value:
            return value, f'metadata/{path.name}'
    return UNKNOWN, None


def _session_log(run_dir):
    """A map_course log in the runs folder that began within 5 s before this run."""
    try:
        run_time = datetime.strptime(run_dir.name, '%Y%m%d-%H%M%S')
    except ValueError:
        return None
    for log in sorted(run_dir.parent.glob('map_course-*.log')):
        try:
            log_time = datetime.strptime(log.stem[len('map_course-'):], '%Y%m%d-%H%M%S')
        except ValueError:
            continue
        if 0 <= (run_time - log_time).total_seconds() <= 5:
            return log.name
    return None


def bag_start_utc(bag_info):
    """The bag's first message time, ISO 8601 in UTC, from bag/metadata.yaml."""
    try:
        ns = bag_info['starting_time']['nanoseconds_since_epoch']
    except (KeyError, TypeError):
        return UNKNOWN
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat(timespec='seconds')


def read_run_info(run_dir, bag_info=None):
    """run_info.yaml if the recorder wrote one, else the same fields rebuilt
    from what the run holds. `source` says which; unknowns stay `unknown`."""
    run_dir = Path(run_dir)
    written = _load_yaml(run_dir / RUN_INFO)
    if written:
        info = dict(written)
        info['source'] = RUN_INFO
    else:
        auto = _load_yaml(run_dir / 'auto_record.yaml') or {}
        car_id, car_source = _car_id(run_dir)
        if auto:
            recorder = 'auto_record'
        elif (run_dir / 'events.jsonl').is_file():
            # race_diag_node's probe file: record_run.py, by hand or from map_course.
            recorder = 'record_run'
        else:
            recorder = UNKNOWN
        info = {
            'schema': SCHEMA, 'run_id': run_dir.name, 'recorder': recorder,
            'started': str(auto.get('started') or UNKNOWN),
            'car_id': car_id, 'host': UNKNOWN,
            'workspace_commit': workspace_commit_of(run_dir),
            'algorithms': auto.get('algorithms') or UNKNOWN, 'topics': UNKNOWN,
            'source': 'reconstructed from the run folder (recorded before run_info.yaml existed)',
        }
        if car_source:
            info['car_id_source'] = car_source
        log = _session_log(run_dir)
        if log:
            info['session_log'] = log
    info['bag_start_utc'] = bag_start_utc(bag_info)
    names, source = algorithm_names(run_dir, info.get('algorithms') if written else None)
    info['algorithm_nodes'] = names if source else UNKNOWN
    info['algorithm_source'] = source or UNKNOWN
    return info


def algorithm_names(run_dir, recorded=None):
    """([node names], where they came from) for the driving code of a run.

    run_info's or auto_record.yaml's {topic: [nodes]} first (the nodes auto_record
    saw publishing a drive topic), else the driving code in
    metadata/processes.txt, as run_triage.py reads it, else the driving nodes
    whose parameters snapshot_params saved in metadata/params/ (a hand-started
    or map_course recording has no processes.txt). (None, None) if none."""
    def names_of(value):
        if isinstance(value, dict):
            return sorted({str(node).lstrip('/') for nodes in value.values() for node in nodes or ()})
        if isinstance(value, list):
            return sorted({str(node).lstrip('/') for node in value})
        return None

    names = names_of(recorded)
    if names:
        return names, RUN_INFO
    auto = _load_yaml(Path(run_dir) / 'auto_record.yaml') or {}
    names = names_of(auto.get('algorithms'))
    if names:
        return names, 'auto_record.yaml'
    processes = Path(run_dir) / 'metadata' / 'processes.txt'
    if processes.is_file():
        return drivers_in(processes.read_text(errors='replace')), 'metadata/processes.txt'
    params = Path(run_dir) / 'metadata' / 'params'
    if params.is_dir():
        return drivers_in(' '.join(sorted(p.stem for p in params.glob('*.yaml')))), 'metadata/params'
    return None, None
