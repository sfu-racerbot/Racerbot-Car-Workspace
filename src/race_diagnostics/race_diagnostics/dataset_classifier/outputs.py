"""
outputs.py -- write qualification.yaml and data/dataset_inventory.csv safely.

Both are written to a temporary file beside the target and moved into place
in one step, so a reader never sees half a file and a crash leaves the old
one. A file whose content would not change is not touched at all, so
regrading is safe to repeat: no new timestamps, no duplicate rows, nothing
for git to show. The inventory has exactly one row per run folder.

Two cars share one inventory in git. A car refreshes only its own rows and
keeps the other car's as they are, so their commits touch different lines;
`run_triage.py --inventory` rebuilds every row from the qualification files.
"""

import csv
import fcntl
import io
import os
from contextlib import contextmanager
from pathlib import Path

import yaml

from .classifier import QUALIFICATION, find_runs, run_location

INVENTORY = 'dataset_inventory.csv'
UNCLASSIFIED = 'unclassified'


class _Dumper(yaml.SafeDumper):
    """Block style for the document, one line per stream's numbers."""


class _Flow(dict):
    pass


_Dumper.add_representer(_Flow, lambda d, data: d.represent_mapping(
    'tag:yaml.org,2002:map', data.items(), flow_style=True))


def _write_if_changed(path, text):
    """Replace path with text in one step, unless it already holds exactly that."""
    path = Path(path)
    try:
        if path.read_text() == text:
            return False
    except (OSError, UnicodeDecodeError):
        pass
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def dump_qualification(doc):
    doc = dict(doc)
    if doc.get('streams'):
        doc['streams'] = {k: _Flow(v) for k, v in doc['streams'].items()}
    header = ('# Written by race_diagnostics.dataset_classifier from this run\'s bag and\n'
              '# metadata; regenerated, never edited. How to read it:\n'
              '# docs/run-diagnostics.md#dataset-qualification\n')
    return header + yaml.dump(doc, Dumper=_Dumper, sort_keys=False, width=100,
                              allow_unicode=True)


def write_qualification(run_dir, doc):
    """Write <run_dir>/qualification.yaml. True if the file changed."""
    return _write_if_changed(Path(run_dir) / QUALIFICATION, dump_qualification(doc))


def read_qualification(run_dir):
    path = Path(run_dir) / QUALIFICATION
    try:
        text = path.read_text()
    except OSError:
        return None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return None


# --------------------------------------------------------------------------- inventory

def columns(rules):
    apps = list(rules['applications'])
    return (['car', 'run_id', 'path', 'recorder', 'started', 'bag_start_utc', 'duration_s',
             'algorithms', 'distance_m', 'moving_s', 'max_speed_mps', 'lb_held_s', 'recording']
            + apps + [f'{app}_why' for app in apps]
            + ['recording_flags', 'classifier_version', 'rules_sha256'])


def _cell(value):
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.1f}' if abs(value) >= 1 else f'{value:.2f}'
    if isinstance(value, list):
        return ';'.join(str(v) for v in value)
    return str(value)


def inventory_row(data_root, run_dir, rules):
    """One CSV row for a run folder, from its qualification.yaml (or unclassified)."""
    run_dir = Path(run_dir)
    rel = run_dir.resolve().relative_to(Path(data_root).resolve()).as_posix()
    row = dict.fromkeys(columns(rules), '')
    row.update(car=rel.split('/')[0], run_id=run_dir.name, path=rel)
    doc = read_qualification(run_dir)
    apps = list(rules['applications'])
    if not doc:
        row['recording'] = UNCLASSIFIED
        for app in apps:
            row[app] = UNCLASSIFIED
        return row
    info = doc.get('run_info') or {}
    metrics = doc.get('metrics') or {}
    recording = doc.get('recording') or {}
    row.update(
        recorder=_cell(info.get('recorder')), started=_cell(info.get('started')),
        bag_start_utc=_cell(info.get('bag_start_utc')),
        duration_s=_cell(metrics.get('duration_s')),
        algorithms=_cell(info.get('algorithm_nodes')),
        distance_m=_cell(metrics.get('distance_m')), moving_s=_cell(metrics.get('moving_s')),
        max_speed_mps=_cell(metrics.get('max_speed_mps')),
        lb_held_s=_cell(metrics.get('lb_held_s')),
        recording=_cell(recording.get('status')),
        recording_flags=_cell(recording.get('flags')),
        classifier_version=_cell((doc.get('classifier') or {}).get('version')),
        rules_sha256=str((doc.get('classifier') or {}).get('rules_sha256', ''))[:12])
    for app in apps:
        graded = (doc.get('applications') or {}).get(app) or {}
        row[app] = graded.get('status', UNCLASSIFIED)
        row[f'{app}_why'] = graded.get('explanation', '')
    return row


@contextmanager
def _locked(data_root):
    """Hold an exclusive lock on the data folder while the inventory is rewritten,
    so two runs finishing together cannot drop each other's rows."""
    fd = os.open(data_root, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _render(rows, rules):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns(rules), lineterminator='\n')
    writer.writeheader()
    for row in sorted(rows, key=lambda r: (r['car'], r['run_id'])):
        writer.writerow(row)
    return out.getvalue()


def _existing_rows(path, rules):
    """The rows already in the inventory, or None if it is missing or has other columns."""
    try:
        with open(path, newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != columns(rules):
                return None
            return list(reader)
    except OSError:
        return None


def update_inventory(data_root, rules, cars=None):
    """Refresh the rows of `cars` (all cars if None) from their run folders,
    keeping the other cars' rows. True if the file changed."""
    data_root = Path(data_root)
    path = data_root / INVENTORY
    with _locked(data_root):
        kept = []
        if cars is not None:
            existing = _existing_rows(path, rules)
            if existing is None:
                cars = None  # no usable file to keep rows from: rebuild it all
            else:
                kept = [row for row in existing if row['car'] not in cars]
        runs = [r for r in find_runs(data_root) if cars is None or r.parent.parent.name in cars]
        rows = kept + [inventory_row(data_root, run, rules) for run in runs]
        return _write_if_changed(path, _render(rows, rules))


def rebuild_inventory(data_root, rules):
    """Rebuild every row of data/dataset_inventory.csv from the run folders."""
    return update_inventory(data_root, rules, cars=None)


__all__ = ['INVENTORY', 'dump_qualification', 'write_qualification', 'read_qualification',
           'update_inventory', 'rebuild_inventory', 'inventory_row', 'run_location']
