"""runs.py against a fake data/ tree. Field names and layout come from
race_diagnostics' auto_record.yaml (e.g. data/rb2/racerbot_runs/20261111-154204/)
and data/README.md."""
import subprocess
from datetime import datetime

import pytest
import yaml

from racerbot_console import runs


def make(ws, rel, record=None, files=None):
    d = ws / rel
    d.mkdir(parents=True)
    if record is not None:
        (d / 'auto_record.yaml').write_text(record if isinstance(record, str)
                                            else yaml.safe_dump(record))
    for name, size in (files or {}).items():
        (d / name).write_bytes(b'x' * size)
    return d


def test_runs_newest_first_across_kinds(tmp_path):
    make(tmp_path, 'data/rb2/racerbot_runs/20261111-154204',
         {'algorithms': {'/drive': ['/disparity_extender_node']}, 'duration_sec': 1256.7,
          'bag_bytes': 344914634, 'activity': {'moved': True}})
    make(tmp_path, 'data/rb2/racerbot_auto/20261116-105215-line')
    make(tmp_path, 'data/rb2/racerbot_sim/auto/20261001-000000')
    make(tmp_path, 'data/rb1/racerbot_runs/20261201-000000')          # other car: not listed
    got = runs.list_runs(tmp_path, 'rb2')
    assert [r.path.name for r in got] == ['20261116-105215-line', '20261111-154204',
                                          '20261001-000000']
    assert [r.kind for r in got] == ['map', 'run', 'sim']
    run = got[1]
    assert run.what == 'disparity_extender' and run.moved is True
    assert run.duration_sec == 1256.7 and run.size_bytes == 344914634
    assert run.started == datetime(2026, 11, 11, 15, 42, 4)
    assert got[0].what == 'line'


@pytest.mark.parametrize('record', ['{not yaml: [', '- a list\n', 'plain', ''])
def test_bad_auto_record_yaml_still_lists_the_run(tmp_path, record):
    make(tmp_path, 'data/rb2/racerbot_runs/20261111-154204', record)
    (r,) = runs.list_runs(tmp_path, 'rb2')
    assert r.moved is None and r.duration_sec is None and r.what == '-'


def test_wrong_types_in_auto_record_are_ignored(tmp_path):
    make(tmp_path, 'data/rb2/racerbot_runs/20261111-154204',
         {'duration_sec': 'long', 'bag_bytes': -5, 'activity': {'moved': 'yes'},
          'algorithms': {'/drive': [3]}})
    (r,) = runs.list_runs(tmp_path, 'rb2')
    assert (r.duration_sec, r.size_bytes, r.moved, r.what) == (None, None, None, '-')


def test_true_is_not_a_number_of_bytes_or_seconds(tmp_path):
    # YAML `yes`/`true` parses to a bool, and bool is an int in Python
    make(tmp_path, 'data/rb2/racerbot_runs/20261111-154204', 'duration_sec: true\nbag_bytes: true\n')
    (r,) = runs.list_runs(tmp_path, 'rb2')
    assert r.duration_sec is None and r.size_bytes is None


def test_folder_without_a_timestamp_sorts_last(tmp_path):
    make(tmp_path, 'data/rb2/racerbot_runs/scratch')
    make(tmp_path, 'data/rb2/racerbot_runs/20200101-000000')
    assert [r.path.name for r in runs.list_runs(tmp_path, 'rb2')] == ['20200101-000000', 'scratch']


def test_no_data_folder_is_an_empty_list(tmp_path):
    assert runs.list_runs(tmp_path, 'rb2') == []


def test_in_git_from_a_real_repo(tmp_path):
    a = make(tmp_path, 'data/rb2/racerbot_runs/20261111-154204', files={'bag.mcap': 10})
    make(tmp_path, 'data/rb2/racerbot_runs/20261111-160000', files={'bag.mcap': 10})
    s = make(tmp_path, 'data/rb2/racerbot_sim/auto/20261001-000000', files={'x': 1})
    git = ['git', '-C', str(tmp_path)]
    subprocess.run(git + ['init', '-q'], check=True)
    subprocess.run(git + ['add', str(a), str(s)], check=True)
    tracked = runs.tracked_folders(tmp_path, 'rb2')
    by = {r.path.name: r.in_git for r in runs.list_runs(tmp_path, 'rb2', tracked=tracked)}
    assert by == {'20261111-154204': True, '20261111-160000': False, '20261001-000000': True}


def test_in_git_unknown_outside_a_repo(tmp_path):
    assert runs.tracked_folders(tmp_path / 'nope', 'rb2') is None


@pytest.mark.parametrize('n,text', [(None, '-'), (0, '0 B'), (999, '999 B'), (1000, '1.0 kB'),
                                    (344914634, '345 MB'), (1_500_000_000, '1.5 GB')])
def test_human_bytes(n, text):
    assert runs.human_bytes(n) == text


@pytest.mark.parametrize('sec,text', [(None, '-'), (0, '0:00'), (59.4, '0:59'), (59.6, '1:00'),
                                      (1256.7, '20:57')])
def test_human_duration(sec, text):
    assert runs.human_duration(sec) == text
