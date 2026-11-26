"""The dataset classifier (src/race_diagnostics/race_diagnostics/dataset_classifier/) on the recorded runs in data/.

Runner: python3 -m pytest tools/run_analysis/tests/ (the tool suites; reads bags with rosbags, unsourced, or
rosbag2_py with ROS sourced). The synthetic-run and closed-form tests are in src/race_diagnostics/test/.

Oracles:
  * recorded measurements: each bag's own bag/metadata.yaml (message counts per topic), and the `activity:` that
    run_activity.py wrote into auto_record.yaml on the car with rosbag2_py, a different reader;
  * invariants: runlog.py's LB holds and moved_between over every scan agree with the classifier's;
  * independent sources: which gap follower ran, from metadata/processes.txt and from the parameters the node
    itself reported (metadata/params/gap_follow_node.yaml), agree on every run that has both;
  * change detectors (not oracles), labelled where used: every committed qualification.yaml regrades from its own
    stored numbers to its stored grades under the current rules, the small runs re-measure to their committed
    files byte for byte, and data/dataset_inventory.csv is what the committed qualification files rebuild to.
    When one fails after a deliberate change to the classifier or its rules, regrade and commit:
        python3 tools/run_analysis/run_triage.py --qualify --all
"""
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data'
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'src' / 'race_diagnostics'))
from run_analysis import bagread, runlog  # noqa: E402
from race_diagnostics.dataset_classifier import classifier as cl, load_rules, outputs  # noqa: E402
from race_diagnostics.dataset_classifier import metrics as mx  # noqa: E402
from race_diagnostics.dataset_classifier import algorithms as al  # noqa: E402
from race_diagnostics.dataset_classifier.run_info import read_run_info  # noqa: E402
from race_diagnostics.dataset_classifier.bag_stats import read_streams  # noqa: E402

RULES = load_rules()
# The smallest recordings with something to show: a late start the car flagged (190408), two LB holds (190904),
# a car that never moved (153409), and car 2's smallest auto recording (183134).
SMALL = ['rb1/racerbot_runs/20261030-190408', 'rb1/racerbot_runs/20261030-190904',
         'rb1/racerbot_runs/20261111-153409', 'rb2/racerbot_runs/20261030-183134']
RUNS = cl.find_runs(DATA)


def _bag_info(run):
    return yaml.safe_load((DATA / run / 'bag' / 'metadata.yaml').read_text())['rosbag2_bagfile_information']


@pytest.fixture(scope='module')
def measured():
    return {run: cl.measure(DATA / run, RULES) for run in SMALL}


@pytest.mark.parametrize('run', SMALL)
def test_every_message_the_bag_says_it_holds_is_read_once(measured, run):
    info = _bag_info(run)
    m = measured[run]
    assert m['recording']['read_error'] is None
    assert m['recording']['messages_read'] == info['message_count']
    for topic in info['topics_with_message_count']:
        name = topic['topic_metadata']['name']
        assert m['streams'][name]['count'] == topic['message_count'], name
        assert m['streams'][name]['type'] == topic['topic_metadata']['type'], name


@pytest.mark.parametrize('run', [r for r in SMALL if 'activity' in (
    yaml.safe_load((DATA / r / 'auto_record.yaml').read_text()) or {})])
def test_the_activity_matches_what_run_activity_recorded_on_the_car(measured, run):
    # recorded measurement: run_activity.py's summary, written on the car after the run.
    recorded = yaml.safe_load((DATA / run / 'auto_record.yaml').read_text())['activity']
    assert measured[run]['activity'] == recorded


def test_the_late_starts_are_the_ones_the_car_flagged():
    # recorded measurement: the car's own run_activity marked this run started_late in auto_record.yaml.
    run = DATA / 'rb1/racerbot_runs/20261030-190408'
    assert yaml.safe_load((run / 'auto_record.yaml').read_text())['activity']['started_late'] is True
    doc = cl.qualification(run, RULES)
    assert doc['recording']['flags'] == ['started_late']
    assert 'conditional: recording: the car was already moving when the bag started' in \
        doc['applications']['algorithm_accuracy']['reasons']
    assert not any('already moving' in r for r in doc['applications']['localization']['reasons'])


@pytest.mark.parametrize('run', SMALL)
def test_lb_holds_are_the_ones_runlog_finds(measured, run):
    joy = [(t, msg.buttons[runlog.DEADMAN_BUTTON] if len(msg.buttons) > runlog.DEADMAN_BUTTON else 0)
           for _, t, msg in bagread.messages(DATA / run, ['/joy'])]
    joy = np.array(joy, dtype=float).reshape(-1, 2)
    holder = runlog.Run.__new__(runlog.Run)
    holder.joy = joy
    expected = holder.lb_windows()
    assert len(expected) == measured[run]['metrics']['lb_holds']
    assert mx.lb_windows(joy[:, 0], joy[:, 1] > 0.5, RULES['measurement']['lb_min_hold_s']) == expected


def test_carrying_from_snapshots_equals_carrying_from_every_scan():
    # invariant: the classifier keeps only the scans next to LB presses and releases; run_triage reads them all.
    run = DATA / 'rb1/racerbot_runs/20261030-190904'
    scan_t, ranges, odom, joy = [], [], [], []
    for topic, t, msg in bagread.messages(run, ['/scan', '/odom', '/joy']):
        if topic == '/scan':
            scan_t.append(t)
            ranges.append(np.asarray(msg.ranges, dtype=np.float32))
        elif topic == '/odom':
            odom.append((t, msg.twist.twist.linear.x))
        else:
            joy.append((t, int(len(msg.buttons) > 4 and msg.buttons[4])))
    scan_t, odom, joy = np.array(scan_t), np.array(odom), np.array(joy, dtype=float)
    windows = mx.lb_windows(joy[:, 0], joy[:, 1], 0.5)
    assert len(windows) == 2
    _, carry, _ = read_streams(run / 'bag', RULES)
    snap_t, snaps = carry.finish()
    # At most one scan per LB press or release, not the run's scans.
    assert len(snaps) <= int(np.sum(np.diff(joy[:, 1]) != 0)) + 1 < len(ranges)
    assert mx.moved_between(snap_t, snaps, odom, windows) == mx.moved_between(scan_t, ranges, odom, windows) \
        == [False]  # recorded: run_triage calls this run one continuous car

    def carried(t, r):  # as if lifted 0.5 m away after the first release
        return r + np.float32(0.5) if t > windows[0][1] else r
    full = mx.moved_between(scan_t, [carried(t, r) for t, r in zip(scan_t, ranges)], odom, windows)
    snapped = mx.moved_between(snap_t, [carried(t, r) for t, r in zip(snap_t, snaps)], odom, windows)
    assert snapped == full == [True]


def test_a_car_that_never_moved_is_unusable_for_everything():
    # recorded measurement: auto_record.yaml says this run's car never moved.
    run = DATA / 'rb1/racerbot_runs/20261111-153409'
    assert yaml.safe_load((run / 'auto_record.yaml').read_text())['activity']['moved'] is False
    doc = cl.qualification(run, RULES)
    assert set(doc['summary'].values()) == {'unusable'}
    assert doc['metrics']['distance_m'] == 0.0


def test_the_folders_without_a_bag_are_graded_unusable_not_skipped():
    empty = [run for run in RUNS if not (run / 'bag').is_dir()]
    assert [run.name for run in empty] == ['20261022-171108', '20261023-165408']
    for run in empty:
        doc = cl.qualification(run, RULES)
        assert doc['recording']['flags'] == ['no_bag'] and set(doc['summary'].values()) == {'unusable'}


# --------------------------------------------------------------------------- committed results

def _committed(run):
    return outputs.read_qualification(run)


def test_every_run_folder_has_a_qualification():
    missing = [str(run.relative_to(DATA)) for run in RUNS if _committed(run) is None]
    assert missing == [], 'run: python3 tools/run_analysis/run_triage.py --qualify --all'


@pytest.mark.parametrize('run', RUNS, ids=[f'{r.parent.parent.name}/{r.name}' for r in RUNS])
def test_committed_grades_follow_from_their_stored_numbers_under_the_current_rules(run):
    # change-detector (not an oracle): grading is deterministic, so the stored numbers and the current rules must
    # give the stored grades. Fails when the rules or the grading changed and the runs were not regraded.
    doc = _committed(run)
    assert doc['classifier']['rules_sha256'] == RULES.sha256, 'rules changed: regrade with --qualify --all'
    assert doc['classifier']['version'] == cl.VERSION
    if doc['error'] is None:
        again = cl.regrade(doc, RULES)
        assert again['applications'] == doc['applications']
        assert again['recording']['flags'] == doc['recording']['flags']
    assert doc['summary'] == {app: a['status'] for app, a in doc['applications'].items()}


@pytest.mark.parametrize('run', SMALL)
def test_the_small_runs_remeasure_to_their_committed_files(run):
    # change-detector (not an oracle): the same bag and rules give the same file, byte for byte.
    path = DATA / run / 'qualification.yaml'
    assert outputs.dump_qualification(cl.qualification(DATA / run, RULES)) == path.read_text()


def test_the_inventory_is_one_row_per_run_folder_rebuilt_from_the_qualifications():
    # change-detector (not an oracle) for the content; invariant for one row per folder.
    path = DATA / outputs.INVENTORY
    rows = [line.split(',', 3)[:2] for line in path.read_text().splitlines()[1:]]
    assert sorted(map(tuple, rows)) == sorted((r.parent.parent.name, r.name) for r in RUNS)
    assert len(rows) == len(set(map(tuple, rows)))
    expected = outputs._render([outputs.inventory_row(DATA, run, RULES) for run in RUNS], RULES)
    assert path.read_text() == expected


def test_a_dry_run_over_one_run_prints_the_table_and_writes_nothing(tmp_path):
    run = DATA / SMALL[0]
    before = sorted((p.name, p.stat().st_mtime_ns) for p in run.iterdir())
    inventory = (DATA / outputs.INVENTORY).stat().st_mtime_ns
    out = subprocess.run([sys.executable, str(ROOT / 'tools/run_analysis/run_triage.py'), '--qualify', '--dry-run',
                          str(run)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert f'rb1/{run.name}: localization=' in out.stdout
    assert 'application' in out.stdout and 'lateral_acceleration' in out.stdout
    assert sorted((p.name, p.stat().st_mtime_ns) for p in run.iterdir()) == before
    assert (DATA / outputs.INVENTORY).stat().st_mtime_ns == inventory


# --------------------------------------------------------------------------- which code drove

# racerbot_a's reactive gap follower, which starts a node named like this workspace's: gap_follow_node.
TEAM_A_GAP_FOLLOW_RUNS = ['rb2/racerbot_runs/20261030-182852', 'rb2/racerbot_runs/20261109-183455',
                          'rb2/racerbot_runs/20261109-183910']


@pytest.mark.parametrize('run', TEAM_A_GAP_FOLLOW_RUNS)
def test_team_a_gap_follower_runs_are_named_as_racerbot_a_code(run):
    # Independent of the processes.txt match: the node reported reactive's own parameter, not this workspace's.
    reported = al.reported_parameters(DATA / run / 'metadata' / 'params' / 'gap_follow_node.yaml')
    assert 'use_fallback_method' in reported and 'enable_deadman' not in reported
    info = read_run_info(DATA / run, known=RULES['algorithms'])
    assert info['algorithm_nodes'] == ['gap_follow_node']
    assert [(c['code'], c['repo'], c['from']) for c in info['algorithm_code']] == [
        ('reactive/gap_follow_node', 'racerbot_a', 'metadata/processes.txt')]


def test_processes_and_the_parameter_snapshot_agree_on_every_recorded_gap_follower(tmp_path):
    known = RULES['algorithms']
    seen = {}
    for run_dir in RUNS:
        processes = run_dir / 'metadata' / 'processes.txt'
        params = run_dir / 'metadata' / 'params' / 'gap_follow_node.yaml'
        if not (processes.is_file() and params.is_file()):
            continue
        by_process = [code for node, code in al.from_processes(processes.read_text(errors='replace'), known)
                      if node == 'gap_follow_node']
        # The same run with its processes.txt left out: only the parameter snapshot can say.
        only_params = tmp_path / run_dir.relative_to(DATA).as_posix().replace('/', '_')
        (only_params / 'metadata' / 'params').mkdir(parents=True)
        shutil.copy(params, only_params / 'metadata' / 'params')
        by_params = [c['code'] for c in al.identify(['gap_follow_node'], known, only_params)]
        assert by_params == by_process, run_dir
        seen[str(run_dir.relative_to(DATA))] = by_process[0]
    # Not vacuous: runs of both packages were compared.
    assert sorted(run for run, code in seen.items() if code == 'reactive/gap_follow_node') == TEAM_A_GAP_FOLLOW_RUNS
    assert 'gap_follow/gap_follow_node' in seen.values()


def test_every_recorded_algorithm_is_on_the_list():
    """A run showing unknown/<node> needs that code added to classification_rules.yaml's algorithms:."""
    unknown, found = {}, set()
    for run_dir in RUNS:
        info = read_run_info(run_dir, known=RULES['algorithms'])
        if not isinstance(info['algorithm_code'], list):
            continue
        for item in info['algorithm_code']:
            found.add((item['code'], item['from'].split(' ')[0]))
            if item['code'].startswith('unknown/'):
                unknown.setdefault(str(run_dir.relative_to(DATA)), []).append(item['from'])
    assert unknown == {}
    # Not vacuous: each way of matching was used, and the map_course runs (no processes.txt) were among them.
    assert {('reactive/gap_follow_node', 'metadata/processes.txt'), ('ftg/ftg_node', 'metadata/processes.txt'),
            ('gap_follow/gap_follow_node', 'metadata/params/gap_follow_node.yaml'),
            ('pure_pursuit/auto_map_race_node', 'node')} <= found
