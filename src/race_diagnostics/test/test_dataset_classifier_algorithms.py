"""
race_diagnostics.dataset_classifier.algorithms: whose code drove a run.

This workspace's gap_follow and racerbot_a's reactive package both start a
node called gap_follow_node, so the classifier names code by
<package>/<executable> and repo, never by node name alone. No ROS needed:

    python3 -m pytest src/race_diagnostics/test/test_dataset_classifier_algorithms.py -v

(also collected by `colcon test --packages-select race_diagnostics`).

Oracles:
  * cited spec: the matching order and the `unknown/<node>` rule in
    config/classification_rules.yaml's `algorithms:` comment and
    docs/run-diagnostics.md#which-code-drove;
  * recorded measurements: process lines copied from metadata/processes.txt of
    rb2 20261030-182852 (team A's reactive gap follower), rb1 20261030-175055
    (ours) and rb2 20261030-184316 (team A's ftg); the parameter snapshot
    format from rb2 20261030-182852's metadata/params/gap_follow_node.yaml;
  * source: node and executable names from racerbot_a main b9bfd35's launch
    files, and enable_deadman from src/gap_follow/gap_follow/gap_follow_node.py.
Every recorded run checked the same way: tools/run_analysis/tests/test_dataset_classifier_bag.py.
"""
import os
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from race_diagnostics.dataset_classifier import algorithms as al  # noqa: E402
from race_diagnostics.dataset_classifier import load_rules  # noqa: E402
from race_diagnostics.dataset_classifier.rules import RulesError, check_rules  # noqa: E402

RULES = load_rules()
KNOWN = RULES['algorithms']
HEADER = '    PID    PPID                  STARTED COMMAND\n'
TEAM_A_GAP_FOLLOW = ('  83301   83289 Fri Oct 30 18:28:47 2026 /home/racerbotcar-2/racerbot-ws/install/reactive/lib/'
                     'reactive/gap_follow_node --ros-args -r __node:=gap_follow_node --params-file '
                     '/tmp/launch_params_l3ijoko0 -r /drive:=/drive -r /scan:=/scan\n')
OUR_GAP_FOLLOW = ('  31967   31923 Fri Oct 30 17:54:13 2026 /usr/bin/python3 /home/racerbotcar-1/racerbot-ws/install/'
                  'gap_follow/lib/gap_follow/gap_follow_node --ros-args -r __node:=gap_follow_node --params-file '
                  '/home/racerbotcar-1/racerbot-ws/install/gap_follow/share/gap_follow/config/gap_follow.yaml\n')
TEAM_A_FTG = (' 111377  111356 Fri Oct 30 18:43:12 2026 /home/racerbotcar-2/racerbot-ws/install/ftg/lib/ftg/ftg_node '
              '--ros-args -r __node:=ftg_node --params-file /tmp/launch_params_d_r1qeq3\n')
BOTH_GAP_FOLLOWERS = ['gap_follow/gap_follow_node', 'reactive/gap_follow_node']


def params_text(*names):
    """A node's parameters as snapshot_params saves them (format of rb2 20261030-182852)."""
    body = ''.join(f'    {name}: true\n' for name in names) or '    {}\n'
    return f'/gap_follow_node:\n  ros__parameters:\n{body}'


def run(tmp_path, processes=None, params=None):
    run_dir = tmp_path / '20261201-120000'
    (run_dir / 'metadata').mkdir(parents=True)
    if processes is not None:
        (run_dir / 'metadata' / 'processes.txt').write_text(HEADER + processes)
    if params is not None:
        (run_dir / 'metadata' / 'params').mkdir()
        path = run_dir / 'metadata' / 'params' / 'gap_follow_node.yaml'
        if isinstance(params, bytes):
            path.write_bytes(params)
        else:
            path.write_text(params)
    return run_dir


def codes(identified):
    return [(item['node'], item['code'], item['repo'], item['role'], item['from']) for item in identified]


# --------------------------------------------------------------------------- processes.txt

def test_the_same_node_name_from_two_packages_is_named_by_its_install_path(tmp_path):
    team_a = al.identify(['/gap_follow_node'], KNOWN, run(tmp_path / 'a', TEAM_A_GAP_FOLLOW))
    ours = al.identify(['/gap_follow_node'], KNOWN, run(tmp_path / 'b', OUR_GAP_FOLLOW))
    assert codes(team_a) == [('/gap_follow_node', 'reactive/gap_follow_node', 'racerbot_a', 'driving',
                              'metadata/processes.txt')]
    assert codes(ours) == [('/gap_follow_node', 'gap_follow/gap_follow_node', 'racerbot-ws', 'driving',
                            'metadata/processes.txt')]


def test_processes_txt_is_believed_before_the_parameter_snapshot(tmp_path):
    # The documented order: the process that ran is exact; a parameter is only evidence.
    run_dir = run(tmp_path, TEAM_A_GAP_FOLLOW, params_text('enable_deadman'))
    assert [item['code'] for item in al.identify(['gap_follow_node'], KNOWN, run_dir)] == \
        ['reactive/gap_follow_node']


def test_a_process_runs_under_its_node_remap_or_else_its_listed_name():
    remapped = TEAM_A_FTG.replace('__node:=ftg_node', '__node:=opponent_ftg')
    assert al.from_processes(remapped, KNOWN) == [('opponent_ftg', 'ftg/ftg_node')]
    bare = '  9 1 Mon Dec  1 12:00:00 2026 /home/x/install/ftg/lib/ftg/ftg_node --ros-args --params-file /tmp/p\n'
    assert al.from_processes(bare, KNOWN) == [('ftg_node', 'ftg/ftg_node')]


def test_only_listed_executables_count_and_each_once():
    text = (HEADER + TEAM_A_FTG + TEAM_A_FTG
            + '   3136    2354 Fri Oct 30 17:28:14 2026 /usr/bin/python3 /home/x/install/web_dashboard/lib/'
              'web_dashboard/dashboard_node\n'
            # Not an executable at .../lib/<package>/<executable>: a module file, a site-packages path, a
            # longer name that only starts like a listed one.
            + '  10 1 Mon Dec  1 12:00:00 2026 /usr/bin/python3 /home/x/install/gap_follow/lib/gap_follow/'
              'gap_follow_node.py\n'
            + '  11 1 Mon Dec  1 12:00:00 2026 /usr/lib/python3.12/site-packages/gap_follow/gap_follow_node\n'
            + '  12 1 Mon Dec  1 12:00:00 2026 /home/x/install/ftg/lib/ftg/ftg_node_old --ros-args\n')
    assert al.from_processes(text, KNOWN) == [('ftg_node', 'ftg/ftg_node')]
    assert al.from_processes('', KNOWN) == []
    assert al.from_processes(None, KNOWN) == []
    assert al.from_processes(HEADER, KNOWN) == []


def test_two_listed_processes_under_one_node_name_are_both_named(tmp_path):
    run_dir = run(tmp_path, TEAM_A_GAP_FOLLOW + OUR_GAP_FOLLOW)
    assert [item['code'] for item in al.identify(['gap_follow_node'], KNOWN, run_dir)] == BOTH_GAP_FOLLOWERS


# --------------------------------------------------------------------------- without processes.txt

@pytest.mark.parametrize('node, code, repo', [
    ('ftg_node', 'ftg/ftg_node', 'racerbot_a'),
    ('disparity_extender_node', 'disparity_extender/disparity_extender_node', 'racerbot_a'),
    # Neither was recognised before 2026-11-26: pure_pursuit_node did not match pure_pursuit_a_node, and the
    # executable obs_detection runs as obs_detection_node (racerbot_a obs_detection_launch.py).
    ('pure_pursuit_a_node', 'pure_pursuit_a/pure_pursuit_a_node', 'racerbot_a'),
    ('obs_detection_node', 'obs_detection/obs_detection', 'racerbot_a'),
    ('pure_pursuit_node', 'pure_pursuit/pure_pursuit_node', 'racerbot-ws'),
    ('pure_pursuit_tech_node', 'pure_pursuit_tech/pure_pursuit_tech_node', 'racerbot-ws'),
])
def test_a_node_only_one_entry_starts_is_named_by_its_node_name(node, code, repo):
    assert codes(al.identify([f'/{node}'], KNOWN)) == [(f'/{node}', code, repo, 'driving', 'node name')]


def test_a_namespaced_node_is_matched_by_its_own_name():
    assert [item['code'] for item in al.identify(['/opp/ftg_node'], KNOWN)] == ['ftg/ftg_node']


@pytest.mark.parametrize('parameter, code', [
    ('use_fallback_method', 'reactive/gap_follow_node'),
    ('enable_deadman', 'gap_follow/gap_follow_node'),
])
def test_a_shared_node_name_is_told_apart_by_the_parameter_it_reported(tmp_path, parameter, code):
    run_dir = run(tmp_path, params=params_text('max_speed', parameter))
    assert codes(al.identify(['gap_follow_node'], KNOWN, run_dir)) == [
        ('gap_follow_node', code, KNOWN[code]['repo'], 'driving',
         f'metadata/params/gap_follow_node.yaml ({parameter})')]


@pytest.mark.parametrize('params', [
    None,                                                   # no snapshot
    '',                                                     # empty file
    params_text(),                                          # no parameters
    params_text('max_speed'),                               # neither parameter
    params_text('enable_deadman', 'use_fallback_method'),   # both: cannot be told apart
    '/gap_follow_node:\n  ros__parameters: [enable_deadman]\n',  # wrong type
    '- enable_deadman\n',                                   # not a mapping at all
    '/gap_follow_node:\n  ros__parameters:\n    enable_deadman: [\n',  # truncated
    b'/gap_follow_node:\n  ros__parameters:\n    enable_deadman\xff: true\n',  # malformed UTF-8
], ids=['missing', 'empty', 'no_parameters', 'neither', 'both', 'wrong_type', 'a_list', 'truncated', 'bad_utf8'])
def test_a_shared_node_name_with_nothing_to_tell_them_apart_is_unknown_not_guessed(tmp_path, params):
    run_dir = run(tmp_path, params=params)
    assert codes(al.identify(['gap_follow_node'], KNOWN, run_dir)) == [
        ('gap_follow_node', 'unknown/gap_follow_node', 'unknown', 'driving',
         'unknown: gap_follow/gap_follow_node or reactive/gap_follow_node, and nothing in the run tells which')]


def test_a_shared_node_name_without_a_run_folder_is_unknown():
    assert [item['code'] for item in al.identify(['gap_follow_node'], KNOWN)] == ['unknown/gap_follow_node']


def test_a_node_on_no_list_is_unknown_and_still_counts_as_driving():
    # A run only names a node auto_record saw publishing a drive topic.
    identified = al.identify(['/my_new_planner'], KNOWN)
    assert codes(identified) == [('/my_new_planner', 'unknown/my_new_planner', 'unknown', 'driving',
                                  "unknown: no entry in the rules' algorithms list starts my_new_planner")]
    assert al.drove(identified) is True


# --------------------------------------------------------------------------- what counts as driving

def test_slam_and_the_gamepad_never_count_as_driving():
    alone = al.identify(['slam_toolbox', 'joy_teleop'], KNOWN)
    assert [(item['code'], item['role']) for item in alone] == [
        ('joy_teleop/joy_teleop', 'manual'), ('slam_toolbox/async_slam_toolbox_node', 'mapping')]
    assert al.drove(alone) is False
    assert al.driving(alone) == ([], [])
    assert al.drove([]) is False
    with_ftg = al.identify(['slam_toolbox', 'joy_teleop', 'ftg_node'], KNOWN)
    assert al.drove(with_ftg) is True
    assert al.driving(with_ftg) == (['ftg/ftg_node'], ['racerbot_a'])


def test_driving_lists_each_code_and_repo_once_sorted(tmp_path):
    run_dir = run(tmp_path, TEAM_A_GAP_FOLLOW + OUR_GAP_FOLLOW + TEAM_A_FTG)
    identified = al.identify(['gap_follow_node', 'ftg_node', 'slam_toolbox'], KNOWN, run_dir)
    assert al.driving(identified) == (['ftg/ftg_node'] + BOTH_GAP_FOLLOWERS, ['racerbot-ws', 'racerbot_a'])


# --------------------------------------------------------------------------- the parameter snapshot

def test_reported_parameters_reads_every_node_block_and_refuses_what_is_not_a_snapshot(tmp_path):
    path = tmp_path / 'p.yaml'
    path.write_text('/a:\n  ros__parameters:\n    x: 1\n    y: {z: 2}\n/b:\n  ros__parameters:\n    w: 3\n')
    assert al.reported_parameters(path) == {'x', 'y', 'w'}
    path.write_text('/a:\n  something_else: {x: 1}\n')
    assert al.reported_parameters(path) == set()
    path.write_text('3\n')
    assert al.reported_parameters(path) is None
    assert al.reported_parameters(tmp_path / 'missing.yaml') is None


def test_the_parameter_that_tells_ours_apart_is_one_our_gap_follow_declares():
    # If gap_follow ever renames it, map_course runs (no processes.txt) would all become unknown.
    source = Path(__file__).resolve().parents[2] / 'gap_follow' / 'gap_follow' / 'gap_follow_node.py'
    parameter = KNOWN['gap_follow/gap_follow_node']['parameter']
    assert f"self.declare_parameter('{parameter}'" in source.read_text()


# --------------------------------------------------------------------------- the rules' list

def _rules_with(change):
    data = yaml.safe_load(open(RULES.path))
    change(data['algorithms'])
    return data


@pytest.mark.parametrize('change, error', [
    (lambda a: a['reactive/gap_follow_node'].pop('parameter'),
     'gap_follow/gap_follow_node, reactive/gap_follow_node all start gap_follow_node: each needs a parameter'),
    (lambda a: a['reactive/gap_follow_node'].update(parameter='enable_deadman'),
     'all start gap_follow_node: each needs a parameter of its own'),
    (lambda a: a.update({'gap_follow_node': {'repo': 'x', 'node': 'y'}}), "'gap_follow_node' is not <package>"),
    (lambda a: a.update({'a/b/c': {'repo': 'x', 'node': 'y'}}), "'a/b/c' is not <package>/<executable>"),
    (lambda a: a['ftg/ftg_node'].update(team='a'), r'algorithms\.ftg/ftg_node: unknown key\(s\) team'),
    (lambda a: a['ftg/ftg_node'].pop('repo'), r'algorithms\.ftg/ftg_node: missing repo'),
    (lambda a: a['ftg/ftg_node'].update(role='racing'), "'racing' is not one of driving, mapping, manual"),
    (lambda a: a['ftg/ftg_node'].update(node='ftg-node'), "node: expected a name, got 'ftg-node'"),
    (lambda a: a['ftg/ftg_node'].update(node=3), 'node: expected a name, got 3'),
    (lambda a: a.clear(), 'algorithms: none listed'),
])
def test_a_list_that_cannot_name_code_is_refused(change, error):
    with pytest.raises(RulesError, match=error):
        check_rules(_rules_with(change))


def test_rules_without_an_algorithms_list_are_refused():
    data = yaml.safe_load(open(RULES.path))
    del data['algorithms']
    with pytest.raises(RulesError, match='rules: missing algorithms'):
        check_rules(data)


def test_a_repository_name_may_hold_a_hyphen_but_a_node_name_may_not():
    check_rules(_rules_with(lambda a: a['ftg/ftg_node'].update(repo='team-a')))
    with pytest.raises(RulesError, match="parameter: expected a name, got 'enable-deadman'"):
        check_rules(_rules_with(lambda a: a['gap_follow/gap_follow_node'].update(parameter='enable-deadman')))
