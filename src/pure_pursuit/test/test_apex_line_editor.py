"""Run with pytest tools/f1tenth_sim/tests/test_apex_line_editor.py.
Oracles: synthetic annulus geometry, export contract, preservation invariants.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_pursuit import apex_line_editor as editor


@pytest.fixture
def run(tmp_path):
    angle = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    xy = 3 * np.column_stack((np.cos(angle), np.sin(angle)))
    np.savetxt(tmp_path / 'raceline_profiled.csv', np.c_[xy, np.ones(len(xy))], delimiter=',')
    yy, xx = np.mgrid[:200, :200]
    radius = np.hypot((xx + .5) * .05 - 5, (yy + .5) * .05 - 5)
    Image.fromarray(np.where((radius > 2) & (radius < 4), 255, 0).astype('uint8')).save(tmp_path / 'map.pgm')
    (tmp_path / 'map.yaml').write_text(yaml.safe_dump(dict(image='map.pgm', resolution=.05,
        origin=[-5, -5, 0], negate=0, occupied_thresh=.65, free_thresh=.196)))
    (tmp_path / 'start_pose.yaml').write_text('x: 3\ny: 0\nyaw: 1.57079632679\n')
    doc = dict(format='racerbot-apexes', version=1, frame='map', line=editor.fingerprint(xy),
               apexes=[dict(x=3.1, y=0, waypoint_index=0, s_m=0, distance_to_line_m=.1)])
    (tmp_path / 'apexes.json').write_text(json.dumps(doc))
    return tmp_path


def test_export_feeds_pure_pursuit_and_preserves_originals(run):
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    command = editor.edit(run, run / 'apexes.json', run / 'edited')
    loaded_xy, loaded_speed = editor.racing_math.load_profiled_csv(str(run / 'edited/raceline_profiled.csv'))
    out = np.c_[loaded_xy, loaded_speed]
    # oracle: invariant, requested apex reached and positive bounded speeds.
    a = out[:, :2]
    delta = np.roll(a, -1, axis=0) - a
    t = np.clip(np.sum((np.array([3.1, 0]) - a) * delta, axis=1) / np.sum(delta**2, axis=1), 0, 1)
    assert np.min(np.linalg.norm(a + t[:, None]*delta - [3.1, 0], axis=1)) <= .02
    assert np.all((out[:, 2] > 0) & (out[:, 2] <= 4))
    assert 'steering_law:=pure_pursuit' in command
    assert f'run_dir:={run}/edited' in command
    assert all((run / name).read_bytes() == data for name, data in before.items())
    assert (run / 'edited/start_pose.yaml').read_bytes() == before['start_pose.yaml']
    assert editor.OccupancyMap.from_yaml(str(run / 'edited/map.yaml')).is_blocked(0, 0)
    checks = json.loads((run / 'edited/apex_edit.json').read_text())['checks']
    assert max(checks['apex_distances_m']) <= .02  # oracle: spec editor tolerance in metres
    with pytest.raises(ValueError, match='already exists'):
        editor.edit(run, run / 'apexes.json', run / 'edited')


@pytest.mark.parametrize('damage', ['fingerprint', 'unassociated', 'nan', 'empty', 'wrong_type'])
def test_bad_requests_refused(run, damage):
    path = run / 'apexes.json'
    doc = json.loads(path.read_text())
    if damage == 'fingerprint':
        doc['line']['fnv1a'] = '00000000'
    elif damage == 'unassociated':
        doc['apexes'][0]['waypoint_index'] = None
    elif damage == 'nan':
        doc['apexes'][0]['x'] = float('nan')
    elif damage == 'empty':
        doc['apexes'] = []
    else:
        doc['apexes'] = [1]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='fingerprint|waypoint|finite|apex'):
        editor.edit(run, path, run / 'edited')
    assert not (run / 'edited').exists()  # oracle: refusal never exports


@pytest.mark.parametrize('radius,reason', [(3.8, 'clearance'), (.8, 'curvature')])
def test_unsafe_geometry_refused(run, radius, reason):
    angle = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    xy = radius * np.c_[np.cos(angle), np.sin(angle)]
    params = yaml.safe_load(editor.CONFIG.read_text())['auto_map_race_node']['ros__parameters']
    with pytest.raises(ValueError, match=reason):
        editor.validate_line(xy, xy[:1], editor.OccupancyMap.from_yaml(str(run / 'map.yaml')), params)


def test_fingerprint_js_negative_half_rounding():
    # oracle: spec apex.js Math.round(-0.5) is 0, unlike floor(-0.5).
    assert editor.fingerprint([[-.0005, .0005]]) == editor.fingerprint([[0, .001]])
    assert editor.fingerprint([[1, 2]]) != editor.fingerprint([[2, 1]])


def test_optimizer_none_is_identical():
    angle = np.linspace(0, 2*np.pi, 60, endpoint=False)
    xy = 3 * np.c_[np.cos(angle), np.sin(angle)]
    args = (xy, np.ones(60), np.ones(60), .155, .1, .3)
    a = editor.optimizer.optimize_minimum_curvature(*args, iterations=2)
    b = editor.optimizer.optimize_minimum_curvature(*args, iterations=2, apex_points=None)
    # oracle: invariant optional pins cannot change legacy results.
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize('name,content', [
    ('apexes.json', '{'), ('apexes.json', '[]'), ('apexes.json', 'null'),
    ('raceline_profiled.csv', 'x,y,speed\n1,2'),
    ('raceline_profiled.csv', '1,2,3\nwrong,2,3'),
    ('raceline_profiled.csv', ''), ('apexes.json', '\ufffd')])
def test_corrupt_inputs(run, name, content):
    (run / name).write_text(content)
    with pytest.raises(ValueError, match='CSV|expected|Expecting|convert|delimiter'):
        editor.edit(run, run / 'apexes.json', run / 'edited')
    assert not (run / 'edited').exists()  # oracle: refusal invariant


@pytest.mark.parametrize('index', [-1, 120, None])
def test_invalid_index(run, index):
    path = run / 'apexes.json'
    doc = json.loads(path.read_text())
    doc['apexes'][0]['waypoint_index'] = index
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='waypoint_index'):
        editor.read_inputs(run, path)


def test_wall_apex_export_refused(run):
    path = run / 'apexes.json'
    doc = json.loads(path.read_text())
    doc['apexes'][0]['x'] = 4.1
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='corridor|clearance|apex'):
        editor.edit(run, path, run / 'edited')
    assert not (run / 'edited').exists()  # oracle: no unsafe export



def test_cli_module_from_another_directory(run):
    import os
    import subprocess
    env = dict(os.environ)
    # Preserve sourced dependencies and the subprocess hardware-isolation hook.
    env['PYTHONPATH'] = (str(Path(editor.__file__).resolve().parents[1])
                         + os.pathsep + env.get('PYTHONPATH', ''))
    result = subprocess.run([sys.executable, '-m', 'pure_pursuit.apex_line_editor',
        '--run', str(run), '--apexes', str(run / 'apexes.json'),
        '--out', str(run / 'cli')], cwd='/tmp', env=env,
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    xy, speed = editor.racing_math.load_profiled_csv(str(run / 'cli/raceline_profiled.csv'))
    # oracle: CLI emits a positive-speed controller-readable profile at the requested apex.
    assert np.all(speed > 0)
    assert np.min(np.linalg.norm(xy - [3.1, 0], axis=1)) <= .02
