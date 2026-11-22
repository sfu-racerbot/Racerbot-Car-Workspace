"""Independent geometry and observer checks; python3 -m pytest tools/dev/tests/."""
import argparse
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'benchmark_map_import.py'
SPEC = importlib.util.spec_from_file_location('benchmark_map_import', PATH)
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


def rectangle(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def grid():
    # Oracle: only image pixel (2,2), world square [0,1] x [0,1], is occupied.
    free = np.ones((5, 5), dtype=bool)
    free[2, 2] = False
    return bench.GridGeometry(free, 1., -2., -2.)


def test_grid_coordinates_and_outside():
    g = grid()
    assert g.pixel([.5, .5]) == (2, 2)
    assert g.pixel([-2., -2.]) == (4, 0)
    np.testing.assert_array_equal(g.centers([[2, 2], [0, 4]]), [[.5, .5], [2.5, 2.5]])
    assert not g.is_free([.5, .5])
    assert g.is_free([-.001, .5])
    assert not g.is_free([-2.001, .5])
    assert not g.is_free([3., .5])


@pytest.mark.parametrize('corners,expected', [
    (rectangle(.1, .1, .9, .9), [[2, 2]]),
    (rectangle(-.1, -.1, 1.1, 1.1), [[2, 2]]),
    (rectangle(1., .1, 1.5, .9), [[2, 2]]),  # closed squares: exact edge contact
    (rectangle(1.001, .1, 1.5, .9), []),
    (rectangle(-5., -5., -4., -4.), []),
    # Crossing edges with no vertex inside the other polygon.
    (rectangle(-1., .4, 2., .6), [[2, 2]]),
    # Rotated rectangle's bbox overlaps the wall but its body does not.
    (np.array([[.8, 1.4], [1.4, .8], [1.5, .9], [.9, 1.5]]), []),
])
def test_cell_contact_uses_all_separating_axes(corners, expected):
    # Oracle: analytic intersections against the unit square described in grid().
    assert grid().contacts(corners).tolist() == expected


def test_empty_and_single_cell_grids():
    g = bench.GridGeometry(np.ones((0, 0), dtype=bool), 1., 0., 0.)
    assert g.contacts(rectangle(0., 0., .1, .1)).tolist() == []
    assert not g.is_free([0., 0.])
    g = bench.GridGeometry(np.zeros((1, 1), dtype=bool), 1., 0., 0.)
    assert g.contacts(rectangle(.1, .1, .2, .2)).tolist() == [[0, 0]]


def test_audit_detects_contact_and_old_region_excursion():
    # Oracle: the candidate's own 5cm square is a wall before and free after.
    x, y = bench.CANDIDATES[0]
    old = bench.GridGeometry(np.zeros((1, 1), dtype=bool), .05, x - .025, y - .025)
    after = bench.GridGeometry(np.ones((1, 1), dtype=bool), .05, x - .025, y - .025)
    corners = rectangle(x - .01, y - .01, x + .01, y + .01)
    pose = np.array([x, y, 0., 0., .3])
    before_audit = bench.EpisodeAudit(old, old)
    assert before_audit.summary()['independent_contact_ticks'] == 0
    assert before_audit.summary()['first_outside_old_region'] is None
    before_audit.record(.025, pose, corners, False)
    s = before_audit.summary()
    assert s['independent_collision']
    assert s['near_candidate_contact_ticks'] == [1, 0]
    assert s['outside_old_region_poses'] == 1
    assert s['first_contact']['occupied_cells_rc'] == [[0, 0]]
    assert s['first_contact']['near_candidate_cells'] == [True, False]
    after_audit = bench.EpisodeAudit(after, old)
    after_audit.record(.025, pose, corners, False)
    s = after_audit.summary()
    assert not s['independent_collision']
    assert s['outside_old_region_poses'] == 1


def test_observer_preserves_results_and_restores_factory_on_error():
    events = []
    reset_result = ({'observation': 1}, {'sim_time': 0.})
    step_result = ({'observation': 2}, 3., False, False, {'sim_time': .025})
    corners = rectangle(-1., -1., -.5, -.5)
    plant = SimpleNamespace(
        reset=lambda *a, **kw: reset_result, step=lambda command: events.append(command) or step_result,
        truth=lambda index: np.array([-1., -1., 0., 0., .2]),
        _chassis_outline=lambda index: corners, _wall_contact=lambda: set())
    factory = lambda seed: plant
    module = SimpleNamespace(make_plant=factory)
    audit = bench.EpisodeAudit(grid(), grid())
    with pytest.raises(RuntimeError, match='stop'):
        with bench.instrument_factory(module, audit):
            observed = module.make_plant(123)
            assert observed.reset([[1., 2., 3.]]) is reset_result
            assert observed.step([(0., 1.)]) is step_result
            raise RuntimeError('stop')
    assert module.make_plant is factory
    assert events == [[(0., 1.)]]
    assert audit.poses == [[0., -1., -1., .2, False], [.025, -1., -1., .2, False]]
    assert not audit.summary()['independent_collision']


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf'])
def test_invalid_timeouts(value):
    with pytest.raises(argparse.ArgumentTypeError, match='positive and finite'):
        bench.positive(value)


def test_positive_timeout():
    assert bench.positive('0.025') == .025
