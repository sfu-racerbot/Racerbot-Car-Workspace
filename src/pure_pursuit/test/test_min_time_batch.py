"""The minimum-time optimizer's batched lap time and gradient against the single-path code they replace.

ROS-free, run by pytest and the pure_pursuit package runner:
    python3 -m pytest src/pure_pursuit/test/test_min_time_batch.py -q

Oracles: each batched function is checked row by row against the single-path function it copies
(min_time_line.speed_profile/lap_time, raceline_optimizer.signed_curvature, numpy.interp), and the
gradient against scipy's own approx_derivative, which L-BFGS-B called before (2026-11-10). The
parallel starts are checked against the same starts run in turn. The optimizer's own result on a
corridor (faster, margins kept) is test_min_time_line.py's test_optimize_stadium_corridor, which runs the
batched path by default.
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from pure_pursuit import min_time_line as mtl
from pure_pursuit import race_line
from pure_pursuit.raceline_optimizer import signed_curvature


def wobbly_paths(rows=6, n=240, seed=0):
    """Closed paths with left and right turns, uneven spacing and a tight hairpin, one per row."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 2.0 * np.pi, n))
    paths = []
    for r in range(rows):
        x = 6.0 * np.cos(t) + 0.3 * r * np.cos(2 * t)
        y = 3.0 * np.sin(t) + (0.6 + 0.1 * r) * np.sin(3 * t)
        paths.append(np.column_stack((x, y)) + rng.normal(0.0, 0.005, (n, 2)))
    return np.stack(paths)


PHYSICS = [mtl.CarPhysics(), mtl.CarPhysics(speed_delay=0.22, a_lat_left=2.8, a_lat_right=2.8),
           mtl.CarPhysics(speed_delay=0.0, steering_delay=0.0)]


@pytest.mark.parametrize('physics', PHYSICS)
def test_batched_speed_profile_and_lap_time_match_the_single_path_functions(physics):
    paths = wobbly_paths()
    speed, kappa, seg = mtl._batch_speed_profile(paths, physics)
    laps = mtl._batch_lap_time(paths, physics)
    for r, xy in enumerate(paths):
        want_speed, want_kappa, want_seg = mtl.speed_profile(xy, physics)
        # oracle: the single-path functions the optimizer called before, row by row.
        np.testing.assert_allclose(speed[r], want_speed, rtol=0, atol=1e-12)  # m/s
        np.testing.assert_allclose(kappa[r], want_kappa, rtol=0, atol=1e-12)  # 1/m
        np.testing.assert_allclose(seg[r], want_seg, rtol=0, atol=1e-12)      # m
        assert laps[r] == pytest.approx(mtl.lap_time(xy, physics), abs=1e-12)  # s


def test_the_speed_delay_and_steering_rate_actually_bind_on_these_paths():
    # Guards the test above against passing vacuously: the response delay and the steering rate must
    # each change the lap on these paths, or matching them would prove nothing about those branches.
    paths = wobbly_paths()
    base = mtl.CarPhysics(speed_delay=0.0, steering_delay=0.0)
    for changed in (replace(base, speed_delay=0.22), replace(base, steering_delay=0.10, steering_rate=0.5)):
        assert np.all(mtl._batch_lap_time(paths, changed) > mtl._batch_lap_time(paths, base) + 1e-3)  # s


def test_batched_curvature_matches_including_degenerate_points():
    paths = wobbly_paths(rows=3)
    paths[1, 10] = paths[1, 11]                                    # a repeated point
    paths[2, 20:23] = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]         # three collinear points
    got = mtl._batch_curvature(paths)
    for r, xy in enumerate(paths):
        # oracle: signed_curvature, which returns 0 wherever its Menger denominator vanishes.
        np.testing.assert_array_equal(got[r], signed_curvature(xy))
    assert got[1, 10] == 0.0 and got[1, 11] == 0.0


def test_row_interp_matches_numpy_including_both_ends():
    rng = np.random.default_rng(3)
    xp = np.cumsum(rng.uniform(0.01, 0.2, (4, 50)), axis=1)
    xp = np.concatenate([np.zeros((4, 1)), xp], axis=1)
    fp = rng.normal(0.0, 1.0, xp.shape)
    x = rng.uniform(0.0, 1.0, (4, 30)) * xp[:, -1:]
    x[:, 0], x[:, 1], x[:, 2] = 0.0, xp[:, -1], xp[:, 7]           # first knot, last knot, an interior knot
    got = mtl._row_interp(x, xp, fp)
    for r in range(4):
        # oracle: numpy.interp, which speed_profile calls.
        np.testing.assert_allclose(got[r], np.interp(x[r], xp[r], fp[r]), rtol=0, atol=1e-12)


def test_row_searchsorted_matches_numpy_per_row():
    rng = np.random.default_rng(4)
    table = np.sort(rng.uniform(-5.0, 50.0, (5, 40)), axis=1)
    query = rng.uniform(-6.0, 51.0, (5, 25))
    query[:, 0] = table[:, 3]                                      # exact ties
    for side in ('left', 'right'):
        got = mtl._row_searchsorted(table, query, side)
        for r in range(5):
            np.testing.assert_array_equal(got[r], np.searchsorted(table[r], query[r], side=side))


def lap_of_offsets(paths_center, normals, physics):
    def batch_cost(controls):
        xy = paths_center[None] + np.repeat(controls, len(paths_center) // controls.shape[1], axis=1)[..., None] \
            * normals[None]
        return mtl._batch_lap_time(xy, physics)
    return batch_cost


@pytest.mark.parametrize('case', ['interior', 'at_upper', 'at_lower', 'unbounded'])
def test_gradient_takes_exactly_scipys_steps(case):
    center = wobbly_paths(rows=1)[0]
    from pure_pursuit.raceline_optimizer import path_frames
    _, normals = path_frames(center)
    cost = lap_of_offsets(center, normals, PHYSICS[1])
    m = 24
    rng = np.random.default_rng(7)
    x0 = rng.uniform(-0.05, 0.05, m)
    lo, hi = np.full(m, -0.1), np.full(m, 0.1)
    if case == 'at_upper':
        x0[:6] = hi[:6]                    # +step would leave the bounds: scipy steps backwards
        x0[6] = hi[6] - 0.5e-4             # inside, but +step overshoots
    elif case == 'at_lower':
        x0[:6] = lo[:6]
    elif case == 'unbounded':
        lo, hi = np.full(m, -np.inf), np.full(m, np.inf)
    f0, grad = mtl.forward_difference_gradient(cost, x0, lo, hi, 1e-4)
    # oracle: scipy's 2-point derivative with L-BFGS-B's absolute eps and bounds, which the optimizer used.
    want = approx_derivative(lambda x: cost(x[None])[0], x0, method='2-point', abs_step=1e-4, bounds=(lo, hi))
    assert f0 == cost(x0[None])[0]
    np.testing.assert_array_equal(grad, want)
    assert np.all(np.isfinite(grad)) and np.any(grad != 0.0)


def test_a_control_fixed_by_its_bounds_gets_zero_not_nan():
    center = wobbly_paths(rows=1)[0]
    from pure_pursuit.raceline_optimizer import path_frames
    _, normals = path_frames(center)
    cost = lap_of_offsets(center, normals, PHYSICS[0])
    m = 12
    x0 = np.zeros(m)
    lo, hi = np.full(m, -0.1), np.full(m, 0.1)
    lo[3] = hi[3] = 0.0                    # a repair pulled both bounds onto the corridor's middle
    _, grad = mtl.forward_difference_gradient(cost, x0, lo, hi, 1e-4)
    # oracle: closed form; a control that cannot move has zero derivative within its bounds. scipy itself
    # divides 0 by 0 here.
    assert grad[3] == 0.0
    with np.errstate(invalid='ignore'):
        scipy_grad = approx_derivative(lambda x: cost(x[None])[0], x0, method='2-point', abs_step=1e-4,
                                       bounds=(lo, hi))
    assert np.isnan(scipy_grad[3])
    np.testing.assert_array_equal(np.delete(grad, 3), np.delete(scipy_grad, 3))


def test_parallel_starts_give_exactly_the_serial_candidates(monkeypatch):
    # Each start runs in a forked worker; the result must be what running them in turn gives.
    from pure_pursuit.occupancy_map import OccupancyMap
    from pure_pursuit.racing_math import load_xy_csv
    from pathlib import Path
    data = Path(__file__).parent / 'data' / 'course_20261106a'
    grid = OccupancyMap.from_yaml(data / 'map.yaml', inflate_cells=0)
    seed = load_xy_csv(data / 'raceline_raw.csv')
    cfg = race_line.RaceLineConfig()
    calls = []
    real = race_line._start_results

    def spy(names, *rest):
        calls.append(list(names))
        return real(names, *rest)

    monkeypatch.setattr(race_line, '_start_results', spy)
    parallel = race_line.build_race_line(grid, seed, cfg)
    assert calls == [list(cfg.min_time_starts)]

    def serial(names, mc, prepared, field_at, cfg):
        return [(n, race_line._min_time_from(race_line._start_offsets(n, mc, prepared, field_at, cfg),
                                             mc, field_at, cfg)) for n in names]

    monkeypatch.setattr(race_line, '_start_results', serial)
    alone = race_line.build_race_line(grid, seed, cfg)
    got = {c.source: c for c in parallel.candidates}
    want = {c.source: c for c in alone.candidates}
    # oracle: invariant, the same deterministic work in another process gives the same numbers.
    assert got.keys() == want.keys()
    for source in want:
        np.testing.assert_array_equal(got[source].xy, want[source].xy)
        assert got[source].reason == want[source].reason
