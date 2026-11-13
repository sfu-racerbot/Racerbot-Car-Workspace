"""racing_math.body_arc_free_distance must equal gap_follow's copy.

The two packages do not import each other, so the function is duplicated
(like vehicle_boundary_distances). Run with `rb`; unsourced, the gap_follow
import is skipped so the file still collects.
"""
import numpy as np
import pytest

from pure_pursuit import racing_math

gap_logic = pytest.importorskip('gap_follow.gap_logic')


@pytest.mark.parametrize('low', [0.05, 0.6])
def test_matches_gap_follows_copy_on_random_scans(low):
    """low=0.05 is the plan's spec (most scans already overlap the body, so
    the result is mostly 0.0); low=0.6 leaves a gap so hits land at varied
    travel distances, which is what actually exercises the arc."""
    rng = np.random.default_rng(20261112)
    angles = np.linspace(-2.35, 2.35, 1081)
    finite_results = 0
    distinct = set()
    for _ in range(200):
        ranges = rng.uniform(low, 5.0, angles.shape)
        valid = rng.random(angles.shape) > 0.05
        curvature = float(rng.uniform(-1.2, 1.2))
        args = (ranges, valid, angles, curvature, 0.175, 0.58, 0.36, 0.26)
        a = gap_logic.body_arc_free_distance(*args)
        b = racing_math.body_arc_free_distance(*args)
        assert a == b
        finite_results += np.isfinite(a)
        distinct.add(a)
    # the comparison must exercise hits, not only inf == inf
    assert finite_results >= 10
    if low > 0.5:
        assert len(distinct) >= 5
