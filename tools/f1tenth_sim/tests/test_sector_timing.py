from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sector_trials import assess


def test_square_times_from_rest_gate_and_partial_finish():
    # Four 1 m edges, one every second. Three closed laps on the line;
    # the shared gate is its last station, 3 m from the start.
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    elapsed = np.arange(241) / 20.
    k = elapsed.astype(int) % 4
    u = (elapsed % 1)[:, None]
    pos = (1-u) * square[k] + u * square[(k+1) % 4]
    wall = np.full(240, .3)
    laps, sectors, err = assess(square, pos, elapsed, wall)
    assert laps == pytest.approx([3., 4., 4.], abs=1e-10)
    assert np.max(err) == pytest.approx(0., abs=1e-10)
    for s in sectors:
        assert s['times'] == pytest.approx([4./6], abs=1e-10)
        assert s['min_wall'] == pytest.approx(.3, abs=1e-10)
    laps, secs, _ = assess(square, pos[:61], elapsed[:61], wall[:60])
    assert laps == pytest.approx([3.], abs=1e-10)
    assert all(s is None for s in secs)
