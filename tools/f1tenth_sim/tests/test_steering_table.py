"""Offline table: analytical bicycle with 75% grip, independent of the fitter."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2]
sys.path.extend(map(str, (TOOLS, TOOLS / 'f1tenth_sim')))
from steering_table import compensate, fit, predict


def calibrated():
    # Same speed range in adjacent asked bins; different force at each wheel angle.
    rows = [(v, math.atan(a * .36 / v**2), .75 * a / v)
            for a in (-1.25, -.75, .75, 1.25)
            for v in np.linspace(2., 2.4, 12)]
    return fit(np.array(rows), .36, indexed=True)


def test_inverse_and_forward_match_separate_signed_bicycle_predictions():
    table = calibrated()
    for sign in (-1, 1):
        v, asked = 2.2, sign * .9
        cmd = math.atan(asked * .36 / v**2)
        assert predict(table, v, cmd) == pytest.approx(.75 * asked / v, abs=1e-10)
        assert compensate(table, v, cmd) == pytest.approx(
            math.atan(asked / .75 * .36 / v**2), abs=1e-10)


def test_sparse_stale_and_out_of_range_fall_back():
    table = calibrated()
    v = 2.2
    steer = math.atan(.8 * .36 / v**2)
    assert compensate(table, v, steer, age=.1) is not None
    for speed, cmd, age in [(v, steer, .10001), (v, steer, -1), (2.5, steer, 0),
                            (0, steer, 0), (v, .4, 0), (v, 0., 0),
                            (float('nan'), steer, 0), (v, float('inf'), 0)]:
        assert compensate(table, speed, cmd, age) is None
    empty = fit(np.array([[v, steer, .5]]), .36, indexed=False)
    assert compensate(empty, v, steer) is None
