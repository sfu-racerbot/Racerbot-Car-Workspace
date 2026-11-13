"""start_replay's geometry helpers against closed forms (a circle of known radius).

Oracles: a counter-clockwise circle of radius 10 m centred on the origin, sampled every 0.5 degree, has arc
length 2*pi*10 and a point at radius r on the left of the direction of travel (inside) is r_line - r to the
left. Run by: python3 -m pytest tools/f1tenth_sim/tests/ -q -p no:cacheprovider
"""
import math
import sys
from pathlib import Path

import pytest

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM.parents[1] / 'tools'))
sys.path.insert(0, str(SIM.parents[1] / 'src' / 'pure_pursuit'))
import numpy as np  # noqa: E402
import start_replay as sr  # noqa: E402

R = 10.0


def circle():
    angle = np.radians(np.arange(0.0, 360.0, 0.5))
    return sr.line_geometry(np.column_stack([R * np.cos(angle), R * np.sin(angle)]))


def test_arc_length_and_projection_follow_the_circle():
    xy, seg, ds, arc = circle()
    assert ds.sum() == pytest.approx(2 * math.pi * R, abs=0.005)       # m, chord error of 0.5 deg steps
    s, off = sr.project(xy, seg, ds, arc, R * math.cos(math.radians(90)), R * math.sin(math.radians(90)))
    assert s == pytest.approx(R * math.pi / 2, abs=0.05)               # m, a quarter turn
    assert off == pytest.approx(0.0, abs=0.01)


def test_offset_is_positive_to_the_left_of_travel():
    xy, seg, ds, arc = circle()
    inside = sr.project(xy, seg, ds, arc, 9.0, 0.0)[1]               # nearer the centre: left of a CCW line
    outside = sr.project(xy, seg, ds, arc, 10.7, 0.0)[1]
    assert inside == pytest.approx(1.0, abs=0.01)    # m
    assert outside == pytest.approx(-0.7, abs=0.01)


def test_lap_times_are_the_gaps_between_start_finish_crossings():
    length = 100.0
    t = np.arange(0.0, 40.0, 0.025)
    arcs = (t * 10.0) % length                       # 10 m/s: a lap every 10 s
    laps = sr.lap_times(t, arcs, length)
    assert laps == pytest.approx([10.0, 10.0], abs=0.03)       # s; crossings at 10, 20, 30 s are two laps
    assert sr.lap_times(t[:300], arcs[:300], length) == []           # under one crossing: no lap
