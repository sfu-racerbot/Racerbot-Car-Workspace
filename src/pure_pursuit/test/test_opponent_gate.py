"""OpponentGate and WallDistance: is a detected cluster a car worth racing against?

Oracles: closed forms (40 Hz sightings, speeds from arc-length deltas), boundary values of each parameter,
and a recorded measurement -- every forward-cone detection replayed from car 2's solo run 20261106-175300,
where a phantom made the real node commit to an overtake and end in body_contact
(test/data/phantom_20261106-175300.csv; tools/run_analysis/replay_opponent_gate.py).
Runner: python3 -m pytest src/pure_pursuit/test/ (and colcon test), no ROS needed.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from pure_pursuit.opponent_gate import OpponentGate, WallDistance

TRACK = 32.0  # m, closed track length used by the synthetic cases
HZ = 40.0     # the Hokuyo's scan rate on both cars
PHANTOM = Path(__file__).parent / 'data' / 'phantom_20261106-175300.csv'


def feed(gate, speed, n, start_arc=3.0, wall=1.0, x0=0.0):
    """n sightings at 40 Hz of a car moving `speed` m/s along the track (and along x in the map)."""
    verdicts = []
    for k in range(n):
        t = k / HZ
        verdicts.append(gate.observe(t, x0 + speed * t, 0.0, (start_arc + speed * t) % TRACK, TRACK,
                                     wall_distance=wall))
    return verdicts


@pytest.mark.parametrize('speed,start_arc', [(1.0, 3.0), (0.0, 3.0), (2.0, 31.95)])
def test_a_car_is_confirmed_on_the_seventh_sighting(speed, start_arc):
    # Closed form: six intervals at 40 Hz span 6/40 = 0.15 s = min_span_sec, so sighting 7 (index 6) is the
    # first with both 6 sightings and the span. The 31.95 m start crosses the finish line mid-window.
    verdicts = feed(OpponentGate(), speed, 7, start_arc)
    assert [v.confirmed for v in verdicts] == [False] * 6 + [True]


def test_the_recorded_phantom_is_never_confirmed():
    # Recorded measurement, run 20261106-175300: no other car was on the track, so every one of these
    # detections is false. The real node overtook on them; the gate must confirm none.
    data = np.loadtxt(PHANTOM, delimiter=',', comments='#', skiprows=4)
    total = float(PHANTOM.read_text().split('# total_length ')[1].split()[0])
    gate = OpponentGate()
    verdicts = [gate.observe(t, x, y, arc, total, wall_distance=wall) for t, x, y, arc, wall in data]
    assert len(verdicts) == 115
    assert not any(v.confirmed for v in verdicts)
    # What rejects it: the phantom is a spot in the map 0.05-0.25 m from a mapped wall.
    assert sum('from a mapped wall' in v.reason for v in verdicts) == len(verdicts)


def test_without_the_wall_check_the_recorded_phantom_gets_through():
    # The same data with the wall check off: time and motion alone confirm it (it sits still in the map,
    # as a stopped car would). Shows the test above can fail, and which check carries it.
    data = np.loadtxt(PHANTOM, delimiter=',', comments='#', skiprows=4)
    total = float(PHANTOM.read_text().split('# total_length ')[1].split()[0])
    gate = OpponentGate(min_wall_distance=0.0)
    confirmed = sum(gate.observe(t, x, y, arc, total, wall_distance=wall).confirmed
                    for t, x, y, arc, wall in data)
    assert confirmed > 10


@pytest.mark.parametrize('wall,confirmed', [(0.30, True), (0.2999, False), (0.0, False)])
def test_wall_distance_boundary(wall, confirmed):
    # Boundary: min_wall_distance 0.30 m is the closest a detection may be to a wall and still count.
    assert feed(OpponentGate(), 1.0, 7, wall=wall)[-1].confirmed is confirmed


def test_a_sighting_near_a_wall_restarts_the_count():
    gate = OpponentGate()
    feed(gate, 0.0, 6)
    verdict = gate.observe(6 / HZ, 0.0, 0.0, 3.0, TRACK, wall_distance=0.1)
    assert not verdict.confirmed
    assert verdict.reason == '0.10 m from a mapped wall, under the 0.30 m a car would be'
    assert gate.observe(7 / HZ, 0.0, 0.0, 3.0, TRACK, wall_distance=1.0).reason == 'seen 1 of 6 needed'


def test_backwards_motion_is_rejected_with_its_speed():
    # Closed form: arc length falling 5.4 m/s (the closing rate the node logged on 175300, 6.9 m/s against
    # the ego's 1.5) over the 0.15 s span reads as 5.4 m/s backwards.
    verdict = feed(OpponentGate(), -5.4, 7, start_arc=20.0, x0=0.0)[-1]
    assert not verdict.confirmed
    assert verdict.reason == 'moved backwards along the track at 5.4 m/s'


@pytest.mark.parametrize('speed,confirmed', [(-1.5, True), (-1.6, False), (8.0, True), (8.1, False)])
def test_speed_limits_are_inclusive(speed, confirmed):
    # Boundaries: max_reverse_speed 1.5 m/s, max_speed 8.0 m/s, judged exactly. 32 Hz timestamps are binary
    # fractions, so at the 7th sighting (6/32 s) arc 10 + 1.5 * 0.1875 and the speed are exact in floating
    # point. Map x is held still so no step exceeds max_jump_m; only the along-track speed varies.
    gate = OpponentGate()
    verdicts = [gate.observe(k / 32.0, 0.0, 0.0, 10.0 + speed * k / 32.0, TRACK, wall_distance=1.0)
                for k in range(7)]
    assert verdicts[-1].confirmed is confirmed


def test_six_sightings_are_needed_even_when_the_span_is_long_enough():
    # Closed form: at 20 Hz, 5 sightings already span 4/20 = 0.2 s > min_span_sec, so only confirm_count (6)
    # holds the 5th back.
    gate = OpponentGate()
    verdicts = [gate.observe(k / 20.0, 0.0, 0.0, 3.0, TRACK, wall_distance=1.0) for k in range(6)]
    assert [v.confirmed for v in verdicts] == [False] * 5 + [True]
    assert verdicts[4].reason == 'seen 5 of 6 needed'


def test_too_fast_names_the_speed():
    gate = OpponentGate()
    verdicts = [gate.observe(k / HZ, 0.0, 0.0, 10.0 + 9.0 * k / HZ, TRACK, wall_distance=1.0) for k in range(7)]
    assert verdicts[-1].reason == 'moved along the track too fast at 9.0 m/s'


def test_a_jump_restarts_the_count():
    gate = OpponentGate()
    feed(gate, 0.0, 7)
    # Contract: a step over max_jump_m (0.5 m) in the map is a different object.
    verdict = gate.observe(7 / HZ, 0.51, 0.0, 3.0, TRACK, wall_distance=1.0)
    assert verdict.reason == 'seen 1 of 6 needed'
    gate2 = OpponentGate()
    feed(gate2, 0.0, 7)
    assert gate2.observe(7 / HZ, 0.49, 0.0, 3.0, TRACK, wall_distance=1.0).confirmed


def test_sightings_older_than_the_window_drop_out():
    gate = OpponentGate()
    feed(gate, 0.0, 7)
    # 0.451 s after the first sighting the window (0.3 s) holds only sightings from 0.151 s on: none.
    verdict = gate.observe(0.151 + 0.3, 0.0, 0.0, 3.0, TRACK, wall_distance=1.0)
    assert verdict.reason == 'seen 1 of 6 needed'


def test_time_going_backwards_restarts_the_count():
    gate = OpponentGate()
    feed(gate, 0.0, 7)
    assert gate.observe(0.1, 0.0, 0.0, 3.0, TRACK, wall_distance=1.0).reason == 'seen 1 of 6 needed'


@pytest.mark.parametrize('index', range(5))
@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf])
def test_non_finite_input_refuses_and_clears(index, bad):
    gate = OpponentGate()
    feed(gate, 0.0, 7)
    args = [7 / HZ, 0.0, 0.0, 3.0, TRACK]
    args[index] = bad
    assert gate.observe(*args, wall_distance=1.0).reason == 'non-finite opponent observation'
    assert gate.observe(8 / HZ, 0.0, 0.0, 3.0, TRACK, wall_distance=1.0).reason == 'seen 1 of 6 needed'


def test_nan_wall_distance_refuses_but_inf_means_no_map():
    gate = OpponentGate()
    assert gate.observe(0.0, 0.0, 0.0, 3.0, TRACK, wall_distance=math.nan).reason == \
        'non-finite opponent observation'
    assert feed(OpponentGate(), 0.0, 7, wall=math.inf)[-1].confirmed


def test_confirm_count_one_and_no_span_confirms_at_once():
    # The settings test_opponent_integration.py's fixtures use, so its pass-side tests stay about passing.
    gate = OpponentGate(confirm_count=1, min_span_sec=0.0)
    assert gate.observe(0.0, 0.0, 0.0, 3.0, TRACK, wall_distance=1.0).confirmed
    assert not gate.observe(1e-3, 0.0, 0.0, 3.0, TRACK, wall_distance=0.1).confirmed


@pytest.mark.parametrize('name', ['confirm_count', 'window_sec', 'min_span_sec', 'max_jump_m',
                                  'max_reverse_speed', 'max_speed', 'min_wall_distance'])
@pytest.mark.parametrize('value', [-1, math.nan, math.inf])
def test_invalid_parameters_are_named(name, value):
    with pytest.raises(ValueError, match=name):
        OpponentGate(**{name: value})


@pytest.mark.parametrize('name', ['confirm_count', 'window_sec', 'max_jump_m', 'max_speed'])
def test_zero_is_refused_where_it_means_nothing(name):
    with pytest.raises(ValueError, match=name):
        OpponentGate(**{name: 0})


def test_fractional_count_and_span_past_window_are_refused():
    with pytest.raises(ValueError, match='confirm_count'):
        OpponentGate(confirm_count=2.5)
    with pytest.raises(ValueError, match='cannot exceed window_sec'):
        OpponentGate(window_sec=0.3, min_span_sec=0.31)


def _grid(width=40, height=20, wall_row=0):
    data = np.zeros((height, width), dtype=np.int16)
    data[wall_row, :] = 100
    return data.flatten().tolist()


def test_wall_distance_closed_form():
    # 0.05 m cells, wall along row 0 (y in [0, 0.05)); origin (-1, 0). The distance transform measures
    # cell centre to blocked cell centre: a point in row r is r * 0.05 m from row 0.
    walls = WallDistance(_grid(), 40, 20, 0.05, -1.0, 0.0)
    assert walls.at(0.0, 0.5 + 0.025) == pytest.approx(0.50, abs=1e-9)  # m, row 10
    assert walls.at(0.0, 0.025) == 0.0                                    # on the wall


def test_wall_distance_counts_unknown_and_off_map_as_wall():
    data = np.zeros((20, 40), dtype=np.int16)
    data[0, :] = 100
    data[15, :] = -1  # unknown row
    walls = WallDistance(data.flatten().tolist(), 40, 20, 0.05, -1.0, 0.0)
    assert walls.at(0.0, 13 * 0.05 + 0.025) == pytest.approx(0.10, abs=1e-9)  # m, 2 rows from unknown
    assert walls.at(5.0, 0.5) == 0.0
    assert walls.at(math.nan, 0.5) == 0.0


def test_wall_distance_refuses_a_rotated_map():
    with pytest.raises(ValueError, match='origin yaw'):
        WallDistance(_grid(), 40, 20, 0.05, -1.0, 0.0, origin_yaw=0.1)
