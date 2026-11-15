"""Static confirmation replayed on rb1 run 20261111-155542 at 31.3-34.8 s of the bag (pure, no rclpy).

Fixture static_gate_rb1_20261111-155542.npz (tools/run_analysis/scan_window_fixture.py --from 29.5 --to 33.0, the
fixture clock; intent_report's clock, in which this is episode 33.53 s of tools/run_analysis/results/
2026-11-13-static-confirmation-gate.md, is 2.099 s later): a real car (track 8 below, 3-5 m/s) and a static object
(track 9, 0.1-0.5 m/s) the tracker briefly mistook for a mover. Replayed cold-started through
tools/f1tenth_sim/gap_mirror.GapController, the node's own per-scan logic (held to the node by test_gap_mirror_parity).
Track ids are the replay's own.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'tools' / 'f1tenth_sim'))
import gap_mirror  # noqa: E402

FIXTURE = HERE / 'data' / 'static_gate_rb1_20261111-155542.npz'
REAL, STATIC = 8, 9


class _Lidar:
    def __init__(self, data):
        self.angle_min = float(data['angle_min'])
        self.angle_increment = float(data['angle_increment'])
        self.range_min = float(data['range_min'])


def replay(radial, tangential):
    data = np.load(FIXTURE)
    params = json.loads(str(data['params_json']))
    params['opponent_follow_static_radial_speed'] = radial
    params['opponent_follow_static_tangential_speed'] = tangential
    ctrl = gap_mirror.GapController(params, _Lidar(data))
    rows = []
    for n, scan in enumerate(data['ranges'].astype(np.float64)):
        yaw = float(data['yaw_rates'][n])
        steering, speed = ctrl.command(scan, float(data['odom_speeds'][n]), n, yaw_rate=None if math.isnan(yaw) else yaw)
        o = ctrl.opponents
        rows.append(dict(n=n, steering=steering, speed=speed, binding=ctrl.binding_cap,
                         follow_cap=ctrl.telemetry.get('follow_cap', math.inf), source=o.follow_cap_track,
                         law=o.follow_law_cap(False), static={t.track_id for t in o.tracker.tracks if t.static_confirmed},
                         moving={t.track_id for t in o.tracker.tracks if t.moving}))
    return rows


@pytest.fixture(scope='module')
def off():
    return replay(0.0, 0.0)


@pytest.fixture(scope='module')
def on():
    return replay(0.35, 1.5)


def test_the_replay_ran_the_whole_window_and_the_fixture_triggers_the_follow_cap(off):
    assert len(off) == 139
    assert sum(r['binding'] == 'follow' for r in off) >= 50                   # measured 59 with the gate off
    assert any(r['source'] == STATIC for r in off)                              # the static object does set the cap, gate off
    assert any(r['source'] == REAL for r in off)


def test_gate_off_never_confirms_and_leaves_the_follow_law_exactly_as_it_was(off):
    assert all(not r['static'] for r in off)
    # today's follow law over the raw moving + held bodies, recomputed here from the public pieces
    data = np.load(FIXTURE)
    params = json.loads(str(data['params_json']))
    params['opponent_follow_static_radial_speed'] = 0.0
    ctrl = gap_mirror.GapController(params, _Lidar(data))
    for n, scan in enumerate(data['ranges'].astype(np.float64)):
        yaw = float(data['yaw_rates'][n])
        ctrl.command(scan, float(data['odom_speeds'][n]), n, yaw_rate=None if math.isnan(yaw) else yaw)
        o = ctrl.opponents
        lateral = o.params.relevant_lateral
        today = min(o._follow_over(o.moving_body(), lateral, True), o._follow_over(o._held_body(), lateral, True))
        gated = min(o._follow_over(o.moving_body(True), lateral, True), o._follow_over(o._held_body(True), lateral, True))
        assert gated == today, n


def test_the_static_object_is_confirmed_while_tracked_and_never_adds_a_follow_cap_tick(off, on):
    first = next(r['n'] for r in on if STATIC in r['static'])
    assert first < 100                                                          # confirmed while still in view
    assert sum(STATIC in r['static'] for r in on) > 40                          # and stays so (measured 49 ticks)
    # while it is a live mover and confirmed it is never the source of the cap; the one tick it sets after confirmation
    # (97, as in the gate-off run) is its held body, which the stopped-car hold exempts
    assert not any(r['source'] == STATIC and STATIC in r['moving'] and STATIC in r['static'] for r in on)
    assert {r['n'] for r in on if r['source'] == STATIC} <= {r['n'] for r in off if r['source'] == STATIC}


def test_the_real_car_is_never_confirmed_and_its_follow_cap_ticks_are_unchanged(off, on):
    assert not any(REAL in r['static'] for r in on)
    real_off = [r['n'] for r in off if r['source'] == REAL]
    real_on = [r['n'] for r in on if r['source'] == REAL]
    assert real_on == real_off and len(real_on) >= 10


def test_residual_cost_once_the_real_car_has_left_the_gate_changes_nothing_while_the_hold_and_memory_run(off, on):
    # change-detector (not an oracle): ticks 116-135 of this fixture, after the real car left view (its last moving tick
    # is 115). The follow cap still binds on all of them with the gate on, as with it off: the real car's 1 s
    # FollowCapMemory (source 8) and the static object's held-body alongside ceiling (stopped-car hold, exempt by the
    # controller's ruling 2026-11-13) set it. If the hold is ever gated this is the test that changes.
    tail = range(116, 136)
    assert [off[n]['binding'] for n in tail] == ['follow'] * 20
    assert [on[n]['binding'] for n in tail] == ['follow'] * 20
    assert all(on[n]['law'] == math.inf for n in tail)                      # the follow LAW itself no longer follows the object


def test_a_legacy_yaml_without_the_new_parameters_replays_with_the_gate_off():
    assert gap_mirror.LEGACY_VALUES['opponent_follow_static_radial_speed'] == 0.0
    assert gap_mirror.LEGACY_VALUES['opponent_follow_static_tangential_speed'] == 0.0
