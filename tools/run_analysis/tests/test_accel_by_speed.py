"""accel_by_speed: acceleration capability against speed from a command and the wheel speed.

Oracles: a synthetic first-order speed loop with a hard acceleration limit a(v) (6 m/s^2 below 2 m/s, falling
linearly to 2 at 5, flat beyond), driven by step commands, so the true capability at each speed is known in
closed form; invariants for what enters the tables. Run by: python3 -m pytest tools/run_analysis/tests/ -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_analysis import accel_by_speed as abs_  # noqa: E402

BINS = (0, 1, 2, 3, 3.5, 4, 4.5, 5, 6)


def a_true(v):
    return float(np.interp(v, [0.0, 2.0, 5.0], [6.0, 6.0, 2.0]))


def simulate(steps, dt=0.01, gain=10.0, brake_limit=4.0):
    """(t, wheel speed, command): a proportional loop whose acceleration is limited to a_true(v) and whose
    braking to ``brake_limit``; ``steps`` is a list of (command, seconds)."""
    t, v, cmd = [], [], []
    speed, clock = 0.0, 0.0
    for command, seconds in steps:
        for _ in range(int(seconds / dt)):
            a = min(gain * (command - speed), a_true(speed)) if command > speed else max(gain * (command - speed), -brake_limit)
            speed += a * dt
            clock += dt
            t.append(clock); v.append(speed); cmd.append(command)
    return np.array(t), np.array(v), np.array(cmd)


def ramps(repeats=12, top=6.0):
    steps = []
    for _ in range(repeats):
        steps += [(top, 2.5), (0.0, 2.5)]
    return simulate(steps)


def test_recovers_the_known_capability_in_every_bin():
    t, v, cmd = ramps()
    rows = abs_.accel_capability(t, v, cmd, bins=BINS)
    assert len(rows) == len(BINS) - 1
    for row in rows:
        centre = 0.5 * (row['low'] + row['high'])
        assert row['n'] >= 30, row
        # oracle: the hard limit a_true at the bin's speed; p75 sits in the saturated upper part
        assert row['capability'] == pytest.approx(a_true(centre), abs=0.3), row     # m/s^2
        assert row['capability'] == row['p75']


def test_samples_where_the_command_is_below_the_wheel_speed_never_enter_the_acceleration_table():
    t, v, cmd = simulate([(5.0, 3.0), (1.0, 3.0)])           # up, then braking down to 1
    up = abs_.accel_capability(t, v, cmd, bins=BINS)
    down = abs_.accel_capability(t, v, cmd, bins=BINS, braking=True)
    n_up = sum(r['n'] for r in up)
    asked = np.sum(cmd - v >= 0.5)
    assert n_up <= asked and n_up > 0
    assert sum(r['n'] for r in down) <= np.sum(v - cmd >= 0.5)
    # a pure braking trace (command always below speed) has nothing in the acceleration table
    t2, v2, cmd2 = simulate([(6.0, 2.0)])
    v2 = v2[200:]; t2 = t2[200:]; cmd2 = np.full_like(v2, 0.0)
    assert sum(r['n'] for r in abs_.accel_capability(t2, v2, cmd2, bins=BINS)) == 0


def test_braking_reports_deceleration_magnitudes_at_the_loops_limit():
    t, v, cmd = ramps()
    rows = abs_.accel_capability(t, v, cmd, bins=BINS, braking=True)
    populated = [r for r in rows if r['n'] >= 20]
    assert populated
    for row in populated:
        assert row['p90'] <= 4.0 + 0.3, row            # the synthetic brake limit, m/s^2
        assert row['median'] > 0.0


def test_empty_input_returns_empty_bins_with_nan_capability():
    rows = abs_.accel_capability([], [], [], bins=BINS)
    assert len(rows) == len(BINS) - 1
    for row in rows:
        assert row['n'] == 0 and math.isnan(row['capability']) and math.isnan(row['median'])


def test_nan_samples_are_dropped():
    t, v, cmd = ramps(repeats=4)
    clean = abs_.accel_capability(t, v, cmd, bins=BINS)
    v_bad, cmd_bad = v.copy(), cmd.copy()
    v_bad[::7] = np.nan
    cmd_bad[3::11] = np.nan
    rows = abs_.accel_capability(t, v_bad, cmd_bad, bins=BINS)
    assert 0.5 * sum(r["n"] for r in clean) < sum(r['n'] for r in rows) < sum(r['n'] for r in clean)   # dropped, not poisoned
    for row in rows:
        assert not math.isinf(row['median'])
        if row['n']:
            assert not math.isnan(row['median'])


def test_a_bin_with_fewer_than_twenty_samples_has_no_capability():
    t, v, cmd = simulate([(6.0, 0.8)])                 # one short run: a handful of samples per bin
    rows = abs_.accel_capability(t, v, cmd, bins=BINS)
    thin = [r for r in rows if 0 < r['n'] < 20]
    assert thin
    assert all(math.isnan(r['capability']) for r in thin)
    assert all(not math.isnan(r['median']) for r in thin)       # the other statistics are still reported
