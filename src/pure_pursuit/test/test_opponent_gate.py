"""Oracles: Task 4 temporal/motion contract and closed-form motion."""

import re

import pytest

from pure_pursuit.opponent_gate import OpponentGate


def observe(gate, t, arc=3.0, ego=0.0, x=None):
    return gate.observe(t, arc if x is None else x, 0.0, arc, ego, 32.0)


@pytest.mark.parametrize('speed,start', [(1.0, 3.0), (0.0, 3.0), (1.0, 31.95)])
def test_closed_form_confirmation(speed, start):
    # Closed form: 6 intervals / 40 Hz = 0.15 s, including finish-line wrap.
    gate = OpponentGate()
    for tick in range(7):
        t = tick / 40
        arc = (start + speed * t) % 32
        verdict = observe(gate, t, arc, ego=(start - 3 + speed * t) % 32,
                          x=speed * t)
        assert verdict.confirmed == (tick == 6)


def test_recorded_backwards_phantom():
    # Recorded measurement: run 20261106-175300, ego 1.5, closing 6.9 m/s.
    gate = OpponentGate()
    speeds = []
    for tick in range(14):
        t = tick / 40
        verdict = observe(gate, t, 3.3 + (1.5 - 6.9) * t, 1.5 * t)
        assert not verdict.confirmed
        if 'backwards' in verdict.reason:
            speeds.append(float(re.search(r'at ([\d.]+) m/s', verdict.reason)[1]))
    assert speeds
    assert speeds == pytest.approx([5.4] * len(speeds), abs=0.2)  # m/s


def test_jump_restarts_count():
    gate = OpponentGate()
    for tick in range(7):
        observe(gate, tick / 40)
    # Contract: a 0.6 m map-frame jump identifies a different object.
    verdict = observe(gate, 0.175, x=3.6)
    assert not verdict.confirmed
    assert verdict.reason == 'seen 1 of 6 needed'


def test_expired_observations_drop_out():
    gate = OpponentGate()
    for tick in range(7):
        observe(gate, tick / 40)
    # Contract: silence exceeding the 0.3 s window loses confirmation.
    verdict = observe(gate, 0.451)
    assert not verdict.confirmed
    assert verdict.reason == 'seen 1 of 6 needed'


@pytest.mark.parametrize('name', ['confirm_count', 'window_sec', 'max_jump_m',
                                 'max_reverse_speed', 'max_speed'])
@pytest.mark.parametrize('value', [-1, float('nan'), float('inf')])
def test_invalid_constructor(name, value):
    # Contract: invalid parameters must name the rejected parameter.
    with pytest.raises(ValueError, match=name):
        OpponentGate(**{name: value})


@pytest.mark.parametrize('name', ['confirm_count', 'window_sec', 'max_jump_m', 'max_speed'])
def test_zero_constructor(name):
    with pytest.raises(ValueError, match=name):
        OpponentGate(**{name: 0})


@pytest.mark.parametrize('index', range(6))
@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_observation_clears_history(index, invalid):
    gate = OpponentGate()
    for tick in range(7):
        observe(gate, tick / 40)
    args = [0.175, 3.0, 0.0, 3.0, 0.0, 32.0]
    args[index] = invalid
    assert not gate.observe(*args).confirmed
    # Contract: invalid input clears history, rather than retaining a car.
    assert observe(gate, 0.2).reason == 'seen 1 of 6 needed'


def test_forward_speed_limit():
    gate = OpponentGate()
    observe(gate, 0)
    # Closed form: 0.225 m / 0.025 s = 9 m/s exceeds the 8 m/s ceiling.
    verdict = observe(gate, 0.025, 3.225)
    assert not verdict.confirmed
    assert 'too fast at 9.0 m/s' in verdict.reason
    assert observe(gate, 0.05, 3.225).reason == 'seen 1 of 6 needed'
