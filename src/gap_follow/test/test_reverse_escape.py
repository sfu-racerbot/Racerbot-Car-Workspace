"""ReverseEscape, rule by rule (pure, no rclpy). Rules are numbered as in the 2026-11-13 brief and
docs/gap-follow.md#reverse-escape. Every number is a closed form or a stated rule, not a recorded output."""
import math
import random

import pytest

from gap_follow.reverse_escape import ReverseEscape

DT = 0.025


def drive(esc, seconds, **kw):
    """Step ``seconds`` of ticks with fixed inputs; returns the list of results."""
    args = dict(deadman=True, blocked_ahead=True, measured_speed=0.0, rear_quarter_clear=True, recent_forward_m=1.0)
    args.update(kw)
    return [esc.step(dt=DT, **args) for _ in range(int(round(seconds / DT)))]


def to_reversing(esc, **kw):
    """Wait out wait_sec with the car stopped and blocked until the first negative command."""
    for _ in range(int(esc.wait_sec / DT) + 3):
        out = esc.step(dt=DT, deadman=True, blocked_ahead=True, measured_speed=0.0, rear_quarter_clear=True,
                       recent_forward_m=kw.get('recent_forward_m', 1.0))
        assert out is None or out <= 0.0, out                         # rule 7 on every tick, including the first
        if out is not None and out < 0.0:
            return out
    raise AssertionError('never started reversing')


# --- rule 1: idle -> waiting --------------------------------------------------------------------

def test_rule1_blocked_and_stopped_starts_waiting_and_commands_nothing():
    esc = ReverseEscape()
    assert esc.state == 'idle'
    assert esc.step(dt=DT, deadman=True, blocked_ahead=True, measured_speed=0.0, rear_quarter_clear=True,
                    recent_forward_m=1.0) is None
    assert esc.state == 'waiting'


@pytest.mark.parametrize('kw', [dict(blocked_ahead=False), dict(deadman=False), dict(measured_speed=None),
                                dict(measured_speed=float('nan')), dict(measured_speed=0.1), dict(measured_speed=-0.5)])
def test_rule1_every_other_input_stays_idle(kw):
    esc = ReverseEscape()
    drive(esc, 0.5, **kw)
    assert esc.state == 'idle'


def test_rule1_waiting_returns_to_idle_when_no_longer_blocked_or_released():
    for kw in (dict(blocked_ahead=False), dict(deadman=False)):
        esc = ReverseEscape()
        drive(esc, 0.5)
        assert esc.state == 'waiting'
        drive(esc, DT, **kw)
        assert esc.state == 'idle'


# --- rule 2: waiting -> reversing ---------------------------------------------------------------

def test_rule2_reverses_after_wait_sec_not_before():
    esc = ReverseEscape(wait_sec=1.0)
    out = drive(esc, 1.0 - 2 * DT)
    assert all(o is None for o in out)
    out = drive(esc, 4 * DT)
    assert any(o is not None and o < 0.0 for o in out)
    assert esc.state == 'reversing'


@pytest.mark.parametrize('kw', [dict(rear_quarter_clear=False), dict(recent_forward_m=0.19), dict(recent_forward_m=0.0)])
def test_rule2_needs_clear_rear_quarters_and_ground_just_driven(kw):
    esc = ReverseEscape()
    out = drive(esc, 3.0, **kw)
    assert all(o is None for o in out) and esc.state == 'waiting'


def test_rule2_the_forward_distance_boundary_is_inclusive():
    esc = ReverseEscape()
    out = drive(esc, 1.2, recent_forward_m=0.20)
    assert any(o is not None and o < 0.0 for o in out)


def test_rule2_target_is_the_smaller_of_max_distance_and_the_ground_driven():
    esc = ReverseEscape(max_distance=0.35)
    to_reversing(esc, recent_forward_m=0.22)
    assert esc.target == pytest.approx(0.22, abs=1e-12)          # m
    esc2 = ReverseEscape(max_distance=0.35)
    to_reversing(esc2, recent_forward_m=2.0)
    assert esc2.target == pytest.approx(0.35, abs=1e-12)


# --- rule 3: reversing --------------------------------------------------------------------------

def test_rule3_commands_minus_speed_and_ends_at_the_distance():
    esc = ReverseEscape(speed=0.30, max_distance=0.35)
    first = to_reversing(esc)
    assert first == pytest.approx(-0.30, abs=1e-12)               # m/s
    cmds = []
    for _ in range(200):
        cmds.append(esc.step(dt=DT, deadman=True, blocked_ahead=True, measured_speed=-0.30, rear_quarter_clear=True,
                             recent_forward_m=1.0))
        if esc.state == 'settling':
            break
    assert esc.state == 'settling' and cmds[-1] == 0.0
    # closed form: 0.35 m at 0.30 m/s is 1.1667 s = 46.67 ticks of 0.025 s; the tick that reaches it is the 47th.
    assert len(cmds) == 47


def test_rule3_remaining_distance_is_target_minus_the_integral_of_measured_speed():
    esc = ReverseEscape(max_distance=0.35)
    to_reversing(esc)
    for _ in range(40):                                           # 40 ticks x 0.025 s = 1.0 s at -0.30 m/s
        esc.step(dt=DT, deadman=True, blocked_ahead=True, measured_speed=-0.30, rear_quarter_clear=True, recent_forward_m=1.0)
    assert esc.remaining == pytest.approx(0.35 - 0.30, abs=1e-9)  # m


def test_rule3_wheels_not_turning_ends_at_max_time_having_moved_zero():
    esc = ReverseEscape(max_time=1.6)
    to_reversing(esc)
    n = 0
    while esc.state == 'reversing':
        esc.step(dt=DT, deadman=True, blocked_ahead=True, measured_speed=0.0, rear_quarter_clear=True, recent_forward_m=1.0)
        n += 1
        assert n < 200
    assert esc.remaining == pytest.approx(esc.target, abs=1e-12)  # nothing travelled
    assert n * DT == pytest.approx(1.6, abs=DT)                   # s, the time cap


def test_rule3_a_blocked_rear_quarter_ends_it_with_zero_that_tick():
    esc = ReverseEscape()
    to_reversing(esc)
    out = esc.step(dt=DT, deadman=True, blocked_ahead=True, measured_speed=-0.3, rear_quarter_clear=False, recent_forward_m=1.0)
    assert out == 0.0 and esc.state == 'settling'


@pytest.mark.parametrize('measured', [None, float('nan'), float('inf'), float('-inf')])
def test_rule3_stale_or_non_finite_speed_ends_it_immediately(measured):
    esc = ReverseEscape()
    to_reversing(esc)
    out = esc.step(dt=DT, deadman=True, blocked_ahead=True, measured_speed=measured, rear_quarter_clear=True, recent_forward_m=1.0)
    assert out == 0.0 and esc.state == 'settling'


# --- rule 4: deadman ----------------------------------------------------------------------------

@pytest.mark.parametrize('when', ['waiting', 'reversing', 'settling', 'cooldown'])
def test_rule4_deadman_released_at_any_time_returns_none_and_goes_idle(when):
    esc = ReverseEscape()
    to_reversing(esc)
    attempts = esc.attempts
    if when == 'waiting':
        esc = ReverseEscape()
        drive(esc, 0.5)
        attempts = 0
    elif when == 'settling':
        esc.step(dt=DT, deadman=True, blocked_ahead=True, measured_speed=-0.3, rear_quarter_clear=False, recent_forward_m=1.0)
    elif when == 'cooldown':
        drive(esc, 1.3, measured_speed=-0.3, rear_quarter_clear=False)      # reverse ends, 0.3 s settle, then cooldown
    assert esc.state == when
    out = esc.step(dt=DT, deadman=False, blocked_ahead=True, measured_speed=0.0, rear_quarter_clear=True, recent_forward_m=1.0)
    assert out is None and esc.state == 'idle'
    assert esc.attempts == attempts                                # the attempt still counts


# --- rule 5: settling and cooldown ----------------------------------------------------------------

def test_rule5_settling_is_zero_for_settle_sec_then_cooldown_returns_none():
    esc = ReverseEscape(settle_sec=0.3, cooldown_sec=4.0)
    to_reversing(esc)
    out = esc.step(dt=DT, deadman=True, blocked_ahead=True, measured_speed=-0.3, rear_quarter_clear=False, recent_forward_m=1.0)
    assert esc.state == 'settling'
    zeros = 0
    while esc.state == 'settling':
        r = esc.step(dt=DT, deadman=True, blocked_ahead=True, measured_speed=0.0, rear_quarter_clear=True, recent_forward_m=1.0)
        if esc.state == 'settling':
            assert r == 0.0
            zeros += 1
        assert zeros < 100
    assert zeros * DT == pytest.approx(0.3, abs=2 * DT)           # s
    assert esc.state == 'cooldown'
    assert all(o is None for o in drive(esc, 3.9))                # cooldown never commands
    assert esc.state == 'cooldown'
    drive(esc, 0.2)
    assert esc.state in ('idle', 'waiting')


def test_rule5_cooldown_blocks_a_new_reverse():
    esc = ReverseEscape(cooldown_sec=4.0, max_attempts=3)
    to_reversing(esc)
    esc.step(dt=DT, deadman=True, blocked_ahead=True, measured_speed=-0.3, rear_quarter_clear=False, recent_forward_m=1.0)
    out = drive(esc, 3.0, measured_speed=0.0)
    assert all(o is None or o == 0.0 for o in out) and not any(o is not None and o < 0.0 for o in out)


# --- rule 6: attempts ---------------------------------------------------------------------------

def run_one_attempt(esc):
    to_reversing(esc)
    esc.step(dt=DT, deadman=True, blocked_ahead=True, measured_speed=-0.3, rear_quarter_clear=False, recent_forward_m=1.0)
    drive(esc, 0.3 + 4.0 + 0.1, blocked_ahead=False)              # settle, cool down, return to idle


def test_rule6_at_most_max_attempts_then_exhausted_until_forward_progress():
    esc = ReverseEscape(max_attempts=2, rearm_progress=3.0)
    run_one_attempt(esc)
    run_one_attempt(esc)
    assert esc.attempts == 2
    out = drive(esc, 5.0)
    assert not any(o is not None and o < 0.0 for o in out)
    assert esc.state == 'exhausted'
    esc.note_forward(2.9)
    assert esc.state == 'exhausted'
    esc.note_forward(0.1)                                         # 3.0 m in all: re-armed
    assert esc.attempts == 0 and esc.state == 'idle'
    assert to_reversing(esc) < 0.0


def test_rule6_progress_before_the_cap_also_resets_the_count():
    esc = ReverseEscape(max_attempts=2, rearm_progress=3.0)
    run_one_attempt(esc)
    assert esc.attempts == 1
    esc.note_forward(3.0)
    assert esc.attempts == 0


# --- rule 7: bounds and refusals ------------------------------------------------------------------

def test_rule7_never_positive_never_above_speed_over_random_sequences():
    for seed in range(500):
        rng = random.Random(seed)
        speed = rng.choice([0.1, 0.3, 0.5])
        esc = ReverseEscape(speed=speed, wait_sec=0.5, settle_sec=0.1, cooldown_sec=0.2, max_attempts=3)
        for _ in range(400):
            out = esc.step(dt=rng.choice([0.0, DT, 0.05]), deadman=rng.random() > 0.05, blocked_ahead=rng.random() > 0.2,
                           measured_speed=rng.choice([None, float('nan'), 0.0, 0.0, -0.3, 0.2, 1.0]),
                           rear_quarter_clear=rng.random() > 0.1, recent_forward_m=rng.choice([0.0, 0.3, 1.0]))
            if rng.random() < 0.02:
                esc.note_forward(rng.random() * 4.0)
            assert out is None or (-speed - 1e-12 <= out <= 0.0), (seed, out)


@pytest.mark.parametrize('kw,name', [
    (dict(speed=0.0), 'speed'), (dict(speed=0.51), 'speed'), (dict(speed=float('nan')), 'speed'),
    (dict(max_distance=0.0), 'max_distance'), (dict(max_distance=0.61), 'max_distance'),
    (dict(wait_sec=0.49), 'wait_sec'), (dict(max_attempts=0), 'max_attempts'), (dict(max_attempts=4), 'max_attempts'),
    (dict(max_time=0.0), 'max_time'), (dict(settle_sec=-0.1), 'settle_sec'), (dict(cooldown_sec=-1.0), 'cooldown_sec'),
    (dict(rearm_progress=0.0), 'rearm_progress'), (dict(rear_quarter_clearance=-0.01), 'rear_quarter_clearance')])
def test_rule7_constructor_refuses_bad_arguments(kw, name):
    with pytest.raises(ValueError, match=name):
        ReverseEscape(**kw)


def test_rule7_the_boundary_values_are_accepted():
    ReverseEscape(speed=0.5, max_distance=0.6, wait_sec=0.5, max_attempts=1)
    ReverseEscape(speed=0.01, max_distance=0.01, max_attempts=3)
