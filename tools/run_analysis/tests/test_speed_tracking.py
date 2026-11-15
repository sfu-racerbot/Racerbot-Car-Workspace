"""speed_tracking.py's pure functions on synthetic traces with closed-form answers. Run by the tool-suite command in CLAUDE.md."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from run_analysis import speed_tracking as st

TAU = 0.3


def lag_trace(step_t=1.0, hold_for=2.0, before=2.0, after=1.0, t_end=4.0):
    """Command 2.0 -> 1.0 at ``step_t`` holding ``hold_for`` s, speed a first-order lag (tau 0.3 s) that was settled at 2.0."""
    cmd_t = np.arange(0.0, t_end, 0.025)
    cmd = np.where((cmd_t >= step_t) & (cmd_t < step_t + hold_for), after, before)
    v_t = np.arange(0.0, t_end, 0.002)
    v = np.where(v_t < step_t, before, after + (before - after) * np.exp(-(v_t - step_t) / TAU))
    return v_t, v, cmd_t, cmd


def test_step_down_time_is_the_first_order_lag_closed_form():
    # oracle: v - 1.0 = exp(-t/tau) reaches 0.3 at t = tau ln(1/0.3) = 0.3612 s
    v_t, v, cmd_t, cmd = lag_trace()
    events = st.step_down_events(v_t, v, cmd_t, cmd)
    assert len(events) == 1
    assert events[0]['time_to_reach'] == pytest.approx(TAU * math.log(1.0 / 0.3), abs=0.01)    # s
    assert events[0]['new'] == pytest.approx(1.0, abs=1e-9)


def test_step_down_not_reached_reports_the_remaining_error_at_0_4_s():
    # tau 1.5 s: at 0.4 s v - 1.0 = exp(-0.4/1.5) = 0.766 m/s, and the command holds only 0.5 s
    cmd_t = np.arange(0.0, 3.0, 0.025)
    cmd = np.where((cmd_t >= 1.0) & (cmd_t < 1.5), 1.0, 2.0)
    v_t = np.arange(0.0, 3.0, 0.002)
    v = np.where(v_t < 1.0, 2.0, 1.0 + np.exp(-(v_t - 1.0) / 1.5))
    events = st.step_down_events(v_t, v, cmd_t, cmd)
    assert len(events) == 1 and math.isnan(events[0]['time_to_reach'])
    assert events[0]['remaining'] == pytest.approx(math.exp(-0.4 / 1.5), abs=0.01)             # m/s


def test_a_drop_that_does_not_hold_for_0_4_s_is_not_a_step_down():
    # the command comes back up after 0.3 s: a glitch, not a step-down (window 0.4 s; a 0.2 s window would count it)
    cmd_t = np.arange(0.0, 3.0, 0.025)
    cmd = np.where((cmd_t >= 1.0) & (cmd_t < 1.3), 1.0, 2.0)
    v_t = np.arange(0.0, 3.0, 0.01)
    assert st.step_down_events(v_t, np.full_like(v_t, 2.0), cmd_t, cmd) == []


def test_overshoot_after_a_ramp_and_drop_has_the_designed_half_metre():
    # command 0 -> 2.0 over 1.0-1.5 s, back to 1.0 at 1.6 s; speed ramps to 1.5 and stays: v - cmd = 0.5 after the drop
    cmd_t = np.arange(0.0, 4.0, 0.025)
    cmd = np.where(cmd_t < 1.0, 0.0, np.where(cmd_t < 1.5, 4.0 * (cmd_t - 1.0), np.where(cmd_t < 1.6, 2.0, 1.0)))
    v_t = np.arange(0.0, 4.0, 0.005)
    v = np.clip((v_t - 1.0) * 3.0, 0.0, 1.5)
    ev = st.overshoot_events(v_t, v, cmd_t, cmd)
    assert len(ev) == 1
    assert ev[0]['t'] == pytest.approx(1.5, abs=0.03)
    assert ev[0]['max_over'] == pytest.approx(0.5, abs=0.01)                                   # m/s
    assert ev[0]['time_over'] == pytest.approx(0.9, abs=0.03)                                  # s, 1.6 s to the window end at 2.5 s


def test_braking_bins_report_the_constant_deceleration():
    # speed falls at exactly 2.0 m/s^2 from 3.0 to 1.5 while the command is 1.0: e from 2.0 down to 0.5
    v_t = np.arange(0.0, 0.75, 0.01)
    v = 3.0 - 2.0 * v_t
    out = st.braking_bins(v_t, v, np.array([0.0]), np.array([1.0]))
    assert out[(1.5, math.inf)]['median'] == pytest.approx(-2.0, abs=0.01)
    assert out[(1.0, 1.5)]['median'] == pytest.approx(-2.0, abs=0.01)
    assert out[(0.5, 1.0)]['n'] > 0 and out[(0.2, 0.5)]['n'] == 0


def test_a_trace_without_events_gives_n_0_and_nan_medians_and_never_raises():
    t = np.arange(0.0, 3.0, 0.025)
    s = st.summarise([(t, np.full_like(t, 1.2), t, np.full_like(t, 1.2))])
    assert s['step_down']['n'] == 0 and math.isnan(s['step_down']['time']['median'])
    assert s['overshoot']['n'] == 0 and math.isnan(s['overshoot']['max_over']['median'])
    for empty in (np.array([]),):
        st.summarise([(empty, empty, empty, empty)])
        assert st.step_down_events(empty, empty, empty, empty) == []
        assert st.overshoot_events(empty, empty, empty, empty) == []


def test_nan_samples_are_skipped():
    v_t, v, cmd_t, cmd = lag_trace()
    v = v.copy()
    v[::7] = math.nan
    cmd = cmd.copy()
    cmd[10] = math.nan
    events = st.step_down_events(v_t, v, cmd_t, cmd)
    assert len(events) == 1
    assert events[0]['time_to_reach'] == pytest.approx(TAU * math.log(1.0 / 0.3), abs=0.02)


def test_jitter_counts_only_fast_ticks():
    cmd_t = np.arange(0.0, 2.0, 0.025)
    cmd = np.where(np.arange(cmd_t.size) % 2 == 0, 2.0, 2.5)                                    # alternates by 0.5
    v_t = np.array([0.0, 2.0])
    fast = st.jitter(cmd_t, cmd, v_t, np.array([2.0, 2.0]))
    slow = st.jitter(cmd_t, cmd, v_t, np.array([1.0, 1.0]))
    assert fast['median'] == pytest.approx(0.5, abs=1e-9) and slow['n'] == 0


def test_corner_shed_is_the_closed_form_distance_law():
    # speed 2.0 m/s constant to a contact at t = 3.0: d(t) = 2 (3 - t). With a = 1 m/s^2 and v_c = 1: v_max = sqrt(1 + 2 d).
    # A 3.0 m/s command exceeds it from t = 1.0; the 2.0 m/s speed exceeds it from t = 2.25 (4 > 1 + 4 (3 - t)).
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import corner_shed
    t = np.arange(0.0, 3.0001, 0.005)
    table, info = corner_shed.shed_table(t, np.full_like(t, 2.0), t, np.full_like(t, 3.0), 3.0, 0.0, [1.0], [1.0])
    assert info['distance'] == pytest.approx(6.0, abs=1e-6)
    t_cmd, t_speed = table[(1.0, 1.0)]
    assert t_cmd == pytest.approx(1.0, abs=0.01) and t_speed == pytest.approx(2.25, abs=0.01)
