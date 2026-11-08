"""Approved measured-clearance contract; run by the pure_pursuit package suite."""
import math

import pytest

from pure_pursuit.pass_completion import CompletionGate, measured_body_lead


def test_body_lead_uses_whole_bodies_uncertainty_and_the_track_seam():
    # oracle: closed form. Ego rear is .11m behind its rear-axle frame;
    # the observed surface may be .58m behind the opponent's front.
    expected = 3 - .11 - .58 - .35
    assert measured_body_lead(5, 2, 40, .58, .36, .35) == pytest.approx(expected, abs=1e-12)  # m
    assert measured_body_lead(1, 38, 40, .58, .36, .35) == pytest.approx(expected, abs=1e-12)
    assert measured_body_lead(15, 12, 40, .58, .36, .35) == pytest.approx(expected, abs=1e-12)
    assert measured_body_lead(2, 5, 40, .58, .36, .35) < -3
    assert measured_body_lead(0, 0, 0, .58, .36, .35) < 0


@pytest.mark.parametrize('index,value', [(0, math.nan), (1, math.inf), (2, -1),
                                         (3, 0), (4, -1), (5, -1)])
def test_invalid_body_geometry_cannot_establish_a_clear_lead(index, value):
    args = [5, 2, 40, .58, .36, .35]
    args[index] = value
    # oracle: invariant; invalid geometry must never authorise a rejoin.
    assert measured_body_lead(*args) == -math.inf


def test_measured_clear_lead_must_arm_and_persist_above_the_margin():
    gate = CompletionGate(clear_margin=1, hold_sec=.5, hysteresis=.1)
    # oracle: approved contract, docs/racing-autonomy.md: .5s measured hold,
    # arm at 1.1m, sustain at >=1.0m, never count an earlier marginal lead.
    assert not gate.observe(0, 1.05, True)
    assert not gate.observe(1, 1.05, True)
    assert not gate.observe(2, 1.1, True)
    assert not gate.observe(2.499, 1.0, True)
    assert gate.observe(2.5, 1.0, True)


@pytest.mark.parametrize('reason', ['lost', 'below', 'nan', 'inf', 'clock_back', 'clock_nan', 'clock_inf'])
def test_loss_invalid_measurement_or_overlap_resets_the_full_hold(reason):
    gate = CompletionGate()
    assert not gate.observe(1, 2, True)
    now, lead, measured = 1.4, 2, True
    if reason == 'lost':
        measured = False
    elif reason == 'below':
        lead = .999
    elif reason == 'nan':
        lead = math.nan
    elif reason == 'inf':
        lead = math.inf
    elif reason == 'clock_nan':
        now = math.nan
    elif reason == 'clock_inf':
        now = math.inf
    else:
        now = .9
    # oracle: invariant; no prediction or old timer survives a denied sample.
    assert not gate.observe(now, lead, measured)
    assert not gate.observe(2, 2, True)
    assert not gate.observe(2.499, 2, True)
    assert gate.observe(2.5, 2, True)


@pytest.mark.parametrize('options', [dict(hold_sec=0), dict(clear_margin=-1),
                                    dict(hysteresis=-1), dict(hold_sec=math.nan)])
def test_invalid_completion_configuration_is_refused(options):
    # oracle: invariant; an invalid hold/margin cannot silently permit a pass.
    with pytest.raises(ValueError, match='completion margins'):
        CompletionGate(**options)
