"""Brake assist, pure (gap_logic.brake_assist_speed) and in the simulator mirror (gap_mirror.GapController). No rclpy.

Oracle: the definition in docs/gap-follow.md#brake-assist, written out: with the wheel speed more than ``threshold``
above the target, publish max(0, target - gain * (measured - target)); otherwise the target. Run by
`python3 -m pytest src/gap_follow/test/` (the gap_follow suite).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from gap_follow import gap_logic

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'tools' / 'f1tenth_sim'))


@pytest.mark.parametrize('target, measured, expected', [
    (2.0, 2.5, 1.5),         # excess 0.5: 2.0 - 1.0 * 0.5
    (2.0, 3.0, 1.0),         # excess 1.0
    (2.0, 2.15, 2.0),        # exactly at the threshold: unchanged
    (2.0, 2.16, 1.84),       # one step past it: 2.0 - 0.16
    (2.0, 1.0, 2.0),         # slower than the target: unchanged (never raises)
    (1.0, 5.0, 0.0),         # 1 - 4 < 0: clamped at zero, never a reverse command
    (0.0, 2.0, 0.0),         # a stop stays a stop
    (-0.3, 0.0, -0.3),       # a reverse command (the reverse escape) passes untouched
])
def test_closed_form(target, measured, expected):
    assert gap_logic.brake_assist_speed(target, measured, 0.15, 1.0) == pytest.approx(expected, abs=1e-12)


def test_gain_scales_the_correction_and_zero_gain_is_off():
    assert gap_logic.brake_assist_speed(2.0, 2.5, 0.15, 2.0) == pytest.approx(1.0, abs=1e-12)    # 2 - 2 * 0.5
    assert gap_logic.brake_assist_speed(2.0, 2.5, 0.15, 0.0) == 2.0


@pytest.mark.parametrize('measured', [math.nan, math.inf, -math.inf])
def test_an_unmeasurable_speed_gives_no_judgement(measured):
    assert gap_logic.brake_assist_speed(2.0, measured, 0.15, 1.0) == 2.0


@pytest.mark.parametrize('target', [math.nan, math.inf])
def test_a_non_finite_target_is_returned_unchanged(target):
    out = gap_logic.brake_assist_speed(target, 2.0, 0.15, 1.0)
    assert (math.isnan(out) and math.isnan(target)) or out == target


def test_it_never_raises_the_command_and_never_goes_negative():
    rng = np.random.default_rng(7)
    for target, measured in rng.uniform(0.0, 6.0, size=(2000, 2)):
        out = gap_logic.brake_assist_speed(float(target), float(measured), 0.15, 1.0)
        assert 0.0 <= out <= target + 1e-12


# --- the mirror (the simulator's copy of the node's logic) -----------------------------------------------------------
import gap_mirror  # noqa: E402


class _Lidar:
    angle_min = -math.radians(135.0)
    angle_increment = math.radians(270.0) / 1080
    range_min = 0.05


def _controller(enable, max_speed=1.0):
    params = gap_mirror.node_params(car_id='rb1')
    params.update(enable_brake_assist=enable, max_speed=max_speed, enable_launch_boost=False,
                  enable_opponent_awareness=False)
    return gap_mirror.GapController(params, _Lidar())


def _open_scan():
    return np.full(1081, 3.0)


def test_the_mirror_publishes_the_assisted_speed_and_keeps_the_planned_one_as_its_basis():
    on, off = _controller(True), _controller(False)
    for step, wheel in enumerate((0.0, 0.5, 1.0, 1.6, 1.6, 1.6)):
        s_on, v_on = on.command(_open_scan(), wheel, step)
        s_off, v_off = off.command(_open_scan(), wheel, step)
        assert s_on == s_off
        assert on.last_commanded_speed == off.last_commanded_speed == v_off       # the plan is untouched
        planned = v_off
        expected = gap_logic.brake_assist_speed(planned, wheel, 0.15, 1.0)
        assert v_on == pytest.approx(expected, abs=1e-12)
        assert on.telemetry['brake_assist'] == (expected < planned)
    assert v_on < v_off                                                            # the last ticks were 0.6 m/s over


def test_parameter_sets_from_before_the_assist_run_with_it_off():
    params = gap_mirror.node_params(car_id='rb1')
    for name in gap_mirror.BRAKE_ASSIST_PARAMS:
        params.pop(name)
    ctrl = gap_mirror.GapController(params, _Lidar())
    assert ctrl.p['enable_brake_assist'] is False


def test_exactly_at_the_threshold_is_unchanged_and_just_past_it_is_assisted():
    # Exactly representable values (0.25 = 2^-2), so the boundary is the comparison and not rounding.
    assert gap_logic.brake_assist_speed(2.0, 2.25, 0.25, 1.0) == 2.0
    assert gap_logic.brake_assist_speed(2.0, 2.25 + 2.0 ** -20, 0.25, 1.0) == pytest.approx(1.75, abs=1e-5)
