"""Tests for trace_deficit, trace_multi.first_stall, pure_course.plan_line, the scorecard's
detail view, and run_validation's new optional limits (whose defaults must not move).
Run by: python3 -m pytest tools/f1tenth_sim/tests/ tools/racerbot_sim/tests/ tools/run_analysis/tests/ tools/dev/tests/ -q
"""
import csv
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path[:0] = [os.path.join(HERE, '..'), os.path.join(HERE, '..', '..'),
                os.path.join(HERE, '..', '..', '..', 'src', 'pure_pursuit')]
import trace_deficit as td  # noqa: E402


def circle_trace(radius: float, speed: float, laps: float = 1.0, dt: float = 0.025) -> dict:
    t = np.arange(0.0, laps * 2 * math.pi * radius / speed, dt)
    a = speed * t / radius
    return dict(t=t, x=radius * np.cos(a), y=radius * np.sin(a), v=np.full(t.size, speed),
                steering=np.full(t.size, math.atan(0.36 / radius)), binding=["curve"] * t.size)


def test_deficit_on_a_circle_is_the_closed_form():
    # Oracle: on a circle of radius R the ideal speed is sqrt(a_lat R); driven at v the loss
    # is length / v - length / sqrt(a_lat R). Ends excluded: the profile starts from rest-free
    # open ends, so compare the middle sections.
    trace = circle_trace(2.0, 1.5)
    d = td.deficit(trace, lateral=2.0, braking=50.0, accel=50.0, top_speed=10.0)
    ideal = math.sqrt(2.0 * 2.0)
    middle = d["sections"][1:-1]
    assert np.median([r["ideal"] for r in middle]) == pytest.approx(ideal, abs=0.03)       # m/s
    assert np.median([r["lost"] for r in middle]) == pytest.approx(2.5 / 1.5 - 2.5 / ideal, abs=0.02)   # s
    assert all(r["binding"] == {"curve": max(r["binding"].values())} for r in middle)


def test_deficit_a_car_faster_than_ideal_gains_time():
    d = td.deficit(circle_trace(2.0, 2.5), lateral=2.0, braking=50.0, accel=50.0, top_speed=10.0)
    assert np.median([r["lost"] for r in d["sections"][1:-1]]) < 0.0


def test_read_trace_prefers_true_speed_and_keeps_blanks_as_nan(tmp_path):
    path = tmp_path / "t.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "x", "y", "speed_meas", "speed_true", "steering", "binding"])
        w.writeheader()
        w.writerow(dict(t=0, x=0, y=0, speed_meas=3.1, speed_true=3.0, steering="", binding="curve"))
    out = td.read_trace(path)
    assert out["v"][0] == 3.0 and math.isnan(out["steering"][0]) and out["binding"] == ["curve"]


def test_first_stall_finds_the_first_long_enough_run():
    import trace_multi
    speeds = [1.0] * 10 + [0.0] * 20 + [1.0] * 5 + [0.0] * 45
    assert trace_multi.first_stall(speeds, 0.025, seconds=1.0) == 35
    assert trace_multi.first_stall([0.0] * 39, 0.025, seconds=1.0) is None


def test_plan_line_respects_the_lateral_limit_and_top_speed():
    import pure_course
    s = np.linspace(0.0, 2 * math.pi, 400, endpoint=False)
    xy = np.column_stack([3.0 * np.cos(s), 3.0 * np.sin(s)])
    plan = pure_course.plan_line(xy, top_speed=4.0, lateral=2.7)
    # Oracle: v^2 / R <= a_lat on a 3 m circle, so v <= sqrt(8.1) = 2.85 < 4.0.
    assert plan.speed.max() == pytest.approx(math.sqrt(2.7 * 3.0), abs=0.05)
    assert pure_course.plan_line(xy, 2.0, 2.7).speed.max() == pytest.approx(2.0, abs=1e-9)


def test_command_shaper_defaults_are_the_pure_pursuit_yaml():
    import run_validation as rv
    shaper = rv.CommandShaper()
    assert shaper.max_speed == rv.PURE_MAX_SPEED and shaper.max_lateral_accel == rv.PURE_MAX_LATERAL_ACCEL


def test_command_shaper_obeys_a_lower_speed_limit():
    import run_validation as rv
    shaper = rv.CommandShaper(max_speed=1.0)
    speed = 0.0
    for _ in range(400):
        _steer, speed = shaper.command(0.0, 9.0, measured_speed=speed)
    assert speed == pytest.approx(1.0, abs=1e-9)


def test_command_shaper_can_go_above_the_yaml_when_asked():
    # pure_course.py --speed 5.0 plans above pure_pursuit.yaml's max_speed; the shaper must
    # then let the car reach it, not quietly hold it at the YAML's value.
    import run_validation as rv
    raised = rv.PURE_MAX_SPEED + 1.0
    shaper = rv.CommandShaper(max_speed=raised)
    speed = 0.0
    for _ in range(400):
        _steer, speed = shaper.command(0.0, 9.0, measured_speed=speed)
    assert speed == pytest.approx(raised, abs=1e-9)


def test_detail_lists_every_section_and_the_grip_bins():
    import validate_calibration as vc
    cards = {"real": dict(sectors=[1.0, 2.0], sector_cmd=[1.1, 2.1], sector_steer=[0.1, 0.2], grip=[(1, 1.5, 9, 1.0)]),
             "sim": dict(sectors=[1.5], sector_cmd=[1.6], sector_steer=[0.15], grip=[])}
    text = vc.detail(cards).splitlines()
    assert len(text) == 1 + 2 + 2
    assert "2.00  2.10 0.200" in text[2] and "nan" in text[2]
    assert text[3] == "grip real: 1:1.00"


def test_command_shaper_ramps_a_falling_target_down_from_the_last_command():
    # Mirrors pure_pursuit_node._shape_normal_command (2026-11-07): the measured-speed basis
    # never rises above this tick's target, so braking is slewed from the last command and
    # not held one step below the car's speed. Closed form: one step is
    # max_braking_decel * CONTROL_DT from the 3.2 m/s last command.
    import run_validation as rv
    shaper = rv.CommandShaper(car_id='rb2', max_speed=5.0)
    step = float(shaper.params['max_braking_decel']) * shaper.control_dt
    shaper.previous_speed = 3.2
    _steer, speed = shaper.command(0.0, 2.0, measured_speed=4.0)
    assert speed == pytest.approx(3.2 - step, abs=1e-9)  # m/s; ramping from 4.0 gives 4.0 - step
    # a target between the last command and the measured speed is commanded as is
    shaper.previous_speed = 3.2
    _steer, speed = shaper.command(0.0, 3.5, measured_speed=4.0)
    assert speed == pytest.approx(3.5, abs=1e-9)  # m/s
