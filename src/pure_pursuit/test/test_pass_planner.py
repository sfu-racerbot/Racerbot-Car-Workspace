"""pass_planner: passing a stopped car (stage 2). ROS-free; run by
`python3 -m pytest src/pure_pursuit/test/` and colcon test.

Oracles: circle geometry and the stopping/reach kinematics are closed forms
recomputed here; the corridor cases use an analytic wall distance on a
synthetic stadium (two 30 m straights, 6 m bends), so "room" is known exactly.
"""
import math

import numpy as np
import pytest

from pure_pursuit import pass_planner as pp

LIMITS = pp.Limits()


def stadium(straight=30.0, radius=6.0, step=0.15):
    pts = [(i * step, 0.0) for i in range(int(straight / step))]
    m = int(math.pi * radius / step)
    pts += [(straight + radius * math.cos(-math.pi / 2 + math.pi * i / m),
             radius + radius * math.sin(-math.pi / 2 + math.pi * i / m)) for i in range(m)]
    pts += [(straight - i * step, 2 * radius) for i in range(int(straight / step))]
    pts += [(radius * math.cos(math.pi / 2 + math.pi * i / m),
             radius + radius * math.sin(math.pi / 2 + math.pi * i / m)) for i in range(m)]
    return np.array(pts)


XY = stadium()
LINE = pp.Line(XY, np.full(len(XY), 4.0))


def corridor(half_width):
    """Walls at half_width either side of the first straight (the line is y = 0
    for 0 <= x <= 30, where every case here stays): distance = half_width - |y|."""
    def walls(xs, ys):
        return half_width - np.abs(np.asarray(ys, float))
    return walls


STOPPED = pp.StalledOpponent(rear_arc=10.0, lateral=0.0, speed_bound=1.0, age=0.0)


def pose_behind(gap):
    """Rear axle on the first straight with the front `gap` metres behind s=10."""
    front = (LIMITS.wheelbase + LIMITS.length) / 2
    return (10.0 - gap - front, 0.0, 0.0)


# --- Frenet frame -----------------------------------------------------------

def test_offset_points_on_a_circle_sit_at_radius_minus_offset_and_project_back():
    # Oracle: a counter-clockwise circle's left normal points at the centre,
    # so the point at offset d is at radius R - d, and frenet() inverts it.
    radius = 5.0
    angles = np.linspace(0, 2 * math.pi, 400, endpoint=False)
    circle = pp.Line(np.column_stack((radius * np.cos(angles), radius * np.sin(angles))),
                     np.full(400, 3.0))
    for s, d in ((1.0, 0.0), (7.3, 0.6), (20.0, -0.8)):
        point = circle.point(np.array(s), np.array(d))
        assert np.linalg.norm(point) == pytest.approx(radius - d, abs=2e-3)  # 2 mm chord sag
        s_back, d_back = circle.frenet(*point)
        assert s_back == pytest.approx(s, abs=5e-3)
        assert d_back == pytest.approx(d, abs=5e-3)


def test_line_refuses_too_few_points_and_non_finite_points():
    with pytest.raises(ValueError, match='>= 4 points'):
        pp.Line(XY[:3], np.ones(3))
    bad = XY.copy()
    bad[5, 0] = math.nan
    with pytest.raises(ValueError, match='finite'):
        pp.Line(bad, np.ones(len(bad)))


def test_frenet_of_a_non_finite_point_is_nan():
    s, d = LINE.frenet(math.nan, 0.0)
    assert math.isnan(s) and math.isnan(d)


# --- Kinematics --------------------------------------------------------------

def test_braking_times_coast_for_the_response_then_brake_to_the_stop():
    v = 2.0
    coast = v * LIMITS.response                       # 0.90 m
    stop = coast + v * v / (2 * LIMITS.brake)         # 1.5667 m
    assert pp.stop_distance(v, LIMITS) == pytest.approx(stop, abs=1e-9)
    times = pp.braking_times(np.array([0.5, coast + 0.5, stop - 1e-9, stop + 0.01]), v, LIMITS)
    assert times[0] == pytest.approx(0.25, abs=1e-9)
    left = v * v - 2 * LIMITS.brake * 0.5
    assert times[1] == pytest.approx(LIMITS.response + (v - math.sqrt(left)) / LIMITS.brake, abs=1e-9)
    assert times[2] == pytest.approx(LIMITS.response + v / LIMITS.brake, abs=1e-4)
    assert times[3] == math.inf


def test_braking_times_from_standstill_reach_nothing_but_the_start():
    assert list(pp.braking_times(np.array([0.0, 0.1]), 0.0, LIMITS)) == [0.0, math.inf]


def test_opponent_reach_runs_for_the_reaction_time_then_brakes_to_a_stop():
    v = 1.0
    assert float(pp.reach_long(0.1, v, LIMITS)) == pytest.approx(0.1, abs=1e-12)
    saturated = v * LIMITS.opponent_react + v * v / (2 * LIMITS.opponent_brake)
    assert float(pp.reach_long(math.inf, v, LIMITS)) == pytest.approx(saturated, abs=1e-12)
    assert float(pp.reach_long(10.0, v, LIMITS)) == pytest.approx(saturated, abs=1e-12)


def test_unseen_time_moves_the_box_forward_at_the_speed_bound():
    seen = pp.opponent_box(STOPPED, 0.0, LIMITS)
    later = pp.opponent_box(pp.StalledOpponent(10.0, 0.0, 1.0, 0.8), 0.0, LIMITS)
    assert seen[0] == later[0] == pytest.approx(10.0 - LIMITS.opponent_long_uncertainty, abs=1e-12)
    assert later[1] - seen[1] == pytest.approx(1.0 * 0.8, abs=1e-12)
    assert later[2] - seen[2] == pytest.approx(math.sin(LIMITS.opponent_heading) * 0.8, abs=1e-12)


# --- Approved rejoin bound (tech lead, 2026-11-08) ---------------------------

def test_worst_case_lead_pushes_the_last_sighting_forward_at_the_speed_bound():
    opponent = pp.StalledOpponent(10.0, 0.0, 1.0, 0.5)
    front = 10.0 + LIMITS.opponent_length + 1.0 * 0.5
    assert pp.worst_case_lead(front + 1.25, opponent, LINE, LIMITS) == pytest.approx(1.25, abs=1e-9)


def test_no_rejoin_without_a_sighting_inside_max_unseen():
    stale = pp.StalledOpponent(10.0, 0.0, 1.0, LIMITS.max_unseen + 0.01)
    assert pp.worst_case_lead(30.0, stale, LINE, LIMITS) == -math.inf


# --- Contingency check -----------------------------------------------------------

def hold_plan(gap, half_width=1.5):
    pose = pose_behind(gap)
    return pp.plan_hold(LINE, pose, 4.0, corridor(half_width), LIMITS)


def test_a_stop_that_ends_short_of_the_opponent_box_is_clear_one_past_it_is_not():
    # On the line, straight at the stopped car. Free room ahead of the front
    # bumper before the box: gap - long_uncertainty - long_margin. Samples
    # are 0.1 m apart and the check runs one sample past the stop, so a stop
    # 0.3 m short of that room is clear and one 0.1 m past it is not.
    gap = 3.0
    plan = hold_plan(gap)
    room = gap - LIMITS.opponent_long_uncertainty - LIMITS.long_margin

    def speed_for(stop):
        r, b = LIMITS.response, LIMITS.brake
        return -b * r + math.sqrt((b * r) ** 2 + 2 * b * stop)
    assert pp.fallback_clear(plan, speed_for(room - 0.3), STOPPED, LINE, LIMITS)
    assert not pp.fallback_clear(plan, speed_for(room + 0.1), STOPPED, LINE, LIMITS)


def test_beside_the_box_the_body_needs_the_car_margin_on_top_of_both_half_widths():
    # Held still beside a stopped car (overlapping it along the line), the
    # box's half-width at its saturated reach sets the lateral threshold:
    # car half-width + box half-width + car_margin (closed form below).
    reach = float(pp.reach_long(math.inf, STOPPED.speed_bound, LIMITS))
    half = (LIMITS.opponent_width / 2 + LIMITS.opponent_lateral_uncertainty
            + math.sin(LIMITS.opponent_heading) * reach)
    threshold = LIMITS.width / 2 + half + LIMITS.car_margin
    beside = pp.StalledOpponent(rear_arc=7.0, lateral=0.0, speed_bound=1.0, age=0.0)

    def clear_at(offset):
        plan = pp.plan_hold(LINE, (7.0, offset, 0.0), 0.0, corridor(1.5), LIMITS)
        return pp.fallback_clear(plan, 0.0, beside, LINE, LIMITS)
    assert clear_at(threshold + 0.03)
    assert not clear_at(threshold - 0.06)


def test_a_car_closing_from_behind_is_checked_against_the_stopped_pose_forever():
    # Ego ahead, stopping from 0.5 m/s; a car behind with a 2 m/s bound. Its
    # reach while ego is still braking is reach(T_stop); afterwards it keeps
    # coming to reach(inf) (it reacts, then brakes). Place it so ego's
    # stopped rear is between the two: clear while moving, not once stopped.
    v, bound = 0.5, 2.0
    t_stop = LIMITS.response + v / LIMITS.brake
    during = float(pp.reach_long(t_stop, bound, LIMITS))
    final = float(pp.reach_long(math.inf, bound, LIMITS))
    assert final - during > 0.15
    s0 = 12.0
    plan = pp.plan_hold(LINE, (s0, 0.0, 0.0), v, corridor(1.5), LIMITS)
    last = float(plan.dist[plan.dist <= pp.stop_distance(v, LIMITS) + LIMITS.spacing][-1])
    ego_rear = s0 + last + (LIMITS.wheelbase - LIMITS.length) / 2
    front_now = ego_rear - LIMITS.long_margin - (during + final) / 2
    rear = front_now - LIMITS.opponent_length - LIMITS.opponent_long_uncertainty
    behind = pp.StalledOpponent(rear_arc=rear, lateral=0.0, speed_bound=bound, age=0.0)
    assert not pp.fallback_clear(plan, v, behind, LINE, LIMITS)
    further = pp.StalledOpponent(rear_arc=rear - (final - during), lateral=0.0, speed_bound=bound, age=0.0)
    assert pp.fallback_clear(plan, v, further, LINE, LIMITS)


def test_the_stop_must_also_clear_the_walls():
    plan = hold_plan(3.0, half_width=0.30)   # 0.30 - 0.155 body = 0.145 < margin + slack
    assert not pp.fallback_clear(plan, 0.5, None, LINE, LIMITS)
    assert pp.fallback_clear(hold_plan(3.0), 0.5, None, LINE, LIMITS)


def test_fallback_refuses_missing_plan_and_bad_speeds():
    plan = hold_plan(3.0)
    assert not pp.fallback_clear(None, 1.0, STOPPED, LINE, LIMITS)
    assert not pp.fallback_clear(plan, math.nan, STOPPED, LINE, LIMITS)
    assert not pp.fallback_clear(plan, -0.1, STOPPED, LINE, LIMITS)


def test_a_plan_shorter_than_the_stop_is_not_clear():
    plan = hold_plan(3.0)
    too_fast = math.sqrt(2 * LIMITS.brake * (plan.dist[-1] + 1.0))
    assert not pp.fallback_clear(plan, too_fast, None, LINE, LIMITS)


def test_safe_speed_is_none_when_even_the_current_speed_cannot_stop_clear():
    plan = hold_plan(0.6)
    assert pp.safe_speed(plan, STOPPED, LINE, LIMITS, 2.0) is None
    assert pp.safe_speed(plan, STOPPED, LINE, LIMITS, -1.0) is None


# --- Entry -----------------------------------------------------------------

def test_a_wide_corridor_admits_a_pass_beside_the_whole_reachable_box():
    plan, manoeuvre, cap, why = pp.entry(LINE, pose_behind(2.0), 0.0, STOPPED, corridor(1.5), LIMITS)
    assert plan is not None, why
    worst = pp.StalledOpponent(10.0, 0.0, 1.0, LIMITS.max_unseen)
    half = pp.opponent_box(worst, math.inf, LIMITS)[2]
    expected = half + LIMITS.width / 2 + LIMITS.car_margin + 0.05
    assert abs(plan.target) == pytest.approx(expected, abs=1e-9)
    assert np.all(plan.wall >= LIMITS.wall_margin)
    assert cap >= 0.3
    # The shift back is planned beyond the stopped car's front.
    assert manoeuvre.s_return > 10.0 + LIMITS.opponent_length


def test_a_corridor_too_narrow_for_two_cars_refuses_with_the_wall_reason():
    plan, _, _, why = pp.entry(LINE, pose_behind(2.0), 0.0, STOPPED, corridor(1.1), LIMITS)
    assert plan is None
    assert 'left: wall' in why and 'right: wall' in why


def test_a_car_whose_bound_the_pass_speed_cannot_outrun_is_not_passed():
    # new_pass needs pass_speed - speed_bound > 0.2 m/s, or the unseen bound
    # would never fall behind and the rejoin could never be authorised.
    # (A wider max_offset isolates this gate from the offset gate.)
    limits = pp.Limits(max_offset=2.0)
    slow = pp.StalledOpponent(10.0, 0.0, limits.pass_speed - 0.25, 0.0)
    fast = pp.StalledOpponent(10.0, 0.0, limits.pass_speed - 0.15, 0.0)
    assert pp.new_pass(LINE, pose_behind(2.0), 0.0, slow, limits, 1) is not None
    assert pp.new_pass(LINE, pose_behind(2.0), 0.0, fast, limits, 1) is None


def test_a_car_beside_or_behind_is_not_a_pass_target():
    behind = pp.StalledOpponent(4.0, 0.0, 1.0, 0.0)
    assert pp.new_pass(LINE, pose_behind(2.0), 0.0, behind, LIMITS, 1) is None


def test_a_pass_offset_beyond_max_offset_is_refused_on_that_side():
    off_left = pp.StalledOpponent(10.0, 0.30, 1.0, 0.0)
    assert pp.target_offset(off_left, 1, LIMITS) is None
    assert pp.target_offset(off_left, -1, LIMITS) is not None
