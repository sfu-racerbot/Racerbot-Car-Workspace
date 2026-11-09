"""pass_planner: passing a stopped car (stage 2). ROS-free; run by
`python3 -m pytest src/pure_pursuit/test/` and colcon test.

Oracles: circle geometry and the stopping/reach kinematics are closed forms
recomputed here; the corridor cases use an analytic wall distance on a
synthetic stadium (two 30 m straights, 6 m bends), so "room" is known exactly.
"""
import math
from dataclasses import replace

import numpy as np
import pytest

from pure_pursuit import pass_planner as pp

LIMITS = pp.Limits()
PACE = 4.0   # the synthetic line's profile speed: nominal_speed() of every case here


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
    # ...and of a finite one, on the first straight (y = 0, heading +x), is exact.
    s, d = LINE.frenet(5.0, 0.2)
    assert (s, d) == (pytest.approx(5.0, abs=1e-9), pytest.approx(0.2, abs=1e-9))


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
    # At exactly max_unseen the bound still applies (closed form).
    edge = pp.StalledOpponent(10.0, 0.0, 1.0, LIMITS.max_unseen)
    front = 10.0 + LIMITS.opponent_length + 1.0 * LIMITS.max_unseen
    assert pp.worst_case_lead(30.0, edge, LINE, LIMITS) == pytest.approx(30.0 - front, abs=1e-9)


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
    # The stopped pose a car from behind closes on: the sample short of the stop.
    last = float(plan.dist[plan.dist <= pp.stop_distance(v, LIMITS)][-1])
    ego_rear = s0 + last + (LIMITS.wheelbase - LIMITS.length) / 2
    front_now = ego_rear - LIMITS.long_margin - (during + final) / 2
    rear = front_now - LIMITS.opponent_length - LIMITS.opponent_long_uncertainty
    behind = pp.StalledOpponent(rear_arc=rear, lateral=0.0, speed_bound=bound, age=0.0)
    assert not pp.fallback_clear(plan, v, behind, LINE, LIMITS)
    further = pp.StalledOpponent(rear_arc=rear - (final - during), lateral=0.0, speed_bound=bound, age=0.0)
    assert pp.fallback_clear(plan, v, further, LINE, LIMITS)


def test_the_stop_must_also_clear_the_walls():
    # The body's clearance must cover the margin AND the sampling slack
    # (closed form, body_points: hypot(L/16, W/8) between body points plus
    # half a step between poses; no turn on the straight). A wall half way
    # into the slack passes the margin alone and must still be refused.
    slack = math.hypot(LIMITS.length / 16, LIMITS.width / 8) + LIMITS.spacing / 2
    half_width = LIMITS.width / 2 + LIMITS.wall_margin + slack / 2
    plan = hold_plan(3.0, half_width=half_width)
    assert not pp.fallback_clear(plan, 0.5, None, LINE, LIMITS)
    assert pp.fallback_clear(hold_plan(3.0), 0.5, None, LINE, LIMITS)


def test_fallback_refuses_missing_plan_and_bad_speeds():
    plan = hold_plan(3.0)
    assert pp.fallback_clear(plan, 0.5, STOPPED, LINE, LIMITS)     # the same plan, a sane speed
    assert not pp.fallback_clear(None, 1.0, STOPPED, LINE, LIMITS)
    assert not pp.fallback_clear(plan, math.nan, STOPPED, LINE, LIMITS)
    assert not pp.fallback_clear(plan, -0.1, STOPPED, LINE, LIMITS)


def test_a_plan_shorter_than_the_stop_is_not_clear():
    plan = hold_plan(3.0)
    too_fast = math.sqrt(2 * LIMITS.brake * (plan.dist[-1] + 1.0))
    assert not pp.fallback_clear(plan, too_fast, None, LINE, LIMITS)
    # A speed whose stop (closed form) fits in the plan, walls clear, is.
    fits = 0.5
    assert pp.stop_distance(fits, LIMITS) + LIMITS.spacing < plan.dist[-1]
    assert pp.fallback_clear(plan, fits, None, LINE, LIMITS)


def test_safe_speed_is_none_when_even_the_current_speed_cannot_stop_clear():
    plan = hold_plan(0.6)
    assert pp.safe_speed(plan, STOPPED, LINE, LIMITS, 2.0) is None
    assert pp.safe_speed(plan, STOPPED, LINE, LIMITS, -1.0) is None
    # From standstill, with 3 m of room, there is a positive checked speed.
    assert pp.safe_speed(hold_plan(3.0), STOPPED, LINE, LIMITS, 0.0) > 0.0


# --- Entry -----------------------------------------------------------------

def test_a_wide_corridor_admits_a_pass_beside_the_whole_reachable_box():
    plan, manoeuvre, cap, why = pp.entry(LINE, pose_behind(2.0), 0.0, STOPPED, corridor(1.5), LIMITS, nominal=PACE)
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
    # Closed form: the lane needs half-width >= target + car half-width +
    # wall margin + body/step slack, about 0.85 m here; 0.80 m is too narrow.
    plan, _, _, why = pp.entry(LINE, pose_behind(2.0), 0.0, STOPPED, corridor(0.80), LIMITS, nominal=PACE)
    assert plan is None
    assert 'left: wall' in why and 'right: wall' in why


def test_a_car_whose_bound_the_pass_pace_cannot_outrun_is_not_passed():
    # new_pass needs nominal - speed_bound > 0.2 m/s, or the unseen bound
    # would never fall behind and the rejoin could never be authorised.
    # (A wider max_offset isolates this gate from the offset gate.)
    limits = pp.Limits(max_offset=2.0)
    slow = pp.StalledOpponent(10.0, 0.0, PACE - 0.25, 0.0)
    fast = pp.StalledOpponent(10.0, 0.0, PACE - 0.15, 0.0)
    assert pp.new_pass(LINE, pose_behind(2.0), 0.0, slow, limits, 1, PACE) is not None
    assert pp.new_pass(LINE, pose_behind(2.0), 0.0, fast, limits, 1, PACE) is None
    assert pp.new_pass(LINE, pose_behind(2.0), 0.0, slow, limits, 1, math.nan) is None


def test_the_pass_pace_is_the_lines_mean_profile_speed_ahead():
    speeds = np.where(np.arange(len(XY)) < 40, 2.0, 5.0)    # 2 m/s for the first 6 m
    line = pp.Line(XY, speeds)
    # Oracle: mean of 20 samples every 0.5 m from s = 0: twelve at 2 m/s
    # (0..5.5 m, before station 40 at 6.0 m) and eight at 5 m/s.
    assert pp.nominal_speed(line, 0.0) == pytest.approx((12 * 2.0 + 8 * 5.0) / 20, abs=0.08)


def test_a_car_beside_or_behind_is_not_a_pass_target():
    behind = pp.StalledOpponent(4.0, 0.0, 1.0, 0.0)
    assert pp.new_pass(LINE, pose_behind(2.0), 0.0, behind, LIMITS, 1, PACE) is None
    # The same car ahead of us is one.
    assert pp.new_pass(LINE, pose_behind(2.0), 0.0, STOPPED, LIMITS, 1, PACE) is not None


def test_a_pass_offset_beyond_max_offset_is_refused_on_that_side():
    off_left = pp.StalledOpponent(10.0, 0.40, 1.0, 0.0)
    assert pp.target_offset(off_left, 1, LIMITS) is None
    assert pp.target_offset(off_left, -1, LIMITS) is not None


def test_a_shift_to_the_right_curves_right_then_left_in_the_laws_signed_convention():
    # The tracking laws feed path curvature forward, signed + left / - right.
    # Oracle: a quintic shift towards -d on a straight has d'' < 0 over its
    # first half and > 0 over its second (d''(t) ~ 60t - 180t^2 + 120t^3).
    shift = pp.Manoeuvre(5.0, 0.0, -0.6, 3.0, math.inf, 3.0, -1)
    plan = pp.manoeuvre_plan(LINE, (5.0, 0.0, 0.0), 1.0, shift, corridor(1.5), LIMITS,
                             mode='pass', hold=True)
    curvature = plan.context.curvature
    first = curvature[(plan.dist > 0.3) & (plan.dist < 1.2)]
    second = curvature[(plan.dist > 1.8) & (plan.dist < 2.7)]
    assert first.size and second.size
    assert np.all(first < 0) and np.all(second > 0)


def test_a_pass_that_cannot_get_wide_before_reaching_the_car_is_refused_though_it_can_stop_now():
    # 1.0 m behind: a stop from standstill is trivially clear, but a 2.1 m
    # shift has the body only ~0.1 m across by the time its front reaches the
    # box (0.55 m of room), so crawling the pass would lose the checked stop.
    m = pp.new_pass(LINE, pose_behind(1.0), 0.0, STOPPED, LIMITS, -1, PACE)
    plan = pp.manoeuvre_plan(LINE, pose_behind(1.0), 0.0, m, corridor(1.5), LIMITS,
                             mode='pass', hold=True)
    assert pp.fallback_clear(plan, 0.0, STOPPED, LINE, LIMITS)
    assert not pp.progress_clear(plan, STOPPED, LINE, LIMITS)
    plan, _, _, why = pp.entry(LINE, pose_behind(1.0), 0.0, STOPPED, corridor(1.5), LIMITS, nominal=PACE)
    # Refused before the crawl check: the lane itself meets the full-reach box.
    assert plan is None and "meets the car's full-reach box" in why
    # From 2.0 m back the same pass can be crawled.
    m = pp.new_pass(LINE, pose_behind(2.0), 0.0, STOPPED, LIMITS, -1, PACE)
    plan = pp.manoeuvre_plan(LINE, pose_behind(2.0), 0.0, m, corridor(1.5), LIMITS,
                             mode='pass', hold=True)
    assert pp.progress_clear(plan, STOPPED, LINE, LIMITS)


def test_a_stop_that_starts_later_meets_a_box_that_has_grown_since():
    # Ego ahead at 2 m/s pulling away from a car (bound 1 m/s) whose reach,
    # starting now, never closes the 0.2 m gap. Starting 5 s from now its
    # reach has saturated (closed form) and covers ego's first sample.
    v, s0 = 2.0, 12.0
    plan = pp.plan_hold(LINE, (s0, 0.0, 0.0), v, corridor(1.5), LIMITS)
    ego_rear = s0 + (LIMITS.wheelbase - LIMITS.length) / 2
    front = ego_rear - LIMITS.long_margin - 0.2
    rear = front - LIMITS.opponent_length - LIMITS.opponent_long_uncertainty
    behind = pp.StalledOpponent(rear_arc=rear, lateral=0.0, speed_bound=1.0, age=0.0)
    assert float(pp.reach_long(math.inf, 1.0, LIMITS)) > 0.2
    assert pp.fallback_clear(plan, v, behind, LINE, LIMITS, t0=0.0)
    assert not pp.fallback_clear(plan, v, behind, LINE, LIMITS, t0=5.0)


def test_a_shift_from_standstill_peaks_at_80_percent_of_the_steering_curvature():
    # Spec (shift_length docstring): from standstill the steering term sizes
    # the shift so its quintic d'' peaks at 80% of tan(steering)/wheelbase.
    # Oracle: the path's curvature there is d''/(1 + d'^2)^1.5, with d'' and
    # d' of the quintic at t* = (3 - sqrt 3)/6 (closed forms below).
    kmax = 0.8 * math.tan(LIMITS.steering) / LIMITS.wheelbase   # spec: 80 % steering headroom
    delta, length = 0.6, pp.shift_length(0.6, 0.0, LIMITS)
    shift = pp.Manoeuvre(5.0, 0.0, -delta, length, math.inf, 3.0, -1)
    plan = pp.manoeuvre_plan(LINE, (5.0, 0.0, 0.0), 0.0, shift, corridor(1.5), LIMITS,
                             mode='pass', hold=True)
    t_star = (3 - math.sqrt(3)) / 6
    d2 = 0.8 * kmax
    assert 10 * math.sqrt(3) / 3 * delta / length ** 2 == pytest.approx(d2, rel=1e-9)
    d1 = 30 * t_star ** 2 * (1 - t_star) ** 2 * delta / length
    peak = float(np.max(np.abs(plan.context.curvature)))
    assert peak == pytest.approx(d2 / (1 + d1 ** 2) ** 1.5, rel=0.03)   # 3 %: 0.1 m sampling


def test_a_pass_lane_that_never_meets_the_full_reach_box_has_no_extra_speed_limit():
    m = pp.new_pass(LINE, pose_behind(2.0), 0.0, STOPPED, LIMITS, -1, PACE)
    plan = pp.manoeuvre_plan(LINE, pose_behind(2.0), 0.0, m, corridor(1.5), LIMITS, mode='pass', hold=True)
    assert pp.box_free_distance(plan, STOPPED, LINE, LIMITS) == math.inf
    assert pp.box_free_distance(plan, None, LINE, LIMITS) == math.inf
    assert pp.box_free_distance(None, STOPPED, LINE, LIMITS) == 0.0


def test_a_lane_that_meets_the_box_caps_speed_to_stop_short_of_it():
    # Held on the line straight at the stopped car: the box (saturated) starts
    # where the body's front plus long_margin reaches its lo, so the cap is
    # the delayed-stop speed for that distance less two samples (closed form).
    plan = hold_plan(3.0)
    free = pp.box_free_distance(plan, STOPPED, LINE, LIMITS)
    room = 3.0 - LIMITS.opponent_long_uncertainty - LIMITS.long_margin
    assert free == pytest.approx(room, abs=LIMITS.spacing + 1e-9)
    r, b = LIMITS.response, LIMITS.brake
    d = free - 2 * LIMITS.spacing
    expected = -b * r + math.sqrt((b * r) ** 2 + 2 * b * d)
    cap = pp.safe_speed(plan, STOPPED, LINE, LIMITS, 0.0)
    assert cap is not None and cap <= expected + 1e-9
    assert pp.stop_distance(cap, LIMITS) <= free - 2 * LIMITS.spacing + 1e-9
    assert cap >= expected - 0.25          # m/s: the 0.25 m/s grid below the bound
    # An entry along that lane is refused for meeting the box.
    walls = corridor(0.6)                   # only the line fits: no offset lane
    plan, _, _, why = pp.entry(LINE, pose_behind(2.0), 0.0, STOPPED, walls, LIMITS, nominal=PACE)
    assert plan is None


def test_the_stopped_pose_a_car_closes_on_is_the_one_short_of_the_stop():
    # Review 2026-11-08: ego on the line at s = 12 m stopping from 0.5 m/s
    # (0.2667 m); a car behind (rear 10.03 m, bound 2 m/s). Checked at the
    # sample past the stop (0.3 m) the saturated box misses; at the sample
    # short of it (0.2 m), the conservative pose for a car closing from
    # behind, it does not. Further back it clears both.
    plan = pp.plan_hold(LINE, (12.0, 0.0, 0.0), 0.5, corridor(1.5), LIMITS)
    behind = pp.StalledOpponent(rear_arc=10.03, lateral=0.0, speed_bound=2.0, age=0.0)
    assert not pp.fallback_clear(plan, 0.5, behind, LINE, LIMITS)
    further = pp.StalledOpponent(rear_arc=9.8, lateral=0.0, speed_bound=2.0, age=0.0)
    assert pp.fallback_clear(plan, 0.5, further, LINE, LIMITS)


def test_the_checks_start_from_where_the_car_is_not_where_the_manoeuvre_says():
    # Review 2026-11-08: holding a pass at d = -0.60 m but really at -0.80 m,
    # beside a wall 1.00 m off the line. The real body edge is 0.045 m from
    # the wall (1.00 - 0.80 - 0.155): the plan must start there and refuse.
    # Numbers derived for a 0.10 m wall margin, pinned here (the default
    # planning margin became 0.15 m on 2026-11-09; this test is about where
    # the checks start, not the margin's value).
    limits = replace(LIMITS, wall_margin=0.10)
    hold = pp.Manoeuvre(4.0, 0.0, -0.60, 2.0, math.inf, 3.0, -1)
    walls = corridor(1.0)
    real = pp.manoeuvre_plan(LINE, (8.0, -0.80, 0.0), 0.5, hold, walls, limits, mode='pass', hold=True)
    assert real.d[0] == pytest.approx(-0.80, abs=1e-3)
    assert not pp.fallback_clear(real, 0.5, None, LINE, limits)
    on_plan = pp.manoeuvre_plan(LINE, (8.0, -0.60, 0.0), 0.5, hold, walls, limits, mode='pass', hold=True)
    assert on_plan.d[0] == pytest.approx(-0.60, abs=1e-3)
    assert pp.fallback_clear(on_plan, 0.5, None, LINE, limits)


def test_a_pose_needing_more_than_the_steering_headroom_counts_as_a_wall():
    # Spec (Limits.steering_headroom): a pass path may need at most 80 % of
    # the steering limit. Oracle: a circle of radius R needs atan(L/R); pick R
    # either side of the headroom's radius and build a short plan on it.
    limit = 0.8 * LIMITS.steering       # spec value, not read back from Limits
    r_edge = LIMITS.wheelbase / math.tan(limit)

    def plan_on(radius):
        angles = np.linspace(0, 2 * math.pi, 600, endpoint=False)
        circle = pp.Line(np.column_stack((radius * np.cos(angles), radius * np.sin(angles))),
                         np.full(600, 2.0))
        open_floor = lambda xs, ys: np.full(len(xs), 5.0)
        return pp.build(circle, 0.0, np.zeros(20), 'hold', LIMITS, open_floor)
    assert np.all(np.isfinite(plan_on(r_edge * 1.1).wall[1:-1]))
    assert not np.any(np.isfinite(plan_on(r_edge * 0.9).wall[1:-1]))


def test_the_offset_blend_is_a_steerable_shift():
    # Spec: the measured offset blends into the manoeuvre over shift_length
    # (steerable by construction). Oracle (closed form): a 0.25 m error
    # blended over 1 m would peak at 0.25 * QUINTIC_PEAK = 1.44 1/m, above
    # the 80 % headroom's 0.8 tan(0.42) / 0.36 = 0.99 1/m; over
    # shift_length(0.25, 1.0) no sample may be marked unsteerable (-inf).
    hold = pp.Manoeuvre(4.0, 0.0, -0.60, 2.0, math.inf, 3.0, -1)
    plan = pp.manoeuvre_plan(LINE, (8.0, -0.85, 0.0), 0.0, hold, corridor(1.5), LIMITS, mode='pass', hold=True)
    assert 0.25 * pp.QUINTIC_PEAK > 0.8 * math.tan(0.42) / 0.36       # the 1 m blend would not steer
    assert np.all(np.isfinite(plan.wall))
    blend = pp.shift_length(0.25, 1.0, LIMITS)
    after = plan.dist >= blend + LIMITS.spacing
    assert np.allclose(plan.d[after], -0.60, atol=1e-6)             # m
    assert plan.d[0] == pytest.approx(-0.85, abs=1e-6)


def test_the_offset_blend_lengthens_with_speed():
    # At 3 m/s the speed term sets the blend: 3 * sqrt(QUINTIC_PEAK * 0.25 /
    # shift_accel) = 2.55 m (closed form, > min_shift 1.5 m). Halfway through,
    # the quintic has removed exactly half the 0.25 m error.
    blend = 3.0 * math.sqrt(pp.QUINTIC_PEAK * 0.25 / LIMITS.shift_accel)
    assert blend > LIMITS.min_shift + 0.5
    hold = pp.Manoeuvre(4.0, 0.0, -0.60, 2.0, math.inf, 3.0, -1)
    plan = pp.manoeuvre_plan(LINE, (8.0, -0.85, 0.0), 3.0, hold, corridor(1.5), LIMITS, mode='pass', hold=True)
    arc = plan.s - 8.0                    # the blend runs along the line's arc, not the path's
    mid = int(np.argmin(np.abs(arc - blend / 2)))
    assert plan.d[mid] == pytest.approx(-0.60 - 0.25 * (1 - pp.smooth(arc[mid] / blend)), abs=1e-6)
    assert plan.d[mid] == pytest.approx(-0.725, abs=0.01)          # m: half the error left
    assert np.allclose(plan.d[arc >= blend], -0.60, atol=1e-6)


def test_keeping_is_the_keep_going_tier():
    keep = pp.keeping(LIMITS)
    assert (keep.wall_margin, keep.car_margin, keep.long_margin) == (
        LIMITS.keep_wall_margin, LIMITS.keep_car_margin, LIMITS.keep_long_margin)
    # Spec (Limits comments, 2026-11-09): plan with 0.15 / 0.08 / 0.15 m, keep going to 0.05 / 0.04 / 0.08 m.
    assert (LIMITS.wall_margin, LIMITS.car_margin, LIMITS.long_margin) == (0.15, 0.08, 0.15)
    assert (keep.wall_margin, keep.car_margin, keep.long_margin) == (0.05, 0.04, 0.08)
    assert keep.brake == LIMITS.brake and keep.width == LIMITS.width   # nothing else changes


@pytest.mark.parametrize('clearance, expected', [
    (0.25, 'go'),        # above the planning margin: a positive checked speed
    (0.10, 'brake'),     # between the tiers: keep the manoeuvre, brake (cap 0)
    (0.03, 'abort'),     # under the keep margin: no checked speed at all
])
def test_safe_speed_plans_with_one_tier_and_gives_up_with_the_other(clearance, expected):
    # A hold on the line, on the straight: the body's wall clearance is the
    # corridor half-width minus half the car's width (closed form), the same
    # along the whole plan. build()'s sampling slack is added to the
    # half-width so `clearance` is what the plan sees.
    probe = pp.plan_hold(LINE, (8.0, 0.0, 0.0), 1.0, corridor(1.0), LIMITS)
    slack = 1.0 - LIMITS.width / 2 - float(np.min(probe.wall))      # sampling slack, m
    plan = pp.plan_hold(LINE, (8.0, 0.0, 0.0), 1.0, corridor(LIMITS.width / 2 + slack + clearance), LIMITS)
    assert float(np.min(plan.wall)) == pytest.approx(clearance, abs=1e-6)   # m
    cap = pp.safe_speed(plan, None, LINE, LIMITS, 1.0)
    if expected == 'go':
        assert cap is not None and cap >= 1.0
    elif expected == 'brake':
        assert cap == 0.0
    else:
        assert cap is None
