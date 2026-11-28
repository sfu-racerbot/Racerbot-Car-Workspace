"""The explorer's decisions (car_explorer/explorer.py), no ROS.

The refusals and back-offs are the half that matters (TEST_QUALITY_STANDARDS
A8): a raise must never happen while LB is released, a limit sign must always
lower the setting, and nothing may leave gap_follow's own ranges.
Run: python3 -m pytest src/car_explorer/test/ -q
"""

import math

import pytest

from car_explorer.explorer import (
    GAP_FOLLOW_MAX_LATERAL, GAP_FOLLOW_MAX_SPEED, LATERAL, MIN_SPEED, SPEED, ConfigError,
    Conductor, CourseMap, Fence, Inputs, Ladder, Rules, path_stop_margin, stopping_distance,
    straight_room)

TICK = 0.05  # s, the node's 20 Hz


def drive(c, seconds, t0=0.0, **kw):
    """Feed ``seconds`` of identical inputs; return every setting applied."""
    defaults = dict(lb_held=True, speed=2.0, steering=0.0, stop_margin=5.0,
                    voltage=12.0, safety_state='gap_follow')
    defaults.update(kw)
    if 'yaw_rate' not in kw:  # turning exactly as the steering asks: no slide
        v, s = defaults['speed'], defaults['steering']
        defaults['yaw_rate'] = (None if v is None or s is None
                                else v * math.tan(s) / Rules().wheelbase)
    applied = []
    n = int(round(seconds / TICK))
    for i in range(n):
        out = c.update(Inputs(now=t0 + i * TICK, **defaults))
        if out is not None:
            applied.append(out)
    return applied, t0 + n * TICK


# --- the ladder -------------------------------------------------------------------------

def test_ladder_starts_at_the_fence_start_and_alternates_axes():
    lad = Ladder(Fence())
    assert lad.setting() == {SPEED: 2.0, LATERAL: 1.5}
    assert lad.raise_one() == {SPEED: 2.5, LATERAL: 1.5}
    assert lad.raise_one() == {SPEED: 2.5, LATERAL: 1.75}
    assert lad.raise_one() == {SPEED: 3.0, LATERAL: 1.75}


def test_ladder_has_no_cap_below_gap_follows_own_maxima():
    lad = Ladder(Fence())
    seen = [lad.setting()]
    while (s := lad.raise_one()) is not None:
        seen.append(s)
    assert seen[-1] == {SPEED: GAP_FOLLOW_MAX_SPEED[1], LATERAL: GAP_FOLLOW_MAX_LATERAL[1]}
    assert max(s[SPEED] for s in seen) == 5.0 and max(s[LATERAL] for s in seen) == 4.0
    assert lad.finished


def test_back_off_lowers_the_last_raised_axis_two_notches_and_freezes_it():
    lad = Ladder(Fence())
    for _ in range(5):
        lad.raise_one()          # speed 3.5, lateral 2.0, last raised speed
    assert lad.setting() == {SPEED: 3.5, LATERAL: 2.0}
    assert lad.back_off(None) == {SPEED: 2.5, LATERAL: 2.0}
    assert lad.limits == {SPEED: 3.5}
    for _ in range(20):
        lad.raise_one()
    assert lad.setting()[SPEED] == 2.5            # frozen: never raised again


def test_back_off_before_any_raise_freezes_both_at_the_start():
    lad = Ladder(Fence())
    assert lad.back_off(None) == {SPEED: 2.0, LATERAL: 1.5}
    assert lad.finished and lad.raise_one() is None


@pytest.mark.parametrize('kw', [
    dict(speed_top=5.1), dict(lateral_top=4.5), dict(speed_start=0.2),
    dict(speed_step=0.0), dict(lateral_step=-0.25), dict(speed_start=3.0, speed_top=2.5),
    dict(speed_step=math.nan), dict(lateral_top=math.inf), dict(back_off_steps=0)])
def test_fence_outside_gap_follows_range_or_malformed_is_refused(kw):
    with pytest.raises(ConfigError):
        Fence(**kw)


# --- the wall check ---------------------------------------------------------------------

def test_stopping_distance_is_reaction_plus_braking():
    # 4 m/s, 0.24 s before the brakes bite, then 3 m/s^2: 0.96 + 16/6 m.
    assert stopping_distance(4.0, 3.0, 0.24) == pytest.approx(0.96 + 16 / 6, abs=1e-9)  # m
    assert stopping_distance(0.0, 3.0, 0.24) == 0.0
    assert stopping_distance(-1.0, 3.0, 0.24) == 0.0


GEOM = dict(wheelbase=0.325, half_width=0.15, lidar_x=0.265, front_reach=0.41,
            delay=0.24)


def beams_ahead(distance, n=1081):
    """A UST-10LX scan with only the beam straight ahead hitting at ``distance``."""
    ranges = [math.inf] * n
    ranges[n // 2] = distance
    return ranges, -math.radians(135), math.radians(270) / (n - 1)


def test_wall_straight_ahead_margin_is_distance_minus_body_minus_reaction():
    ranges, a0, da = beams_ahead(3.0)
    m = path_stop_margin(ranges, a0, da, speed=2.0, steering=0.0, **GEOM)
    # hit at 0.265 + 3.0 from the rear axle, body reaches 0.41, 2 m/s x 0.24 s.
    assert m == pytest.approx(0.265 + 3.0 - 0.41 - 0.48, abs=1e-6)  # m


def test_a_faster_car_has_less_margin_to_the_same_wall():
    ranges, a0, da = beams_ahead(1.0)
    slow = path_stop_margin(ranges, a0, da, speed=1.0, steering=0.0, **GEOM)
    fast = path_stop_margin(ranges, a0, da, speed=4.0, steering=0.0, **GEOM)
    # 0.855 m of room: 0.24 m used at 1 m/s, 0.96 m at 4 m/s.
    assert slow == pytest.approx(0.855 - 0.24, abs=1e-6) and fast == pytest.approx(0.855 - 0.96, abs=1e-6)


def test_a_wall_off_to_the_side_is_not_on_a_straight_path():
    n = 1081
    ranges = [math.inf] * n
    a0, da = -math.radians(135), math.radians(270) / (n - 1)
    left = int(round((math.radians(90) - a0) / da))   # beam at +90 degrees
    ranges[left] = 0.30                                # wall 0.30 m to the left
    assert path_stop_margin(ranges, a0, da, speed=3.0, steering=0.0, **GEOM) == math.inf


def test_turning_toward_a_side_wall_puts_it_on_the_path():
    n = 1081
    a0, da = -math.radians(135), math.radians(270) / (n - 1)
    ranges = [math.inf] * n
    i = int(round((math.radians(45) - a0) / da))       # 45 deg left, 1.2 m out
    ranges[i] = 1.2
    straight = path_stop_margin(ranges, a0, da, speed=2.0, steering=0.0, **GEOM)
    left = path_stop_margin(ranges, a0, da, speed=2.0, steering=0.3, **GEOM)
    right = path_stop_margin(ranges, a0, da, speed=2.0, steering=-0.3, **GEOM)
    assert straight == math.inf and right == math.inf and left < 1.0


def test_straight_room_is_the_distance_ahead_of_the_nose_with_no_reaction_taken_off():
    ranges, a0, da = beams_ahead(3.0)
    g = {k: GEOM[k] for k in ('half_width', 'lidar_x', 'front_reach')}
    # Closed form: hit at lidar_x + 3.0 from the rear axle, nose at front_reach.
    assert straight_room(ranges, a0, da, **g) == pytest.approx(0.265 + 3.0 - 0.41, abs=1e-6)  # m
    side = [math.inf] * len(ranges)
    side[int(round((math.radians(60) - a0) / da))] = 0.5   # off to the side, not ahead
    assert straight_room(side, a0, da, **g) == math.inf


def test_bad_beams_are_ignored_and_an_empty_scan_is_clear():
    ranges, a0, da = beams_ahead(math.nan)
    ranges[0] = 0.0
    ranges[1] = -1.0
    assert path_stop_margin(ranges, a0, da, speed=3.0, steering=0.0, **GEOM) == math.inf
    assert path_stop_margin([], a0, da, speed=3.0, steering=0.0, **GEOM) == math.inf


def test_wall_check_refuses_nonsense_geometry():
    ranges, a0, da = beams_ahead(2.0)
    with pytest.raises(ConfigError):
        path_stop_margin(ranges, a0, da, speed=2.0, steering=0.0, **{**GEOM, 'delay': -0.1})
    with pytest.raises(ConfigError):
        path_stop_margin(ranges, a0, da, speed=math.nan, steering=0.0, **GEOM)


# --- the conductor ----------------------------------------------------------------------

NO_NEEDS = dict(need_grip_sec=0.0, need_braking_sec=0.0, need_steering_sec=0.0)


def make(baseline=None, **rules):
    """A conductor with short segments. With no baseline there are no stop
    tests (gap_follow's own min_speed is unknown), and with NO_NEEDS the
    information ledger never holds a session open: the ladder alone, as in
    the tests written before either existed."""
    c = Conductor(Fence(), Rules(**{'segment_sec': 2.0, **NO_NEEDS, **rules}), baseline)
    first = c.start()
    assert {k: first[k] for k in (SPEED, LATERAL)} == {SPEED: 2.0, LATERAL: 1.5}
    return c


def test_update_before_start_is_refused():
    with pytest.raises(RuntimeError, match='start'):
        Conductor().update(Inputs(now=0.0, lb_held=True))


def test_clean_driving_raises_one_notch_per_segment_on_a_straight():
    c = make()
    applied, _ = drive(c, 2.0 * 3 + 0.1)
    assert applied == [{SPEED: 2.5, LATERAL: 1.5}, {SPEED: 2.5, LATERAL: 1.75},
                       {SPEED: 3.0, LATERAL: 1.75}]


def test_no_raise_while_lb_is_released():
    c = make()
    applied, _ = drive(c, 30.0, lb_held=False)
    assert applied == [] and c.applied == {SPEED: 2.0, LATERAL: 1.5}
    assert c.driving_sec == 0.0


def test_no_raise_while_stopped_or_speed_unknown():
    c = make()
    assert drive(c, 30.0, speed=0.0)[0] == []
    assert drive(c, 30.0, t0=40.0, speed=None)[0] == []
    assert c.applied == {SPEED: 2.0, LATERAL: 1.5}


def test_a_raise_waits_for_a_straight():
    c = make()
    applied, t = drive(c, 5.0, steering=0.2)          # cornering the whole time
    assert applied == [] and c.pending == {SPEED: 2.5, LATERAL: 1.5}
    applied, _ = drive(c, 0.1, t0=t, steering=0.0)
    assert applied == [{SPEED: 2.5, LATERAL: 1.5}]


def test_gap_follow_safety_stop_backs_off_at_once_even_mid_corner():
    c = make()
    _, t = drive(c, 4.05)                              # speed 2.5, lateral 1.75
    applied, _ = drive(c, TICK, t0=t, steering=0.3, safety_state='ttc_brake')
    assert applied == [{SPEED: 2.5, LATERAL: 1.5}]     # lateral (last raised) down 2, floor 0
    assert c.coverage.events[0]['kind'] == 'gap_follow_safety'
    assert c.ladder.lateral_frozen


@pytest.mark.parametrize('state', ['ttc_escape', 'emergency_clearance'])
def test_every_gap_follow_safety_state_counts(state):
    c = make()
    applied, _ = drive(c, TICK, safety_state=state)
    assert applied and c.coverage.events[0]['detail'] == state


def test_wall_check_backs_off_below_the_margin_and_not_above_it():
    c = make(wall_margin=0.03)
    assert drive(c, 1.0, stop_margin=0.031)[0] == []
    applied, _ = drive(c, TICK, t0=1.0, stop_margin=0.029)
    assert applied and c.coverage.events[0]['kind'] == 'wall'


def test_limit_signs_are_ignored_when_lb_is_released():
    c = make()
    assert drive(c, 2.0, lb_held=False, stop_margin=-1.0, safety_state='ttc_brake')[0] == []
    assert c.coverage.events == []


def sliding(speed=3.0, steering=0.2, share=0.5, wheelbase=0.325):
    return dict(speed=speed, steering=steering,
                yaw_rate=share * speed * math.tan(steering) / wheelbase)


def test_a_slide_lasting_the_hold_time_freezes_cornering():
    c = make(slide_hold_sec=0.2, slide_steer_window=0.3)
    applied, _ = drive(c, 0.60, **sliding(share=0.5))     # 0.3 s to trust the steering, then 0.2 s
    assert applied == [{SPEED: 2.0, LATERAL: 1.5}]
    assert c.coverage.events[0]['kind'] == 'slide'
    assert c.ladder.lateral_frozen and not c.ladder.speed_frozen


def test_a_short_slide_or_turning_as_asked_is_not_a_slide():
    c = make(slide_hold_sec=0.2, slide_steer_window=0.3)
    _, t = drive(c, 0.4, **sliding(share=0.9))
    _, t = drive(c, 0.15, t0=t, **sliding(share=0.5))
    assert drive(c, 1.0, t0=t, **sliding(share=0.9))[0] == []
    assert c.coverage.events == []


def test_turning_the_wrong_way_is_a_slide():
    c = make(slide_hold_sec=0.1, slide_steer_window=0.3)
    applied, _ = drive(c, 0.5, **sliding(share=-0.5))
    assert applied and c.coverage.events[0]['kind'] == 'slide'


def test_a_fresh_steering_swing_is_not_a_slide_yet():
    # The servo lags the command: right after the command flips, the car is
    # still turning the old way. That must not count.
    c = make(slide_hold_sec=0.1, slide_steer_window=0.3)
    _, t = drive(c, 0.5, **sliding(steering=-0.2, share=1.0))
    swing = dict(speed=3.0, steering=0.2, yaw_rate=3.0 * math.tan(-0.2) / 0.325)
    assert drive(c, 0.25, t0=t, **swing)[0] == []
    assert c.coverage.events == []


def test_one_limit_sign_backs_off_once_then_waits_out_the_cooldown():
    c = make(event_cooldown_sec=3.0)
    _, t = drive(c, 2.1)                         # one raise: speed 2.5
    first, t = drive(c, 1.0, t0=t, stop_margin=0.0)
    assert len(first) == 1 and len(c.coverage.events) == 1
    _, t = drive(c, 1.9, t0=t, stop_margin=0.0)
    assert len(c.coverage.events) == 1           # still inside the 3 s
    _, t = drive(c, 0.2, t0=t, stop_margin=0.0)
    assert len(c.coverage.events) == 2           # the next one only after 3 s


def test_slow_or_straight_driving_is_never_called_a_slide():
    c = make(slide_hold_sec=0.1)
    assert drive(c, 1.0, speed=1.0, steering=0.3, yaw_rate=0.0)[0] == []
    assert drive(c, 1.0, t0=1.0, speed=3.0, steering=0.02, yaw_rate=0.0)[0] == []
    assert c.coverage.events == []


def test_low_battery_ends_the_session():
    c = make()
    applied, _ = drive(c, 1.0, voltage=10.0)
    assert c.done and 'battery' in c.done_reason
    assert applied == []
    assert drive(c, 10.0, t0=1.0)[0] == []          # nothing more once done


BASE = {MIN_SPEED: 1.429}  # gap_follow's shipped min_speed (src/gap_follow/config/gap_follow.yaml)


def stopper(**rules):
    """Stop tests allowed straight away: braking is short of data, no gap."""
    kw = dict(need_braking_sec=4.0, brake_gap_sec=0.0, brake_min_speed=2.5, brake_clear=1.0,
              brake_hold_sec=3.0, brake_decel=3.0, brake_speed=0.5)
    kw.update(rules)
    return make(baseline=BASE, **kw)


def test_a_stop_test_lowers_max_and_min_speed_then_puts_both_back():
    c = stopper()
    # 3 m/s needs 9/6 = 1.5 m to brake, plus 1.0 m spare: 2.51 m is enough.
    applied, t = drive(c, 0.1, speed=3.0, speed_cmd=3.0, stop_margin=2.51)
    assert applied == [{SPEED: 0.5, LATERAL: 1.5, MIN_SPEED: 0.5}]
    # The car slows to brake_speed + 0.3: the test is over, gap_follow's own min_speed returns.
    applied, _ = drive(c, 0.1, t0=t, speed=0.75, speed_cmd=0.5)
    assert applied == [{SPEED: 2.0, LATERAL: 1.5, MIN_SPEED: 1.429}]
    assert c.coverage.brake_tests == 1


def test_a_stop_test_that_never_slows_ends_at_the_hold_time():
    c = stopper(brake_hold_sec=1.0)
    _, t = drive(c, 0.05, speed=3.0, speed_cmd=3.0, stop_margin=5.0)
    applied, _ = drive(c, 0.9, t0=t, speed=3.0, speed_cmd=0.5)
    assert applied == []
    applied, _ = drive(c, 0.2, t0=t + 0.9, speed=3.0, speed_cmd=0.5)
    assert applied[0] == {SPEED: 2.0, LATERAL: 1.5, MIN_SPEED: 1.429}


def test_a_back_off_during_a_stop_test_puts_min_speed_back_too():
    c = stopper()
    _, t = drive(c, 0.05, speed=3.0, speed_cmd=3.0, stop_margin=5.0)
    applied, _ = drive(c, 0.05, t0=t, speed=3.0, safety_state='ttc_brake')
    assert applied == [{SPEED: 2.0, LATERAL: 1.5, MIN_SPEED: 1.429}] and c.brake_until is None


@pytest.mark.parametrize('why, kw', [
    ('no room: 2.49 m < 2.5 m', dict(speed=3.0, stop_margin=2.49)),
    ('too slow', dict(speed=2.0, stop_margin=9.0)),
    ('cornering', dict(speed=3.0, stop_margin=9.0, steering=0.2)),
    ('room unknown', dict(speed=3.0, stop_margin=None)),
])
def test_no_stop_test_without_room_speed_or_a_straight(why, kw):
    c = stopper()
    applied, _ = drive(c, 1.0, speed_cmd=kw['speed'], **kw)
    assert all(a[SPEED] != 0.5 for a in applied), why
    assert c.coverage.brake_tests == 0 and c.brake_until is None


def test_no_stop_test_while_another_number_is_further_behind():
    # Grip has none of its 20 s, braking has 3 of 4: the ladder's raises come first.
    c = stopper(need_grip_sec=20.0)
    c.ledger.seconds['braking'] = 3.0
    applied, _ = drive(c, 1.0, speed=3.0, speed_cmd=3.0, stop_margin=9.0)
    assert all(a[SPEED] != 0.5 for a in applied) and c.ledger.short() == ['grip', 'braking']


def test_a_low_battery_during_a_stop_test_puts_everything_back():
    c = stopper()
    _, t = drive(c, 0.05, speed=3.0, speed_cmd=3.0, stop_margin=5.0)
    applied, _ = drive(c, 0.05, t0=t, speed=3.0, voltage=10.0)
    assert applied == [{SPEED: 2.0, LATERAL: 1.5, MIN_SPEED: 1.429}] and c.done


def test_no_stop_test_without_gap_follows_min_speed():
    # Without it a max_speed below min_speed would be refused by gap_follow.
    c = make(baseline=None, need_braking_sec=4.0, brake_gap_sec=0.0)
    applied, _ = drive(c, 1.0, speed=3.0, speed_cmd=3.0, stop_margin=9.0)
    assert all(a[SPEED] != 0.5 for a in applied) and not c.can_stop_test
    assert all(MIN_SPEED not in a for a in applied)


def test_no_stop_test_once_braking_has_its_data_or_too_soon_after_the_last():
    c = stopper(need_braking_sec=0.5)
    # Ordinary braking: 1.1 m/s faster than the command for 0.6 s fills the ledger.
    drive(c, 0.6, speed=3.0, speed_cmd=1.9, stop_margin=0.5)
    assert c.ledger.seconds['braking'] >= 0.5
    applied, _ = drive(c, 1.0, t0=1.0, speed=3.0, speed_cmd=3.0, stop_margin=9.0)
    assert all(a[SPEED] != 0.5 for a in applied)
    c = stopper(brake_gap_sec=20.0)
    applied, _ = drive(c, 5.0, speed=3.0, speed_cmd=3.0, stop_margin=9.0)
    assert all(a[SPEED] != 0.5 for a in applied)


# --- the information ledger -------------------------------------------------------------

def test_the_ledger_counts_cornering_near_the_limit_hard_braking_and_fast_steering():
    c = make(need_grip_sec=10.0, need_braking_sec=10.0, need_steering_sec=10.0)
    # Lateral setting 1.5, so 0.8 x 1.5 = 1.2 m/s^2 counts as near the limit.
    # Each stretch is 1 s of ticks after the first; the stretches join end to end.
    _, t = drive(c, 1.05, speed=3.0, steering=0.0, yaw_rate=0.42, speed_cmd=3.0)   # 1.26 m/s^2
    _, t = drive(c, 1.0, t0=t, speed=3.0, steering=0.0, yaw_rate=0.3, speed_cmd=3.0)  # 0.9: not
    assert c.ledger.seconds['grip'] == pytest.approx(1.0, abs=1e-6)  # s; first tick has no dt
    _, t = drive(c, 1.0, t0=t, speed=3.0, speed_cmd=1.9)   # 1.1 m/s over the command: counts
    _, t = drive(c, 1.0, t0=t, speed=3.0, speed_cmd=2.1)   # 0.9 over: not
    assert c.ledger.seconds['braking'] == pytest.approx(1.0, abs=1e-6)  # s
    for k in range(20):   # steering swinging 0.1 rad every 50 ms: >= 1 rad/s, 20 ticks
        c.update(Inputs(now=t + k * TICK, lb_held=True, speed=2.0, steering=0.05 * (-1) ** k,
                        stop_margin=5.0))
    assert c.ledger.seconds['steering'] == pytest.approx(1.0, abs=1e-6)  # s
    # All three short of 10 s; the furthest behind comes first (all at 1/10: by name).
    assert sorted(c.ledger.short()) == ['braking', 'grip', 'steering']


def test_a_session_at_both_limits_still_runs_until_every_number_has_its_data():
    c = Conductor(Fence(speed_top=2.0, lateral_top=1.5),
                  Rules(segment_sec=1.0, need_grip_sec=1.0, need_braking_sec=0.0,
                        need_steering_sec=0.0))
    c.start()
    drive(c, 3.0, speed=2.0, steering=0.0, yaw_rate=0.0)
    assert not c.done                                    # no grip data yet
    drive(c, 3.0, t0=3.0, speed=3.0, steering=0.0, yaw_rate=0.5)
    assert c.done and 'every number' in c.done_reason


def slides_after_clean(c, clean_sec, t):
    """Drive clean (straight, as asked) for ``clean_sec``, then slide once."""
    _, t = drive(c, clean_sec, t0=t)
    applied, t = drive(c, 0.6, t0=t, **sliding(share=0.5))
    return applied, t


def test_once_cornering_is_held_a_slide_lowers_speed_and_then_the_session_ends():
    """The first slide holds cornering. Only speed rises after that, so the next
    slide lowers speed (on a slippery floor, blaming cornering again let speed
    climb to 5 m/s through 14 slides in the simulator). A slide with nothing
    left to lower ends the session."""
    # Grip short of data holds the session open once both settings are held.
    c = make(slide_hold_sec=0.2, slide_steer_window=0.3, event_cooldown_sec=1.0,
             need_grip_sec=600.0)
    applied, t = drive(c, 0.6, **sliding(share=0.5))
    assert applied == [{SPEED: 2.0, LATERAL: 1.5}] and c.ladder.lateral_frozen
    # Three clean 2 s segments raise speed only: 2.0 -> 3.5.
    _, t = drive(c, 6.5, t0=t)
    assert c.applied == {SPEED: 3.5, LATERAL: 1.5}
    applied, t = drive(c, 0.6, t0=t, **sliding(share=0.5))
    assert applied == [{SPEED: 2.5, LATERAL: 1.5}]          # two notches down, cornering kept
    assert c.ladder.limits == {LATERAL: 1.5, SPEED: 3.5} and c.ladder.speed_frozen
    assert 'speed comes down' in c.coverage.events[-1]['detail']
    # Sliding again below the speed already found: the floor may have changed.
    applied, t = slides_after_clean(c, 1.5, t)
    assert applied == [{SPEED: 2.0, LATERAL: 1.5}]
    assert 'floor may have changed' in c.coverage.events[-1]['detail']
    assert not c.done
    # Both at their lowest notch and still sliding: nothing left to lower.
    applied, t = slides_after_clean(c, 1.5, t)
    assert applied == [] and c.done
    assert c.done_reason == 'still sliding at the gentlest settings'
    assert drive(c, 3.0, t0=t)[0] == []                      # and nothing is raised after


def test_a_slide_with_speed_at_its_lowest_lowers_cornering_again_rather_than_ending():
    c = make(slide_hold_sec=0.2, slide_steer_window=0.3, need_grip_sec=600.0)
    c.ladder.raise_one(), c.ladder.raise_one()               # speed 2.5, cornering 1.75
    c.ladder.back_off(SPEED)                                 # speed found at 2.5, back to 2.0
    c.ladder.lateral_frozen = True                           # cornering already held at 1.75
    applied, _ = drive(c, 0.6, **sliding(share=0.5))
    assert applied == [{SPEED: 2.0, LATERAL: 1.5}] and not c.done


def test_a_slide_that_ends_the_session_during_a_stop_test_puts_min_speed_back():
    c = stopper(slide_hold_sec=0.2, slide_steer_window=0.3)
    c.ladder.back_off(None)                                  # both frozen at the start
    applied, t = drive(c, 0.1, speed=3.0, speed_cmd=3.0, stop_margin=5.0)
    assert applied == [{SPEED: 0.5, LATERAL: 1.5, MIN_SPEED: 0.5}]
    applied, _ = drive(c, 0.6, t0=t, speed_cmd=0.5, **sliding(speed=3.0, share=0.5))
    assert applied == [{SPEED: 2.0, LATERAL: 1.5, MIN_SPEED: 1.429}] and c.done


# --- the course map ---------------------------------------------------------------------

def circle(radius=3.0, steps=200, turns=1.0):
    """Poses round a circle, heading along it."""
    for k in range(int(steps * turns) + 1):
        a = 2 * math.pi * k / steps
        yield (radius * math.cos(a), radius * math.sin(a), a + math.pi / 2)


def test_a_lap_closes_only_after_driving_away_and_coming_back():
    m = CourseMap()
    poses = list(circle(turns=0.9))
    for p in poses:
        m.add(p, 5.0)
    assert m.laps == 0 and not m.learned
    for p in list(circle(turns=1.0))[len(poses):]:
        m.add(p, 5.0)
    assert m.laps == 1 and m.learned


def test_a_short_trip_out_and_back_or_coming_back_the_wrong_way_is_not_a_lap():
    m = CourseMap()
    for k in range(61):                       # 3 m out and 3 m back: 6 m, under min_lap_m
        m.add((0.1 * min(k, 60 - k), 0.0, 0.0), 5.0)
    assert m.laps == 0
    m = CourseMap()
    for k in range(121):                      # 6 m out, then back to the start facing backwards
        x = 0.1 * min(k, 120 - k)
        m.add((x, 0.0, 0.0 if k <= 60 else math.pi), 5.0)
    assert m.laps == 0


def test_sitting_near_the_start_is_not_a_lap():
    m = CourseMap()
    for k in range(500):
        m.add((0.01 * (k % 3), 0.0, 0.0), 5.0)
    assert m.laps == 0


def test_the_map_keeps_the_least_room_seen_per_place_and_heading():
    m = CourseMap()
    m.add((1.1, 1.1, 0.0), 4.0)
    m.add((1.2, 1.2, 0.1), 2.5)
    m.add((1.2, 1.2, 0.2), 3.0)
    assert m.room_at((1.15, 1.15, 0.05)) == 2.5
    assert m.room_at((1.15, 1.15, math.pi)) is None       # same place, other way: unknown
    assert m.room_at((9.0, 9.0, 0.0)) is None
    assert m.room_at(None) is None and m.room_at((math.nan, 0.0, 0.0)) is None


def lap_then(c, room=9.0, speed=2.0):
    """Drive one closing lap of the circle (too slow for a stop test); return
    the time after it."""
    t = 0.0
    for p in circle(steps=40):
        c.update(Inputs(now=t, lb_held=True, speed=speed, steering=0.0, yaw_rate=0.0,
                        stop_margin=9.0, speed_cmd=speed, pose=p, straight_room=room))
        t += TICK
    return t


def test_no_raise_while_the_first_lap_maps_the_course():
    c = make(learn_timeout_sec=90.0)
    applied = []
    for k, p in enumerate(circle(steps=200, turns=0.85)):   # 8.5 s of segments, lap still open
        out = c.update(Inputs(now=k * TICK, lb_held=True, speed=2.0, steering=0.0,
                              stop_margin=5.0, pose=p, straight_room=5.0))
        if out:
            applied.append(out)
    assert applied == [] and c.learning
    for k, p in enumerate(list(circle(steps=200))[170:]):
        closed = 8.55 + k * TICK
        out = c.update(Inputs(now=closed, lb_held=True, speed=2.0, steering=0.0,
                              stop_margin=5.0, pose=p, straight_room=5.0))
        assert out is None
        if c.course.learned:
            break
    assert not c.learning and c.course.learned
    # The mapping lap counted for nothing: the first raise needs a full 2 s segment.
    applied, t = drive(c, 1.9, t0=closed + TICK)
    assert applied == []
    applied, _ = drive(c, 0.2, t0=t)
    assert applied and applied[0][SPEED] == 2.5


def test_a_lap_that_never_closes_falls_back_to_the_scan_after_the_timeout():
    c = make(learn_timeout_sec=3.0)
    applied, t = drive(c, 2.9, pose=(0.0, 0.0, 0.0), straight_room=5.0)
    assert applied == [] and c.learning
    applied, _ = drive(c, 2.2, t0=t, pose=(0.0, 0.0, 0.0), straight_room=5.0)
    assert not c.learning and applied and applied[0][SPEED] == 2.5


def test_once_mapped_a_stop_test_needs_room_on_the_map_too_and_each_place_is_used_once():
    c = stopper()
    t = lap_then(c, room=1.0)                       # the map: only 1.0 m ahead anywhere
    p = (3.0, 0.0, math.pi / 2)
    applied = [c.update(Inputs(now=t + k * TICK, lb_held=True, speed=3.0, steering=0.0,
                               stop_margin=9.0, speed_cmd=3.0, pose=p, straight_room=9.0))
               for k in range(5)]
    assert all(a is None or a[SPEED] != 0.5 for a in applied)   # live scan alone is not enough
    c = stopper()
    t = lap_then(c, room=9.0)
    out = c.update(Inputs(now=t, lb_held=True, speed=3.0, steering=0.0, stop_margin=9.0,
                          speed_cmd=3.0, pose=p, straight_room=9.0))
    assert out[SPEED] == 0.5 and c.course.key(*p) in c.course.tested
    t2 = t + 0.05
    for _ in range(80):                             # the test ends, braking still short
        c.update(Inputs(now=t2, lb_held=True, speed=0.6, steering=0.0, stop_margin=9.0,
                        speed_cmd=0.5, pose=p, straight_room=9.0))
        t2 += TICK
    again = [c.update(Inputs(now=t2 + k * TICK, lb_held=True, speed=3.0, steering=0.0,
                             stop_margin=9.0, speed_cmd=3.0, pose=p, straight_room=9.0))
             for k in range(5)]
    assert all(a is None or a[SPEED] != 0.5 for a in again)       # same place: not again


def test_session_time_cap_ends_the_session():
    c = make(max_session_sec=3.0)
    drive(c, 3.1)
    assert c.done and 'session time' in c.done_reason


def test_coverage_counts_seconds_by_speed_and_cornering_bin():
    c = make()
    drive(c, 1.0, speed=3.2, steering=0.0, yaw_rate=0.5)    # a_lat 1.6
    bins = {(b['speed_from'], b['lateral_from']): b['seconds'] for b in c.report()['bins']}
    assert bins == {(3.0, 1.5): pytest.approx(0.95, abs=1e-6)}  # s; first tick has no dt


@pytest.mark.parametrize('kw', [dict(wall_margin=-0.01), dict(slide_ratio=1.0),
                                dict(brake_speed=0.2), dict(brake_speed=3.0),
                                dict(segment_sec=0.0), dict(wheelbase=math.nan),
                                dict(brake_hold_sec=0.0), dict(brake_delay=-0.1),
                                dict(grip_info_ratio=0.0), dict(grip_info_ratio=1.5),
                                dict(need_grip_sec=-1.0), dict(learn_timeout_sec=-1.0)])
def test_rules_out_of_range_are_refused(kw):
    with pytest.raises(ConfigError):
        Rules(**kw)


def test_a_step_that_overshoots_the_top_lands_exactly_on_it():
    lad = Ladder(Fence(speed_step=0.7, lateral_top=1.5))
    speeds = [lad.setting()[SPEED]]
    while (s := lad.raise_one()) is not None:
        speeds.append(s[SPEED])
    assert speeds == pytest.approx([2.0, 2.7, 3.4, 4.1, 4.8, 5.0], abs=1e-9)  # m/s
