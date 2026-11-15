"""Static confirmation of tracks by the range-rate residuals (opponent_tracking, docs/gap-follow.md#static-confirmation).

Pure, no rclpy. Oracle: closed-form geometry. A fixed post or a target with a known world velocity is seen from a
LiDAR at laser_x ahead of the rear axle on a prescribed ego path; the tracker is fed the exact range and bearing of the
nearest point and the wheel speed and yaw rate of the path. Every number below is that geometry or a stated rule.
"""
import math

import pytest

from gap_follow.opponent_tracking import (FollowCapMemory, OpponentAwareness, OpponentParams, OpponentTracker, Track)

DT = 0.025
LX = 0.26


def lidar_view(ego, point):
    x, y, yaw = ego
    lx, ly = x + LX * math.cos(yaw), y + LX * math.sin(yaw)
    dx, dy = point[0] - lx, point[1] - ly
    return math.hypot(dx, dy), math.atan2(dy, dx) - yaw


def run(path, target, ticks, params=None, radial_scale=1.0):
    """path(t) -> (x, y, yaw, v, omega); target(t) -> (px, py). Returns the tracker and its track after ``ticks`` scans."""
    tracker = OpponentTracker(params or OpponentParams())
    for i in range(ticks):
        t = i * DT
        x, y, yaw, v, omega = path(t)
        px, py = target(t)
        r, b = lidar_view((x, y, yaw), (px, py))
        det = [(px, py, 0.0, r, b)]
        if i == 0:
            tracker.tracks.append(Track(px, py, near_range=r, prev_range=r, prev_clock=0.0, prev_bearing=b, track_id=1))
        else:
            tracker.update(det, DT, radial=(v * radial_scale, omega, LX))
    return tracker, tracker.tracks[0]


def straight(t):
    return 2.0 * t, 0.0, 0.0, 2.0, 0.0


def arc(t, speed=2.0, radius=1.5):
    w = speed / radius
    return radius * math.sin(w * t), radius * (1 - math.cos(w * t)), w * t, speed, w


POST = lambda t: (6.0, 0.7)


def test_a_post_the_ego_drives_past_in_a_straight_line_has_both_residuals_near_zero_and_is_confirmed():
    _, tr = run(straight, POST, 16)
    assert tr.static_residual == pytest.approx(0.0, abs=0.05)         # closed form: a fixed point gives 0
    assert tr.static_tangential == pytest.approx(0.0, abs=0.05)
    assert tr.static_confirmed


def test_the_same_post_while_the_ego_drives_a_1_5_m_arc_at_2_ms_is_still_confirmed():
    # omega = v / R exactly; the post at (1.2, 1.4) is ahead-left of the arc's start
    _, tr = run(arc, lambda t: (1.6, 1.8), 16)
    assert tr.static_residual == pytest.approx(0.0, abs=0.05)
    assert tr.static_tangential == pytest.approx(0.0, abs=0.05)
    assert tr.static_confirmed


def test_a_target_crossing_at_2_ms_has_a_tangential_residual_of_about_2_and_is_not_confirmed():
    # the target moves at (0, 2) in the world, 2.3 m ahead of the start: its world velocity's radial part is
    # 2 sin b and its tangential part 2 cos b (the residuals are the world velocity's components), |b| < 0.3
    _, tr = run(straight, lambda t: (2.3, -0.3 + 2.0 * t), 12)
    assert tr.static_residual < 0.6
    assert 1.8 <= tr.static_tangential <= 2.0
    assert not tr.static_confirmed


def test_a_target_moving_away_along_the_line_of_sight_at_2_ms_is_not_confirmed():
    _, tr = run(straight, lambda t: (4.0 + 2.0 * t, 0.0), 16)         # world velocity (2, 0), dead ahead: the residual is its own speed along the line of sight
    assert tr.static_residual == pytest.approx(2.0, abs=0.1)
    assert not tr.static_confirmed


def test_a_wrong_wheel_speed_shows_up_as_a_radial_residual():
    _, tr = run(straight, lambda t: (6.0, 0.0), 16, radial_scale=1.25)          # v reported 0.5 m/s too high, post dead ahead
    assert tr.static_residual == pytest.approx(0.5, abs=0.02)
    assert not tr.static_confirmed                                               # 0.5 >= S 0.35


# --- missing samples never confirm -------------------------------------------------------------------

def test_three_samples_do_not_confirm():
    tracker, tr = run(straight, POST, 4)                                           # tick 0 creates, ticks 1-3 sample
    assert len(tr.samples) == 3 and not tr.static_confirmed
    tracker, tr = run(straight, POST, 5)
    assert len(tr.samples) == 4 and tr.static_confirmed                            # the fourth makes it (>= 4)


def test_four_samples_all_older_than_0_6_s_do_not_confirm():
    tracker, tr = run(straight, POST, 5)
    assert tr.static_confirmed
    for _ in range(25):                                                            # 0.625 s of detections that add no sample
        tracker.update([(6.0, 0.7, 0.0, 4.0, 0.1)], DT, radial=None)
    assert len(tr.samples) == 4 and not tr.static_confirmed


def test_a_nan_range_or_bearing_a_zero_dt_or_a_long_gap_add_no_sample():
    tracker, tr = run(straight, POST, 3)
    n = len(tr.samples)
    for det, dt in (([(6.0, 0.7, 0.0, math.nan, 0.1)], DT), ([(6.0, 0.7, 0.0, 4.0, math.nan)], DT),
                    ([(6.0, 0.7, 0.0, math.inf, 0.1)], DT)):
        tracker.update(det, dt, radial=(2.0, 0.0, LX))
        assert len(tr.samples) == n
    tracker.tracks[0].prev_range, tracker.tracks[0].prev_bearing = 4.0, 0.1
    tracker.tracks[0].prev_clock = tracker.clock + 0.3                               # a 0.3 s gap would be > 0.25 s... (negative)
    tracker.update([(6.0, 0.7, 0.0, 4.0, 0.1)], DT, radial=(2.0, 0.0, LX))
    assert len(tr.samples) == n
    tracker.tracks[0].prev_range, tracker.tracks[0].prev_bearing, tracker.tracks[0].prev_clock = 4.0, 0.1, tracker.clock - 0.2999
    tracker.update([(6.0, 0.7, 0.0, 4.0, 0.1)], 0.0001, radial=(2.0, 0.0, LX))      # gap 0.30 s
    assert len(tr.samples) == n
    tracker.update([(6.0, 0.7, 0.0, 4.0, 0.1)], DT, radial=(math.nan, 0.0, LX))
    assert len(tr.samples) == n


# --- the boundary ------------------------------------------------------------------------------------

def _confirmed_with(radial, tangential, params):
    tracker = OpponentTracker(params)
    t = Track(0.0, 0.0)
    tracker.clock = 1.0
    t.samples = [(1.0 - 0.025 * i, radial, tangential) for i in range(8)]
    tracker._confirm_static(t)
    return t.static_confirmed


def test_exactly_at_the_radial_or_tangential_limit_is_not_confirmed_and_just_under_is():
    p = OpponentParams()
    s, t = p.follow_static_radial_speed, p.follow_static_tangential_speed
    assert _confirmed_with(s - 0.01, t - 0.01, p)
    assert not _confirmed_with(s, t - 0.01, p)
    assert not _confirmed_with(s - 0.01, t, p)
    assert not _confirmed_with(-s, t - 0.01, p)                                    # |residual|: a negative one counts by magnitude


def test_a_limit_of_zero_never_confirms():
    assert not _confirmed_with(0.0, 0.0, OpponentParams(follow_static_radial_speed=0.0))
    assert not _confirmed_with(0.0, 0.0, OpponentParams(follow_static_tangential_speed=0.0))


@pytest.mark.parametrize('name,value', [('follow_static_radial_speed', -0.1), ('follow_static_radial_speed', 1.01),
                                         ('follow_static_radial_speed', math.nan), ('follow_static_tangential_speed', -1.0),
                                         ('follow_static_tangential_speed', 2.01), ('follow_static_tangential_speed', math.inf)])
def test_unusable_limits_are_refused_by_name(name, value):
    with pytest.raises(ValueError, match=name.replace('follow_', 'opponent_follow_', 1)):
        OpponentParams(**{name: value}).validate()


# --- fail-safe restore, only the follow cap is dropped -----------------------------------------------

def _awareness_with_moving_track(params=None):
    """A track in the tracker's own state that the follow law applies to: moving, 3 m dead ahead, confirmed car."""
    aw = OpponentAwareness(params or OpponentParams(), wheelbase=0.36, laser_x=LX, laser_y=0.0, car_length=0.58)
    t = Track(3.0, 0.0, vx=1.0, moving=True, confirmed_car=True, near_dx=-0.1, track_id=5, streak=20, hits=20, near_range=2.7)
    aw.tracker.tracks.append(t)
    aw.tracker.clock = 5.0
    return aw, t


def test_a_confirmed_track_contributes_no_follow_cap_and_is_followed_again_the_tick_its_median_crosses_the_limit():
    aw, t = _awareness_with_moving_track()
    assert math.isfinite(aw.follow_law_cap(passing=False))                          # followed as today without samples
    t.samples = [(5.0 - 0.025 * i, 0.0, 0.0) for i in range(8)]
    aw.tracker._confirm_static(t)
    assert t.static_confirmed and aw.follow_law_cap(passing=False) == math.inf
    assert aw.alongside_cap(passing=False) == math.inf
    assert aw.moving_body() != []                                                    # still an obstacle: inflation, passing, clearance
    flipped = None
    for k in range(5):                                                               # five residuals of +1.5 m/s
        aw.tracker.clock += DT
        t.samples.append((aw.tracker.clock, 1.5, 0.0))
        del t.samples[:-8]
        aw.tracker._confirm_static(t)
        if not t.static_confirmed:
            flipped = k
            break
    assert flipped == 3                                                              # median of 8 = (0 + 1.5)/2 > 0.35 at the 4th
    assert math.isfinite(aw.follow_law_cap(passing=False))                           # back on the very same tick


def test_gate_off_gives_the_follow_cap_of_today():
    off = OpponentParams(follow_static_radial_speed=0.0)
    aw, t = _awareness_with_moving_track(off)
    t.samples = [(5.0 - 0.025 * i, 0.0, 0.0) for i in range(8)]
    aw.tracker._confirm_static(t)
    assert not t.static_confirmed
    ref, _ = _awareness_with_moving_track(off)
    assert aw.follow_law_cap(passing=False) == ref.follow_law_cap(passing=False) < math.inf


# --- the follow-cap memory remembers who set it -------------------------------------------------------

def test_the_memory_is_cleared_only_when_the_track_that_set_it_is_static():
    mem = FollowCapMemory(1.0)
    assert mem.update(0.8, 10.0, 11) == (0.8, False)                                  # track 11 (real) sets the cap, then is lost
    mem.clear_if_static({12})                                                         # a different track is confirmed static
    for k in range(1, 12):                                                            # 0.3 s later, scans every 25 ms
        assert mem.update(math.inf, 10.0 + 0.025 * k) == (0.8, True), k              # A's cap still held
    for k in range(12, 40):
        mem.update(math.inf, 10.0 + 0.025 * k)
    assert mem.update(math.inf, 11.0) == (0.8, True)                                  # its whole memory window
    assert mem.update(math.inf, 11.025)[0] == math.inf
    mem = FollowCapMemory(1.0)
    mem.update(0.8, 10.0, 11)
    mem.clear_if_static({11})                                                         # the remembered track itself is static
    assert mem.update(math.inf, 10.025) == (math.inf, False)


def test_a_cap_stored_without_a_source_is_never_cleared_by_a_confirmation():
    mem = FollowCapMemory(1.0)
    mem.update(0.8, 10.0)
    mem.clear_if_static({1, 2, 3})
    assert mem.update(math.inf, 10.1) == (0.8, True)
