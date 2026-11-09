"""Real decision bodies and independent race accounting; no ROS or gym needed."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import race_two as race


def rectangle():
    # 40 m perimeter, 0.1 m stations: closed-form arc oracle.
    points = np.vstack((np.column_stack((np.arange(0, 10, .1), np.zeros(100))),
                        np.column_stack((np.full(100, 10), np.arange(0, 10, .1))),
                        np.column_stack((np.arange(10, 0, -.1), np.full(100, 10))),
                        np.column_stack((np.zeros(100), np.arange(10, 0, -.1)))))
    return race.rv.PathPlan(points, race.racing_math.compute_segment_lengths(points, closed=True),
                            np.full(400, 4.0))


def test_truth_progress_unwraps_forward_and_reverse_without_counting_grid_advantage():
    p = race.Progress(rectangle(), [0, .2])
    assert p.distance == 0
    assert p.update([.3, 0]) == pytest.approx(.5, abs=1e-9)  # metres
    assert p.update([0, .1]) == pytest.approx(.1, abs=1e-9)
    assert p.update([0, .2]) == pytest.approx(0, abs=1e-9)


def row(time, lead=-2, **kwargs):
    result = dict(time=time, ego_progress=0, opp_progress=-lead - 3,
                  head_start=3, active=False, status='', ego_command=1,
                  profile_speed=4, safety='none', car_contact=[], wall_contact=[])
    result.update(kwargs)
    return result


def test_referee_does_not_believe_controller_completion_or_a_brief_nose_ahead():
    rows = [row(.1, active=True), row(.2, lead=.2, status='overtake complete')]
    rows += [row(.3 + .1 * i, lead=1.2) for i in range(4)]
    result = race.race_metrics(rows, .1, 40, 4)
    assert (result['attempted'], result['completed'], result['abandoned']) == (1, 0, 0)
    rows.append(row(.7, lead=1.2))
    assert race.race_metrics(rows, .1, 40, 4)['completed'] == 1


def test_accounting_common_finish_line_contacts_abandonment_and_stuck_time():
    rows = [row(.1, active=True), row(.2, status='overtake abandoned: stalled',
                                    safety='body_contact', car_contact=[0, 1]),
            row(.3, ego_progress=159.9, opp_progress=157, wall_contact=[1]),
            row(.4, ego_progress=160, opp_progress=157.1, ego_command=4)]
    result = race.race_metrics(rows, .1, 40, 4)
    assert result['race_time_s'] == .4
    assert result['opponent_race_time_s'] == .3  # 3 m starting advantage
    assert result['abandoned'] == 1
    assert result['stuck_s'] == .3
    assert result['emergency_stop_ticks'] == 1
    assert result['car_contact_ticks'] == result['wall_contact_ticks'] == 1
    assert race.race_metrics([], .1, 40, 4)['race_time_s'] is None


def decisions(wall_distance=math.inf):
    lidar = SimpleNamespace(angle_min=-3 * math.pi / 4, angle_increment=math.pi / 720,
                            range_min=.05)
    expected = lambda x, y, yaw, angles: np.full(len(angles), 10.0)
    d = race.NodeDecisions(rectangle(), lidar, expected, lambda x, y: wall_distance,
                           race.rv._pure_pursuit_params('rb2'))
    scan = np.full(1081, 10.0)
    angles = lidar.angle_min + np.arange(1081) * lidar.angle_increment
    scan[np.abs(angles) < .08] = 2.0
    return d, scan


def test_real_car_needs_confirmation_and_wall_phantom_never_commits():
    # Node spec: six sightings spanning >=0.15s; wall distance >=0.30m.
    d, scan = decisions()
    for now in (0, .025, .05, .075, .1, .125):
        d.observe([0, 0, 0, 1, 0], scan, now, 0)
        assert d._update_opponent_and_overtake(0) is None
        assert not d.overtake_active
    d.observe([0, 0, 0, 1, 0], scan, .175, 0)
    target = d._update_opponent_and_overtake(0)
    # x is the 4m preview station; y is exactly the specified offset.
    assert target == pytest.approx((4.0, .35), abs=1e-9)  # metres
    assert d.overtake_active
    phantom, scan = decisions(.299)
    for now in np.arange(0, .6, .025):
        phantom.observe([0, 0, 0, 1, 0], scan, now, 0)
        assert phantom._update_opponent_and_overtake(0) is None
        assert not phantom.overtake_active


def test_node_stalled_pass_gives_up_resets_gate_and_obeys_retry():
    d, scan = decisions()
    for now in np.arange(0, .3, .025):
        d.observe([0, 0, 0, 1, 0], scan, float(now), 0)
        d._update_opponent_and_overtake(0)
    assert d.overtake_active
    for now in np.arange(.3, 2, .025):
        d.observe([0, 0, 0, 1, 0], scan, float(now), 0)
        d._update_opponent_and_overtake(0)
    assert not d.overtake_active
    assert d._overtake_retry_after_sec > 4.5
    assert 'no new pass' in d.last_opponent_status


def test_off_line_recovery_aims_at_line_times_out_and_can_reset():
    d, scan = decisions()
    follower = race.rv.PathFollower(rectangle(), law_name='delay_lqr', car_id='rb2')
    d.observe([2, -1.1, 0, .2, 0], scan, 0, 0)
    steer, speed, _, target, state = d.base_command(follower, [2, -1.1, 0, .2, 0], 0)
    assert target == pytest.approx((2, 0), abs=1e-9)  # nearest point, not a preview
    assert steer > 0  # line lies left of the car
    assert speed == .3
    assert state == 'off_racing_line_recovery'
    d.now = 5.001
    assert d.base_command(follower, [2, -1.1, 0, .2, 0], 0)[1] == 0
    d.now = 5.1
    assert d.base_command(follower, [2, 0, 0, .2, 0], 0)[-1] is None
    assert not d.recovery_exhausted
    assert d.base_command(follower, [2, -2.01, 0, .2, 0], 0)[1] == 0


@pytest.mark.parametrize('scan_kind', ['missing', 'stale', 'flank'])
def test_source_safety_adapter_denies_blind_or_flank_contact(scan_kind):
    d, _ = decisions()
    scan = np.full(1081, 10.0)
    d.observe([0, 0, 0, 2, 0], scan, 1.0, 0)
    if scan_kind == 'missing':
        d.last_scan = None
    elif scan_kind == 'stale':
        d.last_scan_time = 0
    else:
        angles = d.lidar.angle_min + np.arange(1081) * d.lidar.angle_increment
        scan[np.abs(angles - math.pi / 2) < .02] = .16
    steer, speed, state, _ = d._reactive_override(allow_avoidance=False, desired_steering=0)
    assert speed == 0.0
    assert steer is None
    assert state in ('lidar_scan_missing', 'lidar_scan_stale', 'body_contact')


def test_shipped_car2_follows_after_confirmation_without_an_offset_target():
    d, scan = decisions()
    follower = race.rv.PathFollower(rectangle(), law_name='delay_lqr', car_id='rb2')
    for now in np.arange(0, .125, .025):
        d.observe([0, 0, 0, 1, 0], scan, float(now), 0)
        _, _, _, target, _ = d.base_command(follower, [0, 0, 0, 1, 0], 0)
        # oracle: spec; the shipped confirmation gate still spans .15s.
        assert d.follow_speed_cap is None
        assert target is None
    for now in np.arange(.125, .3, .025):
        d.observe([0, 0, 0, 1, 0], scan, float(now), 0)
        _, _, _, target, _ = d.base_command(follower, [0, 0, 0, 1, 0], 0)
    # oracle: invariant; a real confirmed car gets a physical-braking cap,
    # never a point-offset target, in car 2's shipped configuration.
    assert 0 < d.follow_speed_cap < 2
    assert target is None
    assert not d.overtake_active
    assert d.enable_opponent_following and not d.enable_opponent_overtake


@pytest.mark.parametrize('kind', ['clear', 'blocked', 'stale', 'invalid', 'nan', 'zero', 'short'])
def test_lost_opponent_reacquisition_requires_a_visible_stopping_arc(kind):
    d, scan = decisions()
    angles = d.lidar.angle_min + np.arange(1081) * d.lidar.angle_increment
    # Map/scan input boundary: a known wall at y=.20. A 0.31m-wide car
    # fits, but the inflated line corridor is occluded. All math is real.
    def wall(x, y, yaw, beam_angles):
        sine = np.sin(yaw + beam_angles)
        return np.where(sine > .001, np.minimum(10, .2 / np.maximum(sine, .001)), 10)
    scan[:] = wall(0, 0, 0, angles)
    d.map_ray_caster.expected_ranges = wall
    d.opponent.arc_length, d.opponent.last_update_time = 2.0, 0.0
    d.opponent.progress_rate = 1.0
    if kind == 'blocked':
        scan[np.abs(angles) < .1] = .8
    elif kind == 'invalid':
        scan[:] = math.inf
    elif kind == 'nan':
        scan[:] = math.nan
    elif kind == 'zero':
        scan[:] = 0
    elif kind == 'short':
        scan = np.array([10.0])
    d.observe([0, 0, 0, .1, 0], scan, 3.0, 0)
    if kind == 'stale':
        d.last_scan_time = 0.0
    d._update_opponent_and_overtake(0, allow_pass=False, desired_steering=0)
    # oracle: invariant; no blind timeout or escape from a blocked arc.
    # The only permitted lost-track motion is the specified .25m/s crawl.
    assert d.follow_speed_cap == (.25 if kind == 'clear' else 0.0)
    assert not d.overtake_active


def test_confirmed_car_behind_on_the_line_does_not_create_a_forward_follow_cap():
    d, scan = decisions()
    # Ego faces backwards here: the real detected object is in its scan's
    # forward cone, but behind it along the racing line. This isolates
    # the longitudinal gate used at folded/adjacent track sections.
    for now in np.arange(0, .3, .025):
        d.observe([5, 0, 0, 1, math.pi], scan, float(now), 0)
        d._update_opponent_and_overtake(50, allow_pass=False)
    # oracle: invariant; only an opponent ahead on the line triggers following.
    assert d.opponent.arc_length < 5
    assert d.follow_speed_cap is None
    assert not d.overtake_active


def test_following_retains_an_ahead_car_round_a_bend_outside_the_avoidance_cone():
    d, scan = decisions()
    angles = d.lidar.angle_min + np.arange(1081) * d.lidar.angle_increment
    # Rectangle's next leg is x=10, y increasing. Ego is at (8.5,0),
    # opponent surface at (10,2): 3.5m ahead along the line, bearing >30°.
    scan[:] = 10
    bearing = math.atan2(2, 10 - 8.5 - d.laser_offset_x)
    distance = math.hypot(2, 10 - 8.5 - d.laser_offset_x)
    scan[np.abs(angles - bearing) < .08] = distance
    for now in np.arange(0, .3, .025):
        d.observe([8.5, 0, 0, 1, 0], scan, float(now), 0)
        d._update_opponent_and_overtake(85, allow_pass=False)
    # oracle: known geometry and confirmation spec; following must retain
    # the real ahead car without borrowing the narrow avoidance aim cone.
    assert d.opponent.arc_length == pytest.approx(12, abs=.05)  # 5cm surface-centroid discretisation
    assert 0 < d.follow_speed_cap < d.max_speed
    assert not d.overtake_active


def test_speed_credit_needs_a_continuous_fresh_track_and_allows_braking():
    tracker = race.OpponentTracker(.3, 1.0)
    for now in np.arange(0, .351, .025):
        tracker.update((39.5 + 3 * now) % 40, float(now), 40)
    # oracle: closed form; 3m/s, minus 1m/s uncertainty and worst braking
    # through half the .35s measurement interval. Seam wrapping must agree.
    lower = tracker.speed_lower_bound(.35, 40, 1.0, 3.0)
    expected_ema = 3 * (1 - .7 ** 14)
    assert lower == pytest.approx(expected_ema - 1 - 3 * .35 / 2, abs=1e-9)  # m/s
    assert tracker.speed_lower_bound(.461, 40, 1, 3) == 0
    assert tracker.speed_lower_bound(.3, 40, 1, 3) == 0  # genuinely future observation
    assert tracker.speed_lower_bound(.35, 40, -1, 3) == 0
    assert tracker.speed_lower_bound(.35, 40, 1, 0) == 0
    assert tracker.speed_lower_bound(.35, 40, math.nan, 3) == 0
    tracker.update(3, 2.0, 40)
    assert tracker.speed_lower_bound(2.0, 40, 1, 3) == 0
    braking = race.OpponentTracker(1, 1)
    for now in np.arange(0, .351, .025):
        braking.update(3 * now - 1.5 * now ** 2, float(now), 40)
    assert 0 < braking.speed_lower_bound(.35, 40, .1, 3) <= 3 - 3 * .35
    braking.progress_rate = -1
    assert braking.speed_lower_bound(.35, 40, .1, 3) == 0


def test_old_surface_bound_is_retired_once_not_resurrected_each_lap():
    d, scan = decisions()
    scan[:] = 10
    d.opponent.arc_length, d.opponent.last_update_time = 2, 0
    d.observe([4, 0, 0, 1, 0], scan, 3, 0)
    d._update_opponent_and_overtake(40, allow_pass=False)
    # oracle: invariant; a safely traversed old location is not a new car
    # ahead when progress next wraps. New detections still need confirmation.
    assert d.follow_speed_cap is None
    d.observe([0, 0, 0, 1, 0], scan, 20, 0)
    d._update_opponent_and_overtake(0, allow_pass=False)
    assert d.follow_speed_cap is None


def test_opponent_reacquisition_does_not_alias_to_the_old_search_window():
    d, scan = decisions()
    d.opponent.prev_nearest_index = 300  # opposite part of the 40m rectangle
    for now in np.arange(0, .3, .025):
        d.observe([0, 0, 0, 1, 0], scan, float(now), 0)
        d._update_opponent_and_overtake(0, allow_pass=False)
    # oracle: geometry; surface is 2m ahead of laser x=.26 on y=0.
    assert d.opponent.arc_length == pytest.approx(2.26, abs=.025)  # m, scan-centroid discretisation
    assert 0 < d.follow_speed_cap < 2


def test_unconfirmed_traffic_brakes_without_borrowing_gap_steering():
    d, scan = decisions()
    angles = d.lidar.angle_min + np.arange(1081) * d.lidar.angle_increment
    scan[np.abs(angles) < .08] = 1.1
    d.opponent.arc_length = 2  # previously confirmed traffic memory
    d.observe([0, 0, 0, 2, 0], scan, 3, 0)
    steering, speed, state, _ = d._reactive_override(allow_avoidance=True, desired_steering=.25)
    # oracle: delayed physical stop, with visible surface-to-front gap.
    gap = 1.1 + .26 - (.58 + .36) / 2
    expected = math.sqrt((3 * .45) ** 2 + 6 * (gap - .65)) - 3 * .45
    assert steering is None
    assert speed == pytest.approx(expected, abs=1e-9)  # m/s
    assert state is None


def test_source_adapter_completes_only_on_persistent_measured_body_lead():
    d, scan = decisions()
    d.enable_opponent_following = False  # isolate the explicitly opted-in legacy pass
    for now in np.arange(0, .3, .025):
        d.observe([4.8, 0, 0, 1, 0], scan, float(now), 0)
        d._update_opponent_and_overtake(48, allow_pass=True)
    assert d.overtake_active
    # Ego rounds the right bend; the stationary observed surface at (7.06,0)
    # is now behind but still in the physical 270° scan.
    angles = d.lidar.angle_min + np.arange(1081) * d.lidar.angle_increment
    bearing = (math.atan2(-.5 - d.laser_offset_x, 7.06 - 10) - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
    measured = np.full(1081, 10.0)
    measured[np.abs(angles - bearing) < .055] = math.hypot(7.06 - 10, .5 + d.laser_offset_x)
    observations = [(now, measured) for now in (.3, .325, .35, .375, .4)]
    observations += [(.425, np.full(1081, 10.0))]
    observations += [(float(now), measured) for now in np.arange(.45, .95, .025)]
    for now, observation in observations:
        d.observe([10, .5, 0, 1, math.pi / 2], observation, now, 0)
        d._update_opponent_and_overtake(105, allow_pass=True)
        # oracle: approved contract; a missing measurement resets the .5s
        # hold, even though predicted progress already says ego is ahead.
        assert d.overtake_active
        assert not d.last_opponent_status.startswith('overtake complete:')
    d.observe([10, .5, 0, 1, math.pi / 2], measured, .95, 0)
    assert d._update_opponent_and_overtake(105, allow_pass=True) is None
    assert not d.overtake_active
    assert d.last_opponent_status.startswith('overtake complete: measured body lead')


# --- Stage 2: passing a stopped car (node's _local_pass_update bodies) ------

def straight_stadium():
    """30 m straights (y = 0 and y = 12), 6 m bends; 0.15 m stations."""
    step, straight, radius = .15, 30.0, 6.0
    pts = [(i * step, 0.0) for i in range(int(straight / step))]
    m = int(math.pi * radius / step)
    pts += [(straight + radius * math.cos(-math.pi / 2 + math.pi * i / m),
             radius + radius * math.sin(-math.pi / 2 + math.pi * i / m)) for i in range(m)]
    pts += [(straight - i * step, 2 * radius) for i in range(int(straight / step))]
    pts += [(radius * math.cos(math.pi / 2 + math.pi * i / m),
             radius + radius * math.sin(math.pi / 2 + math.pi * i / m)) for i in range(m)]
    points = np.array(pts)
    return race.rv.PathPlan(points, race.racing_math.compute_segment_lengths(points, closed=True),
                            np.full(len(points), 4.0))


def planner_decisions(walls):
    """Planner on; `walls['at']` is swapped by tests (first straight: y = 0)."""
    lidar = SimpleNamespace(angle_min=-3 * math.pi / 4, angle_increment=math.pi / 720, range_min=.05)
    expected = lambda x, y, yaw, angles: np.full(len(angles), 10.0)
    params = dict(race.rv._pure_pursuit_params('rb2'), enable_local_pass_planner=True,
                  enable_opponent_overtake=False)
    d = race.NodeDecisions(straight_stadium(), lidar, expected, lambda x, y: walls['at'](x, y), params)
    d.tracking_law = SimpleNamespace(name='delay_lqr')
    return d


def stopped_car_scan(d, ego_x, car_rear_x):
    """A stopped car's rear face dead ahead on the line (the first straight)."""
    scan = np.full(1081, 10.0)
    angles = d.lidar.angle_min + np.arange(1081) * d.lidar.angle_increment
    scan[np.abs(angles) < .08] = car_rear_x - (ego_x + d.laser_offset_x)
    return scan


def follow_stopped_car(d, ego_x=5.0, car_rear_x=7.5, until=1.8, speed=0.0):
    nearest, _ = race.racing_math.find_nearest_index(d.xy, (ego_x, 0.0), closed=True)
    for now in np.arange(0, until, .025):
        d.observe([ego_x, 0, 0, speed, 0], stopped_car_scan(d, ego_x, car_rear_x), float(now), 0)
        d._update_opponent_and_overtake(nearest)
    return nearest


def test_pass_planner_needs_a_speed_from_behind_then_commits_on_a_checked_path():
    walls = dict(at=lambda x, y: 1.5 - abs(y))
    d = planner_decisions(walls)
    follow_stopped_car(d, until=0.2)
    # Node spec: confirmation (six sightings over 0.15 s) and then a speed
    # measured from behind over at least 0.3 s, before any pass.
    assert d._pass_state == 'idle' and not d.overtake_active and d._opp_speed is None
    follow_stopped_car(d, until=0.8)
    assert d._opp_speed == pytest.approx(0.0, abs=1e-6)   # m/s: it is not moving
    assert d._pass_state == 'pass' and d.overtake_active
    assert d._pass_plan.mode == 'pass'
    # The pass replaces the stage-1 cap only with its own checked stop speed.
    assert 0.3 <= d.follow_speed_cap <= float(np.max(d.speed_profile))
    assert d.last_opponent_status.startswith('overtake active: passing on the')


def test_pass_planner_never_commits_in_a_corridor_too_narrow_for_two_cars():
    walls = dict(at=lambda x, y: 0.80 - abs(y))
    d = planner_decisions(walls)
    follow_stopped_car(d, until=3.0)
    assert d._pass_state == 'idle' and not d.overtake_active
    assert 'no checked pass' in d.last_opponent_status


def started_pass(walls):
    d = planner_decisions(walls)
    nearest = follow_stopped_car(d, until=0.8)
    assert d._pass_state == 'pass'
    return d, nearest


EGO_PAST = (10.5, -0.75)   # past the stopped car (rear 7.5 m), in the right-hand lane


def abort_once(d, ego, now=1.9, speed=2.0):
    d.overtake_max_duration_sec = 0.0          # forces the abort this tick
    # Last sighting 1.05 s ago: past max_unseen, so no rejoin is authorised
    # and the abort (not the return) is what this tick decides.
    d._opp_seen = now - 1.05
    nearest, _ = race.racing_math.find_nearest_index(d.xy, ego, closed=True)
    d.observe([ego[0], ego[1], 0, speed, 0], np.full(1081, 10.0), now, 0)
    d._update_opponent_and_overtake(nearest)


@pytest.mark.parametrize('branch, walls_at, plan_mode', [
    # Lane held clear: brake in it.
    ('brake_in_lane', lambda x, y: 1.5 - abs(y), 'hold'),
    # The lane (y < -0.62) walled off from 1.9 m ahead: inside a 2 m/s stop
    # (1.57 m + body) holding the lane, but a 2.9 m shift back has the body
    # inside y > -0.55 by then (quintic: d(1.43 m) = -0.39 m).
    ('checked_rejoin', lambda x, y: (0.05 if (x > EGO_PAST[0] + 1.9 and y < -0.62) else 1.5 - abs(y)),
     'return'),
    # Walled off ahead everywhere: no checked path, hold the steering.
    ('zero_stop', lambda x, y: (0.0 if x > EGO_PAST[0] + 0.6 else 1.5 - abs(y)), None),
])
def test_an_abort_brakes_to_zero_on_the_first_checked_path_and_keeps_the_car(branch, walls_at, plan_mode):
    # Approved order (2026-11-08): lane, then checked rejoin, then hold steering.
    walls = dict(at=lambda x, y: 1.5 - abs(y))
    d, _ = started_pass(walls)
    anchor, arc = d._opp_anchor, d.opponent.arc_length
    walls['at'] = walls_at
    abort_once(d, EGO_PAST)
    assert d._pass_state == 'abort' and d.overtake_active
    assert d._pass_branch == branch
    assert d.follow_speed_cap == 0.0
    assert (d._pass_plan.mode if d._pass_plan is not None else None) == plan_mode
    assert d._pass_hold_steering is (branch == 'zero_stop')
    # Never forget the car on an abort.
    assert d._opp_anchor == anchor and d.opponent.arc_length == arc
    assert d.last_opponent_status.startswith('overtake abandoned:')


def test_no_rejoin_without_a_sighting_in_the_last_second_however_far_ahead():
    walls = dict(at=lambda x, y: 1.5 - abs(y))
    d, _ = started_pass(walls)
    # Far past the car (rear 7.5 m), offset in the passing lane, but the last
    # consistent sighting is 1.2 s old: the approved bound needs <= 1.0 s.
    ego_x = 14.0
    nearest, _ = race.racing_math.find_nearest_index(d.xy, (ego_x, -0.75), closed=True)
    d._opp_seen = 0.6
    d.observe([ego_x, -0.75, 0, 1.0, 0], np.full(1081, 10.0), 1.8, 0)
    d._update_opponent_and_overtake(nearest)
    assert d._pass_state != 'return'
    # The same place with a fresh sighting is authorised.
    d2, _ = started_pass(walls)
    d2._opp_seen = 1.8
    d2.observe([ego_x, -0.75, 0, 1.0, 0], np.full(1081, 10.0), 1.8, 0)
    d2._update_opponent_and_overtake(nearest)
    assert d2._pass_state == 'return'
    assert d2.last_opponent_status.startswith('overtake rejoining')


def test_no_pass_is_started_on_a_car_that_is_still_speeding_up():
    # A race start: the car ahead pulls away at 2 m/s^2 from rest. Oracle:
    # for constant acceleration the two half-window mean speeds differ by
    # exactly a times the time between their midpoints, so the estimate is
    # 2 m/s^2 > the 1.5 m/s^2 gate at every window length, and no pass may
    # start on that stale speed.
    walls = dict(at=lambda x, y: 1.5 - abs(y))
    d = planner_decisions(walls)
    nearest, _ = race.racing_math.find_nearest_index(d.xy, (5.0, 0.0), closed=True)
    for now in np.arange(0, 1.0, .025):
        rear = 7.5 + 0.5 * 2.0 * float(now) ** 2
        d.observe([5.0, 0, 0, 0.0, 0], stopped_car_scan(d, 5.0, rear), float(now), 0)
        d._update_opponent_and_overtake(nearest)
        assert d._pass_state == 'idle' and not d.overtake_active
    assert d._opp_accelerating


def test_mid_pass_a_lost_car_keeps_its_measured_speed_bound():
    # Review 2026-11-08: losing sight must not turn its speed into zero.
    walls = dict(at=lambda x, y: 1.5 - abs(y))
    d, nearest = started_pass(walls)
    d._opp_speed = 1.0                       # as measured from behind
    bound = d._pass_opponent(1.0).speed_bound
    lost = 0.8 + d.opponent_lost_timeout_sec + 0.5
    d.observe([5.0, -0.5, 0, 0.5, 0], np.full(1081, 10.0), lost, 0)
    d._update_opponent_and_overtake(nearest)
    assert d._opp_speed == 1.0
    assert d._pass_opponent(lost).speed_bound == pytest.approx(bound, abs=1e-12)
    # Idle (a corridor too narrow to pass in), a car reacquired after being
    # lost starts a fresh speed estimate.
    idle = planner_decisions(dict(at=lambda x, y: 0.80 - abs(y)))
    follow_stopped_car(idle, until=0.5)
    assert idle._opp_speed is not None
    back = 0.5 + idle.opponent_lost_timeout_sec + 0.1
    for now in np.arange(back, back + 0.2, 0.025):   # re-confirmed, under 0.3 s of history
        idle.observe([5.0, 0, 0, 0.0, 0], stopped_car_scan(idle, 5.0, 7.5), float(now), 0)
        idle._update_opponent_and_overtake(nearest)
    assert idle._opp_speed is None and idle._pass_state == 'idle'


def test_a_localization_jump_mid_pass_stops_like_the_node_and_keeps_the_pass_state():
    # Node order: beyond off_racing_line_recovery_max_error the hard watchdog
    # stop returns before the planner, so the planner state is untouched.
    walls = dict(at=lambda x, y: 1.5 - abs(y))
    d, _ = started_pass(walls)
    follower = race.rv.PathFollower(straight_stadium(), law_name='delay_lqr', car_id='rb2')
    # Beyond recovery range from the line AND from the pass path (~0.8 m off it).
    far = [9.0, -(d.off_racing_line_recovery_max_error + 1.5), 0, 1.0, 0]
    d.observe(far, np.full(1081, 10.0), 1.9, 0)
    steering, speed, *_, state = d.base_command(follower, far, 0.1)
    assert speed == 0.0 and state == 'off_racing_line'
    assert d._pass_state == 'pass'


# Both lanes walled 0.45 m off the line from x = 6.6 m on: no pass and no
# carrying on in a lane, but a path back to the line is clear of the walls.
def _lanes_walled(x, y):
    return 0.05 if (x > 6.6 and abs(y) > 0.45) else 1.5 - abs(y)


def stopped_in_abort(walls_at):
    walls = dict(at=lambda x, y: 1.5 - abs(y))
    d, nearest = started_pass(walls)
    limit = d.overtake_max_duration_sec
    walls['at'] = walls_at
    ego = (5.5, -0.4)
    nearest, _ = race.racing_math.find_nearest_index(d.xy, ego, closed=True)
    d.overtake_max_duration_sec = 0.0
    d.observe([*ego, 0, 0.0, 0], stopped_car_scan(d, 5.5, 7.5), 0.9, 0)   # still seen
    d._update_opponent_and_overtake(nearest)
    d.overtake_max_duration_sec = limit
    assert d._pass_state == 'abort'
    return d, nearest, ego


def tick_unseen(d, nearest, ego, now):
    d.observe([*ego, 0, 0.0, 0], np.full(1081, 10.0), now, 0)
    d._update_opponent_and_overtake(nearest)


def test_a_stopped_abort_hands_a_car_lost_ahead_back_to_following_after_the_lost_timeout():
    # Spec (node comment): a car last seen ahead of our nose cannot come back
    # beside us unseen; once lost for opponent_lost_timeout_sec, and with a
    # wall-clear path back to the line, the manoeuvre ends.
    d, nearest, ego = stopped_in_abort(_lanes_walled)
    lost = d.opponent_lost_timeout_sec
    tick_unseen(d, nearest, ego, 0.9 + lost - 0.05)            # one tick short: not lost yet
    assert d._pass_state == 'abort' and d.follow_speed_cap == 0.0
    tick_unseen(d, nearest, ego, 0.9 + lost + 0.05)
    assert d._pass_state == 'idle' and not d.overtake_active
    assert d.last_opponent_status.startswith('overtake ended')


def test_a_stopped_abort_keeps_a_lost_car_when_the_way_back_is_walled():
    walled = lambda x, y: 0.0 if x > 6.6 else 1.5 - abs(y)
    d, nearest, ego = stopped_in_abort(walled)
    tick_unseen(d, nearest, ego, 0.9 + d.opponent_lost_timeout_sec + 0.5)
    assert d._pass_state == 'abort' and d.follow_speed_cap == 0.0


def test_a_stopped_abort_keeps_a_car_last_seen_beside_it():
    # Beside us (its rear behind our nose) it may sit in the rear-flank blind
    # spot: losing it is no reason to forget it.
    d, nearest, ego = stopped_in_abort(_lanes_walled)
    d._opp_anchor = (d._opp_anchor[0] - 2.5, *d._opp_anchor[1:])   # rear now level with us
    tick_unseen(d, nearest, ego, 0.9 + d.opponent_lost_timeout_sec + 0.5)
    assert d._pass_state == 'abort'


def test_a_stopped_abort_carries_on_in_a_clear_lane_when_the_way_back_is_blocked():
    # Car 2 stopped in its lane (y = -0.75) just past the stopped car (rear
    # 7.5 m, front 8.12 m; car 2's rear 8.29 m), so it is no pass target, and
    # the racing line ahead is walled off, so there is no way back to it. Only
    # carrying on in the lane, which is clear, can move car 2.
    line_blocked = lambda x, y: 0.0 if (x > 9.2 and y > -0.45) else 1.5 - abs(y)
    ego = (8.4, -0.75)
    nearest, _ = race.racing_math.find_nearest_index(straight_stadium().xy, ego, closed=True)

    def stop_there(walls_at):
        walls = dict(at=lambda x, y: 1.5 - abs(y))
        d, _ = started_pass(walls)
        walls['at'] = walls_at
        limit = d.overtake_max_duration_sec
        d.overtake_max_duration_sec = 0.0
        d.observe([*ego, 0, 0.0, 0], np.full(1081, 10.0), 0.85, 0)
        d._update_opponent_and_overtake(nearest)
        d.overtake_max_duration_sec = limit
        return d
    d = stop_there(line_blocked)
    assert d._pass_state == 'pass' and d._pass_plan.mode == 'pass'
    assert d._pass_plan.d[0] == pytest.approx(-0.75, abs=1e-3)      # m: in its own lane
    assert d.follow_speed_cap >= 0.3
    # The lane also walled 1.2 m ahead of the nose: a stop from rest is
    # clear, but a crawl along it is not, so it stays stopped.
    both = lambda x, y: 0.0 if (x > ego[0] + 0.47 + 1.2 and y < -0.45) else line_blocked(x, y)
    assert stop_there(both)._pass_state == 'abort'


def flank_scan(d, bearing_deg, rng):
    """An unexplained return (expected map range is 10 m) at one bearing, e.g. a car beside us."""
    scan = np.full(1081, 10.0)
    angles = d.lidar.angle_min + np.arange(1081) * d.lidar.angle_increment
    scan[np.abs(angles - math.radians(bearing_deg)) < 0.15] = rng
    return scan


def test_a_stopped_abort_keeps_the_car_while_its_flank_shows_an_unconfirmed_return():
    # Review 2026-11-09: no confirmed sighting is not evidence of an empty
    # flank (a car can fail confirmation, e.g. reversing, or another object
    # can replace the anchor). A return 0.6 m off at +80 deg, on the side
    # car 2 (right of the line) would rejoin toward, keeps the abort and its
    # zero cap however long the anchor has been stale.
    d, nearest, ego = stopped_in_abort(_lanes_walled)
    for now in np.arange(1.0, 0.9 + 3 * d.opponent_lost_timeout_sec, 0.1):
        d.observe([*ego, 0, 0.0, 0], flank_scan(d, 80.0, 0.6), float(now), 0)
        d._update_opponent_and_overtake(nearest)
        assert d._pass_state == 'abort' and d.follow_speed_cap == 0.0
    # The same return on the far side (-80 deg) is not across the rejoin.
    d2, nearest, ego = stopped_in_abort(_lanes_walled)
    for now in np.arange(1.0, 0.9 + d2.opponent_lost_timeout_sec + 0.2, 0.1):
        d2.observe([*ego, 0, 0.0, 0], flank_scan(d2, -80.0, 0.6), float(now), 0)
        d2._update_opponent_and_overtake(nearest)
    assert d2._pass_state == 'idle'


def test_the_flank_must_stay_clear_for_the_whole_lost_timeout():
    d, nearest, ego = stopped_in_abort(_lanes_walled)
    lost = d.opponent_lost_timeout_sec
    gone = 0.9 + 2 * lost                    # the return leaves here
    for now in np.arange(1.0, gone, 0.1):
        d.observe([*ego, 0, 0.0, 0], flank_scan(d, 60.0, 0.8), float(now), 0)
        d._update_opponent_and_overtake(nearest)
    tick_unseen(d, nearest, ego, gone)
    tick_unseen(d, nearest, ego, gone + lost - 0.05)
    assert d._pass_state == 'abort'          # clear, but not for long enough yet
    tick_unseen(d, nearest, ego, gone + lost + 0.05)
    assert d._pass_state == 'idle'


def test_a_nan_or_too_short_beam_in_the_flank_is_not_seen_clear():
    d, nearest, ego = stopped_in_abort(_lanes_walled)
    d.observe([*ego, 0, 0.0, 0], np.full(1081, 10.0), 2.0, 0)
    assert d._pass_flank_clear(True)
    for bad in (math.nan, 0.01):              # NaN; under range_min (0.05 m)
        d.observe([*ego, 0, 0.0, 0], flank_scan(d, 80.0, bad), 2.0, 0)
        assert not d._pass_flank_clear(True)
    # A return beyond the 2 m reach is not across the rejoin.
    d.observe([*ego, 0, 0.0, 0], flank_scan(d, 80.0, 2.5), 2.0, 0)
    assert d._pass_flank_clear(True)
