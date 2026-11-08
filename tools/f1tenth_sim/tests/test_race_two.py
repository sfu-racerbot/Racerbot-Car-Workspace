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
