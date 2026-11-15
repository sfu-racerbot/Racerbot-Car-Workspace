"""event_replay.py: the plant starts where the recording says and the harness reports what it should. Run by the tool-suite command in CLAUDE.md."""
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src', 'gap_follow'))
import event_replay  # noqa: E402
import eval_gap  # noqa: E402


def test_the_reference_start_is_the_recorded_pose_and_speed():
    ref = event_replay.real_reference()
    # recorded measurement: fixture steer_into_wall_rb1_20261111-155542.npz at 1.106 s = clock 85.606 s
    assert ref['pose'][0] == pytest.approx(-7.358, abs=0.005) and ref['pose'][1] == pytest.approx(9.138, abs=0.005)     # m
    assert ref['speed'] == pytest.approx(0.358, abs=0.002)                                                             # m/s
    assert np.all(np.isfinite(ref['pose'])) and ref['t'][0] == pytest.approx(0.0, abs=1e-9) and ref['t'][-1] <= 3.0


def test_the_plant_starts_at_the_given_pose_and_speed(request):
    calibration = eval_gap.calibration_for_car('rb1')
    plant = eval_gap.make_plant(event_replay.TRACK, 12345, calibration=calibration, car_id='rb1')
    pose = np.array([[-7.358, 9.138, 1.738]])
    plant.reset(pose, speeds=[0.358])
    plant.step([(0.0, 0.0)])
    truth = plant.truth(0)
    # oracle: one 25 ms tick at 0.358 m/s moves the car 9 mm; the brief's bound is 0.02 m and 0.02 rad
    assert math.hypot(truth[0] - pose[0, 0], truth[1] - pose[0, 1]) < 0.02
    assert abs(math.remainder(truth[4] - pose[0, 2], 2 * math.pi)) < 0.02
    assert plant.chassis_map_clearance(0) >= 0.2                       # the start is in free space, not against a wall
    plant.close()


def test_a_run_reports_speeds_clearance_and_the_trace(tmp_path):
    data = np.load(event_replay.FIXTURE)
    params = json.loads(str(data['params_json']))
    ref = event_replay.real_reference(data)
    r = event_replay.run_once(12345, False, ref, params)
    assert r['steps'] == 120 and len(r['trace']['v']) == 120
    assert r['start_clearance'] >= 0.2
    assert r['v_07'] > 0.5 and r['min_clearance'] > 0.0 and not r['contact']
    assert r['max_v_minus_cmd'] == pytest.approx(max(v - c for v, c in zip(r['trace']['v'], r['trace']['cmd'])), abs=1e-9)
    # the first tick is at the recorded pose and speed (the harness starts where the recording says)
    assert math.hypot(r['trace']['x'][0] - ref['pose'][0], r['trace']['y'][0] - ref['pose'][1]) < 0.02      # m
    assert r['trace']['v'][0] == pytest.approx(ref['speed'], abs=0.02)                                       # m/s


# --- warm start (2026-11-14) -----------------------------------------------------------------------------------------
def _fixture_warm(data, ref):
    """warm_up inputs built from the fixture alone (its scans before the start, its wheel speed and gyro)."""
    t = np.asarray(data['times'], dtype=float)
    i0 = ref['start_index']
    yaw = np.asarray(data['yaw_rates'], dtype=float)
    ok = np.isfinite(yaw)
    return dict(scans=[data['ranges'][j] for j in range(i0)], scan_t=t[:i0],
                speed_at=lambda s: float(np.interp(s, t, data['odom_speeds'])), gyro_t=t[ok], gyro_wz=yaw[ok],
                press=float(t[0]) - 0.025, start_t=float(t[i0]))


def test_warm_up_feeds_every_scan_on_the_nodes_clock():
    # Oracle: steer_stages.replay_window's clock, step = (t - press) / CONTROL_DT, one controller call per recorded scan.
    data = np.load(event_replay.FIXTURE)
    params = json.loads(str(data['params_json']))
    ref = event_replay.real_reference(data)
    w = _fixture_warm(data, ref)
    calls = []

    class Spy:
        last_command_time = None

        def command(self, scan, v, step, yaw_rate=None):
            calls.append((step, v, yaw_rate))
            return 0.1, 1.0

    spy = Spy()
    out, last = event_replay.warm_up(spy, w['scans'], w['scan_t'], w['speed_at'], w['gyro_t'], w['gyro_wz'], w['press'])
    assert len(calls) == ref['start_index'] == 44
    assert [c[0] for c in calls] == pytest.approx([(s - w['press']) / eval_gap.CONTROL_DT for s in w['scan_t']], abs=1e-9)
    assert spy.last_command_time == pytest.approx(calls[0][0] - 1.0, abs=1e-9)
    assert last == calls[-1][0] and out == (0.1, 1.0)
    assert calls[0][1] == pytest.approx(float(data['odom_speeds'][0]), abs=1e-9)        # the wheel speed the node had
    assert params['wheelbase'] > 0                                                        # the fixture's params load


def test_a_warm_run_plays_the_recording_through_the_plants_empty_first_scans_only():
    # The plant's first two scans after reset are zeros (its LiDAR delay line). A warm run plays the recorded /drive
    # there instead of feeding the warmed controller zeros; a cold run keeps feeding the controller (old behaviour).
    data = np.load(event_replay.FIXTURE)
    params = json.loads(str(data['params_json']))
    ref = event_replay.real_reference(data)
    warm = event_replay.run_once(12345, True, ref, params, warm=_fixture_warm(data, ref), data=data)
    cold = event_replay.run_once(12345, True, ref, params)
    assert warm['open_loop_ticks'] == 2 and cold['open_loop_ticks'] == 0
    assert warm['trace']['cmd'][0] == pytest.approx(float(ref['cmd'][0]), abs=1e-9)      # the recorded command, 0.50 m/s
    assert warm['steps'] == cold['steps'] == 120
    assert math.isfinite(warm['first_command']['speed'])


def test_shared_speed_loop_restores_the_shared_values():
    from sim_fidelity.calibration import SHARED_SPEED_LOOP
    cal = event_replay.shared_speed_loop(eval_gap.calibration_for_car('rb1'))
    assert {k: getattr(cal, k) for k in SHARED_SPEED_LOOP} == SHARED_SPEED_LOOP
