"""Plan Task 12, map racer: the race-day flank stop reproduces, flank creep leaves it, and the arc braking cap
keeps the rejoin off the next wall.

Recorded measurement: rb2 run 20261111-152132 stopped for 30.5 s (19.0-49.5 s) with a wall 2.5 cm off its right
flank and the road open ahead (fixture src/pure_pursuit/test/data/flank_stop_rb2_20261111-152132.npz: finite poses at
(-7.02, 9.27), heading 1.68 rad; its scans give the body 0.021 m of clearance, run_validation._footprint_clearance,
median). Car 2's race-day map draws that wall about 7 cm closer than the scans saw it: at the recorded pose the
simulated body overlaps it by 0.04 m. So the car is placed 0.05 m to its left, where the simulated scan gives the
recorded clearance (checked below), and the scan state, not the map pose, is what is reproduced.

Oracle (the plan's, with the plant's own body-against-map check, never gym's flag): with flank creep off the map
racer stays stopped there; with the shipped config (flank_creep_length 0.5, enable_arc_braking_cap true) it moves off
within 3 s with no chassis contact. With the cap off it reaches a wall: the hazard the cap exists for
(tools/run_analysis/results/2026-11-14-map-racer-stuck-case.md). Run by the tool-suite command in CLAUDE.md.
"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import eval_gap  # noqa: E402  (bootstraps gym and the simulator path)
import run_validation as rv  # noqa: E402

FIXTURE = os.path.join(HERE, '..', '..', '..', 'src', 'pure_pursuit', 'test', 'data', 'flank_stop_rb2_20261111-152132.npz')
TRACK = 'car2_course_2026_11_11'
CAR = 'rb2'
SHIFT_LEFT = 0.05      # m, see the module docstring
SEEDS = (12345, 12346)


def start_pose():
    poses = np.load(FIXTURE)['poses']
    x, y, yaw = np.median(poses[np.isfinite(poses[:, 0])], axis=0)
    yaw = math.remainder(yaw, 2 * math.pi)
    return np.array([[x - SHIFT_LEFT * math.sin(yaw), y + SHIFT_LEFT * math.cos(yaw), yaw]])


def recorded_body_clearance():
    rv.select_profile(rv.PROFILE.name, CAR)
    data = np.load(FIXTURE)
    scans = data['ranges'][np.isfinite(data['poses'][:, 0])]
    return float(np.median([rv._footprint_clearance(s, rv.PURE_PURSUIT_PARAMS) for s in scans]))


def drive(seed, **overrides):
    """The map racer from rest at start_pose() for 3 s: (metres moved, contact steps, min chassis clearance,
    the body clearance its first scans give)."""
    rv.select_profile(rv.PROFILE.name, CAR)
    plant = rv.make_plant(eval_gap.track_path(TRACK), 1, seed, car_id=CAR)
    try:
        plan = rv.PathPlan.from_track(plant.env.unwrapped.track, car_id=CAR)
        follower = rv.PathFollower(plan, car_id=CAR)
        shaper = rv.CommandShaper(CAR, None, None)
        params = rv.PURE_PURSUIT_PARAMS
        saved = {k: params[k] for k in overrides}
        params.update(overrides)
        try:
            obs, _ = plant.reset(start_pose())
            for _ in range(4):                                  # the plant's scan delay line fills
                obs, *_ = plant.step([(0.0, 0.0)])
            first_body = rv._footprint_clearance(obs['agent_0']['scan'], params)
            x0, y0 = plant.truth(0)[:2]
            clearance = math.inf
            for _ in range(int(round(3.0 / rv.CONTROL_DT))):
                state = obs['agent_0']['std_state']
                steering, speed, *_ = follower.command(state, previous_steering=shaper.previous_steering)
                steering, speed, result = rv.apply_map_safety(
                    obs['agent_0']['scan'], steering, speed, car_id=CAR, applied_steering=shaper.previous_steering)
                if result in ('escape', 'creep'):
                    cap = min(shaper.max_speed, float(params['emergency_escape_speed']))
                else:
                    cap = shaper.max_speed
                if result == 'none':
                    arc_cap = rv.map_arc_braking_cap(obs['agent_0']['scan'], steering, shaper.previous_steering, CAR)
                    if arc_cap < speed:
                        speed, cap = arc_cap, min(cap, arc_cap)
                steering, speed = shaper.command(steering, speed, cap, measured_speed=state[3])
                obs, *_ = plant.step([(steering, speed)])
                c = plant.chassis_map_clearance(0)
                if math.isfinite(c):
                    clearance = min(clearance, c)
        finally:
            params.update(saved)
        moved = math.hypot(*(plant.truth(0)[:2] - np.array([x0, y0])))
        return moved, plant.report().get('contact_steps', 0), clearance, first_body
    finally:
        plant.close()


def test_the_start_reproduces_the_recorded_body_clearance():
    # oracle: recorded measurement, the fixture's scans (0.021 m); within 5 mm.
    _moved, _contact, _clr, first_body = drive(SEEDS[0], flank_creep_length=0.0)
    assert first_body == pytest.approx(recorded_body_clearance(), abs=0.005)   # m


@pytest.mark.parametrize('seed', SEEDS)
def test_with_flank_creep_off_the_map_racer_latches_as_on_race_day(seed):
    moved, contact, _clr, _ = drive(seed, flank_creep_length=0.0)
    assert moved < 0.05          # m: stayed stopped
    assert contact == 0


@pytest.mark.parametrize('seed', SEEDS)
def test_the_shipped_config_leaves_the_wall_without_contact(seed):
    assert rv._pure_pursuit_params(CAR)['flank_creep_length'] > 0.0, 'precondition: shipped creep on'
    assert rv._pure_pursuit_params(CAR)['enable_arc_braking_cap'] is True, 'precondition: shipped cap on'
    moved, contact, clearance, _ = drive(seed)
    assert moved > 0.5           # m within 3 s
    assert contact == 0 and clearance > 0.0


@pytest.mark.parametrize('seed', SEEDS)
def test_without_the_arc_braking_cap_the_rejoin_reaches_a_wall(seed):
    # The hazard the cap exists for; if this stops failing over, the scenario no longer tests the cap.
    _moved, contact, clearance, _ = drive(seed, enable_arc_braking_cap=False)
    assert contact > 0 or clearance <= 0.0


# --- The mirror's flank-creep gate on synthetic scans (the node's own cases, src/pure_pursuit/test/test_flank_creep_node.py) ---

def wall_right_scan(distance):
    """Open to 6 m, a straight wall ``distance`` m to the right of the LiDAR, parallel to the car."""
    rv.select_profile(rv.PROFILE.name, CAR)
    lidar = rv._lidar()
    angles = lidar.angle_min + np.arange(lidar.num_beams) * lidar.angle_increment
    s = -np.sin(angles)
    return np.minimum(6.0, np.where(s > 1e-6, distance / np.maximum(s, 1e-6), 6.0))


def body_half_width():
    return float(rv.PURE_PURSUIT_PARAMS['car_width']) / 2.0


def test_the_mirror_creeps_from_a_wall_alongside_that_the_body_is_not_touching():
    # 0.18 m from the LiDAR: 2.5 cm off the 0.31 m body, the recorded stop. Steering straight along it.
    _steering, speed, result = rv.apply_map_safety(wall_right_scan(0.18), 0.0, 2.0, car_id=CAR)
    assert result == 'creep'
    assert speed == pytest.approx(float(rv.PURE_PURSUIT_PARAMS['emergency_escape_speed']), abs=1e-12)  # m/s


def test_the_mirror_stops_with_creep_off():
    scan = wall_right_scan(0.18)                    # selects the profile first: it rebuilds the parameter dict
    params = rv.PURE_PURSUIT_PARAMS
    saved = params['flank_creep_length']
    params['flank_creep_length'] = 0.0
    try:
        assert rv.apply_map_safety(scan, 0.0, 2.0, car_id=CAR)[1:] == (0.0, 'stop')
    finally:
        params['flank_creep_length'] = saved


def test_the_mirror_stops_when_the_wall_touches_the_body():
    # inside the half width: body clearance below zero
    assert rv.apply_map_safety(wall_right_scan(body_half_width() - 0.005), 0.0, 2.0, car_id=CAR)[1:] == (0.0, 'stop')


def test_the_mirror_stops_when_the_wall_is_inside_the_creep_margin():
    # 5 mm off the body: not touching (tier 0 fires at <= 3 cm), but inside the 1 cm flank_creep_margin.
    assert float(rv.PURE_PURSUIT_PARAMS['flank_creep_margin']) > 0.005, 'precondition'
    result = rv.apply_map_safety(wall_right_scan(body_half_width() + 0.005), 0.0, 2.0, car_id=CAR)
    assert result[1:] == (0.0, 'stop')


def test_the_mirror_stops_when_steering_toward_the_wall():
    assert rv.apply_map_safety(wall_right_scan(0.18), -0.3, 2.0, car_id=CAR)[1:] == (0.0, 'stop')
