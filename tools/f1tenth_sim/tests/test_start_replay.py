"""start_replay's geometry helpers against closed forms (a circle of known radius).

Oracles: a counter-clockwise circle of radius 10 m centred on the origin, sampled every 0.5 degree, has arc
length 2*pi*10 and a point at radius r on the left of the direction of travel (inside) is r_line - r to the
left. Run by: python3 -m pytest tools/f1tenth_sim/tests/ -q -p no:cacheprovider
"""
import math
import sys
from pathlib import Path

import pytest

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM.parents[1] / 'tools'))
sys.path.insert(0, str(SIM.parents[1] / 'src' / 'pure_pursuit'))
import numpy as np  # noqa: E402
import start_replay as sr  # noqa: E402

R = 10.0


def circle():
    angle = np.radians(np.arange(0.0, 360.0, 0.5))
    return sr.line_geometry(np.column_stack([R * np.cos(angle), R * np.sin(angle)]))


def test_arc_length_and_projection_follow_the_circle():
    xy, seg, ds, arc = circle()
    assert ds.sum() == pytest.approx(2 * math.pi * R, abs=0.005)       # m, chord error of 0.5 deg steps
    s, off = sr.project(xy, seg, ds, arc, R * math.cos(math.radians(90)), R * math.sin(math.radians(90)))
    assert s == pytest.approx(R * math.pi / 2, abs=0.05)               # m, a quarter turn
    assert off == pytest.approx(0.0, abs=0.01)


def test_offset_is_positive_to_the_left_of_travel():
    xy, seg, ds, arc = circle()
    inside = sr.project(xy, seg, ds, arc, 9.0, 0.0)[1]               # nearer the centre: left of a CCW line
    outside = sr.project(xy, seg, ds, arc, 10.7, 0.0)[1]
    assert inside == pytest.approx(1.0, abs=0.01)    # m
    assert outside == pytest.approx(-0.7, abs=0.01)


def test_lap_times_are_the_gaps_between_start_finish_crossings():
    length = 100.0
    t = np.arange(0.0, 40.0, 0.025)
    arcs = (t * 10.0) % length                       # 10 m/s: a lap every 10 s
    laps = sr.lap_times(t, arcs, length)
    assert laps == pytest.approx([10.0, 10.0], abs=0.03)       # s; crossings at 10, 20, 30 s are two laps
    assert sr.lap_times(t[:300], arcs[:300], length) == []           # under one crossing: no lap


# --- option plumbing (2026-11-12) --------------------------------------------------------------
from dataclasses import replace  # noqa: E402
from sim_fidelity import calibration_for_car, with_floor  # noqa: E402
from sim_fidelity.calibration import sfu_2026_11_11_rb2_grip_curve  # noqa: E402
from sim_fidelity.grip import ProgressiveGrip  # noqa: E402

FLOOR = 'sfu_course_2026_11_11_rb2'


def test_defaults_leave_the_floor_untouched():
    assert sr.floor_calibration(FLOOR) == with_floor(FLOOR)
    assert sr.floor_calibration(FLOOR, None, None) == with_floor(FLOOR)


def test_grip_scale_replaces_the_floors_curve_with_the_measured_one_at_that_scale():
    cal = sr.floor_calibration(FLOOR, grip_scale=1.0)
    assert cal.grip_curve == sfu_2026_11_11_rb2_grip_curve(1.0)
    # Oracle: at scale 1.0 the delivered peak is the measured gyro-basis plateau, 2.94 m/s^2 (grip_truth.py).
    peak = max(d for _, d in cal.grip_curve)
    assert peak == pytest.approx(2.94, abs=0.005)       # m/s^2
    assert max(d for _, d in with_floor(FLOOR).grip_curve) == pytest.approx(2.94 * 1.15, abs=0.01)     # the floor's scale
    assert cal.grip_curve != with_floor(FLOOR).grip_curve
    assert replace(cal, grip_curve=with_floor(FLOOR).grip_curve) == with_floor(FLOOR)    # nothing else changed


def test_grip_scale_is_refused_on_a_floor_that_has_no_scalable_curve():
    with pytest.raises(ValueError, match='grip-scale'):
        sr.floor_calibration('sfu_course_2026_11_06', grip_scale=1.1)


def test_grip_coupling_reaches_the_plants_grip_object():
    import eval_gap
    cal = sr.floor_calibration(FLOOR, grip_coupling=0.5)
    assert cal.longitudinal_grip_coupling == 0.5
    plant = eval_gap.make_plant('car2_course_2026_11_11', 12345, cal)
    try:
        assert isinstance(plant.grip[0], ProgressiveGrip)
        assert plant.grip[0].longitudinal_coupling == 0.5
        assert plant.grip[0]._delivered == [d for _, d in with_floor(FLOOR).grip_curve]   # curve untouched
    finally:
        plant.close()


@pytest.mark.parametrize('bad', [-0.1, 1.5])
def test_grip_coupling_outside_zero_to_one_is_refused(bad):
    with pytest.raises(ValueError, match='grip-coupling'):
        sr.floor_calibration(FLOOR, grip_coupling=bad)


def test_online_lateral_reaches_the_controllers_shaper_limit():
    args = sr.parse_args(['--track', 't', '--floor', FLOOR, '--line', 'l', '--start=0,0,0', '--out', 'o'])
    assert args.lateral == 3.1                    # the race-day config's value, unchanged by default
    args = sr.parse_args(['--track', 't', '--floor', FLOOR, '--line', 'l', '--start=0,0,0', '--out', 'o',
                          '--online-lateral', '99'])
    kwargs = sr.solo_arguments(args.law, args.speed, args.lateral, 3, 0.42)
    assert kwargs['max_lateral_accel'] == 99.0
    import run_validation as rv
    shaper = rv.CommandShaper('rb2', kwargs['max_speed'], kwargs['max_lateral_accel'])
    assert shaper.max_lateral_accel == 99.0       # the object the run drives with


def test_summary_reports_speed_offset_and_where_the_run_ended():
    # A synthetic run round the 10 m circle at 3 m/s (wheel scale 1.0), 0.5 m outside the line, ending a
    # quarter turn round: arc length of the end = R * pi / 2, offset = -0.5 (to the right of a CCW line).
    from types import SimpleNamespace
    angle = np.radians(np.arange(0.0, 360.0, 0.5))
    plan = SimpleNamespace(xy=np.column_stack([R * np.cos(angle), R * np.sin(angle)]), speed=np.full(len(angle), 3.0))
    steps = 400
    theta = np.linspace(0.0, math.pi / 2, steps)
    trace = np.column_stack([np.arange(steps) * 0.025, (R + 0.5) * np.cos(theta), (R + 0.5) * np.sin(theta),
                             np.full(steps, 3.0), np.zeros(steps), np.full(steps, 3.1)])
    cal = SimpleNamespace(odom_speed_scale=1.0)
    result = dict(contact_steps=7, collision=True, passed=False, max_cross_track_m=0.5, laps=0.2)
    out = sr.summarise(cal, plan, trace, result, [5.0])
    assert out['seed_result']['end_arc'] == pytest.approx(R * math.pi / 2, abs=0.05)     # m
    assert out['seed_result']['end_offset_left'] == pytest.approx(-0.5, abs=0.01)        # m
    assert out['at'][5.0]['wheel_speed'] == pytest.approx(3.0, abs=1e-9)
    assert out['at'][5.0]['offset_left'] == pytest.approx(-0.5, abs=0.01)
    assert out['seed_result']['contact'] == 7


# --- starting already moving (2026-11-13) -----------------------------------------------------

def _dense(xy, step=0.005):
    """The closed polyline resampled every ``step`` m by linear interpolation, so a nearest-sample projection is exact to step/2."""
    xy = np.asarray(xy, float)
    closed = np.vstack([xy, xy[:1]])
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
    t = np.arange(0.0, d[-1], step)
    return np.column_stack([np.interp(t, d, closed[:, 0]), np.interp(t, d, closed[:, 1])])


def _where(xy, x, y):
    """(arc s, offset left, heading of the line there, deg) of (x, y), by projection onto the dense line."""
    g, seg, ds, arc = sr.line_geometry(_dense(xy))
    s, off = sr.project(g, seg, ds, arc, x, y)
    j = int(np.argmin(np.abs(arc - s)))
    return s, off, math.degrees(math.atan2(seg[j][1], seg[j][0]))


def test_pose_on_line_projects_back_to_the_requested_arc_offset_and_heading():
    xy, _, _, _ = circle()
    x, y, yaw_deg = sr.pose_on_line(xy, 8.0, right=0.10, heading_error_deg=0.5)
    s, off, tangent = _where(xy, x, y)
    assert s == pytest.approx(8.0, abs=0.02)              # m
    assert off == pytest.approx(-0.10, abs=0.02)          # m, right is negative left
    assert (yaw_deg - tangent) == pytest.approx(0.5, abs=0.5)     # deg


def test_the_plants_first_state_is_the_requested_position_speed_offset_and_heading():
    import eval_gap
    from pure_pursuit import racing_math
    xy = racing_math.load_xy_csv(
        '/home/bryan/racerbot-ws/src/pure_pursuit/test/data/course_20261111/raceline_raw.csv')
    x, y, yaw_deg = sr.pose_on_line(xy, 8.0, right=0.098, heading_error_deg=0.48)
    cal = sr.floor_calibration(FLOOR)
    plant = eval_gap.make_plant('car2_course_2026_11_11', 12345, cal)
    try:
        plant.reset(np.array([[x, y, math.radians(yaw_deg)]]), speeds=[3.42 / cal.odom_speed_scale])
        truth = plant.truth(0)
        s, off, tangent = _where(xy, truth[0], truth[1])
        heading = (math.degrees(truth[4]) - tangent + 180.0) % 360.0 - 180.0
        assert s == pytest.approx(8.0, abs=0.02)           # m, projected back independently
        assert -off == pytest.approx(0.098, abs=0.02)      # m right
        assert truth[3] * cal.odom_speed_scale == pytest.approx(3.42, abs=0.02)    # m/s on the wheels
        assert heading == pytest.approx(0.48, abs=0.5)     # deg
    finally:
        plant.close()


# --- --check-profile-alone (2026-11-13) -------------------------------------------------------
PLANNED_LINE = SIM / 'maps' / 'car2_course_2026_11_11_planned_line.csv'


def _check(line, extra):
    return sr.main(['--track', 'car2_course_2026_11_11', '--floor', FLOOR, '--line', str(line), '--check-profile-alone',
                    '--seeds', '1', '--check-laps', '1', *extra])


def test_a_slow_profile_passes_the_profile_alone_check(tmp_path, monkeypatch):
    monkeypatch.setenv('RACERBOT_CAR', 'rb2')
    # A deliberately slow plan (the race-day line's speeds times 0.6) is drivable with no online cap: exit 0.
    rows = np.loadtxt(PLANNED_LINE, delimiter=',', skiprows=1)
    rows[:, 2] *= 0.6
    slow = tmp_path / 'slow.csv'
    np.savetxt(slow, rows, delimiter=',', header='x,y,speed', comments='')
    assert _check(slow, ['--poses', '2']) == 0


def test_the_planned_race_day_profile_fails_the_profile_alone_check(capsys, monkeypatch):
    monkeypatch.setenv('RACERBOT_CAR', 'rb2')
    # RECORDED SIMULATOR MEASUREMENT and change-detector (not an oracle): 2026-11-13, floor sfu_course_2026_11_11_rb2,
    # the original-guard race-day line (est. 9.404 s), cap off, standing start at s=0: contact near s 7.5 in all
    # 6 seeds (6/6 at s 7.3-8.2 over 15 grip cells). If the plan or the floor changes this may legitimately flip.
    assert _check(PLANNED_LINE, ['--poses', '1']) == 1
    out = capsys.readouterr().out
    assert 'contact at s=' in out and 'FAIL' in out


# --- the floor's closed-loop calibration against the real 152132 (2026-11-13) --------------------

def test_cap_on_flying_lap_is_within_one_percent_faster_to_four_percent_slower_than_the_real_car(monkeypatch):
    monkeypatch.setenv('RACERBOT_CAR', 'rb2')
    # RECORDED MEASUREMENT: real flying laps of run 20261111-152132 are 10.05 s (race-day line, online cap 3.1).
    # The floor's admissible band (advisor's rule, 2026-11-13) is [9.95, 10.45] s: never more than 1% faster than
    # the car. Simulated here on the original-guard race-day line (est. 9.404 s), standing start, 3 laps, 3 seeds.
    laps = []
    for seed in (12345, 12346, 12347):
        cal, plan, trace, result = sr.trace_run('car2_course_2026_11_11', FLOOR, str(PLANNED_LINE), sr.pose_on_line(
            np.loadtxt(PLANNED_LINE, delimiter=',', skiprows=1)[:, :2], 0.0), seed, 80.0, 3, 0.42, 5.0, 3.1, 'delay_lqr')
        xy, seg, ds, arc = sr.line_geometry(plan.xy)
        s = np.array([sr.project(xy, seg, ds, arc, x, y)[0] for x, y in zip(trace[:, 1], trace[:, 2])])
        assert not result['collision'] and result['contact_steps'] == 0
        laps += sr.lap_times(trace[:, 0], s, float(ds.sum()))
    assert len(laps) >= 6
    assert 9.95 - 1e-9 <= float(np.median(laps)) <= 10.45, sorted(laps)


def test_launch_with_the_cap_off_makes_contact_before_the_second_bend_like_the_real_car_could_not(monkeypatch):
    # RECORDED SIMULATOR MEASUREMENT and change-detector (not an oracle), 2026-11-13: from the 152132 launch state
    # at s=8.1 m (3.42 m/s on the wheels, 0.098 m right, +0.48 deg) with the online cap off, the floor's car touches
    # at s 11.7-12.0 in 6/6 seeds. The real car, capped only by the reactive layer, reached 0.40 m right at s=14.
    monkeypatch.setenv('RACERBOT_CAR', 'rb2')     # rb2's race-day controller config (rb1's pure pursuit differs)
    from pure_pursuit import racing_math
    line_xy, _ = racing_math.load_profiled_csv(str(PLANNED_LINE))
    start = sr.pose_on_line(line_xy, 8.10, 0.098, 0.48)
    for seed in (12345, 12346):
        cal, plan, trace, result = sr.trace_run('car2_course_2026_11_11', FLOOR, str(PLANNED_LINE), start, seed, 40.0, 1,
                                                0.42, 5.0, 99.0, 'delay_lqr', start_wheel_speed=3.42)
        row = sr.summarise(cal, plan, trace, result, [])
        assert row['seed_result']['contact'] > 0
        assert 11.0 < row['seed_result']['end_arc'] < 13.0, row['seed_result']['end_arc']     # m


def test_cap_binding_counter_counts_ticks_where_the_online_cap_set_the_speed():
    import run_validation as rv
    with sr.cap_binding_counter() as stats:
        shaper = rv.CommandShaper('rb2', 5.0, 2.0)               # online lateral limit 2.0 m/s^2
        for _ in range(10):
            shaper.command(0.0, 4.0)                              # straight: no cap below the wanted speed
        for _ in range(10):
            shaper.command(0.35, 4.0)                             # tan(.35)/.36 = 0.97 /m: cap sqrt(2/.97) = 1.43
        for _ in range(5):
            shaper.command(0.35, 1.0)                             # wanted speed already below the cap
    assert stats == dict(ticks=25, binding=10)
    import run_validation
    assert not hasattr(run_validation.racing_math.curvature_speed_limit, '__wrapped__')
    assert run_validation.CommandShaper.command.__name__ == 'command' and 'cap_binding_counter' not in repr(
        run_validation.CommandShaper.command)                    # undone on exit


def test_understeer_compensation_steers_harder_and_caps_speed_by_the_delivered_curve_and_is_undone():
    # Oracle: racing_math's own closed forms (understeer_steering / delivered_curvature_speed_limit) at the table
    # rb2's gyro-basis grip gives (asked 0, 2.7, 3.25, 3.75 -> delivered 0, 2.7, 2.88, 2.94).
    import run_validation as rv
    from pure_pursuit import racing_math
    table = dict(asked=[0.0, 2.7, 3.25, 3.75], delivered=[0.0, 2.7, 2.88, 2.94])
    curve = racing_math.grip_curve_from_params(table['asked'], table['delivered'])
    original = (rv.PathFollower.command, rv.racing_math.curvature_speed_limit)
    with sr.understeer_compensation(table):
        assert rv.PathFollower.command is not original[0]
        k = 0.9                                              # 1/m
        assert rv.racing_math.curvature_speed_limit(k, 3.1, 5.0) == pytest.approx(
            racing_math.delivered_curvature_speed_limit(k, 3.1, 5.0, curve), abs=1e-12)    # m/s
        # 3.1 m/s^2 is more than this table ever delivers, so the cap is the last asked (3.75): above the plain one
        assert rv.racing_math.curvature_speed_limit(k, 3.1, 5.0) == pytest.approx(math.sqrt(3.75 / k), abs=1e-9)
        assert rv.racing_math.curvature_speed_limit(k, 2.5, 5.0) == pytest.approx(original[1](k, 2.5, 5.0), abs=1e-9)
    assert (rv.PathFollower.command, rv.racing_math.curvature_speed_limit) == original
    with sr.understeer_compensation(None):
        assert (rv.PathFollower.command, rv.racing_math.curvature_speed_limit) == original


# --- launch_lap_scale (2026-11-13) ----------------------------------------------------------

class _StubFollower:
    """Stands for run_validation.PathFollower: a fixed profiled speed of 2.0 m/s at any state."""
    def command(self, state, *a, **k):
        return 0.0, 2.0, 0, 0, 0.0


def _stub_follower(monkeypatch):
    monkeypatch.setattr(sr.rv, 'PathFollower', _StubFollower)
    monkeypatch.setattr(sr.rv, 'CONTROL_DT', 0.025)         # s, 40 Hz


def test_launch_scale_of_one_leaves_the_follower_alone(monkeypatch):
    _stub_follower(monkeypatch)
    before = _StubFollower.command
    with sr.launch_lap_scale(1.0, 5.0, 40.0) as ls:
        assert ls is None and _StubFollower.command is before
    assert _StubFollower.command is before


def test_launch_scale_multiplies_the_profiled_speed_for_the_distance_then_restores_it(monkeypatch):
    # Oracle: closed form. 2.0 m/s x 0.85 = 1.7 while metres remain; 0.15 m at 2.0 m/s x 0.025 s = 0.05 m a tick,
    # counted before the factor is read: 0.10 m and 0.05 m left (scaled), then 0 (unscaled).
    pytest.importorskip('pure_pursuit.launch_scale', reason='LaunchScale lands with the launch-lap-scale PR; '
                        'remove this skip when that is merged into this branch')
    _stub_follower(monkeypatch)
    follower = _StubFollower()
    state = np.array([0.0, 0.0, 0.0, 2.0, 0.0])
    with sr.launch_lap_scale(0.85, 0.15, 40.0) as ls:
        speeds = [follower.command(state)[1] for _ in range(3)]
        assert ls.remaining == 0.0
    assert speeds[0] == pytest.approx(1.7, abs=1e-12) and speeds[1] == pytest.approx(1.7, abs=1e-12)    # m/s
    assert speeds[2] == pytest.approx(2.0, abs=1e-12)
    assert follower.command(state)[1] == 2.0                  # restored on exit as well
