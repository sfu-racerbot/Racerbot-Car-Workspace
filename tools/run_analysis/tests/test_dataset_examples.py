"""The five dataset examples (tools/run_analysis/dataset_examples/), against closed forms.

Runner: python3 -m pytest tools/run_analysis/tests/ (the tool suites; no ROS needed).

Oracles: every expected value is a closed form computed here (a straight line, a
circle, a known delay or scale injected into synthetic signals), a cited spec
(the signed cross-track convention of src/pure_pursuit/pure_pursuit/racing_math.py
project_onto_path; the base_link <- laser offset of docs/hardware-reference.md), a
second, independent implementation (racing_math.project_onto_path for cross-track
error), or real log lines copied from rb2 20261111-152132's /rosout. Results on the
recorded runs are worked through in docs/dataset-examples.md.
"""
from pathlib import Path
import math
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_analysis.dataset_examples import common as c  # noqa: E402
from run_analysis.dataset_examples import (algorithm_accuracy as aa, lateral_acceleration as la,  # noqa: E402
                                           localization as lo, steering_response as sr,
                                           velocity_estimation as ve)
from pure_pursuit import racing_math  # noqa: E402

L = c.WHEELBASE
READY = ("pure_pursuit_node ready: 186 waypoints from '/home/racerbotcar-2/racerbot-ws/data/rb2/racerbot_auto/"
         "20261111-151026-rebuilt/raceline_profiled.csv' (closed loop), speed profile 1.79-5.00 m/s")
DRIVE = ("DRIVE [delay_lqr] pose=(-16.41,9.32,-81.0deg), nearest waypoint=8, steering target=17, cross-track=0.08m, "
         "lookahead=1.24m from fresh odometry=4.26m/s")


def rec(kind, rows):
    return np.rec.fromrecords(rows, names=c.COLUMNS[kind])


def synthetic_run(tmp_path, t, *, v, steer=None, wz=None, ax=None, ay=None, pose=None, pose_topic="/slam_pose",
                  intents=None, lb=None, drive=None):
    """A Run built from arrays, as c.load would return it. LB held throughout unless lb is given."""
    run = c.Run(path=tmp_path, car="rb9", run_id="20990101-000000")
    n = len(t)
    zero = np.zeros(n)
    run.series["/odom"] = rec("odometry", list(zip(t, zero, zero, zero, v, wz if wz is not None else zero)))
    if wz is not None:
        run.series["/imu/data"] = rec("imu", list(zip(t, zero, zero, wz, ax if ax is not None else zero,
                                                      ay if ay is not None else zero, zero + 9.8)))
    if steer is not None:
        run.series["/ackermann_cmd"] = rec("ackermann", list(zip(t, steer, v)))
        run.series["/drive"] = rec("ackermann", list(zip(t, steer, drive if drive is not None else v)))
    if pose is not None:
        run.series[pose_topic] = rec("pose", list(zip(t, *pose)))
    held = np.ones(n, dtype=int) if lb is None else lb
    run.series["/joy"] = rec("joy", list(zip(t, held)))
    run.intents = intents or []
    return run


# --------------------------------------------------------------------------- common: time

def test_on_grid_interpolates_linearly_and_never_invents_a_silence():
    t = np.array([0.0, 0.1, 0.2, 1.0, 1.1])
    x = 10.0 * t
    out = c.on_grid([0.05, 0.15, 0.5, 1.05, 1.2, -0.1], t, x, max_gap=0.1)
    assert out[:2] == pytest.approx([0.5, 1.5], abs=1e-12)
    assert np.isnan(out[2])                          # 0.3 s from either sample: a silence
    assert out[3] == pytest.approx(10.5, abs=1e-12)
    assert np.isnan(out[4]) and np.isnan(out[5])     # outside the samples
    assert np.isnan(c.on_grid([0.0, 1.0], [], [])).all()
    assert c.on_grid([0.1], [0.0, 0.2, np.nan], [1.0, 3.0, 5.0], 0.15)[0] == pytest.approx(2.0, abs=1e-12)


def test_derivative_of_a_ramp_is_its_slope_and_unknown_at_the_ends():
    x = 3.0 * np.arange(50) / c.FS                   # 3 m/s ramp sampled at FS
    d = c.derivative(x, c.FS, span_s=0.2)
    k = int(round(0.2 * c.FS / 2))
    assert d[k:-k] == pytest.approx(3.0, abs=1e-9)
    assert np.isnan(d[:k]).all() and np.isnan(d[-k:]).all()
    assert np.isnan(c.derivative([1.0, 2.0], c.FS)).all()


def test_moving_average_keeps_a_constant_and_spreads_a_gap():
    x = np.full(20, 2.5)
    assert np.nanmax(np.abs(c.moving_average(x, 5)[2:-2] - 2.5)) < 1e-12
    x[10] = np.nan
    out = c.moving_average(x, 5)
    assert np.isnan(out[8:13]).all() and out[7] == pytest.approx(2.5, abs=1e-12)


# --------------------------------------------------------------------------- common: poses

def straight_track(speed=2.0, seconds=5.0, rate=20.0):
    t = np.arange(0.0, seconds, 1.0 / rate)
    return t, speed * t, np.zeros_like(t), np.zeros_like(t)


def test_pose_jumps_finds_a_step_the_wheels_cannot_explain_and_nothing_else():
    t, x, y, yaw = straight_track()
    assert len(c.pose_jumps(t, x, y, yaw, t, np.full(len(t), 2.0))) == 0
    jumped = x.copy()
    jumped[40:] += 0.5                               # 0.5 m in one 0.05 s step at 2 m/s: 0.4 m too far
    assert c.pose_jumps(t, jumped, y, yaw, t, np.full(len(t), 2.0)).tolist() == [t[40]]
    small = x.copy()
    small[40:] += 0.25                               # 0.15 m beyond the wheels, under the 0.3 m tolerance
    assert len(c.pose_jumps(t, small, y, yaw, t, np.full(len(t), 2.0))) == 0
    turned = yaw.copy()
    turned[60:] += math.radians(30)                  # 30 deg in one step while the gyro reads 0
    assert c.pose_jumps(t, x, y, turned, t, np.full(len(t), 2.0), t, np.zeros(len(t))).tolist() == [t[60]]
    assert len(c.pose_jumps([0.0], [0.0], [0.0], [0.0], t, np.ones(len(t)))) == 0


def test_near_includes_the_pad_boundary():
    assert c.near([0.0, 0.5, 0.51, 2.0], [0.0], pad=0.5).tolist() == [True, True, False, False]


def test_rigid_fit_recovers_a_known_rotation_and_shift():
    rng = np.random.default_rng(1)
    a = rng.uniform(-10, 10, size=(200, 2))
    th = math.radians(10.0)
    r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    b = a @ r.T + [3.0, -2.0]
    fit = c.rigid_fit(a, b)
    assert fit["rotation_deg"] == pytest.approx(10.0, abs=1e-9)
    assert fit["residual_median_m"] == pytest.approx(0.0, abs=1e-9)
    assert fit["centre_shift_m"] == pytest.approx(float(np.hypot(*(b.mean(0) - a.mean(0)))), abs=1e-9)
    assert c.frame_verdict(float(np.median(np.hypot(*(a - b).T))), fit).startswith("frame offset")
    assert math.isnan(c.rigid_fit(a[:1], b[:1])["rotation_deg"])


def test_frame_verdict_calls_unexplained_disagreement_the_same_frame():
    rng = np.random.default_rng(2)
    a = rng.uniform(-10, 10, size=(500, 2))
    b = a + rng.normal(0, 0.1, size=a.shape)          # noise, no rotation or shift
    fit = c.rigid_fit(a, b)
    assert c.frame_verdict(float(np.median(np.hypot(*(a - b).T))), fit) == "same frame"
    assert c.frame_verdict(math.nan, fit) == "unknown"


def test_to_base_link_moves_a_laser_pose_back_along_its_heading():
    # docs/hardware-reference.md: the LiDAR is 0.26 m ahead of base_link on the car's x axis.
    x, y, yaw = c.to_base_link([1.0, 0.0], [0.0, 2.0], [math.pi / 2, 0.0], "laser")
    assert x == pytest.approx([1.0, -0.26], abs=1e-12)
    assert y == pytest.approx([-0.26, 2.0], abs=1e-12)
    assert yaw == pytest.approx([math.pi / 2, 0.0], abs=1e-12)
    same = c.to_base_link([1.0], [2.0], [0.3], "base_link")
    assert [float(a[0]) for a in same] == [1.0, 2.0, 0.3]
    with pytest.raises(ValueError, match="pose frame 'odom'"):
        c.to_base_link([1.0], [2.0], [0.3], "odom")


def test_forward_speed_of_a_circle_is_the_chord_speed_closed_form():
    v, radius = 3.0, 4.0
    w = v / radius
    t = np.arange(0.0, 6.0, 1.0 / c.FS)
    x, y, yaw = radius * np.sin(w * t), radius * (1 - np.cos(w * t)), w * t
    span = c.SPEED_SPAN_S
    k = int(round(span * c.FS / 2))
    half = w * k / c.FS                                # half the angle swept over the span
    out = c.forward_speed(x, y, yaw)
    assert out[k:-k] == pytest.approx(v * math.sin(half) / half, abs=1e-9)
    straight = c.forward_speed(2.0 * t, np.zeros_like(t), np.zeros_like(t))
    assert straight[k:-k] == pytest.approx(2.0, abs=1e-9)
    reversing = c.forward_speed(2.0 * t, np.zeros_like(t), np.full(len(t), math.pi))     # heading -x, moving +x
    assert reversing[k:-k] == pytest.approx(-2.0, abs=1e-9)
    sideways = c.forward_speed(2.0 * t, np.zeros_like(t), np.full(len(t), math.pi / 2))  # a sliding car
    assert sideways[k:-k] == pytest.approx(0.0, abs=1e-9)


def test_accel_axes_accepts_car_axes_and_names_a_rotated_mount():
    t = np.arange(0, 20, 0.02)
    forward, lateral = np.sin(t), np.cos(1.3 * t)
    ok = np.ones(len(t), dtype=bool)
    good = c.accel_axes(forward, lateral, forward, lateral, ok)
    assert good["mapping_ok"] and good["problem"] == ""
    rotated = c.accel_axes(lateral, -forward, forward, lateral, ok)    # car 2's BNO085 on 2026-11-11
    assert not rotated["mapping_ok"]
    assert rotated["problem"] == "ax follows +lateral (r +1.00); ay follows -forward (r -1.00)"
    flipped = c.accel_axes(-forward, -lateral, forward, lateral, ok)   # both VESC IMUs
    assert flipped["problem"] == "ax follows -forward (r -1.00); ay follows -lateral (r -1.00)"


# --------------------------------------------------------------------------- common: bookkeeping

def test_stats_and_line_fit_closed_forms_and_empty_input():
    s = c.stats([3.0, -4.0, np.nan])
    assert s["n"] == 2 and s["mean"] == pytest.approx(-0.5) and s["rms"] == pytest.approx(math.sqrt(12.5))
    assert s["max_abs"] == 4.0
    assert c.stats([])["n"] == 0 and math.isnan(c.stats([])["rms"])
    f = c.line_fit([0, 1, 2, 3], [1, 3, 5, 7])
    assert (f["slope"], f["intercept"]) == pytest.approx((2.0, 1.0), abs=1e-12) and f["r"] == pytest.approx(1.0)
    assert math.isnan(c.line_fit([1, 1, 1], [1, 2, 3])["slope"])


def test_intent_mask_follows_the_latest_intent_and_is_open_without_any(tmp_path):
    run = c.Run(path=tmp_path, car="rb9", run_id="x")
    assert c.intent_mask(run, [0.0, 1.0], lambda r: False).tolist() == [True, True]
    run.intents = [(1.0, {"severity": "drive"}), (2.0, {"severity": "stop"})]
    keep = c.intent_mask(run, [0.5, 1.0, 1.5, 2.0, 3.0], lambda r: r["severity"] != "stop")
    assert keep.tolist() == [False, True, True, False, False]


def test_results_never_go_inside_data(tmp_path):
    run = c.Run(path=tmp_path, car="rb1", run_id="20261026-194514")
    with pytest.raises(ValueError, match="never go inside data/"):
        c.out_dir(run, c.ROOT / "data" / "rb1")
    assert c.out_dir(run, tmp_path) == (tmp_path / "rb1-20261026-194514").resolve()


def test_local_path_and_node_params(tmp_path):
    assert c.local_path("/home/racerbotcar-2/racerbot-ws/data/rb2/x.csv") == c.ROOT / "data/rb2/x.csv"
    assert c.local_path("/elsewhere/x.csv") == Path("/elsewhere/x.csv")
    (tmp_path / "metadata" / "params").mkdir(parents=True)
    (tmp_path / "metadata" / "params" / "n.yaml").write_text("/n:\n  ros__parameters:\n    a: 1\n")
    assert c.node_params(tmp_path, "n") == {"a": 1}
    assert c.node_params(tmp_path, "missing") == {}


# --------------------------------------------------------------------------- localization

def test_body_delta_is_motion_in_the_starting_frame():
    dx, dy, dyaw = lo.body_delta(1.0, 1.0, math.pi / 2, 1.0, 3.0, math.pi)
    assert (dx, dy, float(dyaw)) == pytest.approx((2.0, 0.0, math.pi / 2), abs=1e-12)


def test_relative_pose_errors_zero_for_a_perfect_estimate_and_the_scale_of_a_scaled_one():
    t, x, y, yaw = straight_track(speed=2.0, seconds=20.0)
    s = x.copy()
    ok = np.ones(len(t), dtype=bool)
    same = lo.relative_pose_errors(s, (x, y, yaw), (x, y, yaw), ok, 5.0)
    assert same["n"] > 0 and same["translation_pct_p95"] == pytest.approx(0.0, abs=1e-9)
    scaled = lo.relative_pose_errors(s, (x, y, yaw), (1.05 * x, y, yaw), ok, 5.0)
    assert scaled["_translation"] == pytest.approx(5.0, abs=1e-9)          # 5 % long, every stretch


def test_relative_pose_errors_heading_drift_of_a_gyro_bias_on_a_straight():
    speed, bias = 2.0, math.radians(1.0)                                    # 1 deg/s heading drift
    t, x, y, yaw = straight_track(speed=speed, seconds=20.0)
    res = lo.relative_pose_errors(x, (x, y, yaw), (x, y, bias * t), np.ones(len(t), bool), 4.0)
    # A 4 m stretch at 2 m/s takes 2 s: 2 deg of heading, 0.5 deg/m.
    assert res["_heading"] == pytest.approx(math.degrees(bias) / speed, abs=1e-6)


def test_relative_pose_errors_skips_stretches_that_touch_a_bad_sample():
    t, x, y, yaw = straight_track(speed=2.0, seconds=20.0)
    ok = np.ones(len(t), dtype=bool)
    full = lo.relative_pose_errors(x, (x, y, yaw), (x, y, yaw), ok, 5.0)
    ok[200] = False                                                         # x = 20 m
    cut = lo.relative_pose_errors(x, (x, y, yaw), (x, y, yaw), ok, 5.0)
    starts = x[cut["_starts"]]
    assert cut["n"] < full["n"]
    assert not np.any((starts <= 20.0) & (starts + 5.0 >= 20.0))
    assert lo.relative_pose_errors(x, (x, y, yaw), (x, y, yaw), np.zeros(len(t), bool), 5.0)["n"] == 0


def test_dead_reckoning_a_perfect_estimate_retraces_the_reference():
    t = np.linspace(0, 10, 200)
    ref = (np.cos(t), np.sin(t), t + math.pi / 2)
    th = 0.7                                                               # odometry's own frame: rotated ...
    est = (math.cos(th) * ref[0] - math.sin(th) * ref[1] + 7.0,          # ... and moved, the same motion
           math.sin(th) * ref[0] + math.cos(th) * ref[1] - 3.0, ref[2] + th)
    i0 = 37                                                                # a heading on no axis
    x, y, yaw = lo.dead_reckon((ref[0][i0], ref[1][i0], ref[2][i0]), est, i0)
    assert x == pytest.approx(ref[0], abs=1e-9) and y == pytest.approx(ref[1], abs=1e-9)
    assert c.wrap(np.asarray(yaw) - ref[2]) == pytest.approx(0.0, abs=1e-9)   # headings modulo 2 pi


# --------------------------------------------------------------------------- velocity

def test_speed_errors_reports_a_scale_as_the_distance_ratio():
    ref = np.linspace(1.0, 4.0, 100)
    e = ve.speed_errors(1.05 * ref, ref, np.ones(100, bool))
    assert e["distance_ratio"] == pytest.approx(1.05, abs=1e-12)
    assert e["mean"] == pytest.approx(0.05 * ref.mean(), abs=1e-12)
    assert ve.speed_errors(ref, ref, np.zeros(100, bool))["n"] == 0


def test_velocity_analyse_finds_wheels_reading_high_against_the_map(tmp_path):
    t = np.arange(0.0, 30.0, 1.0 / c.FS)
    v_true = 2.0 + np.sin(0.5 * t)
    x = np.cumsum(v_true) / c.FS
    run = synthetic_run(tmp_path, t, v=1.08 * v_true, wz=np.zeros(len(t)), ax=0.5 * np.cos(0.5 * t),
                        pose=(x, np.zeros(len(t)), np.zeros(len(t))))
    res = ve.analyse(run)
    assert res["wheel"]["distance_ratio"] == pytest.approx(1.08, abs=0.002)   # 2 mm/s of derivative error
    assert res["accel_vs_wheel"]["slope"] == pytest.approx(1 / 1.08, abs=0.01)  # a_x = dv_true/dt = dv_wheel/dt / 1.08


# --------------------------------------------------------------------------- steering

def test_bicycle_model_and_its_inverse():
    v, delta = np.array([1.0, 2.0, 3.0]), np.array([0.1, -0.2, 0.3])
    r = sr.predicted_yaw_rate(v, delta)
    assert r == pytest.approx(v * np.tan(delta) / L, abs=1e-12)
    assert sr.effective_steering(v, r) == pytest.approx(delta, abs=1e-12)
    assert np.isnan(sr.effective_steering([0.4], [0.1])).all()


def test_shift_delays_and_advances():
    x = np.arange(5.0)
    assert np.isnan(sr.shift(x, 2)[:2]).all() and sr.shift(x, 2)[2:].tolist() == [0.0, 1.0, 2.0]
    assert sr.shift(x, -1)[:4].tolist() == [1.0, 2.0, 3.0, 4.0] and np.isnan(sr.shift(x, -1)[4])
    assert sr.shift(x, 0).tolist() == x.tolist()


def test_steering_analyse_recovers_an_injected_delay_and_gain(tmp_path):
    t = np.arange(0.0, 60.0, 1.0 / c.FS)
    v = np.full(len(t), 2.0)
    delta = 0.1 * np.sin(1.1 * t) * np.sin(0.13 * t)
    delay = 0.12                                                            # s, six samples
    r = 0.9 * sr.predicted_yaw_rate(v, sr.shift(delta, int(delay * c.FS)))
    run = synthetic_run(tmp_path, t, v=v, steer=delta, wz=np.nan_to_num(r))
    res = sr.analyse(run)
    assert res["delay_s"] == pytest.approx(delay, abs=0.5 / c.FS)
    assert res["yaw_gain"] == pytest.approx(0.9, abs=1e-3)
    # delta_eff = atan(0.9 tan(delta)): 0.9 delta to O(delta^3), under 3e-4 of slope at 0.1 rad.
    assert res["steering_map"]["slope"] == pytest.approx(0.9, abs=2e-3)
    assert res["steering_map"]["intercept"] == pytest.approx(0.0, abs=1e-4)


def test_turned_over_commanded_by_steering_and_by_speed():
    cmd = np.repeat([0.07, 0.22], 50)
    eff = np.repeat([0.07, 0.11], 50)
    rows = sr.by_steering(cmd, eff)
    assert [(r["low"], r["ratio_median"]) for r in rows] == [(0.05, pytest.approx(1.0)), (0.20, pytest.approx(0.5))]
    speed = np.repeat([1.0, 4.0], 50)
    rows = sr.by_speed(speed, np.full(100, 0.2), np.r_[np.full(50, 0.2), np.full(50, 0.14)])
    assert [(r["low"], r["ratio_median"]) for r in rows] == [(0.5, pytest.approx(1.0)), (3.5, pytest.approx(0.7))]
    assert sr.by_speed(speed, np.full(100, 0.01), np.full(100, 0.01)) == []   # under MIN_STEER


# --------------------------------------------------------------------------- lateral

def test_lateral_analyse_matches_v_r_and_flags_a_flipped_accelerometer(tmp_path):
    t = np.arange(0.0, 40.0, 1.0 / c.FS)
    v = 2.5 + 0.5 * np.sin(0.3 * t)
    delta = 0.15 * np.sin(0.7 * t)
    r = sr.predicted_yaw_rate(v, delta)
    a_lat = la.kinematic_lateral(v, r)
    assert a_lat == pytest.approx(v * r, abs=1e-12)
    dvdt = 0.15 * np.cos(0.3 * t)
    good = la.analyse(synthetic_run(tmp_path, t, v=v, steer=delta, wz=r, ax=dvdt, ay=a_lat))
    assert good["accelerometer_axes"]["mapping_ok"]
    assert good["accel_vs_kinematic"]["slope"] == pytest.approx(1.0, abs=0.02)   # 0.2 s smoothing of a_y only
    assert good["grip"]["steering_delay_s"] == pytest.approx(0.0, abs=0.5 / c.FS)
    flipped = la.analyse(synthetic_run(tmp_path, t, v=v, steer=delta, wz=r, ax=-dvdt, ay=-a_lat))
    assert not flipped["accelerometer_axes"]["mapping_ok"]
    assert "WARNING accelerometer axes" in "\n".join(la.report(c.Run(tmp_path, "rb9", "x"), flipped))


# --------------------------------------------------------------------------- algorithm accuracy: command tracking

def test_command_tracking_recovers_speed_delay_and_ratio(tmp_path):
    t = np.arange(0.0, 60.0, 1.0 / c.FS)
    v_cmd = 2.0 + np.sin(0.7 * t) * np.sin(0.11 * t)
    lag = 0.24
    v = 0.9 * sr.shift(v_cmd, int(round(lag * c.FS)))
    v = np.where(np.isnan(v), v_cmd[0] * 0.9, v)
    steer = 0.1 * np.sin(0.9 * t)
    run = synthetic_run(tmp_path, t, v=v, steer=steer, wz=sr.predicted_yaw_rate(v, steer), drive=v_cmd)
    res = aa.command_tracking(run)
    assert res["speed"]["delay_s"] == pytest.approx(lag, abs=0.5 / c.FS)
    assert res["speed"]["achieved_over_commanded"] == pytest.approx(0.9, abs=1e-3)
    assert res["curvature"]["delay_s"] == pytest.approx(0.0, abs=0.5 / c.FS)
    assert res["curvature"]["gain"] == pytest.approx(1.0, abs=1e-3)


def test_command_tracking_scores_only_while_the_algorithm_drove(tmp_path):
    t = np.arange(0.0, 20.0, 1.0 / c.FS)
    v = np.full(len(t), 2.0)
    lb = (t < 10.0).astype(int)
    run = synthetic_run(tmp_path, t, v=v, steer=np.zeros(len(t)), wz=np.zeros(len(t)), lb=lb,
                        intents=[(0.0, {"severity": "drive", "state": "gap_follow"}),
                                 (5.0, {"severity": "stop", "state": "deadman_released"})])
    assert aa.command_tracking(run)["driving_s"] == pytest.approx(5.0, abs=2.0 / c.FS)


# --------------------------------------------------------------------------- algorithm accuracy: path tracking

SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])  # counter-clockwise


def test_cross_track_sign_follows_the_workspace_convention():
    # + when the point is LEFT of the line's direction (racing_math.project_onto_path).
    e, seg, along = aa.cross_track([5.0, 5.0, 10.5, -0.5], [1.0, -1.0, 5.0, 5.0], SQUARE)
    assert e == pytest.approx([1.0, -1.0, -0.5, -0.5], abs=1e-12)
    assert seg.tolist() == [0, 0, 1, 3] and along == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert aa.cross_track([5.0], [1.0], SQUARE, closed=False)[0] == pytest.approx([1.0])


def test_cross_track_picks_the_nearer_side_of_a_hairpin():
    hairpin = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 0.6], [0.0, 0.6]])
    e, seg, _ = aa.cross_track([5.0], [0.45], hairpin)
    assert seg.tolist() == [2] and e == pytest.approx([0.15], abs=1e-12)


def test_cross_track_agrees_with_pure_pursuits_own_projection():
    """A second, independent implementation: racing_math.project_onto_path on a smooth circle."""
    n = 200
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    line = np.c_[5 * np.cos(a), 5 * np.sin(a)]
    seg_len = racing_math.compute_segment_lengths(line, closed=True)
    rng = np.random.default_rng(3)
    angle, radius = rng.uniform(0, 2 * math.pi, 50), rng.uniform(4.6, 5.4, 50)   # within 0.4 m of the line
    px, py = radius * np.cos(angle), radius * np.sin(angle)
    keep = np.ones(50, dtype=bool)
    e, _, _ = aa.cross_track(px[keep], py[keep], line)
    theirs = [racing_math.project_onto_path(line, seg_len, (x, y),
                                            racing_math.find_nearest_index(line, (x, y), closed=True)[0]).lateral_error
              for x, y in zip(px[keep], py[keep])]
    assert e == pytest.approx(theirs, abs=1e-9)


def test_node_log_parses_the_real_ready_and_drive_lines(tmp_path):
    run = c.Run(path=tmp_path, car="rb2", run_id="x", rosout=[(1.0, "pure_pursuit_node", READY),
                                                               (2.0, "pure_pursuit_node", DRIVE),
                                                               (3.0, "pure_pursuit_node", "STOP [body_contact] ...")])
    ready, drive = aa.node_log(run)
    assert ready == [(186, "/home/racerbotcar-2/racerbot-ws/data/rb2/racerbot_auto/20261111-151026-rebuilt/"
                           "raceline_profiled.csv")]
    assert drive.tolist() == [[2.0, -16.41, 9.32, pytest.approx(math.radians(-81.0)), 0.08]]


def path_run(tmp_path, radius_driven, *, line_file="raceline_profiled.csv", recorded_file=None, intents=None,
             jump_at=None, laser=True):
    """A car on a circle of radius_driven (base_link) round a 200-point circle line of radius 5,
    reported as the particle filter would (the LiDAR's pose), with the run's metadata."""
    folder = tmp_path / "data" / "rb9" / "racerbot_auto" / "line"
    folder.mkdir(parents=True)
    a = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    np.savetxt(folder / line_file, np.c_[5 * np.cos(a), 5 * np.sin(a), np.full(200, 2.0)], delimiter=",",
               header="x,y,speed", comments="")
    (folder / "map.yaml").write_text("image: map.pgm\n")
    params = tmp_path / "run" / "metadata" / "params"
    params.mkdir(parents=True)
    recorded = recorded_file or f"/home/racerbotcar-2/racerbot-ws/data/rb9/racerbot_auto/line/{line_file}"
    (params / "pure_pursuit_node.yaml").write_text(yaml.safe_dump({"/pure_pursuit_node": {"ros__parameters": {
        "waypoints_file": recorded, "pose_topic": "/pf/viz/inferred_pose", "pose_frame": "laser",
        "laser_offset_x": 0.26, "laser_offset_y": 0.0, "steering_law": "pure_pursuit"}}}))
    (params / "map_server.yaml").write_text(yaml.safe_dump({"/map_server": {"ros__parameters": {
        "yaml_filename": "/home/racerbotcar-2/racerbot-ws/data/rb9/racerbot_auto/line/map.yaml"}}}))
    t = np.arange(0.0, 20.0, 1.0 / 40)
    speed = 2.0
    th = speed * t / radius_driven
    bx, by, yaw = radius_driven * np.cos(th), radius_driven * np.sin(th), th + math.pi / 2
    px, py = (bx + 0.26 * np.cos(yaw), by + 0.26 * np.sin(yaw)) if laser else (bx, by)
    if jump_at is not None:                          # a 0.25 s glitch: 1 m away and back
        px = px.copy()
        px[jump_at:jump_at + 10] += 1.0
    run = synthetic_run(tmp_path / "run", t, v=np.full(len(t), speed), wz=np.full(len(t), speed / radius_driven),
                        pose=(px, py, yaw), pose_topic="/pf/viz/inferred_pose",
                        intents=intents or [(0.0, {"state": "pure_pursuit", "severity": "caution"})])
    run.frames["/pf/viz/inferred_pose"] = "/map"
    run.rosout = [(0.0, "pure_pursuit_node", f"pure_pursuit_node ready: 200 waypoints from '{recorded}' (closed loop)")]
    return run, folder / line_file


def test_path_tracking_measures_a_known_offset_from_a_laser_pose(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    run, line = path_run(tmp_path, 5.2)
    res = aa.path_tracking(run)
    assert all(ch["ok"] for ch in res["checks"]), res["checks"]
    # 0.2 m outside a counter-clockwise circle is right of the line: e = -0.2 m, less the
    # chord sag 5 (1 - cos(pi/200)) = 0.6 mm where the line cuts inside the circle.
    sag = 5 * (1 - math.cos(math.pi / 200))
    assert res["cross_track"]["mean"] == pytest.approx(-0.2, abs=sag + 1e-6)
    assert res["cross_track"]["max_abs"] <= 0.2 + sag + 1e-6
    assert res["time_s"]["scored"] == pytest.approx(20.0, abs=0.1)


def test_path_tracking_without_the_laser_offset_would_be_wrong(tmp_path, monkeypatch):
    # The same car with its pose taken as base_link: 0.26 m ahead on a 5.2 m circle sits farther out.
    monkeypatch.setattr(c, "ROOT", tmp_path)
    run, _ = path_run(tmp_path, 5.2, laser=False)
    res = aa.path_tracking(run)
    expected = 5.2 - math.hypot(5.2, 0.26)           # the rear axle placed 0.26 m behind the true base_link
    assert res["cross_track"]["mean"] == pytest.approx(-0.2 + expected, abs=1e-3)


def test_path_tracking_leaves_out_avoidance_and_pose_jumps(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    intents = [(0.0, {"state": "pure_pursuit", "severity": "caution"}),
               (5.0, {"state": "lidar_avoidance", "severity": "caution"}),
               (8.0, {"state": "pure_pursuit", "severity": "caution"})]
    run, _ = path_run(tmp_path, 5.2, intents=intents, jump_at=600)        # a 1 m glitch at 15 s
    res = aa.path_tracking(run)
    assert res["time_s"]["not_tracking"] == pytest.approx(3.0, abs=0.1)
    assert res["pose"]["jumps"] == 2                                      # away, and back
    assert res["time_s"]["near_pose_jump"] > 0
    assert res["states_not_tracking"] == ["lidar_avoidance"]
    assert res["cross_track"]["max_abs"] < 0.21                           # the 1 m jump is never scored


def test_path_tracking_checks_fail_on_a_line_the_run_did_not_load(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    run, line = path_run(tmp_path, 5.2, recorded_file="/home/x/racerbot-ws/data/rb9/racerbot_auto/line/other.csv")
    res = aa.path_tracking(run, line_path=line)
    failed = {ch["check"] for ch in res["checks"] if not ch["ok"]}
    assert failed == {"line is the run's waypoints_file", "waypoint count matches the node's log"}


def test_path_tracking_checks_fail_when_the_node_loaded_another_count(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    run, _ = path_run(tmp_path, 5.2)
    run.rosout = [(0.0, "pure_pursuit_node", run.rosout[0][2].replace("200 waypoints", "199 waypoints"))]
    failed = {ch["check"] for ch in aa.path_tracking(run)["checks"] if not ch["ok"]}
    assert failed == {"waypoint count matches the node's log"}


def test_a_run_with_no_racing_line_gets_no_path_tracking(tmp_path):
    run = synthetic_run(tmp_path, np.arange(0, 5, 0.02), v=np.full(250, 1.0))
    assert aa.path_tracking(run) is None
    lines = aa.report(run, {"driving_s": 0.0}, None)
    assert "command-tracking results only, not path-tracking accuracy" in lines[-1]
