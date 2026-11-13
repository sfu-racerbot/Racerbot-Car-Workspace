#!/usr/bin/env python3
"""A pure_pursuit race controller started from a given pose at rest, with a trace of where it went.

    python3 tools/f1tenth_sim/start_replay.py --track car2_course_2026_11_11 --floor sfu_course_2026_11_11_rb2 \\
        --line raceline_profiled.csv --start -16.5746,10.6178,285.151 --seeds 6 --out failed_start.json

The suite's race settings (course_suite/run.py stage_race: profiled line, map safety, 0.42 rad, delay_lqr,
the car's own pure_pursuit.yaml) with the spawn pose replaced, and every tick's true state recorded. Reports,
per seed, wheel speed (true speed times the floor's odometry scale), signed cross-track to the line (positive
left) and the line's planned speed at chosen arc lengths, and the laps' times from the line's start/finish.
It does not mirror flank creep or the reactive node's other layers: run_validation's map safety stands in.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_gap  # noqa: E402,F401  (bootstraps gym and the sim path)
import run_validation as rv  # noqa: E402
from sim_fidelity import FLOORS, with_floor  # noqa: E402

DEFAULT_ARCS = (10.0, 11.0, 12.0, 13.0, 14.0)


def line_geometry(xy):
    xy = np.asarray(xy, float)
    seg = np.diff(np.vstack([xy, xy[:1]]), axis=0)
    ds = np.hypot(seg[:, 0], seg[:, 1])
    return xy, seg, ds, np.concatenate([[0.0], np.cumsum(ds)])[:-1]


def project(xy, seg, ds, arc, x, y):
    """(arc length, signed offset, left positive) of the nearest line point."""
    j = int(np.argmin(np.hypot(xy[:, 0] - x, xy[:, 1] - y)))
    tx, ty = seg[j] / ds[j]
    return float(arc[j]), float((x - xy[j, 0]) * -ty + (y - xy[j, 1]) * tx)


def pose_on_line(xy, arc_s, right=0.0, heading_error_deg=0.0):
    """(x, y, yaw_deg) at arc length ``arc_s`` along the closed line, ``right`` m to the right of it, the
    heading ``heading_error_deg`` to the left of its tangent."""
    xy, seg, ds, arc = line_geometry(xy)
    arc_s = arc_s % float(ds.sum())
    j = int(np.searchsorted(arc, arc_s, side="right")) - 1
    tx, ty = seg[j] / ds[j]
    x, y = xy[j] + (arc_s - arc[j]) * np.array([tx, ty])
    return (float(x + right * ty), float(y - right * tx),
            float(math.degrees(math.atan2(ty, tx)) + heading_error_deg))


def lap_times(times, arcs, length):
    """Times between successive start/finish crossings (arc wrapping from the end to the start)."""
    crossings = [times[i] for i in range(1, len(arcs)) if arcs[i - 1] > 0.75 * length and arcs[i] < 0.25 * length]
    return [b - a for a, b in zip(crossings, crossings[1:])]


import contextlib  # noqa: E402


@contextlib.contextmanager
def cap_binding_counter():
    """Counts, over the commands the controller's final stage issues, the ticks on which the online curvature
    cap (CommandShaper's curve_speed) was below both the wanted speed and the hard cap: the factor that set
    the speed. Yields a dict with ``ticks`` and ``binding``; undone on exit."""
    stats = dict(ticks=0, binding=0)
    last = {}
    original_limit = rv.racing_math.curvature_speed_limit
    original_command = rv.CommandShaper.command

    def limit(*a, **k):
        last["curve"] = original_limit(*a, **k)
        return last["curve"]

    def command(self, desired_steering, desired_speed, hard_speed_cap=None, measured_speed=None):
        last.clear()
        out = original_command(self, desired_steering, desired_speed, hard_speed_cap, measured_speed)
        if "curve" in last:
            stats["ticks"] += 1
            hard = self.max_speed if hard_speed_cap is None else hard_speed_cap
            if last["curve"] < min(desired_speed, hard) - 1e-9:
                stats["binding"] += 1
        return out

    rv.racing_math.curvature_speed_limit = limit
    rv.CommandShaper.command = command
    try:
        yield stats
    finally:
        rv.racing_math.curvature_speed_limit = original_limit
        rv.CommandShaper.command = original_command


@contextlib.contextmanager
def understeer_compensation(table):
    """The node's understeer compensation (off unless a table is given) in the math-level controller:
    ``racing_math.understeer_steering`` on the tracking law's steering at the measured speed, and the online cap
    from ``delivered_curvature_speed_limit``, as pure_pursuit_node does with ``understeer_curve_asked/delivered``.
    ``table`` is dict(asked=[...], delivered=[...]) or None (nothing changes)."""
    if not table:
        yield
        return
    from pure_pursuit import racing_math
    curve = racing_math.grip_curve_from_params(list(table["asked"]), list(table["delivered"]))
    original_command = rv.PathFollower.command
    original_limit = rv.racing_math.curvature_speed_limit

    def command(self, state, *a, **k):
        steering, speed, *rest = original_command(self, state, *a, **k)
        steering = racing_math.understeer_steering(steering, abs(float(state[3])), rv.WHEELBASE, curve, rv.STEERING_LIMIT)
        return (steering, speed, *rest)

    def limit(curvature, max_lateral_accel, max_speed):
        return racing_math.delivered_curvature_speed_limit(curvature, max_lateral_accel, max_speed, curve)

    rv.PathFollower.command = command
    rv.racing_math.curvature_speed_limit = limit
    try:
        yield
    finally:
        rv.PathFollower.command = original_command
        rv.racing_math.curvature_speed_limit = original_limit


def floor_calibration(floor, grip_scale=None, grip_coupling=None):
    """The floor, with its grip scale and longitudinal coupling overridden only where given."""
    from dataclasses import replace
    from sim_fidelity.calibration import sfu_2026_11_11_rb2_grip_curve
    cal = with_floor(floor)
    if grip_scale is not None:
        if floor != "sfu_course_2026_11_11_rb2":
            raise ValueError(f"--grip-scale needs a floor with a scalable curve (sfu_course_2026_11_11_rb2), not {floor}")
        cal = replace(cal, grip_curve=sfu_2026_11_11_rb2_grip_curve(float(grip_scale)))
    if grip_coupling is not None:
        if not (math.isfinite(grip_coupling) and 0.0 <= grip_coupling <= 1.0):
            raise ValueError(f"--grip-coupling must be in [0, 1], got {grip_coupling!r}")
        cal = replace(cal, longitudinal_grip_coupling=float(grip_coupling))
    return cal


def solo_arguments(law, speed, lateral, laps, steering_limit, safety="map"):
    """What run_pure_solo is called with, the online lateral limit included."""
    return dict(law_name=law, max_speed=float(speed), max_lateral_accel=float(lateral), safety=safety, laps=laps,
                steering_limit=steering_limit)


def trace_run(track, floor, line_csv, start, seed, seconds, laps, steering_limit, speed, lateral, law,
              grip_scale=None, grip_coupling=None, start_wheel_speed=None, understeer=None):
    from pure_pursuit import racing_math
    cal = floor_calibration(floor, grip_scale, grip_coupling)
    rv.RACELINE_SOURCE = "plan"
    xy, profiled = racing_math.load_profiled_csv(line_csv)
    plan = rv.PathPlan(xy=np.asarray(xy, float), seg_len=racing_math.compute_segment_lengths(xy, closed=True),
                       speed=np.minimum(profiled, speed))
    trace = []
    pose = np.array([[start[0], start[1], math.radians(start[2])]], dtype=float)
    make_plant, plan_pose = rv.make_plant, rv.plan_initial_pose

    def spawn(plan_, line_):
        return pose

    def wrapped(*a, **k):
        plant = make_plant(*a, **k)
        step = plant.step
        if start_wheel_speed is not None:
            reset = plant.reset         # already moving: the wheels read true speed times the odometry scale
            plant.reset = lambda poses: reset(poses, speeds=[start_wheel_speed / cal.odom_speed_scale])

        def recording(commands):
            truth = plant.truth(0)
            trace.append((len(trace) * rv.CONTROL_DT, float(truth[0]), float(truth[1]), float(truth[3]),
                          float(commands[0][0]), float(commands[0][1])))
            return step(commands)
        plant.step = recording
        return plant
    rv.make_plant, rv.plan_initial_pose = wrapped, spawn
    try:
        with understeer_compensation(understeer), cap_binding_counter() as cap:
            result = rv.run_pure_solo(eval_gap.track_path(track), seed, seconds, calibration=cal, plan=plan,
                                      **solo_arguments(law, speed, lateral, laps, steering_limit))
        result["cap_binding_ticks"], result["command_ticks"] = cap["binding"], cap["ticks"]
    finally:
        rv.make_plant, rv.plan_initial_pose = make_plant, plan_pose
    return cal, plan, np.array(trace, float), result


def _flying_worst(arc, off, length, sign):
    """Worst offset (right for -1, left for +1) after the first start/finish crossing; None before one."""
    crossed = [i for i in range(1, len(arc)) if arc[i - 1] > 0.75 * length and arc[i] < 0.25 * length]
    if not crossed:
        return None
    tail = off[crossed[0]:]
    return float(max(0.0, -tail.min() if sign < 0 else tail.max()))


def summarise(cal, plan, trace, result, arcs_wanted):
    xy, seg, ds, arc = line_geometry(plan.xy)
    length = float(ds.sum())
    t, x, y, v = trace[:, 0], trace[:, 1], trace[:, 2], trace[:, 3] * cal.odom_speed_scale
    proj = [project(xy, seg, ds, arc, a, b) for a, b in zip(x, y)]
    s = np.array([p[0] for p in proj])
    off = np.array([p[1] for p in proj])
    at = {}
    for target in arcs_wanted:
        k = next((i for i in range(len(s)) if abs((s[i] - target + length / 2) % length - length / 2) < 0.08), None)
        if k is not None:
            at[target] = dict(wheel_speed=float(v[k]), offset_left=float(off[k]), command=float(trace[k, 5]),
                              t=float(t[k]))
    return dict(seed_result=dict(contact=int(result.get("contact_steps", 0) or 0), collision=bool(result["collision"]),
                                 passed=bool(result["passed"]), max_cross_track=result["max_cross_track_m"],
                                 laps=float(result["laps"]), end_t=float(t[-1]), end_arc=float(s[-1]),
                                 end_offset_left=float(off[-1]), max_right=float(max(0.0, -off.min())),
                                 max_left=float(max(0.0, off.max())), max_right_flying=_flying_worst(s, off, length, -1),
                                 max_left_flying=_flying_worst(s, off, length, +1)),
                at=at, laps_s=lap_times(t, s, length), profile_at={a: float(np.interp(a, np.r_[arc, length],
                np.r_[plan.speed, plan.speed[0]])) for a in arcs_wanted}, start_arc=float(s[0]), start_offset=float(off[0]))


def check_profile_alone(args, line_xy):
    """The planned profile driven without the online cap's help: standing starts round the line, then flying laps.

    Every run is cap-off (online lateral 99). Prints per run the worst right and left offset and any contact;
    returns (exit code, runs): 1 when any run touched anything. The gate for a new speed plan: it must be
    drivable on its own (2026-11-13).
    """
    _, _, ds, _ = line_geometry(line_xy)
    length = float(ds.sum())
    jobs = [(f"standing start at s={i * length / args.poses:.1f}", pose_on_line(line_xy, i * length / args.poses), 1)
            for i in range(args.poses)]
    jobs.append((f"{args.check_laps} flying laps after a standing start at s=0", pose_on_line(line_xy, 0.0), args.check_laps + 1))
    runs, touched = [], False
    for label, start, laps in jobs:
        for k in range(args.seeds):
            seed = args.first_seed + k
            cal, plan, trace, result = trace_run(args.track, args.floor, args.line, start, seed, args.seconds, laps,
                                                 args.steering_limit, args.speed, 99.0, args.law,
                                                 args.grip_scale, args.grip_coupling)
            row = summarise(cal, plan, trace, result, [])
            row.update(label=label, seed=seed)
            hit = bool(row["seed_result"]["contact"] > 0 or row["seed_result"]["collision"])
            touched |= hit
            worst = row["seed_result"]
            where = f" contact at s={worst['end_arc']:.1f} m ({worst['end_offset_left']:+.2f} m)" if hit else ""
            flying = worst["max_right_flying"], worst["max_left_flying"]
            fly = "" if flying[0] is None else f", flying laps worst right {flying[0]:.2f} left {flying[1]:.2f}"
            print(f"{label}, seed {seed}: worst right {worst['max_right']:.2f} m, left {worst['max_left']:.2f} m{fly};"
                  f"{where if hit else ' no contact'}", flush=True)
            runs.append(row)
    print("PROFILE ALONE: " + ("FAIL, contact" if touched else "pass, no contact"), flush=True)
    return (1 if touched else 0), runs


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True)
    ap.add_argument("--floor", required=True, choices=sorted(FLOORS))
    ap.add_argument("--line", required=True, help="profiled x,y,speed CSV")
    ap.add_argument("--start", help="x,y,yaw_deg of the spawn pose, at rest")
    ap.add_argument("--start-s", type=float, help="spawn at this arc length of the line, already moving")
    ap.add_argument("--start-speed", type=float, default=0.0, help="wheel speed at --start-s, m/s")
    ap.add_argument("--start-right", type=float, default=0.0, help="metres right of the line at --start-s")
    ap.add_argument("--start-heading-error", type=float, default=0.0,
                    help="degrees left of the line's heading at --start-s")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--first-seed", type=int, default=12345)
    ap.add_argument("--seconds", type=float, default=80.0)
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--steering-limit", type=float, default=0.42)
    ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--lateral", "--online-lateral", dest="lateral", type=float, default=3.1,
                    help="the controller's online curvature ceiling (max_lateral_accel), m/s^2; "
                         "99 turns it effectively off, so the planned profile is driven unprotected")
    ap.add_argument("--grip-scale", type=float, default=None,
                    help="scale of the sfu_course_2026_11_11_rb2 floor's measured grip curve (default: the floor's 1.25)")
    ap.add_argument("--grip-coupling", type=float, default=None,
                    help="share of longitudinal acceleration that competes with cornering, 0 to 1 "
                         "(default: the floor's, 0)")
    ap.add_argument("--law", default="delay_lqr")
    ap.add_argument("--arcs", default=",".join(str(a) for a in DEFAULT_ARCS))
    ap.add_argument("--check-profile-alone", action="store_true",
                    help="drive the profile with the online cap off from --poses standing starts and --check-laps "
                         "flying laps, every seed; exit 1 on any contact")
    ap.add_argument("--poses", type=int, default=5, help="standing starts evenly spaced round the line")
    ap.add_argument("--check-laps", type=int, default=3)
    ap.add_argument("--out")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.check_profile_alone:
        from pure_pursuit import racing_math
        line_xy, _ = racing_math.load_profiled_csv(args.line)
        code, runs = check_profile_alone(args, line_xy)
        if args.out:
            Path(args.out).write_text(json.dumps(runs, indent=1))
        return code
    if not args.out:
        raise SystemExit("start_replay: --out is required")
    start_wheel_speed = None
    if args.start_s is not None:
        from pure_pursuit import racing_math
        line_xy, _ = racing_math.load_profiled_csv(args.line)
        start = pose_on_line(line_xy, args.start_s, args.start_right, args.start_heading_error)
        start_wheel_speed = args.start_speed
    elif args.start:
        start = tuple(float(v) for v in args.start.split(","))
    else:
        raise SystemExit("start_replay: give --start x,y,yaw_deg or --start-s")
    arcs = [float(a) for a in args.arcs.split(",")]
    rows = []
    for k in range(args.seeds):
        cal, plan, trace, result = trace_run(args.track, args.floor, args.line, start, args.first_seed + k,
                                             args.seconds, args.laps, args.steering_limit, args.speed,
                                             args.lateral, args.law, args.grip_scale, args.grip_coupling, start_wheel_speed)
        row = summarise(cal, plan, trace, result, arcs)
        row["seed"] = args.first_seed + k
        rows.append(row)
        at = ", ".join(f"s{a:g}: {d['wheel_speed']:.2f} m/s {d['offset_left']:+.2f} m" for a, d in row["at"].items())
        print(f"seed {row['seed']}: {at}; contact {row['seed_result']['contact']} laps {row['laps_s']}", flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "tools"), str(root / "src" / "pure_pursuit")]
    raise SystemExit(main())
