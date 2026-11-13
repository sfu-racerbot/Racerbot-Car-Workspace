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


def lap_times(times, arcs, length):
    """Times between successive start/finish crossings (arc wrapping from the end to the start)."""
    crossings = [times[i] for i in range(1, len(arcs)) if arcs[i - 1] > 0.75 * length and arcs[i] < 0.25 * length]
    return [b - a for a, b in zip(crossings, crossings[1:])]


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
              grip_scale=None, grip_coupling=None):
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

        def recording(commands):
            truth = plant.truth(0)
            trace.append((len(trace) * rv.CONTROL_DT, float(truth[0]), float(truth[1]), float(truth[3]),
                          float(commands[0][0]), float(commands[0][1])))
            return step(commands)
        plant.step = recording
        return plant
    rv.make_plant, rv.plan_initial_pose = wrapped, spawn
    try:
        result = rv.run_pure_solo(eval_gap.track_path(track), seed, seconds, calibration=cal, plan=plan,
                                  **solo_arguments(law, speed, lateral, laps, steering_limit))
    finally:
        rv.make_plant, rv.plan_initial_pose = make_plant, plan_pose
    return cal, plan, np.array(trace, float), result


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
                                 end_offset_left=float(off[-1])),
                at=at, laps_s=lap_times(t, s, length), profile_at={a: float(np.interp(a, np.r_[arc, length],
                np.r_[plan.speed, plan.speed[0]])) for a in arcs_wanted}, start_arc=float(s[0]), start_offset=float(off[0]))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True)
    ap.add_argument("--floor", required=True, choices=sorted(FLOORS))
    ap.add_argument("--line", required=True, help="profiled x,y,speed CSV")
    ap.add_argument("--start", required=True, help="x,y,yaw_deg of the spawn pose")
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
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    start = tuple(float(v) for v in args.start.split(","))
    arcs = [float(a) for a in args.arcs.split(",")]
    rows = []
    for k in range(args.seeds):
        cal, plan, trace, result = trace_run(args.track, args.floor, args.line, start, args.first_seed + k,
                                             args.seconds, args.laps, args.steering_limit, args.speed,
                                             args.lateral, args.law, args.grip_scale, args.grip_coupling)
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
