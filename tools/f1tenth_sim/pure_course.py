#!/usr/bin/env python3
"""pure_pursuit on a real course in the simulator, on a racing line of your choice.

    M=data/rb2/racerbot_auto/20261030-185036-fixed
    python3 tools/f1tenth_sim/pure_course.py --track course_2026_10_30_scans --floor sfu_course_2026_10_30 \\
        --line $M/raceline_raw.csv --speed 4.0,5.0 --lateral 2.5,3.0

Each combination of planned top speed and lateral limit drives a solo run with
run_validation's pure pursuit controller (the same math as pure_pursuit_node), starting
on the line, and prints the lap time, whether it finished, cross-track error, and how
often the reactive safety layer took over. Without --line it follows the track's own
centre line, as run_validation.py does.

    python3 tools/f1tenth_sim/pure_course.py --safety map --laps 3
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_gap  # noqa: E402,F401  (bootstraps gym and the sim path)
import run_validation as rv  # noqa: E402
from sim_fidelity import FLOORS, with_floor, with_friction  # noqa: E402


def plan_line(xy, top_speed: float, lateral: float, brake: float = 8.0, accel: float = 3.0):
    """A PathPlan along ``xy`` with run_validation's profile, at these limits."""
    from pure_pursuit import racing_math
    xy = np.asarray(xy, float)
    seg = racing_math.compute_segment_lengths(xy, closed=True)
    kappa = racing_math.estimate_path_curvature(xy, closed=True)
    speed = racing_math.compute_velocity_profile(seg, kappa, v_max=top_speed, v_min=0.5, a_lat_max=lateral,
                                                 a_accel_max=accel, a_brake_max=brake, closed=True,
                                                 smoothing_passes=8, friction_ellipse=True)
    return rv.PathPlan(xy=xy, seg_len=seg, speed=speed)


def main(argv=None) -> int:
    from run_analysis.loc import load_line
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="course_2026_10_30_scans")
    ap.add_argument("--line", help="x,y CSV of the line to follow (default: the track's centre line)")
    ap.add_argument("--speed", default="4.0", help="comma list of planned top speeds, m/s")
    ap.add_argument("--lateral", default="2.5", help="comma list of lateral limits, m/s^2")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--friction", type=float)
    ap.add_argument("--floor", choices=sorted(FLOORS))
    ap.add_argument("--safety", choices=rv.SAFETY_MODES, default="fallback")
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--brake", type=float, default=8.0, help="profile braking limit, m/s^2 (car 2 measured 3.0-4.5)")
    ap.add_argument("--accel", type=float, default=3.0, help="profile acceleration limit, m/s^2")
    ap.add_argument("--profiled", action="store_true",
                    help="--line is a profiled x,y,speed CSV: drive its own speeds instead of re-profiling it "
                         "(--speed and --lateral then only set the controller's ceilings)")
    ap.add_argument("--law", default="pure_pursuit", choices=("pure_pursuit", "stanley", "lqr", "mpc"),
                    help="steering law, as pure_pursuit_node's steering_law")
    ap.add_argument("--set", action="append", default=[],
                    help="a pure_pursuit.yaml steering-law parameter, name=value (e.g. lookahead_speed_gain=0.25)")
    ap.add_argument("--online-lateral", type=float, default=None,
                    help="lateral limit of the controller's online curvature ceiling, m/s^2 (default: the planned one)")
    ap.add_argument("--steering-limit", type=float,
                    help="steering limit in radians (default: car configuration)")
    args = ap.parse_args(argv)
    calibration = with_friction(args.friction, with_floor(args.floor))
    track = eval_gap.track_path(args.track)
    xy = load_line(args.line) if args.line else None
    profiled = None
    if args.profiled:
        from pure_pursuit import racing_math
        if not args.line:
            ap.error("--profiled needs --line")
        xy, profiled = racing_math.load_profiled_csv(args.line)
    for top, lateral in itertools.product(*([float(v) for v in s.split(",")] for s in (args.speed, args.lateral))):
        if profiled is not None:
            from pure_pursuit import racing_math
            plan = rv.PathPlan(xy=np.asarray(xy, float), seg_len=racing_math.compute_segment_lengths(xy, closed=True),
                               speed=np.minimum(profiled, top))
        else:
            plan = plan_line(xy, top, lateral, args.brake, args.accel) if xy is not None else None
        r = rv.run_pure_solo(track, args.seed, args.timeout, calibration=calibration, plan=plan, law_name=args.law,
                             law_params={k: float(v) for k, v in (item.split("=", 1) for item in args.set)} or None,
                             max_speed=top, max_lateral_accel=args.online_lateral or lateral, safety=args.safety,
                             laps=args.laps, steering_limit=args.steering_limit)
        print(f"top {top:.1f} lateral {lateral:.1f}: lap {r['lap_time_s']} finished {r['passed']} contact {r['contact_steps']} "
              f"cross-track max {r['max_cross_track_m']} safety avoid {r['avoid_steps']} stop {r['stop_steps']} escape {r.get('escape_steps', 0)} "
              f"lap times {[round(value, 2) for value in r['lap_times_s']]}", flush=True)
    return 0


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "tools"), str(root / "src" / "pure_pursuit")]
    raise SystemExit(main())
