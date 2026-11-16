#!/usr/bin/env python3
"""Every stall and slow spot of a gap_follow run, by course position.

    python3 tools/run_analysis/gap_stalls.py data/rb1/racerbot_runs/20261111-155542 \\
        --loc ~/.cache/racerbot/loc --line tools/f1tenth_sim/maps/car2_course_2026_11_11_planned_line.csv

A stall is the node's published /drive speed at exactly zero for 0.5 s or more while LB is held. A slow spot is a
run of ticks with the true speed under 1.0 m/s that is not a stall, outside the first 3 s after an LB press (the
standing start). Each is placed by loc.py's true pose on the course line (``s`` from the line's first point, the
race-day start pose) and explained by replaying the LB hold through steer_stages.replay_window: the factor that
bound (stop reason when the mirror stopped) and the left and right side clearance (minimum raw range 60-120
degrees). Times are seconds from the first /joy, /odom or /drive_intent message (intent_report.py's clock).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "tools", ROOT / "tools" / "f1tenth_sim", ROOT / "src" / "gap_follow", ROOT / "src" / "pure_pursuit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_analysis import runlog  # noqa: E402
from run_analysis.loc import COL  # noqa: E402
from run_analysis.steer_stages import bag_origin, gyro_series, recorded_params, replay_window  # noqa: E402

STALL_MIN = 0.5      # s
SLOW = 1.0           # m/s
LAUNCH_GRACE = 3.0   # s after an LB press


def line_projector(line_csv):
    """s(x, y): arc length along the closed course line to the nearest point, from the line's first point."""
    pts = np.loadtxt(line_csv, delimiter=",", comments="#", skiprows=1)[:, :2]
    seg = np.hypot(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(seg)[:-1]])

    def project(x, y):
        k = int(np.argmin(np.hypot(pts[:, 0] - x, pts[:, 1] - y)))
        return float(arc[k]), float(np.hypot(pts[k, 0] - x, pts[k, 1] - y))
    return project, float(seg.sum())


def groups(mask, t, min_len, gap=0.06):
    """Runs of True in ``mask`` (times ``t``), allowing gaps shorter than ``gap`` s, at least ``min_len`` s long."""
    out, start, last = [], None, None
    for i, m in enumerate(mask):
        if m:
            if start is None:
                start = i
            elif t[i] - t[last] > gap:
                out.append((start, last)); start = i
            last = i
        elif start is not None and t[i] - t[last] > gap:
            out.append((start, last)); start = None
    if start is not None:
        out.append((start, last))
    return [(a, b) for a, b in out if t[b] - t[a] >= min_len]


def analyse(run_dir, loc_dir, line_csv):
    run_dir = Path(run_dir).resolve()
    run = runlog.load_run(run_dir)
    origin = bag_origin(run_dir)
    loc = np.load(Path(loc_dir).expanduser() / f"{run_dir.name}.npz")["a"]
    project, length = line_projector(line_csv)
    params, gyro = recorded_params(run_dir), gyro_series(run_dir)
    rows = []
    for press, release in run.lb_windows(min_duration=1.0):
        w = replay_window(run, params, press + 1e-3, release, gyro, loc=loc)
        for r in w:
            r["press"] = press
        rows.extend(w)
    t = np.array([r["t"] for r in rows])
    real = np.array([r["real_speed"] for r in rows])
    true_v = np.array([r.get("true_speed", math.nan) for r in rows])
    press = np.array([r["press"] for r in rows])

    def describe(a, b, kind):
        mid = (a + b) // 2
        r0 = rows[a]
        x, y = rows[mid].get("x", math.nan), rows[mid].get("y", math.nan)
        s, off = project(x, y) if math.isfinite(x) else (math.nan, math.nan)
        return dict(kind=kind, start=r0["t"] - origin, duration=t[b] - t[a], s=s, off_line=off, x=x, y=y,
                    binding=r0["binding"] + (f"({r0['stop_reason']})" if r0["stopped"] else ""),
                    opponent=r0["opponent_mode"], left=r0["left_60_120"], right=r0["right_60_120"],
                    forward=r0["forward"], speed=r0["speed"], since_press=r0["t"] - r0["press"],
                    replay_speed=r0["speed_cmd"], real_speed=r0["real_speed"])
    stalls = [describe(a, b, "stall") for a, b in groups(real == 0.0, t, STALL_MIN)]
    in_stall = np.zeros(len(rows), bool)
    for a, b in groups(real == 0.0, t, STALL_MIN):
        in_stall[a:b + 1] = True
    slow_mask = (true_v < SLOW) & ~in_stall & (t - press > LAUNCH_GRACE)
    slow = [describe(a, b, "slow") for a, b in groups(slow_mask, t, 0.0, gap=0.12)]
    paired = [r for r in rows if math.isfinite(r["real_steering"])]
    err = np.array([r["steering"] - r["real_steering"] for r in paired])
    verr = np.array([r["speed_cmd"] - r["real_speed"] for r in paired])
    parity = dict(ticks=len(rows), paired=len(paired), steer_max=float(np.abs(err).max()),
                  steer_rms=float(np.sqrt(np.mean(err ** 2))), speed_median=float(np.median(np.abs(verr))),
                  speed_over_0p05=float(np.mean(np.abs(verr) > 0.05)))
    return stalls, slow, length, parity


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--loc", required=True)
    ap.add_argument("--line", required=True, help="x,y[,speed] CSV of the course line (a header row, then points; the race-day start pose first)")
    args = ap.parse_args(argv)
    stalls, slow, length, parity = analyse(args.run, args.loc, args.line)
    print(f"{args.run}: line length {length:.2f} m")
    print("replay parity with /drive: {ticks} ticks, {paired} paired, steering max {steer_max:.4f} rad rms {steer_rms:.4f}, "
          "speed median |err| {speed_median:.3f} m/s, {speed_over_0p05:.1%} of ticks more than 0.05 m/s off".format(**parity))
    for title, items in (("STALLS (commanded speed 0 for 0.5 s or more, LB held)", stalls), ("SLOW SPOTS (true speed under 1.0 m/s, not a stall)", slow)):
        print(title)
        print("   start    dur     s   off     x       y    bound               opponent   left  right   fwd   v   after_press replay/real cmd")
        for d in items:
            print(f"{d['start']:8.2f} {d['duration']:6.2f} {d['s']:6.2f} {d['off_line']:5.2f} {d['x']:7.2f} {d['y']:7.2f}  "
                  f"{d['binding']:<20}{d['opponent']:<10}{d['left']:6.2f} {d['right']:6.2f} {d['forward']:5.2f} {d['speed']:4.2f} {d['since_press']:7.2f}   {d['replay_speed']:.2f}/{d['real_speed']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
