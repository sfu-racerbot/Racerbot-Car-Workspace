#!/usr/bin/env python3
"""Question 3 of the corner-entry study: when was the real command above what the car could still shed before the corner?

    python3 tools/run_analysis/corner_shed.py ~/racerbot-ws/data/rb1/racerbot_runs/20261111-155542 --contact 87.34 --from 84.5

From the wheel speed integrated along the path, d(t) = the distance still to drive to the contact point. With a braking deceleration a (the
measured median of speed_tracking a.) and a speed v_c wanted at the corner, the highest speed from which the car can still reach v_c is
v_max(t) = sqrt(v_c^2 + 2 a d(t)). Prints, per (a, v_c), the first time the command exceeded v_max and the first time the speed did.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_analysis import speed_tracking  # noqa: E402


def shed_table(v_t, v, cmd_t, cmd, t_contact, t_from, decels, targets):
    """{(a, v_c): (first time command > v_max, first time speed > v_max)} for t in [t_from, t_contact]."""
    sel = (v_t >= t_from) & (v_t <= t_contact)
    t, vv = v_t[sel], v[sel]
    s = np.concatenate([[0.0], np.cumsum(0.5 * (vv[1:] + vv[:-1]) * np.diff(t))])
    d = s[-1] - s                                           # metres still to drive to the contact point
    c = speed_tracking.hold(cmd_t, cmd, t)
    out = {}
    for a in decels:
        for vc in targets:
            vmax = np.sqrt(vc * vc + 2.0 * a * d)
            over_c = np.where(c > vmax + 1e-9)[0]
            over_v = np.where(vv > vmax + 1e-9)[0]
            out[(a, vc)] = (float(t[over_c[0]]) if over_c.size else math.nan, float(t[over_v[0]]) if over_v.size else math.nan)
    return out, dict(distance=float(s[-1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--contact", type=float, required=True, help="contact time on intent_report's clock")
    ap.add_argument("--from", dest="t_from", type=float, required=True)
    ap.add_argument("--decels", type=float, nargs="+", default=[0.65, 1.08, 2.25])
    ap.add_argument("--targets", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    args = ap.parse_args(argv)
    segs = [s for s in speed_tracking.run_segments(args.run) if s[2][0] <= args.contact <= s[2][-1]]
    cmd_t, cmd, v_t, v = segs[0]
    table, info = shed_table(v_t, v, cmd_t, cmd, args.contact, args.t_from, args.decels, args.targets)
    print(f"distance driven from {args.t_from} s to the contact at {args.contact} s: {info['distance']:.2f} m")
    print("   decel m/s^2  v_c m/s | first t command > v_max | first t speed > v_max")
    for (a, vc), (tc, tv) in table.items():
        print(f"   {a:10.2f} {vc:8.1f} | {tc:22.2f} | {tv:20.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
