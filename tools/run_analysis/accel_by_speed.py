#!/usr/bin/env python3
"""How hard the car can accelerate (and brake), against its speed. Pure functions plus a bag-reading main().

    python3 tools/run_analysis/accel_by_speed.py <run_dir> [<run_dir> ...] [--car rb2] [--json out.json]

Only samples where the speed loop is asking for more are used: the command exceeds the wheel speed by at least
``min_ask`` m/s (for braking, the wheel speed exceeds the command), so what the wheels do is what the car can
give, not what was asked. The acceleration is the centred difference of the wheel speed over ``smooth`` seconds.
The loop is not saturated on every such sample, so the UPPER QUARTILE (p75) of a speed bin is taken as that
bin's capability, and only bins with 20 or more samples report one (NaN otherwise). Written 2026-11-13 after the
2026-11-11 launch gained 6 m/s^2 at 1-3 m/s but 0.8-2.6 at 3.4-4.9 m/s (the floor and the planner assumed a flat
4.0). Reading a run needs its bag (ROS sourced) through runlog.load_run; the functions need only numpy.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

BINS = (0, 1, 2, 3, 3.5, 4, 4.5, 5, 6)
MIN_SAMPLES = 20


def _clean(t, wheel_speed, command_speed):
    t, v, c = (np.asarray(x, dtype=float).ravel() for x in (t, wheel_speed, command_speed))
    if not (len(t) == len(v) == len(c)):
        raise ValueError('t, wheel_speed and command_speed must have the same length')
    keep = np.isfinite(t) & np.isfinite(v) & np.isfinite(c)
    t, v, c = t[keep], v[keep], c[keep]
    order = np.argsort(t, kind='stable')
    return t[order], v[order], c[order]


def accel_capability(t, wheel_speed, command_speed, *, min_ask=0.5, smooth=0.2, bins=BINS, braking=False):
    """Per speed bin: n, median, p75, p90 of the acceleration (deceleration magnitude if ``braking``) over the
    samples where the loop is asking for more (less); ``capability`` = p75 when n >= 20, else NaN."""
    t, v, c = _clean(t, wheel_speed, command_speed)
    half = 0.5 * smooth
    accel = np.full(len(t), np.nan)
    if len(t) >= 2:
        inside = (t - half >= t[0]) & (t + half <= t[-1])
        accel[inside] = (np.interp(t[inside] + half, t, v) - np.interp(t[inside] - half, t, v)) / smooth
    asked = (v - c >= min_ask) if braking else (c - v >= min_ask)
    value = -accel if braking else accel
    use = asked & np.isfinite(value)
    rows = []
    for low, high in zip(bins[:-1], bins[1:]):
        chosen = value[use & (v >= low) & (v < high)]
        if len(chosen):
            row = dict(low=float(low), high=float(high), n=int(len(chosen)), median=float(np.median(chosen)),
                       p75=float(np.percentile(chosen, 75)), p90=float(np.percentile(chosen, 90)))
        else:
            row = dict(low=float(low), high=float(high), n=0, median=math.nan, p75=math.nan, p90=math.nan)
        row['capability'] = row['p75'] if row['n'] >= MIN_SAMPLES else math.nan
        rows.append(row)
    return rows


def run_series(run_dir):
    """(t, wheel speed, command speed) on a 40 Hz grid over every LB hold of the run."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_analysis.runlog import load_run
    run = load_run(Path(run_dir))
    t_all, v_all, c_all = [], [], []
    for a, b in run.lb_windows():
        grid = np.arange(a, b, 0.025)
        if len(grid) < 40:
            continue
        t_all.append(grid)
        v_all.append(np.interp(grid, run.odom[:, 0], run.odom[:, 1]))
        c_all.append(np.interp(grid, run.drive[:, 0], run.drive[:, 3]))
    return t_all, v_all, c_all


def pooled(runs):
    """Capability rows over several runs: each hold is differentiated on its own, then the samples are pooled
    by concatenating them with gaps (the centred difference never straddles two holds)."""
    t, v, c, offset = [], [], [], 0.0
    for run_dir in runs:
        for tt, vv, cc in zip(*run_series(run_dir)):
            t.append(tt - tt[0] + offset)
            v.append(vv)
            c.append(cc)
            offset += (tt[-1] - tt[0]) + 10.0
    if not t:
        return None
    return np.concatenate(t), np.concatenate(v), np.concatenate(c)


def format_rows(rows):
    lines = ['  speed m/s     n   median     p75     p90   capability']
    for r in rows:
        lines.append(f"  {r['low']:.1f}-{r['high']:.1f}  {r['n']:6d}  {r['median']:7.2f} {r['p75']:7.2f} {r['p90']:7.2f}   "
                     f"{r['capability']:.2f}")
    return '\n'.join(lines)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--car', default='')
    ap.add_argument('--json')
    ap.add_argument('--min-ask', type=float, default=0.5, help='command above wheel speed by at least this, m/s')
    args = ap.parse_args(argv)
    series = pooled(args.runs)
    if series is None:
        print('no LB hold long enough', file=sys.stderr)
        return 1
    accel = accel_capability(*series, min_ask=args.min_ask)
    brake = accel_capability(*series, min_ask=args.min_ask, braking=True)
    print('acceleration (command above wheel speed):\n' + format_rows(accel))
    print('braking (wheel speed above command):\n' + format_rows(brake))
    if args.json:
        Path(args.json).write_text(json.dumps(dict(
            car=args.car, runs=[Path(r).name for r in args.runs],
            criteria=dict(min_ask=args.min_ask, smooth=0.2, capability='p75, bins of 20+ samples'),
            accel=accel, brake=brake), indent=1, allow_nan=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
