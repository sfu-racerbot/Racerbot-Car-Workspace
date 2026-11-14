#!/usr/bin/env python3
"""How well the car's speed follows the speed the node commands (real runs and the simulator, same functions).

    python3 tools/run_analysis/speed_tracking.py ~/racerbot-ws/data/rb1/racerbot_runs/20261111-155542 [--json out.json]

Reads /drive (the command the node published) and /odom (wheel speed) with runlog's loaders, per LB hold, and reports

  a. braking: over samples with cmd >= 0.3, dv/dt (central difference over 0.1 s) in bins of e = v - cmd;
  b. step-downs: the command falls by >= 0.8 m/s within 0.15 s and stays within +-0.3 of the new value for >= 0.4 s: the time until
     v is within 0.3 m/s of the new command (NaN = not reached; then the remaining error 0.4 s after the step);
  c. overshoots: the command rises >= 1.0 m/s within 0.6 s, then falls >= 0.5 m/s within the next 0.4 s: max(v - cmd) over the
     following 1.0 s and the time v stays above cmd + 0.3 m/s in that window;
  d. command jitter: |cmd[k] - cmd[k-1]| at speeds above 1.5 m/s.

The functions take plain arrays (command stamps and values, speed stamps and values), no bag I/O, so the simulator's per-tick trace goes
through exactly the same code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

BINS = ((0.2, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, math.inf))
STEP_DROP, STEP_WITHIN, STEP_HOLD, STEP_BAND, STEP_REACHED = 0.8, 0.15, 0.4, 0.3, 0.3
RISE, RISE_WITHIN, FALL, FALL_WITHIN, OVERSHOOT_WINDOW, OVERSHOOT_BAND = 1.0, 0.6, 0.5, 0.4, 1.0, 0.3


def hold(cmd_t, cmd, t):
    """Zero-order hold of the command onto the stamps ``t`` (NaN before the first command)."""
    i = np.searchsorted(cmd_t, t, side="right") - 1
    out = np.where(i >= 0, np.asarray(cmd, dtype=float)[np.maximum(i, 0)], np.nan)
    return out


def _clean(t, x):
    t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    ok = np.isfinite(t) & np.isfinite(x)
    return t[ok], x[ok]


def _stats(values):
    v = np.asarray([x for x in values if math.isfinite(x)], dtype=float)
    if not v.size:
        return dict(n=0, median=math.nan, p25=math.nan, p75=math.nan, p90=math.nan)
    return dict(n=int(v.size), median=float(np.median(v)), p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)),
                p90=float(np.percentile(v, 90)))


def braking_bins(v_t, v, cmd_t, cmd, bins=BINS, half=0.05):
    """{bin: stats of dv/dt} over samples with cmd >= 0.3, dv/dt the central difference over 2*half seconds."""
    v_t, v = _clean(v_t, v)
    cmd_t, cmd = _clean(cmd_t, cmd)
    if v_t.size < 3 or not cmd_t.size:
        return {b: _stats([]) for b in bins}
    c = hold(cmd_t, cmd, v_t)
    e = v - c
    dvdt = (np.interp(v_t + half, v_t, v) - np.interp(v_t - half, v_t, v)) / (2 * half)
    usable = (c >= 0.3) & (v_t - half >= v_t[0]) & (v_t + half <= v_t[-1])
    return {b: _stats(dvdt[usable & (e >= b[0]) & (e < b[1])]) for b in bins}


def step_down_events(v_t, v, cmd_t, cmd):
    """List of dicts (t, new, time_to_reach, remaining): the command fell by >= 0.8 m/s within 0.15 s to a value >= 0.3 that it
    held (+-0.3) for >= 0.4 s. time_to_reach is NaN when v does not get within 0.3 m/s of the new command while it holds."""
    v_t, v = _clean(v_t, v)
    cmd_t, cmd = _clean(cmd_t, cmd)
    out = []
    k, n = 0, cmd_t.size
    while k < n:
        j = np.searchsorted(cmd_t, cmd_t[k] - STEP_WITHIN - 1e-9)
        window_max = cmd[j:k + 1].max()
        new = cmd[k]
        if k > 0 and window_max - new >= STEP_DROP - 1e-9 and new >= 0.3 and cmd[k - 1] > new:
            end = np.searchsorted(cmd_t, cmd_t[k] + STEP_HOLD - 1e-9)
            seg = cmd[k:end + 1]
            if cmd_t[-1] >= cmd_t[k] + STEP_HOLD - 1e-9 and np.all(np.abs(seg - new) <= STEP_BAND + 1e-9):
                # how long the command stays near the new value, at most until it leaves the band
                stay = k
                while stay + 1 < n and abs(cmd[stay + 1] - new) <= STEP_BAND + 1e-9:
                    stay += 1
                t0 = cmd_t[k]
                grid = v_t[(v_t >= t0) & (v_t <= cmd_t[stay])]
                reached = grid[np.interp(grid, v_t, v) - new <= STEP_REACHED] if grid.size else grid
                if reached.size:
                    out.append(dict(t=float(t0), new=float(new), time_to_reach=float(reached[0] - t0), remaining=0.0))
                else:
                    t_end = min(t0 + STEP_HOLD, v_t[-1]) if v_t.size else t0
                    out.append(dict(t=float(t0), new=float(new), time_to_reach=math.nan,
                                    remaining=float(np.interp(t_end, v_t, v) - new) if v_t.size else math.nan))
                k = np.searchsorted(cmd_t, t0 + STEP_HOLD)
                continue
        k += 1
    return out


def overshoot_events(v_t, v, cmd_t, cmd):
    """List of dicts (t, max_over, time_over): a peak of the command reached by a rise of >= 1.0 m/s within 0.6 s and followed by a
    fall of >= 0.5 m/s within 0.4 s. max_over = max(v - cmd) over the following 1.0 s; time_over = seconds in that window with
    v > cmd + 0.3."""
    v_t, v = _clean(v_t, v)
    cmd_t, cmd = _clean(cmd_t, cmd)
    out, last = [], -math.inf
    for p in range(cmd_t.size):
        t = cmd_t[p]
        if t - last < FALL_WITHIN + RISE_WITHIN:
            continue
        before = cmd[np.searchsorted(cmd_t, t - RISE_WITHIN - 1e-9):p + 1]
        after = cmd[p:np.searchsorted(cmd_t, t + FALL_WITHIN + 1e-9)]
        if cmd[p] < after.max() - 1e-12 or cmd[p] - before.min() < RISE - 1e-9 or cmd[p] - after.min() < FALL - 1e-9:
            continue
        grid = v_t[(v_t >= t) & (v_t <= t + OVERSHOOT_WINDOW)]
        if grid.size < 2:
            continue
        e = np.interp(grid, v_t, v) - hold(cmd_t, cmd, grid)
        dt = np.diff(grid, prepend=grid[0])
        out.append(dict(t=float(t), max_over=float(np.nanmax(e)), time_over=float(np.sum(dt[e > OVERSHOOT_BAND]))))
        last = t
    return out


def jitter(cmd_t, cmd, v_t, v, min_speed=1.5):
    """Stats of |cmd[k] - cmd[k-1]| over ticks where the speed is above ``min_speed``."""
    cmd_t, cmd = _clean(cmd_t, cmd)
    v_t, v = _clean(v_t, v)
    if cmd_t.size < 2 or v_t.size < 2:
        return _stats([])
    fast = np.interp(cmd_t[1:], v_t, v) > min_speed
    return _stats(np.abs(np.diff(cmd))[fast])


def summarise(segments):
    """segments: list of (cmd_t, cmd, v_t, v), one per driving hold; events and bins are pooled."""
    pooled = {b: [] for b in BINS}
    steps, overs, jit = [], [], []
    for cmd_t, cmd, v_t, v in segments:
        v_t, v = _clean(v_t, v)
        cmd_t, cmd = _clean(cmd_t, cmd)
        if v_t.size >= 3 and cmd_t.size:
            c = hold(cmd_t, cmd, v_t)
            e = v - c
            half = 0.05
            dvdt = (np.interp(v_t + half, v_t, v) - np.interp(v_t - half, v_t, v)) / (2 * half)
            ok = (c >= 0.3) & (v_t - half >= v_t[0]) & (v_t + half <= v_t[-1])
            for b in BINS:
                pooled[b] += list(dvdt[ok & (e >= b[0]) & (e < b[1])])
        steps += step_down_events(v_t, v, cmd_t, cmd)
        overs += overshoot_events(v_t, v, cmd_t, cmd)
        if cmd_t.size > 1 and v_t.size > 1:
            fast = np.interp(cmd_t[1:], v_t, v) > 1.5
            jit += list(np.abs(np.diff(cmd))[fast])
    reached = [s["time_to_reach"] for s in steps if math.isfinite(s["time_to_reach"])]
    return dict(
        braking={f"{b[0]}-{b[1]}": _stats(pooled[b]) for b in BINS},
        step_down=dict(n=len(steps), reached=len(reached), time=_stats(reached),
                       not_reached=[dict(t=s["t"], remaining=s["remaining"]) for s in steps if not math.isfinite(s["time_to_reach"])],
                       events=steps),
        overshoot=dict(n=len(overs), max_over=_stats([o["max_over"] for o in overs]),
                       time_over=_stats([o["time_over"] for o in overs]), times=[o["t"] for o in overs], events=overs),
        jitter=_stats(jit))


def run_segments(run_dir):
    """Per LB hold of a real run: (cmd_t, cmd, v_t, v) with /drive and /odom receive times (the clock of intent_report)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_analysis import runlog
    run = runlog.load_run(run_dir)
    origin = float(min(run.joy[0, 0] if len(run.joy) else math.inf, run.odom[0, 0], run.drive[0, 0]))
    segs = []
    for press, release in run.lb_windows(min_duration=1.0):
        d = (run.drive[:, 0] >= press) & (run.drive[:, 0] <= release)
        o = (run.odom[:, 0] >= press) & (run.odom[:, 0] <= release)
        segs.append((run.drive[d, 0] - origin, run.drive[d, 3], run.odom[o, 0] - origin, run.odom[o, 1]))
    return segs


def print_summary(name, s):
    print(f"== {name}")
    print("a. braking dv/dt (m/s^2) by e = v - cmd:")
    for b, st in s["braking"].items():
        print(f"   e {b:>8}: n {st['n']:5d}  median {st['median']:6.2f}  p25 {st['p25']:6.2f}  p75 {st['p75']:6.2f}")
    sd = s["step_down"]
    print(f"b. step-downs: n {sd['n']}, reached {sd['reached']}; time to within 0.3 m/s: median {sd['time']['median']:.3f} s, p90 {sd['time']['p90']:.3f} s; "
          f"not reached: {[(round(x['t'], 2), round(x['remaining'], 2)) for x in sd['not_reached']]}")
    ov = s["overshoot"]
    print(f"c. overshoots: n {ov['n']}; max(v - cmd) median {ov['max_over']['median']:.2f} p90 {ov['max_over']['p90']:.2f} m/s; time over cmd+0.3: "
          f"median {ov['time_over']['median']:.2f} p90 {ov['time_over']['p90']:.2f} s; at {[round(t, 2) for t in ov['times']]}")
    j = s["jitter"]
    print(f"d. command jitter at > 1.5 m/s: n {j['n']}, |cmd[k]-cmd[k-1]| p50 {j['median']:.3f} p90 {j['p90']:.3f} m/s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    s = summarise(run_segments(args.run))
    print_summary(Path(args.run).name, s)
    if args.json:
        Path(args.json).write_text(json.dumps(s, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
