#!/usr/bin/env python3
"""Fit the simulator's speed loop to a car's real runs: acceleration gain and limit, braking deadband, gain and
limit, and the transport delay.

    python3 tools/run_analysis/fit_speed_loop.py data/rb1/racerbot_runs/20261111-155542 \\
        data/rb1/racerbot_runs/20261111-153721 --car rb1 --table --json OUT.json

Method. Every LB hold of each run gives the commanded speed (/drive) and the wheel speed (/odom), resampled on
the plant's 0.025 s control grid (command held, wheel speed interpolated). The speed loop ALONE is replayed, one
dimension, no steering and no map: v[k+1] = v[k] + sim_fidelity.actuation.speed_loop_accel(cal, cmd[k - delay],
v[k]) * dt, the same function the plant calls. So that model error cannot drift over a long hold, the replay
restarts at the real wheel speed every 0.25 s and runs 1.0 s ahead (moving horizon). Score = RMS of (replayed -
real) wheel speed over every horizon sample whose command is at least 0.3 m/s, every run pooled; the same RMS
over the accelerating samples (command > real + 0.2) and the braking samples (real > command + 0.2) is reported
beside it. The grid is the product of the six parameters below; the chosen set has the lowest score.

``--table`` prints speed_tracking's rows a-d (braking by excess, step-downs, overshoots, jitter) for the real runs,
for the shared speed loop and for the chosen set, each replayed over whole LB holds (no restarts).
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "tools", ROOT / "tools" / "f1tenth_sim"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sim_fidelity.actuation import speed_loop_accel  # noqa: E402
from sim_fidelity.calibration import SHARED_SPEED_LOOP, calibration_for_car  # noqa: E402

DT = 0.025                 # s, the plant's control tick (eval_gap.CONTROL_DT)
RESTART, HORIZON = 0.25, 1.0
MIN_CMD, MIN_EXCESS = 0.3, 0.2

GRID = dict(
    speed_loop_accel_gain=(10.0, 14.0, 20.0, 30.0),
    speed_loop_accel_max=(5.0, 5.5, 6.0),
    speed_loop_brake_deadband=(0.0, 0.2, 0.3, 0.4),
    speed_loop_brake_gain=(2.0, 3.0, 4.0),
    speed_loop_brake_max=(3.0, 4.0),
    speed_delay_ticks=(1, 2, 3),
)
LOOP_FIELDS = tuple(GRID)


def to_grid(cmd_t, cmd, v_t, v):
    """(t, cmd, v) on a DT grid over the overlap of both series; the command held (zero-order), v interpolated."""
    cmd_t, cmd, v_t, v = (np.asarray(x, dtype=float) for x in (cmd_t, cmd, v_t, v))
    ok_c = np.isfinite(cmd_t) & np.isfinite(cmd)
    ok_v = np.isfinite(v_t) & np.isfinite(v)
    cmd_t, cmd, v_t, v = cmd_t[ok_c], cmd[ok_c], v_t[ok_v], v[ok_v]
    if cmd_t.size == 0 or v_t.size < 2:
        return np.empty(0), np.empty(0), np.empty(0)
    t0, t1 = max(cmd_t[0], v_t[0]), min(cmd_t[-1], v_t[-1])
    if t1 - t0 < DT:
        return np.empty(0), np.empty(0), np.empty(0)
    t = t0 + DT * np.arange(int(math.floor((t1 - t0) / DT)) + 1)
    idx = np.searchsorted(cmd_t, t, side="right") - 1
    return t, cmd[np.clip(idx, 0, cmd.size - 1)], np.interp(t, v_t, v)


def replay(cal, cmd, v0, delay, start, steps):
    """The speed loop alone from v0 at grid index ``start`` for ``steps`` ticks; cmd[k - delay] drives tick k (the
    command history before ``start`` is the real one). Returns the speeds at start .. start + steps."""
    out = np.empty(steps + 1)
    v = out[0] = float(v0)
    for j in range(steps):
        k = max(start + j - delay, 0)
        v = v + speed_loop_accel(cal, float(cmd[k]), v) * DT
        out[j + 1] = v
    return out


def scores(cal, grids):
    """{'all', 'accel', 'brake'}: RMS of replayed - real wheel speed over the moving-horizon replays' samples."""
    every, steps = int(round(RESTART / DT)), int(round(HORIZON / DT))
    parts = {"all": [], "accel": [], "brake": []}
    for _t, cmd, v in grids:
        for s in range(0, len(v) - steps, every):
            sim = replay(cal, cmd, v[s], cal.speed_delay_ticks, s, steps)
            real, c = v[s:s + steps + 1], cmd[s:s + steps + 1]
            ok, e = c >= MIN_CMD, sim - real
            parts["all"].append(e[ok])
            parts["accel"].append(e[ok & (c - real > MIN_EXCESS)])
            parts["brake"].append(e[ok & (real - c > MIN_EXCESS)])
    out = {}
    for name, chunks in parts.items():
        e = np.concatenate(chunks) if chunks else np.empty(0)
        out[name] = float(np.sqrt(np.mean(e * e))) if e.size else math.nan
        out[f"n_{name}"] = int(e.size)
    return out


def _job(job):
    base, grids, values = job
    cal = dataclasses.replace(base, **dict(zip(LOOP_FIELDS, values)))
    return dict(zip(LOOP_FIELDS, values), **scores(cal, grids))


def shared_loop(base):
    """``base`` with the shared speed-loop values (calibration.SHARED_SPEED_LOOP)."""
    return dataclasses.replace(base, **{name: SHARED_SPEED_LOOP[name] for name in LOOP_FIELDS})


def whole_hold_segments(cal, grids):
    """speed_tracking segments of a replay of each whole hold (no restarts): (t, cmd, t, replayed v)."""
    return [(t, cmd, t, replay(cal, cmd, v[0], cal.speed_delay_ticks, 0, len(v) - 1)) for t, cmd, v in grids]


def load_grids(runs):
    from run_analysis import speed_tracking
    grids, real_segments = [], []
    for run in runs:
        for seg in speed_tracking.run_segments(run):
            g = to_grid(*seg)
            if g[0].size > int(round(HORIZON / DT)) + 1:
                grids.append(g)
                real_segments.append(seg)
    return grids, real_segments


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--car", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--json")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args(argv)
    from run_analysis import speed_tracking
    shared = shared_loop(calibration_for_car(args.car))
    grids, real_segments = load_grids(args.runs)
    jobs = [(shared, grids, values) for values in itertools.product(*GRID.values())]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_job, jobs, chunksize=8))
    rows.sort(key=lambda r: r["all"])
    today = _job((shared, grids, tuple(getattr(shared, f) for f in LOOP_FIELDS)))
    best = rows[0]
    nodb = next(r for r in rows if r["speed_loop_brake_deadband"] == 0.0)

    def line(r):
        return (f"all {r['all']:.4f} accel {r['accel']:.4f} brake {r['brake']:.4f} | accel gain {r['speed_loop_accel_gain']:g} "
                f"max {r['speed_loop_accel_max']:g} | brake deadband {r['speed_loop_brake_deadband']:g} gain "
                f"{r['speed_loop_brake_gain']:g} max {r['speed_loop_brake_max']:g} | delay {r['speed_delay_ticks']}")
    print(f"{len(grids)} holds from {len(args.runs)} runs; samples: all {best['n_all']}, accel {best['n_accel']}, brake {best['n_brake']}")
    print("shared values: " + line(today))
    print("top 10:")
    for r in rows[:10]:
        print("   " + line(r))
    print("best without a deadband: " + line(nodb))
    print(f"chosen against shared: all {100.0 * (1.0 - best['all'] / today['all']):.1f}% lower, "
          f"accel {100.0 * (1.0 - best['accel'] / today['accel']):.1f}%, brake {100.0 * (1.0 - best['brake'] / today['brake']):.1f}%")
    out = dict(car=args.car, runs=[Path(r).name for r in args.runs], method=__doc__.split("Method. ")[1].split("``--table``")[0].strip(),
               grid=GRID, shared=today, best_without_deadband=nodb, top=rows[:10],
               chosen={f: best[f] for f in LOOP_FIELDS}, chosen_scores={k: v for k, v in best.items() if k not in LOOP_FIELDS})
    if args.table:
        chosen = dataclasses.replace(shared, **out["chosen"])
        tables = {}
        for name, cal in (("real", None), ("shared", shared), ("chosen", chosen)):
            segs = real_segments if cal is None else whole_hold_segments(cal, grids)
            tables[name] = s = speed_tracking.summarise(segs)
            speed_tracking.print_summary(name if cal is None else f"{name} (1-D replay)", s)
        out["tables"] = {k: dict(braking=v["braking"],
                                 step_down=dict(n=v["step_down"]["n"], reached=v["step_down"]["reached"], time=v["step_down"]["time"]),
                                 overshoot=dict(n=v["overshoot"]["n"], max_over=v["overshoot"]["max_over"],
                                                time_over=v["overshoot"]["time_over"]),
                                 jitter=v["jitter"])
                         for k, v in tables.items()}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
