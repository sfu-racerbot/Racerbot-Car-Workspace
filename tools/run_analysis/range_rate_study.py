#!/usr/bin/env python3
"""Range-rate test: does d(range)/dt separate a wall from a car without any dead-reckoned heading?

    python3 tools/run_analysis/range_rate_study.py data/rb1/racerbot_runs/20261111-155542 \\
        --loc ~/.cache/racerbot/loc --map src/pure_pursuit/test/data/course_20261111/map.yaml --line LINE.csv

For every tracked object and scan, residual = d(near_range)/dt + v cos(b) + omega * laser_x * sin(b), b the body-frame
bearing of the object's NEAREST beam from the LiDAR, near_range its range, v the wheel speed and omega the yaw rate.
For any static surface the residual is zero (the range changes only because the LiDAR moves toward it); for a car
moving away along the line of sight at u it is +u. d(near_range)/dt is the difference between this tick's and the previous
tick's associated detections (both consecutive; dt in (0, 0.1] s), smoothed by a running median over the last 8 valid
ticks (4 needed). omega from the gyro (/imu/data, fresh within 0.1 s, else the steering model) and from the steering model
alone. Detection bearing is derived here from the scan; the shipped segment_objects is untouched. Episodes are
ego_yaw_study.py's (the shipped follow episodes): wall = the U-turn inner corner, real = free-space objects that moved
over 0.5 m. Association of a track to its detection is exact (the tracker stores the detection's near_dx and near_range).
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
from run_analysis.ego_yaw_study import MATCH, Truth, classify, run_variant  # noqa: E402
from run_analysis.follow_cost import episodes_of  # noqa: E402
from run_analysis.gap_stalls import line_projector  # noqa: E402
from run_analysis.steer_stages import bag_origin, gyro_series, recorded_params  # noqa: E402

WINDOW, MIN_SAMPLES = 8, 4
THRESHOLDS = (0.8, 0.5)
DT_MAX = 0.25            # s, a larger gap between two sightings is no sample (the shipped gate's rule)
EXPIRY, NEWEST, NEED = 0.6, 8, 4


def make_extra(run, wheelbase, laser_x):
    """Hook: per tick, each associated track's raw and smoothed residual (gyro omega and steering-model omega)."""
    angles = run.beam_angles()
    state = {}

    def extra(ctrl, row):
        opp = ctrl.opponents
        si = int(np.searchsorted(run.scan_t, row["t"]))
        out = {}
        for tr in opp.tracker.tracks:
            if tr.misses != 0:
                state.pop(tr.track_id, None)
                continue
            det = next((d for d in opp.detections if d.near_x - d.x == tr.near_dx and d.near_range == tr.near_range), None)
            if det is None:
                state.pop(tr.track_id, None)
                continue
            seg = np.asarray(run.scan[si][det.lo:det.hi + 1], dtype=float)
            seg = np.where(np.isfinite(seg) & (seg >= run.range_min), seg, np.inf)
            b = float(angles[det.lo + int(np.argmin(seg))])
            st = state.setdefault(tr.track_id, dict(prev=None, bg=[], bs=[]))
            v = row["speed"]
            omega_g = row["yaw_rate"] if math.isfinite(row["yaw_rate"]) else None
            omega_s = v * math.tan(opp.steer.angle) / wheelbase
            res = {}
            if st["prev"] is not None:
                dt = row["t"] - st["prev"][0]
                if 0.0 < dt <= DT_MAX:
                    rate = (det.near_range - st["prev"][1]) / dt
                    for key, buf, om in (("g", st["bg"], omega_g if omega_g is not None else omega_s), ("s", st["bs"], omega_s)):
                        buf.append(rate + v * math.cos(b) + om * laser_x * math.sin(b))
                        del buf[:-WINDOW]
                        res[key] = (buf[-1], float(np.median(buf)) if len(buf) >= MIN_SAMPLES else math.nan)
            st["prev"] = (row["t"], det.near_range)
            out[tr.track_id] = dict(res=res, bearing=b)
        row["rr"] = out
    return extra


def study(run_dir, loc_dir, map_yaml, line_csv):
    from scipy.ndimage import distance_transform_edt
    from pure_pursuit.occupancy_map import OCCUPIED, OccupancyMap
    run_dir = Path(run_dir).resolve()
    run = runlog.load_run(run_dir)
    origin = bag_origin(run_dir)
    loc = np.load(Path(loc_dir).expanduser() / f"{run_dir.name}.npz")["a"]
    truth = Truth(loc)
    project, _ = line_projector(line_csv)
    grid = OccupancyMap.from_yaml(map_yaml, inflate_cells=0)
    field = distance_transform_edt(grid.grid != OCCUPIED) * grid.resolution

    def dist(x, y):
        c, r = grid.world_to_cell(x, y)
        r, c = int(r), int(c)
        return float(field[r, c]) if 0 <= r < field.shape[0] and 0 <= c < field.shape[1] else math.inf
    params, gyro = recorded_params(run_dir), gyro_series(run_dir)
    rows = run_variant("steering", run, params, gyro, truth, loc, extra=make_extra(run, params["wheelbase"], params["laser_offset_x"]))
    eps = episodes_of(rows, origin, project, dist)
    return rows, eps


def episode_stats(rows, ep, key):
    a, b = ep["tick_idx"]
    vals = []
    for k in range(a, b + 1):
        o = rows[k]["obj"]
        if o is None:
            vals.append(math.nan)
            continue
        best = min(rows[k]["tracks"], key=lambda tr: math.hypot(tr[0] - o["bx"], tr[1] - o["by"]), default=None)
        if best is None or math.hypot(best[0] - o["bx"], best[1] - o["by"]) > MATCH:
            vals.append(math.nan)
            continue
        r = rows[k]["rr"].get(best[6], {}).get("res", {}).get(key)
        vals.append(r[1] if r else math.nan)
    v = np.array(vals)
    good = v[np.isfinite(v)]
    out = dict(n=len(v), valid=len(good))
    out["median"], out["p10"], out["p90"] = ((float(np.median(good)), float(np.percentile(good, 10)), float(np.percentile(good, 90)))
                                             if good.size else (math.nan,) * 3)
    for T in THRESHOLDS:
        out[f"ge{T}"] = float(np.sum(good >= T)) / len(v)             # NaN ticks count as below
        out[f"lt{T}"] = 1.0 - out[f"ge{T}"]
    return out


def report(rows, eps, label):
    wall, real = classify(eps)
    print(f"== {label}: {len(wall)} wall episodes, {len(real)} real moving-car episodes")
    verdict = {}
    for key, name in (("g", "gyro omega"), ("s", "steering-model omega")):
        for title, group in (("WALL", wall), ("REAL", real)):
            print(f"-- {title}, {name}")
            print("   start   dur   n valid |  median    p10    p90 | frac>=0.8 frac>=0.5")
            stats = []
            for e in group:
                st = episode_stats(rows, e, key)
                stats.append(st)
                print(f"{e['start']:8.2f} {e['duration']:5.2f} {st['n']:3d} {st['valid']:5d} | {st['median']:7.2f} {st['p10']:6.2f} {st['p90']:6.2f} | {st['ge0.8']:8.2f} {st['ge0.5']:9.2f}")
            verdict[(key, title)] = stats
        for T in THRESHOLDS:
            wall_ok = sum(1 for st in verdict[(key, "WALL")] if st[f"lt{T}"] >= 0.95)
            real_ok = sum(1 for st in verdict[(key, "REAL")] if st[f"ge{T}"] >= 0.70)
            print(f"   [{name}] T={T}: wall episodes below T on >=95% of ticks: {wall_ok}/{len(wall)}; "
                  f"real episodes at or above T on >=70% of ticks: {real_ok}/{len(real)}")
    return verdict


def static_confirmed_series(rows, track_id, k, S):
    """Static-confirmed at tick k for ``track_id``: >= NEED raw steering-model residual samples among the newest NEWEST
    no older than EXPIRY s, median |residual| < S. Anything else is False (not confirmed). Also the window median."""
    t_now = rows[k]["t"]
    samples = []
    j = k
    while j >= 0 and rows[j]["t"] >= t_now - EXPIRY and len(samples) < NEWEST:
        r = rows[j]["rr"].get(track_id, {}).get("res", {}).get("s")
        if r is not None and math.isfinite(r[0]):
            samples.append(r[0])
        j -= 1
    if len(samples) < NEED:
        return False, math.nan
    med = float(np.median(np.abs(samples)))
    return med < S, med


def gate_report(rows, eps, label, S_values=(0.5, 0.35)):
    wall, real = classify(eps)
    out = {}
    for S in S_values:
        wt = wc = 0
        wall_rows = []
        for e in wall:
            a, b = e["tick_idx"]
            n = c = 0
            for k in range(a, b + 1):
                o = rows[k]["obj"]
                n += 1
                if o is not None and static_confirmed_series(rows, o["track"], k, S)[0]:
                    c += 1
            wt += n
            wc += c
            wall_rows.append((e["start"], n, c))
        real_rows = []
        for e in real:
            a, b = e["tick_idx"]
            n = c = longest = run = 0
            mins = []
            for k in range(a, b + 1):
                o = rows[k]["obj"]
                n += 1
                conf = False
                if o is not None:
                    conf, med = static_confirmed_series(rows, o["track"], k, S)
                    if math.isfinite(med):
                        mins.append(med)
                c += conf
                run = run + 1 if conf else 0
                longest = max(longest, run)
            real_rows.append((e["start"], n, c, longest * DT_TICK, min(mins) if mins else math.nan))
        out[S] = (wt, wc, wall_rows, real_rows)
        print(f"== GATE {label} S={S}: wall ticks confirmed {wc}/{wt} = {wc / max(wt, 1):.3f}")
        for st, n, c in wall_rows:
            print(f"   wall {st:8.2f}  {c:3d}/{n:3d} = {c / n:.2f}")
        for st, n, c, lng, mn in real_rows:
            print(f"   REAL {st:8.2f}  confirmed {c:3d}/{n:3d} = {c / n:.3f}  longest run {lng:.3f}s  min full-window median {mn:.2f}")
    return out


DT_TICK = 0.025


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--loc", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--line", required=True)
    args = ap.parse_args(argv)
    rows, eps = study(args.run, args.loc, args.map, args.line)
    report(rows, eps, Path(args.run).name)
    gate_report(rows, eps, Path(args.run).name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
