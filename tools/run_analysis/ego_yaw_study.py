#!/usr/bin/env python3
"""Does the tracker's ego yaw make walls look like cars? Run OpponentAwareness three ways and compare follow episodes.

    python3 tools/run_analysis/ego_yaw_study.py data/rb1/racerbot_runs/20261111-155542 \\
        --loc ~/.cache/racerbot/loc --map src/pure_pursuit/test/data/course_20261111/map.yaml --line LINE.csv

Variants of the ego motion that gap_follow's opponent tracker dead-reckons (opponent_tracking.EgoPose):
  steering : as shipped, yaw rate = speed * tan(steering estimate) / wheelbase;
  gyro     : yaw rate = /imu/data angular_velocity.z (the topic the node's IMU features read; a sample no older than
             imu_timeout_sec 0.1 s, else the steering model), wheel speed for translation;
  truth    : yaw rate and speed from loc.py's pose (central difference over +-0.1 s).
The follow episodes are the shipped variant's (follow_cost.py). For each, per variant, over the episode's ticks: the
tracker's track nearest the shipped episode's object in the body frame (within 0.6 m): its world speed (median, p90),
the fraction of ticks it is ``moving`` and feeds the follow law, and the fraction of ticks any follow cap applies. For the
truth variant also the map-frame path length of the tracked point (the detection centroid) over the episode: the
sliding-centroid effect. No behaviour change: EgoPose.advance is patched in this process only.
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

from gap_follow import opponent_tracking  # noqa: E402
from run_analysis import runlog  # noqa: E402
from run_analysis.follow_cost import DT, MOVED, WALL_DISTANCE, episodes_of, follow_hook  # noqa: E402
from run_analysis.gap_stalls import line_projector  # noqa: E402
from run_analysis.loc import COL  # noqa: E402
from run_analysis.steer_stages import bag_origin, gyro_series, recorded_params, replay_window  # noqa: E402

CORNER = (-8.1, 9.95)    # m, the inner corner of the U-turn (wall)
MATCH = 0.6              # m, body-frame distance to call a track "the" object
VARIANTS = ("steering", "gyro", "truth")


class Truth:
    """loc.py's pose interpolated: yaw rate and speed by central difference."""

    def __init__(self, loc):
        t = loc[:, COL["t"]]
        keep = np.concatenate([[True], np.diff(t) > 1e-6])
        self.t, self.x, self.y = t[keep], loc[keep, COL["x"]], loc[keep, COL["y"]]
        self.yaw = np.unwrap(loc[keep, COL["yaw"]])

    def pose(self, t):
        return float(np.interp(t, self.t, self.x)), float(np.interp(t, self.t, self.y)), float(np.interp(t, self.t, self.yaw))

    def rates(self, t, h=0.1):
        a, b = self.pose(t - h), self.pose(t + h)
        return (b[2] - a[2]) / (2 * h), math.hypot(b[0] - a[0], b[1] - a[1]) / (2 * h)


def run_variant(name, run, params, gyro, truth, loc):
    ctx = {}
    original = opponent_tracking.EgoPose.advance

    def advance(self, speed, steering, wheelbase, dt):
        if name == "gyro" and ctx.get("yaw") is not None and math.isfinite(ctx["yaw"]):
            yaw_rate, v = ctx["yaw"], speed
        elif name == "truth" and ctx.get("truth") is not None:
            yaw_rate, v = ctx["truth"]
        else:
            return original(self, speed, steering, wheelbase, dt)
        mid = self.yaw + 0.5 * yaw_rate * dt
        self.x += v * math.cos(mid) * dt
        self.y += v * math.sin(mid) * dt
        self.yaw += yaw_rate * dt

    def pre(ctrl, t, yaw, v):
        ctx["yaw"] = yaw
        ctx["truth"] = truth.rates(t)

    def hook(ctrl, row):
        follow_hook(ctrl, row)
        opp = ctrl.opponents
        fed = {(round(m[0], 9), round(m[1], 9)) for m in opp.followed_body()}
        moving = {(round(m[0], 9), round(m[1], 9)) for m in opp.moving_body()}
        tracks = []
        for tr in opp.tracker.tracks:
            bx, by = opp.ego.to_body(tr.x, tr.y)
            key = (round(bx, 9), round(by, 9))
            tracks.append((bx, by, tr.speed, bool(tr.moving), key in fed, key in moving, tr.track_id, bool(tr.confirmed_car),
                           float(tr.stopped_sec), int(tr.streak)))
        row["tracks"] = tracks

    opponent_tracking.EgoPose.advance = advance
    rows = []
    try:
        for press, release in run.lb_windows(min_duration=1.0):
            rows += replay_window(run, params, press + 1e-3, release, gyro, loc=loc, hook=hook, pre_hook=pre)
    finally:
        opponent_tracking.EgoPose.advance = original
    return rows


def episode_metrics(rows_v, rows_i, ep, truth):
    a, b = ep["tick_idx"]
    speeds, moving, fed, applies, n_match = [], 0, 0, 0, 0
    path, last = 0.0, None
    for k in range(a, b + 1):
        o = rows_i[k]["obj"]
        rv = rows_v[k]
        if rv["follow_cap_tel"] < rv["next_lowest"] - 1e-9 and not rv["stopped"]:
            applies += 1
        if o is None:
            continue
        best = min(rv["tracks"], key=lambda tr: math.hypot(tr[0] - o["bx"], tr[1] - o["by"]), default=None)
        if best is None or math.hypot(best[0] - o["bx"], best[1] - o["by"]) > MATCH:
            continue
        n_match += 1
        speeds.append(best[2])
        moving += best[3]
        fed += best[4]
        x, y, yaw = truth.pose(rows_i[k]["t"])
        m = (x + best[0] * math.cos(yaw) - best[1] * math.sin(yaw), y + best[0] * math.sin(yaw) + best[1] * math.cos(yaw))
        if last is not None:
            path += math.hypot(m[0] - last[0], m[1] - last[1])
        last = m
    n = b - a + 1
    return dict(ticks=n, matched=n_match, speed_med=float(np.median(speeds)) if speeds else math.nan,
                speed_p90=float(np.percentile(speeds, 90)) if speeds else math.nan,
                moving=moving / n, fed=fed / n, applies=applies / n, path=path)


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
    rows = {v: run_variant(v, run, params, gyro, truth, loc) for v in VARIANTS}
    eps = episodes_of(rows["steering"], origin, project, dist)
    for e in eps:
        e["metrics"] = {v: episode_metrics(rows[v], rows["steering"], e, truth) for v in VARIANTS}
    return rows, eps


def latch_report(rows_v, rows_i, eps, truth, name):
    """Why a wall episode keeps its follow cap: for the track nearest the episode's object at its middle tick, the first
    tick it was ``moving`` (and the world speed, range and true yaw rate then), how many ticks it was moving, and the
    ticks it fed the cap while not moving (a held, confirmed car: stopped_release_sec)."""
    out = []
    for e in eps:
        a, b = e["tick_idx"]
        mid = (a + b) // 2
        o = rows_i[mid]["obj"]
        if o is None:
            continue
        cand = min(rows_v[mid]["tracks"], key=lambda tr: math.hypot(tr[0] - o["bx"], tr[1] - o["by"]), default=None)
        if cand is None or math.hypot(cand[0] - o["bx"], cand[1] - o["by"]) > MATCH:
            continue
        tid = cand[6]
        life = []
        for k, r in enumerate(rows_v):
            for tr in r["tracks"]:
                if tr[6] == tid:
                    life.append((k, r["t"], tr))
        moving = [x for x in life if x[2][3]]
        held = [x for x in life if x[2][4] and not x[2][3]]
        first = moving[0] if moving else None
        out.append(dict(start=e["start"], dur=e["duration"], track=tid, life_ticks=len(life), moving_ticks=len(moving), held_ticks=len(held),
                        first_moving_rel=(first[1] - rows_v[a]["t"]) if first else math.nan,
                        first_speed=first[2][2] if first else math.nan,
                        first_range=math.hypot(first[2][0], first[2][1]) if first else math.nan,
                        first_yaw_rate=truth.rates(first[1])[0] if first else math.nan,
                        peak_speed=max(x[2][2] for x in life)))
    print(f"LATCH ({name}): why the follow cap holds on the wall episodes")
    print("   start   dur  track  life moving held  first_moving(s from ep start)  speed  range  yaw_rate  peak_speed")
    for d in out:
        print(f"{d['start']:8.2f} {d['dur']:5.2f} {d['track']:6d} {d['life_ticks']:5d} {d['moving_ticks']:6d} {d['held_ticks']:5d} "
              f"{d['first_moving_rel']:12.2f} {d['first_speed']:12.2f} {d['first_range']:6.2f} {d['first_yaw_rate']:8.2f} {d['peak_speed']:8.2f}")
    return out


def classify(eps):
    wall = [e for e in eps if e["kind"] == "wall/static" and math.hypot(e["obj_x"] - CORNER[0], e["obj_y"] - CORNER[1]) < 0.25]
    real = [e for e in eps if e["kind"].startswith("free") and e["moved"] > MOVED]
    return wall, real


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--loc", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--line", required=True)
    args = ap.parse_args(argv)
    rows, eps = study(args.run, args.loc, args.map, args.line)
    wall, real = classify(eps)
    for title, group in (("WALL episodes at the U-turn inner corner", wall), ("REAL moving-car episodes (moved > 0.5 m)", real)):
        print(f"{title}: {len(group)}")
        print("   start   dur     s | " + " | ".join(f"{v:^38}" for v in VARIANTS))
        print("                      | " + " | ".join("  vmed  vp90  moving  fed  cap_applies  path" for _ in VARIANTS))
        for e in group:
            cells = []
            for v in VARIANTS:
                m = e["metrics"][v]
                cells.append(f"{m['speed_med']:5.2f} {m['speed_p90']:5.2f} {m['moving']:6.2f} {m['fed']:5.2f} {m['applies']:7.2f} {m['path']:7.2f}")
            print(f"{e['start']:8.2f} {e['duration']:5.2f} {e['s']:6.2f} | " + " | ".join(cells))
        for v in VARIANTS:
            ms = [e["metrics"][v] for e in group]
            lose = sum(1 for m in ms if m["applies"] <= 0.2)
            keep = sum(1 for m in ms if m["applies"] >= 0.8)
            print(f"  {v:9} episodes losing the cap on >=80% of ticks: {lose}/{len(ms)}; keeping it on >=80%: {keep}/{len(ms)}; "
                  f"median of per-episode speed median {np.nanmedian([m['speed_med'] for m in ms]):.2f} m/s")
    for v in ("steering", "truth"):
        latch_report(rows[v], rows["steering"], wall, Truth(np.load(Path(args.loc).expanduser() / f"{Path(args.run).resolve().name}.npz")["a"]), v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
