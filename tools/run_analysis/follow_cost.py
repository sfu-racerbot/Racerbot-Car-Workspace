#!/usr/bin/env python3
"""What gap_follow's follow cap cost a run, episode by episode, and what it was following.

    python3 tools/run_analysis/follow_cost.py data/rb1/racerbot_runs/20261111-155542 \\
        --loc ~/.cache/racerbot/loc --map src/pure_pursuit/test/data/course_20261111/map.yaml \\
        --line /path/to/car2_course_2026_11_11_planned_line.csv

Replays each LB hold through gap_mirror.GapController with the run's recorded parameters (steer_stages.py). A tick is
"under the follow cap" when the follow cap (the gap-only follow law and the alongside-abort ceiling over the tracked
bodies) is below the next-lowest speed ceiling (curve, clearance, side, and the corner cap on a fallback gap, and the
configured maximum); an episode is a run of such ticks. For each episode: start, duration, the car's course position
``s`` on the line, the follow cap, the next-lowest cap (the speed given up is their difference) and the tracked object
that set the cap, placed in the map frame by loc.py's pose. Each object is classified on the map: within 0.15 m of an
occupied cell is a wall or static object, otherwise free space is a real object (a car); ``moved`` is whether it was
more than 0.5 m from its first to its last sighting in the episode. Two readings of the speed lost: the brief's
(next-lowest cap minus follow cap, summed over the ticks) and a counterfactual replay of the same scans with opponent
awareness off (what the node would have commanded without it, open loop). Times are seconds from the first joy, odom or
drive_intent message.
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
from run_analysis.gap_stalls import line_projector  # noqa: E402
from run_analysis.steer_stages import bag_origin, gyro_series, recorded_params, replay_window  # noqa: E402

WALL_DISTANCE = 0.15     # m, an object this close to an occupied cell is a wall or static
MOVED = 0.5              # m, between first and last sighting
DT = 0.025               # s, the 40 Hz scan


def follow_hook(ctrl, row):
    """Per-tick: the follow cap, the next-lowest ceiling and the body that set the cap, in the body frame."""
    tel = ctrl.telemetry
    opp = ctrl.opponents
    p = ctrl.p
    others = [tel.get("curve_cap", math.inf), tel.get("clearance_cap", math.inf), tel.get("side_cap", math.inf),
              float(p["max_speed"])]
    if tel.get("fallback"):
        others.append(tel.get("corner_cap", math.inf))
    row["next_lowest"] = min(others)
    row["follow_cap_tel"] = tel.get("follow_cap", math.inf)
    row["obj"] = None
    if opp is None or not math.isfinite(row["follow_cap_tel"]):
        return
    passing = ctrl.opponent_mode == "clear"
    lateral = None if passing else opp.params.relevant_lateral
    best = None
    for m in opp.followed_body():
        cap = opp._follow_over([m], lateral if m in opp.moving_body() else opp.params.relevant_lateral,
                               opp._on_my_arc(m[4], m[1]))
        if best is None or cap < best[0]:
            best = (cap, m)
    if best is None:
        return
    m = best[1]
    ident = next((t.track_id for t in opp.tracker.tracks
                  if np.hypot(opp.ego.to_body(t.x, t.y)[0] - m[0], opp.ego.to_body(t.x, t.y)[1] - m[1]) < 1e-9), -1)
    row["obj"] = dict(bx=m[0], by=m[1], vx=m[2], vy=m[3], cap=best[0], track=ident)


def episodes_of(rows, origin, project, dist_to_wall):
    t = np.array([r["t"] for r in rows])
    under = np.array([r["follow_cap_tel"] < r["next_lowest"] - 1e-9 and not r["stopped"] for r in rows])
    out, i = [], 0
    while i < len(rows):
        if not under[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(rows) and under[j + 1] and t[j + 1] - t[j] < 0.1:
            j += 1
        seg = rows[i:j + 1]
        pos = []
        for r in seg:
            o = r["obj"]
            if o is not None and "x" in r:
                c, s_ = math.cos(r["yaw"]), math.sin(r["yaw"])
                pos.append((r["x"] + o["bx"] * c - o["by"] * s_, r["y"] + o["bx"] * s_ + o["by"] * c, o["track"], o["cap"]))
        given = [r["next_lowest"] - r["follow_cap_tel"] for r in seg]
        e = dict(start=seg[0]["t"] - origin, duration=(len(seg)) * DT, ticks=len(seg),
                 s=project(seg[len(seg) // 2].get("x", math.nan), seg[len(seg) // 2].get("y", math.nan))[0]
                 if "x" in seg[0] else math.nan,
                 follow_cap=float(np.median([r["follow_cap_tel"] for r in seg])),
                 next_lowest=float(np.median([r["next_lowest"] for r in seg])),
                 given_up=float(np.mean(given)), metres_brief=float(np.sum(given) * DT), tick_idx=(i, j))
        if pos:
            arr = np.array(pos)
            e.update(obj_x=float(np.median(arr[:, 0])), obj_y=float(np.median(arr[:, 1])),
                     moved=float(np.hypot(arr[-1, 0] - arr[0, 0], arr[-1, 1] - arr[0, 1])),
                     tracks=sorted({int(k) for k in arr[:, 2]}), wall_distance=dist_to_wall(np.median(arr[:, 0]), np.median(arr[:, 1])))
            e["kind"] = "wall/static" if e["wall_distance"] <= WALL_DISTANCE else "free space (real object)"
        else:
            e.update(obj_x=math.nan, obj_y=math.nan, moved=math.nan, tracks=[], wall_distance=math.nan, kind="no object (memory/held)")
        out.append(e)
        i = j + 1
    return out


def analyse(run_dir, loc_dir, map_yaml, line_csv, static_gate=False):
    from scipy.ndimage import distance_transform_edt
    from pure_pursuit.occupancy_map import OCCUPIED, OccupancyMap
    run_dir = Path(run_dir).resolve()
    run = runlog.load_run(run_dir)
    origin = bag_origin(run_dir)
    loc = np.load(Path(loc_dir).expanduser() / f"{run_dir.name}.npz")["a"]
    project, _length = line_projector(line_csv)
    grid = OccupancyMap.from_yaml(map_yaml, inflate_cells=0)
    field = distance_transform_edt(grid.grid != OCCUPIED) * grid.resolution

    def dist_to_wall(x, y):
        c, r = grid.world_to_cell(x, y)
        return float(field[int(r), int(c)]) if 0 <= int(r) < field.shape[0] and 0 <= int(c) < field.shape[1] else math.inf
    params, gyro = recorded_params(run_dir), gyro_series(run_dir)
    if static_gate:      # the run predates the gate: switch it on at the shipped limits (gap_follow.yaml)
        params.update(opponent_follow_static_radial_speed=0.35, opponent_follow_static_tangential_speed=1.5)
    off = dict(params, enable_opponent_awareness=False)
    rows, rows_off = [], []
    for press, release in run.lb_windows(min_duration=1.0):
        rows += replay_window(run, params, press + 1e-3, release, gyro, loc=loc, hook=follow_hook)
        rows_off += replay_window(run, off, press + 1e-3, release, gyro, loc=loc)
    eps = episodes_of(rows, origin, project, dist_to_wall)
    for e in eps:
        a, b = e["tick_idx"]
        lost = [max(0.0, rows_off[k]["speed_cmd"] - rows[k]["speed_cmd"]) for k in range(a, b + 1)]
        e["metres_counterfactual"] = float(np.sum(lost) * DT)
    return rows, eps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--loc", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--line", required=True)
    ap.add_argument("--static-gate", action="store_true", help="replay with the static-confirmation gate on (S 0.35, T 1.5)")
    args = ap.parse_args(argv)
    rows, eps = analyse(args.run, args.loc, args.map, args.line, args.static_gate)
    print(f"{args.run}: {len(rows)} LB-held ticks ({len(rows) * DT:.1f} s), {len(eps)} follow episodes")
    print("  start   dur     s  follow next  given  m(brief) m(cf)   obj x     y    wall_d moved tracks     kind")
    for e in eps:
        print(f"{e['start']:7.2f} {e['duration']:5.2f} {e['s']:6.2f} {e['follow_cap']:6.2f} {e['next_lowest']:5.2f} {e['given_up']:6.2f} "
              f"{e['metres_brief']:7.3f} {e['metres_counterfactual']:6.3f} {e['obj_x']:7.2f} {e['obj_y']:6.2f} {e['wall_distance']:6.2f} "
              f"{e['moved']:5.2f} {str(e['tracks'])[:10]:10} {e['kind']}")
    tot = lambda f, k: sum(e[k] for e in eps if f(e))
    for name, f in (("all", lambda e: True), ("wall/static", lambda e: e["kind"] == "wall/static"),
                    ("real (free space)", lambda e: e["kind"].startswith("free")),
                    ("  of which moved > 0.5 m", lambda e: e["kind"].startswith("free") and e["moved"] > MOVED),
                    ("  of which did not move", lambda e: e["kind"].startswith("free") and e["moved"] <= MOVED),
                    ("no object", lambda e: e["kind"].startswith("no object"))):
        n = sum(1 for e in eps if f(e))
        print(f"{name:20} episodes {n:3d}  time {tot(f, 'duration'):7.2f} s  progress lost: brief {tot(f, 'metres_brief'):7.2f} m, "
              f"counterfactual {tot(f, 'metres_counterfactual'):7.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
