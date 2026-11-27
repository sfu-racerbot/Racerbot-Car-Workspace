#!/usr/bin/env python3
"""Localization: how far the car's own odometry drifts from a map pose.

    python3 tools/run_analysis/dataset_examples/localization.py data/rb1/racerbot_runs/20261026-194514 \\
        --map data/rb1/racerbot_auto/20261026-194514-clean

Model (relative pose error). Odometry is only trusted over short distances, so
it is compared with the map pose one stretch at a time. For a stretch from i to
j, each source's motion is expressed in the car's own frame at i:

    dT_src = T_src(i)^-1 T_src(j)     (dx, dy, dyaw in base_link at i)
    translation drift = |dp_odom - dp_map| / distance driven      (%)
    heading drift     = wrap(dyaw_odom - dyaw_map) / distance driven (deg/m)

Two odometries are compared: /odom (wheel speed, heading from the steering
command through the bicycle model) and /odom/imu (wheel speed, heading from
the gyro). The reference is the map pose: SLAM's /slam_pose, else the particle
filter's /pf/viz/inferred_pose moved from the LiDAR to base_link (common.map_pose).
With --map, loc.py's LiDAR match on that map is a second, independent estimate
(not ground truth).

Stretches touching a jump of the map pose (a SLAM correction, a
relocalization), LB released, or the car under 0.3 m/s are left out.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_analysis.dataset_examples import common as c  # noqa: E402

TOPICS = ["/odom", "/odom/imu", *c.MAP_POSES, "/imu/data", "/joy"]
SOURCES = {"/odom": "wheels + steering model", "/odom/imu": "wheels + gyro"}
STRETCHES_M = (2.0, 5.0)  # driven distance per compared stretch
STEP_M = 0.5              # a new stretch starts every this many metres


def body_delta(x0, y0, yaw0, x1, y1, yaw1):
    """Motion from pose 0 to pose 1, in the car's frame at pose 0: (dx, dy, dyaw)."""
    dx, dy = x1 - x0, y1 - y0
    c0, s0 = np.cos(yaw0), np.sin(yaw0)
    return c0 * dx + s0 * dy, -s0 * dx + c0 * dy, c.wrap(yaw1 - yaw0)


def relative_pose_errors(s, ref, est, ok, stretch_m: float, step_m: float = STEP_M) -> dict:
    """Translation drift (% of distance) and heading drift (deg/m) of ``est`` against
    ``ref`` over stretches of ``stretch_m`` driven metres.

    s: distance driven at each sample; ref, est: (x, y, yaw) arrays on the same samples;
    ok: samples allowed (a stretch with any sample not ok is skipped)."""
    s = np.asarray(s, dtype=float)
    ok = np.asarray(ok, dtype=bool) & np.isfinite(s)
    for a in (*ref, *est):
        ok &= np.isfinite(a)
    bad = np.concatenate([[0], np.cumsum(~ok)])
    trans, head, starts = [], [], []
    next_start = -math.inf
    for i in range(len(s)):
        if not ok[i] or s[i] < next_start:
            continue
        j = int(np.searchsorted(s, s[i] + stretch_m))
        if j >= len(s) or bad[j + 1] - bad[i] > 0:
            continue
        dr = body_delta(*(a[i] for a in ref), *(a[j] for a in ref))
        de = body_delta(*(a[i] for a in est), *(a[j] for a in est))
        driven = s[j] - s[i]
        trans.append(100.0 * math.hypot(de[0] - dr[0], de[1] - dr[1]) / driven)
        head.append(math.degrees(float(c.wrap(de[2] - dr[2]))) / driven)
        starts.append(i)
        next_start = s[i] + step_m
    trans, head = np.array(trans), np.array(head)
    return dict(stretch_m=stretch_m, n=int(len(trans)),
                translation_pct_median=float(np.median(trans)) if len(trans) else math.nan,
                translation_pct_p95=float(np.percentile(trans, 95)) if len(trans) else math.nan,
                heading_deg_per_m_median_abs=float(np.median(np.abs(head))) if len(head) else math.nan,
                heading_deg_per_m_mean=float(np.mean(head)) if len(head) else math.nan,
                _translation=trans, _heading=head, _starts=np.array(starts, dtype=int))


def dead_reckon(ref0, est, i0):
    """est's trajectory placed on the map at sample i0: T_ref(i0) T_est(i0)^-1 T_est(t)."""
    x, y, yaw = (np.asarray(a, dtype=float) for a in est)
    dx, dy, dyaw = body_delta(x[i0], y[i0], yaw[i0], x, y, yaw)
    rx, ry, ryaw = ref0
    return (rx + math.cos(ryaw) * dx - math.sin(ryaw) * dy,
            ry + math.sin(ryaw) * dx + math.cos(ryaw) * dy, ryaw + dyaw)


def locate_with_loc(run: c.Run, map_dir, out: Path):
    """loc.py's LiDAR match of the run on map_dir: rows of loc.COLUMNS, cached in ``out``.
    The run's own map-pose path seeds loc.py's global search (where to look first)."""
    from run_analysis import loc
    cache = out / "loc_track.npz"
    if cache.exists():
        return np.load(cache)["a"]
    _, _, x, y, _ = c.map_pose(run)
    seed = np.c_[x, y]
    rows, _ = loc.locate(run.path, loc.Grid.from_map(map_dir), seed)
    np.savez_compressed(cache, a=rows)
    return rows


def frame_check(ref_t, ref_xy, loc_rows, window_s: float = 20.0, ref_yaw=None) -> dict:
    """Are the map pose and loc.py in the same frame, and how far apart are they?

    If they were in different frames, the best rotation and shift (common.rigid_fit)
    would line loc.py's well-matched positions up with the map pose's and leave little
    behind; if what is left is about as large as the distance itself, there is no frame
    offset and the distance is the two estimates disagreeing (common.frame_verdict).
    Also: their heading difference, and their distance in each ``window_s`` of the run."""
    from run_analysis.loc import COL, GOOD_H10
    good = loc_rows[loc_rows[:, COL["h10"]] >= GOOD_H10]
    t = good[:, COL["t"]]
    keep = (t >= ref_t[0]) & (t <= ref_t[-1])
    t = t[keep]
    a = good[keep][:, [COL["x"], COL["y"]]]
    b = np.c_[np.interp(t, ref_t, ref_xy[:, 0]), np.interp(t, ref_t, ref_xy[:, 1])]
    d = np.hypot(*(a - b).T)
    windows = []
    for start in np.arange(0.0, (t[-1] - ref_t[0]) if len(t) else 0.0, window_s):
        m = (t - ref_t[0] >= start) & (t - ref_t[0] < start + window_s)
        if m.any():
            windows.append(dict(from_s=float(start), to_s=float(start + window_s), n=int(m.sum()),
                                distance_median_m=float(np.median(d[m]))))
    fit = c.rigid_fit(a, b)
    median = float(np.median(d)) if len(d) else math.nan
    heading = (float(np.median(np.degrees(c.wrap(good[keep][:, COL["yaw"]] - np.interp(t, ref_t, np.unwrap(ref_yaw))))))
               if ref_yaw is not None and len(t) else math.nan)
    return dict(rigid_fit=fit, verdict=c.frame_verdict(median, fit), heading_difference_median_deg=heading,
                distance_median_m=median,
                distance_p95_m=float(np.percentile(d, 95)) if len(d) else math.nan,
                windows=windows)


def analyse(run: c.Run, loc_rows=None) -> dict:
    topic, t, rx, ry, ryaw = c.map_pose(run)
    yaw_ref = np.unwrap(ryaw)
    odom = run["/odom"]
    v = np.interp(t, odom.t, odom.v)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(rx), np.diff(ry)))])
    gyro = run["/imu/data"] if run.has("/imu/data") else None
    jumps = c.pose_jumps(t, rx, ry, ryaw, odom.t, odom.v,
                         gyro.t if gyro is not None else None, gyro.wz if gyro is not None else None)
    ok = c.inside(t, c.lb_spans(run)) & (np.abs(v) >= c.MOVING_SPEED) & ~c.near(t, jumps)
    result = dict(reference=topic, samples=int(len(t)), driving_samples=int(ok.sum()),
                  distance_m=float(s[-1]), map_pose_jumps=[float(j - t[0]) for j in jumps], sources={})
    curves = {}
    for topic, label in SOURCES.items():
        if not run.has(topic):
            continue
        o = run[topic]
        est = (np.interp(t, o.t, o.x), np.interp(t, o.t, o.y), np.interp(t, o.t, np.unwrap(o.yaw)))
        rows = [relative_pose_errors(s, (rx, ry, yaw_ref), est, ok, d) for d in STRETCHES_M]
        result["sources"][topic] = dict(model=label, stretches=[{k: v for k, v in r.items() if not k.startswith("_")}
                                                                for r in rows])
        curves[topic] = (est, rows)
    if loc_rows is not None and len(loc_rows):
        from run_analysis.loc import COL
        result["loc_py"] = frame_check(t, np.c_[rx, ry], loc_rows, ref_yaw=ryaw)
        lt = loc_rows[:, COL["t"]]
        est = (np.interp(t, lt, loc_rows[:, COL["x"]]), np.interp(t, lt, loc_rows[:, COL["y"]]),
               np.interp(t, lt, np.unwrap(loc_rows[:, COL["yaw"]])))
        covered = (t >= lt[0]) & (t <= lt[-1])
        rows = [relative_pose_errors(s, (rx, ry, yaw_ref), est, ok & covered, d) for d in STRETCHES_M]
        result["loc_py"]["stretches"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        curves["loc.py"] = (est, rows)
    result["_curves"] = curves
    result["_ok"] = ok
    result["_ref"] = (t, rx, ry, ryaw)
    return result


def plots(run: c.Run, result: dict, folder: Path) -> list:
    plt = c.pyplot()
    if plt is None:
        return []
    t, rx, ry, ryaw = result["_ref"]
    ok = result["_ok"]
    paths = []
    i0 = int(np.argmax(ok)) if ok.any() else 0
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rx, ry, "k-", lw=1.5, label=f"map pose ({result['reference']}, base_link)")
    colours = {"/odom": "tab:red", "/odom/imu": "tab:blue", "loc.py": "tab:green"}
    for name, (est, _) in result["_curves"].items():
        if name == "loc.py":
            ax.plot(est[0], est[1], "-", color=colours[name], lw=0.8, alpha=0.8, label="loc.py (LiDAR on the map)")
            continue
        x, y, _ = dead_reckon((rx[i0], ry[i0], ryaw[i0]), est, i0)
        ax.plot(x, y, "-", color=colours[name], lw=1, label=f"{name} dead-reckoned ({SOURCES[name]})")
    for j in result["map_pose_jumps"]:
        k = min(int(np.searchsorted(t, t[0] + j)), len(t) - 1)
        ax.plot(rx[k], ry[k], "kx", ms=9)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, map)")
    ax.set_ylabel("y (m, map)")
    ax.set_title(f"{run.car} {run.run_id}: odometry placed on the map at the first LB hold\n"
                 f"(x: a map-pose jump, left out of the drift numbers)", fontsize=9)
    ax.legend(fontsize=7, loc="best")
    paths.append(c.save(fig, folder / "localization_paths.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, (_, rows) in result["_curves"].items():
        for row in rows:
            if row["n"]:
                axes[0].hist(row["_translation"], bins=30, histtype="step", label=f"{name}, {row['stretch_m']:g} m")
                axes[1].hist(row["_heading"], bins=30, histtype="step", label=f"{name}, {row['stretch_m']:g} m")
    axes[0].set_xlabel("translation drift (% of distance driven)")
    axes[1].set_xlabel("heading drift (deg per m driven)")
    for ax in axes:
        ax.set_ylabel("stretches")
        ax.legend(fontsize=7)
    fig.suptitle("Drift against the map pose, per stretch")
    paths.append(c.save(fig, folder / "localization_drift.png"))
    plt.close(fig)
    return paths


def report(run: c.Run, result: dict) -> list:
    lines = [f"localization  {run.car}/{run.run_id}  reference {result['reference']}, {result['distance_m']:.0f} m",
             f"  map-pose jumps left out: {len(result['map_pose_jumps'])}"]
    for topic, row in result["sources"].items():
        for st in row["stretches"]:
            lines.append(f"  {topic:10} over {st['stretch_m']:g} m: translation {c.fmt(st['translation_pct_median'])}% "
                         f"(p95 {c.fmt(st['translation_pct_p95'])}%), "
                         f"heading {c.fmt(st['heading_deg_per_m_median_abs'])}"
                         f" deg/m  [{st['n']} stretches]")
    if "loc_py" in result:
        lp = result["loc_py"]
        lines.append(f"  loc.py vs map pose: {lp['verdict']} "
                     f"(a rigid fit leaves {c.fmt(lp['rigid_fit']['residual_median_m'])}"
                     f" m of {c.fmt(lp['distance_median_m'])} m); distance p95 {c.fmt(lp['distance_p95_m'])} m, "
                     f"heading difference {c.fmt(lp['heading_difference_median_deg'])} deg (medians)")
        windows = [f"{w['from_s']:.0f}-{w['to_s']:.0f} s {c.fmt(w['distance_median_m'], 2)} m" for w in lp["windows"]]
        lines.append("    by time: " + ", ".join(windows))
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run directory, data/<car>/racerbot_runs/<time>")
    ap.add_argument("--map", help="map folder (map.yaml) the run's map pose is in: adds loc.py as a second estimate")
    ap.add_argument("--out", default=str(c.DEFAULT_OUT), help="results folder (default %(default)s)")
    args = ap.parse_args(argv)
    run = c.load(args.run, TOPICS)
    if c.map_pose(run) is None:
        print(f"{args.run}: no map pose ({', '.join(c.MAP_POSES)}), so nothing to compare odometry against",
              file=sys.stderr)
        return 1
    folder = c.out_dir(run, args.out)
    loc_rows = locate_with_loc(run, args.map, folder) if args.map else None
    result = analyse(run, loc_rows)
    paths = plots(run, result, folder)
    print("\n".join(report(run, result)))
    c.write_summary(folder, "localization", {k: v for k, v in result.items() if not k.startswith("_")})
    print(f"  results: {folder}" + (f" ({len(paths)} plots)" if paths else " (no matplotlib: numbers only)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
