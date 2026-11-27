#!/usr/bin/env python3
"""Algorithm accuracy, as two separate results that answer different questions.

    python3 tools/run_analysis/dataset_examples/algorithm_accuracy.py data/rb2/racerbot_runs/20261111-152132
    python3 tools/run_analysis/dataset_examples/algorithm_accuracy.py data/rb1/racerbot_runs/20261026-194514

1. Command tracking (every run with /drive): did the car do what the algorithm
   commanded? Speed: commanded (/drive, held between messages) against the
   wheel speed. Curvature: commanded kappa_cmd = tan(delta)/L against achieved
   kappa = r / v (gyro yaw rate over speed). Each after the delay that best
   lines them up. This says nothing about where the car went.

2. Path tracking (only when the run followed a recorded racing line): how far
   was the car from that line? Cross-track error e = signed distance from the
   car's rear axle (base_link) to the closest point of the closed line,
   + when the car is left of the line's direction. Computed here against every
   segment of the line, independently of the controller's own projection.

   The line is the one the run loaded (pure_pursuit_node's waypoints_file),
   checked against the node's own "ready: N waypoints from ..." log line; the
   map is the one map_server loaded. The car's position is the particle
   filter's (a LiDAR pose, moved to base_link with the node's own
   laser_offset_x/y), and loc.py's LiDAR match on the same map is a second,
   independent estimate. Neither is ground truth: e includes their errors.

   Only ticks where the node was tracking (its /drive_intent state is its
   steering law), LB was held, the car moved, and the particle filter had not
   just jumped are scored; time spent stopped, avoiding or released is
   reported, not scored.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_analysis.dataset_examples import common as c  # noqa: E402
from run_analysis.dataset_examples.steering_response import predicted_yaw_rate, shift  # noqa: E402
from run_analysis.speed_tracking import hold  # noqa: E402
from run_analysis.yaw_tracking import ls_gain, xcorr_lag  # noqa: E402
from pure_pursuit import racing_math  # noqa: E402  (pure Python, no ROS)

TOPICS = ["/drive", "/odom", "/imu/data", "/joy", "/drive_intent", *c.MAP_POSES]
NODE = "pure_pursuit_node"
SPEED_MAX_LAG_S = 1.0       # s, the speed loop is slower than steering
STEER_MAX_LAG_S = 0.5
MIN_SPEED = 0.5             # m/s, curvature r / v is noise below this
CORNER_CURVATURE = 0.3      # 1/m, |line curvature| from which a point counts as a corner
READY = re.compile(r"ready: (\d+) waypoints from '([^']+)'")
DRIVE_LOG = re.compile(r"^DRIVE .*?pose=\(([-\d.]+),([-\d.]+),([-+\d.]+)deg\).*?cross-track=([-\d.]+)m")


# --------------------------------------------------------------------------- command tracking

def command_tracking(run: c.Run) -> dict:
    """Speed and curvature commanded on /drive against what the car did, while the algorithm drove."""
    drive, odom, imu = run["/drive"], run["/odom"], run["/imu/data"]
    grid = c.time_grid(max(drive.t[0], odom.t[0], imu.t[0]), min(drive.t[-1], odom.t[-1], imu.t[-1]))
    driving = c.inside(grid, c.lb_spans(run)) & c.intent_mask(run, grid, lambda row: row["severity"] != "stop")
    v_cmd, steer = hold(drive.t, drive.speed, grid), hold(drive.t, drive.steer, grid)
    v = c.on_grid(grid, odom.t, odom.v)
    r = c.on_grid(grid, imu.t, imu.wz)
    span = np.flatnonzero(driving)
    out = dict(driving_s=float(driving.sum() / c.FS))
    if not len(span):
        return out
    a, b = span[0], span[-1] + 1
    lag_v, corr_v = xcorr_lag(np.nan_to_num(v_cmd[a:b]), np.nan_to_num(v[a:b]), c.FS, SPEED_MAX_LAG_S)
    k = int(round(lag_v * c.FS)) if math.isfinite(lag_v) else 0
    v_cmd_d = shift(v_cmd, k)
    ok = driving & (v_cmd_d > c.MOVING_SPEED) & np.isfinite(v)
    out["speed"] = dict(delay_s=lag_v, correlation=corr_v, error=c.stats((v - v_cmd_d)[ok]),
                        achieved_over_commanded=float(np.sum(v[ok]) / np.sum(v_cmd_d[ok])) if ok.any() else math.nan)
    lag_k, corr_k = xcorr_lag(np.nan_to_num(predicted_yaw_rate(v, steer))[a:b], np.nan_to_num(r[a:b]),
                              c.FS, STEER_MAX_LAG_S)
    j = int(round(lag_k * c.FS)) if math.isfinite(lag_k) else 0
    kappa_cmd = np.tan(shift(steer, j)) / c.WHEELBASE
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(np.abs(v) >= MIN_SPEED, r / v, np.nan)
    okk = driving & np.isfinite(kappa) & np.isfinite(kappa_cmd)
    out["curvature"] = dict(delay_s=lag_k, correlation=corr_k, error=c.stats((kappa - kappa_cmd)[okk]),
                            gain=ls_gain(kappa[okk], kappa_cmd[okk]))
    # r / v with the wheel speed is too small if the wheels read high; the map pose's speed removes that.
    topic, v_map, _ = c.map_speed(run, grid)
    if topic is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa_m = np.where(v_map >= c.MIN_MAP_SPEED, r / v_map, np.nan)
        okm = driving & np.isfinite(kappa_m) & np.isfinite(kappa_cmd)
        out["curvature"]["with_map_speed"] = dict(speed_from=topic, gain=ls_gain(kappa_m[okm], kappa_cmd[okm]),
                                                  error=c.stats((kappa_m - kappa_cmd)[okm]))
    out["_curves"] = dict(t=grid - grid[0], v_cmd=v_cmd_d, v=v, kappa_cmd=kappa_cmd, kappa=kappa, driving=driving)
    return out


# --------------------------------------------------------------------------- path tracking

def load_line(path) -> np.ndarray:
    """(N, 3) x, y, speed of a racing-line CSV with a header row (speed NaN if absent)."""
    rows = np.loadtxt(Path(path), delimiter=",", skiprows=1, ndmin=2)
    return rows[:, :3] if rows.shape[1] >= 3 else np.c_[rows[:, :2], np.full(len(rows), np.nan)]


def cross_track(px, py, line_xy, closed: bool = True):
    """(e, segment, t): signed distance (m) from each point to the closest point of the
    polyline (+ = left of the line's direction), the segment it lies on and how far
    along it (0..1). Every segment is considered, so a hairpin cannot pick the wrong side."""
    px, py = np.atleast_1d(np.asarray(px, dtype=float)), np.atleast_1d(np.asarray(py, dtype=float))
    a = np.asarray(line_xy, dtype=float)[:, :2]
    b = np.roll(a, -1, axis=0) if closed else a[1:]
    a = a if closed else a[:-1]
    d = b - a
    length2 = np.sum(d * d, axis=1)
    keep = length2 > 1e-12
    a, d, length2, index = a[keep], d[keep], length2[keep], np.flatnonzero(keep)
    t = np.clip(((px[:, None] - a[:, 0]) * d[:, 0] + (py[:, None] - a[:, 1]) * d[:, 1]) / length2, 0.0, 1.0)
    qx, qy = a[:, 0] + t * d[:, 0], a[:, 1] + t * d[:, 1]
    dist2 = (px[:, None] - qx) ** 2 + (py[:, None] - qy) ** 2
    best = np.argmin(dist2, axis=1)
    rows = np.arange(len(px))
    dx, dy = d[best, 0], d[best, 1]
    side = np.sign(dx * (py - qy[rows, best]) - dy * (px - qx[rows, best]))
    return side * np.sqrt(dist2[rows, best]), index[best], t[rows, best]


def node_log(run: c.Run):
    """(ready, drive): the node's 'ready: N waypoints from PATH' lines, and its DRIVE lines
    as rows (t, x, y, yaw_rad, cross_track_m)."""
    ready = [(int(m.group(1)), m.group(2)) for _, _, text in run.rosout for m in [READY.search(text)] if m]
    drive = [(t, float(m.group(1)), float(m.group(2)), math.radians(float(m.group(3))), float(m.group(4)))
             for t, _, text in run.rosout for m in [DRIVE_LOG.search(text)] if m]
    return ready, np.array(drive).reshape(-1, 5)


def checks(run: c.Run, params: dict, line_path: Path, line, map_dir) -> list:
    """Each check that the racing line, map and frames are the ones the run really used."""
    out = []
    recorded = params.get("waypoints_file", "")
    out.append(dict(check="line is the run's waypoints_file", ok=bool(recorded) and c.local_path(recorded) == line_path,
                    detail=f"run loaded {recorded or '(no waypoints_file parameter)'}; analysed {line_path}"))
    ready, _ = node_log(run)
    if ready:
        n, path = ready[-1]
        out.append(dict(check="waypoint count matches the node's log",
                        ok=n == len(line) and c.local_path(path) == line_path,
                        detail=f"node: {n} waypoints from {path}; file: {len(line)}"))
    else:
        out.append(dict(check="waypoint count matches the node's log", ok=False, detail="no 'ready:' line in /rosout"))
    out.append(dict(check="map is the line's folder",
                    ok=map_dir is not None and Path(map_dir).resolve() == line_path.parent.resolve(),
                    detail=f"map_server loaded {map_dir}"))
    topic = params.get("pose_topic", "")
    out.append(dict(check="pose topic and frame", ok=run.has(topic) and run.frames.get(topic, "").strip("/") == "map",
                    detail=f"{topic} in frame '{run.frames.get(topic, '?')}', pose_frame '{params.get('pose_frame')}', "
                           f"laser offset ({params.get('laser_offset_x')}, {params.get('laser_offset_y')}) m"))
    return out


def path_tracking(run: c.Run, line_path=None, loc_rows=None) -> dict | None:
    """Cross-track error against the run's racing line, or None if the run followed no line."""
    params = c.node_params(run.path, NODE)
    if line_path is None:
        if not params.get("waypoints_file"):
            return None
        line_path = c.local_path(params["waypoints_file"])
    line_path = Path(line_path).resolve()
    if not line_path.is_file():
        return dict(error=f"{line_path}: racing line not found")
    line = load_line(line_path)
    map_yaml = c.node_params(run.path, "map_server").get("yaml_filename")
    map_dir = c.local_path(map_yaml).parent if map_yaml else None
    result = dict(line=str(line_path.relative_to(c.ROOT)) if c.ROOT in line_path.parents else str(line_path),
                  steering_law=params.get("steering_law"), checks=checks(run, params, line_path, line, map_dir))
    topic = params.get("pose_topic", "/pf/viz/inferred_pose")
    if not run.has(topic):
        result["error"] = f"no {topic} in the bag"
        return result
    pose = run[topic]
    frame = params.get("pose_frame", "laser")
    x, y, yaw = c.to_base_link(pose.x, pose.y, pose.yaw, frame, float(params.get("laser_offset_x", c.LASER_X)),
                               float(params.get("laser_offset_y", 0.0)))
    t = pose.t
    odom = run["/odom"]
    gyro = run["/imu/data"] if run.has("/imu/data") else None
    jumps = c.pose_jumps(t, x, y, yaw, odom.t, odom.v, gyro.t if gyro is not None else None,
                         gyro.wz if gyro is not None else None)
    v = np.interp(t, odom.t, odom.v)
    law = params.get("steering_law")
    tracking = c.intent_mask(run, t, lambda row: row["state"] == law)
    held = c.inside(t, c.lb_spans(run))
    moving = np.abs(v) >= c.MOVING_SPEED
    jumped = c.near(t, jumps)
    ok = tracking & held & moving & ~jumped
    e, seg, along = cross_track(x, y, line)
    kappa_line = racing_math.estimate_signed_path_curvature(line[:, :2], closed=True)
    corner = np.abs(kappa_line[seg]) >= CORNER_CURVATURE
    dt = np.diff(t, append=t[-1])
    result.update(pose=dict(topic=topic, frame=frame, jumps=int(len(jumps))),
                  time_s=dict(scored=float(dt[ok].sum()), released=float(dt[moving & ~held].sum()),
                              not_tracking=float(dt[held & moving & ~tracking].sum()),
                              near_pose_jump=float(dt[held & moving & tracking & jumped].sum())),
                  states_not_tracking=sorted({row["state"] for _, row in run.intents if row["state"] != law}),
                  cross_track=c.stats(e[ok]), straights=c.stats(e[ok & ~corner]), corners=c.stats(e[ok & corner]))
    profile = line[seg, 2] + along * (line[(seg + 1) % len(line), 2] - line[seg, 2])
    result["speed_vs_profile"] = c.stats((v - profile)[ok])
    _, drive = node_log(run)
    if len(drive):
        ours_e = np.abs(np.interp(drive[:, 0], t, e))
        ours_x, ours_y = np.interp(drive[:, 0], t, x), np.interp(drive[:, 0], t, y)
        result["agrees_with_node_log"] = dict(
            lines=int(len(drive)),
            pose_distance_median_m=float(np.median(np.hypot(ours_x - drive[:, 1], ours_y - drive[:, 2]))),
            cross_track_difference_median_m=float(np.median(np.abs(ours_e - np.abs(drive[:, 4])))))
    if loc_rows is not None and len(loc_rows):
        from run_analysis.loc import COL, GOOD_H10
        good = loc_rows[(loc_rows[:, COL["h10"]] >= GOOD_H10)]
        lt = good[:, COL["t"]]
        inside_pf = (lt >= t[0]) & (lt <= t[-1])
        good, lt = good[inside_pf], lt[inside_pf]
        ok_l = (np.interp(lt, t, ok.astype(float)) > 0.999)
        e_loc, _, _ = cross_track(good[:, COL["x"]], good[:, COL["y"]], line)
        pf_x, pf_y = np.interp(lt, t, x), np.interp(lt, t, y)
        median = (float(np.median(np.hypot(good[:, COL["x"]] - pf_x, good[:, COL["y"]] - pf_y)[ok_l]))
                  if ok_l.any() else math.nan)
        fit = c.rigid_fit(good[ok_l][:, [COL["x"], COL["y"]]], np.c_[pf_x, pf_y][ok_l])
        result["loc_py"] = dict(scans=int(len(good)), cross_track=c.stats(e_loc[ok_l]),
                                distance_to_pf_median_m=median, rigid_fit=fit, verdict=c.frame_verdict(median, fit))
        result["_loc"] = (lt - t[0], e_loc, ok_l)
    result["_curves"] = dict(t=t - t[0], x=x, y=y, e=e, ok=ok, line=line, map_dir=map_dir,
                             node=drive if len(drive) else None, t0=t[0])
    return result


def locate_with_loc(run: c.Run, map_dir, line, out: Path):
    """loc.py on the run's own map, seeded by the racing line; cached in ``out``."""
    from run_analysis import loc
    cache = out / "loc_track.npz"
    if cache.exists():
        return np.load(cache)["a"]
    rows, _ = loc.locate(run.path, loc.Grid.from_map(map_dir), line[:, :2])
    np.savez_compressed(cache, a=rows)
    return rows


# --------------------------------------------------------------------------- output

def plots(run: c.Run, cmd: dict, path: dict | None, folder: Path) -> list:
    plt = c.pyplot()
    if plt is None:
        return []
    paths = []
    if "_curves" in cmd:
        k = cmd["_curves"]
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(k["t"], k["v_cmd"], "-", color="tab:orange", lw=1,
                     label=f"commanded (/drive), delayed {cmd['speed']['delay_s'] * 1000:.0f} ms")
        axes[0].plot(k["t"], k["v"], "k-", lw=0.8, label="wheel speed (/odom)")
        axes[1].plot(k["t"], k["kappa_cmd"], "-", color="tab:orange", lw=1, label="commanded tan(delta)/L, delayed")
        axes[1].plot(k["t"], k["kappa"], "k-", lw=0.6, label="achieved r / v")
        axes[1].set_ylim(-3, 3)
        for ax in axes:
            ax.fill_between(k["t"], 0, 1, where=k["driving"], transform=ax.get_xaxis_transform(), color="0.92")
            ax.legend(fontsize=7, loc="upper right")
        axes[0].set_ylabel("speed (m/s)")
        axes[1].set_ylabel("curvature (1/m)")
        axes[1].set_xlabel("time (s); shaded: the algorithm driving")
        axes[0].set_title(f"{run.car} {run.run_id}: command tracking")
        paths.append(c.save(fig, folder / "command_tracking.png"))
        plt.close(fig)
    if path and "_curves" in path:
        k = path["_curves"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
        if k["map_dir"] is not None and (Path(k["map_dir"]) / "map.yaml").is_file():
            from run_analysis.loc import Grid
            g = Grid.from_map(k["map_dir"])
            axes[0].imshow(g.occ, origin="lower", cmap="Greys", alpha=0.6,
                           extent=[g.ox, g.ox + g.occ.shape[1] * g.res, g.oy, g.oy + g.occ.shape[0] * g.res])
        closed = np.vstack([k["line"][:, :2], k["line"][:1, :2]])
        axes[0].plot(closed[:, 0], closed[:, 1], "-", color="tab:green", lw=1.5, label="racing line")
        axes[0].plot(k["x"][~k["ok"]], k["y"][~k["ok"]], ".", ms=1, color="0.75", label="not scored")
        sc = axes[0].scatter(k["x"][k["ok"]], k["y"][k["ok"]], c=np.abs(k["e"][k["ok"]]), s=2, cmap="viridis",
                             vmin=0, vmax=max(0.3, float(np.nanpercentile(np.abs(k["e"][k["ok"]]), 95)))
                             if k["ok"].any() else 1)
        fig.colorbar(sc, ax=axes[0], label="|cross-track error| (m)")
        pad = 1.0
        axes[0].set_xlim(np.nanmin(k["line"][:, 0]) - pad, np.nanmax(k["line"][:, 0]) + pad)
        axes[0].set_ylim(np.nanmin(k["line"][:, 1]) - pad, np.nanmax(k["line"][:, 1]) + pad)
        axes[0].set_aspect("equal")
        axes[0].legend(fontsize=7, loc="best")
        axes[0].set_title(f"{run.car} {run.run_id}: path tracking ({path['steering_law']})", fontsize=9)
        e = np.where(k["ok"], k["e"], np.nan)
        axes[1].plot(k["t"], e, "-", color="tab:blue", lw=0.8, label="particle filter (scored)")
        if "_loc" in path:
            lt, el, okl = path["_loc"]
            axes[1].plot(lt[okl], el[okl], ".", ms=2, color="tab:green", label="loc.py (scored)")
        if k["node"] is not None:
            axes[1].plot(k["node"][:, 0] - k["t0"], k["node"][:, 4], "kx", ms=4, label="|e| the node logged")
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set_xlabel("time (s)")
        axes[1].set_ylabel("cross-track error (m), + left of the line")
        axes[1].legend(fontsize=7)
        paths.append(c.save(fig, folder / "path_tracking.png"))
        plt.close(fig)
    return paths


def report(run: c.Run, cmd: dict, path: dict | None) -> list:
    lines = [f"algorithm_accuracy  {run.car}/{run.run_id}"]
    lines.append(f"  COMMAND TRACKING ({cmd['driving_s']:.0f} s with the algorithm driving)")
    if "speed" in cmd:
        s, k = cmd["speed"], cmd["curvature"]
        lines.append(f"    speed: delay {c.fmt(s['delay_s'] * 1000)} ms, "
                     f"achieved - commanded mean {c.fmt(s['error']['mean'])} m/s, "
                     f"RMS {c.fmt(s['error']['rms'])} m/s, achieved/commanded {c.fmt(s['achieved_over_commanded'])}")
        lines.append(f"    curvature: delay {c.fmt(k['delay_s'] * 1000)} ms, gain {c.fmt(k['gain'])}, "
                     f"RMS error {c.fmt(k['error']['rms'])} 1/m")
        if "with_map_speed" in k:
            w = k["with_map_speed"]
            lines.append(f"    curvature with {w['speed_from']}'s speed: gain {c.fmt(w['gain'])}, "
                         f"RMS error {c.fmt(w['error']['rms'])} 1/m")
    if path is None:
        lines.append("  PATH TRACKING: none. The run followed no recorded racing line, so these are command-tracking "
                     "results only, not path-tracking accuracy.")
        return lines
    if "error" in path and "cross_track" not in path:
        lines.append(f"  PATH TRACKING: not scored: {path['error']}")
        return lines
    lines.append(f"  PATH TRACKING against {path['line']} ({path['steering_law']} steering law)")
    for ch in path["checks"]:
        lines.append(f"    [{'ok' if ch['ok'] else 'FAIL'}] {ch['check']}: {ch['detail']}")
    ts = path["time_s"]
    lines.append(f"    scored {ts['scored']:.0f} s; left out: {ts['not_tracking']:.0f} s not tracking "
                 f"({', '.join(path['states_not_tracking'])}), {ts['released']:.0f} s LB released while moving, "
                 f"{ts['near_pose_jump']:.0f} s near {path['pose']['jumps']} particle-filter jumps")
    for name in ("cross_track", "straights", "corners"):
        e = path[name]
        lines.append(f"    {name:11}: mean {c.fmt(e['mean'])} m, RMS {c.fmt(e['rms'])} m, "
                     f"p95 |e| {c.fmt(e['p95_abs'])} m, "
                     f"max {c.fmt(e['max_abs'])} m")
    if "agrees_with_node_log" in path:
        a = path["agrees_with_node_log"]
        lines.append(f"    vs the node's own DRIVE log ({a['lines']} lines): "
                     f"pose {c.fmt(a['pose_distance_median_m'])} m apart, "
                     f"|e| differs {c.fmt(a['cross_track_difference_median_m'])} m (medians)")
    if "loc_py" in path:
        lp = path["loc_py"]
        lines.append(f"    loc.py (second estimate): RMS {c.fmt(lp['cross_track']['rms'])} m, p95 |e| "
                     f"{c.fmt(lp['cross_track']['p95_abs'])} m; {c.fmt(lp['distance_to_pf_median_m'])} m from the "
                     f"particle filter (median); {lp['verdict']} (a rigid fit leaves "
                     f"{c.fmt(lp['rigid_fit']['residual_median_m'])} m)")
    s = path["speed_vs_profile"]
    lines.append(f"    speed - line's profile speed: mean {c.fmt(s['mean'])} m/s, RMS {c.fmt(s['rms'])} m/s")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run directory, data/<car>/racerbot_runs/<time>")
    ap.add_argument("--raceline", help="racing line CSV (default: the one the run loaded, if any)")
    ap.add_argument("--no-loc", action="store_true", help="skip loc.py's second position estimate")
    ap.add_argument("--out", default=str(c.DEFAULT_OUT), help="results folder (default %(default)s)")
    args = ap.parse_args(argv)
    run = c.load(args.run, TOPICS, rosout_from=(NODE,))
    missing = [t for t in ("/drive", "/odom", "/imu/data") if not run.has(t)]
    if missing:
        print(f"{args.run}: needs {', '.join(missing)}", file=sys.stderr)
        return 1
    folder = c.out_dir(run, args.out)
    cmd = command_tracking(run)
    path = path_tracking(run, args.raceline)
    if path is not None and "_curves" in path and not args.no_loc and path["_curves"]["map_dir"] is not None:
        rows = locate_with_loc(run, path["_curves"]["map_dir"], path["_curves"]["line"], folder)
        path = path_tracking(run, args.raceline, rows)
    paths = plots(run, cmd, path, folder)
    print("\n".join(report(run, cmd, path)))
    summary = {"command_tracking": {k: v for k, v in cmd.items() if not k.startswith("_")}}
    if path is not None:
        summary["path_tracking"] = {k: v for k, v in path.items() if not k.startswith("_")}
    c.write_summary(folder, "algorithm_accuracy", summary)
    print(f"  results: {folder}" + (f" ({len(paths)} plots)" if paths else " (no matplotlib: numbers only)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
