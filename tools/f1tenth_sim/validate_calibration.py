#!/usr/bin/env python3
"""Does the simulator behave like the real car? Score both the same way, side by side.

    R=data/rb2/racerbot_runs/20261030; M=data/rb2/racerbot_auto/20261030-185036-fixed
    python3 tools/f1tenth_sim/validate_calibration.py real $R-184826 --line $M/raceline_raw.csv \\
        --out /tmp/real.json
    python3 tools/f1tenth_sim/validate_calibration.py sim --params-from $R-184826 \\
        --line $M/raceline_raw.csv --floor sfu_course_2026_10_30 --out /tmp/sim.json
    python3 tools/f1tenth_sim/validate_calibration.py table real=/tmp/real.json after=/tmp/sim.json

The sim drives the settings the real run was launched with, read from the run's own
metadata, for several flying laps. Neither side's verdict comes from the thing being
judged: laps, speeds and sections are measured from positions, the same way for both.

    python3 tools/f1tenth_sim/validate_calibration.py fit-plant $R-185036 $R-184556 \
        --floor sfu_course_2026_10_30 --grid '{"steering_delay_ticks": [1, 2, 3]}' --out /tmp/fit.json

fit-plant replays the real commands through the plant on an empty floor and ranks each
setting by yaw rate error against the gyro: no controller in the loop, so a setting
cannot win by changing what the controller asks for.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools" / "f1tenth_sim"), str(ROOT / "src" / "gap_follow")]

WHEELBASE = 0.36
BEND = (6.0, 8.0, -10.8, -9.3)      # x0, x1, y0, y1 m: the bottom-right bend of the 2026-10-30 course
SECTOR = 2.5                        # m of course
INTENT_HZ = 16.5                    # Hz of /drive_intent on run 20261030-185036
FULL_LOCK = 0.46
STEER_LAG = 0.14
GRIP_BINS = ((0.25, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4), (4, 5),
             (5, 6), (6, 8), (8, 12))
# What the code of 2026-10-30's runs did, on today's parameters: see docs/sim-calibration.md.
OLD_CODE = dict(aim_depth_bias=0.0, aim_depth_bias_fallback=0.0, swept_clearance_max_length=3.0,
                opponent_follow_floor=0.0)


# --- the scoring, on plain arrays ------------------------------------------------------

def in_box(x, y, box=BEND) -> np.ndarray:
    return (np.asarray(x) >= box[0]) & (np.asarray(x) < box[1]) & (np.asarray(y) >= box[2]) & (np.asarray(y) < box[3])


def flicker(x, y, fallback, box=BEND) -> dict:
    """How the mode behaves inside ``box``: share of samples in fallback, changes per pass."""
    inside = in_box(x, y, box)
    fallback = np.asarray(fallback, bool)
    if not inside.any():
        return dict(passes=0, share=math.nan, changes_per_pass=math.nan)
    edges = np.flatnonzero(np.diff(np.r_[0, inside.astype(int), 0]))
    changes = 0
    for start, end in zip(edges[::2], edges[1::2]):
        changes += int(np.sum(fallback[start + 1:end] != fallback[start:end - 1]))
    return dict(passes=len(edges) // 2, share=float(fallback[inside].mean()),
                changes_per_pass=changes / max(len(edges) // 2, 1))


def grip_rows(asked, achieved, bins=GRIP_BINS, min_samples: int = 10) -> list:
    """(low, high, n, median achieved) per bin of asked, both given as magnitudes."""
    asked, achieved = np.abs(asked), np.abs(achieved)
    rows = []
    for low, high in bins:
        k = (asked >= low) & (asked < high)
        if k.sum() >= min_samples:
            rows.append((low, high, int(k.sum()), float(np.median(achieved[k]))))
    return rows


def grip_error(reference, measured, limit: float = 12.0) -> float:
    """RMS m/s^2 between two grip tables over the bins they share."""
    a = {(r[0], r[1]): r[3] for r in reference if r[1] <= limit}
    b = {(r[0], r[1]): r[3] for r in measured}
    shared = sorted(set(a) & set(b))
    if not shared:
        return math.nan
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in shared) / len(shared))


def sector_error(reference, measured) -> dict:
    """RMS and worst m/s between two sector speed lists, where both have a value."""
    a, b = np.array(reference, float), np.array(measured, float)
    k = np.isfinite(a) & np.isfinite(b)
    if not k.any():
        return dict(rms=math.nan, worst=math.nan, bias=math.nan, sectors=0)
    d = b[k] - a[k]
    return dict(rms=float(np.sqrt(np.mean(d ** 2))), worst=float(np.max(np.abs(d))),
                bias=float(np.mean(d)), sectors=int(k.sum()))


def summarise_laps(t, progress, length: float, skip_laps: int = 1) -> dict:
    from run_analysis.laps import lap_times
    laps = lap_times(t, progress, length)
    flying = laps[skip_laps:] if len(laps) > skip_laps else []
    return dict(laps=[round(x, 2) for x in laps], flying=[round(x, 2) for x in flying])


# --- the real run ----------------------------------------------------------------------

def real_card(run, line, loc_dir) -> dict:
    """The driving behaviour of a located run."""
    from run_analysis.bagread import messages
    from run_analysis.car_measure import edge_statistics, read_series
    from run_analysis.laps import Course, holds
    from run_analysis.loc import COL, load_line, load_track
    from run_analysis.oa_replay import bag_frames, replay
    from run_analysis.sectors import true_speed, sector_medians
    from gap_follow import opponent_tracking as ot
    import yaml
    track = load_track(run, loc_dir)
    course = Course(load_line(line))
    t, x, y = track[:, COL["t"]], track[:, COL["x"]], track[:, COL["y"]]
    a, b = max(holds(track[:, COL["lb"]] > 0), key=lambda h: h[1] - h[0])
    arc, _ = course.project(x[a:b], y[a:b])
    s = course.unwrap(arc)
    v = true_speed(t[a:b], x[a:b], y[a:b])
    card = dict(run=Path(run).name, hold_s=float(t[b - 1] - t[a]), **summarise_laps(t[a:b], s, course.length))
    card["sectors"] = [float(q) for q in sector_medians(arc, v, course.length, SECTOR, mask=s > course.length)]
    held = np.array(read_series(run)["drive"])
    steer = np.interp(t[a:b], held[:, 0], np.abs(held[:, 1]))
    speed = np.interp(t[a:b], held[:, 0], held[:, 2])
    card["sector_steer"] = [float(q) for q in sector_medians(arc, steer, course.length, SECTOR, mask=s > course.length)]
    card["sector_cmd"] = [float(q) for q in sector_medians(arc, speed, course.length, SECTOR, mask=s > course.length)]
    # The mode, as the node reported it.
    times, states = [], []
    for _topic, stamp, m in messages(run, ["/drive_intent"]):
        d = json.loads(m.data)
        if d.get("state") in ("gap_follow", "corner_fallback"):
            times.append(stamp)
            states.append(d["state"] == "corner_fallback")
    times = np.array(times)
    card["flicker"] = flicker(np.interp(times, t, x), np.interp(times, t, y), states)
    # Scans.
    values = yaml.safe_load((ROOT / "src/gap_follow/config/gap_follow.yaml").read_text())["gap_follow_node"]["ros__parameters"]
    params = ot.OpponentParams.from_mapping(values)
    r = replay(bag_frames(run), params, values["wheelbase"], values["laser_offset_x"], values["laser_offset_y"], values["car_length"])
    card["false_movers"] = r.movers
    card["tracked_s"] = r.seconds
    card["false_movers_per_100s"] = 100.0 * r.movers / max(r.seconds, 1e-3)
    scans = []
    for i, (_topic, _stamp, m) in enumerate(messages(run, ["/scan"])):
        if i % 20 == 0:
            scans.append(np.array(m.ranges, float))
    card["scan_edges"] = edge_statistics(np.array(scans))
    return card


# --- the simulated run -----------------------------------------------------------------

def params_from_run(run, old_code: bool = True) -> dict:
    """The run's launched parameters that the mirror knows, with today's code paths set as that run's.

    ``old_code`` (the default) overlays OLD_CODE: what the 2026-10-30 code did, for runs of that code.
    A run made on later code was launched with those parameters itself (e.g. car 2's 20261106-171030 with
    aim_depth_bias 1.0); pass ``old_code=False`` (``--launched-code``) to simulate exactly what it ran.
    """
    import yaml
    import gap_mirror
    raw = yaml.safe_load((Path(run) / "metadata" / "params" / "gap_follow_node.yaml").read_text())
    values = raw[next(iter(raw))]["ros__parameters"] if "ros__parameters" not in raw else raw["ros__parameters"]
    known = gap_mirror.node_params()
    params = {k: v for k, v in values.items() if k in known}
    if old_code:
        params.update(OLD_CODE)
    return params


def sim_drive(job) -> dict:
    """Several laps of one seed, with the same measures as :func:`real_card`."""
    from run_analysis.car_measure import best_lag, edge_statistics
    from run_analysis.laps import Course
    from run_analysis.loc import load_line
    from run_analysis.oa_replay import replay
    from run_analysis.sectors import sector_medians
    from gap_follow import gap_logic, opponent_tracking as ot
    import eval_gap
    import yaml
    track, params, seed, seconds, calibration, line_path, grip_only, *rest = job
    start = rest[0] if rest else None
    plant = eval_gap.make_plant(track, seed, calibration)
    centre = plant.env.unwrapped.track.raceline
    pose = start if start is not None else (centre.xs[0], centre.ys[0], centre.yaws[0])
    obs, _ = plant.reset(np.array([pose], dtype=float))

    class _Lidar:
        angle_min = float(-eval_gap.components_half_fov())
        angle_increment = float(eval_gap.components_increment())
        range_min = float(eval_gap.components_range_min())
    ctrl = eval_gap.GapController(params, _Lidar)
    dt = eval_gap.CONTROL_DT
    rows, scans, frames = [], [], []
    contact = False
    settled = 0
    angles = None
    for step in range(int(seconds / dt)):
        ego = obs["agent_0"]
        wheel = float(np.asarray(ego["std_state"], float)[3])
        scan = np.asarray(ego["scan"], float)
        steering, speed = ctrl.command(scan, wheel, step, horizontal_accel=0.0, lb_pressed=True,
                                       yaw_rate=float(ego["std_state"][5]), fused_speed=float(plant.truth(0)[3]))
        truth = plant.truth(0)
        rows.append((step * dt, truth[0], truth[1], truth[3], truth[5], wheel, steering, speed,
                     bool(ctrl.telemetry.get("fallback", False)), truth[2]))
        if not grip_only:
            if angles is None:
                angles = _Lidar.angle_min + np.arange(scan.size) * _Lidar.angle_increment
            if step % 20 == 0:
                scans.append(scan.copy())
            clean, valid = gap_logic.sanitize_ranges(scan, 10.0, _Lidar.range_min)
            frames.append((clean, valid, angles, step * dt, wheel, steering))
        settled = settled + 1 if abs(truth[3]) < 0.05 else 0
        obs, _r, _done, _t, _info = plant.step([(steering, speed)])
        if plant.chassis_contact_steps > 0:
            contact = True
            break
        if settled > 200:          # 5 s standing: wedged, as a person would call it
            break
    wedged = settled > 200
    plant.close()
    a = np.array(rows, float)
    t, x, y, v, gz, w, cmd, cmd_speed, fb, angle = a.T
    late = np.interp(t - STEER_LAG, t, np.clip(cmd, -FULL_LOCK, FULL_LOCK))
    moving = v > 1.0
    out = dict(seed=seed, grip=grip_rows((v ** 2 * np.tan(late) / WHEELBASE)[moving], (v * gz)[moving]),
               ask=(v ** 2 * np.tan(late) / WHEELBASE)[moving].tolist() if grip_only else None,
               got=(v * gz)[moving].tolist() if grip_only else None,
               rack=(late[moving]).tolist() if grip_only else None,
               v=(v[moving]).tolist() if grip_only else None,
               contact=contact, wedged=wedged, end=[float(x[-1]), float(y[-1])], seconds=float(t[-1]))
    if grip_only:
        return out
    course = Course(load_line(line_path))
    arc, _ = course.project(x, y)
    s = course.unwrap(arc)
    out.update(summarise_laps(t, s, course.length))
    out["sectors"] = [float(q) for q in sector_medians(arc, v, course.length, SECTOR, mask=s > course.length)]
    out["sector_steer"] = [float(q) for q in sector_medians(arc, np.abs(cmd), course.length, SECTOR, mask=s > course.length)]
    out["sector_cmd"] = [float(q) for q in sector_medians(arc, cmd_speed, course.length, SECTOR, mask=s > course.length)]
    # /drive_intent reports the mode at about 16.5 Hz; count changes at that rate on both sides.
    every = np.round(np.arange(0.0, len(t) - 1, 40.0 / INTENT_HZ)).astype(int)
    out["flicker"] = flicker(x[every], y[every], (fb > 0.5)[every])
    values = yaml.safe_load((ROOT / "src/gap_follow/config/gap_follow.yaml").read_text())["gap_follow_node"]["ros__parameters"]
    r = replay(frames, ot.OpponentParams.from_mapping(values), values["wheelbase"], values["laser_offset_x"],
               values["laser_offset_y"], values["car_length"])
    out["false_movers"] = r.movers
    out["tracked_s"] = r.seconds
    out["false_movers_per_100s"] = 100.0 * r.movers / max(r.seconds, 1e-3)
    out["scan_edges"] = edge_statistics(np.array(scans))
    # /odom's yaw, computed as vesc_to_odom computes it, against true yaw rate.
    k = v > 1.0
    odom = w * np.tan(cmd) / WHEELBASE
    lead = best_lag(t[k], gz[k], t, odom, np.arange(-0.1, 0.4, 0.0125))[0]
    late_odom = np.interp(t[k] - lead, t, odom)
    out["odom_yaw_leads_gyro_s"] = lead
    out["odom_yaw_over_gyro"] = float(np.sum(late_odom * gz[k]) / np.sum(gz[k] ** 2))
    out["wheel_over_true"] = float(np.sum(w[k] * v[k]) / np.sum(v[k] ** 2))
    return out


def open_loop(calibration) -> dict:
    """Stop and top speed on an empty floor, where nothing but the plant is at work."""
    from run_analysis.fit_sim import simulate_commands
    out = {}
    fast = replace(calibration, speed_max=10.0)
    # The real stop began at 5.41 on the wheels: 5.41 / 1.033 true, the wheel over-read measured.
    target = 5.41 / 1.033
    ramp = np.column_stack([np.zeros(400), np.full(400, target / calibration.speed_command_scale)])
    stop = np.zeros((160, 2))
    speed, _ = simulate_commands(np.vstack([ramp, stop]), fast)
    v0 = float(speed[399])
    end = 400 + int(np.argmax(speed[400:] * calibration.odom_speed_scale <= 0.29))
    out["stop_start_true"] = v0
    out["stop_m"] = float(np.sum(speed[400:end]) * 0.025)
    out["stop_s"] = float((end - 400) * 0.025)
    top, _ = simulate_commands(np.column_stack([np.zeros(320), np.full(320, 6.0)]), calibration)
    out["top_true"] = float(top[-1])
    out["top_wheel"] = float(top[-1]) * calibration.odom_speed_scale
    return out


def plant_job(job) -> dict:
    """Real commands through the plant on an empty floor, scored against the gyro."""
    from run_analysis.car_measure import best_lag
    from run_analysis.fit_sim import simulate_commands
    calibration, setting, commands, t, gyro, true_v = job
    cal = replace(calibration, **setting)
    speed, yaw = simulate_commands(commands, cal)
    k = (true_v > 1.0) & (speed > 1.0)
    return dict(setting=setting, yaw_rms=float(np.sqrt(np.mean((yaw[k] - gyro[k]) ** 2))),
                lag=best_lag(t[k], yaw[k], t, gyro, np.arange(-0.15, 0.15, 0.0125))[0],
                speed_rms=float(np.sqrt(np.mean((speed[k] - true_v[k]) ** 2))))


def fit_plant(calibration, runs, loc_dir, grid: dict, workers: int, seconds: float = 60.0) -> list:
    """Every setting of ``grid``, best first, by yaw rate error against the gyro."""
    import itertools
    from run_analysis.car_measure import _gyro_bias, measure_run
    measured = [measure_run(run, loc_dir) for run in runs]
    bias = _gyro_bias(measured)
    jobs = []
    names = sorted(grid)
    for values in itertools.product(*(grid[n] for n in names)):
        setting = dict(zip(names, values))
        for r in measured:
            start = int(np.argmax(r["good"]))
            end = start + int(seconds / 0.025)
            jobs.append((calibration, setting, np.column_stack([r["steer"][start:end], r["speed_cmd"][start:end]]),
                         r["t"][start:end], r["gyro"][start:end] - bias, r["v"][start:end]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(plant_job, jobs))
    rows = []
    for i in range(0, len(results), len(measured)):
        chunk = results[i:i + len(measured)]
        rows.append(dict(setting=chunk[0]["setting"], yaw_rms=float(np.mean([c["yaw_rms"] for c in chunk])),
                         lag=float(np.mean([c["lag"] for c in chunk])),
                         speed_rms=float(np.mean([c["speed_rms"] for c in chunk]))))
    return sorted(rows, key=lambda row: row["yaw_rms"])


# --- the table -------------------------------------------------------------------------

def _mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
    return float(np.mean(values)) if values else math.nan


def combine(results: list) -> dict:
    """Seeds of a sim card as one."""
    flying = [lap for r in results for lap in r.get("flying", [])]
    sectors = np.nanmedian(np.array([r["sectors"] for r in results], float), axis=0)
    fl = [r["flicker"] for r in results if r["flicker"]["passes"]]
    extra = {key: np.nanmedian(np.array([r[key] for r in results], float), axis=0).tolist()
             for key in ("sector_steer", "sector_cmd")}
    return dict(flying=flying, laps=flying, sectors=sectors.tolist(), **extra,
                contact=sum(bool(r["contact"]) for r in results), wedged=sum(bool(r["wedged"]) for r in results),
                seeds=len(results), ends=[r["end"] for r in results if r["contact"] or r["wedged"]],
                flicker=dict(passes=sum(f["passes"] for f in fl), share=_mean(fl, "share"),
                             changes_per_pass=_mean(fl, "changes_per_pass")),
                false_movers=sum(r["false_movers"] for r in results), tracked_s=sum(r["tracked_s"] for r in results),
                false_movers_per_100s=100.0 * sum(r["false_movers"] for r in results)
                / max(sum(r["tracked_s"] for r in results), 1e-3),
                scan_edges=dict(jumps=_mean([r["scan_edges"] for r in results], "jumps"),
                                isolated=_mean([r["scan_edges"] for r in results], "isolated")),
                grip=results[0]["grip"], odom_yaw_leads_gyro_s=_mean(results, "odom_yaw_leads_gyro_s"),
                odom_yaw_over_gyro=_mean(results, "odom_yaw_over_gyro"), wheel_over_true=_mean(results, "wheel_over_true"))


def table(cards: dict) -> str:
    names = list(cards)
    real = cards[names[0]]
    lines = []

    def row(label, fn):
        cells = []
        for n in names:
            try:
                cells.append(fn(cards[n]))
            except (KeyError, TypeError, ValueError, IndexError):
                cells.append("-")
        lines.append(f"{label:30s} " + " ".join(f"{c:>20s}" for c in cells))
    lines.append(f"{'':30s} " + " ".join(f"{n:>20s}" for n in names))
    row("flying lap median, s", lambda c: f"{np.median(c['flying']):.2f} ({len(c['flying'])})")
    row("flying lap range, s", lambda c: f"{min(c['flying']):.2f}-{max(c['flying']):.2f}")
    row("seeds with contact", lambda c: f"{c['contact']}/{c['seeds']}")
    row("seeds wedged", lambda c: f"{c['wedged']}/{c['seeds']}")
    row("bend: fallback share", lambda c: f"{c['flicker']['share']:.2f}")
    row("bend: mode changes per pass", lambda c: f"{c['flicker']['changes_per_pass']:.1f}")
    row("false movers per 100 s", lambda c: f"{c['false_movers_per_100s']:.1f}")
    row("scan: range jumps per scan", lambda c: f"{c['scan_edges']['jumps']:.1f}")
    row("scan: lone returns per scan", lambda c: f"{c['scan_edges']['isolated']:.1f}")
    row("wheel over true", lambda c: f"{c['wheel_over_true']:.3f}")
    row("/odom yaw over gyro", lambda c: f"{c['odom_yaw_over_gyro']:.3f}")
    row("/odom yaw leads gyro, s", lambda c: f"{c['odom_yaw_leads_gyro_s']:.3f}")
    row("section speed: rms vs real", lambda c: f"{sector_error(real['sectors'], c['sectors'])['rms']:.2f}")
    row("section speed: worst", lambda c: f"{sector_error(real['sectors'], c['sectors'])['worst']:.2f}")
    row("section speed: bias", lambda c: f"{sector_error(real['sectors'], c['sectors'])['bias']:+.2f}")
    row("grip curve: rms vs real", lambda c: f"{grip_error(real['grip'], c['grip']):.2f}")
    for key, label in (("stop_m", "stop from 5.41, m"), ("stop_s", "stop from 5.41, s"),
                       ("top_true", "top speed true, m/s"), ("top_wheel", "top speed on wheels, m/s")):
        row(label, lambda c, k=key: f"{c[k]:.2f}")
    return "\n".join(lines)


def detail(cards: dict) -> str:
    """Section by section speed, commanded speed and |steering|, and each card's grip bins."""
    names = list(cards)
    lines = ["section m  " + "  ".join(f"{n:>22s}" for n in names) + "   (true speed / command / |steer|)"]
    count = max(len(c.get("sectors", [])) for c in cards.values())
    for i in range(count):
        cells = []
        for n in names:
            c = cards[n]
            values = [c.get(k, [])[i] if i < len(c.get(k, [])) else math.nan for k in ("sectors", "sector_cmd", "sector_steer")]
            cells.append("{:5.2f} {:5.2f} {:5.3f}".format(*values).rjust(22))
        lines.append(f"{i * SECTOR:9.1f}  " + "  ".join(cells))
    for n in names:
        grip = cards[n].get("grip") or []
        lines.append(f"grip {n}: " + " ".join(f"{g[0]}:{g[3]:.2f}" for g in grip))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("real", help="score located runs")
    r.add_argument("runs", nargs="+")
    r.add_argument("--line", required=True, help="the course line, x,y csv")
    r.add_argument("--loc", default=None, help="where loc.py wrote")
    r.add_argument("--measured", help="car_measure.py --json of these runs, for the grip curve and odometry")
    r.add_argument("--stop-m", type=float, help="a measured stop distance to show, m")
    r.add_argument("--stop-s", type=float, help="a measured stop time to show, s")
    r.add_argument("--out", required=True)
    s = sub.add_parser("sim", help="drive the simulator")
    s.add_argument("--params-from", required=True, help="a run whose launched settings to drive")
    s.add_argument("--line", required=True)
    s.add_argument("--track", default=None, help="default: the floor's own track, else course_2026_10_30")
    s.add_argument("--floor", default=None)
    s.add_argument("--friction", type=float, default=None)
    s.add_argument("--curve", help="json of a grip curve, from fit")
    s.add_argument("--seeds", type=int, default=6)
    s.add_argument("--first-seed", type=int, default=12345)
    s.add_argument("--seconds", type=float, default=115.0)
    s.add_argument("--workers", type=int, default=5)
    s.add_argument("--set", action="append", default=[], help="a parameter, name=value")
    s.add_argument("--start", default=None,
                   help="x,y,yaw_deg to start from, map frame; default the track centreline's first point. "
                        "Start where the real runs started: a standing start elsewhere is a different corner "
                        "approach (2026-11-06: the centreline start is 3 m before the hairpin, every real run "
                        "started past it)")
    s.add_argument("--launched-code", action="store_true",
                   help="drive exactly the run's launched parameters, without OLD_CODE (the 2026-10-30 code "
                        "paths); for runs made on code after 2026-11-01")
    s.add_argument("--cal", action="append", default=[], help="a calibration field, name=value, for experiments")
    s.add_argument("--out", required=True)
    g = sub.add_parser("fit-plant", help="fit plant settings: real commands against the gyro")
    g.add_argument("runs", nargs="+")
    g.add_argument("--floor", required=True)
    g.add_argument("--curve")
    g.add_argument("--loc", default=None)
    g.add_argument("--grid", required=True, help='json, {"field": [values], ...}')
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--out", required=True)
    t = sub.add_parser("table", help="print cards side by side, the first being the reference")
    t.add_argument("cards", nargs="+", help="name=path")
    t.add_argument("--detail", action="store_true", help="also every section's speed and the grip bins")
    args = ap.parse_args(argv)

    if args.mode == "real":
        from run_analysis.loc import DEFAULT_OUT
        cards = [real_card(run, args.line, args.loc or DEFAULT_OUT) for run in args.runs]
        out = dict(flying=[x for c in cards for x in c["flying"]], laps=[x for c in cards for x in c["laps"]],
                   sectors=np.nanmedian(np.array([c["sectors"] for c in cards], float), axis=0).tolist(),
                   sector_steer=np.nanmedian(np.array([c["sector_steer"] for c in cards], float), axis=0).tolist(),
                   sector_cmd=np.nanmedian(np.array([c["sector_cmd"] for c in cards], float), axis=0).tolist(),
                   contact=0, seeds=len(cards), wedged=0, per_run=cards,
                   flicker=dict(passes=sum(c["flicker"]["passes"] for c in cards),
                                share=_mean([c["flicker"] for c in cards], "share"),
                                changes_per_pass=_mean([c["flicker"] for c in cards], "changes_per_pass")),
                   false_movers=sum(c["false_movers"] for c in cards), tracked_s=sum(c["tracked_s"] for c in cards),
                   false_movers_per_100s=100.0 * sum(c["false_movers"] for c in cards)
                   / max(sum(c["tracked_s"] for c in cards), 1e-3),
                   scan_edges=dict(jumps=_mean([c["scan_edges"] for c in cards], "jumps"),
                                   isolated=_mean([c["scan_edges"] for c in cards], "isolated")))
        if args.measured:
            m = json.loads(Path(args.measured).read_text())
            out["grip"] = [(g["low"], g["high"], g["n"], g["both"]) for g in m["grip"]]
            for key in ("wheel_over_true", "odom_yaw_over_gyro", "odom_yaw_leads_gyro_s"):
                out[key] = m[key]
        if args.stop_m is not None:
            out["stop_m"] = args.stop_m
        if args.stop_s is not None:
            out["stop_s"] = args.stop_s
        Path(args.out).write_text(json.dumps(out))
        return 0

    if args.mode == "table":
        cards = {n: json.loads(Path(p).read_text()) for n, p in (c.split("=", 1) for c in args.cards)}
        print(table(cards))
        if args.detail:
            print(detail(cards))
        return 0

    import eval_gap
    from sim_fidelity import with_floor, with_friction
    from sim_fidelity.calibration import FLOORS
    if args.mode == "fit-plant":
        from run_analysis.loc import DEFAULT_OUT
        calibration = with_floor(args.floor)
        if args.curve:
            calibration = replace(calibration, grip_curve=tuple(map(tuple, json.loads(Path(args.curve).read_text())["curve"])))
        rows = fit_plant(calibration, args.runs, args.loc or DEFAULT_OUT, json.loads(args.grid), args.workers)
        Path(args.out).write_text(json.dumps(rows))
        for row in rows[:8] + rows[-2:]:
            print(f"yaw rms {row['yaw_rms']:.3f} rad/s, lag {row['lag']:+.3f} s, speed rms {row['speed_rms']:.3f} m/s  {row['setting']}")
        return 0
    calibration = with_friction(args.friction, with_floor(args.floor))
    track = args.track or (FLOORS[args.floor].track if args.floor else "course_2026_10_30")
    params = params_from_run(args.params_from, old_code=not args.launched_code)

    if args.curve:
        calibration = replace(calibration, grip_curve=tuple(map(tuple, json.loads(Path(args.curve).read_text())["curve"])))
    for item in args.cal:
        name, _, raw = item.partition("=")
        calibration = replace(calibration, **{name: json.loads(raw)})
    # parse_set returns every default with these applied; only the named ones may
    # replace what the run was launched with.
    named = {item.partition("=")[0] for item in args.set}
    params.update({k: v for k, v in eval_gap.parse_set(args.set).items() if k in named})
    start = None
    if args.start:
        x, y, yaw_deg = (float(v) for v in args.start.split(","))
        start = (x, y, math.radians(yaw_deg))
    jobs = [(track, params, args.first_seed + k, args.seconds, calibration, args.line, False, start)
            for k in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(sim_drive, jobs))
    out = combine(results)
    out.update(open_loop(calibration))
    out["per_seed"] = [{k: r[k] for k in ("seed", "laps", "contact", "wedged", "end", "seconds", "false_movers")} for r in results]
    Path(args.out).write_text(json.dumps(out))
    print(table({"this": out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
