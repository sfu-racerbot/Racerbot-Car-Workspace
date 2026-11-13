#!/usr/bin/env python3
"""Replay a gap_follow run's scans through the mirror and print what each steering stage contributed.

    python3 tools/run_analysis/steer_stages.py data/rb1/racerbot_runs/20261111-155542 \\
        --loc ~/.cache/racerbot/loc --from 84.5 --to 88.2 [--csv out.csv]

Runs tools/f1tenth_sim/gap_mirror.GapController (the node's own per-scan logic without ROS) with the run's recorded
parameters (metadata/params/gap_follow_node.yaml), from the start of the LB hold that contains the window so the
steering basis, the opponent tracker and the launch state are warm, feeding it the bag's scans, the wheel speed
the node had and the /imu/data gyro it used for steering damping. For every scan in the window it prints, in the
order the node applies them: the gap it aimed at, the bearing term, centering, damping, the clipped sum, the arc
guard's correction, the slewed command, the recorded /drive steering for the same scan (``parity``), the opponent
mode and the speed factor that bound. Side clearance is the minimum raw range 60-120 degrees each side, forward is
the minimum raw range within the forward stop cone.

Times are seconds from the first /joy, /odom or /drive_intent message (the clock intent_report.py prints); the
analysis itself keys on absolute receive time. Reads the bag; writes only --csv.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "tools", ROOT / "tools" / "f1tenth_sim", ROOT / "src" / "gap_follow"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gap_mirror  # noqa: E402
from run_analysis import runlog  # noqa: E402
from run_analysis.bagread import messages  # noqa: E402
from run_analysis.loc import COL  # noqa: E402

DT = gap_mirror.CONTROL_DT


class Lidar:
    def __init__(self, run):
        self.angle_min, self.angle_increment, self.range_min = run.angle_min, run.angle_increment, run.range_min


def recorded_params(run_dir) -> dict:
    """The parameters the node ran with, as the mirror takes them (metadata/params/gap_follow_node.yaml)."""
    data = yaml.safe_load((Path(run_dir) / "metadata" / "params" / "gap_follow_node.yaml").read_text())
    block = data.get("/gap_follow_node") or data["gap_follow_node"]
    params = dict(block["ros__parameters"])
    params = {k: v for k, v in params.items() if k not in gap_mirror.NOT_MIRRORED}
    # A parameter the run's node did not have yet is filled from the mirror's legacy table, else it is off (flank
    # creep, 2026-11-11, came after 20261111-155542: length 0 is no flank creep).
    for name in sorted(gap_mirror.REQUIRED_PARAMS - set(params)):
        params[name] = {"flank_creep_length": 0.0, "flank_creep_margin": 0.05}.get(name, gap_mirror.LEGACY_VALUES.get(name))
        print(f"note: {name} not in the recorded params, using {params[name]!r}", file=sys.stderr)
    return params


def bag_origin(run_dir) -> float:
    """Absolute time of the first /joy, /odom or /drive_intent message: intent_report.py's zero."""
    for _topic, stamp, _m in messages(run_dir, ["/joy", "/odom", "/drive_intent"]):
        return stamp
    raise ValueError("no messages")


def gyro_series(run_dir):
    """(t, wz) of /imu/data, the gyro the node's steering damping reads."""
    t, wz = [], []
    for _topic, stamp, m in messages(run_dir, ["/imu/data"]):
        t.append(stamp)
        wz.append(m.angular_velocity.z)
    return np.array(t), np.array(wz)


def raw_min(scan, angles, lo_deg, hi_deg, range_min, max_range):
    sel = (angles >= math.radians(lo_deg)) & (angles <= math.radians(hi_deg))
    r = scan[sel]
    r = r[np.isfinite(r) & (r > range_min) & (r < max_range)]
    return float(r.min()) if r.size else math.inf


def replay_window(run, params, t_from, t_to, gyro, imu_timeout=0.1, loc=None):
    """Rows for the scans in (t_from, t_to) absolute, replayed from the start of the LB hold that contains t_from."""
    press = max((a for a, b in run.lb_windows(min_duration=0.0) if a <= t_from < b), default=None)
    if press is None:
        raise ValueError("t_from is not inside an LB hold")
    ctrl = gap_mirror.GapController(dict(params), Lidar(run))
    idx = np.where((run.scan_t > press) & (run.scan_t < t_to))[0]
    ctrl.last_command_time = (run.scan_t[idx[0]] - press) / DT - 1.0
    angles = run.beam_angles()
    drive_t = run.drive[:, 0]
    gain_ref = params["steering_gain_speed_ref"]
    rows = []
    for n, i in enumerate(idx):
        t = run.scan_t[i]
        v = run.odom_speed_at(t)
        k = int(np.searchsorted(gyro[0], t, side="right")) - 1
        yaw = float(gyro[1][k]) if k >= 0 and t - gyro[0][k] <= imu_timeout else None
        scan = run.scan[i].astype(np.float64)
        steer_basis_before = ctrl.steering_basis
        last_speed_before = ctrl.last_commanded_speed
        steer, speed = ctrl.command(scan, v, (t - press) / DT, yaw_rate=yaw)
        if t < t_from:
            continue
        nxt = run.scan_t[idx[n + 1]] if n + 1 < len(idx) else t + 0.05
        j = int(np.searchsorted(drive_t, t, side="right"))
        real = run.drive[j, 2:4] if j < len(drive_t) and drive_t[j] < nxt else (math.nan, math.nan)
        tel = ctrl.telemetry
        movers = ctrl.opponents.moving_body() if ctrl.opponents is not None else []
        gain_basis = max(last_speed_before, min(abs(v), params["max_speed"]))
        gain = params["steering_gain"] * float(min(1.0, gain_ref / gain_basis)) if gain_basis > gain_ref else params["steering_gain"]
        target = tel.get("target_angle", math.nan)
        bearing = gain * target if math.isfinite(target) else math.nan
        centering = tel.get("centering_bias", math.nan)
        damping = tel.get("steering_damping", math.nan)
        pre = tel.get("desired_steering", math.nan)
        guard = tel.get("arc_guard", 0.0)
        row = dict(
            t=t, t_rel=0.0, speed=v, yaw_rate=yaw if yaw is not None else math.nan,
            left_60_120=raw_min(scan, angles, 60, 120, run.range_min, params["max_range"]),
            right_60_120=raw_min(scan, angles, -120, -60, run.range_min, params["max_range"]),
            forward=raw_min(scan, angles, -params["forward_stop_fov_deg"] / 2, params["forward_stop_fov_deg"] / 2,
                            run.range_min, params["max_range"]),
            side_left=tel.get("side_left", math.nan), side_right=tel.get("side_right", math.nan),
            gap_bearing=math.degrees(target) if math.isfinite(target) else math.nan,
            gap_width_deg=math.degrees(tel.get("gap_width", math.nan)) if "gap_width" in tel else math.nan,
            gap_depth=tel.get("target_distance", math.nan), fallback=bool(tel.get("fallback", False)),
            bearing=bearing, centering=centering, damping=damping,
            clipped_sum=float(np.clip(bearing + centering + damping, -params["max_steering_angle"],
                                      params["max_steering_angle"])) if math.isfinite(bearing) else math.nan,
            arc_guard=guard, desired=pre, steering=steer, slew_basis_before=steer_basis_before,
            real_steering=float(real[0]), real_speed=float(real[1]), speed_cmd=speed,
            opponent_mode=ctrl.opponent_mode or "", n_movers=len(movers),
            mover_xy=";".join(f"({m[0]:.2f},{m[1]:.2f})" for m in movers),
            binding=ctrl.binding_cap or "", follow_cap=tel.get("follow_cap", math.inf),
            stopped=ctrl.binding_cap == "stop", stop_reason=ctrl.stop_reason if ctrl.binding_cap == "stop" else "",
            creeping=bool(tel.get("creeping", False)),
        )
        if loc is not None:
            k2 = int(np.searchsorted(loc[:, COL["t"]], t))
            k2 = min(max(k2, 0), len(loc) - 1)
            row.update(x=float(loc[k2, COL["x"]]), y=float(loc[k2, COL["y"]]), yaw=float(loc[k2, COL["yaw"]]),
                       true_speed=float(loc[k2, COL["wheel_v"]]))
        rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--loc", help="loc.py output directory (true pose and speed)")
    ap.add_argument("--from", dest="t0", type=float, required=True, help="seconds from the bag's first joy/odom/intent")
    ap.add_argument("--to", dest="t1", type=float, required=True)
    ap.add_argument("--csv")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a parameter for a counterfactual (YAML value), e.g. enable_opponent_awareness=false")
    args = ap.parse_args(argv)
    run_dir = Path(args.run).expanduser().resolve()
    run = runlog.load_run(run_dir)
    origin = bag_origin(run_dir)
    loc = None
    if args.loc:
        loc = np.load(Path(args.loc).expanduser() / f"{run_dir.name}.npz")["a"]
    params = recorded_params(run_dir)
    for item in args.set:
        name, value = item.split("=", 1)
        params[name] = yaml.safe_load(value)
    rows = replay_window(run, params, origin + args.t0, origin + args.t1, gyro_series(run_dir), loc=loc)
    for r in rows:
        r["t_rel"] = r["t"] - origin
    paired = [r for r in rows if math.isfinite(r["real_steering"])]
    err = np.array([r["steering"] - r["real_steering"] for r in paired])
    print(f"origin {origin:.3f} (first joy/odom/intent); {len(rows)} scans, {len(paired)} paired with /drive; "
          f"steering error max {np.abs(err).max():.4f} rad, rms {np.sqrt(np.mean(err ** 2)):.4f} rad")
    cols = ["t_rel", "speed", "left_60_120", "right_60_120", "forward", "gap_bearing", "gap_width_deg", "bearing",
            "centering", "damping", "clipped_sum", "arc_guard", "steering", "real_steering", "opponent_mode",
            "n_movers", "binding", "speed_cmd", "real_speed"]
    print(" ".join(f"{c[:11]:>11}" for c in cols))
    for r in rows:
        print(" ".join(f"{r[c]:11.3f}" if isinstance(r[c], float) else f"{str(r[c]):>11}" for c in cols))
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
