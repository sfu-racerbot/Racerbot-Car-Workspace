#!/usr/bin/env python3
"""Save every scan of a time window of a run, with loc.py's pose for each, as a replay test fixture.

    python3 tools/run_analysis/scan_window_fixture.py data/rb2/racerbot_runs/20261106-175300 \\
        --loc ~/.cache/racerbot/loc_car2_1106 --from 3.0 --to 4.5 \\
        --out src/pure_pursuit/test/data/course_20261106a/false_opponent_20261106-175300.npz

Times are seconds from the first message in the bag, the same clock intent_report.py and rosout replays
print. Writes ranges (n x beams), angle_min, angle_increment, range_min, range_max, times (seconds from
the window start), poses (x, y, yaw of base_link on the map, from loc.py's row for that scan; NaN where
loc.py has none), speeds (loc.py wheel speed at that scan, NaN without a pose), odom_speeds (the latest /odom linear.x at or before it,
what the node had), yaw_rates (the /imu/data gyro z the node's steering damping
reads, NaN where no sample is within 0.1 s), drive_steering and drive_speed (the /drive the node published for that
scan, NaN where none) and, with --gap-params, params_json (that gap_follow_node.yaml as steer_stages.recorded_params
gives it to the mirror). Reads the bag; writes only --out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_analysis.bagread import messages  # noqa: E402
from run_analysis.loc import COL  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--loc", required=True)
    ap.add_argument("--from", dest="t0", type=float, required=True)
    ap.add_argument("--to", dest="t1", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap-params", help="the run's metadata/params/gap_follow_node.yaml, stored in the fixture")
    args = ap.parse_args(argv)
    run = Path(args.run).expanduser().resolve()
    rows = np.load(Path(args.loc).expanduser() / f"{run.name}.npz")["a"]
    by_time = {round(float(t), 6): row for t, row in zip(rows[:, COL["t"]], rows)}
    start = None
    ranges, times, poses, speeds, scan_abs = [], [], [], [], []
    imu, drive, odom = [], [], []
    meta = {}
    for topic, t, msg in messages(run, ["/scan", "/joy", "/imu/data", "/drive", "/odom"]):
        if start is None and topic in ("/scan", "/joy"):
            start = t
        if topic == "/imu/data":
            imu.append((t, msg.angular_velocity.z))
            continue
        if topic == "/odom":
            odom.append((t, msg.twist.twist.linear.x))
            continue
        if topic == "/drive":
            drive.append((t, msg.drive.steering_angle, msg.drive.speed))
            continue
        if topic != "/scan" or not (args.t0 <= t - start <= args.t1):
            continue
        scan_abs.append(t)
        ranges.append(np.asarray(msg.ranges, dtype=np.float32))
        meta = dict(angle_min=float(msg.angle_min), angle_increment=float(msg.angle_increment),
                    range_min=float(msg.range_min), range_max=float(msg.range_max))
        times.append(t - start - args.t0)
        row = by_time.get(round(t, 6))
        poses.append(row[[COL["x"], COL["y"], COL["yaw"]]] if row is not None else [np.nan] * 3)
        speeds.append(row[COL["wheel_v"]] if row is not None else np.nan)
    if not ranges:
        raise SystemExit("no scans in that window")
    imu_a, drive_a = np.asarray(imu, float).reshape(-1, 2), np.asarray(drive, float).reshape(-1, 3)
    odom_a = np.asarray(odom, float).reshape(-1, 2)
    yaw_rates, d_steer, d_speed, odom_speeds = [], [], [], []
    for n, t in enumerate(scan_abs):
        k = int(np.searchsorted(imu_a[:, 0], t, side="right")) - 1
        ko = int(np.searchsorted(odom_a[:, 0], t, side="right")) - 1
        odom_speeds.append(odom_a[ko, 1] if ko >= 0 else np.nan)
        yaw_rates.append(imu_a[k, 1] if k >= 0 and t - imu_a[k, 0] <= 0.1 else np.nan)
        nxt = scan_abs[n + 1] if n + 1 < len(scan_abs) else t + 0.05
        j = int(np.searchsorted(drive_a[:, 0], t, side="right"))
        has = j < len(drive_a) and drive_a[j, 0] < nxt
        d_steer.append(drive_a[j, 1] if has else np.nan)
        d_speed.append(drive_a[j, 2] if has else np.nan)
    extra = {}
    if args.gap_params:
        import json
        from run_analysis.steer_stages import recorded_params
        extra["params_json"] = json.dumps(recorded_params(Path(args.gap_params).parents[2]), sort_keys=True)
    np.savez_compressed(args.out, ranges=np.stack(ranges), times=np.asarray(times), poses=np.asarray(poses, float),
                        speeds=np.asarray(speeds, float), odom_speeds=np.asarray(odom_speeds, float), yaw_rates=np.asarray(yaw_rates, float),
                        drive_steering=np.asarray(d_steer, float), drive_speed=np.asarray(d_speed, float),
                        run=run.name, window=(args.t0, args.t1), **extra, **meta)
    located = int(np.isfinite(np.asarray(poses, float)[:, 0]).sum())
    print(f"wrote {args.out}: {len(ranges)} scans, {located} with a loc.py pose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
