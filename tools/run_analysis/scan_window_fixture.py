#!/usr/bin/env python3
"""Save every scan of a time window of a run, with loc.py's pose for each, as a replay test fixture.

    python3 tools/run_analysis/scan_window_fixture.py data/rb2/racerbot_runs/20261106-175300 \\
        --loc ~/.cache/racerbot/loc_car2_1106 --from 3.0 --to 4.5 \\
        --out src/pure_pursuit/test/data/course_20261106a/false_opponent_20261106-175300.npz

Times are seconds from the first message in the bag, the same clock intent_report.py and rosout replays
print. Writes ranges (n x beams), angle_min, angle_increment, range_min, range_max, times (seconds from
the window start), poses (x, y, yaw of base_link on the map, from loc.py's row for that scan; NaN where
loc.py has none) and speeds (wheel odometry at that scan). Reads the bag; writes only --out.
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
    args = ap.parse_args(argv)
    run = Path(args.run).expanduser().resolve()
    rows = np.load(Path(args.loc).expanduser() / f"{run.name}.npz")["a"]
    by_time = {round(float(t), 6): row for t, row in zip(rows[:, COL["t"]], rows)}
    start = None
    ranges, times, poses, speeds = [], [], [], []
    meta = {}
    for topic, t, msg in messages(run, ["/scan", "/joy"]):
        if start is None:
            start = t
        if topic != "/scan" or not (args.t0 <= t - start <= args.t1):
            continue
        ranges.append(np.asarray(msg.ranges, dtype=np.float32))
        meta = dict(angle_min=float(msg.angle_min), angle_increment=float(msg.angle_increment),
                    range_min=float(msg.range_min), range_max=float(msg.range_max))
        times.append(t - start - args.t0)
        row = by_time.get(round(t, 6))
        poses.append(row[[COL["x"], COL["y"], COL["yaw"]]] if row is not None else [np.nan] * 3)
        speeds.append(row[COL["wheel_v"]] if row is not None else np.nan)
    if not ranges:
        raise SystemExit("no scans in that window")
    np.savez_compressed(args.out, ranges=np.stack(ranges), times=np.asarray(times), poses=np.asarray(poses, float),
                        speeds=np.asarray(speeds, float), run=run.name, window=(args.t0, args.t1), **meta)
    located = int(np.isfinite(np.asarray(poses, float)[:, 0]).sum())
    print(f"wrote {args.out}: {len(ranges)} scans, {located} with a loc.py pose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
