#!/usr/bin/env python3
"""Save a handful of a run's real scans, with where loc.py placed the car for each, as a test fixture.

    python3 tools/run_analysis/scan_fixture.py data/rb2/racerbot_runs/20261106-164401 \\
        --loc ~/.cache/racerbot/loc_car2_1106 --count 5 --out src/pure_pursuit/test/data/scans_20261106-164401.npz

Picks scans spread evenly by distance driven over the run's best-fitting stretch (loc.py's h10, the share
of beams within 10 cm of a wall, at least --min-fit), so the poses are trustworthy truth for a localizer's
test. Writes ranges, angle_min, angle_increment, and per scan the loc.py pose (x, y, yaw of base_link on
the map), its time and its fit. Reads the bag; writes only --out. Needs ROS sourced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_analysis.bagread import messages  # noqa: E402
from run_analysis.loc import COL  # noqa: E402


def pick_rows(rows: np.ndarray, count: int, min_fit: float) -> np.ndarray:
    """`count` row indices spread evenly by distance along the well-fitting rows."""
    good = np.flatnonzero(rows[:, COL["h10"]] >= min_fit)
    if len(good) < count:
        raise SystemExit(f"only {len(good)} scans fit at {min_fit}; need {count}")
    xy = rows[good][:, [COL["x"], COL["y"]]]
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    targets = np.linspace(0.0, along[-1], count + 1)[:-1] + along[-1] / (2 * count)
    return good[np.searchsorted(along, targets).clip(0, len(good) - 1)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--loc", required=True, help="where loc.py wrote this run's .npz")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--min-fit", type=float, default=0.97)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    run = Path(args.run).expanduser().resolve()
    rows = np.load(Path(args.loc).expanduser() / f"{run.name}.npz")["a"]
    chosen = rows[pick_rows(rows, args.count, args.min_fit)]
    wanted = list(chosen[:, COL["t"]])
    ranges, angle_min, angle_increment = [], None, None
    for _topic, t, msg in messages(run, ["/scan"]):
        if wanted and abs(t - wanted[0]) < 1e-6:
            ranges.append(np.asarray(msg.ranges, dtype=np.float32))
            angle_min, angle_increment = float(msg.angle_min), float(msg.angle_increment)
            wanted.pop(0)
        if not wanted:
            break
    if wanted:
        raise SystemExit(f"{len(wanted)} chosen scan times were not found in the bag")
    np.savez_compressed(
        args.out, ranges=np.stack(ranges), angle_min=angle_min, angle_increment=angle_increment,
        poses=chosen[:, [COL["x"], COL["y"], COL["yaw"]]], times=chosen[:, COL["t"]],
        fits=chosen[:, COL["h10"]], run=run.name)
    print(f"wrote {args.out}: {len(ranges)} scans from {run.name}, fits {np.round(chosen[:, COL['h10']], 3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
