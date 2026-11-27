#!/usr/bin/env python3
"""Run all five dataset examples on one run, reading its bag once.

    python3 tools/run_analysis/dataset_examples/run_all.py data/rb2/racerbot_runs/20261111-152132
    python3 tools/run_analysis/dataset_examples/run_all.py data/rb1/racerbot_runs/20261026-194514 \\
        --map data/rb1/racerbot_auto/20261026-194514-clean

An example whose topics the run lacks says so and is skipped. --map adds loc.py
to localization (the folder the run's map pose is in); path tracking finds the
run's own racing line and map by itself. Results: <out>/<car>-<run>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_analysis.dataset_examples import common as c  # noqa: E402
from run_analysis.dataset_examples import (algorithm_accuracy, lateral_acceleration, localization,  # noqa: E402
                                           steering_response, velocity_estimation)

NEEDS = {
    "localization": ("/odom",),
    "velocity_estimation": ("/odom",),
    "steering_response": ("/ackermann_cmd", "/odom", "/imu/data"),
    "lateral_acceleration": ("/odom", "/imu/data"),
    "algorithm_accuracy": ("/drive", "/odom", "/imu/data"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run directory, data/<car>/racerbot_runs/<time>")
    ap.add_argument("--map", help="map folder for loc.py in the localization example")
    ap.add_argument("--no-loc", action="store_true", help="skip loc.py everywhere")
    ap.add_argument("--out", default=str(c.DEFAULT_OUT), help="results folder (default %(default)s)")
    args = ap.parse_args(argv)
    modules = (localization, velocity_estimation, steering_response, lateral_acceleration, algorithm_accuracy)
    topics = sorted({t for m in modules for t in m.TOPICS})
    run = c.load(args.run, topics, rosout_from=(algorithm_accuracy.NODE,))
    folder = c.out_dir(run, args.out)
    plots = 0
    for name, module in zip(NEEDS, modules):
        missing = [t for t in NEEDS[name] if not run.has(t)]
        if name in ("localization", "velocity_estimation") and c.map_pose(run) is None:
            missing.append("a map pose")
        if missing:
            print(f"{name}: skipped, the run has no {', '.join(missing)}\n")
            continue
        if module is algorithm_accuracy:
            cmd = module.command_tracking(run)
            path = module.path_tracking(run)
            if path is not None and "_curves" in path and not args.no_loc and path["_curves"]["map_dir"] is not None:
                rows = module.locate_with_loc(run, path["_curves"]["map_dir"], path["_curves"]["line"], folder)
                path = module.path_tracking(run, None, rows)
            plots += len(module.plots(run, cmd, path, folder))
            print("\n".join(module.report(run, cmd, path)) + "\n")
            summary = {"command_tracking": {k: v for k, v in cmd.items() if not k.startswith("_")}}
            if path is not None:
                summary["path_tracking"] = {k: v for k, v in path.items() if not k.startswith("_")}
            c.write_summary(folder, name, summary)
            continue
        if module is localization:
            loc_rows = module.locate_with_loc(run, args.map, folder) if args.map and not args.no_loc else None
            result = module.analyse(run, loc_rows)
        else:
            result = module.analyse(run)
        plots += len(module.plots(run, result, folder))
        print("\n".join(module.report(run, result)) + "\n")
        c.write_summary(folder, name, {k: v for k, v in result.items() if not k.startswith("_")})
    print(f"results: {folder} ({plots} plots, summary.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
