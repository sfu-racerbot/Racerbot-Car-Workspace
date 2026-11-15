"""Build one race line with the planner's physics overridden: lateral grip, speed delay, grip coupling.

    PYTHONPATH=<checkout with profile_grip_coupling>/src/pure_pursuit python3 plan_build.py --run-dir D --lat 2.95 \\
        --speed-delay 0.22 --coupling 0.5

D holds map.yaml, its image and raceline_raw.csv (as pure_pursuit.race_line expects). Writes the same outputs as
``python -m pure_pursuit.race_line`` (raceline_profiled.csv, race_line.json) for rb2's effective parameters with
those three values replaced, and exits 1 (writing build_failed.txt) when no candidate passes. The command line of
race_line has no flag for the speed delay or the coupling, hence this script.
"""
import argparse
import sys
from pathlib import Path

from pure_pursuit.occupancy_map import OccupancyMap
from pure_pursuit.race_line import RaceLineConfig, build_race_line, load_params, write_race_line
from pure_pursuit.racing_math import load_xy_csv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--car", default="rb2")
    ap.add_argument("--lat", type=float, required=True, help="planner lateral accel, left and right, m/s^2")
    ap.add_argument("--speed-delay", type=float, required=True, help="planner speed delay, s")
    ap.add_argument("--coupling", type=float, required=True, help="profile_grip_coupling, 0 to 1")
    args = ap.parse_args(argv)
    values = load_params(args.car)
    values.update(profile_max_lateral_accel=args.lat, profile_max_lateral_accel_right=args.lat,
                  profile_speed_delay=args.speed_delay, profile_grip_coupling=args.coupling)
    cfg = RaceLineConfig.from_params(values)
    grid = OccupancyMap.from_yaml(args.run_dir / "map.yaml", inflate_cells=0)
    seed = load_xy_csv(args.run_dir / "raceline_raw.csv")
    line = build_race_line(grid, seed, cfg)
    for candidate in line.candidates:
        print(f'{candidate.source}: {"pass" if candidate.ok else "fail"}; {candidate.est_lap_s:.3f} s; {candidate.reason}')
    print(f'choice: {line.source if line.ok else "none"}; {line.reason}')
    if not line.ok:
        (args.run_dir / "build_failed.txt").write_text(line.reason + "\n")
        return 1
    write_race_line(args.run_dir, line, values["floor_test_speed"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
