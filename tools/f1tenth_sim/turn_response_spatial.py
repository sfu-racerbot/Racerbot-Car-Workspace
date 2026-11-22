"""Score a frozen response model against localized November evaluation runs.

Requires slam_rollout.py and run_analysis/slam_motion.py from the SLAM study.
Omit --model to run the reference on exactly the same starts.
"""
import argparse
import json
from pathlib import Path

from fit_turn_response import apply_response, frozen_model, study_calibration


def main():
    from slam_rollout import aggregate, analyze_run, make_plant, split_role

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('car', choices=('rb1', 'rb2'))
    ap.add_argument('loc', type=Path, help='directory of loc.py NPZ tracks')
    ap.add_argument('output', type=Path)
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--model', type=Path)
    args = ap.parse_args()
    if any(split_role(args.car, p.name) != 'nov-eval' for p in args.runs):
        ap.error('expected only November evaluation runs for this car')
    cal = study_calibration(args.car)
    model = frozen_model(args.model, args.car) if args.model else None

    def factory(calibration, seed):
        return apply_response(make_plant(calibration, seed), model)

    runs = [analyze_run(p, args.car, args.loc, cal, factory=factory) for p in args.runs]
    rows = [w for r in runs for w in r['windows']]
    with args.output.open('x') as f:
        json.dump(dict(runs=runs, pooled=aggregate(rows)), f, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
