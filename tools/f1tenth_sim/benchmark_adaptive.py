#!/usr/bin/env python3
"""ROS-free paired pursuit benchmark; writes one row per seeded simulation.

RACERBOT_CAR=rb2 PYTHONPATH="$PWD/.sim/python" python3 tools/f1tenth_sim/benchmark_adaptive.py --output .sim/adaptive-judging.json
Tune separately with --tracks car2_course_2026_11_06b --seeds 1000 1001.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_gap
import run_validation as rv


def run(job):
    track, course, seed, speed, law, overrides, laps = job
    start = time.monotonic()
    result = rv.run_pure_solo(
        course, seed, 120.0 * laps, car_id='rb2', planned_top_speed=speed, law_name=law,
        law_params=overrides if law == 'adaptive_pure_pursuit' else None,
        max_speed=speed, max_lateral_accel=2.5, safety='map', laps=laps)
    result.update(track=track, seed=seed, top_speed=speed, law=law, requested_laps=laps,
                  wall_seconds=round(time.monotonic() - start, 3),
                  overrides=overrides if law == 'adaptive_pure_pursuit' else {})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks', nargs='+', default=[
        'indoor_wide', 'car2_course_2026_11_06a', 'Spielberg', 'car2_course_2026_11_11'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[12345, 12346])
    parser.add_argument('--speeds', nargs='+', type=float, default=[4, 5, 6])
    parser.add_argument('--laws', nargs='+', choices=rv.tracking_laws.LAW_NAMES, default=[
        'pure_pursuit', 'adaptive_pure_pursuit', 'delay_lqr'])
    parser.add_argument('--set', action='append', default=[],
                        help='adaptive_pure_pursuit.KEY=VALUE (adaptive runs only)')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--laps', type=int, default=1, help='laps per run; use 4 to compare flying laps')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    overrides = rv._parse_law_params(args.set)
    if any(not key.startswith('adaptive_pure_pursuit.') for key in overrides):
        parser.error('--set only accepts adaptive_pure_pursuit parameters')
    if args.workers < 1 or args.laps < 1 or any(not math.isfinite(s) or s <= 0 for s in args.speeds):
        parser.error('workers, laps and speeds must be positive and finite')
    # Build generated layouts and fetch missing circuits before workers share them.
    from f1tenth_gym.envs.track.utils import find_track_dir
    courses = {track: eval_gap.track_path(track) for track in args.tracks}
    for track in courses:
        if track in eval_gap.GYM_TRACKS:
            find_track_dir(track)
    jobs = [(track, course, seed, speed, law, overrides, args.laps) for track, course in courses.items()
            for speed in args.speeds for seed in args.seeds for law in args.laws]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run, jobs):
            rows.append(row)
            args.output.write_text(json.dumps(rows, indent=2) + '\n')
            keys = ('track', 'seed', 'top_speed', 'law', 'lap_time_s', 'lap_times_s', 'passed',
                    'contact_steps', 'peak_speed_mps', 'max_cross_track_m', 'mean_cross_track_m',
                    'avoid_steps', 'stop_steps', 'escape_steps', 'wall_seconds')
            print(json.dumps({k: row.get(k) for k in keys}), flush=True)


if __name__ == '__main__':
    main()
