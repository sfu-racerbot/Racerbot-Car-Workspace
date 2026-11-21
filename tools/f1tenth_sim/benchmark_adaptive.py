#!/usr/bin/env python3
"""ROS-free paired pursuit benchmark; writes every result before summarising.

RACERBOT_CAR=rb2 python3 tools/f1tenth_sim/benchmark_adaptive.py --output .sim/adaptive-judging.json
Tune separately with --tracks car2_course_2026_11_06b --seeds 1000 1001.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_gap
import run_validation as rv


def run(job):
    track, seed, speed, law, overrides = job
    start = time.monotonic()
    result = rv.run_pure_solo(
        eval_gap.track_path(track), seed, 90.0, law_name=law,
        law_params=overrides if law == 'adaptive_pure_pursuit' else None,
        max_speed=speed, max_lateral_accel=2.5, safety='map', laps=1)
    result.update(track=track, seed=seed, top_speed=speed, law=law,
                  wall_seconds=round(time.monotonic() - start, 3), overrides=overrides)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks', nargs='+', default=[
        'car2_course_2026_11_06a', 'Spielberg', 'car2_course_2026_11_11'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[12345, 12346])
    parser.add_argument('--speeds', nargs='+', type=float, default=[4, 5, 6])
    parser.add_argument('--laws', nargs='+', default=[
        'pure_pursuit', 'adaptive_pure_pursuit', 'delay_lqr'])
    parser.add_argument('--set', action='append', default=[])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    overrides = {k: float(v) for k, v in (item.split('=', 1) for item in args.set)}
    jobs = [(track, seed, speed, law, overrides) for track in args.tracks
            for speed in args.speeds for seed in args.seeds for law in args.laws]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run, jobs):
            rows.append(row)
            args.output.write_text(json.dumps(rows, indent=2) + '\n')
            keys = ('track', 'seed', 'top_speed', 'law', 'lap_time_s', 'passed',
                    'contact_steps', 'max_cross_track_m', 'mean_cross_track_m',
                    'avoid_steps', 'stop_steps', 'escape_steps', 'wall_seconds')
            print(json.dumps({k: row.get(k) for k in keys}), flush=True)


if __name__ == '__main__':
    main()
