#!/usr/bin/env python3
"""Fixed empty-course, three-lap comparison using eval_gap's fidelity plant.

Run from the workspace root with PYTHONPATH=.sim/python. Each arm is an
addition to the selected car's current time-trial parameters. Output contains
the resolved parameters and every run, including unfinished/contact runs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import eval_gap
from floor_matrix import parse_arm
from gap_follow.launch_speed import time_trial_overrides
from gap_mirror import node_params
from sim_fidelity import calibration_for_car, with_floor


COURSES = (
    ('indoor_oval', None),
    ('car2_course_2026_11_11', 'sfu_course_2026_11_11_rb2'),
    ('asb_10000', None),
    ('narrow_1p2', None),
    ('Spielberg', None),
)
SEEDS = (12345, 12346, 12347)
LAPS = 3


def run(job):
    car, arm, track, floor, seed, params = job
    result = eval_gap.run_track(
        track, params, seed, 240.0, car_id=car, max_laps=LAPS,
        calibration=with_floor(floor, calibration_for_car(car)))
    return dict(car=car, arm=arm, floor=floor, **result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', action='append', required=True,
                        help='name:key=value,...; first arm is baseline')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    arms = [parse_arm(value) for value in args.arm]
    if len({name for name, _ in arms}) != len(arms):
        parser.error('arm names must be unique')
    params = {car: {name: node_params(car_id=car) | time_trial_overrides(car) | extra
                    for name, extra in arms} for car in ('rb1', 'rb2')}
    for track, _ in COURSES:
        eval_gap.track_path(track)
    jobs = [(car, name, track, floor, seed, params[car][name])
            for car in params for name, _ in arms
            for track, floor in COURSES for seed in SEEDS]
    report = dict(courses=COURSES, seeds=SEEDS, laps=LAPS, params=params, runs=[])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            report['runs'].append(result)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(f"{result['car']} {result['arm']} {result['track']} seed={result['seed']} "
                  f"laps={result['lap_times_s']} contact={result['contact_steps']} "
                  f"clearance={result['min_chassis_map_clearance_m']} "
                  f"stops={result['stop_steps']} stuck={result['max_stuck_steps']}", flush=True)
    return 0 if all(r['laps'] >= LAPS and not r['collision'] for r in report['runs']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
