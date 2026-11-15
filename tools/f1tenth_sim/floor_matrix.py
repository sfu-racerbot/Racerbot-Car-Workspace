#!/usr/bin/env python3
"""Floor matrix: compare gap_follow settings arms over courses x floors x grip scales x seeds.

    python3 tools/f1tenth_sim/floor_matrix.py --car rb2 --time-trial \\
        --arm shared: --arm p71:steering_gain=0.8,max_steering_rate=4.0 \\
        --track maps --floors car,sfu_course_2026_11_11_rb2 --grip 0.85,1.0,1.15 --seeds 3

A single floor hides what a setting does on another: on its own suite #71 had zero contact, but on the
race-day floor it stalled on one course and made contact on another (Fable's review, 2026-11-15). So the
acceptance for a gap_follow change is this matrix: each course on the car's own calibration ("car") and on
each named floor (sim_fidelity.calibration.FLOORS), each at several grip scales (#74's set_grip_scale).

The first arm is the reference. A later arm is acceptable (``verdict``) when, in every cell (course, floor,
grip): it has no stall (fewer than ``--laps`` laps in the timeout) or contact unless the reference has it too
in that cell, and its worst minimum chassis-to-map clearance is no lower than the reference's minus
``--clearance-slack`` (0.03 m). Ground truth only: contact and clearance come from the plant's chassis
check, never gym's flag. One process per run; ``--workers`` is capped at 6 and waits while load is above 5.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src' / 'gap_follow'))

MAX_WORKERS = 6
MAX_LOAD = 5.0


def parse_arm(text: str) -> tuple[str, dict]:
    """'name:k=v,k=v' -> (name, {k: v}); values parsed as bool, int, float, else string."""
    name, _, body = text.partition(':')
    if not name:
        raise ValueError(f'arm needs a name before ":", got {text!r}')
    params = {}
    for item in filter(None, body.split(',')):
        key, sep, raw = item.partition('=')
        if not sep or not key:
            raise ValueError(f'arm {name}: expected key=value, got {item!r}')
        params[key] = _value(raw)
    return name, params


def _value(raw: str):
    low = raw.lower()
    if low in ('true', 'false'):
        return low == 'true'
    for kind in (int, float):
        try:
            return kind(raw)
        except ValueError:
            pass
    return raw


def cell_summary(runs: list[dict], laps: int) -> dict:
    """One cell's runs (seeds) -> stalls, contacts, worst clearance, median flying lap."""
    stalls = sum(1 for r in runs if not r['collision'] and r['laps'] < laps)
    contacts = sum(1 for r in runs if r['collision'])
    clear = [r['min_chassis_map_clearance_m'] for r in runs if r['min_chassis_map_clearance_m'] is not None]
    flying = sorted(t for r in runs if not r['collision'] for t in r['lap_times_s'][1:])
    return {'runs': len(runs), 'stalls': stalls, 'contacts': contacts,
            'min_clearance_m': min(clear) if clear else None,
            'median_flying_s': flying[len(flying) // 2] if flying else None}


def verdict(reference: dict, arm: dict, slack: float) -> list[str]:
    """Reasons ``arm`` is worse than ``reference`` cell by cell ([] = acceptable). Both map cell -> summary."""
    reasons = []
    for cell, ref in reference.items():
        got = arm[cell]
        if got['stalls'] > ref['stalls']:
            reasons.append(f'{cell}: stalls {got["stalls"]} > reference {ref["stalls"]}')
        if got['contacts'] > ref['contacts']:
            reasons.append(f'{cell}: contacts {got["contacts"]} > reference {ref["contacts"]}')
        if ref['min_clearance_m'] is not None and (
                got['min_clearance_m'] is None or got['min_clearance_m'] < ref['min_clearance_m'] - slack):
            reasons.append(f'{cell}: clearance {got["min_clearance_m"]} < reference {ref["min_clearance_m"]} - {slack}')
    return reasons


def _wait_for_load():
    while os.getloadavg()[0] > MAX_LOAD:
        time.sleep(5)


def _one(job):
    import eval_gap
    from sim_fidelity import calibration_for_car, with_floor
    arm, track, floor, grip, seed, params, timeout, laps, car = job
    cal = calibration_for_car(car) if floor == 'car' else with_floor(floor, calibration_for_car(car))
    r = eval_gap.run_track(track, params, seed, timeout, calibration=cal, car_id=car, max_laps=laps,
                           grip_scale=None if grip == 1.0 else grip)
    keep = ('laps', 'collision', 'lap_times_s', 'min_chassis_map_clearance_m', 'steering_reversals_per_s')
    return arm, track, floor, grip, seed, {k: r.get(k) for k in keep}


def main(argv=None) -> int:
    import eval_gap
    import gap_mirror
    from gap_follow.launch_speed import time_trial_overrides
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--car', choices=('rb1', 'rb2'), required=True)
    ap.add_argument('--arm', action='append', required=True, help="name:k=v,... (first = reference)")
    ap.add_argument('--time-trial', action='store_true', help="apply the car's time_trial overrides under each arm")
    ap.add_argument('--track', default='maps', help="as eval_gap --track")
    ap.add_argument('--floors', default='car', help="comma list: 'car' (own calibration) and FLOORS names")
    ap.add_argument('--grip', default='1.0', help='comma list of grip scales')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--first-seed', type=int, default=12345)
    ap.add_argument('--laps', type=int, default=2)
    ap.add_argument('--timeout', type=float, default=90.0)
    ap.add_argument('--clearance-slack', type=float, default=0.03)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--output', help='write all runs and the verdict as JSON')
    args = ap.parse_args(argv)
    arms = [parse_arm(a) for a in args.arm]
    base = gap_mirror.node_params(car_id=args.car)
    if args.time_trial:
        base = base | time_trial_overrides(args.car)
    floors = [f.strip() for f in args.floors.split(',') if f.strip()]
    grips = [float(g) for g in args.grip.split(',')]
    tracks = eval_gap.select_tracks(args.track)
    jobs = [(name, t, f, g, args.first_seed + k, base | extra, args.timeout, args.laps, args.car)
            for name, extra in arms for t in tracks for f in floors for g in grips for k in range(args.seeds)]
    results: dict = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, MAX_WORKERS)) as pool:
        futures = []
        for job in jobs:
            _wait_for_load()
            futures.append(pool.submit(_one, job))
        for fut in futures:
            arm, track, floor, grip, seed, r = fut.result()
            results.setdefault(arm, {}).setdefault(f'{track}|{floor}|{grip}', []).append(r)
    summaries = {arm: {cell: cell_summary(runs, args.laps) for cell, runs in cells.items()}
                 for arm, cells in results.items()}
    ref_name = arms[0][0]
    verdicts = {name: verdict(summaries[ref_name], summaries[name], args.clearance_slack) for name, _ in arms[1:]}
    for cell in sorted(summaries[ref_name]):
        print(cell, ' | '.join(f"{name}: lap {s['median_flying_s']} clr {s['min_clearance_m']} "
                                f"stall {s['stalls']} contact {s['contacts']}"
                                for name, s in ((n, summaries[n][cell]) for n, _ in arms)))
    for name, reasons in verdicts.items():
        print(f'VERDICT {name}:', 'acceptable' if not reasons else f'{len(reasons)} worse cells')
        for reason in reasons:
            print('   ', reason)
    if args.output:
        Path(args.output).write_text(json.dumps({'args': vars(args), 'runs': results, 'summaries': summaries,
                                                 'verdicts': verdicts}, indent=1))
    return 0 if all(not r for r in verdicts.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
