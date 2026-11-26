#!/usr/bin/env python3
"""What a recording really is, before anyone draws a conclusion from it.

    python3 tools/run_analysis/run_triage.py data/<car>/racerbot_runs/<time> [more runs]

For each run: which driving code ran, how long LB was held and in how many holds, how far the wheels went and
how fast, and whether the car was moved by hand between holds. A run carried between holds cannot map: nothing the
car senses saw it move. The answer then names where a clean replay can start from
(replay_map_course.py --from-seconds).

Why: on 2026-10-31, most of the 30 Oct recordings that "failed to map" turned out to be other teams' algorithms
or a few seconds of driving (docs/superpowers/specs/2026-10-30-slam-reliability.md). Carrying is judged from the
car's own scans and wheels: placing single scans on a reference map once called a car carried 12 m that wasn't.

Dataset qualification -- what each recording is usable for (race_diagnostics.dataset_classifier, the same code
bringup's recorder runs after every recording; docs/run-diagnostics.md#dataset-qualification):

    python3 tools/run_analysis/run_triage.py --qualify <run> [more runs]   # grade these runs
    python3 tools/run_analysis/run_triage.py --qualify --all              # grade every run in data/
    python3 tools/run_analysis/run_triage.py --inventory                  # rebuild data/dataset_inventory.csv

Each writes <run>/qualification.yaml and refreshes data/dataset_inventory.csv, only where something changed;
--dry-run prints the grades and writes nothing. Reading a bag needs ROS sourced or rosbags installed.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'src' / 'race_diagnostics'))

# The driving-code names and the carried-between-holds check live with the classifier now, so the two can
# never disagree; imported back under their old names.
from race_diagnostics.dataset_classifier.metrics import (  # noqa: E402,F401
    DRIVERS, MOVED_M, STILL_MPS, algorithms, moved_between, scan_change)

SHORT_S = 5.0  # less LB than this is not a mapping run


def hold_stats(odom, windows):
    """[(start, seconds, metres, top m/s)] per LB hold, from odom rows (t, v)."""
    odom = np.asarray(odom, dtype=float).reshape(-1, 2)
    out = []
    for start, end in windows:
        rows = odom[(odom[:, 0] >= start) & (odom[:, 0] <= end)]
        if len(rows) > 1:
            metres = float(np.sum(np.abs(rows[1:, 1]) * np.diff(rows[:, 0])))
            top = float(np.max(rows[:, 1]))
        else:
            metres, top = 0.0, 0.0
        out.append((float(start), float(end - start), metres, top))
    return out


def verdict(holds, carried):
    """One line: whether this run can be judged as a mapping run, and from where."""
    if not holds or sum(h[1] for h in holds) < SHORT_S:
        return 'not a mapping run: LB held under 5 s'
    if any(carried):
        last = max(i for i, c in enumerate(carried) if c) + 1
        return (f'carried between holds {last} and {last + 1}: nothing before can map. A clean replay starts '
                f'at the hold after it (--from-seconds {holds[last][0] - holds[0][0] + 0.0:.0f} or later)')
    return 'one continuous car: usable'


def triage(run_dir):
    from run_analysis import runlog
    run = runlog.load_run(run_dir)
    processes = Path(run_dir) / 'metadata' / 'processes.txt'
    windows = run.lb_windows()
    odom = run.odom[:, :2] if len(run.odom) else np.zeros((0, 2))
    holds = hold_stats(odom, windows)
    carried = moved_between(run.scan_t, run.scan, odom, windows)
    t0 = run.scan_t[0] if len(run.scan_t) else 0.0
    return {
        'algorithms': (algorithms(processes.read_text(errors='replace')) if processes.exists()
                       else ['unknown: no metadata/processes.txt']),
        'seconds': float(run.scan_t[-1] - t0) if len(run.scan_t) else 0.0,
        'holds': [(start - t0, seconds, metres, top) for start, seconds, metres, top in holds],
        'carried': carried, 'verdict': verdict(holds, carried)}


def qualify(args, parser):
    """--qualify / --inventory: grade runs with the dataset classifier and print the counts."""
    from race_diagnostics.dataset_classifier import classifier, load_rules, outputs
    rules = load_rules(args.rules)
    data_root = Path(args.data_root)
    if args.inventory and not args.qualify:
        changed = outputs.rebuild_inventory(data_root, rules)
        print(f"{data_root / outputs.INVENTORY}: {'rebuilt' if changed else 'already up to date'}")
        return 0
    if args.all == bool(args.runs):
        parser.error('--qualify needs run folders or --all, not both')
    runs = classifier.find_runs(data_root) if args.all else args.runs
    apps = list(rules['applications'])
    counts = {app: dict.fromkeys(('usable', 'conditional', 'unusable'), 0) for app in apps}
    for run_dir, doc in classifier.classify_all(data_root if args.all else None, rules, runs=runs,
                                                jobs=args.jobs, dry_run=args.dry_run):
        for app in apps:
            counts[app][doc['summary'][app]] += 1
        error = f"  ERROR {doc['error']}" if doc.get('error') else ''
        print(f"{doc['car']}/{doc['run_id']}: " + ' '.join(
            f"{app.split('_')[0]}={doc['summary'][app]}" for app in apps) + error, flush=True)
    if not args.all and not args.dry_run:
        for data_root_of_run in {classifier.run_location(r)[0] for r in runs} - {None}:
            outputs.update_inventory(data_root_of_run, rules,
                                     cars={classifier.run_location(r)[1] for r in runs})
    print(f"\n{len(runs)} run(s){' (dry run: nothing written)' if args.dry_run else ''}")
    print(f"{'application':22s} {'usable':>7s} {'conditional':>12s} {'unusable':>9s}")
    for app in apps:
        c = counts[app]
        print(f"{app:22s} {c['usable']:7d} {c['conditional']:12d} {c['unusable']:9d}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('runs', type=Path, nargs='*')
    parser.add_argument('--qualify', action='store_true',
                        help='grade the runs for the five dataset uses (writes qualification.yaml)')
    parser.add_argument('--all', action='store_true', help='with --qualify: every run under --data-root')
    parser.add_argument('--inventory', action='store_true',
                        help='rebuild dataset_inventory.csv from the qualification files')
    parser.add_argument('--rules', help='rules file (default: src/race_diagnostics/config/classification_rules.yaml)')
    parser.add_argument('--data-root', default=str(ROOT / 'data'), help='the data/ folder (default: this checkout\'s)')
    parser.add_argument('--jobs', type=int, default=1, help='runs graded in parallel')
    parser.add_argument('--dry-run', action='store_true', help='print the grades, write nothing')
    args = parser.parse_args(argv)
    if args.qualify or args.inventory:
        return qualify(args, parser)
    if not args.runs:
        parser.error('give run folders, or --qualify / --inventory')
    for run in args.runs:
        try:
            result = triage(run)
        except (OSError, ValueError) as exc:
            print(f'{run.name}: unreadable: {exc}')
            continue
        metres = sum(h[2] for h in result['holds'])
        print(f"{run.name}: {result['seconds']:.0f} s, {', '.join(result['algorithms']) or 'no driving code'}; "
              f"{len(result['holds'])} LB hold(s), {metres:.0f} m by the wheels")
        for index, (start, seconds, hold_m, top) in enumerate(result['holds']):
            moved = '  then carried' if index < len(result['carried']) and result['carried'][index] else ''
            print(f'    hold {index + 1}: at {start:5.1f} s for {seconds:5.1f} s, {hold_m:5.1f} m, top {top:.2f} m/s{moved}')
        print(f"  -> {result['verdict']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
