#!/usr/bin/env python3
"""Archive per-seed verdicts and decision transitions from race_two outputs."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def archive(root):
    runs = []
    table = []
    for path in sorted(root.glob('*/*/results.json')):
        results = json.loads(path.read_text())
        times = [r['race_time_s'] for r in results if r['race_time_s'] is not None]
        for result in results:
            with (path.parent / f"{result['seed']}.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            events = []
            previous = None
            for index, row in enumerate(rows):
                category = (row['active'], row['safety'])
                if category != previous or index == len(rows) - 1:
                    events.append(row)
                previous = category
            runs.append(dict(result=result, events=events))
        table.append(dict(course=path.parts[-3], opponent=path.parts[-2], seeds=len(results),
                          attempted=sum(r['attempted'] for r in results),
                          completed=sum(r['completed'] for r in results),
                          abandoned=sum(r['abandoned'] for r in results),
                          declared_completed=sum(r['declared_completed'] for r in results),
                          stuck_median_s=statistics.median(r['stuck_s'] for r in results),
                          stuck_range_s=[min(r['stuck_s'] for r in results), max(r['stuck_s'] for r in results)],
                          car_contact_runs=sum(r['car_contact_ticks'] > 0 for r in results),
                          wall_contact_runs=sum(r['wall_contact_ticks'] > 0 for r in results),
                          emergency_stop_ticks=sum(r['emergency_stop_ticks'] for r in results),
                          finished=sum(r['race_time_s'] is not None for r in results),
                          race_time_median_s=statistics.median(times) if times else None,
                          race_time_range_s=[min(times), max(times)] if times else None,
                          elapsed_range_s=[min(r['elapsed_s'] for r in results), max(r['elapsed_s'] for r in results)]))
    return dict(table=table, runs=runs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--diagnose-seed', type=int)
    args = ap.parse_args()
    report = archive(args.root)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['table'], indent=2))
    if args.diagnose_seed is not None:
        for run in report['runs']:
            r, events = run['result'], run['events']
            if r['seed'] != args.diagnose_seed or r['opponent'] == 'solo':
                continue
            commit = next((e for e in events if e['active'] == 'True'), None)
            emergency = next((e for e in events if e['safety'] in (
                'emergency_obstacle', 'body_contact', 'emergency_escape')), None)
            print(r['track'], r['opponent'], r['seed'])
            for label, event in (('commit', commit), ('emergency', emergency), ('end', events[-1])):
                if event is None:
                    continue
                lead = float(event['ego_progress']) - float(event['opp_progress']) - float(event['head_start'])
                print(label, {k: event[k] for k in ('time', 'ego_speed', 'ego_command',
                      'desired_steering', 'target', 'arc_clearance', 'mapped_forward',
                      'status', 'safety', 'safety_detail', 'car_contact', 'wall_contact')}, 'truth lead', round(lead, 3))


if __name__ == '__main__':
    main()
