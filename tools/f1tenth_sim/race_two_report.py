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
    args = ap.parse_args()
    report = archive(args.root)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['table'], indent=2))


if __name__ == '__main__':
    main()
