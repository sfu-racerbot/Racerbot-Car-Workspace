"""python3 -m race_diagnostics.dataset_classifier <run> [<run> ...]

Grade recorded runs: writes each <run>/qualification.yaml and refreshes
data/dataset_inventory.csv. The batch commands (--all, --inventory) are in
tools/run_analysis/run_triage.py.
"""

import argparse
import sys

from . import classify_run, load_rules


def main(argv=None):
    parser = argparse.ArgumentParser(description='Grade recorded runs for five uses.')
    parser.add_argument('runs', nargs='+', help='run folders, data/<car>/racerbot_runs/<time>')
    parser.add_argument('--rules', help='rules file (default: config/classification_rules.yaml)')
    parser.add_argument('--dry-run', action='store_true', help='print the grades, write nothing')
    args = parser.parse_args(argv)
    rules = load_rules(args.rules)
    for run in args.runs:
        doc = classify_run(run, rules, dry_run=args.dry_run)
        grades = ' '.join(f'{app}={status}' for app, status in doc['summary'].items())
        print(f"{doc['car']}/{doc['run_id']}: {grades}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
