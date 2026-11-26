"""dataset_classifier: grades every recorded run for five uses.

    localization, velocity_estimation, steering_response,
    lateral_acceleration, algorithm_accuracy

Each one is graded usable | conditional | unusable, with the reason. A run
can be usable for several uses at once. The thresholds live in
config/classification_rules.yaml, not in this code. The classifier only
reads the bag and never changes or deletes a recording: it adds
<run>/qualification.yaml and a row in data/dataset_inventory.csv.

Public interface (everything else in this folder is internal):
    classify_run(run_dir)         grade one run, write <run>/qualification.yaml
    classify_all(data_root)       grade every run under data/
    rebuild_inventory(data_root)  regenerate data/dataset_inventory.csv
    load_rules(path=None)         read and check classification_rules.yaml
    write_run_info(run_dir, ...)  write <run>/run_info.yaml when recording starts

Run it by hand:
    python3 -m race_diagnostics.dataset_classifier <run> [<run> ...]
    python3 tools/run_analysis/run_triage.py --qualify --all
How a grade is reached: docs/run-diagnostics.md#dataset-qualification
"""

from .classifier import classify_all, classify_run
from .outputs import rebuild_inventory
from .rules import load_rules
from .run_info import write_run_info

__all__ = ['classify_run', 'classify_all', 'rebuild_inventory', 'load_rules', 'write_run_info']
