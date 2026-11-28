"""Lets the console's tests run unsourced: its own package and the two it reads
(racerbot_doctor's thresholds, racerbot_car's identity) are importable from
source. Markers are registered here, not in a pytest.ini (see the root pytest.ini)."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
for pkg in ('racerbot_console', 'racerbot_doctor', 'racerbot_car'):
    path = str(SRC / pkg)
    if path not in sys.path:
        sys.path.insert(0, path)
