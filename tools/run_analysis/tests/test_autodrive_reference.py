import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_analysis.autodrive.november_reference import replay_scan_count


def test_replay_counts_only_eligible_hold_interiors():
    # First hold is short; second is exactly 1 s; third ends with the recording.
    buttons_ms = [(0, True), (999, False), (2000, True), (3000, False),
                  (4000, True), (5500, True)]
    scans_ms = [500, 2001, 2002, 2999, 3000, 4001, 4002, 5499, 5500]
    windows = [(t * 10**6, h) for t, h in buttons_ms]
    assert replay_scan_count([t * 10**6 for t in scans_ms], windows) == 6
    assert replay_scan_count([500 * 10**6, 3000 * 10**6], windows) == 0
