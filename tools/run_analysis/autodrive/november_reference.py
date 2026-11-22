"""Read only the three named development bags; print sensor metrics, write nothing."""

import argparse
import json
import statistics
from pathlib import Path


def replay_scan_count(scans, buttons):
    # Same selection as follow_cost.analyse -> steer_stages.replay_window:
    # LB holds >= 1 s; scans in [press + 1 ms, release).
    spans, start = [], None
    for t, held in buttons:
        if held and start is None:
            start = t
        elif not held and start is not None:
            spans.append((start, t))
            start = None
    if start is not None:
        spans.append((start, buttons[-1][0]))
    return sum(sum(a + 1_000_000 <= t < b for t in scans)
               for a, b in spans if b - a >= 1_000_000_000)


def measure(run):
    from mcap_ros2.reader import read_ros2_messages

    scans, buttons, geometry = [], [], set()
    for part in (run / "bag").glob("*.mcap"):
        for message in read_ros2_messages(part, topics=["/scan", "/joy"]):
            t, m = message.log_time_ns, message.ros_msg
            if message.channel.topic == "/joy":
                buttons.append((t, m.buttons[4]))
            else:
                scans.append(t)
                geometry.add((len(m.ranges), m.range_max))
    if len(scans) < 2 or not buttons:
        raise ValueError(f"Missing scan or joy samples: {run}")
    scans.sort()
    buttons.sort()
    return dict(run=run.name, scans=len(scans), joy=len(buttons),
                scan_beams_and_max_m=sorted(geometry),
                median_scan_hz=1e9 / statistics.median(b - a for a, b in zip(scans, scans[1:])),
                replay_scans=replay_scan_count(scans, buttons))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Workspace data/ directory")
    data = parser.parse_args().data
    for car, run in [("rb1", "153721"), ("rb1", "155542"), ("rb2", "152132")]:
        print(json.dumps(measure(data / car / "racerbot_runs" / f"20261111-{run}")), flush=True)
