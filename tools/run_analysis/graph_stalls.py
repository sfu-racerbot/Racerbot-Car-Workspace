#!/usr/bin/env python3
"""Find moments when every ROS topic in a run went quiet at once.

    python3 tools/run_analysis/graph_stalls.py data/rb2/racerbot_runs/<ts> [<ts> ...]

On 2026-11-09 six of car 2's fourteen recorded runs had a gap of about 0.35 s
in which /scan, /odom, the particle filter's pose and /drive all stopped
together, each within seconds of a recorder joining the ROS graph. A gap in
one topic is a slow sensor or node; the same gap in all of them at once is
the graph itself stalling, and a car at 5 m/s drives 1.75 m on its last
command through one. This finds those: intervals where *every* watched topic
that the bag holds received nothing for longer than ``--min-gap``.

The detection (``graph_stalls``) is plain Python on lists of receive times and
is tested unsourced (tests/test_graph_stalls.py). Reading the bag needs ROS
sourced. It writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The topics that stalled together on 2026-11-09, at their usual rates: the
# LiDAR (40 Hz), wheel odometry (50 Hz), the particle filter (~50 Hz) and the
# controller (40 Hz). The VESC's own motor and servo commands follow /drive.
DEFAULT_TOPICS = ("/scan", "/odom", "/pf/viz/inferred_pose", "/drive",
                  "/commands/motor/speed")
# Above the slowest of those topics' normal spacing (25 ms at 40 Hz) by a wide
# margin, and well under the 0.30-0.36 s stalls seen on 2026-11-09.
DEFAULT_MIN_GAP = 0.15


def topic_gaps(times, min_gap):
    """[(start, end)] for each pair of consecutive receive times more than
    ``min_gap`` apart. ``times`` need not be sorted."""
    ordered = sorted(times)
    return [(a, b) for a, b in zip(ordered, ordered[1:]) if b - a > min_gap]


def graph_stalls(times_by_topic, min_gap=DEFAULT_MIN_GAP):
    """[(start, end)] intervals longer than ``min_gap`` in which no topic of
    ``times_by_topic`` ({topic: [receive time, ...]}) received anything.

    Only topics with at least two messages count: a topic the run never
    published cannot vouch for anything. Fewer than two such topics returns
    [] -- one quiet topic is not the graph stalling. The interval is bounded
    by the last message any topic got before it and the first any got after.
    """
    if min_gap <= 0:
        raise ValueError("min_gap must be positive")
    live = {topic: sorted(times) for topic, times in times_by_topic.items() if len(times) >= 2}
    if len(live) < 2:
        return []
    start = max(times[0] for times in live.values())
    end = min(times[-1] for times in live.values())
    merged = sorted(t for times in live.values() for t in times if start <= t <= end)
    return [(a, b) for a, b in zip(merged, merged[1:]) if b - a > min_gap]


def read_times(run, topics):
    """{topic: [receive time, ...]} from the run's bag."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bagread
    out = {topic: [] for topic in topics}
    for topic, t, _msg in bagread.messages(run, topics):
        out[topic].append(t)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--topics", default=",".join(DEFAULT_TOPICS))
    ap.add_argument("--min-gap", type=float, default=DEFAULT_MIN_GAP, help="seconds")
    args = ap.parse_args(argv)
    topics = [t for t in args.topics.split(",") if t]
    worst = 0
    for run in args.runs:
        times = read_times(run, topics)
        present = [t for t in topics if len(times[t]) >= 2]
        stalls = graph_stalls(times, args.min_gap)
        first = min((ts[0] for ts in times.values() if ts), default=0.0)
        print(f"{run}: {len(stalls)} graph stall(s) over {', '.join(present) or 'no topics'}")
        for a, b in stalls:
            print(f"  {a - first:7.2f} s after the bag starts: {b - a:.3f} s with nothing on any topic")
        worst = max(worst, len(stalls))
    return 1 if worst else 0


if __name__ == "__main__":
    raise SystemExit(main())
