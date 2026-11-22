"""Count frozen November replay windows that pass only the mux-command age gate."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from mcap.reader import make_reader


GROUPS = (
    ("rb1", "development", "20261111", "153409 153436 153721 155542"),
    ("rb1", "evaluation", "20261116", "095058 095242 095652"),
    ("rb2", "development", "20261111", "135934 142254 152132"),
    ("rb2", "evaluation", "20261111", "135453 140322 141227 142036 142604 152844 154204"),
)


def count_windows(times):
    t = np.sort(np.asarray(times, dtype=np.int64))
    if not t.size:
        return 0, 0
    complete = fresh = 0
    ticks = np.arange(480, dtype=np.int64) * 25_000_000
    for b in range(int(t[0]), int(t[-1]) - 12_000_000_000 + 1, 10_000_000_000):
        grid = b + ticks
        age = grid - t[np.searchsorted(t, grid, side="right") - 1]
        complete += 1
        fresh += int(age.max() <= 100_000_000)
    return complete, fresh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="pinned Racerbot data directory")
    args = parser.parse_args()
    out = csv.writer(sys.stdout)
    out.writerow(["car", "domain", "run", "all_12s_blocks", "mux_age_pass", "mux_age_reject"])
    for car, domain, day, ids in GROUPS:
        for run_id in ids.split():
            run = f"{day}-{run_id}"
            chunks = sorted((args.data / car / "racerbot_runs" / run / "bag").glob("*.mcap"))
            if not chunks:
                raise FileNotFoundError(f"no MCAPs for {car}/{run}")
            times = []
            for chunk in chunks:
                with chunk.open("rb") as stream:
                    msgs = make_reader(stream).iter_messages(topics=["/ackermann_cmd"])
                    times.extend(msg.log_time for _, _, msg in msgs)
            complete, fresh = count_windows(times)
            out.writerow([car, domain, run, complete, fresh, complete - fresh])


if __name__ == "__main__":
    main()
