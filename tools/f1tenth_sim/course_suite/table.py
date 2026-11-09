"""Summarise suite_run results: per track, map racer vs the best clean gap_follow configuration.

    python3 suite_table.py <out dir> [tag]
gap_follow's best: the lowest flying median over configurations where at least half the seeds had no
contact and did not wedge, and at least 2 flying laps exist. Map racer: flying median (laps 2..), runs with
contact, runs unfinished. margin = (gf - mr) / gf.
"""
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else "mr"
rows = [json.loads(l) for l in open(out / "results.jsonl")]
race, gap, build = defaultdict(list), defaultdict(dict), {}
for r in rows:
    if r["stage"] == "race" and r.get("tag") == tag:
        race[r["track"]].append(r)
    elif r["stage"] == "gap":
        gap[r["track"]][r["cfg"]] = r
    elif r["stage"] == "build":
        build[r["track"]] = r
margins, wins, total_contact, lines = [], 0, 0, []
for t in sorted(set(race) | set(gap)):
    mr = race.get(t, [])
    fly = [x for r in mr if "laps" in r for x in r["laps"][1:]]
    contact = sum(1 for r in mr if r.get("contact", 0) > 0)
    unfinished = sum(1 for r in mr if not r.get("finished"))
    total_contact += contact
    best, best_cfg = math.inf, None
    for cfg, g in gap.get(t, {}).items():
        if "seeds" not in g:
            continue
        clean = sum(1 for s in g["seeds"] if not s["contact"] and not s["wedged"])
        if clean * 2 >= len(g["seeds"]) and len(g["flying"]) >= 2:
            m = statistics.median(g["flying"])
            if m < best:
                best, best_cfg = m, cfg
    mrm = statistics.median(fly) if fly else math.inf
    margin = (best - mrm) / best if math.isfinite(best) and math.isfinite(mrm) else math.nan
    if math.isfinite(margin):
        margins.append(margin); wins += margin >= 0.05
    b = build.get(t, {})
    lines.append(f"{t:32s} mr {mrm:7.2f} s  contact {contact}/{len(mr)} unfin {unfinished}  "
                 f"gf {best:7.2f} ({best_cfg})  margin {100 * margin:+6.1f}%  build {b.get('choice')} {b.get('ok')}")
print("\n".join(lines))
if margins:
    print(f"\n{len(margins)} tracks compared: median margin {100 * statistics.median(margins):+.1f}%, "
          f"mean {100 * statistics.mean(margins):+.1f}%, >=5% on {wins}, worst {100 * min(margins):+.1f}%, "
          f"map-racer runs with contact {total_contact}")
