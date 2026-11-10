#!/usr/bin/env python3
"""Parallel steering-law comparison: the same runs and the same verdict as
``run_validation.py --compare``, spread over worker processes.

    # tune on the tuning seeds (never the judging ones)
    python3 tools/f1tenth_sim/law_sweep.py --laws stanley --seed 1000 \\
        --law-param stanley.gain=2.0
    # judge
    python3 tools/f1tenth_sim/law_sweep.py --laws stanley lqr mpc --seed 12345 \\
        --output docs/tracking-law-comparison.json

Every run is run_validation.run_pure_solo on the ``car`` fidelity profile;
the verdict is law_compare.verdict against pure pursuit on the same tracks x
seeds. Pure pursuit always runs at its shipped parameters.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import law_compare  # noqa: E402
import run_validation as rv  # noqa: E402
from sim_fidelity.calibration import (  # noqa: E402
    add_car_argument, calibration_for_car, car_from_args, selected_car)
from racerbot_car.layers import effective_config  # noqa: E402

TRACKS = ("Spielberg", "Silverstone", "BrandsHatch")


def _one(job):
    law, track, seed, params, timeout, car_id = job
    result = rv.run_pure_solo(track, seed, timeout, law_name=law,
                              law_params=params if law != "pure_pursuit" else {}, car_id=car_id)
    result["seed"] = seed
    result["raceline"] = rv.RACELINE_SOURCE
    return result


def _json_safe(value):
    """Spell non-finite floats as strings for the cache digest.

    Calibration uses math.inf for "no limit" (speed_loop_accel_fade_from /
    _no_load), which strict JSON cannot encode; 'inf' still hashes differently
    from any finite value, so the digest keeps telling calibrations apart.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def prepare_jobs(laws, tracks, seeds, params, timeout, car_id, cache_path=None):
    """Return reusable baseline rows, actual worker jobs, and cache identity.

    Car identity is separate from the digest: identical calibrations on two
    cars still must not make their results interchangeable. Hash the complete
    plant calibration and effective controller config so edits also invalidate
    a cache for the same car. Legacy row-only cache files are cache misses.
    """
    car_id = selected_car(car_id)
    values = {'plant': asdict(calibration_for_car(car_id)),
              'controller': effective_config('pure_pursuit.yaml', 'pure_pursuit_node', car_id)}
    digest = hashlib.sha256(json.dumps(
        _json_safe(values), sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')).hexdigest()
    key = {'format': 1, 'car': car_id, 'calibration': digest,
           'tracks': sorted(tracks), 'seeds': sorted(seeds), 'timeout': timeout,
           'raceline': rv.RACELINE_SOURCE, 'fidelity': rv.PROFILE.name}
    cached = []
    if cache_path is not None and Path(cache_path).is_file():
        try:
            document = json.loads(Path(cache_path).read_text())
        except (OSError, ValueError):
            document = None
        if isinstance(document, dict) and document.get('key') == key:
            rows = document.get('rows')
            wanted = {(track, seed) for track in tracks for seed in seeds}
            if (isinstance(rows, list) and len(rows) == len(wanted)
                    and all(isinstance(row, dict)
                            and isinstance(row.get('track'), str)
                            and type(row.get('seed')) is int
                            and row.get('steering_law') == 'pure_pursuit'
                            for row in rows)
                    and {(row.get('track'), row.get('seed')) for row in rows} == wanted):
                cached = rows
    run_laws = laws[1:] if cached else laws
    jobs = [(law, track, seed, params, timeout, car_id)
            for law in run_laws for track in tracks for seed in seeds]
    return cached, jobs, key


def write_baseline_cache(path, key, rows):
    Path(path).write_text(json.dumps({'key': key, 'rows': rows}, sort_keys=True) + '\n')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_car_argument(ap)
    ap.add_argument("--laws", nargs="+", required=True)
    ap.add_argument("--tracks", nargs="+", default=list(TRACKS))
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--repeat-seeds", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--law-param", action="append", default=[])
    ap.add_argument("--output", type=Path)
    ap.add_argument("--baseline-cache", type=Path,
                    help="reuse pure pursuit's rows from this JSON if it matches "
                         "car, calibration, controller config and tracks x seeds; "
                         "otherwise recompute it")
    args = ap.parse_args(argv)
    car_id = car_from_args(ap, args)

    params = rv._parse_law_params(args.law_param)
    seeds = list(range(args.seed, args.seed + args.repeat_seeds))
    laws = ["pure_pursuit"] + [law for law in args.laws if law != "pure_pursuit"]
    cached, jobs, cache_key = prepare_jobs(
        laws, args.tracks, seeds, params, args.timeout, car_id, args.baseline_cache)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = cached + list(pool.map(_one, jobs))
    if args.baseline_cache and not cached:
        write_baseline_cache(args.baseline_cache, cache_key,
                             [r for r in results if r["steering_law"] == "pure_pursuit"])

    rows = {law: [r for r in results if r["steering_law"] == law] for law in laws}
    print(f"seeds {seeds[0]}..{seeds[-1]}  params {params or '{}'}")
    for law in laws:
        for track in args.tracks:
            rs = [r for r in rows[law] if r["track"] == track]
            ok = [r for r in rs if r["passed"]]
            laps = [r["lap_time_s"] for r in ok if r["lap_time_s"]]
            print(f"  {law:<12} {track:<12} pass {len(ok)}/{len(rs)}  "
                  f"mean_ct {statistics.mean(r['mean_cross_track_m'] for r in rs):.4f}  "
                  f"max_ct {max(r['max_cross_track_m'] for r in rs):.3f}  "
                  f"lap {statistics.mean(laps) if laps else float('nan'):7.2f}  "
                  f"stops {sum(r['stop_steps'] for r in rs):5d}  "
                  f"avoid {sum(r['avoid_steps'] for r in rs):5d}  "
                  f"contact {sum(1 for r in rs if r['collision'])}  "
                  f"p99ms {max((r['law_compute_ms_p99'] or 0) for r in rs):.2f}")
    verdicts = {}
    for law in laws[1:]:
        v = law_compare.verdict(rows["pure_pursuit"], rows[law])
        verdicts[law] = v.verdict
        print(f"{law}: {v.verdict}  (cross-track {v.candidate_mean_cross_track:.4f} vs "
              f"{v.baseline_mean_cross_track:.4f}, lap {v.candidate_mean_lap_time:.2f} vs "
              f"{v.baseline_mean_lap_time:.2f})")
        for reason in v.reasons[:6]:
            print(f"    - {reason}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "schema_version": 2, "compare": True, "seed": args.seed,
            "seeds": seeds, "tracks": args.tracks, "raceline": rv.RACELINE_SOURCE,
            "fidelity_profile": rv.PROFILE.name, "laws": laws,
            "law_params": params, "verdicts": verdicts, "results": results,
        }, indent=2, sort_keys=True) + "\n")
    return 0 if all(v == "PROMOTE" for v in verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
