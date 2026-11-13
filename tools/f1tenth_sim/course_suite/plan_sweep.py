#!/usr/bin/env python3
"""The speed-plan sweep (post-race plan, Task 8's evidence): planner arms x courses x seeds, online cap at 3.1.

    rb python3 tools/f1tenth_sim/course_suite/plan_sweep.py /dev/shm/sweep --base /tmp/cmp/base \\
        --builder-src <checkout with profile_grip_coupling>/src/pure_pursuit --workers 4

Arms vary the planner's lateral grip (left = right), speed delay and grip coupling; every arm is raced by the
map racer (delay_lqr, rb2's race-day config, map safety, online lateral cap 3.1) on the floor chosen for rb2
(sfu_course_2026_11_11_rb2), 3 laps from a standing start, 3 seeds. Stages: build (plan_build.py in a subprocess
with the builder checkout on PYTHONPATH), race, extras (race-day course: --check-profile-alone, and the standing
start from the real failed launch pose with the cap on), report. Results accumulate in <out>/results.jsonl; a
finished stage is not repeated. At most --workers jobs, and none starts while the 1-minute load is 5 or more.
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tools" / "f1tenth_sim"), str(ROOT / "tools"), str(ROOT / "src" / "pure_pursuit")]

FLOOR = "sfu_course_2026_11_11_rb2"
COURSES = ["car2_course_2026_11_11", "car2_course_2026_11_06a_scans", "car2_course_2026_11_06b_scans",
           "course_2026_10_30_scans", "indoor_tight", "indoor_oval", "levine_blocked", "mini_Oschersleben"]
# name: (planner lateral left = right, planner speed delay, grip coupling)
ARMS = {"C0": (3.2, 0.22, 0.0), "P1": (2.95, 0.22, 0.0), "P2": (2.95, 0.22, 0.5), "P3": (2.95, 0.22, 1.0),
        "P4": (3.2, 0.22, 1.0), "P5": (2.95, 0.30, 1.0), "P6": (2.80, 0.22, 0.5)}
ONLINE_LATERAL = 3.1
# rb2's VESC-IMU gyro-basis table (grip_truth.py); below 2.7 the car delivers what is asked.
UNDERSTEER = dict(asked=[0.0, 2.7, 3.25, 3.75], delivered=[0.0, 2.7, 2.88, 2.94])
FAILED_POSE = (-16.5746, 10.6178, 285.151)     # race-day 152132, LB press: s 0.00, 0.134 m right, +14.61 deg
LAPS = 3


def wait_for_load(limit=5.0):
    while os.getloadavg()[0] >= limit:
        time.sleep(10)


def track_files(track):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run as suite
    return suite.track_files(track)


def build_job(arm, track, out, base, builder_src):
    lat, delay, coupling = ARMS[arm]
    rd = out / arm / track / "run"
    if (rd / "race_line.json").exists() or (rd / "build_failed.txt").exists():
        return None
    rd.mkdir(parents=True, exist_ok=True)
    for item in (base / track / "run").iterdir():
        if item.is_file():
            (rd / item.name).write_bytes(item.read_bytes())
    env = dict(os.environ, PYTHONPATH=f"{builder_src}:{os.environ.get('PYTHONPATH', '')}", RACERBOT_CAR="rb2")
    started = time.monotonic()
    with open(out / arm / track / "build.log", "w") as log:
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "plan_build.py"), "--run-dir", str(rd),
                             "--lat", str(lat), "--speed-delay", str(delay), "--coupling", str(coupling)],
                            stdout=log, stderr=subprocess.STDOUT, env=env, timeout=4 * 3600, cwd=ROOT).returncode
    info = {}
    try:
        info = json.load(open(rd / "race_line.json"))
    except Exception:
        pass
    reason = ""
    if (rd / "build_failed.txt").exists():
        reason = (rd / "build_failed.txt").read_text().strip()
    return dict(stage="build", arm=arm, track=track, ok=rc == 0, est=info.get("est_lap_s"), choice=info.get("choice"),
                reason=reason or info.get("reason"), seconds=round(time.monotonic() - started, 1))


def race_job(arm, track, seed, out, understeer=False):
    import numpy as np
    import start_replay as sr
    from pure_pursuit import racing_math
    spec, _, _, length = track_files(track)
    line = out / arm / track / "run" / "raceline_profiled.csv"
    xy, _ = racing_math.load_profiled_csv(str(line))
    start = sr.pose_on_line(xy, 0.0)
    timeout = max(60.0, 4.0 * length / 1.5)
    os.environ["RACERBOT_CAR"] = "rb2"
    cal, plan, trace, result = sr.trace_run(spec, FLOOR, str(line), start, seed, timeout, LAPS, 0.42, 5.0,
                                            ONLINE_LATERAL, "delay_lqr", understeer=UNDERSTEER if understeer else None)
    row = sr.summarise(cal, plan, trace, result, [])
    sr_ = row["seed_result"]
    return dict(stage="race", arm=arm + ("+U" if understeer else ""), track=track, seed=seed,
                contact=bool(sr_["contact"] > 0 or sr_["collision"]), unfinished=bool(sr_["laps"] < LAPS - 1e-6),
                laps_done=sr_["laps"], flying=row["laps_s"], cap_binding=result["cap_binding_ticks"],
                ticks=result["command_ticks"], reactive=int(result.get("avoid_steps", 0) + result.get("stop_steps", 0)
                                                           + result.get("escape_steps", 0)),
                end_arc=sr_["end_arc"])


def failed_pose_job(arm, out):
    """The standing start from the real failed launch pose with the cap on, two laps so that s = 14 is reached (one
    lap ends at the gym's start line, which the failed pose has already crossed: 9.6 m in)."""
    import start_replay as sr
    from pure_pursuit import racing_math
    os.environ["RACERBOT_CAR"] = "rb2"
    track = "car2_course_2026_11_11"
    line = out / arm / track / "run" / "raceline_profiled.csv"
    spec, *_ = track_files(track)
    failed = []
    for k in range(3):
        cal, plan, trace, result = sr.trace_run(spec, FLOOR, str(line), FAILED_POSE, 12345 + k, 60.0, 2, 0.42, 5.0,
                                                ONLINE_LATERAL, "delay_lqr")
        row = sr.summarise(cal, plan, trace, result, [14.0])
        at = row["at"].get(14.0)
        failed.append(dict(contact=bool(row["seed_result"]["contact"] > 0 or row["seed_result"]["collision"]),
                           right14=None if at is None else -at["offset_left"], end_arc=row["seed_result"]["end_arc"]))
    return dict(stage="failedpose", arm=arm, failed_pose=failed)


def extras_job(arm, out, understeer=False):
    import start_replay as sr
    from pure_pursuit import racing_math
    os.environ["RACERBOT_CAR"] = "rb2"
    track = "car2_course_2026_11_11"
    line = out / arm / track / "run" / "raceline_profiled.csv"
    xy, _ = racing_math.load_profiled_csv(str(line))
    spec, *_ = track_files(track)
    args = argparse.Namespace(track=spec, floor=FLOOR, line=str(line), seeds=3, first_seed=12345, seconds=90.0,
                              steering_limit=0.42, speed=5.0, law="delay_lqr", grip_scale=None, grip_coupling=None,
                              poses=5, check_laps=3)
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code, runs = sr.check_profile_alone(args, xy)
    worst_right = max(r["seed_result"]["max_right"] for r in runs)
    worst_left = max(r["seed_result"]["max_left"] for r in runs)
    contacts = [(r["label"], r["seed"], round(r["seed_result"]["end_arc"], 1)) for r in runs
                if r["seed_result"]["contact"] > 0 or r["seed_result"]["collision"]]
    failed = []
    for k in range(3):
        cal, plan, trace, result = sr.trace_run(spec, FLOOR, str(line), FAILED_POSE, 12345 + k, 60.0, 2, 0.42, 5.0,
                                                ONLINE_LATERAL, "delay_lqr")
        row = sr.summarise(cal, plan, trace, result, [14.0])
        at = row["at"].get(14.0)
        failed.append(dict(contact=bool(row["seed_result"]["contact"] > 0 or row["seed_result"]["collision"]),
                           right14=None if at is None else -at["offset_left"], end_arc=row["seed_result"]["end_arc"]))
    return dict(stage="extras", arm=arm, alone_pass=(code == 0), alone_worst_right=worst_right,
                alone_worst_left=worst_left, alone_contacts=contacts, failed_pose=failed)


def load_results(out):
    path = out / "results.jsonl"
    return [json.loads(line) for line in open(path)] if path.exists() else []


def record(out, row):
    with open(out / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row)[:260], flush=True)


def report(out, arms, courses):
    rows = load_results(out)
    build = {(r["arm"], r["track"]): r for r in rows if r["stage"] == "build"}
    races = {}
    for r in rows:
        if r["stage"] == "race":
            races.setdefault((r["arm"], r["track"]), []).append(r)
    extras = {r["arm"]: r for r in rows if r["stage"] == "extras"}
    lines = ["| arm | course | line | est lap s | median flying lap s | contact seeds | unfinished seeds | online cap binding | reactive ticks |",
             "|---|---|---|---|---|---|---|---|---|"]
    ratios = {}
    medians = {}
    for arm in arms:
        for track in courses:
            b = build.get((arm.split("+")[0], track))
            rs = races.get((arm, track), [])
            if b is None:
                lines.append(f"| {arm} | {track} | not built | | | | | | |")
                continue
            if not b["ok"]:
                lines.append(f"| {arm} | {track} | NO: {str(b['reason'])[:70]} | | | | | | |")
                continue
            fly = [x for r in rs for x in r["flying"]]
            med = statistics.median(fly) if fly else math.nan
            medians[(arm, track)] = med
            ticks = sum(r["ticks"] for r in rs) or 1
            lines.append(f"| {arm} | {track} | {b['choice']} | {b['est']:.2f} | {med:.2f} ({len(fly)}) | "
                         f"{sum(r['contact'] for r in rs)}/{len(rs)} | {sum(r['unfinished'] for r in rs)}/{len(rs)} | "
                         f"{sum(r['cap_binding'] for r in rs) / ticks:.3f} | {sum(r['reactive'] for r in rs)} |")
    return "\n".join(lines), build, races, extras, medians


def ranking(out, arms, courses):
    """(markdown, eligible ranking) by the stated rule: an arm is eligible when every course that builds under C0
    builds under it, there is no contact and no unfinished seed anywhere, and it passes --check-profile-alone on
    the race-day course; eligible arms are ranked by the mean over courses of median flying lap / C0's."""
    _, build, races, extras, medians = report(out, arms, courses)
    c0_built = [t for t in courses if build.get(("C0", t), {}).get("ok")]
    rows, eligible = [], []
    for arm in arms:
        base_arm = arm.split("+")[0]
        builds_all = all(build.get((base_arm, t), {}).get("ok") for t in c0_built)
        runs = [r for t in courses for r in races.get((arm, t), [])]
        contact = sum(r["contact"] for r in runs)
        unfinished = sum(r["unfinished"] for r in runs)
        alone = extras.get(base_arm, {}).get("alone_pass")
        ratios = [medians[(arm, t)] / medians[("C0", t)] for t in c0_built
                  if (arm, t) in medians and ("C0", t) in medians]
        mean_ratio = sum(ratios) / len(ratios) if ratios else math.nan
        ok = builds_all and contact == 0 and unfinished == 0 and bool(alone) and len(ratios) == len(c0_built)
        failed = [name for name, bad in (("a course C0 builds did not build", not builds_all),
                                         (f"{contact} contact run(s)", contact > 0),
                                         (f"{unfinished} unfinished run(s)", unfinished > 0),
                                         ("profile-alone check not passed", not alone)) if bad]
        rows.append((arm, ok, mean_ratio, failed))
        if ok:
            eligible.append((mean_ratio, arm))
    eligible.sort()
    lines = ["| arm | eligible | mean flying-lap ratio to C0 | failed conditions |", "|---|---|---|---|"]
    for arm, ok, ratio, failed in rows:
        lines.append(f"| {arm} | {'yes' if ok else 'no'} | {ratio:.4f} | {'; '.join(failed) or '-'} |")
    lines.append("")
    lines.append("Ranking (eligible only, fastest first): " + (", ".join(f"{a} ({r:.4f})" for r, a in eligible) or "none eligible"))
    return "\n".join(lines), eligible


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--base", required=True, help="a course-suite output dir holding <course>/run/ with the raw lines")
    ap.add_argument("--builder-src", required=True, help="src/pure_pursuit of a checkout with profile_grip_coupling")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--courses", default=",".join(COURSES))
    ap.add_argument("--stage", default="build,race,extras,report")
    ap.add_argument("--understeer-arms", default="", help="comma list of arms to also race with the understeer curve on")
    args = ap.parse_args(argv)
    out, base = Path(args.out).resolve(), Path(args.base).resolve()
    out.mkdir(parents=True, exist_ok=True)
    arms, courses = args.arms.split(","), args.courses.split(",")
    stages = args.stage.split(",")
    done = load_results(out)
    built = {(r["arm"], r["track"]) for r in done if r["stage"] == "build"}
    raced = {(r["arm"], r["track"], r["seed"]) for r in done if r["stage"] == "race"}
    extra_done = {r["arm"] for r in done if r["stage"] == "extras"}
    understeer = [a for a in args.understeer_arms.split(",") if a]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = set()
        if "build" in stages:
            for arm in arms:
                for track in courses:
                    if (arm, track) not in built:
                        wait_for_load()
                        futures.add(pool.submit(build_job, arm, track, out, base, args.builder_src))
            for fut in as_completed(futures):
                row = fut.result()
                if row:
                    record(out, row)
        built = {(r["arm"], r["track"]): r for r in load_results(out) if r["stage"] == "build"}
        futures = set()
        if "race" in stages:
            for arm in arms + [a + "+U" for a in understeer]:
                base_arm = arm.split("+")[0]
                for track in courses:
                    if not built.get((base_arm, track), {}).get("ok"):
                        continue
                    for k in range(args.seeds):
                        if (arm, track, 12345 + k) not in raced:
                            wait_for_load()
                            futures.add(pool.submit(race_job, base_arm, track, 12345 + k, out, arm.endswith("+U")))
            for fut in as_completed(futures):
                record(out, fut.result())
        futures = set()
        if "extras" in stages:
            for arm in arms:
                if arm not in extra_done and built.get((arm, "car2_course_2026_11_11"), {}).get("ok"):
                    wait_for_load()
                    futures.add(pool.submit(extras_job, arm, out))
            for fut in as_completed(futures):
                record(out, fut.result())
    if "report" in stages:
        every = arms + [a + "+U" for a in understeer]
        table, *_ = report(out, every, courses)
        ranks, _ = ranking(out, every, courses)
        (out / "table.md").write_text(table + "\n\n" + ranks + "\n")
        print(table)
        print(ranks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
