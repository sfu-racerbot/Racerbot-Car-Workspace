"""Course suite: map racer vs gap_follow on every track. Run from the workspace root (see README.md).

    python3 suite_run.py <out dir> [--tracks a,b] [--workers N] [--seeds K] [--stage all|raw|build|race|gap]
        [--lead S] [--online-lat A] [--tag NAME]

Per track: (raw) two gap_follow laps at mapping settings, lap 2 -> run_dir/raceline_raw.csv (centreline if
gap_follow cannot lap); (build) python -m pure_pursuit.race_line on that run dir; (race) the map racer on
the built line, K seeds x 3 laps; (gap) gap_follow default / time trial / time trial 3.5 and 4.0 m/s,
K seeds. Results: <out>/<track>/*.json, summarised by suite_table.py. All on car 2's 6 Nov floor.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("RACERBOT_CAR", "rb2")
WS = Path(os.environ.get("WS", Path.cwd()))
SUITE = Path(os.environ.get("SUITE_TRACKS", Path(__file__).resolve().parents[3] / ".sim" / "course_suite" / "tracks"))
MR = Path(__file__).resolve().parent
FLOOR = "sfu_course_2026_11_06"
OURS = ["car2_course_2026_11_06a_scans", "car2_course_2026_11_06b_scans", "course_2026_10_30_scans",
        "car1_course_2026_10_26", "room_2026_07_27", "levine_obs", "levine_blocked", "circular_obs"]
INDOOR = ["indoor_oval", "indoor_tight", "indoor_wide", "asb_10000", "asb_sharp", "narrow_1p2", "narrow_1p0"]
GF_CONFIGS = {
    "default": [],
    "tt": ["--set", "aim_depth_bias=1.0", "--set", "enable_opponent_awareness=false"],
    "tt35": ["--set", "aim_depth_bias=1.0", "--set", "enable_opponent_awareness=false", "--set", "max_speed=3.5"],
    "tt40": ["--set", "aim_depth_bias=1.0", "--set", "enable_opponent_awareness=false", "--set", "max_speed=4.0"],
}


def py():
    return [sys.executable]


def track_files(track):
    """(spec for the simulators, map yaml, centreline csv, length m)."""
    maps = WS / "tools" / "f1tenth_sim" / "maps"
    if track in OURS:
        spec, yml, cl = track, maps / f"{track}_map.yaml", maps / f"{track}_centerline.csv"
    elif track in INDOOR:
        sys.path[:0] = [str(WS / "tools" / "f1tenth_sim"), str(WS / "tools")]
        import eval_gap
        stem = Path(eval_gap.track_path(track))
        spec, yml, cl = track, stem.parent / f"{stem.name}_map.yaml", stem.parent / f"{stem.name}_centerline.csv"
    else:
        stem = SUITE / track / track
        spec, yml, cl = str(stem), stem.parent / f"{track}_map.yaml", stem.parent / f"{track}_centerline.csv"
    xy = load_xy(cl)
    import numpy as np
    length = float(np.sum(np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)))
    return spec, yml, cl, length


def load_xy(path):
    """x, y of a CSV whose header (if any) is '#'-commented or plain text: rows that do not parse are skipped."""
    import numpy as np
    pts = []
    for line in open(path):
        parts = line.replace("#", "").split(",")
        try:
            pts.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            continue
    return np.array(pts)


def run(cmd, log, timeout):
    with open(log, "w") as f:
        try:
            return subprocess.run(cmd, cwd=WS, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write("\nTIMEOUT\n")
            return -9


def stage_raw(track, out):
    spec, yml, cl, length = track_files(track)
    rd = out / track / "run"
    rd.mkdir(parents=True, exist_ok=True)
    meta = __import__("yaml").safe_load(open(yml))
    img = Path(yml).parent / meta["image"]
    shutil.copy(img, rd / img.name)
    shutil.copy(yml, rd / "map.yaml")
    trace = out / track / "gf_mapping_trace.csv"
    timeout_sim = max(60.0, 3.0 * length / 1.5)
    rc = run(py() + ["tools/f1tenth_sim/trace_gap.py", "--car", "rb2", "--track", spec, "--floor", FLOOR,
                     "--laps", "2", "--timeout", f"{timeout_sim:.0f}", "--set", "enable_opponent_awareness=false",
                     "--out", str(trace)], out / track / "raw.log", 3600)
    import csv
    import numpy as np
    source = "centerline"
    try:
        rows = list(csv.DictReader(open(trace)))
        xy = np.array([[float(r["x"]), float(r["y"])] for r in rows])
        st = xy[0]; far = False; cuts = []
        for k, p in enumerate(xy):
            d = float(np.hypot(*(p - st))); far = far or d > 3.0
            if far and d < 0.6:
                cuts.append(k); far = False
        if len(cuts) >= 2 and not any(r["contact"] == "1" for r in rows):
            lap = xy[cuts[0]:cuts[1]]
            source = "gap_follow lap 2"
        else:
            raise ValueError("gap_follow did not lap twice cleanly")
    except Exception:
        lap = load_xy(cl)
    with open(rd / "raceline_raw.csv", "w") as f:
        f.write("x,y\n")
        for x, y in lap:
            f.write(f"{x:.4f},{y:.4f}\n")
    return dict(track=track, stage="raw", source=source, rc=rc, length_m=round(length, 1))


def stage_build(track, out):
    rd = out / track / "run"
    vmax = os.environ.get("SUITE_VMAX")
    builder = [os.environ["SUITE_BUILDER"]] if os.environ.get("SUITE_BUILDER") else ["-m", "pure_pursuit.race_line"]
    rc = run(py() + builder + ["--run-dir", str(rd), "--car", "rb2"]
             + (["--v-max", vmax] if vmax else []), out / track / "build.log", 4 * 3600)
    info = {}
    try:
        info = json.load(open(rd / "race_line.json"))
    except Exception:
        pass
    return dict(track=track, stage="build", rc=rc, ok=info.get("ok"), choice=info.get("choice"),
                est=info.get("est_lap_s"), reason=info.get("reason"))


def stage_race(track, out, seed, lead, online_lat, tag):
    spec, _, _, length = track_files(track)
    line = out / track / "run" / "raceline_profiled.csv"
    if not line.exists():
        return dict(track=track, stage="race", tag=tag, seed=seed, error="no line")
    timeout = max(60.0, 4.0 * length / 1.5)
    env = dict(os.environ, LEAD=str(lead))
    vmax = os.environ.get("SUITE_VMAX")
    if vmax:
        env["VMAX"] = vmax
    log = out / track / f"mr_{tag}_{seed}.log"
    with open(log, "w") as f:
        subprocess.run(py() + [str(MR / "race.py"), "--track", spec, "--floor", FLOOR, "--line", str(line),
                               "--profiled", "--safety", "map", "--laps", "3", "--steering-limit", "0.42",
                               "--speed", os.environ.get("SUITE_VMAX", "5.0"), "--lateral", str(online_lat), "--timeout", f"{timeout:.0f}",
                               "--law", "delay_lqr", "--seed", str(seed)],
                       cwd=WS, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=4 * 3600)
    text = log.read_text()
    m = re.search(r"^top .*$", text, re.M)
    if not m:
        return dict(track=track, stage="race", tag=tag, seed=seed, error=text[-300:])
    t = m.group(0)
    laps = json.loads(re.search(r"lap times (\[.*\])", t).group(1))
    return dict(track=track, stage="race", tag=tag, seed=seed, laps=laps,
                contact=int(re.search(r"contact (\d+)", t).group(1)),
                stop=int(re.search(r" stop (\d+)", t).group(1)),
                finished=re.search(r"finished (\w+)", t).group(1) == "True")


def stage_gap(track, out, cfg, seeds):
    spec, _, cl, length = track_files(track)
    seconds = max(90.0, 4.0 * length / 1.5)
    params = Path(os.environ.get("SUITE_GF_PARAMS", WS / ".sim" / "course_suite" / "gf_rb2"))
    if not (params / "metadata" / "params" / "gap_follow_node.yaml").exists():
        return dict(track=track, stage="gap", cfg=cfg,
                    error=f"no {params}: run tools/f1tenth_sim/course_suite/gf_params.py {params} rb2 first")
    res = out / track / f"gf_{cfg}.json"
    rc = run(py() + ["tools/f1tenth_sim/validate_calibration.py", "sim", "--params-from", str(params), "--launched-code",
                     "--floor", FLOOR, "--track", spec, "--line", str(cl), "--seeds", str(seeds), "--workers", "1",
                     "--seconds", f"{seconds:.0f}"] + GF_CONFIGS[cfg] + ["--out", str(res)],
             out / track / f"gf_{cfg}.log", 4 * 3600)
    try:
        d = json.load(open(res))
        return dict(track=track, stage="gap", cfg=cfg, flying=d["flying"],
                    seeds=[dict(laps=s["laps"], contact=s["contact"], wedged=s["wedged"]) for s in d["per_seed"]])
    except Exception as exc:
        return dict(track=track, stage="gap", cfg=cfg, error=f"rc {rc}: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tracks", default="")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--lead", type=float, default=0.25)
    ap.add_argument("--online-lat", type=float, default=3.3)
    ap.add_argument("--tag", default="mr")
    ap.add_argument("--gap-configs", default="", help="comma list: only these gap_follow configurations")
    ap.add_argument("--vmax", type=float, default=None,
                    help="raised top speed: builder --v-max, the plant's VESC cap, and a gap_follow time trial at it")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if a.vmax:
        v = f"{a.vmax:g}"
        os.environ["SUITE_VMAX"] = v
        GF_CONFIGS["ttv"] = GF_CONFIGS["tt"] + ["--set", f"max_speed={v}", "--cal", f"speed_max={v}",
                                                "--cal", f"speed_min=-{v}"]
    out.mkdir(parents=True, exist_ok=True)
    suite = sorted(p.name for p in SUITE.iterdir() if p.is_dir()) if SUITE.exists() else []
    tracks = a.tracks.split(",") if a.tracks else OURS + INDOOR + suite
    for t in tracks:
        (out / t).mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"

    def record(fut):
        try:
            r = fut.result()
        except Exception as exc:  # one bad track must not end the run
            r = dict(stage="error", error=f"{type(exc).__name__}: {exc}")
        with open(results, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps(r)[:240], flush=True)
        return r

    def length(t):
        try:
            return track_files(t)[3]
        except Exception:
            return math.inf

    tracks = sorted(tracks, key=length)          # short courses first: their verdicts arrive early
    stages = ["raw", "build", "race", "gap"] if a.stage == "all" else a.stage.split(",")
    from concurrent.futures import FIRST_COMPLETED, wait
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        if "raw" in stages:
            for f in as_completed([pool.submit(stage_raw, t, out) for t in tracks
                                   if not (out / t / "run" / "raceline_raw.csv").exists()]):
                record(f)

        def races(t):
            return [pool.submit(stage_race, t, out, 12345 + k, a.lead, a.online_lat, a.tag) for k in range(a.seeds)] \
                if "race" in stages else []
        pending = set()
        for t in tracks:
            # A course's races start as soon as its own line exists (built now or by an earlier run).
            if "build" in stages and not (out / t / "run" / "race_line.json").exists():
                pending.add(pool.submit(stage_build, t, out))
            else:
                pending.update(races(t))
        if "gap" in stages:
            cfgs = a.gap_configs.split(",") if a.gap_configs else list(GF_CONFIGS)
            pending.update(pool.submit(stage_gap, t, out, c, a.seeds) for t in tracks for c in cfgs)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                r = record(f)
                if r.get("stage") == "build" and r.get("ok"):
                    pending.update(races(r["track"]))


if __name__ == "__main__":
    main()
