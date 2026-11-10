"""Build the test-course suite (in .sim/course_suite/, never committed): F1TENTH racetracks as-is, plus indoor-sized versions.

    python3 gen_suite.py <f1tenth_racetracks dir> <out dir>

Writes <out>/<name>/<name>_map.yaml, <name>.png and <name>_centerline.csv (x_m, y_m, w_right, w_left) per
track, the triplet the simulator loads from a path (Track.from_track_path), and <out>/index.json.

f1t_<Track>: the published 1:10 map, renamed (no spaces).
mini_<Track>: the same circuit's centreline scaled to TARGET_M of lap (grown to MAX_M if two parts of
the circuit would come closer than the corridor plus a wall), smoothed until no bend is tighter than
MIN_RADIUS_M (the car turns 0.86 m at 0.42 rad; the planner's 0.38 rad gives 0.90 m), WIDTH_M wall to wall.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d

SRC, OUT = None, None
TARGET_M, MAX_M = 80.0, 200.0
WIDTH_M = 2.0
MIN_RADIUS_M = 1.3
RES = 0.05
WALL_M = 0.15            # occupied band drawn round the corridor (the rest of the image is occupied too)


def resample(xy, step):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1))]
    s = np.arange(0, d[-1], step)
    closed = np.vstack([xy, xy[:1]])
    return np.stack([np.interp(s, d, closed[:, 0]), np.interp(s, d, closed[:, 1])], 1)


def curvature(xy):
    p, n = np.roll(xy, 1, 0), np.roll(xy, -1, 0)
    a, b, c = np.linalg.norm(xy - p, axis=1), np.linalg.norm(n - xy, axis=1), np.linalg.norm(n - p, axis=1)
    cross = (xy[:, 0] - p[:, 0]) * (n[:, 1] - p[:, 1]) - (xy[:, 1] - p[:, 1]) * (n[:, 0] - p[:, 0])
    return 2 * cross / np.maximum(a * b * c, 1e-12)


def too_close(xy, min_gap, skip_m):
    """True if two parts of the loop more than skip_m apart along it come within min_gap."""
    step = 0.25
    pts = resample(xy, step)
    n = len(pts)
    k = int(skip_m / step)
    for i in range(n):
        d = np.linalg.norm(pts - pts[i], axis=1)
        idx = np.arange(n)
        sep = np.minimum(np.abs(idx - i), n - np.abs(idx - i))
        if np.any((sep > k) & (d < min_gap)):
            return True
    return False


def write_track(name, xy, half_width):
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    lo = xy.min(0) - half_width - 1.0
    hi = xy.max(0) + half_width + 1.0
    w, h = (np.ceil((hi - lo) / RES)).astype(int)
    dense = resample(xy, RES / 2)
    mask = np.zeros((h, w), bool)
    ix = ((dense[:, 0] - lo[0]) / RES).astype(int)
    iy = (h - 1 - (dense[:, 1] - lo[1]) / RES).astype(int)
    mask[iy.clip(0, h - 1), ix.clip(0, w - 1)] = True
    dist = ndimage.distance_transform_edt(~mask) * RES
    img = np.where(dist <= half_width, 254, 0).astype(np.uint8)
    Image.fromarray(img).save(folder / f"{name}.png")
    (folder / f"{name}_map.yaml").write_text(yaml.safe_dump(dict(
        image=f"{name}.png", resolution=RES, origin=[float(lo[0]), float(lo[1]), 0.0],
        negate=0, occupied_thresh=0.65, free_thresh=0.196)))
    with open(folder / f"{name}_centerline.csv", "w") as f:
        f.write("# x_m, y_m, w_tr_right_m, w_tr_left_m\n")
        for x, y in resample(xy, 0.1):
            f.write(f"{x:.4f}, {y:.4f}, {half_width:.3f}, {half_width:.3f}\n")
    length = float(np.sum(np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)))
    return dict(name=name, path=str(folder / name), length_m=round(length, 1), width_m=round(2 * half_width, 2))


def load_centerline(path):
    rows = [l for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    arr = np.array([[float(v) for v in r.split(",")[:4]] for r in rows])
    return arr[:, :2], arr[:, 2], arr[:, 3]




def import_f1tenth(src_dir, out_dir):
    global SRC, OUT
    SRC, OUT = Path(src_dir), Path(out_dir)
    index = []

    OUT.mkdir(parents=True, exist_ok=True)
    for d in sorted(p for p in SRC.iterdir() if p.is_dir() and not p.name.startswith(".")):
        key = d.name.replace(" ", "")
        cl = next(iter(d.glob("*_centerline.csv")), None)
        my = next(iter(d.glob("*_map.yaml")), None)
        if cl is None or my is None:
            print("skip (files)", d.name)
            continue
        # 1:10 as published
        name = f"f1t_{key}"
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        meta = yaml.safe_load(my.read_text())
        img_src = d / meta["image"]
        shutil.copy(img_src, folder / f"{name}{img_src.suffix}")
        meta["image"] = f"{name}{img_src.suffix}"
        (folder / f"{name}_map.yaml").write_text(yaml.safe_dump(meta))
        shutil.copy(cl, folder / f"{name}_centerline.csv")
        xy, wr, wl = load_centerline(cl)
        length = float(np.sum(np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)))
        index.append(dict(name=name, path=str(folder / name), length_m=round(length, 1),
                          width_m=round(float(np.median(wr + wl)), 2), kind="f1tenth_1to10"))
        # indoor-sized version
        made = False
        for target in np.arange(TARGET_M, MAX_M + 1, 20.0):
            base = resample(xy * (target / length), 0.1)
            small, sigma = base, 0.0
            # blur along the lap, sigma in metres, until no bend is tighter than MIN_RADIUS_M
            while np.abs(curvature(resample(small, 0.25))).max() > 1.0 / MIN_RADIUS_M and sigma < 4.0:
                sigma += 0.25
                small = resample(np.stack([gaussian_filter1d(base[:, 0], sigma / 0.1, mode="wrap"),
                                           gaussian_filter1d(base[:, 1], sigma / 0.1, mode="wrap")], 1), 0.1)
            if np.abs(curvature(resample(small, 0.25))).max() > 1.0 / MIN_RADIUS_M:
                continue
            if too_close(small, WIDTH_M + 2 * WALL_M + 0.3, skip_m=3 * WIDTH_M):
                continue
            entry = write_track(f"mini_{key}", small, WIDTH_M / 2)
            entry.update(kind="mini_f1", smoothing=sigma)
            index.append(entry)
            made = True
            break
        print(f"{key:15s} 1:10 {length:6.1f} m   mini {'%5.1f m (smooth %g)' % (entry['length_m'], sigma) if made else 'skipped'}")
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(len(index), "tracks")
    return index


def random_tracks(out_dir, count=8):
    global OUT
    OUT = Path(out_dir)
    index_path = OUT / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    index = [e for e in index if not e["name"].startswith("rand_")]

    made, seed = 0, 0
    while made < count and seed < 500:
        rng = np.random.default_rng(1000 + seed); seed += 1
        t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        r = np.ones_like(t)
        for k in range(2, 6):
            r += rng.uniform(0, 0.35 / k) * np.cos(k * t + rng.uniform(0, 2 * np.pi))
        stretch = rng.uniform(1.0, 2.2)
        xy = np.stack([r * np.cos(t) * stretch, r * np.sin(t)], 1)
        length0 = np.sum(np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1))
        target = rng.uniform(25, 60)
        xy = resample(xy * target / length0, 0.1)
        width = rng.uniform(1.4, 2.4)
        if np.abs(curvature(resample(xy, 0.25))).max() > 1 / 1.3:
            continue
        if too_close(xy, width + 2 * 0.15 + 0.3, skip_m=3 * width):
            continue
        entry = write_track(f"rand_{made}", xy, width / 2)
        entry.update(kind="random_indoor", seed=1000 + seed - 1)
        index.append(entry); made += 1
        print(entry)
    index_path.write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    import subprocess
    root = Path(__file__).resolve().parents[3] / ".sim" / "course_suite"
    src = root / "src" / "f1tenth_racetracks"
    if not src.exists():
        # GPL-3.0 maps (github.com/f1tenth/f1tenth_racetracks): downloaded, never committed.
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/f1tenth/f1tenth_racetracks", str(src)],
                       check=True)
    import_f1tenth(src, root / "tracks")
    random_tracks(root / "tracks", 8)
