"""Read-only check linking the November 11 Car 2 rebuild to the gap-follow benchmark.

Run from workspace revision 76db7abe, where the benchmark was measured. PR #107
changed track_import.py afterwards, so on later revisions the regenerated raster
and centerline no longer match the tracked asset byte for byte; see
tools/run_analysis/results/2026-11-22-gap-follow-line-choice.md.
"""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "f1tenth_sim"))
from track_import import extract_track, load_map, write_gym_track


def verify(root):
    source = Path("data/rb2/racerbot_auto/20261111-151026-rebuilt")
    fixture = root / "src/pure_pursuit/test/data/course_20261111"
    maps = root / "tools/f1tenth_sim/maps"
    name = "car2_course_2026_11_11"
    spec = yaml.safe_load((root / source / "map.yaml").read_text())
    for filename in ("map.yaml", spec["image"]):
        if (root / source / filename).read_bytes() != (fixture / filename).read_bytes():
            raise ValueError(f"Source and benchmark fixture differ: {filename}")
    meta = json.loads((root / source / "rebuild.json").read_text())
    if Path(meta["run"]).name != "20261111-151026" or meta["car_id"] != "rb2":
        raise ValueError("Unexpected source mapping run")
    params = root / "data/rb2/racerbot_runs/20261111-152132/metadata/params"
    for node, key, filename in [
        ("map_server", "yaml_filename", "map.yaml"),
        ("pure_pursuit_node", "waypoints_file", "raceline_profiled.csv"),
    ]:
        recorded = yaml.safe_load((params / f"{node}.yaml").read_text())
        path = next(iter(recorded.values()))["ros__parameters"][key]
        if not path.endswith("/" + str(source / filename)):
            raise ValueError(f"Unexpected recorded {key}: {path}")
    track = extract_track(load_map(root / source / "map.yaml"))
    with tempfile.TemporaryDirectory() as tmp:
        write_gym_track(track, tmp, name)
        for suffix in ("_centerline.csv", "_map.yaml"):
            filename = name + suffix
            if (Path(tmp) / filename).read_bytes() != (maps / filename).read_bytes():
                raise ValueError(f"Regenerated {filename} differs")
        with Image.open(Path(tmp) / f"{name}.png") as a, Image.open(maps / f"{name}.png") as b:
            if not np.array_equal(np.asarray(a), np.asarray(b)):
                raise ValueError("Regenerated occupancy raster differs")
            print(f"Matched {a.width * a.height:,} cells and {len(track.centerline)} centerline points.")


if __name__ == "__main__":
    verify(ROOT)
