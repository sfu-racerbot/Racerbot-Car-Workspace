#!/usr/bin/env python3
"""Open an offline comparison of real SFU Racerbot recordings (Python 3, NumPy, PyYAML).

    python3 tools/run_analysis/run_visualizer.py
    python3 tools/run_analysis/run_visualizer.py RUN_DIR [RUN_DIR ...] -o comparison.html

With no paths, includes the workspace's cached runs. Accepts a recorder directory
with analysis/run.npz (runlog format) or bag/*.mcap. Uncached MCAP needs
`python3 -m pip install mcap-ros2-support==0.5.7` in your Python environment.
No ROS, server or network is needed to open the generated HTML.
"""
import argparse
import json
import sys
import tempfile
import webbrowser
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
TOPICS = ("/odom", "/drive", "/ackermann_cmd", "/joy")


def cached_streams(cache):
    # Read just the small arrays; the unused scan matrix can be much larger.
    with np.load(cache, allow_pickle=False) as z:
        arrays = {}
        for topic, key, cols in [
            ("/odom", "odom", [0, 3, 4, 1]),
            ("/drive", "drive", [0, 3, 2]),
            ("/ackermann_cmd", "ackermann", [0, 3, 2]),
            ("/joy", "joy", [0, 1]),
        ]:
            if key in z:
                arr = z[key]
                if arr.ndim != 2 or arr.shape[1] <= max(cols):
                    raise ValueError(f"invalid {key} columns in {cache.name}")
                arrays[topic] = arr[:, cols]
        return arrays


def mcap_streams(files):
    try:
        from mcap.exceptions import McapError
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory
    except ImportError as exc:
        raise ValueError("uncached bag: install mcap-ros2-support==0.5.7") from exc

    arrays = {topic: [] for topic in TOPICS}
    for path in files:
        try:
            with path.open("rb") as stream:
                reader = make_reader(stream, decoder_factories=[DecoderFactory()])
                for _, channel, record, msg in reader.iter_decoded_messages(topics=TOPICS):
                    topic, t = channel.topic, record.log_time / 1e9
                    if topic == "/odom":
                        pos = msg.pose.pose.position
                        row = [t, pos.x, pos.y, msg.twist.twist.linear.x]
                    elif topic == "/joy":
                        row = [t, msg.buttons[4] if len(msg.buttons) > 4 else 0]
                    else:
                        row = [t, msg.drive.speed, msg.drive.steering_angle]
                    arrays[topic].append(row)
        except (McapError, EOFError, AttributeError) as exc:
            raise ValueError(f"could not read {path.name}: {exc}") from exc
    return arrays


def prepare(streams):
    """Use receive times, not header stamps. Never resample one topic onto another."""
    rows = {}
    for topic, values in streams.items():
        if len(values):
            a = np.array(values, dtype=float, copy=True)
            if not np.isfinite(a).all():
                raise ValueError(f"non-finite sample on {topic}")
            rows[topic] = a[np.argsort(a[:, 0], kind="stable")]
    if "/odom" not in rows:
        raise ValueError("no /odom samples for a trajectory")
    cmd_topic = next((t for t in ("/drive", "/ackermann_cmd") if t in rows), None)
    odom = rows["/odom"]
    cmd = rows.get(cmd_topic, np.empty((0, 3)))
    joy = rows.get("/joy", np.empty((0, 2)))
    used = [a for a in (odom, cmd, joy) if len(a)]
    start = min(a[0, 0] for a in used)
    for a in used:
        a[:, 0] -= start
    duration = max(a[-1, 0] for a in used)
    odom[:, 1:3] -= odom[0, 1:3].copy()  # origins differ between recordings
    cmd[:, 2] = np.degrees(cmd[:, 2])
    return dict(duration=round(duration, 5), distance=np.linalg.norm(
        np.diff(odom[:, 1:3], axis=0), axis=1).sum(), peak=np.abs(odom[:, 3]).max(),
        odom=odom.round(5).tolist(), cmd=cmd.round(5).tolist(),
        joy=joy.round(5).tolist(), cmdTopic=cmd_topic)


def load(path):
    path = Path(path).expanduser().resolve()
    cache = path / "analysis/run.npz"
    if cache.is_file():
        data = prepare(cached_streams(cache))
        source = "NPZ"
    else:
        files = sorted((path / "bag").glob("*.mcap"))
        if not files:
            raise ValueError("expected a run directory with analysis/run.npz or bag/*.mcap")
        data = prepare(mcap_streams(files))
        source = "MCAP"
    algorithms = []
    started_late = False
    metadata = path / "auto_record.yaml"
    if metadata.is_file():
        meta = yaml.safe_load(metadata.read_text(encoding="utf-8"))
        if isinstance(meta, dict) and isinstance(meta.get("activity"), dict):
            started_late = meta["activity"].get("started_late") is True
        if isinstance(meta, dict) and isinstance(meta.get("algorithms"), dict):
            nodes = meta["algorithms"].get(data["cmdTopic"])
            if isinstance(nodes, list):
                algorithms = [n for n in nodes if isinstance(n, str)]
    car = path.parent.parent.name if path.parent.name == "racerbot_runs" else ""
    return dict(data, car=car, name=path.name, source=source, algorithms=algorithms,
                startedLate=started_late)


def render(runs):
    assets = HERE / "visualizer"
    payload = json.dumps(runs, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return ((assets / "index.html").read_text(encoding="utf-8")
            .replace("/* STYLES */", (assets / "style.css").read_text(encoding="utf-8"))
            .replace("/* COMMANDS */", (assets / "commands.js").read_text(encoding="utf-8"))
            .replace("/* PLAYER */", (assets / "player.js").read_text(encoding="utf-8"))
            .replace("<!-- RUN_DATA -->", payload))


def default_runs():
    paths = sorted(p.parent.parent for p in DATA.glob("*/racerbot_runs/*/analysis/run.npz"))
    # Put one recording from each car in the initial comparison.
    by_car = {}
    for path in paths:
        by_car.setdefault(path.parents[1], path)
    first = list(by_car.values())
    return first + [p for p in paths if p not in first]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="one or more recorder directories")
    parser.add_argument("-o", "--output", type=Path, help="write a new HTML file instead of opening a browser")
    args = parser.parse_args()
    paths = args.runs or default_runs()
    if not paths:
        parser.error("no cached runs found; pass a recorder directory")
    runs = []
    for path in paths:
        try:
            runs.append(load(path))
        except (ValueError, OSError, BadZipFile, yaml.YAMLError) as exc:
            parser.exit(1, f"{path}: {exc}\n")
    html = render(runs)
    output = args.output or Path(tempfile.mkdtemp(prefix="racerbot-")) / "comparison.html"
    try:
        with output.open("x", encoding="utf-8") as out:
            out.write(html)
    except OSError as exc:
        parser.exit(1, f"Could not write report: {exc}\n")
    print(f"{len(runs)} recording(s): {output.resolve().as_uri()}", flush=True)
    if not args.output:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
