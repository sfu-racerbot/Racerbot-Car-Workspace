"""Offline report: clocks, units and topic provenance must survive export."""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_analysis import run_visualizer as viz


def test_independent_receive_times_and_full_resolution_metrics():
    # 3-4-5 displacement then 3 m in reverse; commands start before odometry.
    streams = {"/odom": [[14, 5, 7, -2], [11, 2, 3, 5], [16, 2, 7, -6]],
               "/drive": [[10, -1, -math.pi / 4], [12, 2, math.pi / 6]],
               "/ackermann_cmd": [[8, 4, 0]], "/joy": [[12, 1], [13, 1], [15, 0], [17, 0]]}
    r = viz.prepare(streams)
    assert (r["distance"], r["peak"], r["duration"]) == (8, 6, 7)
    assert r["odom"] == [[1, 0, 0, 5], [4, 3, 4, -2], [6, 0, 4, -6]]
    assert r["cmdTopic"] == "/drive"
    assert r["cmd"] == [[0, -1, -45], [2, 2, 30]]
    assert r["joy"] == [[2, 1], [5, 0], [7, 0]]


def test_single_pose_optional_topics_and_manual_fallback():
    streams = {"/odom": [[50, 4, 5, 0]]}
    r = viz.prepare(streams)
    assert [r[k] for k in ("distance", "duration", "odom", "cmdTopic", "joy")] == [
        0, 0, [[0, 0, 0, 0]], None, []]
    r = viz.prepare(dict(streams, **{"/ackermann_cmd": [[51, 2, -math.pi / 2]]}))
    assert (r["cmdTopic"], r["cmd"]) == ("/ackermann_cmd", [[1, 2, -90]])


@pytest.mark.parametrize("streams,reason", [
    ({"/odom": []}, "no /odom"), ({"/odom": [[0, float("nan"), 0, 1]]}, "non-finite sample"),
    ({"/odom": [[0, 0, 0, 1]], "/drive": [[float("inf"), 1, 0]]}, "non-finite sample"),
])
def test_unusable_motion_has_a_reason(streams, reason):
    with pytest.raises(ValueError, match=reason):
        viz.prepare(streams)


def test_wrong_cache_layout(tmp_path):
    cache = tmp_path / "analysis"
    cache.mkdir()
    np.savez(cache / "run.npz", odom=np.ones((1, 3)))
    with pytest.raises(ValueError, match="invalid odom columns"):
        viz.load(tmp_path)


def test_metadata_remains_text_inside_report():
    name = "sample</script> & comparison"
    report = viz.render([{"name": name}])
    raw = re.search(r'<script id="runs" type="application/json">([^<]+)</script>', report)[1]
    assert json.loads(raw) == [{"name": name}]


def test_cached_runs_cli_does_not_clobber_output(tmp_path):
    # Independent oracle: raw wheel-odometry XY and speed columns in runlog.Run.
    paths = [viz.DATA / "rb1/racerbot_runs/20261026-194514", viz.DATA / "rb2/racerbot_runs/20261030-184826"]
    out = tmp_path / "report.html"
    cmd = [sys.executable, viz.__file__, *paths, "-o", out]
    subprocess.run(cmd, cwd=tmp_path, check=True, timeout=30)
    report = out.read_text()
    runs = json.loads(re.search(r'<script id="runs" type="application/json">([^<]+)</script>', report)[1])
    assert [r["car"] for r in runs] == ["rb1", "rb2"]
    for r, path in zip(runs, paths):
        with np.load(path / "analysis/run.npz") as z:
            odom = z["odom"]
            xy = np.diff(odom[:, 3:5], axis=0)
            assert r["distance"] == pytest.approx(np.hypot(xy[:, 0], xy[:, 1]).sum(), abs=1e-8)  # m
            assert r["peak"] == max(abs(odom[:, 1]))
            assert len(r["odom"]) == len(odom)
    out.write_text("preserve me")
    fail = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert fail.returncode != 0 and "Could not write" in fail.stderr
    assert out.read_text() == "preserve me"


def test_split_mcap_matches_independently_decoded_cache():
    pytest.importorskip("mcap_ros2", reason="raw MCAP reader test requires the optional mcap-ros2-support package")
    # runlog originally decoded this split bag with rosbag2_py, not mcap_ros2.
    path = viz.DATA / "rb1/racerbot_runs/20261026-194514"
    actual = viz.prepare(viz.mcap_streams(sorted((path / "bag").glob("*.mcap"))))
    expected = viz.load(path)
    assert actual["cmdTopic"] == expected["cmdTopic"] == "/drive"
    for key in ("odom", "cmd", "joy"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1.1e-5)


def test_short_joystick_message_releases_lb(tmp_path):
    writer = pytest.importorskip("mcap_ros2.writer", reason="raw MCAP tests require mcap-ros2-support")
    path = tmp_path / "buttons.mcap"
    with writer.Writer(str(path)) as bag:
        schema = bag.register_msgdef("test_msgs/msg/Buttons", "int32[] buttons")
        for second, buttons in enumerate(([0, 0, 0, 0, 1], [], [0, 0, 0, 0, 1]), 1):
            bag.write_message("/joy", schema, {"buttons": buttons}, log_time=second * 10**9)
    # Match runlog.read_bag: no button at index 4 means LB was not pressed.
    assert viz.mcap_streams([path])["/joy"] == [[1, 1], [2, 0], [3, 1]]
