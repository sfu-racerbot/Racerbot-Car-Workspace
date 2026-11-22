"""Fixed-input oracles for the three selectable-driver traces; run with pytest."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def load(stem):
    summary = json.loads(Path(stem + ".json").read_text())
    trace = Path(stem + ".csv")
    assert hashlib.sha256(trace.read_bytes()).hexdigest() == summary["trace_sha256"]
    with trace.open() as f:
        rows = list(csv.DictReader(f))
    assert all(r["contact"] == "False" for r in rows)
    assert summary["track"] == "car2_course_2026_11_11" and summary["car"] == "rb2"
    return summary, rows


def columns(rows, *names):
    return np.array([[float(row[k]) for k in names] for row in rows])


def test_saved_policy_reaches_plant_only_after_wrapper():
    summary, rows = load("saved-policy")
    held = [r for r in rows if r["phase"] == "held"]
    assert len(held) == summary["phase_counts"]["exact_policy_wrapper_to_plant"] == 2400
    expected = columns(held, "wrapped_steer", "wrapped_speed")
    actual = columns(held, "plant_input_steer", "plant_input_speed")
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()
    assert (
        np.max(np.abs(actual[:, 0])) <= 0.349
        and np.max(np.abs(np.diff(actual[:, 0]))) <= 0.05000001
    )
    assert np.all((actual[:, 1] >= 0) & (actual[:, 1] <= 1))
    assert np.count_nonzero(actual[:, 1] > 0) > 0
    assert np.any((actual[:, 1] == 0) & (columns(held, "raw_speed")[:, 0] > 0.5))
    assert (
        np.ptp(columns(rows, "truth_x")[:, 0]) > 1 and max(float(r["lap_count"]) for r in rows) == 0
    )


def test_release_while_moving():
    _, rows = load("moving-release")
    held = [r for r in rows if r["phase"] == "held"]
    off = [r for r in rows if r["phase"] == "released"]
    assert len(held) == 40 and len(off) == 60 and float(held[-1]["truth_speed"]) > 0.5
    for phase in ("not_held", "released"):
        actual = columns(
            [r for r in rows if r["phase"] == phase], "plant_input_steer", "plant_input_speed"
        )
        np.testing.assert_array_equal(actual, np.zeros_like(actual))
    assert abs(float(off[-1]["truth_speed"])) < 0.01


def test_gap_selection_is_distinct_on_same_grid():
    s, baseline = load("gap-follow")
    check, _ = load("saved-policy")
    assert s["spawn_xy_yaw"] == check["spawn_xy_yaw"] and s["seed"] == check["seed"]
    held = [r for r in baseline if r["phase"] == "held"]
    assert s["driver"] == "gap-follow" and s["checkpoint_sha256"] is None
    assert all(r["limit"] == "gap_follow" and not r["raw_speed"] for r in held)
    assert float(held[-1]["lap_count"]) == 1 and len(held) < 2400
    assert max(float(r["plant_input_speed"]) for r in held) > 1
