"""Fixed-input integration oracles; run from the directory containing both JSON results."""

import json
from pathlib import Path

import numpy as np
import pytest


def result(name):
    return json.loads(Path(name).read_text())


def test_bag_and_brake():
    r = result("mapped-result.json")
    assert r["map_grid_equal"] and r["map_cells"] == 141696 and r["reference_points"] == 186
    assert r["topics"]["/scan"] == sum(r["scan_outcomes"].values()) == 8784
    n = r["scan_outcomes"]["map_inference"]
    assert n == 6958 and r["bindings"] == {"braking": n}
    assert r["raw_speed_min_median_max"][0] > 0.75
    assert r["clearance_min_median_max"][2] < 0
    assert r["max_bounded_speed_mps"] == r["nonzero_bounded_speed"] == 0
    assert 0 < r["max_abs_steer_rad"] <= 0.349
    assert r["projected_path_hits"] == 1423 and r["imu_health_messages"] == 0


def test_strict_shadow_refusal():
    r = result("wire-result.json")
    assert r["golden_max_abs_error"] <= 1e-5
    s = r["default_node"]
    assert "not approved" in s["actual_node_refusal"] and not s["imu_health_ok"]
    assert s["map_path_pose_callback_preview_equal"] and s["captured_in_memory"]
    assert s["shadow_topic"] == "/neural_racer/shadow_drive"
    assert s["zero_shadow_messages"] == sum(s["reasons"].values()) == 8784
    assert s["reasons"] == {"refused": 6959, "deadman_released": 1809, "waiting_for_joy": 16}


def test_converter_matches_bag_profile():
    s = result("wire-result.json")["translator"]
    assert s["private_namespace"].startswith("/test_only/") and not s["hardware_driver_started"]
    assert s["global_command_publishers"] == dict.fromkeys(
        (
            "/drive",
            "/teleop",
            "/ackermann_cmd",
            "/commands/motor/speed",
            "/commands/servo/position",
        ),
        0,
    )
    samples = result("mapped-result.json")["examples"]
    cases = s["results"]
    assert len(cases) == 10 and len(samples) == 8
    for sample, case in zip(samples, cases):
        assert case["case"] == "mapped_proposal"
        assert [case["steering_rad"], case["speed_mps"]] == sample["wrapped"]
        assert case["erpm"] == 0 and case["servo"] == pytest.approx(
            0.5904 - 0.78 * float(np.float32(sample["wrapped"][0])), abs=1e-10
        )
    stop, positive = cases[-2:]
    assert (stop["case"], stop["erpm"], stop["servo"]) == ("stop", 0.0, 0.5904)
    assert (positive["case"], positive["erpm"], positive["servo"]) == (
        "synthetic_unit_check",
        2307.0,
        0.4929,
    )
