"""The scorecard that compares the simulator with the real car (validate_calibration.py).

Pure functions on synthetic data whose answer the test built. The simulated and real
sides are scored by these same functions, so a fault here would flatter both alike.
Run by: python3 -m pytest tools/f1tenth_sim/tests/ tools/racerbot_sim/tests/ tools/run_analysis/tests/ tools/dev/tests/ -q
"""
import math
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import validate_calibration as vc  # noqa: E402

BOX = (0.0, 1.0, 0.0, 1.0)


def test_flicker_counts_changes_per_pass_and_the_fallback_share():
    # Two passes through the box, outside samples in between.
    x = np.r_[np.full(2, 5.0), np.full(6, 0.5), np.full(2, 5.0), np.full(4, 0.5), 5.0]
    y = np.full(x.size, 0.5)
    fb = np.r_[[1, 1], [0, 1, 1, 0, 0, 0], [1, 1], [0, 0, 1, 1], 0].astype(bool)
    out = vc.flicker(x, y, fb, BOX)
    assert out["passes"] == 2
    assert out["share"] == pytest.approx(4 / 10, abs=1e-12)      # inside samples only
    assert out["changes_per_pass"] == pytest.approx((2 + 1) / 2, abs=1e-12)


def test_flicker_never_counts_a_change_across_two_passes():
    x = np.r_[0.5, 5.0, 0.5]
    out = vc.flicker(x, np.full(3, 0.5), np.array([False, True, True]), BOX)
    assert out["passes"] == 2 and out["changes_per_pass"] == 0.0


def test_flicker_with_no_pass_is_nan_not_zero():
    out = vc.flicker(np.full(5, 9.0), np.full(5, 9.0), np.zeros(5, bool), BOX)
    assert out["passes"] == 0 and math.isnan(out["share"])


def test_grip_rows_takes_medians_of_magnitudes_and_drops_thin_bins():
    asked = np.r_[np.full(10, -1.2), np.full(10, 3.2), np.full(3, 5.5)]
    achieved = np.r_[np.full(10, -1.0), np.r_[np.full(9, 2.6), 9.0], np.full(3, 3.3)]
    rows = vc.grip_rows(asked, achieved)
    assert rows == [(1, 1.5, 10, pytest.approx(1.0)), (3, 3.5, 10, pytest.approx(2.6))]


def test_grip_error_is_rms_over_the_shared_bins_only():
    real = [(1, 2, 10, 1.0), (2, 3, 10, 2.0), (3, 4, 10, 3.0)]
    sim = [(1, 2, 10, 1.3), (3, 4, 10, 2.6), (9, 10, 10, 50.0)]
    assert vc.grip_error(real, sim) == pytest.approx(math.sqrt((0.09 + 0.16) / 2), abs=1e-12)
    assert math.isnan(vc.grip_error(real, [(9, 10, 10, 1.0)]))


def test_sector_error_rms_worst_and_bias():
    real = [2.0, 3.0, np.nan, 1.0]
    sim = [2.5, 2.0, 4.0, 1.0]
    e = vc.sector_error(real, sim)
    assert e["sectors"] == 3
    assert e["rms"] == pytest.approx(math.sqrt((0.25 + 1.0 + 0.0) / 3), abs=1e-12)
    assert e["worst"] == pytest.approx(1.0, abs=1e-12)
    assert e["bias"] == pytest.approx(-0.5 / 3, abs=1e-12)


def test_summarise_laps_leaves_out_the_standing_lap():
    t = np.arange(0.0, 31.0, 0.1)
    progress = np.where(t < 12.0, t * 10.0 / 12.0, 10.0 + (t - 12.0))   # 12 s, then 10 s laps
    out = vc.summarise_laps(t, progress, 10.0)
    assert out["laps"] == [pytest.approx(12.0, abs=0.01), pytest.approx(10.0, abs=0.01)]
    assert out["flying"] == [pytest.approx(10.0, abs=0.01)]


def test_params_from_run_takes_the_launched_values_and_that_days_code_paths(tmp_path):
    params = tmp_path / "metadata" / "params"
    params.mkdir(parents=True)
    launched = {"max_speed": 3.5, "max_lateral_accel": 2.5, "aim_depth_bias": 1.0, "not_a_parameter": 7}
    (params / "gap_follow_node.yaml").write_text(yaml.safe_dump({"/gap_follow_node": {"ros__parameters": launched}}))
    out = vc.params_from_run(tmp_path)
    assert out["max_speed"] == 3.5 and out["max_lateral_accel"] == 2.5
    assert "not_a_parameter" not in out
    for name, value in vc.OLD_CODE.items():          # the run's code had none of these paths
        assert out[name] == value


def test_params_from_run_with_launched_code_keeps_every_launched_value(tmp_path):
    """A run made on today's code (e.g. car 2's 20261106-171030, launched with aim_depth_bias 1.0) must be
    simulated with what it was launched with: OLD_CODE describes the 2026-10-30 code only."""
    params = tmp_path / "metadata" / "params"
    params.mkdir(parents=True)
    launched = {"max_speed": 5.0, "aim_depth_bias": 1.0, "swept_clearance_max_length": 4.5}
    (params / "gap_follow_node.yaml").write_text(yaml.safe_dump({"/gap_follow_node": {"ros__parameters": launched}}))
    out = vc.params_from_run(tmp_path, old_code=False)
    # oracle: the values this test launched with.
    assert out["aim_depth_bias"] == 1.0
    assert out["swept_clearance_max_length"] == 4.5
    assert out["max_speed"] == 5.0
    # OLD_CODE keys the run was not launched with stay unset, so the mirror uses today's defaults.
    assert "opponent_follow_floor" not in out
