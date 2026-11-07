import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import run_validation as rv  # noqa: E402


def _scan_at(angle, distance):
    rv._ensure_config("rb2")
    lidar = rv._lidar()
    scan = np.full(lidar.num_beams, np.inf)
    index = round((angle - lidar.angle_min) / lidar.angle_increment)
    scan[index] = distance
    return scan


def test_apply_map_safety_stops_only_inside_raw_emergency_cone():
    steering, speed = 0.12, 2.4
    # pure_pursuit.yaml sets emergency_stop_distance to 0.35 m.
    assert rv.apply_map_safety(_scan_at(0.0, 0.30), steering, speed, "rb2") == (
        steering, 0.0, "stop")
    assert rv.apply_map_safety(_scan_at(0.0, 0.40), steering, speed, "rb2") == (
        steering, speed, "none")
    scan = _scan_at(0.0, 0.50)
    fallback = rv.apply_fallback_safety(scan, steering, speed, car_id="rb2")
    assert fallback[1:] == (1.0, "avoid")
    assert rv.apply_map_safety(scan, steering, speed, "rb2") == (
        steering, speed, "none")
    assert rv.apply_map_safety(_scan_at(math.pi / 2.0, 0.20), steering, speed,
                               "rb2") == (steering, speed, "none")
    assert rv.apply_map_safety(np.full_like(scan, np.inf), steering, speed,
                               "rb2") == (steering, speed, "none")
    assert rv.apply_map_safety(np.full_like(scan, np.nan), steering, speed,
                               "rb2") == (steering, speed, "none")


def test_lap_times_from_cumulative_counts():
    # Closed form: completion times 3 s and 7 s; durations are 3 and 4 s.
    assert rv.lap_times_from_counts(list(range(10)), [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]) == [3.0, 4.0]
    # Oracle: cumulative fractions below the next integer complete no lap.
    assert rv.lap_times_from_counts([0.0, 1.0, 2.0], [0.0, 0.4, 0.9]) == []
    # Oracle: empty and single-sample series contain no completed interval.
    assert rv.lap_times_from_counts([], []) == []
    assert rv.lap_times_from_counts([0.0], [0.0]) == []


@pytest.mark.parametrize("kwargs, message", [
    ({"safety": "nowhere"}, "safety"),
    ({"laps": 0}, "laps"),
    ({"steering_limit": 0.0}, "steering_limit"),
    ({"steering_limit": float("nan")}, "steering_limit"),
])
def test_run_pure_solo_validates_options_before_building_simulator(kwargs, message,
                                                                   monkeypatch):
    def should_not_build(*args, **kwargs):
        raise AssertionError("simulator built before argument validation")
    monkeypatch.setattr(rv, "make_plant", should_not_build)
    with pytest.raises(ValueError, match=message):
        rv.run_pure_solo("indoor_wide", 1, 1.0, **kwargs)


def test_run_pure_solo_map_safety_completes_two_laps_and_restores_limit():
    import eval_gap
    old_limit = rv.STEERING_LIMIT
    result = rv.run_pure_solo(eval_gap.track_path("indoor_wide"), 12345, 90.0, car_id="rb2",
                              safety="map", laps=2, steering_limit=0.30)
    assert len(result["lap_times_s"]) == 2
    assert rv.STEERING_LIMIT == old_limit
