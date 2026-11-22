"""Offline regression checks for the stock-asset adapter; run with tools/run_analysis/tests/."""
from io import StringIO
import math
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from run_analysis.gazebo import prepare_world, replay  # noqa: E402


def test_mux_clock_gate_uses_inclusive_age_and_integer_step_boundaries():
    from run_analysis.gazebo.command_coverage import count_windows

    b = 1_800_000_000_000_000_001
    # 125 ms updates aligned with a 25 ms grid: last held age is exactly 100 ms.
    clock = [b + k * 125_000_000 for k in range(177)]
    assert count_windows(clock) == (2, 2)
    assert count_windows(clock[:-1]) == (1, 1)
    # First update arrives 1 ns after the tick; at that tick the old sample is stale.
    clock[1] += 1
    assert count_windows(clock) == (2, 1)
    assert count_windows([]) == (0, 0)


def test_command_curvature_has_the_requested_signed_turn_radius():
    # A bicycle making a circle of radius R has wheel angle atan(L/R) and yaw speed v/R.
    for v, radius, wheelbase in ((3, -2, 0.32), (-1, 4, 1.0), (0, 2, 0.8)):
        angular = replay.command_yaw_rate(v, math.atan(wheelbase / radius), wheelbase)
        assert angular == pytest.approx(v / radius, abs=1e-12)  # rad/s


def test_csv_uses_named_columns_in_order_including_reverse_and_stop():
    stream = StringIO("steer_rad,extra,speed_mps\n-0.2,x,-1.2\n0,y,0\n")
    assert replay.read_actions(stream) == [(-1.2, -0.2), (0, 0)]


@pytest.mark.parametrize("data,error", [
    ("speed_mps,steer_rad\n", "empty"),
    ("x,y\n1,2\n", "columns"),
    ("speed_mps,steer_rad\n1\n", "numeric"),
    ("speed_mps,steer_rad\nno,0.1\n", "numeric"),
    ("speed_mps,steer_rad\n1,nan\n", "nonfinite"),
    ("speed_mps,steer_rad\ninf,0\n", "nonfinite"),
])
def test_bad_csv_is_rejected_before_simulation(data, error):
    with pytest.raises(ValueError, match=error):
        replay.read_actions(StringIO(data))


def test_world_preserves_the_floor_and_places_only_mapped_cars_at_the_recorded_pose(tmp_path):
    start = tmp_path / "start_pose.yaml"
    start.write_text("yaw: 0.25\ny: 4.0\nx: -3.5\n")
    for pose, expected, includes in ((None, [0, 0, 0.08, 0, 0, 0], []),
                                     (start, [-3.5, 4, 0.08, 0, 0, 0.25], ["model://map"])):
        sdf = ET.fromstring('<sdf><world name="empty"><model name="floor"/></world></sdf>')
        car = ET.fromstring('<model name="car"><link name="base_link"/></model>')
        prepare_world.assemble(sdf, car, pose)
        world = sdf.find("world[@name='simple_world']")
        assert world.find("model[@name='floor']") is not None
        assert world.find("model[@name='car']/link[@name='base_link']") is not None
        assert list(map(float, world.findtext("model[@name='car']/pose").split())) == expected
        assert [uri.text for uri in world.findall("include/uri")] == includes


@pytest.mark.parametrize("data", ["x: 1\ny: 2\n", "x: .inf\ny: 1\nyaw: 0\n", "[]"])
def test_invalid_map_start_cannot_silently_fall_back_to_origin(tmp_path, data):
    start = tmp_path / "start_pose.yaml"
    start.write_text(data)
    with pytest.raises(ValueError, match="map start pose"):
        prepare_world.assemble(ET.fromstring("<sdf><world/></sdf>"), ET.Element("model"), start)
