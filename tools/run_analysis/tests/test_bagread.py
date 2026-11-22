"""Read real tiny bags without ROS, retaining timestamps, values and filtering."""

from pathlib import Path
import sys
from types import ModuleType

import pytest
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_analysis.bagread import messages  # noqa: E402


@pytest.fixture(params=["rosbag2_py", "rclpy"])
def without_ros(request, monkeypatch):
    monkeypatch.setitem(sys.modules, "rosbag2_py", ModuleType("rosbag2_py"))
    monkeypatch.setitem(sys.modules, request.param, None)


@pytest.fixture
def recorded_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    store = get_typestore(Stores.ROS2_JAZZY)
    name = "std_msgs/msg/String"
    message = store.types[name]
    with Writer(run / "bag", version=9) as writer:
        kept = writer.add_connection("/kept", name, typestore=store)
        ignored = writer.add_connection("/ignored", name, typestore=store)
        for connection, stamp, value in [(kept, 1_250_000_000, "first"),
                                          (ignored, 1_500_000_000, "not selected"),
                                          (kept, 2_750_000_000, "second")]:
            writer.write(connection, stamp, store.serialize_cdr(message(value), name))
    return run


@pytest.mark.parametrize("bag_path", [False, True])
def test_fallback_decodes_selected_messages_in_receive_order(without_ros, recorded_run, bag_path):
    path = recorded_run / "bag" if bag_path else recorded_run
    rows = list(messages(path, ["/kept", "/absent"]))
    assert [(topic, stamp, msg.data) for topic, stamp, msg in rows] == [
        ("/kept", 1.25, "first"), ("/kept", 2.75, "second")]


@pytest.mark.parametrize("topics", [[], ["/absent"]])
def test_fallback_yields_nothing_when_no_topics_match(without_ros, recorded_run, topics):
    assert list(messages(recorded_run, topics)) == []


def test_fallback_does_not_hide_missing_bag(without_ros, tmp_path):
    with pytest.raises(FileNotFoundError):
        list(messages(tmp_path / "missing", ["/kept"]))
