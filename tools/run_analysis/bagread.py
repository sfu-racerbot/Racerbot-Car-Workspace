"""Read chosen topics from a recorded run's bag, in time order, writing nothing.

runlog.load_run caches a fixed set of topics in <run>/analysis/, inside data/.
The tools built on this module need other topics (/drive_intent, the VESC IMU)
and must leave data/ untouched, so they read the bag directly. Without ROS, uses
rosbags to deserialize the bag's embedded message definitions.
"""

from __future__ import annotations

from pathlib import Path


def bag_dir(run) -> Path:
    """The bag of run directory ``run``, or ``run`` itself if it holds the bag."""
    run = Path(run).expanduser()
    return run / "bag" if (run / "bag").is_dir() else run


def messages(run, topics):
    """Yield (topic, receive time in seconds, message) for ``topics``, skipping
    any the bag does not have."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError:
        from rosbags.highlevel import AnyReader

        with AnyReader([bag_dir(run)]) as reader:
            connections = [c for c in reader.connections if c.topic in topics]
            if not connections:
                return
            for connection, stamp_ns, data in reader.messages(connections=connections):
                yield (connection.topic, stamp_ns / 1e9,
                       reader.deserialize(data, connection.msgtype))
        return

    path = bag_dir(run)
    storage = "mcap" if any(path.glob("*.mcap")) else "sqlite3"
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=storage),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        yield topic, stamp_ns / 1e9, deserialize_message(data, get_message(types[topic]))
