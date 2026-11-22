"""Offline-only ROS contract checks; never launch the hardware stack."""

import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from mcap.reader import make_reader
from nav_msgs.msg import OccupancyGrid, Odometry, Path as ROSPath
from rclpy.clock import ClockType
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from sensor_msgs.msg import Imu, Joy, LaserScan
from std_msgs.msg import Float64
import numpy as np
import yaml

from mapped_replay import RUN, load_assets, model_store, params
from neural_racer.neural_racer_node import NeuralRacerNode
from neural_racer.policy_numpy import PolicyNumpy

NS = "/test_only/mapped_wire"
REAL = ("/drive", "/teleop", "/ackermann_cmd", "/commands/motor/speed", "/commands/servo/position")


def isolate():
    if (
        not __debug__
        or os.environ.get("ROS_DISCOVERY_SERVER")
        or os.environ.get("ROS_STATIC_PEERS")
    ):
        raise RuntimeError("use normal Python with no external discovery peers")
    if (
        os.environ.get("ROS_DOMAIN_ID") != "79"
        or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
        or os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST"
    ):
        raise RuntimeError("isolated localhost ROS domain required")
    if Path("/proc/device-tree/serial-number").exists() or Path("/dev/sensors").exists():
        raise RuntimeError("never run the harness on a vehicle")


def no_real_publishers(node):
    for topic in REAL:
        if node.count_publishers(topic) != 0:
            raise RuntimeError(f"unexpected publisher on {topic}, terminating test")


def under_review_checkpoint():
    model = Path(os.environ["MODEL"])
    assert model_store.weights_hash(model / "weights.npz") == "a0eef2af0a3757b6"
    w, card = model_store.load_dir(model)
    error = model_store.golden_error(PolicyNumpy(w, card), model / "golden.npz")
    assert error < model_store.GOLDEN_ATOL
    for driving in (False, True):
        try:
            model_store.load_approved(model.name, models_dir=model.parent, driving=driving)
        except model_store.ModelStoreError as e:
            assert "not approved" in str(e)
        else:
            raise AssertionError("expected a review-only bundle with no approval")
    return error


def live_refusal():
    """Unmodified shadow node, all bag scans, original Joy/IMU and no manifest."""
    model = Path(os.environ["MODEL"])
    args = [
        "--ros-args",
        "-p",
        "mode:=map",
        "-p",
        "pose_topic:=/pf/viz/inferred_pose",
        "-p",
        "frozen_map_topic:=/map",
        "-p",
        f"model_hash:={model.name}",
        "-p",
        f"models_dir:={model.parent}",
        "-p",
        f"intent_topic:={NS}/intent",
        "-p",
        f"drive_topic:={NS}/unused_drive",
    ]
    rclpy.init(args=args)
    n = NeuralRacerNode()
    try:
        assert (
            n.require_imu
            and n.enable_deadman
            and n.shadow
            and n.policy is None
            and "not approved" in n.refusal
        )
        assert n.cmd_pub.topic_name == "/neural_racer/shadow_drive"
        no_real_publishers(n)
        # The node's output is captured instead of sent over DDS.
        captured = []
        n.cmd_pub.publish = lambda msg: captured.append((msg.drive.steering_angle, msg.drive.speed))
        bag_clock = SimpleNamespace(
            now=lambda: Time(nanoseconds=stamp, clock_type=ClockType.ROS_TIME)
        )
        n.get_clock = lambda: bag_clock
        _, offline_grid, xy, reference = load_assets()
        path = ROSPath()
        path.header.frame_id = "map"
        for x, y in xy:
            pt = PoseStamped()
            pt.header.frame_id = "map"
            pt.pose.position.x = float(x)
            pt.pose.position.y = float(y)
            pt.pose.orientation.w = 1.0
            path.poses.append(pt)
        n.path_callback(path)
        routes = {
            "/joy": (Joy, n.joy_callback),
            "/map": (OccupancyGrid, n.map_callback),
            "/pf/viz/inferred_pose": (PoseStamped, n.pose_callback),
            "/imu/data": (Imu, n.imu_callback),
            "/odom": (Odometry, n.odom_callback),
            "/scan": (LaserScan, n.scan_callback),
        }
        reasons = Counter()
        counts = Counter()
        parts = yaml.safe_load((RUN / "bag/metadata.yaml").read_text())[
            "rosbag2_bagfile_information"
        ]["relative_file_paths"]
        for part in parts:
            with (RUN / "bag" / part).open("rb") as f:
                for schema, ch, m in make_reader(f).iter_messages(topics=list(routes)):
                    stamp = m.log_time
                    typ, callback = routes[ch.topic]
                    assert schema.name.split("/")[-1] == typ.__name__
                    callback(deserialize_message(m.data, typ))
                    counts[ch.topic] += 1
                    if ch.topic == "/scan":
                        reasons[n.last_decision_state] += 1
        assert len(captured) == counts["/scan"] == 8784 and all(
            pair == (0.0, 0.0) for pair in captured
        )
        assert (
            reasons["refused"]
            and reasons["deadman_released"]
            and not n.imu_status_ok
            and n.imu_sample is not None
        )
        assert n.ref_path is not None and n.grid is not None and n.pose is not None
        assert np.array_equal(n.grid.blocked, offline_grid.blocked)
        assert np.allclose(
            reference.preview_block(*n.pose)[0], n.ref_path.preview_block(*n.pose)[0]
        )
        no_real_publishers(n)
        return dict(
            shadow_topic=n.cmd_pub.topic_name,
            zero_shadow_messages=len(captured),
            captured_in_memory=True,
            actual_node_refusal=n.refusal,
            reasons=dict(reasons),
            typed_message_counts=dict(counts),
            map_path_pose_callback_preview_equal=True,
            imu_health_ok=n.imu_status_ok,
        )
    finally:
        n.destroy_node()
        rclpy.shutdown()


def converter():
    """Test the real Ackermann message builder and C++ converter, not the VESC driver."""
    rclpy.init(args=[])
    n = Node("mapped_wire_observer", namespace=NS)
    p = params("ackermann_to_vesc_node")
    cmd = [
        "ros2",
        "run",
        "vesc_ackermann",
        "ackermann_to_vesc_node",
        "--ros-args",
        "-r",
        f"__ns:={NS}",
    ]
    for key in (
        "speed_to_erpm_gain",
        "speed_to_erpm_offset",
        "steering_angle_to_servo_gain",
        "steering_angle_to_servo_offset",
    ):
        cmd += ["-p", f"{key}:={p[key]}"]
    proc = None
    try:
        no_real_publishers(n)
        if n.get_node_names_and_namespaces() != [(n.get_name(), NS)]:
            raise RuntimeError("run on an empty localhost domain")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, start_new_session=True)
        cmds = n.create_publisher(AckermannDriveStamped, "ackermann_cmd", 10)
        motor = []
        servo = []
        sm = n.create_subscription(
            Float64, "commands/motor/speed", lambda m: motor.append(m.data), 10
        )
        ss = n.create_subscription(
            Float64, "commands/servo/position", lambda m: servo.append(m.data), 10
        )

        def until(test):
            limit = time.monotonic() + 8
            while not test():
                if time.monotonic() > limit or proc.poll() is not None:
                    raise RuntimeError("DDS discovery or converter response timed out")
                rclpy.spin_once(n, timeout_sec=0.02)

        until(
            lambda: cmds.get_subscription_count() == 1
            and n.count_publishers(sm.topic_name) == 1
            and n.count_publishers(ss.topic_name) == 1
        )
        sink = SimpleNamespace(cmd_pub=cmds, get_clock=n.get_clock)
        # Discovery is asynchronous in both directions. Establish data-plane
        # readiness with stop messages; never retry a measured sample.
        deadline = time.monotonic() + 5
        while not motor or not servo:
            if time.monotonic() > deadline:
                raise RuntimeError("neutral handshake failed")
            NeuralRacerNode._publish(sink, 0.0, 0.0)
            end = time.monotonic() + 0.12
            while time.monotonic() < end:
                rclpy.spin_once(n, timeout_sec=0.01)
        end = time.monotonic() + 0.2
        while time.monotonic() < end:
            rclpy.spin_once(n, timeout_sec=0.01)
        assert all(i == 0.0 for i in motor) and all(
            i == p["steering_angle_to_servo_offset"] for i in servo
        )
        motor.clear()
        servo.clear()
        samples = json.loads(Path("mapped-result.json").read_text())["examples"]
        if len(samples) != 8 or not all(
            r["reason"] == "braking" and r["wrapped"][1] == 0.0 and abs(r["wrapped"][0]) <= 0.349
            for r in samples
        ):
            raise ValueError("not the eight bounded November diagnostic samples")
        cases = [("mapped_proposal", *r["wrapped"]) for r in samples]
        cases += [("stop", 0.0, 0.0), ("synthetic_unit_check", 0.125, 0.5)]
        outputs = []
        for case, steer, speed in cases:
            before = len(motor)
            NeuralRacerNode._publish(sink, steer, speed)
            until(lambda: len(motor) > before and len(servo) > before)
            expected = [
                p["speed_to_erpm_gain"] * float(np.float32(speed)) + p["speed_to_erpm_offset"],
                p["steering_angle_to_servo_gain"] * float(np.float32(steer))
                + p["steering_angle_to_servo_offset"],
            ]
            assert [motor[-1], servo[-1]] == expected and len(motor) == len(servo) == before + 1
            outputs.append(
                dict(
                    case=case, speed_mps=speed, steering_rad=steer, erpm=motor[-1], servo=servo[-1]
                )
            )
        no_real_publishers(n)
        return dict(
            private_namespace=NS,
            results=outputs,
            hardware_driver_started=False,
            global_command_publishers={k: n.count_publishers(k) for k in REAL},
        )
    finally:
        if proc is not None:
            import signal

            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=6)
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    isolate()
    result = {
        "golden_max_abs_error": under_review_checkpoint(),
        "default_node": live_refusal(),
        "translator": converter(),
    }
    print(json.dumps(result, indent=2, allow_nan=False))
