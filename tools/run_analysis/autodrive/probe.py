"""Measure the stock 2026 IROS practice simulator in its isolated devkit container.

No Racerbot command publishers. Three identical wall-timed, unseeded throttle
pulses test motion after a soft reset; they are not matched November rollouts.
"""

import json
import math
import os
import time

import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Bool, Float32

CAR = "/autodrive/roboracer_1/"


def main():
    if os.environ.get("ROS_DOMAIN_ID") != "79" or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise SystemExit("Use domain 79 and localhost-only discovery inside the isolated devkit.")
    rclpy.init(args=[])
    node = rclpy.create_node("autodrive_practice_probe")
    throttle = node.create_publisher(Float32, CAR + "throttle_command", 1)
    steering = node.create_publisher(Float32, CAR + "steering_command", 1)
    reset = node.create_publisher(Bool, "/autodrive/reset_command", 1)
    samples = {k: [] for k in ("scan", "pose", "lap_time", "camera")}
    reply_to_scan = False

    def on_odom(msg):
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        samples["pose"].append(((pos.x, pos.y), (vel.x, vel.y)))

    def on_scan(msg):
        samples["scan"].append((len(msg.ranges), msg.range_max))
        if reply_to_scan:
            throttle.publish(Float32(data=0.0))
            steering.publish(Float32(data=0.0))

    def on_image(msg):
        samples["camera"].append((msg.width, msg.height, min(msg.data), max(msg.data)))

    def spin(seconds):
        start = time.monotonic()
        while time.monotonic() - start < seconds:
            rclpy.spin_once(node, timeout_sec=0.02)
        elapsed = time.monotonic() - start
        if elapsed > seconds + 0.5:
            raise SystemExit("Wall clock overran measurement window; discard this run.")
        return elapsed

    def last_pose():
        if not samples["pose"]:
            raise SystemExit("No odometry received during trial; discard this run.")
        return samples["pose"][-1]

    node.create_subscription(Odometry, CAR + "odom", on_odom, 1)
    node.create_subscription(LaserScan, CAR + "lidar", on_scan, 1)
    node.create_subscription(Image, CAR + "front_camera", on_image, 1)
    node.create_subscription(Float32, CAR + "lap_time", lambda m: samples["lap_time"].append(m.data), 1)
    try:
        spin(2)
        if not all(samples.values()) or not all(p.get_subscription_count() for p in (throttle, steering, reset)):
            raise SystemExit("Missing stock practice feeds or command subscribers; no measurement.")
        steering.publish(Float32(data=0.0))
        throttle.publish(Float32(data=0.0))
        for values in samples.values():
            values.clear()
        elapsed = spin(10)
        if not all(samples.values()):
            raise SystemExit("Missing measurements during observation.")
        print(json.dumps(dict(
            elapsed_s=elapsed, counts={k: len(v) for k, v in samples.items()},
            scan_beams_and_max_m=sorted(set(samples["scan"])),
            camera_width_height_min_max=sorted(set(samples["camera"])),
            lap_timer_elapsed_s=samples["lap_time"][-1] - samples["lap_time"][0],
        )), flush=True)
        samples["scan"].clear()
        reply_to_scan = True
        elapsed = spin(10)
        reply_to_scan = False
        if not samples["scan"]:
            raise SystemExit("No observations for the no-op policy.")
        print(json.dumps(dict(noop_commands=len(samples["scan"]), elapsed_s=elapsed,
                              scan_to_action_hz=len(samples["scan"]) / elapsed)), flush=True)
        for trial in range(3):
            reset.publish(Bool(data=True))
            spin(0.5)
            reset.publish(Bool(data=False))
            samples["pose"].clear()
            spin(1)
            start, _ = last_pose()
            throttle.publish(Float32(data=0.1))
            samples["pose"].clear()
            elapsed = spin(1)
            throttle.publish(Float32(data=0.0))
            end, velocity = last_pose()
            print(json.dumps(dict(trial=trial + 1, reset_xy_m=start, end_xy_m=end,
                                  odom_twist_end_xy_m_s=velocity, pulse_wall_s=elapsed,
                                  displacement_m=math.dist(start, end))), flush=True)
            spin(1)
    finally:
        throttle.publish(Float32(data=0.0))
        steering.publish(Float32(data=0.0))
        reset.publish(Bool(data=False))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
