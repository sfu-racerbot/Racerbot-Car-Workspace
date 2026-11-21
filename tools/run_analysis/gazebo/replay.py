"""Step the upstream car using a 40 Hz command CSV. Offline only; no ROS graph."""

import argparse
import csv
import json
import math
import os
import tempfile
import time
import uuid
from pathlib import Path
from xml.etree import ElementTree as ET


def run(world, actions, seed):
    # Never deliver replay commands to another Gazebo instance.
    os.environ["GZ_PARTITION"] = f"racerbot_bench_{uuid.uuid4().hex}"
    os.environ["GZ_IP"] = "127.0.0.1"
    import gz.sim8 as gz
    from gz.math7 import Rand
    from gz.msgs10.twist_pb2 import Twist
    from gz.transport13 import Node

    tree = ET.parse(world)
    physics = tree.find("world/physics")
    if float(physics.findtext("max_step_size")) != 0.001:
        raise ValueError("benchmark expects the stock 1 ms physics step")
    physics.find("real_time_factor").text = "0"
    physics.find("real_time_update_rate").text = "0"
    car = tree.find("world/model[@name='f1tenth_testbed']")
    drive = car.find("plugin[@name='gz::sim::systems::AckermannSteering']")
    # Harmonic's plugin default (not the 0.32 m wheel-joint separation).
    wheelbase = float(drive.findtext("wheel_base", "1.0"))
    radius = float(drive.findtext("wheel_radius"))
    Rand.seed(seed)
    start = time.perf_counter()
    # Use a sibling so relative model paths keep their meaning.
    with tempfile.NamedTemporaryFile(suffix=".sdf", dir=world.parent) as f:
        tree.write(f)
        f.flush()
        fixture = gz.TestFixture(f.name)
    parts = {}
    sample = {}

    def pre(_info, ecm):
        if not parts:
            model = gz.Model(gz.World(gz.world_entity(ecm)).model_by_name(ecm, "f1tenth_testbed"))
            parts["base"] = gz.Link(model.link_by_name(ecm, "base_link"))
            parts["base"].enable_velocity_checks(ecm)
            for side in ("left", "right"):
                joint = gz.Joint(model.joint_by_name(ecm, f"front_{side}_wheel_joint"))
                joint.enable_velocity_check(ecm)
                parts[side] = joint

    def post(info, ecm):
        if info.iterations % 25 == 0:
            base = parts["base"]
            pose = base.world_pose(ecm)
            speed = pose.rot().rotate_vector_reverse(base.world_linear_velocity(ecm))
            wheel = (parts["left"].velocity(ecm)[0] + parts["right"].velocity(ecm)[0]) * radius / 2
            sample.update(x=pose.pos().x(), y=pose.pos().y(), yaw=pose.rot().yaw(),
                          body_mps=speed.x(), wheel_mps=wheel,
                          yaw_rps=base.world_angular_velocity(ecm).z())

    fixture.on_pre_update(pre)
    fixture.on_post_update(post)
    fixture.finalize()
    server = fixture.server()
    if not server.run(True, 250, False):
        raise RuntimeError("initial 250 ms settling step failed")
    node = Node()
    publisher = node.advertise("/cmd_vel", Twist)
    deadline = time.monotonic() + 5
    while not publisher.has_connections():
        if time.monotonic() > deadline:
            raise RuntimeError("stock Ackermann plugin did not subscribe")
        time.sleep(0.01)
    rows = [dict(t_s=0.0, **sample)]
    ready = time.perf_counter()
    for i, (v, steer) in enumerate(actions):
        msg = Twist()
        msg.linear.x, msg.angular.z = v, v * math.tan(steer) / wheelbase
        if not publisher.publish(msg) or not server.run(True, 25, False):
            raise RuntimeError(f"command/step failed at tick {i}")
        rows.append(dict(t_s=(i + 1) * 0.025, **sample))
    return rows, dict(cold_start_s=ready - start, step_wall_s=time.perf_counter() - ready,
                      steps=len(actions), seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world", type=Path)
    parser.add_argument("actions", type=Path, help="CSV with speed_mps,steer_rad, one row per tick")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    with args.actions.open() as f:
        actions = [(float(r["speed_mps"]), float(r["steer_rad"])) for r in csv.DictReader(f)]
    if not actions or not all(math.isfinite(x) for pair in actions for x in pair):
        parser.error("a nonempty finite command stream is required")
    rows, stats = run(args.world.resolve(), actions, args.seed)
    with args.output.open("x") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
