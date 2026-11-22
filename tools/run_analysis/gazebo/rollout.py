"""One fresh Gazebo fixture for one blind action block, no ROS graph or targets."""
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
from replay import command_yaw_rate, read_actions

os.environ["GZ_PARTITION"] = "racerbot_bench_" + uuid.uuid4().hex
os.environ["GZ_IP"] = "127.0.0.1"
from gz.math7 import Pose3d, Rand, Vector3d
from gz.msgs10.twist_pb2 import Twist
from gz.sim8 import TestFixture, World, Model, Link, Joint, world_entity
from gz.transport13 import Node
import fixture_bridge


def replay(world, actions, v0):
    if len(actions) != 480 or not all(math.isfinite(v) and math.isfinite(d) for v, d in actions):
        raise ValueError("need 480 finite speed/steer pairs")
    if not math.isfinite(v0):
        raise ValueError("invalid initial speed")
    xml = ET.parse(world)
    drive = xml.find(".//model[@name='f1tenth_testbed']/plugin[@name='gz::sim::systems::AckermannSteering']")
    radius = float(drive.findtext("wheel_radius"))
    wheelbase = float(drive.findtext("wheel_base", "1.0"))
    if float(xml.findtext("./world/physics/max_step_size")) != .001:
        raise ValueError("need stock 1 ms physics step")
    names = [m.get("name") for m in xml.findall("./world/model")]
    if xml.findall("./world/include") or sorted(names) != ["f1tenth_testbed", "ground_plane"]:
        raise ValueError("use the stock open-floor world, not a mapped course")
    xml.find("./world/physics/real_time_factor").text = "0"
    xml.find("./world/physics/real_time_update_rate").text = "0"
    Rand.seed(0)
    with tempfile.NamedTemporaryFile(suffix=".sdf", dir=world.parent) as f:
        xml.write(f)
        f.flush()
        fixture = TestFixture(f.name)
    parts = {}
    state = {}
    stage = "settle"
    count = 0
    rows = []

    def pre(info, ecm):
        if not parts:
            model = Model(World(world_entity(ecm)).model_by_name(ecm, "f1tenth_testbed"))
            base = Link(model.link_by_name(ecm, "base_link"))
            base.enable_velocity_checks(ecm)
            parts.update(model=model, base=base, wheels=[], steer=[])
            for side in ("left", "right"):
                joint = Joint(model.joint_by_name(ecm, f"front_{side}_steer_joint"))
                joint.enable_position_check(ecm)
                parts["steer"].append(joint)
                for end in ("front", "rear"):
                    wheel = Joint(model.joint_by_name(ecm, f"{end}_{side}_wheel_joint"))
                    wheel.enable_velocity_check(ecm)
                    parts["wheels"].append(wheel)
        if stage == "reset":
            base = parts["base"]
            parts["model"].set_world_pose_cmd(ecm, Pose3d(0, 0, state["z"], 0, 0, 0))
            for joint in parts["steer"]:
                joint.reset_position(ecm, [0.0])
                joint.reset_velocity(ecm, [0.0])
            for joint in parts["wheels"]:
                joint.reset_velocity(ecm, [v0 / radius])
            base.set_linear_velocity(ecm, Vector3d(v0, 0, 0))
            base.set_angular_velocity(ecm, Vector3d(0, 0, 0))
        if stage == "run" and count == 0:
            if not fixture_bridge.release_velocity(ecm, parts["base"].entity()):
                raise RuntimeError("could not release velocity reset")
        if stage == "run" and count % 25 == 0:
            v, d = actions[count // 25]
            cmd = Twist()
            cmd.linear.x, cmd.angular.z = v, command_yaw_rate(v, d, wheelbase)
            if not publisher.publish(cmd):
                raise RuntimeError("could not publish action")

    def post(info, ecm):
        nonlocal count
        if stage == "run":
            count += 1
            if count % 25:
                return
        base = parts["base"]
        pose = base.world_pose(ecm)
        v = pose.rot().rotate_vector_reverse(base.world_linear_velocity(ecm))
        w = pose.rot().rotate_vector_reverse(base.world_angular_velocity(ecm))
        state.update(x=pose.pos().x(), y=pose.pos().y(), z=pose.pos().z(), yaw=pose.rot().yaw(),
                     body_mps=v.x(), body_yaw_rps=w.z())
        if stage == "reset":
            for i, joint in enumerate(parts["steer"]):
                state["steer" + str(i)] = joint.position(ecm)[0]
            for i, joint in enumerate(parts["wheels"]):
                state["wheel" + str(i)] = joint.velocity(ecm)[0] * radius
        if stage == "run":
            rows.append((count // 25 - 1, state["body_mps"], state["body_yaw_rps"]))

    fixture.on_pre_update(pre)
    fixture.on_post_update(post)
    fixture.finalize()
    server = fixture.server()
    if not server.run(True, 250, False):
        raise RuntimeError("failed to settle")
    node = Node()
    publisher = node.advertise("/cmd_vel", Twist)
    deadline = time.monotonic() + 5
    while not publisher.has_connections():
        if time.monotonic() > deadline:
            raise TimeoutError("no Ackermann subscriber")
        time.sleep(.005)
    # Seed the stock actuator's own command history without changing its gains/limits.
    cmd = Twist()
    cmd.linear.x = v0
    if not publisher.publish(cmd) or not server.run(True, 100 + math.ceil(abs(v0) / 10 / .001), False):
        raise RuntimeError("failed to prime actuator")
    stage = "reset"
    if not fixture_bridge.run_paused_once(server):
        raise RuntimeError("paused reset failed")
    initial = dict(state)
    print(json.dumps(dict(initial=initial, requested=v0)), flush=True)
    # Native readback at t=0, before the first scored or warm-up physics tick.
    for k, target in dict(x=0, y=0, yaw=0, body_mps=v0, body_yaw_rps=0,
                          steer0=0, steer1=0, **{f"wheel{i}": v0 for i in range(4)}).items():
        if not math.isclose(initial[k], target, rel_tol=0, abs_tol=1e-6):
            raise RuntimeError(f"unsupported reset: {k}={initial[k]}, expected {target}")
    stage = "run"
    if not server.run(True, 480 * 25, False):
        raise RuntimeError("failed to step 12 s")
    if len(rows) != 480 or not all(math.isfinite(v) and math.isfinite(w) for _, v, w in rows):
        raise RuntimeError("incomplete rollout")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("world", type=Path)
    p.add_argument("actions", type=Path)
    p.add_argument("v0", type=float)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    with a.actions.open(encoding="utf-8", newline="") as f:
        commands = read_actions(f)
    t = time.monotonic()
    data = replay(a.world.resolve(), commands, a.v0)
    with a.out.open("x") as f:
        w = csv.writer(f)
        w.writerow(["step", "body_mps", "body_yaw_rps"])
        w.writerows(data)
    print(json.dumps(dict(wall_s=time.monotonic() - t, steps=len(data))))
