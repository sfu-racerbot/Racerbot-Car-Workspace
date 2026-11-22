"""Read-only release check, run on the simulator host with its ROS Python and websockets."""
import asyncio
import json
import struct
import sys
from pathlib import Path

import numpy as np
import websockets
from PIL import Image
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

TOPICS = {"/scan": LaserScan, "/odom": Odometry, "/sim/track_map": OccupancyGrid,
          "/sim/ground_truth_pose": PoseStamped, "/sim/car2/ground_truth_pose": PoseStamped,
          "/tf_static": TFMessage, "/sim_only/ackermann_cmd": AckermannDriveStamped,
          "/sim/status": String}


async def verify(track_png):
    ids, seen, poses, speeds, times, transforms = {}, set(), [], [], [], set()
    info = False
    async with websockets.connect("ws://127.0.0.1:8766",
                                  subprotocols=["foxglove.websocket.v1", "foxglove.sdk.v1"]) as ws:
        print("WebSocket:", ws.subprotocol)
        while True:
            data = await ws.recv()
            if isinstance(data, str):
                msg = json.loads(data)
                if msg["op"] == "serverInfo":
                    assert msg["capabilities"] == ["connectionGraph"], msg["capabilities"]
                    print("Server capabilities:", msg["capabilities"], "(no client writes)")
                    info = True
                if msg["op"] == "advertise":
                    subs = []
                    for c in msg["channels"]:
                        assert c["topic"] in TOPICS.keys() | {"/tf"}
                        if c["topic"] in TOPICS:
                            i = c["id"]
                            ids[i] = c["topic"]
                            subs.append({"id": i, "channelId": i})
                    if subs:
                        await ws.send(json.dumps({"op": "subscribe", "subscriptions": subs}))
                continue
            if data[0] != 1:
                continue
            topic = ids[struct.unpack_from("<I", data, 1)[0]]
            val = deserialize_message(data[13:], TOPICS[topic])
            if topic == "/scan":
                assert len(val.ranges) == 1081 and val.header.frame_id == "laser"
            elif topic == "/sim/status":
                status = json.loads(val.data)
                assert status["track"] == "indoor_oval"
                times.append(status["sim_time_s"])
            elif topic == "/sim/track_map":
                assert val.header.frame_id == "sim_world"
                image = np.asarray(Image.open(track_png).convert("L"))
                expected = np.flipud(np.where(image == 0, 100, 0))
                received = np.array(val.data).reshape(val.info.height, val.info.width)
                assert np.array_equal(expected, received)
                if topic not in seen:
                    print("Match: all", received.size, "map cells, same Gym image; frame sim_world")
            elif topic.endswith("ground_truth_pose"):
                assert val.header.frame_id == "sim_world"
                if topic == "/sim/ground_truth_pose":
                    poses.append((val.pose.position.x, val.pose.position.y))
            elif topic == "/odom":
                assert (val.header.frame_id, val.child_frame_id) == ("odom", "base_link")
                speeds.append(val.twist.twist.linear.x)
            elif topic == "/tf_static":
                transforms.update((t.header.frame_id, t.child_frame_id) for t in val.transforms)
            seen.add(topic)
            if (info and seen == TOPICS.keys() and ("base_link", "laser") in transforms
                    and len(times) > 1 and times[-1] - times[0] >= 2 and max(speeds) > 0.2
                    and abs(poses[-1][0] - poses[0][0]) + abs(poses[-1][1] - poses[0][1]) > 0.25):
                print("Physics time (s):", times[0], "->", times[-1])
                print("Truth XY (m):", poses[0], "->", poses[-1])
                print("Wheel odometry speed range (m/s):", min(speeds), max(speeds))
                print("Scan: 1,081 rays, laser. Static TF:", sorted(transforms))
                print("PASS: read-only live feed incl. command, odometry and both cars' truth poses.")
                return


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(verify(Path(sys.argv[1])), timeout=30))
