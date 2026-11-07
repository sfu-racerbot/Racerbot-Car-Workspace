"""Live-scan seeding against the recorded 20261106-164401 course."""
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_pursuit.seed_start_pose import SeedStartPose, seed_message_fields  # noqa: E402

DATA = Path(__file__).parent / 'data' / 'course_20261106a'
WRONG_MAP = Path(__file__).resolve().parents[3] / 'data/rb2/racerbot_auto/20261030-185036-fixed/map.yaml'


@pytest.fixture
def scans():
    with np.load(DATA / 'scans_20261106-164401.npz') as data:
        yield {key: data[key] for key in data.files}


def args(map_path=DATA / 'map.yaml', line=DATA / 'raceline_raw.csv', give_up=60.0):
    return ['--ros-args', '-p', 'mode:=anywhere', '-p', f'map_yaml:={map_path}',
            '-p', f'waypoints_file:={line}', '-p', 'scan_topic:=/test_only/scan',
            '-p', 'initialpose_topic:=/test_only/initialpose',
            '-p', 'pose_topic:=/test_only/pf_pose', '-p', 'period_sec:=0.05',
            '-p', f'give_up_sec:={give_up}']


def scan(data, index):
    msg = LaserScan()
    msg.ranges = data['ranges'][index].tolist()
    msg.angle_min = float(data['angle_min'])
    msg.angle_increment = float(data['angle_increment'])
    msg.range_max = 30.0
    return msg


@pytest.fixture
def rig():
    nodes = []

    def make(arguments):
        rclpy.init(args=arguments)
        seeder = SeedStartPose()
        helper = rclpy.create_node('anywhere_test_helper')
        nodes.extend([seeder, helper])
        got = []
        helper.create_subscription(PoseWithCovarianceStamped, '/test_only/initialpose', got.append, 10)
        pub = helper.create_publisher(LaserScan, '/test_only/scan', 10)
        return seeder, helper, pub, got

    yield make
    for node in nodes:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def spin(nodes, predicate, timeout, publish=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        if publish:
            publish()
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.02)
    return predicate()


def test_finds_the_car_where_loc_put_it_and_seeds_there(rig, scans):
    seeder, helper, pub, got = rig(args())
    msg = scan(scans, 2)
    assert spin([helper, seeder], lambda: seeder.seeded, 45, lambda: pub.publish(msg))
    spin([helper], lambda: len(got) >= 3, 2)
    # Recorded loc.py base_link pose, run 20261106-164401, scan 2.
    expected = seed_message_fields(*scans['poses'][2], 0.26, 0.0)
    assert len(got) == 3  # seeder repeats specification
    for received in got:
        assert received.pose.pose.position.x == pytest.approx(expected[0], abs=0.15)  # m
        assert received.pose.pose.position.y == pytest.approx(expected[1], abs=0.15)  # m


def test_a_filter_settling_half_a_metre_off_is_not_accepted(rig, scans):
    seeder, helper, pub, got = rig(args() + ['-p', 'confirm_radius:=0.30'])
    pose_pub = helper.create_publisher(PoseStamped, '/test_only/pf_pose', 10)
    msg = scan(scans, 2)
    assert spin([helper, seeder], lambda: seeder.seeded and bool(got), 45,
                lambda: pub.publish(msg))
    assert spin([helper, seeder], lambda: pose_pub.get_subscription_count() > 0, 5)
    seed = got[-1].pose.pose.position
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.pose.orientation.w = 1.0
    pose.pose.position.y = seed.y
    # Closed form: equal y and a 0.50 m x offset give a distance of 0.50 m.
    pose.pose.position.x = seed.x + 0.50
    spin([helper, seeder], lambda: seeder.done, 2, lambda: pose_pub.publish(pose))
    assert not seeder.done
    assert not seeder.succeeded
    # Closed form: equal y and a 0.20 m x offset give a distance of 0.20 m.
    pose.pose.position.x = seed.x + 0.20
    pose_pub.publish(pose)
    assert spin([helper, seeder], lambda: seeder.done, 5)
    assert seeder.succeeded


def test_nothing_is_seeded_before_the_scans_agree(rig, scans):
    seeder, helper, pub, got = rig(args())
    assert spin([helper, seeder], lambda: pub.get_subscription_count() > 0, 5)
    pub.publish(scan(scans, 2))
    assert spin([helper, seeder], lambda: seeder.fix is not None, 45)
    # Spec: moving to scan 4 cannot confirm the scan 2 candidate. The disagreeing scan refuses that
    # candidate; the seeder may then search afresh and confirm the car where it now is, scan 4's
    # recorded pose, if that search finishes inside these 2 s (it does once the suite has warmed up).
    # So: whatever is found or published is scan 4's pose (base_link; laser frame on the wire),
    # never scan 2's, which is 3.9 m away. Recorded poses: loc.py, run 20261106-164401.
    msg = scan(scans, 4)
    spin([helper, seeder], lambda: False, 2, lambda: pub.publish(msg))
    there = scans['poses'][4]
    assert seeder.found_pose is None or math.hypot(
        seeder.found_pose[0] - there[0], seeder.found_pose[1] - there[1]) < 0.15  # m
    wire = seed_message_fields(*there, 0.26, 0.0)
    assert all(math.hypot(r.pose.pose.position.x - wire[0], r.pose.pose.position.y - wire[1]) < 0.15
               for r in got)  # m


def test_a_scan_from_another_course_is_never_seeded(rig, scans):
    seeder, helper, pub, got = rig(args(map_path=WRONG_MAP, give_up=15.0))
    msg = scan(scans, 0)
    assert spin([helper, seeder], lambda: seeder.done, 90, lambda: pub.publish(msg))
    # Spec: mismatched map must be refused without publishing.
    assert not seeder.succeeded and not got
    assert seeder.fix is not None and not seeder.fix.ok


def test_a_car_facing_backwards_is_refused(rig, scans, tmp_path):
    lines = (DATA / 'raceline_raw.csv').read_text().splitlines()
    reversed_line = tmp_path / 'reversed.csv'
    reversed_line.write_text('\n'.join([lines[0], *reversed(lines[1:])]) + '\n')
    seeder, helper, pub, got = rig(args(line=reversed_line, give_up=15.0))
    msg = scan(scans, 2)
    assert spin([helper, seeder], lambda: seeder.done, 90, lambda: pub.publish(msg))
    # Spec: racing line reversed makes the recorded heading backwards.
    assert not seeder.succeeded and not got
    assert seeder.fix is not None and seeder.fix.ok
    assert 'turn it round' in seeder.last_reason


def test_anywhere_needs_a_map_and_a_line():
    # Parameter contract: invalid modes and missing paths are refused at construction.
    for arguments, name in [(['mode:=anywhere'], 'map_yaml'),
                            (['mode:=anywhere', f'map_yaml:={DATA / "map.yaml"}'], 'waypoints_file'),
                            (['mode:=nowhere'], 'mode')]:
        rclpy.init(args=['--ros-args'] + [item for arg in arguments for item in ('-p', arg)])
        try:
            with pytest.raises(ValueError, match=name):
                SeedStartPose()
        finally:
            rclpy.shutdown()
