"""Integration test; run separately on an unused local ROS domain with the upstream model."""

import os
import time

import pytest


def test_ackermann_feed_and_hardware_interlock():
    path = os.environ.get('MUSHR_MODEL')
    if not path:
        pytest.skip('set MUSHR_MODEL to the pinned upstream one_car.xml')
    assert os.environ.get('ROS_DOMAIN_ID') == '79' and (
        os.environ.get('ROS_LOCALHOST_ONLY') == '1' or
        os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE') == 'LOCALHOST')

    from live_feed import AckermannDriveStamped, Imu, MuSHRFeed, Node, Odometry, rclpy
    from rclpy.executors import SingleThreadedExecutor
    from tf2_msgs.msg import TFMessage

    rclpy.init()
    feed = MuSHRFeed(path)
    probe = Node('test_mushr_probe')
    hardware = Node('urg_node')
    executor = SingleThreadedExecutor()
    for node in (feed, probe, hardware):
        executor.add_node(node)
    received, imu, frames = [], [], []
    probe.create_subscription(Odometry, '/mujoco/odom', received.append, 10)
    probe.create_subscription(Imu, '/mujoco/imu', imu.append, 10)
    probe.create_subscription(TFMessage, '/tf', frames.append, 10)
    cmd = probe.create_publisher(AckermannDriveStamped, '/mujoco/command', 1)

    def spin(seconds, action=None):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if action:
                action()
            executor.spin_once(timeout_sec=.02)

    try:
        spin(6.5)
        assert not (received or imu or frames)
        executor.remove_node(hardware)
        hardware.destroy_node()
        spin(4.)
        assert received and imu and frames
        stationary = received[-1].pose.pose.position
        msg = AckermannDriveStamped()
        msg.drive.speed, msg.drive.steering_angle = 1.5, .15
        spin(1.5, lambda: cmd.publish(msg))
        moved = received[-1]
        assert moved.pose.pose.position.x > stationary.x + .45
        assert moved.pose.pose.position.y > stationary.y + .05
        assert moved.twist.twist.linear.x > .6
        assert moved.twist.twist.angular.z > .1 and imu[-1].angular_velocity.z > .1
        assert imu[-1].orientation_covariance[0] == -1.
        assert [(tf.header.frame_id, tf.child_frame_id) for tf in frames[-1].transforms] == [
            ('mujoco/world', 'mujoco/buddy'), ('mujoco/buddy', 'mujoco/imu')]
        spin(2.)
        assert abs(received[-1].twist.twist.linear.x) < .1
    finally:
        executor.shutdown()
        probe.destroy_node()
        feed.destroy_node()
        rclpy.shutdown()
