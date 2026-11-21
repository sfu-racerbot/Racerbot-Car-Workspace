"""Expose the upstream MuSHR one-car scene to an isolated ROS 2 viewer."""

import argparse
import math
import os
import time

import mujoco
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from racerbot_sim.hardware_guard import HardwareInterlock
from rclpy.node import Node
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster


def xyz(field, values):
    field.x, field.y, field.z = map(float, values)


class MuSHRFeed(Node):
    def __init__(self, model_path):
        super().__init__('mushr_feed', namespace='/mujoco')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.body = self.data.body('buddy')
        self.speed = self.data.actuator('buddy_throttle_velocity')
        self.steering = self.data.actuator('buddy_steering_pos')
        self.last_cmd = -math.inf
        self.guard = HardwareInterlock(self, 'MuSHR sensor output', clear_confirm_sec=3.)
        self.odom = self.create_publisher(Odometry, 'odom', 5)
        self.imu = self.create_publisher(Imu, 'imu', 5)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(AckermannDriveStamped, 'command', self.on_command, 1)
        self.create_timer(self.model.opt.timestep, self.tick)

    def on_command(self, msg):
        speed, steer = msg.drive.speed, msg.drive.steering_angle
        if self.guard.blocked or not all(map(math.isfinite, (speed, steer))):
            self.last_cmd = -math.inf
            return
        self.speed.ctrl[0], self.steering.ctrl[0] = speed, steer
        self.last_cmd = time.monotonic()

    def tick(self):
        if not self.guard.safe():
            self.last_cmd = -math.inf
            self.data.ctrl[:] = 0
            return
        if time.monotonic() - self.last_cmd > 0.25:
            self.data.ctrl[:] = 0
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id, odom.child_frame_id = 'mujoco/world', 'mujoco/buddy'
        xyz(odom.pose.pose.position, self.body.xpos)
        quat = odom.pose.pose.orientation
        quat.w, quat.x, quat.y, quat.z = map(float, self.body.xquat)
        twist = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,
                                 self.body.id, twist, 1)
        xyz(odom.twist.twist.angular, twist[:3])
        xyz(odom.twist.twist.linear, twist[3:])
        transform = TransformStamped()
        transform.header, transform.child_frame_id = odom.header, odom.child_frame_id
        xyz(transform.transform.translation, self.body.xpos)
        transform.transform.rotation = quat
        imu = Imu()
        imu.header.stamp = odom.header.stamp
        imu.header.frame_id = 'mujoco/imu'
        imu.orientation_covariance[0] = -1.
        xyz(imu.angular_velocity, self.data.sensor('buddy_gyro').data)
        xyz(imu.linear_acceleration, self.data.sensor('buddy_accelerometer').data)
        mount = TransformStamped()
        mount.header.stamp = odom.header.stamp
        mount.header.frame_id, mount.child_frame_id = odom.child_frame_id, imu.header.frame_id
        site = self.model.site('buddy_imu')
        xyz(mount.transform.translation, site.pos)
        q = mount.transform.rotation
        q.w, q.x, q.y, q.z = map(float, site.quat)
        self.tf.sendTransform([transform, mount])
        self.odom.publish(odom)
        self.imu.publish(imu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='path to upstream models/one_car.xml')
    args = parser.parse_args()
    if not (int(os.environ.get('ROS_DOMAIN_ID', 0)) > 0 and
            (os.environ.get('ROS_LOCALHOST_ONLY') == '1' or
             os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE') == 'LOCALHOST')):
        parser.error('use a dedicated nonzero ROS_DOMAIN_ID and localhost-only discovery')
    rclpy.init()
    node = MuSHRFeed(args.model)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
