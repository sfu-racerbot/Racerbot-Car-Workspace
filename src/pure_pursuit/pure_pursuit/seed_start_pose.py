"""seed_start_pose: tell the particle filter where the car is, with no RViz.

In anywhere mode, match live LiDAR scans against the saved map, reject an
ambiguous or backwards fix, and require agreeing scans before seeding.

In start_pose mode, a mapping run (``map_course``) saves where the car stopped
at the end, in the map frame, as start_pose.yaml -- on the racing line it recorded,
measured by SLAM while it was still running. A car left there (or put back
on that spot, facing the same way) is at that pose; race_launch.py passes it
here as x, y, yaw, and this node hands the filter exactly that, as the "2D
Pose Estimate" RViz would have sent. Default (0, 0, 0): the map origin.

The filter spreads its particles 0.5 m and 0.4 rad (about 23 deg) around the
seed (particle_filter.initialize_particles_pose), so "the same spot" means
within a few tens of centimetres and degrees, not millimetres.

Publishes only ``/initialpose``. It cannot move the car and is not driving
code (docs/adding-your-own-code.md). It waits until the filter is listening,
sends the seed ``repeats`` times ``period_sec`` apart (one message can be
lost while the filter is still loading its map), then waits for the filter
to report a pose within ``confirm_radius`` of the seed, and exits 0. Exit 1
means it never did -- no filter, or a filter that ignored the seed.

race_launch.py starts pure_pursuit only after this exits 0. Before that the
filter publishes its own unseeded guess, and in simulation that guess
jumped 6 m; a controller already driving on it steered into a wall.

    ros2 run pure_pursuit seed_start_pose
    ros2 run pure_pursuit seed_start_pose --ros-args -p x:=1.0 -p yaw:=3.14
"""

import math

import numpy as np
import rclpy
import sys

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from pure_pursuit import map_localize, racing_math
from pure_pursuit.occupancy_map import OccupancyMap


def seed_message_fields(x, y, yaw, laser_offset_x, laser_offset_y):
    """(x, y, qz, qw) of the filter's seed for base_link pose (x, y, yaw).

    The filter's particles are LiDAR poses, not base_link poses -- the same
    conversion auto_map_race_node._seed_particle_filter makes.
    """
    lx, ly, lyaw = racing_math.base_link_pose_to_laser(
        x, y, yaw, laser_offset_x, laser_offset_y)
    return lx, ly, math.sin(lyaw / 2.0), math.cos(lyaw / 2.0)


class SeedStartPose(Node):
    def __init__(self):
        super().__init__('seed_start_pose')
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('yaw', 0.0)
        # Keep in step with gap_follow.yaml / the bringup laser transform.
        self.declare_parameter('laser_offset_x', 0.26)
        self.declare_parameter('laser_offset_y', 0.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('initialpose_topic', '/initialpose')
        self.declare_parameter('repeats', 3)
        self.declare_parameter('period_sec', 0.5)
        self.declare_parameter('give_up_sec', 60.0)
        self.declare_parameter('pose_topic', '/pf/viz/inferred_pose')
        self.declare_parameter('confirm_radius', 1.0)
        self.declare_parameter('mode', 'start_pose')
        self.declare_parameter('map_yaml', '')
        self.declare_parameter('waypoints_file', '')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('confirm_scans', 2)
        self.declare_parameter('agree_position_m', 0.10)
        self.declare_parameter('agree_yaw_deg', 3.0)
        self.declare_parameter('max_heading_error_deg', 60.0)
        self.declare_parameter('retry_sec', 2.0)

        def value(name):
            return self.get_parameter(name).value

        self.mode = value('mode')
        if self.mode not in ('start_pose', 'anywhere'):
            raise ValueError(f'invalid mode: {self.mode!r}')
        self.fix = None
        self.found_pose = None
        self.started = self.get_clock().now()
        if self.mode == 'anywhere':
            for name in ('map_yaml', 'waypoints_file'):
                if not value(name):
                    raise ValueError(f'{name} is required in anywhere mode')
            self.field = map_localize.WallField.from_occupancy(
                OccupancyMap.from_yaml(value('map_yaml')))
            self.line = np.loadtxt(value('waypoints_file'), delimiter=',', skiprows=1,
                                   usecols=(0, 1))
            self.laser_x = float(value('laser_offset_x'))
            self.laser_y = float(value('laser_offset_y'))
            self.confirm_scans = int(value('confirm_scans'))
            self.agree_position = float(value('agree_position_m'))
            self.agree_yaw = math.radians(float(value('agree_yaw_deg')))
            self.max_heading_error = math.radians(float(value('max_heading_error_deg')))
            self.retry_sec = float(value('retry_sec'))
            self.latest_scan = None
            self.scan_number = 0
            self.used_scan = 0
            self.last_attempt = None
            self.candidate = None
            self.agreements = 0
            self.last_reason = 'No usable scan was received.'
            self.logged_reasons = set()
            self.create_subscription(LaserScan, value('scan_topic'), self._scan,
                                     qos_profile_sensor_data)
        self.fields = seed_message_fields(
            float(value('x')), float(value('y')), float(value('yaw')),
            float(value('laser_offset_x')), float(value('laser_offset_y')))
        self.map_frame = value('map_frame')
        self.remaining = int(value('repeats'))
        self.give_up_sec = float(value('give_up_sec'))
        self.done = False
        self.succeeded = False
        self.seeded = False
        self.confirm_radius = float(value('confirm_radius'))
        self.create_subscription(PoseStamped, value('pose_topic'), self._pose, 10)
        self.pub = self.create_publisher(
            PoseWithCovarianceStamped, value('initialpose_topic'), 1)
        self.timer = self.create_timer(float(value('period_sec')), self._tick)
        if self.mode == 'start_pose':
            self.get_logger().info(
                f"Waiting for the particle filter to listen on "
                f"'{value('initialpose_topic')}', then seeding it at the mapping "
                f"start pose. The car must be where mapping left it, facing the "
                f"same way.")

    def _scan(self, msg):
        self.latest_scan = msg
        self.scan_number += 1

    def _refuse(self, reason):
        self.last_reason = reason
        if reason not in self.logged_reasons:
            self.get_logger().error(reason)
            self.logged_reasons.add(reason)
        self.candidate = None
        self.agreements = 0

    def _find_pose(self):
        if self.scan_number == self.used_scan:
            return
        msg = self.latest_scan
        if self.candidate is None:
            now = self.get_clock().now()
            if (self.last_attempt is not None and
                    (now - self.last_attempt).nanoseconds / 1e9 < self.retry_sec):
                return
            self.used_scan = self.scan_number
            self.last_attempt = now
            self.fix = map_localize.localize_scan(
                self.field, msg.ranges, msg.angle_min, msg.angle_increment, laser_x=self.laser_x)
            if not self.fix.ok:
                self._refuse(self.fix.reason)
                return
            error = map_localize.line_heading_error(self.line, self.fix.pose)
            if error > self.max_heading_error:
                self._refuse(f'The car faces against the racing direction by '
                             f'{math.degrees(error):.0f} degrees; turn it round.')
                return
            self.candidate = self.fix.pose
        else:
            self.used_scan = self.scan_number
            bx, by = map_localize.scan_points(msg.ranges, msg.angle_min, msg.angle_increment,
                                              laser_x=self.laser_x)
            refined = map_localize.match_scan(self.field, bx, by, self.candidate)
            dx = math.hypot(refined[0] - self.candidate[0], refined[1] - self.candidate[1])
            dyaw = abs(math.atan2(math.sin(refined[2] - self.candidate[2]),
                                  math.cos(refined[2] - self.candidate[2])))
            if (dx >= self.agree_position or dyaw >= self.agree_yaw or
                    map_localize.scan_fit(self.field, bx, by, refined) < map_localize.MIN_GLOBAL_FIT):
                self._refuse('The scans disagree; the car must be stationary.')
                return
            self.agreements += 1
        if self.agreements >= self.confirm_scans:
            self.found_pose = self.candidate
            self.fields = seed_message_fields(*self.found_pose, self.laser_x, self.laser_y)
            self.get_logger().info(f'Found car at {self.found_pose}. {self.fix.reason}')

    def _pose(self, msg):
        """Done once the filter's pose is near the seed: it has taken it."""
        if not self.seeded or self.done:
            return
        x, y, _qz, _qw = self.fields
        if math.hypot(msg.pose.position.x - x, msg.pose.position.y - y) <= self.confirm_radius:
            self.get_logger().info('Particle filter is at the start pose; racing can begin.')
            self.succeeded = True
            self.done = True

    def _tick(self):
        if self.done:
            return
        waited = (self.get_clock().now() - self.started).nanoseconds / 1e9
        if waited > self.give_up_sec:
            if self.mode == 'anywhere' and self.found_pose is None:
                self.get_logger().error(f'The car was never found; giving up. {self.last_reason}')
                self.done = True
                return
            self.get_logger().error(
                'The particle filter never ' +
                ('reported a pose near the seed' if self.seeded else
                 'subscribed to the seed topic') +
                f' in {self.give_up_sec:.0f}s; giving up, and pure_pursuit will '
                'not start. Is localization running, and is this the right map?')
            self.done = True
            return
        if self.mode == 'anywhere' and self.found_pose is None:
            self._find_pose()
            return
        if self.seeded:
            return
        if self.pub.get_subscription_count() == 0:
            return
        x, y, qz, qw = self.fields
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        self.pub.publish(msg)
        self.remaining -= 1
        if self.remaining <= 0:
            self.get_logger().info(
                f'Seeded the particle filter at laser pose ({x:.2f}, {y:.2f}).')
            self.seeded = True


def main(args=None):
    rclpy.init(args=args)
    node = SeedStartPose()
    succeeded = False
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        succeeded = node.succeeded
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if succeeded else 1)


if __name__ == '__main__':
    main()
