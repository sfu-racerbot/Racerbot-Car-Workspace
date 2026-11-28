"""explorer_node: steps gap_follow's speed and cornering settings up between
stretches of clean driving, and backs them off when the car shows its limit.

It publishes no drive topic and cannot move the car. It only calls
gap_follow_node's ``set_parameters`` service (the dashboard's live-tuning
path) with ``max_speed`` and ``max_lateral_accel``. gap_follow still needs
LB held, and keeps all its own braking. The decisions are in explorer.py
(no rclpy); this file only wires topics to them. docs/explorer.md.
"""

from __future__ import annotations

import json
import math
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.srv import GetParameters, SetParameters
from sensor_msgs.msg import Imu, Joy, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from vesc_msgs.msg import VescStateStamped

from car_explorer.explorer import (MIN_SPEED, ConfigError, Conductor, Fence, Inputs, Rules,
                                   check_status_topic, path_stop_margin, straight_room)


FENCE_KEYS = ('speed_start', 'speed_step', 'speed_top', 'lateral_start', 'lateral_step',
              'lateral_top', 'back_off_steps')
RULE_KEYS = tuple(Rules.__dataclass_fields__)


class ExplorerNode(Node):
    def __init__(self):
        super().__init__('car_explorer')
        defaults_f, defaults_r = Fence(), Rules()
        for key in FENCE_KEYS:
            self.declare_parameter(key, getattr(defaults_f, key))
        for key in RULE_KEYS:
            self.declare_parameter(key, getattr(defaults_r, key))
        self.declare_parameter('target_node', '/gap_follow_node')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('intent_topic', '/drive_intent')
        self.declare_parameter('status_topic', '/car_explorer/status')
        self.declare_parameter('deadman_button', 4)
        self.declare_parameter('joy_timeout_sec', 0.5)
        self.declare_parameter('input_timeout_sec', 0.5)
        self.declare_parameter('rate_hz', 20.0)
        # The wall check: car 2's body (measured 2026-11-23; front_reach bounded,
        # see config/car_explorer.yaml). The braking delay is a rule (brake_delay).
        self.declare_parameter('half_width', 0.135)
        self.declare_parameter('lidar_x', 0.265)
        self.declare_parameter('front_reach', 0.445)
        self.declare_parameter('pose_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('report_dir', '~/.ros/car_explorer')

        p = lambda k: self.get_parameter(k).value  # noqa: E731
        fence = Fence(**{k: (int(p(k)) if k == 'back_off_steps' else float(p(k)))
                         for k in FENCE_KEYS})
        rules = Rules(**{k: float(p(k)) for k in RULE_KEYS})
        check_status_topic(p('status_topic'))
        self.fence, self.rules = fence, rules
        # Built once gap_follow's own min_speed is known (stop tests lower it).
        self.conductor = None
        self.geometry = dict(wheelbase=rules.wheelbase, half_width=float(p('half_width')),
                             lidar_x=float(p('lidar_x')), front_reach=float(p('front_reach')),
                             delay=rules.brake_delay)
        self.pose_frame, self.base_frame = p('pose_frame'), p('base_frame')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.deadman_button = int(p('deadman_button'))
        self.joy_timeout = float(p('joy_timeout_sec'))
        self.input_timeout = float(p('input_timeout_sec'))
        self.report_dir = os.path.expanduser(p('report_dir'))
        self.session = time.strftime('%Y%m%d-%H%M%S')

        self.latest = {}  # name -> (monotonic stamp, value)
        self.lb = (None, False)
        self.create_subscription(LaserScan, p('scan_topic'), self._scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, p('odom_topic'), self._odom, 10)
        self.create_subscription(Imu, p('imu_topic'), self._imu, qos_profile_sensor_data)
        self.create_subscription(AckermannDriveStamped, p('drive_topic'), self._drive, 10)
        self.create_subscription(String, p('intent_topic'), self._intent, 10)
        self.create_subscription(Joy, '/joy', self._joy, 10)
        self.create_subscription(VescStateStamped, '/sensors/core', self._core, 10)
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, p('status_topic'), latched)

        target = p('target_node').rstrip('/')
        self.client = self.create_client(SetParameters, f'{target}/set_parameters')
        self.getter = self.create_client(GetParameters, f'{target}/get_parameters')
        self.target = target
        self.asked_baseline = False
        self.started = False
        self.in_flight = None
        self.queued = None
        self.create_timer(1.0 / float(p('rate_hz')), self._tick)
        self.get_logger().info(
            f'explorer ready: will tune {target} from {fence.speed_start} m/s, '
            f'{fence.lateral_start} m/s^2; wall margin {rules.wall_margin} m; course map in '
            f'{self.pose_frame}. '
            'It never drives: hold LB as usual.')

    # --- inputs -------------------------------------------------------------------------
    def _now(self):
        return time.monotonic()

    def _put(self, name, value):
        self.latest[name] = (self._now(), value)

    def _get(self, name):
        stamp, value = self.latest.get(name, (None, None))
        if stamp is None or self._now() - stamp > self.input_timeout:
            return None
        return value

    def _scan(self, msg):
        g = self.geometry
        try:
            room = straight_room(msg.ranges, msg.angle_min, msg.angle_increment,
                                 half_width=g['half_width'], lidar_x=g['lidar_x'],
                                 front_reach=g['front_reach'])
        except ConfigError:
            room = None
        self._put('straight_room', room)
        speed = self._get('speed')
        steering = self._get('steering')
        if speed is None or steering is None:
            self._put('stop_margin', None)
            return
        try:
            margin = path_stop_margin(msg.ranges, msg.angle_min, msg.angle_increment,
                                      speed=speed, steering=steering, **self.geometry)
        except ConfigError:
            margin = None
        self._put('stop_margin', margin)

    def _odom(self, msg):
        self._put('speed', float(msg.twist.twist.linear.x))

    def _imu(self, msg):
        self._put('yaw_rate', float(msg.angular_velocity.z))

    def _drive(self, msg):
        self._put('steering', float(msg.drive.steering_angle))
        self._put('speed_cmd', float(msg.drive.speed))

    def _pose(self):
        """(x, y, yaw) of the car in the pose frame, or None if SLAM has not
        published it lately. The course map needs a fixed frame; without one
        the explorer runs on the live scan alone."""
        try:
            tf = self.tf_buffer.lookup_transform(self.pose_frame, self.base_frame, Time())
        except TransformException:
            return None
        age = (self.get_clock().now() - Time.from_msg(tf.header.stamp)).nanoseconds * 1e-9
        if age > self.input_timeout:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (float(t.x), float(t.y), yaw)

    def _intent(self, msg):
        try:
            state = json.loads(msg.data).get('state')
        except (ValueError, AttributeError):
            state = None
        self._put('safety_state', state if isinstance(state, str) else None)

    def _joy(self, msg):
        held = (0 <= self.deadman_button < len(msg.buttons)
                and msg.buttons[self.deadman_button] == 1)
        self.lb = (self._now(), held)

    def _core(self, msg):
        self._put('voltage', float(msg.state.voltage_input))

    def _lb_held(self):
        stamp, held = self.lb
        return bool(held) and stamp is not None and self._now() - stamp <= self.joy_timeout

    # --- the loop -----------------------------------------------------------------------
    def _ask_baseline(self):
        request = GetParameters.Request()
        request.names = [MIN_SPEED]
        self.getter.call_async(request).add_done_callback(self._got_baseline)

    def _got_baseline(self, future):
        baseline = {}
        try:
            value = future.result().values[0]
            baseline[MIN_SPEED] = float(value.double_value)
        except Exception as exc:  # noqa: BLE001 - no baseline only means no stop tests
            self.get_logger().warning(f'could not read {self.target} min_speed ({exc}); '
                                      'no stop tests this session')
        self.conductor = Conductor(self.fence, self.rules, baseline)

    def _tick(self):
        if self.conductor is None:
            if not self.asked_baseline and self.getter.service_is_ready():
                self.asked_baseline = True
                self._ask_baseline()
            return
        if not self.started:
            if not self.client.service_is_ready():
                return
            self._send(self.conductor.start())
            self.started = True
            return
        out = self.conductor.update(Inputs(
            now=self._now(), lb_held=self._lb_held(), speed=self._get('speed'),
            steering=self._get('steering'), yaw_rate=self._get('yaw_rate'),
            stop_margin=self._get('stop_margin'), voltage=self._get('voltage'),
            safety_state=self._get('safety_state'), speed_cmd=self._get('speed_cmd'),
            pose=self._pose(), straight_room=self._get('straight_room')))
        if out is not None:
            self._send(out)

    def _send(self, setting):
        """One request at a time; a newer setting replaces a queued one, so the
        latest decision (a back-off above all) is always the one that lands."""
        if self.in_flight is not None:
            self.queued = setting
            return
        request = SetParameters.Request()
        request.parameters = [Parameter(k, value=float(v)).to_parameter_msg()
                              for k, v in setting.items()]
        self.in_flight = self.client.call_async(request)
        self.in_flight.add_done_callback(lambda f, s=dict(setting): self._sent(f, s))

    def _sent(self, future, setting):
        self.in_flight = None
        try:
            results = future.result().results
            refused = [r.reason for r in results if not r.successful]
        except Exception as exc:  # noqa: BLE001 - a failed call is reported, never raised
            refused = [str(exc)]
        if refused:
            self.get_logger().error(f'{self.target} refused {setting}: {refused}')
        else:
            self.get_logger().info(f'applied {setting}')
        self._publish_status(setting, refused)
        if self.queued is not None:
            nxt, self.queued = self.queued, None
            self._send(nxt)

    def _publish_status(self, setting, refused):
        report = self.conductor.report() if self.conductor else {}
        report.update({'last_sent': setting, 'refused': refused, 'session': self.session})
        self.status_pub.publish(String(data=json.dumps(report)))
        self.write_report(report)

    def write_report(self, report=None):
        if report is None:
            if self.conductor is None:
                return
            report = self.conductor.report()
        try:
            os.makedirs(self.report_dir, exist_ok=True)
            path = os.path.join(self.report_dir, f'{self.session}.json')
            with open(path, 'w') as fh:
                json.dump(report, fh, indent=1)
        except OSError as exc:
            self.get_logger().warning(f'could not write the session report: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = ExplorerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
