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
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, qos_profile_sensor_data

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import Imu, Joy, LaserScan
from std_msgs.msg import String
from vesc_msgs.msg import VescStateStamped

from car_explorer.explorer import (ConfigError, Conductor, Fence, Inputs, Rules,
                                   check_status_topic, path_stop_margin)


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
        # The wall check: car 2's braking delay (tools/run_analysis/braking_lag.py
        # on run 20261111-152132: 0.24 s) and its body (measured 2026-11-23).
        self.declare_parameter('brake_delay', 0.24)
        self.declare_parameter('half_width', 0.135)
        self.declare_parameter('lidar_x', 0.265)
        self.declare_parameter('front_reach', 0.41)
        self.declare_parameter('report_dir', '~/.ros/car_explorer')

        p = lambda k: self.get_parameter(k).value  # noqa: E731
        fence = Fence(**{k: (int(p(k)) if k == 'back_off_steps' else float(p(k)))
                         for k in FENCE_KEYS})
        rules = Rules(**{k: (int(p(k)) if k == 'brake_every' else float(p(k)))
                         for k in RULE_KEYS})
        check_status_topic(p('status_topic'))
        self.conductor = Conductor(fence, rules)
        self.geometry = dict(wheelbase=rules.wheelbase, half_width=float(p('half_width')),
                             lidar_x=float(p('lidar_x')), front_reach=float(p('front_reach')),
                             delay=float(p('brake_delay')))
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
        self.target = target
        self.started = False
        self.in_flight = None
        self.queued = None
        self.create_timer(1.0 / float(p('rate_hz')), self._tick)
        self.get_logger().info(
            f'explorer ready: will tune {target} from {fence.speed_start} m/s, '
            f'{fence.lateral_start} m/s^2; wall margin {rules.wall_margin} m. '
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
    def _tick(self):
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
            safety_state=self._get('safety_state')))
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
        report = self.conductor.report()
        report.update({'last_sent': setting, 'refused': refused, 'session': self.session})
        self.status_pub.publish(String(data=json.dumps(report)))
        self.write_report(report)

    def write_report(self, report=None):
        report = report or self.conductor.report()
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
