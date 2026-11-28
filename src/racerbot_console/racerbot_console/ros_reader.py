"""Live readings from the car's ROS graph. Subscribes only.

The only module in the package that imports rclpy. It runs its own node on
a background thread and keeps just the latest value of each topic, so a
slow browser terminal can never slow a topic down, and the screen thread
never waits on ROS. It creates no publisher, no service and no action
client (test_no_drive_paths.py checks the source).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field

from .status import RateMeter


@dataclass
class Readings:
    """Latest values, each with the monotonic time it arrived."""
    volts: float | None = None
    volts_t: float | None = None
    speed: float | None = None
    speed_t: float | None = None
    buttons: tuple | None = None
    joy_t: float | None = None
    joy_publishers: int | None = None
    imu_state: str | None = None
    scan: RateMeter = field(default_factory=RateMeter)
    core: RateMeter = field(default_factory=RateMeter)
    drive: RateMeter = field(default_factory=RateMeter)
    error: str | None = None


@dataclass(frozen=True)
class Live:
    """A consistent copy of the readings at one moment (`now`, monotonic)."""
    now: float
    volts: float | None = None
    volts_t: float | None = None
    speed: float | None = None
    speed_t: float | None = None
    buttons: tuple | None = None
    joy_t: float | None = None
    joy_publishers: int | None = None
    imu_state: str | None = None
    scan_hz: float | None = None
    core_hz: float | None = None
    drive_hz: float | None = None
    error: str | None = None


class RosReader:
    """Owns one rclpy node on a daemon thread. `readings` is updated in
    place under `lock`; read it with `snapshot()`."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.readings = Readings()
        self._thread = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name='ros-reader', daemon=True)
        self._thread.start()

    def snapshot(self) -> 'Live':
        """Everything the screen needs, worked out under the lock so the ROS
        thread can't change a rate meter while it is being read."""
        with self.lock:
            r = self.readings
            t = self.clock()
            return Live(now=t, volts=r.volts, volts_t=r.volts_t, speed=r.speed,
                        speed_t=r.speed_t, buttons=r.buttons, joy_t=r.joy_t,
                        joy_publishers=r.joy_publishers, imu_state=r.imu_state,
                        scan_hz=r.scan.hz(t), core_hz=r.core.hz(t), drive_hz=r.drive.hz(t),
                        error=r.error)

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                                   qos_profile_sensor_data)
            from ackermann_msgs.msg import AckermannDriveStamped
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import Joy, LaserScan
            from std_msgs.msg import String
            from vesc_msgs.msg import VescStateStamped
        except ImportError as exc:
            with self.lock:
                self.readings.error = (f'ROS is not importable ({exc}); start the menu with '
                                       'tools/console/racerbot, which sources ROS')
            return

        try:
            rclpy.init()
            node = rclpy.create_node('racerbot_console_reader')
        except Exception as exc:  # noqa: BLE001 -- shown on screen, never fatal
            with self.lock:
                self.readings.error = f'could not join the ROS graph: {exc}'
            return

        now = self.clock
        sensor = qos_profile_sensor_data
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)

        def on_core(msg):
            with self.lock:
                self.readings.volts = float(msg.state.voltage_input)
                self.readings.volts_t = now()
                self.readings.core.add(now())

        def on_odom(msg):
            with self.lock:
                self.readings.speed = float(msg.twist.twist.linear.x)
                self.readings.speed_t = now()

        def on_joy(msg):
            with self.lock:
                self.readings.buttons = tuple(msg.buttons)
                self.readings.joy_t = now()

        def on_scan(_msg):
            with self.lock:
                self.readings.scan.add(now())

        def on_drive(_msg):
            with self.lock:
                self.readings.drive.add(now())

        def on_imu(msg):
            try:
                state = json.loads(msg.data).get('state')
            except (ValueError, AttributeError):
                state = 'unreadable'
            with self.lock:
                self.readings.imu_state = state if isinstance(state, str) else 'unreadable'

        node.create_subscription(VescStateStamped, '/sensors/core', on_core, sensor)
        node.create_subscription(Odometry, '/odom', on_odom, sensor)
        node.create_subscription(Joy, '/joy', on_joy, sensor)
        node.create_subscription(LaserScan, '/scan', on_scan, sensor)
        node.create_subscription(AckermannDriveStamped, '/drive', on_drive, sensor)
        node.create_subscription(String, '/imu/status', on_imu, latched)

        def count_joy():
            count = node.count_publishers('/joy')
            with self.lock:
                self.readings.joy_publishers = count

        node.create_timer(1.0, count_joy)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        try:
            while not self._stop.is_set() and rclpy.ok():
                executor.spin_once(timeout_sec=0.2)
        finally:
            executor.shutdown()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
