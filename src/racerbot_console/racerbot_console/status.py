"""What the status bar and the Car panel show, worked out from raw readings.

No rclpy and no Textual here, so every rule can be tested on its own. The
ROS reader (ros_reader.py) and the system reader (sysinfo.py) only collect
numbers; this module decides what they mean.

Nothing here is a safety device. The LB line tells the person at the
terminal what the car can see; the driving nodes enforce the deadman
themselves (docs/architecture.md, workspace policy).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from racerbot_doctor import car as doctor_car


class Level(Enum):
    OK = 'ok'
    WARN = 'warn'
    FAIL = 'fail'
    OFF = 'off'          # not running / no reading


@dataclass(frozen=True)
class Shown:
    """One value as shown on screen: its level and its words."""
    level: Level
    text: str


# Seconds after which a reading is no longer "live". /sensors/core comes at
# 50 Hz (racerbot_doctor car.VESC_CORE_MIN_HZ cites vesc_driver.cpp), /joy at
# 20 Hz autorepeat (joy_teleop.yaml), /odom with /sensors/core, /scan at
# 40 Hz. One second is 20 or more missed messages for each of them.
STALE_SEC = 1.0


def is_fresh(stamp: float | None, now: float, timeout: float = STALE_SEC) -> bool:
    """True if a reading taken at `stamp` (monotonic seconds) is still live."""
    if stamp is None or not math.isfinite(stamp) or not math.isfinite(now):
        return False
    age = now - stamp
    return 0.0 <= age <= timeout


def battery(volts: float | None) -> Shown:
    """Battery level from /sensors/core's voltage_input, with the doctor's
    3S thresholds (racerbot_doctor/car.py), so the menu and the doctor agree."""
    if volts is None or not math.isfinite(volts):
        return Shown(Level.OFF, 'no reading')
    if volts < doctor_car.BATTERY_ABSENT_V:
        return Shown(Level.OFF, f'{volts:.1f} V: battery off (VESC on USB only)')
    cell = volts / doctor_car.BATTERY_CELLS
    words = f'{volts:.2f} V {cell:.2f}/cell'
    if volts < doctor_car.BATTERY_FAIL_V:
        return Shown(Level.FAIL, words + ' CHARGE NOW')
    if volts < doctor_car.BATTERY_WARN_V:
        return Shown(Level.WARN, words + ' low')
    if volts > doctor_car.BATTERY_HIGH_V:
        return Shown(Level.WARN, words + ' above a full charge')
    return Shown(Level.OK, words)


def lb(buttons: tuple | list | None, stamp: float | None, now: float,
       button: int = doctor_car.JOY_LB_BUTTON) -> Shown:
    """What /joy says about LB. Display only: never used to allow driving."""
    if buttons is None or stamp is None:
        return Shown(Level.OFF, 'no controller')
    if not is_fresh(stamp, now):
        return Shown(Level.WARN, 'controller silent')
    if button < 0 or button >= len(buttons):
        return Shown(Level.WARN, 'no LB button in /joy')
    if buttons[button] == 1:
        return Shown(Level.OK, 'HELD')
    return Shown(Level.OFF, 'released')


def joy_publishers(count: int | None) -> Shown:
    """One /joy publisher per car. Two means the cars share a ROS graph and
    one car's LB could drive the other (docs/launchstart.md, first run)."""
    if count is None:
        return Shown(Level.OFF, '/joy ?')
    if count == 1:
        return Shown(Level.OK, '/joy 1 pub')
    if count == 0:
        return Shown(Level.OFF, '/joy 0 pub')
    return Shown(Level.FAIL, f'/joy {count} pubs: cars share a network')


def rate(hz: float | None, min_hz: float, name: str) -> Shown:
    """A topic's rate against the doctor's minimum for it."""
    if hz is None or not math.isfinite(hz) or hz <= 0.0:
        return Shown(Level.OFF, f'{name} off')
    level = Level.OK if hz >= min_hz else Level.WARN
    return Shown(level, f'{name} {hz:.0f} Hz')


def speed(mps: float | None, stamp: float | None, now: float) -> Shown:
    if mps is None or not math.isfinite(mps) or not is_fresh(stamp, now):
        return Shown(Level.OFF, 'speed -')
    return Shown(Level.OK, f'{mps:+.2f} m/s')


def imu(state: str | None) -> Shown:
    """/imu/status's JSON 'state' (imu_bridge/imu_health.py)."""
    if state is None:
        return Shown(Level.OFF, 'IMU off')
    return {
        'ok': Shown(Level.OK, 'IMU ok'),
        'waiting': Shown(Level.WARN, 'IMU waiting'),
        'fault': Shown(Level.FAIL, 'IMU fault'),
    }.get(state, Shown(Level.WARN, f'IMU {state}'))


@dataclass
class RateMeter:
    """Message rate over the last `window` seconds of arrivals."""
    window: float = 2.0
    _times: deque = field(default_factory=lambda: deque(maxlen=400))

    def add(self, t: float) -> None:
        if math.isfinite(t):
            self._times.append(t)

    def hz(self, now: float) -> float | None:
        recent = [t for t in self._times if 0.0 <= now - t <= self.window]
        if len(recent) < 2 or now - recent[-1] > STALE_SEC:
            return None
        span = recent[-1] - recent[0]
        if span <= 0.0:
            return None
        return (len(recent) - 1) / span
