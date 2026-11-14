"""Steering servo and command transport lag.

Stock gym turns a steering-angle command into a steering velocity with
``pid_steer``, which despite the name has no proportional term at all::

    sv = sign(target - current) * sv_max

Always full slew rate. At a 25 ms control period and 3.2 rad/s that is a
0.08 rad quantum, so any target off that grid is overshot and the actuator
oscillates forever. Measured on this checkout, commanding a constant 0.15 rad
produced a permanent 0.08 rad (4.58 deg) peak-to-peak limit cycle at 20 Hz,
and 0.26 rad commands actually reached 0.32 rad -- past the controller's own
clamp, and past this car's real right-hand steering limit. See F6 in
``docs/sim-fidelity-audit.md``; it is also why ``max_steering_rate: 1.0``
appeared to help in earlier results.

The audit's suggested fix was a 5 ms control step, which reduces the limit
cycle five-fold but does not remove it and costs 3.1x in run time. Gym exposes
a better route: ``SteerActionType.STEERING_SPEED`` hands the steering velocity
straight to the plant, so the actuator model can live here instead. A real
hobby servo runs a proportional position loop against a slew ceiling, which is
what :class:`SteeringServo` implements -- more faithful than bang-bang, free
of the limit cycle, and free in run time.
"""

from __future__ import annotations

import math

import numpy as np
from collections import deque
from dataclasses import dataclass


@dataclass
class SteeringServo:
    """Proportional position loop with a slew-rate ceiling.

    ``gain`` is the loop's proportional term in 1/s. The discrete update is
    ``delta_next = delta + gain * (target - delta) * dt``, so ``gain * dt``
    must stay below 1 for a monotonic approach; at 40 Hz that caps gain at 40
    and the default of 25 leaves useful margin.
    """

    gain: float
    rate_max: float
    angle_min: float
    angle_max: float

    def velocity(self, target: float, current: float) -> float:
        """Steering velocity to command, rad/s."""
        target = min(max(target, self.angle_min), self.angle_max)
        demand = self.gain * (target - current)
        return min(max(demand, -self.rate_max), self.rate_max)


class TransportDelay:
    """Fixed whole-tick FIFO lag on a scalar command.

    Models the path from a command being published to the actuator acting on
    it: ROS transport, USB serial, VESC firmware, drivetrain lash. Gym's own
    ``steer_delay_steps`` delays the actuator's *output*; delaying the
    *command* here is the physical ordering, and it lets the servo above see
    the same staleness the real one does.
    """

    def __init__(self, ticks: int, initial: float = 0.0):
        if ticks < 0:
            raise ValueError(f"ticks must be >= 0, got {ticks}")
        self.ticks = ticks
        self.initial = initial
        self._queue: deque[float] = deque()
        self.reset()

    def reset(self) -> None:
        self._queue = deque([self.initial] * self.ticks, maxlen=self.ticks + 1)

    def push(self, value: float) -> float:
        """Accept a new command, return the one that takes effect now."""
        if self.ticks == 0:
            return value
        self._queue.append(value)
        return self._queue.popleft()


def speed_loop_accel_ceiling(accel_max: float, fade_from: float, no_load: float, speed: float) -> float:
    """The speed loop's acceleration ceiling at ``speed``: ``accel_max`` up to ``fade_from``, then falling
    linearly to zero at ``no_load``, as an electric motor's force falls with speed.

    Car 2 was measured traction-limited (4 m/s^2, no fall-off) up to the 5.4 m/s it reached on 2026-11-06.
    Above that the drive force of a DC motor falls linearly towards its no-load speed (an estimate for this
    motor and battery, not a measurement), so a raised VESC speed limit is not simulated as free
    acceleration. ``fade_from`` infinite (the default calibration) returns ``accel_max`` unchanged.
    """
    if not math.isfinite(fade_from) or speed <= fade_from:
        return accel_max
    if not (math.isfinite(no_load) and no_load > fade_from):
        raise ValueError("no_load must be finite and above fade_from")
    return accel_max * max(no_load - speed, 0.0) / (no_load - fade_from)


def accel_table_ceiling(table, speed: float) -> float:
    """The speed loop's acceleration ceiling at wheel ``speed`` from a measured table ((speed, accel), ...):
    linear between points, flat beyond the ends. Speeds strictly ascending, accelerations finite and positive,
    at least two points (opt-in per floor: ``Calibration.accel_by_speed``)."""
    rows = [tuple(row) for row in table]
    if len(rows) < 2:
        raise ValueError("an acceleration table needs at least two points")
    if not all(math.isfinite(x) for row in rows for x in row):
        raise ValueError("an acceleration table must be finite")
    if any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise ValueError("acceleration table speeds must be strictly ascending")
    if any(row[1] <= 0.0 for row in rows):
        raise ValueError("acceleration table values must be positive")
    return float(np.interp(speed, [row[0] for row in rows], [row[1] for row in rows]))


def speed_loop_accel(cal, command: float, measured: float) -> float:
    """The acceleration (m/s^2) the car's speed loop delivers toward ``command`` from wheel speed ``measured``:
    proportional each way with its own gain and limit. Accelerating, the ceiling is the measured table
    (``cal.accel_by_speed``) or ``speed_loop_accel_max`` with its fade-off. Braking starts only once the speed
    is more than ``speed_loop_brake_deadband`` above the command: rb1's VESC coasts through a 0.2-0.5 m/s
    excess (median -0.01 m/s^2, tools/run_analysis/results/2026-11-13-corner-entry-study.md). Pure: the plant
    (``FidelityPlant._measured_speed_loop``) and the fitting tool (tools/run_analysis/fit_speed_loop.py)
    both call this one function."""
    diff = command - measured
    if diff >= 0.0:
        if cal.accel_by_speed:
            ceiling = accel_table_ceiling(cal.accel_by_speed, measured * cal.odom_speed_scale)
        else:
            ceiling = speed_loop_accel_ceiling(cal.speed_loop_accel_max, cal.speed_loop_accel_fade_from,
                                               cal.speed_loop_accel_no_load, measured)
        return min(cal.speed_loop_accel_gain * diff, ceiling)
    excess = max(0.0, -diff - cal.speed_loop_brake_deadband)
    return max(-cal.speed_loop_brake_gain * excess, -cal.speed_loop_brake_max)
