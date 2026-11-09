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
