"""A friction circle for a plant that does not have one.

The single-track model in F1TENTH Gym uses a purely linear tyre: lateral force
is ``F_y = C_alpha * alpha`` with no upper bound, so the simulated car can
generate unlimited cornering force. Measured on this workspace's checkout, it
sustained 17.74 m/s^2 (1.81 g) at full lock and never plateaued -- see F2 in
``docs/sim-fidelity-audit.md``. Nor does the model couple longitudinal and
lateral force, so braking hard mid-corner costs nothing (F4).

Both gaps are structural: they live inside upstream's numba-compiled
derivative function, which we keep pristine. So the envelope is imposed from
outside instead, by limiting the steering angle handed to the plant to what
the floor could actually support at the current speed and braking effort.

    F_x^2 + F_y^2  <=  (mu * F_z)^2

Be clear about what this buys. It reproduces *the envelope* -- the car can no
longer corner harder than physics allows, and spending grip on braking now
costs cornering. It does not reproduce *loss of control*: at the limit the
simulated car understeers along the envelope rather than sliding or spinning.
That is still a large improvement over an infinite-grip car, and, because
:class:`GripEnvelope` counts every tick it intervenes, the limit becomes
visible in the results instead of silently flattering the controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# Below this speed the single-track model falls back to a kinematic bicycle
# (single_track.py: `if V < 0.5`), where slip angles and hence grip limits are
# not modelled at all. Clamping there would be meaningless.
KINEMATIC_SPEED = 0.5


@dataclass
class GripEnvelope:
    """Steering limiter enforcing a friction circle, and its own audit trail.

    Attributes:
        friction_accel_limit: mu * g, the radius of the friction circle.
        wheelbase: metres, for the steady-state cornering inversion.
        limited_steps: ticks on which the clamp actually bound.
        max_demanded_lateral: hardest cornering asked for, m/s^2.
        max_allowed_lateral: hardest cornering permitted, m/s^2.
    """

    friction_accel_limit: float
    wheelbase: float
    # How much of the longitudinal demand competes for grip, 0 to 1. Full
    # coupling is the honest ellipse, but it interacts badly with upstream's
    # longitudinal controller: pid_accl answers any speed error at all with a
    # demand that clips to a_max, so the plant is almost always braking at
    # full authority and the lateral budget collapses to zero far more often
    # than on a real car. Setting this below 1 models a drivetrain that does
    # not put the whole friction circle into the longitudinal axis. See the
    # tuning note in tools/f1tenth_sim/README.md.
    longitudinal_coupling: float = 1.0
    limited_steps: int = 0
    total_steps: int = 0
    max_demanded_lateral: float = 0.0
    max_allowed_lateral: float = 0.0
    min_lateral_budget: float = math.inf

    def lateral_budget(self, longitudinal_accel: float) -> float:
        """Cornering left over after longitudinal force takes its share.

        The friction circle solved for its lateral component. Longitudinal
        demand is given priority, which is the usual approximation and the
        conservative one here: it is what makes trail-braking cost grip.
        """
        limit = self.friction_accel_limit
        used = longitudinal_accel * self.longitudinal_coupling
        remaining = limit * limit - used * used
        budget = math.sqrt(remaining) if remaining > 0.0 else 0.0
        self.min_lateral_budget = min(self.min_lateral_budget, budget)
        return budget

    def max_steering(self, speed: float, lateral_budget: float) -> float:
        """Largest steering angle whose steady-state corner fits the budget.

        For a bicycle model at steady state the path radius is
        ``R = L / tan(delta)`` and lateral acceleration is ``v^2 / R``, so
        ``a_lat = v^2 * tan(delta) / L``. Inverting for ``a_lat <= budget``
        gives ``delta <= atan(L * budget / v^2)``.
        """
        if speed < KINEMATIC_SPEED:
            return math.inf
        return math.atan(self.wheelbase * lateral_budget / (speed * speed))

    def apply(
        self,
        steering: float,
        speed: float,
        longitudinal_accel: float,
    ) -> float:
        """Clamp ``steering`` to the friction circle, recording what happened."""
        self.total_steps += 1
        budget = self.lateral_budget(longitudinal_accel)
        limit = self.max_steering(speed, budget)

        demanded = abs(math.tan(steering)) * speed * speed / self.wheelbase
        self.max_demanded_lateral = max(self.max_demanded_lateral, demanded)

        if not math.isfinite(limit) or abs(steering) <= limit:
            self.max_allowed_lateral = max(self.max_allowed_lateral, demanded)
            return steering

        self.limited_steps += 1
        clamped = math.copysign(limit, steering)
        allowed = abs(math.tan(clamped)) * speed * speed / self.wheelbase
        self.max_allowed_lateral = max(self.max_allowed_lateral, allowed)
        return clamped

    def report(self) -> dict:
        """Metrics for the scenario result, in g as well as SI."""
        limit = self.friction_accel_limit
        return {
            "grip_limit_mps2": round(limit, 3),
            "grip_limit_g": round(limit / 9.81, 3),
            "grip_limited_steps": self.limited_steps,
            "grip_limited_fraction": (
                round(self.limited_steps / self.total_steps, 4)
                if self.total_steps
                else 0.0
            ),
            "max_demanded_lateral_mps2": round(self.max_demanded_lateral, 3),
            "max_allowed_lateral_mps2": round(self.max_allowed_lateral, 3),
            "min_lateral_budget_mps2": (
                round(self.min_lateral_budget, 3)
                if math.isfinite(self.min_lateral_budget)
                else None
            ),
        }


@dataclass
class ProgressiveGrip:
    """Tyres that roll off gradually, as the real car's do, instead of a sharp limit.

    ``curve`` is ((asked, delivered), ...) in m/s^2: the lateral acceleration the rack
    asks for at this speed, and what the tyres deliver for it. Measured on 2026-10-30,
    the car follows the request, then keeps gaining ever more slowly: there is no one
    friction number. Measured too: braking and accelerating cost it almost nothing
    (2.7 m/s^2 achieved from 3 m/s^2 of braking through 1.5 of acceleration), so unlike
    GripEnvelope this takes no share for longitudinal force.

    Same interface as GripEnvelope, so the plant and its report need not care which.
    """

    curve: tuple
    wheelbase: float
    limited_steps: int = 0
    total_steps: int = 0
    max_demanded_lateral: float = 0.0
    max_allowed_lateral: float = 0.0
    # True for a measured curve whose delivered column falls past a peak (rb2, 2026-11-11:
    # 2.94 m/s^2 at 3.75 asked, 2.7 beyond). The default still refuses one, so a typo in a
    # table cannot make the tyres weaker as they are pushed harder unnoticed.
    falling: bool = False
    # How much of the longitudinal demand competes for grip, 0 to 1, as GripEnvelope's: lateral is
    # limited to sqrt(peak^2 - (coupling * a_long)^2), peak the curve's largest delivered value. 0 (the
    # default, and every floor so far) takes no share: grip_by_long_accel measured braking costing nothing.
    longitudinal_coupling: float = 0.0

    def __post_init__(self):
        if not (math.isfinite(self.longitudinal_coupling) and 0.0 <= self.longitudinal_coupling <= 1.0):
            raise ValueError(f"longitudinal_coupling must be in [0, 1], got {self.longitudinal_coupling!r}")
        asked = [row[0] for row in self.curve]
        delivered = [row[1] for row in self.curve]
        if len(asked) < 2 or asked != sorted(asked) or (not self.falling and delivered != sorted(delivered)):
            columns = "asked" if self.falling else "both columns"
            raise ValueError(f"grip curve must be ascending in {columns}, got {self.curve!r}")
        self._asked, self._delivered = asked, delivered

    def deliver(self, asked: float) -> float:
        """What the tyres deliver for ``asked``; past the table it stays at the last value."""
        return float(np.interp(asked, self._asked, self._delivered))

    def apply(self, steering: float, speed: float, longitudinal_accel: float = 0.0) -> float:
        """The wheel angle that turns the car as hard as the tyres actually will."""
        self.total_steps += 1
        if speed < KINEMATIC_SPEED:
            return steering
        demanded = abs(math.tan(steering)) * speed * speed / self.wheelbase
        allowed = self.deliver(demanded)
        if self.longitudinal_coupling > 0.0:
            used = self.longitudinal_coupling * longitudinal_accel
            allowed = min(allowed, math.sqrt(max(0.0, max(self._delivered) ** 2 - used * used)))
        self.max_demanded_lateral = max(self.max_demanded_lateral, demanded)
        self.max_allowed_lateral = max(self.max_allowed_lateral, allowed)
        if allowed < demanded:
            self.limited_steps += 1
        return math.copysign(math.atan(self.wheelbase * allowed / (speed * speed)), steering)

    def report(self) -> dict:
        return {
            "grip_limit_mps2": round(max(self._delivered), 3),
            "grip_limit_g": round(max(self._delivered) / 9.81, 3),
            "grip_limited_steps": self.limited_steps,
            "grip_limited_fraction": (round(self.limited_steps / self.total_steps, 4)
                                      if self.total_steps else 0.0),
            "max_demanded_lateral_mps2": round(self.max_demanded_lateral, 3),
            "max_allowed_lateral_mps2": round(self.max_allowed_lateral, 3),
            "min_lateral_budget_mps2": None,
        }


def longitudinal_accel(
    speed_command: float,
    measured_speed: float,
    params,
) -> float:
    """What the plant's own longitudinal controller is about to do.

    Calls exactly the functions the gym calls in ``SPEED`` mode, so the
    figure fed to the friction circle is the acceleration the plant will
    genuinely apply rather than an independent guess at it.
    """
    from f1tenth_gym.envs.dynamic_models import pid_accl
    from f1tenth_gym.envs.dynamic_models.utils import accl_constraints

    demand = pid_accl(
        speed_command,
        measured_speed,
        params.a_max,
        params.v_max,
        params.v_min,
    )
    return float(
        accl_constraints(
            measured_speed,
            demand,
            params.v_switch,
            params.a_max,
            params.v_min,
            params.v_max,
        )
    )
