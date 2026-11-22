"""Effective body turning response, not a tyre-force or rack-angle measurement.

Forward motion: saturate the bicycle's lateral acceleration request, then relax
the yaw rate towards that target. Close the unmeasured CG slip with the rear
no-slip geometry (v sin(beta) = lr * yaw_rate) and the same relaxation time.
Gym still integrates the chassis pose, steering and longitudinal dynamics.
"""

from dataclasses import dataclass
from functools import lru_cache
import math


@dataclass(frozen=True)
class TurnResponse:
    gain: float
    trim_rad: float
    lateral_limit_mps2: float
    lag_s: float

    def __post_init__(self):
        if not (.6 <= self.gain <= 1.4 and abs(self.trim_rad) <= .04
                and 1.5 <= self.lateral_limit_mps2 <= 10 and .025 <= self.lag_s <= .35):
            raise ValueError("turn response outside the fitted parameter bounds")


def steady_rate(speed, steer, wheelbase, gain, trim, limit):
    """Forward, positive-speed yaw target (rad/s). Compiled inside the Gym callback."""
    return limit / speed * math.tanh(gain * speed**2 * math.tan(steer + trim) / (wheelbase * limit))


@lru_cache(maxsize=16)
def make_dynamics(model: TurnResponse):
    from numba import njit
    from f1tenth_gym.envs.dynamic_models.single_track import vehicle_dynamics_st

    target = njit(steady_rate)
    gain, trim, limit, lag = model.gain, model.trim_rad, model.lateral_limit_mps2, model.lag_s

    @njit
    def dynamics(x, u, params):
        out = vehicle_dynamics_st(x, u, params)
        speed, yaw_rate, beta = x[3], x[5], x[6]
        if speed >= .5:  # Gym's own transition; retain its stopped and reverse branch.
            out[5] = (target(speed, x[2], params[3] + params[4], gain, trim, limit) - yaw_rate) / lag
            slip = math.asin(min(1., max(-1., params[4] * yaw_rate / speed)))
            out[6] = (slip - beta) / lag
        return out

    return dynamics
