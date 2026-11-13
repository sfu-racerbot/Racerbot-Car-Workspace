"""Launch-lap speed scale: a bounded fraction of the racing line's speed right after a standing start.

Off by default (scale 1.0). The first metres of a standing start are driven at the plan's speed on tyres
and a speed loop that have not yet shown they can hold it; scaling the profile for the first ``distance``
metres trades a little time there for margin. Pure Python, no rclpy: the node feeds it the deadman state and
the measured speed, and multiplies the profiled speed by ``factor``.
"""

import math

# A press below this measured speed (m/s) is a standing start and arms the scale.
ARM_SPEED = 0.5


class LaunchScale:
    """Armed on a deadman press from standstill, counts down by distance driven, then steps back to 1.0.

    * ``on_deadman(engaged, speed, line_length)`` is called with the deadman state; only a released -> held
      edge can arm, and only when the measured speed is below ARM_SPEED (None, stale odometry, counts as 0).
      A press while rolling at ARM_SPEED or more neither arms nor clears what remains.
    * ``step(speed, dt)`` counts ``abs(speed) * dt`` metres off; a stale tick (speed None or not finite, or
      dt not positive and finite) counts nothing, so a car that is not reporting motion never uses it up.
    * ``factor`` is ``scale`` while metres remain, else 1.0; ``remaining`` is the metres left.
    """

    def __init__(self, scale: float, distance: float):
        if not (math.isfinite(scale) and 0.5 < scale <= 1.0):
            raise ValueError(f'launch_speed_scale must be finite and in (0.5, 1.0], got {scale!r}')
        if not (math.isfinite(distance) and distance >= 0.0):
            raise ValueError(f'launch_scale_distance must be finite and >= 0 m, got {distance!r}')
        self.scale = float(scale)
        self.distance = float(distance)
        self.remaining = 0.0
        self._engaged = False

    @property
    def factor(self) -> float:
        return self.scale if (self.remaining > 0.0 and self.scale < 1.0) else 1.0

    def on_deadman(self, engaged: bool, speed, line_length: float) -> None:
        rising = bool(engaged) and not self._engaged
        self._engaged = bool(engaged)
        if not rising or self.scale >= 1.0:
            return
        v = 0.0 if speed is None or not math.isfinite(speed) else abs(float(speed))
        if v >= ARM_SPEED:
            return
        length = self.distance if self.distance > 0.0 else float(line_length)
        if math.isfinite(length) and length > 0.0:
            self.remaining = length

    def step(self, speed, dt: float) -> None:
        if speed is None or not math.isfinite(speed) or not math.isfinite(dt) or dt <= 0.0:
            return
        self.remaining = max(0.0, self.remaining - abs(float(speed)) * float(dt))
