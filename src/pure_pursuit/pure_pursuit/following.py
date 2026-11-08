"""Conservative longitudinal envelope, independent of ROS and command slew."""
import math

import numpy as np


def project_arc(xy, cumulative, point, index):
    """Continuous local line projection; avoids 0.15 m waypoint speed impulses."""
    best = (math.inf, 0.0)
    for i in ((index - 1) % len(xy), index):
        delta = xy[(i + 1) % len(xy)] - xy[i]
        length = float(np.linalg.norm(delta))
        fraction = float(np.clip(np.dot(np.asarray(point) - xy[i], delta)
                                 / max(length * length, 1e-12), 0, 1))
        error = float(np.linalg.norm(np.asarray(point) - xy[i] - fraction * delta))
        if error < best[0]:
            best = error, float(cumulative[i]) + fraction * length
    return best[1]


def speed_ceiling(gap, opponent_speed, age, *, brake=3.0, accel=4.0,
                  delay=0.45, reserve=0.65, speed_uncertainty=1.0):
    """Bound ego's delayed stop before the opponent's earliest stop.

    Gap is visible rear surface to ego front, along the line. The reserve
    covers centroid/pose error, curvature and tracking. Opponent may stop
    instantaneously: deliberately give NO stopping-distance credit to its
    speed. Its tracked speed only increases the uncertainty allowance during
    lost sightings; it never creates free space. Thus hard braking cannot
    invalidate an optimistic opponent-velocity prediction.

    Ego can accelerate at 4 m/s² through the 0.45 s response allowance
    (0.10 transport + 1/7 speed-loop lag + sensing/tracking budget), then
    brakes at 3 m/s². Solve v*t+a*t²/2+(v+a*t)²/(2*b) <= gap-reserve.
    """
    values = (gap, opponent_speed, age, brake, accel, delay, reserve, speed_uncertainty)
    if not all(math.isfinite(v) for v in values):
        return 0.0
    if min(age, accel, delay, reserve, speed_uncertainty) < 0 or brake <= 0:
        return 0.0
    available = gap - reserve - age * (abs(opponent_speed) + speed_uncertainty)
    linear = delay * (brake + accel)
    constant = brake * accel * delay ** 2 + (accel * delay) ** 2
    return max(0.0, math.sqrt(max(0.0, linear ** 2 + 2 * brake * available - constant)) - linear)
