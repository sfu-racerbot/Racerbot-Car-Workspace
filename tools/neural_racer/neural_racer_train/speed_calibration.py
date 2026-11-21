"""Fit a bounded steering gain per odometry speed band; no ROS or Torch."""
from collections import Counter
import math

import numpy as np

from neural_racer import observation as obs


def speed_bucket(v):
    if v < 0:
        return 'reverse'
    if v < 1:
        return '[0,1)'
    if v < 3:
        return '[1,3)'
    return '[3,inf)'


def calibrated_steer(raw, speed, gains):
    if not math.isfinite(raw):
        return raw  # Preserve the wrapper's non-finite-output stop.
    gain = gains.get(speed_bucket(speed), 1.)
    if not 0.5 <= gain <= 1.5:
        raise ValueError('calibration gain outside [0.5, 1.5]')
    return float(np.clip(raw * gain, -obs.STEER_LIMIT, obs.STEER_LIMIT))


class SteeringCalibration:
    """Minimize sum((gain * raw - target)^2) + (gain - 1)^2 in each band."""
    def __init__(self):
        self.xx = Counter()
        self.xy = Counter()
        self.n = 0

    def add(self, speed, raw, target):
        bucket = speed_bucket(speed)
        self.xx[bucket] += raw * raw
        self.xy[bucket] += raw * target
        self.n += 1

    def gains(self):
        return {b: float(np.clip((self.xy[b] + 1.) / (xx + 1.), 0.5, 1.5))
                for b, xx in self.xx.items()}
