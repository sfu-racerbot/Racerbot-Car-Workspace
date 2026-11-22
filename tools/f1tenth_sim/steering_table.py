"""Simulation-only piecewise-linear command / delivered lateral acceleration model."""
import math

import numpy as np

from run_analysis.grip_truth import BINS


def fit(rows, wheelbase, indexed, min_count=10):
    """Fit adjacent well-populated bins on each side, up to the first grip peak."""
    v, cmd, yaw = rows.T
    asked = v**2 * np.abs(np.tan(cmd)) / wheelbase
    speed_band = np.floor(v).astype(int) if indexed else np.zeros(len(v), dtype=int)
    segments = []
    for sign in (-1, 1):
        for band in np.unique(speed_band):
            group = (speed_band == band) & (cmd * sign > 0)
            previous = None
            for low, high in BINS:
                mask = group & (asked >= low) & (asked < high)
                if mask.sum() < min_count:
                    previous = None
                    continue
                node = [float(np.median(asked[mask])), float(np.median(sign*v[mask]*yaw[mask])),
                        float(min(v[mask])), float(max(v[mask]))]
                if node[1] <= 0:
                    previous = None
                    continue
                if previous is not None:
                    if node[1] <= previous[1]:
                        break
                    lo, hi = max(previous[2], node[2]), min(previous[3], node[3])
                    if lo <= hi:
                        segments.append(dict(sign=sign, v=[lo, hi], asked=[previous[0], node[0]],
                                             delivered=[previous[1], node[1]]))
                previous = node
    return dict(wheelbase=float(wheelbase), segments=segments)


def lookup(model, speed, steering, inverse):
    if not math.isfinite(speed) or speed <= 0 or not math.isfinite(steering):
        return None
    x = speed**2 * abs(math.tan(steering)) / model['wheelbase']
    source, dest = ('delivered', 'asked') if inverse else ('asked', 'delivered')
    for s in model['segments']:
        a, b = s[source]
        if (s['sign']*steering > 0 and s['v'][0] <= speed <= s['v'][1] and a <= x <= b):
            y, z = s[dest]
            return s['sign'] * (y + (z-y)*(x-a)/(b-a))
    return None


def predict(model, speed, cmd):
    delivered = lookup(model, speed, cmd, inverse=False)
    return None if delivered is None else delivered / speed


def compensate(model, speed, steering, age=0.0):
    """Return None for fallback. No speed/acceleration extrapolation or stale speeds."""
    if not 0 <= age <= .1:
        return None
    asked = lookup(model, speed, steering, inverse=True)
    return None if asked is None else math.atan(model['wheelbase']*asked/speed**2)
