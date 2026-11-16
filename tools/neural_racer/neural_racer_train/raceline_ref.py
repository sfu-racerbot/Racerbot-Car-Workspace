"""The lap-time reference of gate G5: the minimum-curvature line inside a
track's width (pure_pursuit's raceline_optimizer, Heilmeier et al. 2019),
paced at a race's caps.

The pace is the textbook speed profile: the cornering limit min(v_cap,
sqrt(a_lat / kappa)), then a forward pass at the car's acceleration and a
backward pass at its braking, from a standing start, for `laps` laps. No
friction ellipse, so it is a little optimistic in a braking zone that is also
a bend: a time near it is near the best line, and a time under it is possible.
"""
import numpy as np

from neural_racer_train.gpu_sim.params import G

HALF_WIDTH = 0.155          # m, WrapperConfig.car_width / 2
MARGIN = 0.05               # m off each wall
SPACING = 0.10              # m, as tracks.SPACING
STENCIL = 3                 # points either side for the curvature (0.3 m)
ACCEL, BRAKE = 4.5, 4.0     # m/s^2, the speed loop's fitted limits (CarParams)


def optimize(spec) -> dict:
    """The line for a TrackSpec: points (n, 2), segment lengths (n,) and curvature (n,)."""
    from pure_pursuit.raceline_optimizer import optimize_minimum_curvature
    res = optimize_minimum_curvature(spec.centre, spec.w_left, spec.w_right, HALF_WIDTH, MARGIN, SPACING)
    line = np.asarray(res['line'], dtype=np.float64)
    return {'line': line, 'ds': np.hypot(*(np.roll(line, -1, axis=0) - line).T), 'kappa': curvature(line)}


def curvature(line: np.ndarray, k: int = STENCIL) -> np.ndarray:
    """Menger curvature of each point of a closed line, on chords of k points."""
    a, b, c = np.roll(line, k, axis=0), line, np.roll(line, -k, axis=0)
    ab, bc, ca = b - a, c - b, a - c
    cross = np.abs(ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0])
    den = np.hypot(*ab.T) * np.hypot(*bc.T) * np.hypot(*ca.T)
    return np.where(den > 1e-12, 2 * cross / np.maximum(den, 1e-12), 0.0)


def race_time(ref: dict, start_xy, v_cap: float, a_lat: float, mu: float, laps: int = 3) -> float:
    """Seconds for `laps` laps of the line from rest at the point nearest start_xy."""
    i0 = int(np.argmin(np.hypot(*(ref['line'] - np.asarray(start_xy)[:2]).T)))
    ds = np.tile(np.roll(ref['ds'], -i0), laps)
    kappa = np.tile(np.roll(ref['kappa'], -i0), laps)
    grip = mu * G
    lat, acc, brk = min(a_lat, grip), min(ACCEL, grip), min(BRAKE, grip)
    v = np.minimum(v_cap, np.sqrt(lat / np.maximum(kappa, 1e-6)))
    v = np.append(v, v[0])                    # the speed at the far end of the last segment
    v[0] = 0.0
    for i in range(1, len(v)):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * acc * ds[i - 1]))
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * brk * ds[i]))
    return float(np.sum(2 * ds / np.maximum(v[:-1] + v[1:], 1e-6)))
