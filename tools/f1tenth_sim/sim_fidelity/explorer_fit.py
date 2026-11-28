"""Fit the car's unmeasured physics numbers from recorded driving.

The explorer (``src/car_explorer``) drives laps at a spread of speeds, corner
limits and stops. This module takes what it recorded -- the commands the car
was given and how it really moved -- and finds the simulator numbers that make
the plant (``sim_fidelity.plant.FidelityPlant``) move the same way when given
the same commands.

Design rules (docs/explorer.md):

* **Measured numbers are never fitted.** ``FIXED`` lists the fields that come
  from a tape measure or a scale; asking to fit one raises ``ValueError``.
* **Every fitted number is readable.** Each is an existing ``CarCalibration``
  field with physical units, so a fit can be checked and argued with.
* **Every fit says how sure it is.** ``sensitivity`` reports how much the
  score worsens when each fitted number moves a little. A number the score
  barely notices is not pinned down by the data, whatever value it landed on.

The score is the replay error the club's other fits use: recorded commands
replayed open-loop through the plant for a few seconds from the recorded
speed, compared with the measured speed (from scan matching) and turn rate
(gyro). It is pure NumPy around the plant, with no ROS.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

DT = 0.025  # s, the plant's control tick (40 Hz, the scan rate)

# Numbers that come from measuring the car, never from a fit
# (project rule 2026-11-23: a measured value stays, and whatever was
# compensating for a wrong one gets fixed instead).
FIXED = frozenset({
    'mass', 'cog_front_fraction', 'wheelbase', 'width', 'length', 'body_width',
    'body_length', 'lidar_offset_x', 'steering_max_left', 'steering_max_right',
    'speed_to_erpm_gain', 'steering_angle_to_servo_gain',
    'steering_angle_to_servo_offset', 'servo_min', 'servo_max', 'speed_max', 'speed_min',
})

# What the explorer's data can teach, with the search range for each.
# (low, high, whole number?). Ranges are physical: a fit that ends on an edge
# says so in its report (``FitResult.at_bound``).
PARAMETERS = {
    'surface_friction': (0.25, 1.2, False),            # grip, x g
    'cornering_stiffness_front': (2.0, 10.0, False),   # tyre stiffness, gym units
    'cornering_stiffness_rear': (2.0, 10.0, False),
    'yaw_inertia': (0.02, 0.2, False),                 # kg m^2
    'steering_delay_ticks': (0, 8, True),              # x 25 ms
    'steering_servo_gain': (5.0, 40.0, False),         # 1/s
    'speed_delay_ticks': (0, 8, True),                 # x 25 ms
    'speed_loop_accel_gain': (1.0, 20.0, False),       # 1/s
    'speed_loop_accel_max': (1.0, 8.0, False),         # m/s^2
    'speed_loop_brake_gain': (1.0, 20.0, False),       # 1/s
    'speed_loop_brake_max': (1.0, 8.0, False),         # m/s^2
}

# Each error is divided by a typical size so speed and turning count alike
# (the same normalisation as fit_slam_tyres.py's yaw term).
SPEED_SCALE = 0.30  # m/s
YAW_SCALE = 0.30    # rad/s


@dataclass(frozen=True)
class Window:
    """A few seconds of recorded driving, ready to replay."""
    commands: np.ndarray        # (N, 2): steering rad, speed m/s, at 40 Hz
    speed: np.ndarray           # (N,) measured true speed, m/s
    yaw_rate: np.ndarray        # (N,) measured turn rate, rad/s
    initial_speed: float        # m/s at the window's first tick
    prior_speed_commands: tuple  # speed commands just before it, oldest first


@dataclass
class FitResult:
    values: dict                # fitted number per name
    cost_before: float
    cost_after: float
    sensitivity: dict           # name -> rise in cost for a small step either way
    at_bound: list              # names that ended on their search edge
    evaluations: int


def check_names(names) -> list:
    names = list(names)
    if not names:
        raise ValueError('nothing to fit: name at least one parameter')
    fixed = sorted(set(names) & FIXED)
    if fixed:
        raise ValueError(f'measured, never fitted: {", ".join(fixed)}')
    unknown = sorted(set(names) - set(PARAMETERS))
    if unknown:
        raise ValueError(f'not a fittable parameter: {", ".join(unknown)} '
                         f'(choose from {", ".join(sorted(PARAMETERS))})')
    if len(set(names)) != len(names):
        raise ValueError('a parameter is named twice')
    return names


def windows_from_series(t, steer, speed_cmd, speed, yaw_rate, good, *,
                        horizon: float = 3.0, spacing: float = 1.0,
                        min_speed: float = 0.5, prior: int = 8) -> list:
    """Cut one run's uniform 40 Hz arrays into replay windows.

    A window starts only where the car is moving (``min_speed``) and every tick
    in it is trusted (``good``), so a window never spans a localisation dropout.
    """
    arrays = [np.asarray(a, float) for a in (t, steer, speed_cmd, speed, yaw_rate)]
    good = np.asarray(good, bool)
    n = len(arrays[0])
    if any(len(a) != n for a in arrays) or len(good) != n:
        raise ValueError('every series must have the same length')
    width = int(round(horizon / DT))
    step = max(1, int(round(spacing / DT)))
    if width < 2:
        raise ValueError(f'horizon {horizon} s is shorter than two ticks')
    t, steer, speed_cmd, speed, yaw_rate = arrays
    out = []
    start = prior
    while start + width <= n:
        stop = start + width
        block = slice(start, stop)
        if (good[block].all() and speed[start] >= min_speed
                and all(np.isfinite(a[block]).all() for a in arrays)
                and np.isfinite(speed_cmd[start - prior:start]).all()):
            out.append(Window(np.column_stack([steer[block], speed_cmd[block]]),
                              speed[block].copy(), yaw_rate[block].copy(),
                              float(speed[start]),
                              tuple(float(v) for v in speed_cmd[start - prior:start])))
            start = stop - width + step
        else:
            start += 1
    return out


def windows_from_run(run, loc_dir, **kw) -> list:
    """Replay windows from a recorded, located run (``loc.py`` output)."""
    from run_analysis.car_measure import _gyro_bias, measure_run
    r = measure_run(run, loc_dir)
    bias = _gyro_bias([r])
    return windows_from_series(r['t'], r['steer'], r['speed_cmd'], r['v'],
                               r['gyro'] - bias, r['good'], **kw)


def replay(calibration, window: Window):
    """(speed, yaw_rate) the plant produces for one window's commands."""
    from run_analysis.fit_sim import simulate_commands
    return simulate_commands(window.commands, calibration,
                             initial_speed=window.initial_speed,
                             prior_speed_commands=window.prior_speed_commands)


def window_error(calibration, window: Window) -> float:
    speed, yaw = replay(calibration, window)
    speed_rms = math.sqrt(float(np.mean((speed - window.speed) ** 2)))
    yaw_rms = math.sqrt(float(np.mean((yaw - window.yaw_rate) ** 2)))
    return 0.5 * (speed_rms / SPEED_SCALE + yaw_rms / YAW_SCALE)


def cost(calibration, windows) -> float:
    """Mean normalised replay error over the windows; each window counts once."""
    if not windows:
        raise ValueError('no windows to score')
    errors = [window_error(calibration, w) for w in windows]
    value = float(np.mean(errors))
    return value if math.isfinite(value) else math.inf


def apply(calibration, names, vector):
    values = {}
    for name, x in zip(names, vector):
        low, high, whole = PARAMETERS[name]
        x = min(max(float(x), low), high)
        values[name] = int(round(x)) if whole else x
    return replace(calibration, **values), values


class _Objective:
    """Picklable so differential_evolution can farm it out to processes."""

    def __init__(self, calibration, names, windows):
        self.calibration, self.names, self.windows = calibration, names, windows

    def __call__(self, vector):
        return cost(apply(self.calibration, self.names, vector)[0], self.windows)


def sensitivity(calibration, names, windows, base_cost=None) -> dict:
    """How much the score rises when each fitted number moves a small step.

    The step is one tick for a whole-number delay, else 10% of the value. The
    larger of the two directions is reported; a value near 0 means the data
    cannot tell that number apart from its neighbours.
    """
    base = cost(calibration, windows) if base_cost is None else base_cost
    out = {}
    for name in names:
        low, high, whole = PARAMETERS[name]
        value = getattr(calibration, name)
        step = 1 if whole else 0.1 * abs(value)
        rises = []
        for moved in (value - step, value + step):
            if low <= moved <= high:
                moved = int(moved) if whole else moved
                rises.append(cost(replace(calibration, **{name: moved}), windows) - base)
        out[name] = max(rises) if rises else 0.0
    return out


def fit(calibration, windows, names, *, maxiter: int = 30, popsize: int = 8,
        seed: int = 0, workers: int = 1, tol: float = 1e-3) -> FitResult:
    """Search the named numbers for the lowest replay error.

    Differential evolution: a global search that copes with noisy scores and
    with whole-number delays (``integrality``), and parallelises with
    ``workers``. Every other calibration field stays as given.
    """
    from scipy.optimize import differential_evolution
    names = check_names(names)
    if not windows:
        raise ValueError('no windows to fit')
    bounds = [PARAMETERS[n][:2] for n in names]
    integrality = [PARAMETERS[n][2] for n in names]
    objective = _Objective(calibration, names, windows)
    before = cost(calibration, windows)
    x0 = [min(max(float(getattr(calibration, n)), lo), hi) for n, (lo, hi) in zip(names, bounds)]
    result = differential_evolution(
        objective, bounds, integrality=integrality, x0=x0, maxiter=maxiter,
        popsize=popsize, seed=seed, tol=tol, polish=False,
        workers=workers, updating='deferred' if workers != 1 else 'immediate')
    fitted, values = apply(calibration, names, result.x)
    after = cost(fitted, windows)
    if after > before:  # never hand back something worse than the start
        fitted, values, after = calibration, {n: getattr(calibration, n) for n in names}, before
    at_bound = [n for n in names
                if math.isclose(values[n], PARAMETERS[n][0]) or math.isclose(values[n], PARAMETERS[n][1])]
    return FitResult(values, before, after, sensitivity(fitted, names, windows, after),
                     at_bound, int(result.nfev))
