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
* **Every fit says how sure it is.** ``interval`` gives each fitted number
  a 90% range by resampling the windows, and ``sensitivity`` how much the
  score worsens when it moves a little. A wide range means the data did not
  pin that number down, whatever value it landed on.
* **Every fit is checked on driving it never saw.** ``fit`` holds back every
  fifth window and reports the score on those before and after.
* **A floor that changes mid-session is caught.** ``drift`` fits grip (or any
  number) separately on the start and the end of a session and flags ranges
  that do not overlap.

The score is the replay error the club's other fits use: recorded commands
replayed open-loop through the plant for a few seconds from the recorded
speed, compared with the measured speed (from scan matching) and turn rate
(gyro). Each replay first drives a short unscored lead-in (``lead_in``), so
the simulated wheels and turn rate have caught up with the real car's before
scoring starts; without it every window began with the wheels straight and
the car not turning, an error no parameter can remove. Pure NumPy around the
plant, with no ROS.
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
    commands: np.ndarray        # (L + N, 2): steering rad, speed m/s, at 40 Hz
    speed: np.ndarray           # (N,) measured true speed over the scored part, m/s
    yaw_rate: np.ndarray        # (N,) measured turn rate over the scored part, rad/s
    initial_speed: float        # m/s at the first command (the lead-in's start)
    prior_speed_commands: tuple  # speed commands just before that, oldest first
    lead_in: int = 0            # L: ticks replayed first and not scored
    t0: float = 0.0             # s, recording time of the first scored tick


@dataclass
class FitResult:
    values: dict                # fitted number per name
    cost_before: float
    cost_after: float
    sensitivity: dict           # name -> rise in cost for a small step either way
    at_bound: list              # names that ended on their search edge
    evaluations: int
    holdout_before: float | None = None  # score on the held-back windows
    holdout_after: float | None = None
    intervals: dict = None      # name -> (low, high) 90% range, from ``interval``


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
                        min_speed: float = 0.5, prior: int = 8,
                        lead_in: float = 0.75) -> list:
    """Cut one run's uniform 40 Hz arrays into replay windows.

    A window starts only where the car is moving (``min_speed``) and every tick
    in it, lead-in included, is trusted (``good``), so a window never spans a
    localisation dropout. ``lead_in`` seconds of commands come before the
    scored ``horizon`` and are replayed but not scored.
    """
    arrays = [np.asarray(a, float) for a in (t, steer, speed_cmd, speed, yaw_rate)]
    good = np.asarray(good, bool)
    n = len(arrays[0])
    if any(len(a) != n for a in arrays) or len(good) != n:
        raise ValueError('every series must have the same length')
    width = int(round(horizon / DT))
    lead = int(round(lead_in / DT))
    step = max(1, int(round(spacing / DT)))
    if width < 2:
        raise ValueError(f'horizon {horizon} s is shorter than two ticks')
    if lead < 0:
        raise ValueError(f'lead_in {lead_in} s is negative')
    t, steer, speed_cmd, speed, yaw_rate = arrays
    out = []
    start = prior + lead            # first scored tick
    while start + width <= n:
        first, stop = start - lead, start + width
        block, whole = slice(start, stop), slice(first, stop)
        if (good[whole].all() and speed[first] >= min_speed
                and all(np.isfinite(a[whole]).all() for a in arrays)
                and np.isfinite(speed_cmd[first - prior:first]).all()):
            out.append(Window(np.column_stack([steer[whole], speed_cmd[whole]]),
                              speed[block].copy(), yaw_rate[block].copy(),
                              float(speed[first]),
                              tuple(float(v) for v in speed_cmd[first - prior:first]),
                              lead, float(t[start])))
            start += step
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
    """(speed, yaw_rate) the plant produces over a window's scored part."""
    from run_analysis.fit_sim import simulate_commands
    speed, yaw = simulate_commands(window.commands, calibration,
                                   initial_speed=window.initial_speed,
                                   prior_speed_commands=window.prior_speed_commands)
    return speed[window.lead_in:], yaw[window.lead_in:]


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


def _errors_for(job):
    calibration, windows = job
    return [window_error(calibration, w) for w in windows]


def _map(jobs, workers):
    if workers == 1:
        return [_errors_for(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_errors_for, jobs))


def profile_grid(calibration, name, *, points: int = 13, span: float = 0.4) -> list:
    """Values of ``name`` to profile around the calibration's: every tick
    within +-2 for a delay, else ``points`` values over +-``span`` of it,
    always inside the search range."""
    low, high, whole = PARAMETERS[name]
    value = getattr(calibration, name)
    if whole:
        return [v for v in range(int(value) - 2, int(value) + 3) if low <= v <= high]
    lo, hi = max(low, value * (1 - span)), min(high, value * (1 + span))
    return [float(v) for v in np.linspace(lo, hi, points)]


def profile(calibration, windows, name, grid, *, workers: int = 1) -> np.ndarray:
    """Every window's error with ``name`` set to each grid value and every
    other number as in ``calibration``: shape (len(grid), len(windows))."""
    check_names([name])
    if not windows:
        raise ValueError('no windows to score')
    jobs = [(replace(calibration, **{name: v}), list(windows)) for v in grid]
    return np.asarray(_map(jobs, workers), float)


def _best_on_grid(grid, curve):
    """The minimum of a sampled curve, refined by a parabola through the
    lowest point and its neighbours (not past them)."""
    i = int(np.argmin(curve))
    if 0 < i < len(grid) - 1:
        x0, x1, x2 = grid[i - 1], grid[i], grid[i + 1]
        y0, y1, y2 = curve[i - 1], curve[i], curve[i + 1]
        den = (x0 - x1) * (x0 - x2) * (x1 - x2)
        a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
        b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den
        if a > 0:
            return float(min(max(-b / (2 * a), x0), x2))
    return float(grid[i])


def interval(grid, errors, *, whole=False, resamples: int = 400, level: float = 0.9,
             seed: int = 0) -> tuple:
    """(best, low, high): the grid value with the lowest mean error, and the
    ``level`` range of where that lowest point lands when the windows are
    resampled with replacement (bootstrap). A delay's range is whole ticks."""
    grid = np.asarray(grid, float)
    errors = np.asarray(errors, float)
    if errors.ndim != 2 or errors.shape[0] != len(grid) or errors.shape[1] == 0:
        raise ValueError('errors must be (grid points, windows) with at least one window')
    pick = (lambda c: float(grid[int(np.argmin(c))])) if whole else (lambda c: _best_on_grid(grid, c))
    best = pick(errors.mean(axis=1))
    rng = np.random.default_rng(seed)
    n = errors.shape[1]
    draws = [pick(errors[:, rng.integers(0, n, n)].mean(axis=1)) for _ in range(resamples)]
    tail = (1 - level) / 2
    return best, float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))


def polish(calibration, windows, names, *, workers: int = 1, **kw):
    """One pass over the named numbers: profile each around its value, move it
    to the profile's lowest point when that lowers the score, and keep the 90%
    range from the same profile. A global search stops on a coarse grid; this
    puts each number where its own profile says, so it lies inside its range.
    Returns (calibration, cost, {name: (low, high)})."""
    current = calibration
    best_cost = cost(current, windows)
    ranges = {}
    for name in check_names(names):
        grid = profile_grid(current, name)
        best, low, high = interval(grid, profile(current, windows, name, grid, workers=workers),
                                   whole=PARAMETERS[name][2], **kw)
        moved, _ = apply(current, [name], [best])
        moved_cost = cost(moved, windows)
        if moved_cost < best_cost:
            current, best_cost = moved, moved_cost
        ranges[name] = (low, high)
    return current, best_cost, ranges


def intervals(calibration, windows, names, *, workers: int = 1, **kw) -> dict:
    """``interval`` for each named number around ``calibration``'s value."""
    out = {}
    for name in check_names(names):
        grid = profile_grid(calibration, name)
        _, low, high = interval(grid, profile(calibration, windows, name, grid, workers=workers),
                                whole=PARAMETERS[name][2], **kw)
        out[name] = (low, high)
    return out


def drift(calibration, windows, name='surface_friction', *, parts: int = 3,
          workers: int = 1, **kw) -> dict:
    """Fit ``name`` alone on the first and last ``parts``-th of the session
    (by time) and say whether the two 90% ranges miss each other: a floor
    that got dustier or a battery that sagged shows up here, where one fit
    over the whole session would just average it away."""
    if parts < 2:
        raise ValueError('parts must be 2 or more')
    ordered = sorted(windows, key=lambda w: w.t0)
    size = len(ordered) // parts
    if size < 3:
        raise ValueError(f'too few windows ({len(ordered)}) to split into {parts}')
    grid = profile_grid(calibration, name)
    whole = PARAMETERS[name][2]
    first = interval(grid, profile(calibration, ordered[:size], name, grid, workers=workers),
                     whole=whole, **kw)
    last = interval(grid, profile(calibration, ordered[-size:], name, grid, workers=workers),
                    whole=whole, **kw)
    changed = first[2] < last[1] or last[2] < first[1]
    return {'name': name, 'start': first, 'end': last, 'changed': bool(changed)}


def split_holdout(windows, every: int = 5):
    """(fit windows, held-back windows): every ``every``-th window by time is
    held back, so both halves span the whole session."""
    if every < 2:
        raise ValueError('every must be 2 or more')
    ordered = sorted(windows, key=lambda w: w.t0)
    held = ordered[every - 1::every]
    kept = [w for i, w in enumerate(ordered) if (i + 1) % every]
    return kept, held


def fit(calibration, windows, names, *, maxiter: int = 30, popsize: int = 8,
        seed: int = 0, workers: int = 1, tol: float = 1e-3, holdout_every: int = 0,
        with_intervals: bool = False) -> FitResult:
    """Search the named numbers for the lowest replay error.

    Differential evolution: a global search that copes with noisy scores and
    with whole-number delays (``integrality``), and parallelises with
    ``workers``. Every other calibration field stays as given.
    ``holdout_every`` > 0 fits on all but every N-th window and scores those
    before and after; ``with_intervals`` finishes with ``polish`` and adds
    each number's 90% range.
    """
    from scipy.optimize import differential_evolution
    names = check_names(names)
    if not windows:
        raise ValueError('no windows to fit')
    held = []
    if holdout_every:
        windows, held = split_holdout(windows, holdout_every)
        if not windows or not held:
            raise ValueError('too few windows to hold some back')
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
    ranges = None
    if with_intervals:
        fitted, after, ranges = polish(fitted, windows, names, workers=workers)
        values = {n: getattr(fitted, n) for n in names}
    at_bound = [n for n in names
                if math.isclose(values[n], PARAMETERS[n][0]) or math.isclose(values[n], PARAMETERS[n][1])]
    out = FitResult(values, before, after, sensitivity(fitted, names, windows, after),
                    at_bound, int(result.nfev), intervals=ranges)
    if held:
        out.holdout_before, out.holdout_after = cost(calibration, held), cost(fitted, held)
    return out
