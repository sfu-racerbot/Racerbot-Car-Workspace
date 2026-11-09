"""Minimum-time racing line from a car's measured physics. No ROS.

The minimum-curvature line (raceline_optimizer.py) makes every corner as gentle as the track allows, which is
not the same as the fastest lap: it takes no account of where the car is fast or slow. This module optimizes
the lap time itself, with a speed profile built from what the car was measured to do:

    grip          lateral acceleration it achieves, per turning side (car 2, 2026-11-06: 2.45 left,
                  1.95 right, at the asks a racing line makes; tools/run_analysis/car_measure.py)
    braking       longitudinal limits from the fitted speed loop (car 2: brake 3.0, accel 4.0 m/s^2)
    steering      the planning angle under the controller's cap, through the measured linkage (car 2:
                  0.349 commanded gave 0.345 at the wheels), so the controller keeps steering in hand
    steering rate the controller's slew limit: where curvature changes faster than the rack can follow at
                  the planned speed, the planned speed comes down
    delay         the servo's delay: that curvature change is judged over the distance the car covers
                  while the servo has not yet responded

Minimizing time rather than curvature is what takes corners wide-in and late-apex (the car straightens the
exit, where it is accelerating, at the cost of a tighter entry, where it is braking anyway), and it
straightens wiggly sections because every wiggle costs time.

Safety is a constraint, not a weight: every station of the result keeps the car's body clear of the mapped
walls by `wall_margin`, and the LiDAR's forward cone clear by `cone_margin` beyond the controller's
emergency_stop_distance, at the line's own heading. Where the optimum breaks either, the line is pulled in
there and re-optimized. Neither safety threshold is changed by any of this; the line keeps away from them.
"""

from dataclasses import dataclass
import math

import numpy as np

from pure_pursuit.raceline_optimizer import path_frames, resample_closed_path, signed_curvature


@dataclass(frozen=True)
class CarPhysics:
    """What a line may ask of the car. Defaults are car 2's, measured 2026-11-06 (see the module docstring)."""
    a_lat_left: float = 2.45
    a_lat_right: float = 1.95
    a_accel: float = 4.0
    a_brake: float = 3.0
    v_max: float = 5.0
    v_min: float = 0.5
    wheelbase: float = 0.36
    max_steering_angle: float = 0.38   # planning angle, under the controller's 0.42 cap
    linkage_gain: float = 0.99         # wheel angle per commanded angle at full lock
    steering_rate: float = 1.0         # rad/s, pure_pursuit.yaml max_steering_rate
    steering_delay: float = 0.10       # s, the servo's fitted delay (4 ticks at 40 Hz)
    speed_delay: float = 0.10          # s, the speed loop's fitted delay (3 ticks) plus one command period

    def __post_init__(self):
        for name in ('a_lat_left', 'a_lat_right', 'a_accel', 'a_brake', 'v_max', 'v_min', 'wheelbase',
                     'max_steering_angle', 'linkage_gain', 'steering_rate'):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f'{name} must be finite and positive, got {value!r}')
        for name in ('steering_delay', 'speed_delay'):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f'{name} must be finite and >= 0, got {value!r}')
        if self.v_min > self.v_max:
            raise ValueError('v_min cannot exceed v_max')

    @property
    def curvature_limit(self) -> float:
        """Largest |curvature| a line may plan, 1/m."""
        return math.tan(self.linkage_gain * self.max_steering_angle) / self.wheelbase

    @property
    def curvature_rate(self) -> float:
        """Fastest curvature change the steering can make, 1/m per second (small-angle, so conservative)."""
        return self.steering_rate * self.linkage_gain / self.wheelbase


def _segments(xy):
    d = np.roll(xy, -1, axis=0) - xy
    return np.hypot(d[:, 0], d[:, 1])


def speed_profile(xy, physics: CarPhysics, passes: int = 3):
    """(speed, signed curvature, segment lengths) of a closed path for this car. Left turns are positive.

    speed[i] is the fastest the car can be at xy[i] and still make every corner, every steering change and
    every braking zone of the lap.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 4 or not np.isfinite(xy).all():
        raise ValueError('need a closed path of at least 4 finite xy points')
    seg = _segments(xy)
    if np.any(seg <= 0.0):
        raise ValueError('duplicate consecutive points')
    kappa = signed_curvature(xy)
    grip = np.where(kappa >= 0.0, physics.a_lat_left, physics.a_lat_right)
    with np.errstate(divide='ignore'):
        v_corner = np.sqrt(grip / np.maximum(np.abs(kappa), 1e-9))
    cap = np.minimum(v_corner, physics.v_max)
    n = len(xy)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    speed = cap.copy()
    for _ in range(passes):
        # Steering: a curvature change, judged over the distance covered during the servo delay (and at
        # least one segment either side), must fit the rack's rate at the planned speed:
        # v * |dkappa/ds| <= curvature_rate.
        window = np.maximum(speed * physics.steering_delay, 2.0 * np.mean(seg))
        ahead = np.interp((s[:-1] + window) % total, s, np.concatenate([kappa, kappa[:1]]))
        behind = np.interp((s[:-1] - window) % total, s, np.concatenate([kappa, kappa[:1]]))
        slope = np.maximum(np.abs(ahead - kappa), np.abs(kappa - behind)) / window
        with np.errstate(divide='ignore'):
            v_rate = physics.curvature_rate / np.maximum(slope, 1e-9)
        speed = np.minimum(cap, v_rate)
        speed = _accel_brake(speed, seg, physics)
        speed = _respond_early(speed, s, physics.speed_delay)
    return np.maximum(speed, physics.v_min), kappa, seg


def _respond_early(speed, s, delay):
    """Each station's speed no higher than the plan (`speed` as passed in) anywhere within the next
    speed * delay metres: braking points move earlier by the distance covered while the car responds.

    The car reaches a commanded speed `delay` late, so a braking point planned exactly where braking must
    begin arrives too fast: car 2's simulated hairpin entry ran 0.3 m wide at 2.5 m/s against a planned 2.2.
    """
    if delay <= 0.0:
        return speed
    # The minimum is over the plan as it stood before this shift, never over shifted values: windows of
    # neighbouring stations overlap, so taking minima of minima would carry every slow corner backwards
    # round the whole lap (it did: 12.4 s became 21.4 s on the 6 Nov course). The physics is a shift of the
    # braking points by the response delay, not a chain.
    total = s[-1]
    n = len(speed)
    pos = np.concatenate([s[:-1], s[:-1] + total])
    plan = np.concatenate([speed, speed]).astype(float)
    out = speed.astype(float).copy()
    # out[i] = min(plan[i : max(j_i, i + 1)]), vectorized over i: the windows are only a few stations
    # long (speed * delay over the station spacing), so loop over the offset k within a window instead
    # of over the stations (the optimizer calls this once per lap-time evaluation, 2026-11-09).
    idx = np.arange(n)
    j = np.searchsorted(pos, pos[:n] + np.asarray(speed, dtype=float) * delay, side='right')
    width = np.maximum(j - idx, 1)
    for k in range(1, int(width.max())):
        reach = width > k
        out[reach] = np.minimum(out[reach], plan[idx[reach] + k])
    return out


def _accel_brake(cap, seg, physics):
    """Forward (acceleration) and backward (braking) passes round a closed loop."""
    # In squared speed w = v^2 the forward pass w[j] = min(cap[j], w[j-1] + 2 a ds) has the closed form
    # w[j] = 2 a S[j] + min over k <= j of (cap[k] - 2 a S[k]), S the distance along the path: a running
    # minimum. Two laps' worth of stations make the loop's closure exact (no corner is further than a lap
    # behind or ahead); the backward pass is the same thing run in reverse.
    # The two envelopes are each taken from the corner caps and combined with min(): lowering a speed for
    # braking can never break the acceleration limit, which only bounds how fast speed may rise.
    n = len(cap)
    w = np.tile(np.asarray(cap, dtype=float) ** 2, 3)
    ds = np.tile(seg, 3)
    S = np.concatenate([[0.0], np.cumsum(ds)[:-1]])          # distance from the start to station j
    R = np.cumsum(ds[::-1])[::-1]                             # distance from station j to the end
    two_a, two_b = 2.0 * physics.a_accel, 2.0 * physics.a_brake
    accel = two_a * S + np.minimum.accumulate(w - two_a * S)
    brake = two_b * R + np.minimum.accumulate((w - two_b * R)[::-1])[::-1]
    return np.sqrt(np.minimum(accel, brake)[n:2 * n])


def lap_time(xy, physics: CarPhysics) -> float:
    speed, _, seg = speed_profile(xy, physics)
    return float(np.sum(2.0 * seg / (speed + np.roll(speed, -1))))


def cone_clearance(field_at, xy, laser_x=0.26, half_fov=math.radians(30.0), beams=13, max_range=1.5,
                   step_floor=0.01):
    """Shortest forward-cone LiDAR range at each station, driving along the path's own heading.

    `field_at(x, y)` is the distance in metres to the nearest mapped wall (0 inside one). Rays are marched
    by that distance and stop within `step_floor` of a wall, measured perpendicular to it, so a range is
    never longer than the true one and short by at most step_floor / sin(angle between ray and wall):
    the conservative direction for a safety check.
    """
    xy = np.asarray(xy, dtype=float)
    tangents, _ = path_frames(xy)
    heading = np.arctan2(tangents[:, 1], tangents[:, 0])
    ox = xy[:, 0] + laser_x * np.cos(heading)
    oy = xy[:, 1] + laser_x * np.sin(heading)
    angles = heading[:, None] + np.linspace(-half_fov, half_fov, beams)[None, :]
    dx, dy = np.cos(angles), np.sin(angles)
    t = np.zeros_like(angles)
    for _ in range(64):
        d = field_at(ox[:, None] + t * dx, oy[:, None] + t * dy)
        hit = d < step_floor
        t = np.where(hit | (t >= max_range), t, t + np.maximum(d, step_floor))
    return np.minimum(t, max_range).min(axis=1)


def footprint_clearance(field_at, xy, front=0.43, rear=0.15, half_width=0.155):
    """Smallest wall distance of the body's outline at each station (rear-axle reference, path heading).

    Defaults: car_length 0.58 m with base_link 0.15 m ahead of the rear bumper; 0.30 m tire to tire.
    """
    xy = np.asarray(xy, dtype=float)
    tangents, normals = path_frames(xy)
    along = np.linspace(-rear, front, 7)
    across = np.linspace(-half_width, half_width, 5)
    pa, pc = np.meshgrid(along, across)
    outline = (np.abs(pc) == half_width) | (pa == along[0]) | (pa == along[-1])
    pa, pc = pa[outline], pc[outline]
    px = xy[:, 0, None] + pa[None, :] * tangents[:, 0, None] + pc[None, :] * normals[:, 0, None]
    py = xy[:, 1, None] + pa[None, :] * tangents[:, 1, None] + pc[None, :] * normals[:, 1, None]
    return field_at(px, py).min(axis=1)


@dataclass(frozen=True)
class MinTimeLine:
    ok: bool
    xy: np.ndarray
    speed: np.ndarray
    lap_s: float
    start_lap_s: float
    min_cone_m: float
    min_body_m: float
    reason: str


def optimize_min_time(center, width_left, width_right, physics: CarPhysics, field_at, *, start=None,
                      half_width=0.155, wall_margin=0.10, emergency_stop_distance=0.35, cone_margin=0.10,
                      body_floor=0.08, control_spacing=1.0, smoothing=0.5, max_iter=300, repairs=8,
                      body_floor_tight=None,
                      laser_x=0.26):
    """The fastest closed line inside the corridor that keeps every safety margin.

    center: (n, 2) closed, smooth reference line (the minimum-curvature line, normally); width_left/right:
    wall distance either side of it (m). start: optional (n,) lateral offsets to start from.
    The offsets are a periodic cubic spline through control points `control_spacing` apart, so every
    candidate line is smooth: free per-point offsets make kinks that the curvature, and so the speed
    profile, cannot tell from real corners.
    Returns MinTimeLine; ok is False, with the reason, when no line keeps the margins.
    """
    from scipy.interpolate import CubicSpline
    from scipy.optimize import minimize

    center = np.asarray(center, dtype=float)
    left, right = np.asarray(width_left, dtype=float), np.asarray(width_right, dtype=float)
    n = len(center)
    if left.shape != (n,) or right.shape != (n,) or not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise ValueError('width_left and width_right must be finite, one per center point')
    if not (math.isfinite(control_spacing) and control_spacing > 0.0):
        raise ValueError('control_spacing must be finite and positive')
    _, normals = path_frames(center)
    room = half_width + wall_margin
    upper = left - room
    lower = -(right - room)
    if np.any(upper < lower):
        worst = int(np.argmin(upper - lower))
        return MinTimeLine(False, center, np.zeros(n), math.inf, math.inf, 0.0, 0.0,
                           f'the track is {left[worst] + right[worst]:.2f} m wide at point {worst}, under the '
                           f'{2 * room:.2f} m the car needs with a {wall_margin:.2f} m margin each side')
    s = np.concatenate([[0.0], np.cumsum(_segments(center))])
    total = s[-1]
    m = max(8, int(round(total / control_spacing)))
    knots = np.linspace(0.0, total, m + 1)
    owner = np.minimum((s[:-1] / total * m).round().astype(int), m - 1)   # nearest control point per station

    def stations(c):
        return CubicSpline(knots, np.concatenate([c, c[:1]]), bc_type='periodic')(s[:-1])

    def control_bounds():
        lo = np.full(m, -np.inf)
        hi = np.full(m, np.inf)
        np.maximum.at(lo, owner, lower)
        np.minimum.at(hi, owner, upper)
        return lo, np.maximum(hi, lo)

    start = np.zeros(n) if start is None else np.asarray(start, dtype=float)
    control = np.interp(knots[:-1], s[:-1], start)

    def line(o):
        return center + o[:, None] * normals

    limit = physics.curvature_limit

    def cost(c):
        xy = line(stations(c))
        second = np.roll(c, -1) - 2.0 * c + np.roll(c, 1)
        excess = np.maximum(np.abs(signed_curvature(xy)) - limit, 0.0)
        # The steering limit is a wall, not a trade: no speed makes an over-tight corner drivable.
        return (lap_time(xy, physics) + smoothing * float(np.sum(second ** 2))
                + 1000.0 * float(np.sum(excess ** 2)))

    start_lap = lap_time(line(stations(control)), physics)
    need_cone = emergency_stop_distance + cone_margin
    for _ in range(repairs + 1):
        lo, hi = control_bounds()
        control = np.clip(control, lo, hi)
        result = minimize(cost, control, method='L-BFGS-B', bounds=list(zip(lo, hi)),
                          options={'maxiter': max_iter, 'eps': 1e-4})
        control = np.clip(result.x, lo, hi)
        offsets = stations(control)
        xy = line(offsets)
        cone = cone_clearance(field_at, xy, laser_x=laser_x)
        body = footprint_clearance(field_at, xy, half_width=half_width)
        too_tight = np.abs(signed_curvature(xy)) > limit + 0.02
        bad = (cone < need_cone) | (body < body_floor_at(xy, body_floor, body_floor_tight, limit))
        if not bad.any() and not too_tight.any():
            speed, _, _ = speed_profile(xy, physics)
            lap = lap_time(xy, physics)
            return MinTimeLine(True, xy, speed, lap, start_lap, float(cone.min()), float(body.min()),
                               f'{lap:.2f} s (from {start_lap:.2f} s), forward cone {cone.min():.2f} m, '
                               f'body {body.min():.2f} m from the walls')
        if not bad.any():
            break   # only the curvature is left over: pulling the line in would not fix it
        # Pull the line towards the middle at and around each offending station, then re-optimize.
        idx = np.flatnonzero(bad)
        near = np.unique(np.concatenate([(idx + k) % n for k in range(-4, 5)]))
        middle = 0.5 * (upper + lower)
        upper[near] = np.maximum(middle[near], upper[near] - 0.05)
        lower[near] = np.minimum(middle[near], lower[near] + 0.05)
    offsets = stations(control)
    xy = line(offsets)
    speed, _, _ = speed_profile(xy, physics)
    cone = cone_clearance(field_at, xy, laser_x=laser_x)
    body = footprint_clearance(field_at, xy, half_width=half_width)
    tight = float(np.abs(signed_curvature(xy)).max())
    return MinTimeLine(False, xy, speed, lap_time(xy, physics), start_lap, float(cone.min()), float(body.min()),
                       f'no line keeps every limit: forward cone {cone.min():.2f} m (need {need_cone:.2f}), '
                       f'body {body.min():.2f} m (need {body_floor:.2f}), curvature {tight:.2f}/m '
                       f'(limit {limit:.2f}/m)')


def body_floor_at(xy, body_floor, body_floor_tight, curvature_limit, window_m=0.6):
    """The body-clearance floor at each station of a closed line.

    ``body_floor`` on straights; rising linearly to ``body_floor_tight`` as the line's curvature
    goes from half the curvature limit to 90% of it, and held over ``window_m`` either side of
    each such point (corner entry and exit). Tracking error is largest where the steering is
    near its limit and has no reserve to correct (car 2, 2026-11-06 floor: max cross-track
    0.28 m at the 06b hairpin against 0.09 m mean), so that is where the line needs room.
    ``body_floor_tight`` None or equal to ``body_floor`` gives the constant floor.
    """
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    if body_floor_tight is None or body_floor_tight == body_floor:
        return np.full(n, float(body_floor))
    if not (math.isfinite(body_floor_tight) and body_floor_tight >= 0.0):
        raise ValueError('body_floor_tight must be finite and non-negative')
    if not (math.isfinite(curvature_limit) and curvature_limit > 0.0):
        raise ValueError('curvature_limit must be finite and positive')
    k = np.abs(signed_curvature(xy))
    weight = np.clip((k / curvature_limit - 0.5) / 0.4, 0.0, 1.0)
    spacing = float(np.median(_segments(xy))) if n > 1 else 1.0
    half = max(0, int(round(window_m / max(spacing, 1e-6))))
    if half:
        weight = np.max([np.roll(weight, j) for j in range(-half, half + 1)], axis=0)
    return body_floor + (body_floor_tight - body_floor) * weight


def offsets_of(center, line_xy):
    """Signed lateral offsets of `line_xy` from `center`, one per center point (nearest-point projection)."""
    center = np.asarray(center, dtype=float)
    line_xy = resample_closed_path(np.asarray(line_xy, dtype=float), 0.02)
    _, normals = path_frames(center)
    d = line_xy[None, :, :] - center[:, None, :]
    nearest = np.argmin(np.hypot(d[..., 0], d[..., 1]), axis=1)
    delta = line_xy[nearest] - center
    return delta[:, 0] * normals[:, 0] + delta[:, 1] * normals[:, 1]
