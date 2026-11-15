"""Contingency planner for passing a slower car. No ROS.

Scope (tech lead, 2026-11-08): "as long as the car physically fits it should
attempt to pass". Any car ahead that is slower than the line's pace by more
than its speed bound's margin, and not accelerating, is a candidate (the node
decides that; see _local_pass_update). On the 2026-11-06 courses the longest
single-side stretch with room for two cars is 8.4 m (06a) / 6.0 m (06b), so
there it nearly always declines (docs/racing-autonomy.md, "Passing: stage 2").

Frame. Everything is in the closed racing line's Frenet frame: s = arc
length, d = signed lateral offset, left positive. Offset points use
interpolated vertex normals, so an offset path has no kinks at stations.

The opponent (StalledOpponent) is an *anchor*, not a prediction: the arc of
its rear face, measured while following it from directly behind (where the
cluster centroid is that face), and its lateral centre. Its front is
bounded by rear + opponent_length (0.62 m: our 0.58 m chassis plus 4 cm);
length_extra adds a car length when the sighting may be any face of it.
speed_bound is its speed measured from behind + pass_speed_uncertainty
(0.5 m/s by default), and never falls during a manoeuvre.

Two opponent motion models, both deliberate:
  * Unseen time (age since the last consistent sighting): it may have moved
    forward at speed_bound the whole time, with no braking. This is the
    approved rejoin bound (2026-11-08): car 2's +/-135 deg LiDAR loses a car
    it is passing ~0.35 m ahead, so a measured lead can never reach 1 m.
  * Each tick's contingency check, over car 2's own stopping time: it may
    keep moving at speed_bound for opponent_react (0.3 s), then brakes at
    opponent_brake. A car that is being passed and keeps accelerating into
    the car beside it is not a model any planner can be safe against.

The contingency check (fallback_clear) is the safety core: from the current
pose and speed V, car 2 coasts for `response` (0.45 s: transport + speed-loop
lag + sensing, as stage 1) then brakes at 3 m/s^2 along the plan; every body
pose up to the stop, and the stopped pose forever after, must clear the walls
by wall_margin and the opponent's reachable box by car_margin/long_margin.
A pass continues only while that holds; the node aborts otherwise.
"""
from dataclasses import dataclass, replace
import math

import numpy as np

from pure_pursuit import racing_math, tracking_laws


@dataclass(frozen=True)
class Limits:
    wheelbase: float = 0.36
    length: float = 0.58
    width: float = 0.31
    steering: float = 0.42
    lateral_accel: float = 2.7        # total, line curvature + shift (car 2's online cap)
    shift_accel: float = 2.0          # lateral accel budget that sizes a shift's length
    accel: float = 4.0                # car 2's measured speed loop
    brake: float = 3.0
    response: float = 0.45
    # Two tiers. A pass is planned, and its speed chosen, with the first; it
    # is abandoned only when the second fails. With one tier the car's own
    # tracking error tripped the margins it was planned to (indoor_wide,
    # 2026-11-09: most aborts at 0.08-0.10 m from the wall at the car's own
    # pose), and a car stopped at a margin could never move off again.
    wall_margin: float = 0.15         # body to wall, planning
    car_margin: float = 0.08          # body to the opponent's reachable box, laterally, planning
    long_margin: float = 0.15         # same, longitudinally, planning
    keep_wall_margin: float = 0.05    # the same three, to keep going (and to move off from a stop)
    keep_car_margin: float = 0.04
    keep_long_margin: float = 0.08
    max_offset: float = 0.85          # inside max_cross_track_error (1.0 m)
    steering_headroom: float = 0.8    # a pass path may need at most this share of the steering limit
    min_shift: float = 1.5            # metres
    opponent_length: float = 0.62     # an F1TENTH car (ours is 0.58 m) plus 4 cm
    opponent_width: float = 0.32
    opponent_lateral_uncertainty: float = 0.08
    opponent_long_uncertainty: float = 0.10
    opponent_react: float = 0.3
    opponent_brake: float = 3.0
    opponent_heading: float = 0.10    # rad; a racing car holds its line within its tracking error
    clear_margin: float = 0.5         # worst-case rejoin body gap (pass_clear_margin)
    max_closing: float = 2.0          # m/s over the car's speed bound while a pass is under way
    max_unseen: float = 1.0           # approved: a sighting within the last second
    spacing: float = 0.10
    horizon: float = 12.0
    body_grid: tuple = (9, 5)         # body sample points, along x across; finer = less slack in build()


def keeping(limits):
    """The keep-going tier of `limits`: what an executing pass must still satisfy."""
    return replace(limits, wall_margin=limits.keep_wall_margin, car_margin=limits.keep_car_margin,
                   long_margin=limits.keep_long_margin)


@dataclass(frozen=True)
class StalledOpponent:
    rear_arc: float       # anchor: rear-face arc on the line
    lateral: float        # anchor: lateral centre
    speed_bound: float    # tracked speed + uncertainty, m/s
    age: float            # seconds since the last consistent sighting
    length_extra: float = 0.0   # a car that moved: the sighting may be any face of it


@dataclass
class Plan:
    mode: str                         # 'pass', 'return' or 'hold'
    context: tracking_laws.PathContext
    s: np.ndarray                     # arc of each sample, unwrapped from the ego's arc
    d: np.ndarray
    heading: np.ndarray               # relative to the line tangent
    dist: np.ndarray                  # distance along the path from sample 0
    wall: np.ndarray                  # body-to-wall clearance at each sample, metres
    side: int = 0
    target: float = 0.0


class Line:
    """Closed racing line as a Frenet frame."""

    def __init__(self, xy, speed):
        xy = np.asarray(xy, float)
        speed = np.asarray(speed, float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 4 or speed.shape != (len(xy),):
            raise ValueError('line needs >= 4 points and one speed per point')
        if not (np.all(np.isfinite(xy)) and np.all(np.isfinite(speed))):
            raise ValueError('line must be finite')
        seg = np.linalg.norm(np.roll(xy, -1, 0) - xy, axis=1)
        if np.any(seg < 1e-6):
            raise ValueError('line has a repeated point')
        self.xy, self.speed, self.seg = xy, speed, seg
        self.cum = np.r_[0.0, np.cumsum(seg[:-1])]
        self.total = float(seg.sum())
        tangent = (np.roll(xy, -1, 0) - xy) / seg[:, None]
        seg_normal = np.stack((-tangent[:, 1], tangent[:, 0]), 1)
        vertex = seg_normal + np.roll(seg_normal, 1, 0)
        self.normal = vertex / np.linalg.norm(vertex, axis=1)[:, None]
        self.tangent = tangent
        self.curvature = racing_math.estimate_signed_path_curvature(xy, closed=True)

    def _locate(self, s):
        s = np.asarray(s, float) % self.total
        i = np.clip(np.searchsorted(self.cum, s, side='right') - 1, 0, len(self.xy) - 1)
        f = (s - self.cum[i]) / self.seg[i]
        return i, (i + 1) % len(self.xy), f

    def point(self, s, d):
        i, j, f = self._locate(s)
        f = f[..., None]
        base = self.xy[i] * (1 - f) + self.xy[j] * f
        normal = self.normal[i] * (1 - f) + self.normal[j] * f
        normal /= np.linalg.norm(normal, axis=-1)[..., None]
        return base + np.asarray(d, float)[..., None] * normal

    def heading(self, s):
        i, _, _ = self._locate(s)
        return np.arctan2(self.tangent[i, 1], self.tangent[i, 0])

    def sample(self, values, s):
        i, j, f = self._locate(s)
        return values[i] * (1 - f) + values[j] * f

    def frenet(self, x, y, hint=None):
        """(s, d) of a world point: projection onto the nearest segment."""
        p = np.array([x, y], float)
        if not np.all(np.isfinite(p)):
            return math.nan, math.nan
        n = len(self.xy)
        idx = range(n) if hint is None else [(hint + k) % n for k in range(-8, 9)]
        best = (math.inf, 0.0, 0.0)
        for i in idx:
            a, b = self.xy[i], self.xy[(i + 1) % n]
            t = float(np.clip(np.dot(p - a, b - a) / self.seg[i] ** 2, 0.0, 1.0))
            q = a + t * (b - a)
            dist = float(np.linalg.norm(p - q))
            if dist < best[0]:
                best = (dist, float(self.cum[i] + t * self.seg[i]), i)
        s = best[1] % self.total
        side = float(np.dot(p - self.point(s, 0.0), self.point(s, 1.0) - self.point(s, 0.0)))
        return s, math.copysign(best[0], side)

    def wrap(self, delta):
        return (delta + self.total / 2) % self.total - self.total / 2


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)


QUINTIC_PEAK = 10 * math.sqrt(3) / 3   # max of the quintic smoothstep's second derivative


def shift_length(delta_d, speed, limits):
    """Length of a quintic shift of delta_d.

    Its curvature peaks at 5.77*delta/L^2: keep v^2 times that inside
    shift_accel, and the curvature itself inside 80% of what the steering
    limit allows. build() then refuses any path whose total curvature (the
    line's own bend plus the shift) still needs more steering than the car has.
    """
    curvature = 0.8 * limits.steering_headroom * math.tan(limits.steering) / limits.wheelbase
    return max(limits.min_shift,
               speed * math.sqrt(QUINTIC_PEAK * abs(delta_d) / limits.shift_accel),
               math.sqrt(QUINTIC_PEAK * abs(delta_d) / curvature))


def lateral_profile(s, d0, target, shift_out, s_return, shift_back):
    """d(s): d0 -> target over shift_out, hold, target -> 0 from s_return over shift_back."""
    d = d0 + (target - d0) * smooth(s / shift_out)
    if s_return is not None:
        d = d * (1 - smooth((s - s_return) / shift_back))
    return d


def body_points(limits):
    rear = (limits.wheelbase - limits.length) / 2
    nx, ny = limits.body_grid
    gx, gy = np.meshgrid(np.linspace(rear, rear + limits.length, nx),
                         np.linspace(-limits.width / 2, limits.width / 2, ny))
    # Any body point is within half a cell diagonal of a sample point.
    slack = math.hypot(limits.length / (2 * (nx - 1)), limits.width / (2 * (ny - 1)))
    return gx.ravel(), gy.ravel(), slack


def build_slack(limits, curvature):
    """What build() subtracts from a sample's wall clearance for a path of this curvature: the body grid's
    slack, half a sample step and the body's swing over half a step."""
    _, _, slack = body_points(limits)
    radius = math.hypot(limits.length, limits.width / 2)
    return slack + limits.spacing / 2 + radius * abs(curvature) * limits.spacing / 2


def build(line, s0, d, mode, limits, walls, *, side=0, target=0.0):
    """Plan from Frenet samples.

    A sample needing more steering than the car has is marked blocked, like a
    wall: a stop that reaches it is refused, and an entry needs none anywhere
    along the whole manoeuvre, but a bend far beyond the stop does not veto
    the plan. None only for degenerate geometry.
    """
    s_rel = np.arange(len(d)) * limits.spacing
    s = s0 + s_rel
    xy = line.point(s, d)
    if not np.all(np.isfinite(xy)):
        return None
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if np.any(step < 1e-6):
        return None
    seg = np.r_[step, step[-1]]
    curvature = racing_math.estimate_signed_path_curvature(xy, closed=False)  # signed: the laws feed it forward
    # Leave the tracking law headroom: a path needing the full limit cannot be
    # followed through delay_lqr's modelled delays (indoor_wide, 2026-11-08:
    # pass paths at full lock ran wide into the wall in 5 of 6 blocker races).
    unsteerable = (np.abs(np.arctan(limits.wheelbase * curvature))
                   > limits.steering_headroom * limits.steering)
    speed = np.minimum(line.sample(line.speed, s),
                       np.sqrt(limits.lateral_accel / np.maximum(np.abs(curvature), 1e-9)))
    for i in range(len(speed) - 2, -1, -1):
        speed[i] = min(speed[i], math.sqrt(speed[i + 1] ** 2 + 2 * limits.brake * step[i]))
    yaw = np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))
    yaw = np.r_[yaw, yaw[-1]]
    heading = (yaw - line.heading(s) + math.pi) % (2 * math.pi) - math.pi
    gx, gy, slack = body_points(limits)
    c, sn = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    px = xy[:, 0:1] + c * gx - sn * gy
    py = xy[:, 1:2] + sn * gx + c * gy
    clear = np.asarray(walls(px.ravel(), py.ravel()), float).reshape(px.shape)
    clear = np.where(np.isfinite(clear), clear, -math.inf).min(axis=1)
    # Motion between samples: half a step plus the body's swing.
    turn = np.abs(np.r_[np.diff(np.unwrap(yaw)), 0.0])
    radius = math.hypot(limits.length, limits.width / 2)
    clear = clear - slack - seg / 2 - radius * turn / 2
    clear = np.where(unsteerable, -math.inf, clear)
    cumulative = racing_math.compute_cumulative_arc_length(seg)
    context = tracking_laws.PathContext(xy, seg, cumulative, speed, curvature, False)
    return Plan(mode, context, s, np.asarray(d, float), heading, np.r_[0.0, np.cumsum(step)],
                clear, side, target)


def reach_long(t, v, limits):
    """Opponent travel within t of now under the per-tick model (react, then brake)."""
    t = np.asarray(t, float)
    react = np.minimum(t, limits.opponent_react)
    after = np.clip(t - limits.opponent_react, 0.0, v / limits.opponent_brake)
    return v * react + v * after - limits.opponent_brake * after ** 2 / 2


def opponent_box(opponent, t, limits):
    """(s_lo, s_hi, lateral half-width) of everywhere the opponent's body can be."""
    unseen = opponent.speed_bound * opponent.age
    travel = unseen + reach_long(t, opponent.speed_bound, limits)
    lo = opponent.rear_arc - limits.opponent_long_uncertainty
    hi = (opponent.rear_arc + limits.opponent_length + opponent.length_extra
          + limits.opponent_long_uncertainty + travel)
    half = (limits.opponent_width / 2 + limits.opponent_lateral_uncertainty
            + math.sin(limits.opponent_heading) * travel)
    return lo, hi, half


def braking_times(dist, speed, limits):
    """Time to reach each distance while coasting `response` then braking; inf if never."""
    dist = np.asarray(dist, float)
    if speed <= 0:
        return np.where(dist <= 0, 0.0, math.inf)
    coast = speed * limits.response
    times = dist / speed
    beyond = dist > coast
    left = speed ** 2 - 2 * limits.brake * (dist - coast)
    braking = limits.response + (speed - np.sqrt(np.maximum(left, 0.0))) / limits.brake
    times = np.where(beyond, np.where(left >= 0, braking, math.inf), times)
    return times


def stop_distance(speed, limits):
    return speed * limits.response + speed ** 2 / (2 * limits.brake)


def _overlaps(plan, k, t, opponent, line, limits):
    rear = (limits.wheelbase - limits.length) / 2
    centre_offset = rear + limits.length / 2
    h = plan.heading[k]
    stretch = max(1e-3, 1 - plan.d[k] * line.sample(line.curvature, plan.s[k]))
    s_c = plan.s[k] + centre_offset * math.cos(h) / stretch
    d_c = plan.d[k] + centre_offset * math.sin(h)
    long_half = (limits.length / 2 * abs(math.cos(h)) + limits.width / 2 * abs(math.sin(h))) / stretch
    lat_half = limits.width / 2 * abs(math.cos(h)) + limits.length / 2 * abs(math.sin(h))
    lo, hi, half = opponent_box(opponent, t, limits)
    rel_lo = line.wrap(lo - plan.s[0]) + plan.s[0]
    rel_hi = rel_lo + (hi - lo)
    longitudinal = (s_c + long_half + limits.long_margin > rel_lo
                    and s_c - long_half - limits.long_margin < rel_hi)
    lateral = abs(d_c - opponent.lateral) < lat_half + half + limits.car_margin
    return longitudinal and lateral


def fallback_clear(plan, speed, opponent, line, limits, t0=0.0):
    """Can the car stop along this plan from `speed`, clear of walls and the opponent?

    t0 is when the car is at the plan's first sample, seconds from now (the
    opponent's reachable box keeps growing until then).
    """
    if plan is None or not math.isfinite(speed) or speed < 0 or not math.isfinite(t0) or t0 < 0:
        return False
    stop = stop_distance(speed, limits)
    reached = plan.dist <= stop + limits.spacing
    if plan.dist[-1] < stop + limits.spacing:
        return False   # the plan must contain the whole stop
    if np.any(plan.wall[reached] < limits.wall_margin):
        return False
    if opponent is None:
        return True
    times = braking_times(plan.dist, speed, limits) + t0
    last = int(np.flatnonzero(reached)[-1])
    for k in np.flatnonzero(reached):
        t = times[k] if math.isfinite(times[k]) else t0 + limits.response + speed / limits.brake
        if _overlaps(plan, k, t, opponent, line, limits):
            return False
    # The stopped car stays there; the opponent's reach saturates. The stop
    # lies between two samples: the one before it is the conservative pose
    # for a car closing from behind, the one after it for a car ahead.
    before = int(np.flatnonzero(plan.dist <= stop)[-1])
    return not any(_overlaps(plan, k, math.inf, opponent, line, limits) for k in {before, last})


def _suffix(plan, k):
    """The plan from sample k on, distances re-based (the tracked path is unchanged)."""
    return Plan(plan.mode, plan.context, plan.s[k:], plan.d[k:], plan.heading[k:],
                plan.dist[k:] - plan.dist[k], plan.wall[k:], plan.side, plan.target)


def progress_clear(plan, opponent, line, limits, speed=0.6, every=3):
    """Could the car crawl the whole plan at `speed`, able to stop clear from every point?

    fallback_clear only proves the car can stop now. A pass that may start
    must also be able to get somewhere: this checks the stop from each point
    along the way, at the time the crawl would reach it.
    """
    if plan is None or speed <= 0:
        return False
    for k in range(0, len(plan.dist), every):
        sub = _suffix(plan, k)
        if sub.dist[-1] < stop_distance(speed, limits) + limits.spacing:
            break
        if not fallback_clear(sub, speed, opponent, line, limits, t0=float(plan.dist[k]) / speed):
            return False
    return True


def box_free_distance(plan, opponent, line, limits):
    """Distance along the plan to the first pose inside the opponent's box at its full reach.

    inf when the planned path clears everywhere the opponent could ever be
    (its box once its reach has saturated): a pass on such a path can always
    keep going. A finite value means the path, still shifting out, would
    meet the box there; no stop may run past it.
    """
    if plan is None:
        return 0.0
    if opponent is None:
        return math.inf
    for k in range(len(plan.dist)):
        if _overlaps(plan, k, math.inf, opponent, line, limits):
            return float(plan.dist[k])
    return math.inf


def safe_speed(plan, opponent, line, limits, current_speed, grid=0.25):
    """Largest commandable speed whose stop is clear; None if even the current one is not.

    Also never faster than stops short of where the path meets the
    opponent's full-reach box (box_free_distance): a car allowed whatever
    stops clear *now* would otherwise run itself into a pass it cannot finish.
    """
    if plan is None or not math.isfinite(current_speed) or current_speed < 0:
        return None
    # Still safe to keep going at all (keep tier)? Then pick the speed with
    # the planning tier, braking towards it if even the current one fails it.
    if not fallback_clear(plan, current_speed, opponent, line, keeping(limits)):
        return None
    top = float(plan.context.speed_profile[0])
    free = box_free_distance(plan, opponent, line, limits) - 2 * limits.spacing
    if math.isfinite(free):
        r, b = limits.response, limits.brake
        top = min(top, max(0.0, -b * r + math.sqrt((b * r) ** 2 + 2 * b * max(free, 0.0))))
    candidates = np.arange(math.floor(top / grid) * grid, current_speed, -grid)
    for v in candidates:
        if fallback_clear(plan, float(v), opponent, line, limits):
            return float(min(v, top))
    if fallback_clear(plan, current_speed, opponent, line, limits):
        return float(min(current_speed, top))
    for v in np.arange(math.floor(current_speed / grid) * grid, -grid / 2, -grid):
        if fallback_clear(plan, float(v), opponent, line, limits):
            return float(min(v, top))
    return 0.0


def worst_case_lead(ego_rear_arc, opponent, line, limits):
    """Approved rejoin bound: ego rear minus the furthest the opponent's front can be."""
    if opponent.age > limits.max_unseen:
        return -math.inf
    front = (opponent.rear_arc + limits.opponent_length + opponent.length_extra
             + opponent.speed_bound * opponent.age)
    return line.wrap(ego_rear_arc - front)


def target_offset(opponent, side, limits):
    """Lateral target clearing the opponent's whole reachable box, or None."""
    # Sized for the longest it may go unseen (max_unseen) plus a full stop.
    worst = StalledOpponent(opponent.rear_arc, opponent.lateral, opponent.speed_bound,
                            max(opponent.age, limits.max_unseen), opponent.length_extra)
    _, _, half = opponent_box(worst, math.inf, limits)
    target = opponent.lateral + side * (half + limits.width / 2 + limits.car_margin + 0.05)
    return None if abs(target) > limits.max_offset else float(target)


@dataclass(frozen=True)
class Manoeuvre:
    """A lateral manoeuvre anchored to the line (absolute arcs), fixed when it starts.

    Re-planning every tick from the car's own position would restart a smooth
    shift each time, and a smooth shift starts with zero slope: the car would
    never actually move across. Each tick instead re-samples this one profile
    from wherever the car now is.
    """
    s_start: float
    d_start: float
    target: float
    out: float              # shift length, metres
    s_return: float         # absolute arc where the shift back starts; inf = hold
    back: float
    side: int = 0

    def lateral(self, s_abs, line):
        s_rel = line.wrap(np.asarray(s_abs, float) - self.s_start)
        s_rel = np.maximum(s_rel, 0.0)
        ret = None if not math.isfinite(self.s_return) else line.wrap(self.s_return - self.s_start)
        return lateral_profile(s_rel, self.d_start, self.target, self.out, ret, self.back)

    def held(self):
        """The same manoeuvre with no shift back: it holds its offset."""
        return replace(self, s_return=math.inf)


@dataclass(frozen=True)
class Bypass:
    """A stopped car's bypass: a crawl path found by rollout (crawl_rollout), anchored to the line once as offsets
    ``d`` at arcs ``s_rel`` past ``s_start``, then a smooth shift back to the line over ``back``.

    The same interface as Manoeuvre, so manoeuvre_plan, build and every stop check treat it unchanged.
    """
    s_start: float
    s_rel: tuple
    d: tuple
    back: float
    side: int
    target: float

    @property
    def d_start(self):
        return self.d[0]

    @property
    def out(self):
        return self.s_rel[-1]

    @property
    def s_return(self):
        return self.s_start + self.s_rel[-1]

    def lateral(self, s_abs, line):
        from scipy.interpolate import CubicSpline
        r = np.maximum(line.wrap(np.asarray(s_abs, float) - self.s_start), 0.0)
        end = self.s_rel[-1]
        # A C2 spline through the rollout's points: linear interpolation would put a curvature spike at each
        # point, which build() reads as unsteerable.
        spline = CubicSpline(self.s_rel, self.d, bc_type='natural')
        d = spline(np.clip(r, 0.0, end))
        if math.isfinite(self.back):
            # Back to the line by a quintic Hermite that starts with the path's own offset and slope (no kink
            # where the path ends) and ends on the line with zero slope and curvature.
            t = np.clip((r - end) / self.back, 0.0, 1.0)
            h0 = 1 - 10 * t ** 3 + 15 * t ** 4 - 6 * t ** 5
            h1 = t - 6 * t ** 3 + 8 * t ** 4 - 3 * t ** 5
            ret = self.d[-1] * h0 + float(spline(end, 1)) * self.back * h1
            d = np.where(r > end, ret, d)
        return d

    def held(self):
        return replace(self, back=math.inf)


def nominal_speed(line, s0, ahead=10.0):
    """The line's mean profile speed over the next `ahead` metres: the pace a pass runs at."""
    return float(np.mean(line.sample(line.speed, s0 + np.arange(0.0, ahead, 0.5))))


def new_pass(line, pose, speed, opponent, limits, side, nominal):
    """The pass manoeuvre from here, or None: offset, outrunning the bound, length."""
    s0, d0 = line.frenet(pose[0], pose[1])
    if not math.isfinite(s0) or side not in (-1, 1):
        return None
    target = target_offset(opponent, side, limits)
    if target is None:
        return None
    rel_rear = line.wrap(opponent.rear_arc - s0)
    if rel_rear < -limits.opponent_length:
        return None    # it is beside or behind us: that is a return, not a pass
    if not math.isfinite(nominal) or nominal - opponent.speed_bound <= 0.2:
        return None    # the unseen bound would never fall behind
    # Sized at the speed the car has now; the plan's speed profile then holds
    # it to what that curvature allows (lateral_accel), so it cannot be run faster.
    out = shift_length(target - d0, speed, limits)
    # Nominal: the front corner leaves the scan ~0.35 m before ego's rear
    # passes it; from there ego gains (v - speed_bound) on the unseen bound.
    rear_offset = (limits.wheelbase - limits.length) / 2
    unseen = max(0.0, limits.clear_margin - 0.35) / (1 - opponent.speed_bound / nominal)
    s_return = max(out, rel_rear + limits.opponent_length + 0.35 + unseen - rear_offset)
    back = shift_length(target, nominal, limits)
    return Manoeuvre(s0, d0, target, out, s0 + s_return, back, side)


def new_return(line, pose, speed, limits):
    s0, d0 = line.frenet(pose[0], pose[1])
    if not math.isfinite(s0):
        return None
    back = shift_length(d0, max(speed, 1.0), limits)
    return Manoeuvre(s0, d0, d0, 1.0, s0, back, 0)


def manoeuvre_plan(line, pose, speed, manoeuvre, walls, limits, *, mode, hold=False):
    """Sample the anchored manoeuvre ahead of the car.

    hold=True drops the shift back (a pass holds its lane until the rejoin
    bound clears); otherwise the plan runs through the shift back to the line.
    """
    if manoeuvre is None:
        return None
    s0, _ = line.frenet(pose[0], pose[1])
    if not math.isfinite(s0):
        return None
    if hold:
        m = manoeuvre.held()
        ahead = max(0.0, line.wrap(manoeuvre.s_start + manoeuvre.out - s0))
    else:
        m = manoeuvre
        ahead = max(0.0, line.wrap(manoeuvre.s_return + manoeuvre.back - s0))
    # Long enough for a stop from the fastest the line allows along it.
    fastest = max(speed, float(np.max(line.sample(line.speed, s0 + np.arange(0.0, ahead + 3.0, 0.5)))))
    length = ahead + stop_distance(fastest, limits) + 1.5
    if length > 40.0:
        return None
    s_rel = np.arange(0.0, length, limits.spacing)
    d = m.lateral(s0 + s_rel, line)
    # Start from where the car really is, not where the manoeuvre says it
    # should be: carry the measured offset and blend it out, so every check
    # sees the actual body (review 2026-11-08). The blend is as long as a
    # steerable shift of that size: a fixed metre made a 0.15 m tracking
    # error need ~0.9 1/m of curvature and aborted passes as "unsteerable".
    _, d0 = line.frenet(pose[0], pose[1])
    if math.isfinite(d0):
        blend = shift_length(abs(d0 - d[0]), max(speed, 1.0), limits)
        d = d + (d0 - d[0]) * (1 - smooth(s_rel / blend))
    return build(line, s0, d, mode, limits, walls, side=manoeuvre.side, target=manoeuvre.target)


def plan_hold(line, pose, speed, walls, limits):
    s0, d0 = line.frenet(pose[0], pose[1])
    if not math.isfinite(s0):
        return None
    length = stop_distance(max(speed, 0.5), limits) + 1.0
    return build(line, s0, np.full(len(np.arange(0.0, length, limits.spacing)), d0), 'hold',
                 limits, walls)


def new_hold(line, pose, speed, limits, side=0):
    """Carry on at the current offset (a stopped abort's way forward past a car beside it)."""
    s0, d0 = line.frenet(pose[0], pose[1])
    if not math.isfinite(s0):
        return None
    back = shift_length(d0, max(speed, 1.0), limits)
    return Manoeuvre(s0, d0, d0, 1.0, math.inf, back, side)


def plan_return(line, pose, speed, walls, limits):
    """A fresh shift back to the line from here (the abort's checked rejoin)."""
    return manoeuvre_plan(line, pose, speed, new_return(line, pose, speed, limits), walls, limits,
                          mode='return')


def entry(line, pose, speed, opponent, walls, limits, *, nominal):
    """The best feasible pass: whole nominal manoeuvre wall-clear, stop clear now.

    Returns (plan, manoeuvre, speed_cap, reason); plan is None with a reason
    per side when there is no pass.
    """
    best, reasons = None, []
    for side in (-1, 1):
        name = 'left' if side > 0 else 'right'
        if target_offset(opponent, side, limits) is None:
            reasons.append(f'{name}: needs more than {limits.max_offset:.2f}m offset')
            continue
        manoeuvre = new_pass(line, pose, speed, opponent, limits, side, nominal)
        whole = manoeuvre_plan(line, pose, speed, manoeuvre, walls, limits, mode='pass')
        if whole is None:
            reasons.append(f'{name}: no pass path (steering, speed bound or length)')
            continue
        if np.any(whole.wall < limits.wall_margin):
            k = int(np.argmax(whole.wall < limits.wall_margin))
            what = 'wall' if math.isfinite(whole.wall[k]) else 'unsteerable'
            reasons.append(f'{name}: {what} {whole.dist[k]:.1f}m along a {whole.dist[-1]:.1f}m pass')
            continue
        plan = manoeuvre_plan(line, pose, speed, manoeuvre, walls, limits, mode='pass', hold=True)
        if math.isfinite(box_free_distance(plan, opponent, line, limits)):
            reasons.append(f'{name}: the pass lane meets the car\'s full-reach box')
            continue
        # A crawl is enough to start: the cap rises as the shift clears the
        # car. new_pass already required the pass speed to outrun the bound.
        v = safe_speed(plan, opponent, line, limits, speed)
        if v is None or v < 0.3:
            reasons.append(f'{name}: no checked stop at a crawl')
            continue
        score = float(whole.wall.min())
        if best is None or score > best[0]:
            best = (score, plan, manoeuvre, v)
    if best is None:
        return None, None, None, '; '.join(reasons)
    return best[1], best[2], best[3], ''


def explain(plan, speed, opponent, line, limits):
    """Why fallback_clear refuses, for the decision log (empty when it is clear)."""
    if plan is None:
        return 'no steerable path'
    stop = stop_distance(speed, limits)
    if plan.dist[-1] < stop + limits.spacing:
        return f'path {plan.dist[-1]:.1f}m shorter than the {stop:.1f}m stop'
    reached = plan.dist <= stop + limits.spacing
    walls = np.flatnonzero(reached & (plan.wall < limits.wall_margin))
    if walls.size:
        if not math.isfinite(plan.wall[walls[0]]):
            return f'unsteerable at {plan.dist[walls[0]]:.1f}m of the stop'
        return f'wall {plan.wall[walls[0]]:.2f}m at {plan.dist[walls[0]]:.1f}m of the stop'
    if opponent is not None:
        times = braking_times(plan.dist, speed, limits)
        for k in np.flatnonzero(reached):
            t = times[k] if math.isfinite(times[k]) else limits.response + speed / limits.brake
            if _overlaps(plan, k, t, opponent, line, limits):
                lo, hi, half = opponent_box(opponent, t, limits)
                return (f'car reachable at {plan.dist[k]:.1f}m (d={plan.d[k]:+.2f}, t={t:.2f}s; '
                        f'car d={opponent.lateral:+.2f} +/-{half:.2f}, arc {line.wrap(lo - plan.s[k]):+.2f}'
                        f'..{line.wrap(hi - plan.s[k]):+.2f} from ours, bound {opponent.speed_bound:.2f}m/s '
                        f'unseen {opponent.age:.2f}s)')
        if _overlaps(plan, int(np.flatnonzero(reached)[-1]), math.inf, opponent, line, limits):
            return 'stopped car reaches the stopped pose'
    return ''


CAR_BUFFER = 0.02


def crawl_rollout(line, pose, opponent, walls, limits, *, levels=5, segments=None, seg=0.5, step=None, beam=40,
                  crawl=1.0, rate=2.5, past=0.3):
    """A crawl path round a stopped car, or None.

    Beam search over piecewise-constant steering: ``segments`` pieces of ``seg`` metres (None: enough to clear
    the box's front along the line from here, plus two), each one of ``levels``
    steering angles across the usable range (steering_headroom of the limit), the rack slewing at ``rate`` rad/s
    while crawling at ``crawl`` m/s, integrated as a kinematic bicycle about the rear axle every ``step`` metres
    (None: the plan's sample spacing).
    A rollout survives while every body point clears the walls by wall_margin plus build()'s sampling slack, and
    its body's bounding box clears the opponent's reachable box (opponent_box, full reach) as _overlaps() judges
    it, in the frame of the line's tangent at the box; it succeeds once the
    rear face is ``past`` metres beyond the box's front with the heading within 0.6 rad of the line's. Of the
    successes the one with the most clearance wins. Returns the rear-axle points (n x 2), the first being ``pose``.
    The shapes the shift-and-hold manoeuvre allows (a constant offset, ramps of at least min_shift) do not fit
    beside a car parked in a bend 1-2 m ahead; on the 2026-11-11 course every stall seed has such a path.
    """
    # One rollout pose per plan sample (None: limits.spacing), so build() checks the poses checked here.
    step = limits.spacing if step is None else step
    lo, hi, half = opponent_box(opponent, math.inf, limits)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    sc = (lo + hi) / 2
    cx, cy = line.point(np.array([sc]), np.array([opponent.lateral]))[0]
    h = float(line.heading(np.array([sc]))[0])
    ch, sh, half_len = math.cos(h), math.sin(h), (hi - lo) / 2
    gx, gy, _ = body_points(limits)
    rear = (limits.wheelbase - limits.length) / 2
    # The rollout charges what build() will, so a path it finds survives build()'s check.
    wall_need = limits.wall_margin + build_slack(limits, math.tan(limits.steering * limits.steering_headroom)
                                                 / limits.wheelbase)
    # 85% of what build() allows: at a crawl the rack reaches its angle within a sample, and build()'s
    # three-point curvature estimate overshoots that step by ~13% (1.18 against 1.04 1/m, 2026-11-15).
    steer = np.linspace(-1.0, 1.0, levels) * limits.steering * limits.steering_headroom * 0.85
    slew = rate * step / crawl
    # state rows: x, y, yaw, delta, clearance; paths as lists of points
    x0, y0, yaw0 = (float(v) for v in pose)
    if segments is None:
        need = (cx - x0) * ch + (cy - y0) * sh + half_len + past - rear
        segments = int(min(12, max(2, math.ceil(need / seg) + 2)))
    states = [(x0, y0, yaw0, 0.0, math.inf, [(x0, y0)])]
    wins = []
    n_sub = max(1, int(round(seg / step)))
    for _ in range(segments):
        grown = []
        for (x, y, yaw, delta, clear, path) in states:
            for target in steer:
                grown.append([x, y, yaw, delta, clear, list(path), target, True])
        if not grown:
            break
        X = np.array([g[0] for g in grown]); Y = np.array([g[1] for g in grown]); YAW = np.array([g[2] for g in grown])
        D = np.array([g[3] for g in grown]); C = np.array([g[4] for g in grown]); T = np.array([g[6] for g in grown])
        alive = np.ones(len(grown), bool); won = np.zeros(len(grown), bool)
        for _ in range(n_sub):
            D = D + np.clip(T - D, -slew, slew)
            X = X + np.cos(YAW) * step
            Y = Y + np.sin(YAW) * step
            YAW = YAW + np.tan(D) / limits.wheelbase * step
            c, sn = np.cos(YAW)[:, None], np.sin(YAW)[:, None]
            px = X[:, None] + c * gx - sn * gy
            py = Y[:, None] + sn * gx + c * gy
            wall = np.asarray(walls(px.ravel(), py.ravel()), float).reshape(px.shape)
            wall = np.where(np.isfinite(wall), wall, -math.inf).min(axis=1)
            # The car test _overlaps() applies: our body's bounding box in the line's frame (here the box's
            # frame, the line's tangent at the box) against the box, with long_margin along and car_margin across.
            rel = YAW - h
            centre = rear + limits.length / 2
            lxc = (X + np.cos(YAW) * centre - cx) * ch + (Y + np.sin(YAW) * centre - cy) * sh
            lyc = -(X + np.cos(YAW) * centre - cx) * sh + (Y + np.sin(YAW) * centre - cy) * ch
            long_half = limits.length / 2 * np.abs(np.cos(rel)) + limits.width / 2 * np.abs(np.sin(rel))
            lat_half = limits.width / 2 * np.abs(np.cos(rel)) + limits.length / 2 * np.abs(np.sin(rel))
            car = np.maximum(np.abs(lxc) - (half_len + long_half + limits.long_margin),
                             np.abs(lyc) - (half + lat_half + limits.car_margin))
            # 2 cm for the spline through these points (and its headings) differing from the rollout itself.
            ok = alive & ~won & (wall >= wall_need) & (car >= CAR_BUFFER)
            alive = alive & (won | ok)
            C = np.where(ok, np.minimum(C, np.minimum(wall - wall_need, car - CAR_BUFFER)), C)
            for i in np.flatnonzero(ok):
                grown[i][5].append((float(X[i]), float(Y[i])))
            rear_along = (X - cx) * ch + (Y - cy) * sh + rear
            heading_ok = np.abs((YAW - h + math.pi) % (2 * math.pi) - math.pi) < 0.6
            won = won | (ok & (rear_along >= half_len + past) & heading_ok)
        for i in np.flatnonzero(won & alive):
            wins.append((float(C[i]), grown[i][5]))
        keep = np.flatnonzero(alive & ~won)
        # Prune by clearance within 0.1 m bins of lateral offset from the box: a path already committed to going
        # round has less clearance than one still heading straight at it, and must not be pruned for that.
        lateral = (-(X - cx) * sh + (Y - cy) * ch)[keep]
        bins = np.round(lateral / 0.1).astype(int)
        order = keep[np.lexsort((-C[keep], bins))]
        ranked, seen = [], {}
        for i, b in zip(order, np.round((-(X[order] - cx) * sh + (Y[order] - cy) * ch) / 0.1).astype(int)):
            if seen.get(b, 0) < max(2, beam // 10):
                seen[b] = seen.get(b, 0) + 1
                ranked.append(i)
        keep = np.asarray(sorted(ranked, key=lambda i: -C[i])[:beam], int)
        states = [(float(X[i]), float(Y[i]), float(YAW[i]), float(D[i]), float(C[i]), grown[i][5]) for i in keep]
    if not wins:
        return None
    return np.asarray(max(wins, key=lambda w: w[0])[1], float)


def bypass_entry(line, pose, speed, opponent, walls, limits, *, nominal):
    """entry() for a stopped car: the crawl_rollout path anchored as a Bypass, then the same checks as a pass
    (whole manoeuvre wall-clear, hold plan clear of the car's full reach, a checked stop at a crawl).
    Returns (plan, manoeuvre, speed_cap, reason)."""
    path = crawl_rollout(line, pose, opponent, walls, limits)
    if path is None:
        return None, None, None, 'bypass: no crawl path round the stopped car'
    s0, _ = line.frenet(pose[0], pose[1])
    rel, ds = [0.0], []
    hint = None
    for x, y in path:
        s, d = line.frenet(x, y)
        r = line.wrap(s - s0)
        if not ds:
            ds.append(d)
            continue
        if r > rel[-1] + 1e-3:
            rel.append(r)
            ds.append(d)
    if len(rel) < 3:
        return None, None, None, 'bypass: crawl path does not advance along the line'
    k = int(np.argmin(np.abs(np.asarray(rel) - line.wrap(opponent.rear_arc - s0))))
    target = float(ds[k])
    back = shift_length(ds[-1], max(nominal, 1.0), limits)
    m = Bypass(float(s0), tuple(rel), tuple(ds), back, 1 if target > opponent.lateral else -1, target)
    whole = manoeuvre_plan(line, pose, speed, m, walls, limits, mode='pass')
    if whole is None:
        return None, None, None, 'bypass: no plan along the crawl path'
    if np.any(whole.wall < limits.wall_margin):
        j = int(np.argmax(whole.wall < limits.wall_margin))
        what = 'wall' if math.isfinite(whole.wall[j]) else 'unsteerable'
        return None, None, None, f'bypass: {what} {whole.dist[j]:.1f}m along the crawl path'
    plan = manoeuvre_plan(line, pose, speed, m, walls, limits, mode='pass', hold=True)
    if math.isfinite(box_free_distance(plan, opponent, line, limits)):
        return None, None, None, "bypass: the crawl path meets the car's full-reach box"
    v = safe_speed(plan, opponent, line, limits, speed)
    if v is None or v < 0.3:
        return None, None, None, 'bypass: no checked stop at a crawl'
    return plan, m, v, ''
