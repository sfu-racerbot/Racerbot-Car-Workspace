"""Contingency planner for passing a stopped or crawling car. No ROS.

Scope (tech lead, 2026-11-08): car 2 only passes a car its tracker has held
at <= stall_speed (0.5 m/s) for >= stall_sec (1.5 s): a crashed, stalled or
crawling car that stage-1 following would otherwise wait behind forever.
Passing moving cars is out of scope: on the 2026-11-06 courses the longest
single-side stretch with room for two cars is 8.4 m (06a) / 6.0 m (06b), and a
1.5 m/s car needs ~12 m (docs/racing-autonomy.md, "Passing a stopped car").

Frame. Everything is in the closed racing line's Frenet frame: s = arc
length, d = signed lateral offset, left positive. Offset points use
interpolated vertex normals, so an offset path has no kinks at stations.

The opponent (StalledOpponent) is an *anchor*, not a prediction: the arc of
its rear face, measured while following it from directly behind (where the
cluster centroid is that face), and its lateral centre. Its front is
bounded by rear + opponent_length (0.70 m: our 0.58 m chassis plus margin
for other teams' cars). speed_bound is the tracked speed + 1 m/s.

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
from dataclasses import dataclass
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
    wall_margin: float = 0.10         # body to wall, covers tracking error
    car_margin: float = 0.10          # body to the opponent's reachable box, laterally
    long_margin: float = 0.30         # same, longitudinally
    max_offset: float = 0.85          # inside max_cross_track_error (1.0 m)
    min_shift: float = 1.5            # metres
    opponent_length: float = 0.70
    opponent_width: float = 0.32
    opponent_lateral_uncertainty: float = 0.08
    opponent_long_uncertainty: float = 0.15
    opponent_react: float = 0.3
    opponent_brake: float = 3.0
    opponent_heading: float = 0.15    # rad; a car pulling away holds near its line
    clear_margin: float = 1.0         # approved rejoin body gap
    max_unseen: float = 1.0           # approved: a sighting within the last second
    spacing: float = 0.10
    horizon: float = 12.0
    pass_speed: float = 2.0           # m/s: plan speed beside a stopped car; sizes the shifts


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
        self.curvature = racing_math.estimate_path_curvature(xy, closed=True)

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


def shift_length(delta_d, speed, limits):
    """Length of a quintic shift of delta_d.

    Its curvature peaks at 5.77*delta/L^2: keep v^2 times that inside
    shift_accel, and the curvature itself inside half of what the steering
    limit allows (the other half is left for the line's own bends).
    """
    curvature = 0.5 * math.tan(limits.steering) / limits.wheelbase
    return max(limits.min_shift,
               speed * math.sqrt(5.7735 * abs(delta_d) / limits.shift_accel),
               math.sqrt(5.7735 * abs(delta_d) / curvature))


def lateral_profile(s, d0, target, shift_out, s_return, shift_back):
    """d(s): d0 -> target over shift_out, hold, target -> 0 from s_return over shift_back."""
    d = d0 + (target - d0) * smooth(s / shift_out)
    if s_return is not None:
        d = d * (1 - smooth((s - s_return) / shift_back))
    return d


def body_points(limits):
    rear = (limits.wheelbase - limits.length) / 2
    gx, gy = np.meshgrid(np.linspace(rear, rear + limits.length, 9),
                         np.linspace(-limits.width / 2, limits.width / 2, 5))
    # Any body point is within half a cell diagonal of a sample point.
    slack = math.hypot(limits.length / 16, limits.width / 8)
    return gx.ravel(), gy.ravel(), slack


def build(line, s0, d, mode, limits, walls, *, side=0, target=0.0):
    """Plan from Frenet samples; None if it needs more steering than the car has."""
    s_rel = np.arange(len(d)) * limits.spacing
    s = s0 + s_rel
    xy = line.point(s, d)
    if not np.all(np.isfinite(xy)):
        return None
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if np.any(step < 1e-6):
        return None
    seg = np.r_[step, step[-1]]
    curvature = racing_math.estimate_path_curvature(xy, closed=False)
    if np.any(np.abs(np.arctan(limits.wheelbase * curvature)) > limits.steering):
        return None
    speed = np.minimum(line.sample(line.speed, s),
                       np.sqrt(limits.lateral_accel / np.maximum(np.abs(curvature), 1e-9)))
    if mode != 'return':
        speed = np.minimum(speed, limits.pass_speed)
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


def fallback_clear(plan, speed, opponent, line, limits):
    """Can the car stop along this plan from `speed`, clear of walls and the opponent?"""
    if plan is None or not math.isfinite(speed) or speed < 0:
        return False
    stop = stop_distance(speed, limits)
    reached = plan.dist <= stop + limits.spacing
    if plan.dist[-1] < stop + limits.spacing:
        return False   # the plan must contain the whole stop
    if np.any(plan.wall[reached] < limits.wall_margin):
        return False
    if opponent is None:
        return True
    times = braking_times(plan.dist, speed, limits)
    last = int(np.flatnonzero(reached)[-1])
    for k in np.flatnonzero(reached):
        t = times[k] if math.isfinite(times[k]) else limits.response + speed / limits.brake
        if _overlaps(plan, k, t, opponent, line, limits):
            return False
    # The stopped car stays there; the opponent's reach saturates.
    return not _overlaps(plan, last, math.inf, opponent, line, limits)


def safe_speed(plan, opponent, line, limits, current_speed, grid=0.25):
    """Largest commandable speed whose stop is clear; None if even the current one is not."""
    if plan is None or not math.isfinite(current_speed) or current_speed < 0:
        return None
    if not fallback_clear(plan, current_speed, opponent, line, limits):
        return None
    top = float(plan.context.speed_profile[0])
    candidates = np.arange(math.floor(top / grid) * grid, current_speed, -grid)
    for v in candidates:
        if fallback_clear(plan, float(v), opponent, line, limits):
            return float(min(v, top))
    return float(min(current_speed, top))


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


def new_pass(line, pose, speed, opponent, limits, side):
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
    if limits.pass_speed - opponent.speed_bound <= 0.2:
        return None    # the unseen bound would never fall behind
    out = shift_length(target - d0, max(speed, limits.pass_speed), limits)
    # Nominal: the front corner leaves the scan ~0.35 m before ego's rear
    # passes it; from there ego gains (v - speed_bound) on the unseen bound.
    rear_offset = (limits.wheelbase - limits.length) / 2
    unseen = (limits.clear_margin - 0.35) / (1 - opponent.speed_bound / limits.pass_speed)
    s_return = max(out, rel_rear + limits.opponent_length + 0.35 + unseen - rear_offset)
    back = shift_length(target, limits.pass_speed, limits)
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
        m = Manoeuvre(manoeuvre.s_start, manoeuvre.d_start, manoeuvre.target, manoeuvre.out,
                      math.inf, manoeuvre.back, manoeuvre.side)
        ahead = max(0.0, line.wrap(manoeuvre.s_start + manoeuvre.out - s0))
        length = ahead + stop_distance(max(speed, limits.pass_speed), limits) + 1.5
    else:
        m = manoeuvre
        ahead = max(0.0, line.wrap(manoeuvre.s_return + manoeuvre.back - s0))
        length = ahead + stop_distance(max(speed, limits.pass_speed), limits) + 1.5
    if length > 40.0:
        return None
    s_rel = np.arange(0.0, length, limits.spacing)
    return build(line, s0, m.lateral(s0 + s_rel, line), mode, limits, walls,
                 side=manoeuvre.side, target=manoeuvre.target)


def plan_hold(line, pose, speed, walls, limits):
    s0, d0 = line.frenet(pose[0], pose[1])
    if not math.isfinite(s0):
        return None
    length = stop_distance(max(speed, 0.5), limits) + 1.0
    return build(line, s0, np.full(len(np.arange(0.0, length, limits.spacing)), d0), 'hold',
                 limits, walls)


def plan_return(line, pose, speed, walls, limits):
    """A fresh shift back to the line from here (the abort's checked rejoin)."""
    return manoeuvre_plan(line, pose, speed, new_return(line, pose, speed, limits), walls, limits,
                          mode='return')


def entry(line, pose, speed, opponent, walls, limits):
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
        manoeuvre = new_pass(line, pose, speed, opponent, limits, side)
        nominal = manoeuvre_plan(line, pose, speed, manoeuvre, walls, limits, mode='pass')
        if nominal is None:
            reasons.append(f'{name}: no pass path (steering, speed bound or length)')
            continue
        if np.any(nominal.wall < limits.wall_margin):
            short = float(nominal.dist[int(np.argmax(nominal.wall < limits.wall_margin))])
            reasons.append(f'{name}: wall {short:.1f}m along a {nominal.dist[-1]:.1f}m pass')
            continue
        plan = manoeuvre_plan(line, pose, speed, manoeuvre, walls, limits, mode='pass', hold=True)
        # A crawl is enough to start: the cap rises as the shift clears the
        # car. new_pass already required the pass speed to outrun the bound.
        v = safe_speed(plan, opponent, line, limits, speed)
        if v is None or v < 0.3:
            reasons.append(f'{name}: no checked stop at a crawl')
            continue
        score = float(nominal.wall.min())
        if best is None or score > best[0]:
            best = (score, plan, manoeuvre, v)
    if best is None:
        return None, None, None, '; '.join(reasons)
    return best[1], best[2], best[3], ''
