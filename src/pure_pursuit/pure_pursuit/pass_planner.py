"""Bounded local offset search. No ROS, ground truth, or simulator flags.

The opponent occupies a *reachable tube*, not just a constant-velocity
prediction. Its lower progress bound includes an immediate stop, and its
lateral tube includes arbitrary acceleration up to the configured grip bound.
This is deliberately conservative: a narrow track may offer no passing path.
Entry/hold and rejoin are separate candidates; the caller must authorise rejoin
using CompletionGate (or a checked abort, never prediction-only completion).
"""
from dataclasses import dataclass
import copy
import math

import numpy as np

from pure_pursuit import racing_math, tracking_laws


@dataclass(frozen=True)
class Limits:
    wheelbase: float = 0.36
    length: float = 0.58
    width: float = 0.31
    steering: float = 0.42
    steering_rate: float = 1.0
    lateral_accel: float = 2.7
    accel: float = 4.0
    brake: float = 3.0
    response: float = 0.45
    tracking_margin: float = 0.25
    wall_margin: float = 0.10
    opponent_uncertainty: float = 0.35
    opponent_speed_uncertainty: float = 1.0
    opponent_lateral_accel: float = 2.7
    opponent_max_speed: float = 8.0
    horizon: float = 12.0
    spacing: float = 0.15


@dataclass(frozen=True)
class Opponent:
    arc: float
    lateral: float
    speed: float
    age: float = 0.0
    lateral_speed: float = 0.0


@dataclass(frozen=True)
class Candidate:
    path: tracking_laws.PathContext
    side: int
    mode: str
    speed_cap: float
    clearance: float
    cost: float


def valid_limits(limits):
    values = vars(limits)
    return (all(math.isfinite(v) and v > 0 for v in values.values())
            and limits.spacing <= 0.15 and limits.horizon <= 20
            and limits.steering < math.pi / 2)


def frame(xy, cumulative, arc):
    """Linear sampling of a closed line, including its real closing segment."""
    seg = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    total = cumulative[-1] + seg[-1]
    s = np.asarray(arc) % total
    i = np.clip(np.searchsorted(cumulative, s, side='right') - 1, 0, len(xy) - 1)
    tangent = (xy[(i + 1) % len(xy)] - xy[i]) / seg[i][..., None]
    point = xy[i] + (s - cumulative[i])[..., None] * tangent
    normal = np.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)
    return point, tangent, normal


def smooth_shift(t):
    t = np.clip(t, 0, 1)
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)


def temporary_path(xy, speed):
    """An open context: its end must clamp, never wrap across a pass corridor."""
    seg = racing_math.compute_segment_lengths(xy, closed=False)
    return tracking_laws.PathContext(
        xy, seg, racing_math.compute_cumulative_arc_length(seg), speed,
        racing_math.estimate_path_curvature(xy, closed=False), False)


def trajectory(xy, cumulative, arc, lateral, target, limits, mode):
    s = np.arange(0, limits.horizon + limits.spacing / 2, limits.spacing)
    centre, _, normal = frame(xy, cumulative, arc + s)
    # Zero slope and curvature at either end; no target-point shortcut.
    shift = lateral + (target - lateral) * smooth_shift(s / 4.0)
    path = centre + shift[:, None] * normal
    return temporary_path(path, np.full(len(path), 5.0))


def feasible_speed(path, limits):
    """Steering, steering-rate, grip and measured acceleration/braking limits."""
    k = np.abs(path.curvature)
    steer = np.arctan(limits.wheelbase * path.curvature)
    if np.any(np.abs(steer) > limits.steering):
        return None
    speed = np.minimum(path.speed_profile, np.sqrt(limits.lateral_accel / np.maximum(k, 1e-9)))
    ds = np.maximum(np.linalg.norm(np.diff(path.xy, axis=0), axis=1), 1e-9)
    rate = np.abs(np.diff(steer)) / ds
    rate_cap = limits.steering_rate / np.maximum(rate, 1e-9)
    speed[:-1] = np.minimum(speed[:-1], rate_cap)
    speed[1:] = np.minimum(speed[1:], rate_cap)
    for i in range(len(speed) - 2, -1, -1):
        speed[i] = min(speed[i], math.sqrt(speed[i + 1] ** 2 + 2 * limits.brake * ds[i]))
    for i in range(1, len(speed)):
        speed[i] = min(speed[i], math.sqrt(speed[i - 1] ** 2 + 2 * limits.accel * ds[i - 1]))
    return speed


def wall_clearance(path, wall_at, limits):
    """Cover the entire rectangle and motion between samples with discs.

    Every body point is within half a grid diagonal of a sample; each motion
    interval adds half its translation plus corner rotation. The distance
    transform is 1-Lipschitz. A further 0.075 m covers map-cell quantisation.
    Invalid/unknown distances are occupied, never open space.
    """
    delta = np.diff(path.xy, axis=0)
    heading = np.arctan2(delta[:, 1], delta[:, 0])
    heading = np.r_[heading, heading[-1]]
    rear = (limits.wheelbase - limits.length) / 2
    grid_step = 0.05
    gx, gy = np.meshgrid(np.linspace(rear, rear + limits.length, 13),
                         np.linspace(-limits.width / 2, limits.width / 2, 8))
    gx, gy = gx.ravel(), gy.ravel()
    radius = math.hypot(rear + limits.length, limits.width / 2)
    turn = np.abs((np.diff(heading) + math.pi) % (2 * math.pi) - math.pi)
    sweep = np.r_[np.linalg.norm(delta, axis=1) / 2 + radius * turn / 2, 0]
    minimum = math.inf
    for i, (point, yaw) in enumerate(zip(path.xy, heading)):
        c, s = math.cos(yaw), math.sin(yaw)
        distances = np.array([wall_at(x, y) for x, y in zip(
            point[0] + c * gx - s * gy, point[1] + s * gx + c * gy)])
        if not np.all(np.isfinite(distances)):
            return -math.inf
        reserve = (grid_step / math.sqrt(2) + sweep[i] + 0.075
                   + limits.tracking_margin + limits.wall_margin)
        minimum = min(minimum, float(min(distances)) - reserve)
        if minimum <= 0:
            return minimum
    return minimum


def opponent_clearance(path, times, xy, cumulative, opponent, limits):
    """Conservative body separation from every reachable opponent position.

    A surface centroid can lie anywhere on the body. Bound the axle by a
    whole length in either direction, then cover both bodies by circles.
    Longitudinal reach includes stopped, braking, coasting and accelerating
    cars. Lateral reach includes a weave in either direction. No lost-track
    advancement is used as the lower bound.
    """
    total = float(cumulative[-1] + np.linalg.norm(xy[0] - xy[-1]))
    # Circle about the chassis centre (wheelbase/2 ahead of the axle).
    body_radius = math.hypot(limits.length / 2, limits.width / 2)
    delta = np.diff(path.xy, axis=0)
    tangents = delta / np.maximum(np.linalg.norm(delta, axis=1)[:, None], 1e-9)
    body_centres = path.xy + limits.wheelbase / 2 * np.vstack((tangents, tangents[-1]))
    minimum = math.inf
    for point, time in zip(body_centres, times):
        t = time + opponent.age
        u = min(limits.opponent_max_speed,
                abs(opponent.speed) + limits.opponent_speed_uncertainty)
        reach = min(limits.opponent_max_speed * t, u * t + limits.accel * t * t / 2)
        # Never sample a full wrapped lap as if it were a small local interval.
        if reach + 2 * limits.length >= total / 2:
            return -math.inf
        arcs = np.arange(opponent.arc - limits.length,
                         opponent.arc + reach + limits.length + limits.spacing,
                         limits.spacing)
        centre, _, normal = frame(xy, cumulative, arcs)
        centre = centre + opponent.lateral * normal
        lateral = (limits.opponent_uncertainty + abs(opponent.lateral_speed) * t
                   + limits.opponent_lateral_accel * t * t / 2)
        distance = float(np.min(np.linalg.norm(centre - point, axis=1)))
        clearance = (distance - 2 * body_radius - lateral
                     - limits.tracking_margin - limits.spacing)
        minimum = min(minimum, clearance)
        if minimum <= 0:
            return minimum
    return minimum


def check(path, xy, cumulative, opponent, wall_at, limits, speed, *, braking=False):
    """Check the path and a delayed stopping trajectory; return a speed bound.

    A stop includes response coasting then 3 m/s² physical braking, not the
    command slew. The finite horizon must contain that entire stop. For a
    braking candidate only the reachable stopping prefix is traversed, with
    opponent uncertainty evaluated at the *later* stopping time.
    """
    profile = feasible_speed(path, limits)
    if profile is None:
        return None
    ds = np.linalg.norm(np.diff(path.xy, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(ds)]
    stop = speed * limits.response + speed ** 2 / (2 * limits.brake)
    if stop + limits.tracking_margin >= arc[-1]:
        return None
    cap = min(float(profile[0]),
              math.sqrt((limits.brake * limits.response) ** 2
                        + 2 * limits.brake * (arc[-1] - limits.tracking_margin))
              - limits.brake * limits.response)
    # Conservative times: do not assume the speed loop jumps to the target.
    moving_speed = max(0.3, min(cap, speed if speed > 0.3 else 0.3))
    times = arc / moving_speed
    checked = path
    if braking:
        n = min(len(arc), int(np.searchsorted(arc, stop + limits.tracking_margin)) + 2)
        checked = temporary_path(path.xy[:max(3, n)], profile[:max(3, n)])
        times = np.full(len(checked.xy), limits.response + speed / limits.brake)
    walls = wall_clearance(checked, wall_at, limits)
    if walls <= 0:
        return None
    cars = opponent_clearance(checked, times, xy, cumulative, opponent, limits)
    if cars <= 0:
        return None
    return cap, min(walls, cars), profile


def plan(xy, cumulative, pose, speed, opponent, wall_at, *, limits=Limits(),
         side=0, rejoin=False, braking=False):
    """Select a checked candidate, or None on absent/invalid/degenerate data."""
    xy, cumulative = np.asarray(xy), np.asarray(cumulative)
    if (not valid_limits(limits) or xy.ndim != 2 or xy.shape[1:] != (2,)
            or len(xy) < 4 or cumulative.shape != (len(xy),)
            or not np.all(np.isfinite(xy)) or not np.all(np.isfinite(cumulative))
            or len(pose) != 3 or not all(math.isfinite(v) for v in (*pose, speed))
            or speed < 0 or opponent is None or wall_at is None
            or not all(math.isfinite(v) for v in vars(opponent).values())
            or opponent.age < 0 or side not in (-1, 0, 1)):
        return None
    seg = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    if np.any(seg < 1e-6) or np.any(np.diff(cumulative) <= 0):
        return None
    nearest = int(np.argmin(np.linalg.norm(xy - pose[:2], axis=1)))
    arc = float(cumulative[nearest])
    centre, tangent, normal = frame(xy, cumulative, arc)
    lateral = float(np.dot(np.asarray(pose[:2]) - centre, normal))
    yaw_error = (pose[2] - math.atan2(tangent[1], tangent[0]) + math.pi) % (2 * math.pi) - math.pi
    # No geometric teleport from a badly aligned pose into a smooth path.
    if abs(yaw_error) > 0.25 or abs(lateral) > 1.5:
        return None
    targets = [0.0] if rejoin else [s * d for s in ((side,) if side else (-1, 1)) for d in (0.8, 1.2)]
    best = None
    for target in targets:
        mode = 'rejoin' if rejoin else 'hold' if braking else 'pass'
        path = trajectory(xy, cumulative, arc, lateral, target, limits, mode)
        verdict = check(path, xy, cumulative, opponent, wall_at, limits, speed, braking=braking)
        if verdict is None:
            continue
        cap, clearance, profile = verdict
        path = temporary_path(path.xy, profile)
        # Time-gain surrogate relative to continuing behind the tracked car;
        # larger clearance wins when travel times are similar.
        travel = float(np.sum(np.linalg.norm(np.diff(path.xy, axis=0), axis=1)
                              / np.maximum(profile[:-1], 0.3)))
        follow_time = limits.horizon / max(0.3, abs(opponent.speed))
        cost = travel - follow_time + 0.3 / max(clearance, 0.01)
        candidate = Candidate(path, 1 if target > 0 else -1 if target < 0 else side,
                              mode, cap, clearance, cost)
        if best is None or candidate.cost < best.cost:
            best = candidate
    return best


def rollout_clear(candidate, law, inp, xy, cumulative, opponent, wall_at, limits,
                  *, braking=False):
    """Recheck the delay_lqr-executed body, including a physical stop.

    Use a private copy: rejected candidates must not advance the real law's
    command history. Pose and command queues, servo gain/rate and steering
    limits are the law's own model. Speed transport and 7/s loop use car 2's
    measured floor model. The geometry check's tracking tube additionally
    covers initial servo/pose uncertainty; this is a model check, not truth.
    """
    if law.name != 'delay_lqr':
        return False
    model = copy.deepcopy(law)
    dt = law.limits.control_dt
    if not (0 < dt <= 0.05):
        return False
    p = law.params
    state = np.array([inp.x, inp.y, inp.yaw, max(0, inp.speed), inp.previous_steering], float)
    positions = [state.copy()] * (model.sensing_ticks + 1)
    steering_queue = [inp.previous_steering] * (model.actuation_ticks + 1)
    speed_queue = [state[3]] * (int(math.ceil(0.10 / dt)) + 1)
    previous = inp.previous_steering
    points, times = [], []
    elapsed = 0.0
    duration = (limits.response + inp.speed / limits.brake + 0.1) if braking else min(6.0, limits.horizon / max(0.3, inp.speed))
    for _ in range(int(math.ceil(duration / dt))):
        sensed = positions[0]
        nearest = int(np.argmin(np.linalg.norm(candidate.path.xy - sensed[:2], axis=1)))
        measured = tracking_laws.TrackingInput(*map(float, sensed[:3]), float(sensed[3]), nearest, previous)
        previous = model.steer(measured, candidate.path).steering
        steering_queue.append(previous)
        command = steering_queue.pop(0)
        rate = float(np.clip(p.servo_gain * (command - state[4]), -p.servo_rate, p.servo_rate))
        state[4] += dt * rate
        target = 0.0 if braking else min(candidate.speed_cap, float(candidate.path.speed_profile[nearest]))
        speed_queue.append(target)
        acceleration = float(np.clip(7.0 * (speed_queue.pop(0) - state[3]), -limits.brake, limits.accel))
        state[3] = max(0.0, state[3] + dt * acceleration)
        state[2] += dt * state[3] * math.tan(state[4]) / limits.wheelbase
        state[:2] += dt * state[3] * np.array([math.cos(state[2]), math.sin(state[2])])
        positions.append(state.copy())
        positions.pop(0)
        points.append(state[:2].copy())
        elapsed += dt
        times.append(elapsed)
        if nearest >= len(candidate.path.xy) - 3:
            break
    path = temporary_path(np.asarray(points), np.full(len(points), candidate.speed_cap))
    return (wall_clearance(path, wall_at, limits) > 0
            and opponent_clearance(path, times, xy, cumulative, opponent, limits) > 0)
