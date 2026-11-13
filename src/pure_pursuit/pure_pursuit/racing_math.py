"""
racing_math.py

Pure, dependency-light math for the pure-pursuit race stack. Every
function in this file takes plain numbers/arrays in and returns plain
numbers/arrays out -- nothing here imports rclpy or talks to ROS topics.

That is deliberate. It means every formula actually used to steer and
pace the car can be unit-tested in complete isolation (see
test/test_racing_math.py) without a running robot, a simulator, or even
ROS installed -- and it means you can read this one file top to bottom
to understand the *entire* algorithm, with the ROS plumbing kept
completely separate in pure_pursuit_node.py / waypoint_recorder_node.py /
generate_velocity_profile.py.

See docs/racing-autonomy.md for the full write-up with derivations and
diagrams. Short version of what lives in each section below:

  1. Frame conversions   -- world (map) frame  <->  car body frame
  2. Pure pursuit         -- the steering-geometry formula itself
  3. Path indexing        -- "where am I on the line" / "where should I aim"
  4. Raceline generation  -- curvature -> cornering speed -> smoothed profile
  5. CSV I/O              -- reading/writing waypoint files
  6. Reactive avoidance   -- gap-finding, for anything not in the map
  7. Opponent racing      -- detect another car, track it, decide/steer an overtake
  8. Swept-path clearance -- the arc check behind the Tier 1 emergency stop
"""

import csv
import math
from dataclasses import dataclass

import numpy as np


# ============================================================================
# 1. Frame conversions
# ============================================================================

def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw (rotation about +Z) from a geometry_msgs Quaternion.

    The car only ever needs its heading in the 2D ground plane -- roll and
    pitch are irrelevant to steering a car that can't leave the ground --
    so this is the standard atan2-based yaw-only extraction from a unit
    quaternion, not a full 3D Euler decomposition:

        yaw = atan2( 2*(w*z + x*y), 1 - 2*(y^2 + z^2) )
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def world_to_body(dx: float, dy: float, yaw: float):
    """Rotate a world/map-frame vector (dx, dy) into the car's body frame.

    `yaw` is the car's current heading in the map frame. The body frame
    follows REP-103 (x out the nose, y to the left). Rotating a world
    vector into body coordinates means rotating it by -yaw:

        x_body =  cos(yaw)*dx + sin(yaw)*dy
        y_body = -sin(yaw)*dx + cos(yaw)*dy

    Sanity check: car heading along world +X (yaw=0), point at world +Y
    (i.e. to the car's left): dx=0, dy>0 -> y_body = dy > 0. Left stays
    left. Good.
    """
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    x_body = cos_yaw * dx + sin_yaw * dy
    y_body = -sin_yaw * dx + cos_yaw * dy
    return x_body, y_body


def laser_pose_to_base_link(x: float, y: float, yaw: float,
                            laser_offset_x: float, laser_offset_y: float):
    """The rear-axle (base_link) pose of a car whose LiDAR is at (x, y, yaw).

    particle_filter ray-casts each particle's scan from the particle's own
    pose, so every particle -- and therefore /pf/viz/inferred_pose -- is a
    pose of the *LiDAR*, not of base_link. The LiDAR sits
    (laser_offset_x, laser_offset_y) from base_link in the car's own frame
    (0.26 m forward on this car, docs/hardware-reference.md); both frames
    share one heading, so only the position moves:

        base = laser - R(yaw) @ offset
    """
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (x - (laser_offset_x * cos_yaw - laser_offset_y * sin_yaw),
            y - (laser_offset_x * sin_yaw + laser_offset_y * cos_yaw),
            yaw)


def base_link_pose_to_laser(x: float, y: float, yaw: float,
                            laser_offset_x: float, laser_offset_y: float):
    """Inverse of laser_pose_to_base_link: where the LiDAR is, given base_link."""
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (x + laser_offset_x * cos_yaw - laser_offset_y * sin_yaw,
            y + laser_offset_x * sin_yaw + laser_offset_y * cos_yaw,
            yaw)


# ============================================================================
# 2. Pure pursuit steering geometry
# ============================================================================

def steering_arc_curvature(x_body: float, y_body: float) -> float:
    """The Pure Pursuit formula. Curvature of the arc from the car's rear
    axle (the origin, in body-frame coordinates) to a target point
    (x_body, y_body) ahead of it, that is tangent to the car's current
    heading (the body-frame x-axis).

    Derivation sketch: a circle tangent to the x-axis at the origin has
    its center at (0, R) for some radius R (positive R = center to the
    left = a left turn). For such a circle to also pass through
    (x_body, y_body):

        x_body^2 + (y_body - R)^2 = R^2
        x_body^2 + y_body^2 - 2*R*y_body = 0
        R = (x_body^2 + y_body^2) / (2 * y_body)

    Curvature kappa = 1/R, so:

        kappa = 2 * y_body / (x_body^2 + y_body^2)

    Using the *actual* squared distance to the target rather than the
    nominal lookahead distance parameter keeps this exact even though the
    chosen waypoint is essentially never sitting at precisely the
    requested lookahead distance (see find_lookahead_index). Positive
    y_body (target to the left) gives positive curvature -> a left turn,
    matching AckermannDriveStamped's "positive steering_angle = left"
    convention.
    """
    dist_sq = x_body * x_body + y_body * y_body
    if dist_sq < 1e-9:
        # Target essentially on top of the car -- no meaningful direction
        # to curve toward. Drive straight rather than divide by ~0.
        return 0.0
    return 2.0 * y_body / dist_sq


def steering_from_curvature(kappa: float, wheelbase: float) -> float:
    """Bicycle-model steering angle that produces a given path curvature.

    Collapsing the car's front and rear wheel pairs to a single front and
    single rear "wheel" (the standard car-like-robot bicycle model), a
    vehicle with wheelbase L needs a front steer angle:

        delta = atan(L * kappa)

    to drive an arc of curvature kappa.
    """
    return math.atan(wheelbase * kappa)


def adaptive_lookahead(speed: float, gain: float, min_lookahead: float, max_lookahead: float) -> float:
    """Speed-scaled lookahead distance: L_d = clip(gain*speed + min, min, max).

    A *fixed* lookahead distance is a bad compromise: short enough to
    corner tightly at parking-lot speed and the car oscillates/overshoots
    at race speed; long enough to be smooth at race speed and it cuts
    corners at low speed. Scaling lookahead with the current speed (a
    standard, well-known extension to textbook Pure Pursuit) fixes both
    problems with one parameter (`gain`, meters of extra lookahead per
    m/s of speed).
    """
    ld = gain * abs(speed) + min_lookahead
    return float(np.clip(ld, min_lookahead, max_lookahead))


def curvature_speed_limit(curvature: float, max_lateral_accel: float,
                          max_speed: float) -> float:
    """Online speed ceiling from the final commanded curvature.

    The offline profile accounts for the recorded path's curvature. This
    second ceiling covers runtime corrections, avoidance, and overtakes whose
    steering can be sharper than the recorded line.
    """
    if not all(math.isfinite(value) for value in (
            curvature, max_lateral_accel, max_speed)):
        raise ValueError('curvature speed-limit inputs must be finite')
    if max_lateral_accel <= 0.0 or max_speed < 0.0:
        raise ValueError('max_lateral_accel must be positive and max_speed non-negative')
    if abs(curvature) < 1e-9:
        return max_speed
    return min(max_speed, math.sqrt(max_lateral_accel / abs(curvature)))


@dataclass(frozen=True)
class GripCurve:
    """What the car delivers against what the steering asks for, in lateral acceleration.

    ``asked`` is the kinematic lateral acceleration (v^2 * tan(delta) / wheelbase), strictly
    increasing from 0; ``delivered`` is what the car achieves (measured by
    tools/run_analysis/grip_truth.py), strictly increasing from 0 and never above ``asked``.
    Past the last point the tyre is beyond its peak: ``deliver`` is flat there and ``ask_for``
    never returns more than the last ``asked``.
    """
    asked: tuple
    delivered: tuple

    def deliver(self, asked: float) -> float:
        """Delivered lateral acceleration for an asked one: piecewise linear, flat beyond the last point."""
        return float(np.interp(asked, self.asked, self.delivered))

    def ask_for(self, wanted: float) -> float:
        """The asked lateral acceleration that delivers ``wanted``; the last asked value beyond the
        last delivered one."""
        if wanted >= self.delivered[-1]:
            return float(self.asked[-1])
        return float(np.interp(wanted, self.delivered, self.asked))


def grip_curve_from_params(asked, delivered):
    """A GripCurve from two parameter lists. None for two empty lists (compensation off).
    ValueError, naming the rule broken, for unequal lengths, fewer than 2 points, a non-finite
    value, non-increasing values, a first point other than (0, 0), or delivered > asked."""
    asked, delivered = [float(v) for v in asked], [float(v) for v in delivered]
    if not asked and not delivered:
        return None
    if len(asked) != len(delivered):
        raise ValueError('understeer curve: asked and delivered must have equal length')
    if len(asked) < 2:
        raise ValueError('understeer curve: needs at least 2 points')
    if not all(math.isfinite(v) for v in asked + delivered):
        raise ValueError('understeer curve: every value must be finite')
    for name, values in (('asked', asked), ('delivered', delivered)):
        if any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError(f'understeer curve: {name} must be strictly increasing')
    if asked[0] != 0.0 or delivered[0] != 0.0:
        raise ValueError('understeer curve: must start at (0, 0)')
    if any(d > a + 1e-12 for a, d in zip(asked, delivered)):
        raise ValueError('understeer curve: delivered must not exceed asked')
    return GripCurve(tuple(asked), tuple(delivered))


def understeer_steering(steering: float, speed: float, wheelbase: float, curve,
                        max_steering_angle: float, min_speed: float = 1.0) -> float:
    """The steering that makes the car follow the curvature ``steering`` asks for kinematically.

    Unchanged (clipped) when ``curve`` is None or ``speed`` is below ``min_speed`` or not finite.
    Past the curve's last delivered value the output holds the steering for the last asked value
    (``ask_for`` caps there) until the kinematic ask exceeds it, then equals the clipped input, so
    the output is continuous across the grip limit and never steers less than the input; the
    speed cap handles the car being too fast. Sign preserved; clipped to
    +/- ``max_steering_angle``. A non-finite ``steering`` returns 0.0, never NaN.
    """
    if not math.isfinite(steering):
        return 0.0 if math.isnan(steering) else math.copysign(max_steering_angle, steering)
    clipped = float(np.clip(steering, -max_steering_angle, max_steering_angle))
    if curve is None or not math.isfinite(speed) or speed < min_speed:
        return clipped
    wanted = speed * speed * abs(math.tan(steering)) / wheelbase
    if wanted <= 0.0:
        return clipped
    asked = max(wanted, curve.ask_for(wanted))
    compensated = math.copysign(math.atan(asked * wheelbase / (speed * speed)), steering)
    return float(np.clip(compensated, -max_steering_angle, max_steering_angle))


def delivered_curvature_speed_limit(curvature: float, max_lateral_accel: float, max_speed: float,
                                    curve) -> float:
    """Online speed ceiling from the curvature the car must follow, using what it delivers.

    The largest speed v <= max_speed with curve.deliver(v^2 * |curvature|) <= max_lateral_accel,
    in closed form: v = sqrt(curve.ask_for(max_lateral_accel) / |curvature|). If the curve never
    delivers max_lateral_accel the asked lateral is limited at the curve's last asked value, so
    speed is still bounded. With ``curve`` None it is exactly curvature_speed_limit. NaN curvature
    returns 0.0 (command zero).
    """
    if not math.isfinite(curvature):
        return 0.0
    if curve is None:
        return curvature_speed_limit(curvature, max_lateral_accel, max_speed)
    if not all(math.isfinite(v) for v in (max_lateral_accel, max_speed)):
        raise ValueError('curvature speed-limit inputs must be finite')
    if max_lateral_accel <= 0.0 or max_speed < 0.0:
        raise ValueError('max_lateral_accel must be positive and max_speed non-negative')
    if abs(curvature) < 1e-9:
        return max_speed
    return min(max_speed, math.sqrt(curve.ask_for(max_lateral_accel) / abs(curvature)))


def slew_rate_limit(target: float, previous: float, dt: float,
                    increase_rate: float, decrease_rate: float = None) -> float:
    """Limit a normal command's rise and fall per second.

    Safety stops bypass this helper and remain immediate.
    """
    if decrease_rate is None:
        decrease_rate = increase_rate
    if not all(math.isfinite(value) for value in (
            target, previous, dt, increase_rate, decrease_rate)):
        raise ValueError('slew-rate inputs must be finite')
    if dt < 0.0 or increase_rate < 0.0 or decrease_rate < 0.0:
        raise ValueError('dt and slew rates must be non-negative')
    lower = previous - decrease_rate * dt
    upper = previous + increase_rate * dt
    return float(np.clip(target, lower, upper))


# ============================================================================
# 3. Path indexing
# ============================================================================

def compute_segment_lengths(xy: np.ndarray, closed: bool = True) -> np.ndarray:
    """Euclidean distance from each waypoint to the next one.

    seg_len[i] = |xy[i+1] - xy[i]|, wrapping xy[-1] -> xy[0] if the track
    is a closed loop. For an open path, seg_len[-1] is set to 0 since
    there is no "next" waypoint after the last one.
    """
    nxt = np.roll(xy, -1, axis=0)
    seg_len = np.hypot(nxt[:, 0] - xy[:, 0], nxt[:, 1] - xy[:, 1])
    if not closed:
        seg_len[-1] = 0.0
    return seg_len


def find_nearest_index(xy: np.ndarray, car_xy, closed: bool = True,
                        prev_index=None, search_window=None):
    """Index of the waypoint closest to the car, and the distance to it
    (the car's instantaneous cross-track error).

    If `prev_index` and a positive `search_window` are both given, the
    search is restricted to the +/- search_window waypoints around the
    previous tick's result (wrapping around the array if `closed`), for
    two reasons:

      1. Speed: O(window) instead of O(N) every control tick.
      2. Correctness: on a track that comes close to itself (a hairpin, a
         figure-eight, a pit-lane split), the *globally* closest waypoint
         can be on a completely different part of the track than the one
         the car is actually on. Searching only near where the car was
         last tick keeps it locked onto the correct branch instead of
         "teleporting" its target to the wrong side of the track.

    Falls back to a full-array search on the very first call (no
    `prev_index` yet) or when `search_window` disables windowing.
    """
    n = len(xy)
    if prev_index is None or not search_window or search_window <= 0 or search_window >= n:
        candidate_idx = np.arange(n)
    else:
        offsets = np.arange(-search_window, search_window + 1)
        candidate_idx = prev_index + offsets
        if closed:
            candidate_idx = np.mod(candidate_idx, n)
        else:
            candidate_idx = candidate_idx[(candidate_idx >= 0) & (candidate_idx < n)]

    deltas = xy[candidate_idx] - np.asarray(car_xy)
    dists = np.hypot(deltas[:, 0], deltas[:, 1])
    best_local = int(np.argmin(dists))
    return int(candidate_idx[best_local]), float(dists[best_local])


def find_lookahead_index(seg_len: np.ndarray, nearest_index: int, lookahead_dist: float,
                          closed: bool = True) -> int:
    """Walk forward along the path from `nearest_index`, accumulating
    segment lengths, and return the index reached once >= lookahead_dist
    has been covered.

    This is a simple, robust stand-in for the textbook description of
    Pure Pursuit target selection ("intersect the path with a circle of
    radius lookahead_dist centered on the car"). Snapping to the next
    recorded waypoint instead of solving that circle-line intersection
    exactly is less precise, but trivial to get right, and the error it
    introduces is bounded by the spacing between recorded waypoints --
    keep that spacing small (waypoint_recorder_node's default is 0.15m)
    and the difference is negligible.
    """
    n = len(seg_len)
    acc = 0.0
    idx = nearest_index
    for _ in range(n):
        acc += seg_len[idx]
        nxt = idx + 1
        if nxt >= n:
            if not closed:
                return n - 1
            nxt = 0
        if acc >= lookahead_dist:
            return nxt
        idx = nxt
    # lookahead_dist is longer than the entire path (a very short/degenerate
    # recording, or badly misconfigured lookahead) -- aim at the farthest
    # point reachable rather than raising mid-race.
    return idx


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]. Used for heading errors so a path just
    across the +/-pi seam reads as a small turn, not a full revolution."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


# A projected point on the path: which segment it sits on, how far along
# that segment (t in [0, 1]), the projected point itself, the path heading
# there, the signed lateral offset of the query point from the path, and
# how far along the whole path (from waypoint 0) the projection is.
# Returned by project_onto_path; the tracking laws read it, nothing else
# writes it.
@dataclass(frozen=True)
class PathProjection:
    """Orthogonal projection of a point onto the path (project_onto_path).

    segment_index is the start waypoint of the chosen segment, t in [0,1]
    the fraction along it, (qx,qy) the projected point, heading the segment
    direction psi_p, lateral_error the signed cross-track offset
    (+ = the point is left of the path), arc_length the distance from
    waypoint 0 along the path to (qx,qy).
    """
    segment_index: int
    t: float
    qx: float
    qy: float
    heading: float
    lateral_error: float
    arc_length: float


def project_onto_path(xy: np.ndarray, seg_len: np.ndarray, point,
                      nearest_idx: int, closed: bool = True):
    """Orthogonal projection of `point` onto the polyline near `nearest_idx`.

    Considers the two segments adjacent to `nearest_idx` -- (i-1 -> i) and
    (i -> i+1), wrapping when `closed` -- and returns the closer projection
    (a `PathProjection`). At the ends of an open path only the one existing
    segment is a candidate. `t` is clamped to [0, 1], so the projection of
    a point past the end of an open path is the last waypoint (t=1 on the
    last segment) rather than running off the path.

    Signed lateral error follows the workspace convention (see
    docs/superpowers/specs/2026-09-27-tracking-00-overview.md):

        e = -sin(psi_p)*(px - qx) + cos(psi_p)*(py - qy)

    positive when `point` is to the LEFT of the path's direction of travel.

    Zero-length segments (duplicate waypoints) are skipped; if every
    candidate segment is zero-length, raises ValueError -- a projection is
    meaningless on a degenerate path. A non-finite `point` raises
    ValueError for the same reason: downstream laws must never silently
    track a NaN.
    """
    px, py = float(point[0]), float(point[1])
    if not (math.isfinite(px) and math.isfinite(py)):
        raise ValueError(
            f'project_onto_path: point must be finite, got ({point[0]}, {point[1]})')
    n = len(xy)
    cumulative = compute_cumulative_arc_length(seg_len)

    candidates = []
    # Segment (i-1 -> i): the point *arriving at* nearest_idx.
    prev_idx = nearest_idx - 1
    if closed:
        prev_idx %= n
    if closed or nearest_idx > 0:
        candidates.append(prev_idx)
    # Segment (i -> i+1): the point *leaving* nearest_idx.
    if closed or nearest_idx < n - 1:
        candidates.append(nearest_idx % n)

    best = None
    for seg_idx in candidates:
        a = xy[seg_idx]
        b = xy[(seg_idx + 1) % n]
        length = float(seg_len[seg_idx])
        if length < 1e-9:
            # Duplicate waypoints: zero-length segment has no direction to
            # project onto; skip it.
            continue
        ax, ay = float(a[0]), float(a[1])
        abx, aby = float(b[0]) - ax, float(b[1]) - ay
        # Parametric position of the projection onto the (infinite) segment
        # line, clamped into the segment.
        t = ((px - ax) * abx + (py - ay) * aby) / (length * length)
        t = min(1.0, max(0.0, t))
        qx = ax + t * abx
        qy = ay + t * aby
        dx = px - qx
        dy = py - qy
        dist = math.hypot(dx, dy)
        heading = math.atan2(aby, abx)
        lateral = -math.sin(heading) * dx + math.cos(heading) * dy
        arc_length = float(cumulative[seg_idx]) + t * length
        projection = PathProjection(
            segment_index=int(seg_idx), t=float(t), qx=qx, qy=qy,
            heading=float(heading), lateral_error=float(lateral),
            arc_length=arc_length)
        if best is None or dist < best[0]:
            best = (dist, projection)

    if best is None:
        raise ValueError(
            'project_onto_path: every candidate segment is zero-length '
            '(duplicate waypoints) -- cannot project onto a degenerate path')
    return best[1]


def interpolate_along(values: np.ndarray, projection: PathProjection,
                      closed: bool = True) -> float:
    """Linear interpolation of a per-waypoint field at a path projection.

    `values[i]` is the field at waypoint i; the value at the projected point
    `t` of the way along segment (segment_index -> segment_index+1) is

        (1 - t) * values[segment_index] + t * values[segment_index + 1]

    wrapping the end index on a closed loop. Every tracking law reads its
    per-projection quantities (curvature, feedforward) through this one
    helper so they all interpolate identically.
    """
    n = len(values)
    start = projection.segment_index
    end = start + 1
    if end >= n:
        end = 0 if closed else n - 1
    a = float(values[start])
    b = float(values[end])
    return a + projection.t * (b - a)


# ============================================================================
# 4. Offline raceline generation
# ============================================================================

def smooth_path(xy: np.ndarray, half_window: int, closed: bool = True) -> np.ndarray:
    """Moving-average smoothing of a waypoint path: each point becomes the
    mean of itself and `half_window` neighbors on each side.

    The recorded waypoints come from particle-filter poses sampled every
    ~0.15m, and localization jitter of even a couple of centimeters at
    that spacing reads as *curvature* to estimate_path_curvature -- three
    nearly-colinear points wiggled slightly produce a spuriously tight
    circumcircle. That fake curvature then flows straight into
    compute_velocity_profile as phantom braking zones in the middle of
    straights. Averaging out the jitter before estimating curvature is
    the single highest-leverage cleanup on profile quality.

    For a closed loop the window wraps around the start/finish seam; for
    an open path the ends are averaged over whatever neighbors actually
    exist (the window shrinks at the boundaries rather than wrapping).
    `half_window <= 0` returns the input unchanged. Smoothing pulls
    corners slightly inward (toward the chord), which shortens and
    *tightens* the line a little -- a conservative direction for the
    velocity profile to err in.
    """
    n = len(xy)
    if half_window <= 0 or n < 3:
        return np.asarray(xy, dtype=np.float64).copy()
    half_window = min(half_window, (n - 1) // 2)

    kernel_size = 2 * half_window + 1
    if closed:
        padded = np.vstack([xy[-half_window:], xy, xy[:half_window]])
        cumsum = np.cumsum(np.vstack([[[0.0, 0.0]], padded]), axis=0)
        return (cumsum[kernel_size:] - cumsum[:-kernel_size]) / kernel_size

    smoothed = np.empty_like(xy, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        smoothed[i] = xy[lo:hi].mean(axis=0)
    return smoothed


def estimate_path_curvature(xy: np.ndarray, closed: bool = True) -> np.ndarray:
    """Per-waypoint path curvature via the Menger curvature of each point
    and its immediate neighbors -- no calculus or spline-fitting needed.

    For three points A, B, C, the (unique) circle passing through all
    three has radius:

        R = (|AB| * |BC| * |CA|) / (4 * area(A,B,C))

    (from the identity area = a*b*c / (4R) for a triangle with sides
    a,b,c and circumradius R). Curvature is 1/R:

        kappa = 4 * area(A,B,C) / (|AB| * |BC| * |CA|)

    Tight corners -> small R -> large kappa. Straights (A,B,C nearly
    colinear) -> area ~ 0 -> kappa ~ 0. Using each point's immediate
    neighbors (A=previous waypoint, B=this waypoint, C=next waypoint)
    means this works directly on raw recorded waypoints.

    For an open (non-closed) path the first and last points have no
    "previous"/"next" neighbor respectively; their curvature is left at 0
    (irrelevant in practice -- real tracks are closed loops, which is the
    default).
    """
    n = len(xy)
    prev_pts = np.roll(xy, 1, axis=0)
    next_pts = np.roll(xy, -1, axis=0)

    if not closed:
        prev_pts = prev_pts.copy()
        next_pts = next_pts.copy()
        prev_pts[0] = xy[0]
        next_pts[-1] = xy[-1]

    a = np.hypot(next_pts[:, 0] - xy[:, 0], next_pts[:, 1] - xy[:, 1])              # |B->C|
    b = np.hypot(xy[:, 0] - prev_pts[:, 0], xy[:, 1] - prev_pts[:, 1])              # |A->B|
    c = np.hypot(next_pts[:, 0] - prev_pts[:, 0], next_pts[:, 1] - prev_pts[:, 1])  # |A->C|

    # Twice the *signed* area of triangle ABC via the shoelace/cross-product
    # formula; abs() + 0.5 gives the true (unsigned) area.
    cross = ((xy[:, 0] - prev_pts[:, 0]) * (next_pts[:, 1] - prev_pts[:, 1]) -
             (xy[:, 1] - prev_pts[:, 1]) * (next_pts[:, 0] - prev_pts[:, 0]))
    area = 0.5 * np.abs(cross)

    denom = a * b * c
    kappa = np.zeros(n)
    valid = denom > 1e-9
    kappa[valid] = 4.0 * area[valid] / denom[valid]
    return kappa


def estimate_signed_path_curvature(xy: np.ndarray, closed: bool = True) -> np.ndarray:
    """Menger curvature with sign: + for a left (CCW) turn, - for right, 0 on
    straights. Same magnitude as estimate_path_curvature(xy, closed) everywhere.

    estimate_path_curvature is deliberately *unsigned*: the velocity profile
    (compute_velocity_profile) only ever needs "how tight is this bend", and
    braking for a left hairpin is identical to braking for a right one. The
    tracking laws, by contrast, steer -- and a feedforward term
    (delta_ff = atan(L * kappa)) has to know which *way* the path turns,
    so they read curvature from this signed variant instead.

    The sign comes from the shoelace cross product of (prev -> this) x
    (prev -> next): positive (counterclockwise turn) means the path bends
    left. Magnitude, the degenerate-colinear zero, and the open-path
    endpoints staying 0 are all inherited unchanged from
    estimate_path_curvature -- do not "fix" those here without fixing the
    unsigned original too, or the two drift apart.
    """
    n = len(xy)
    prev_pts = np.roll(xy, 1, axis=0)
    next_pts = np.roll(xy, -1, axis=0)

    if not closed:
        prev_pts = prev_pts.copy()
        next_pts = next_pts.copy()
        prev_pts[0] = xy[0]
        next_pts[-1] = xy[-1]

    a = np.hypot(next_pts[:, 0] - xy[:, 0], next_pts[:, 1] - xy[:, 1])              # |B->C|
    b = np.hypot(xy[:, 0] - prev_pts[:, 0], xy[:, 1] - prev_pts[:, 1])              # |A->B|
    c = np.hypot(next_pts[:, 0] - prev_pts[:, 0], next_pts[:, 1] - prev_pts[:, 1])  # |A->C|

    # Signed twice-area: positive when (A -> B -> C) turns counterclockwise
    # (left). No abs() here -- the sign IS the answer.
    cross = ((xy[:, 0] - prev_pts[:, 0]) * (next_pts[:, 1] - prev_pts[:, 1]) -
             (xy[:, 1] - prev_pts[:, 1]) * (next_pts[:, 0] - prev_pts[:, 0]))
    area = 0.5 * cross

    denom = a * b * c
    kappa = np.zeros(n)
    valid = denom > 1e-9
    kappa[valid] = 4.0 * area[valid] / denom[valid]
    return kappa


def compute_velocity_profile(seg_len: np.ndarray, curvature: np.ndarray,
                              v_max: float, v_min: float,
                              a_lat_max: float, a_accel_max: float, a_brake_max: float,
                              closed: bool = True, smoothing_passes: int = 5,
                              friction_ellipse: bool = True) -> np.ndarray:
    """Turn per-waypoint path curvature into a per-waypoint target speed.
    This is the core of "how fast should the car go here" -- three passes,
    each a standard, well-established piece of racing-line theory:

    1. Cornering limit (simplified friction circle, lateral-only): taking
       a corner of curvature kappa at speed v produces lateral
       acceleration a_lat = v^2 * kappa (uniform circular motion). Solving
       for the fastest speed that keeps a_lat under the tire/chassis limit
       a_lat_max:

           v_corner = sqrt(a_lat_max / kappa)

    2. Forward (acceleration) pass: the car can only speed up as fast as
       a_accel_max allows between two waypoints ds apart:

           v_i <= sqrt(v_(i-1)^2 + 2 * a_accel_max * ds)

       Sweeping the array forward and capping each point at this limits
       how quickly the profile is allowed to ramp speed up, e.g. coming
       out of a hairpin onto a straight.

    3. Backward (braking) pass: symmetric to (2), using a_brake_max, swept
       from the end of the array backward. This is the pass that actually
       creates *braking zones*: it propagates a low corner-speed limit
       backward along the straight leading into that corner, so the
       profile tells the car to start slowing down early enough to
       actually make the corner, instead of arriving too fast and only
       "finding out" at the apex.

    A closed loop has no single first point to seed sweep (1)/(2) from --
    index 0's "previous" point is the *last* index, whose value hasn't
    been finalized yet on the first sweep. Repeating the forward+backward
    pair (`smoothing_passes` times, default 5) lets that start/finish seam
    converge. Both passes only ever lower values (never raise them), so
    repeating them on an already-converged open path is a harmless no-op
    -- there is no need to special-case open vs. closed here.

    With `friction_ellipse` (default on), passes (2)/(3) don't get the
    tire's *full* longitudinal budget while the car is also cornering:
    grip is one shared resource, and lateral acceleration spends it too.
    The available longitudinal fraction follows the standard friction
    ellipse -- (a_long/a_long_max)^2 + (a_lat/a_lat_max)^2 <= 1, so

        a_long_available = a_long_max * sqrt(1 - (v^2*kappa / a_lat_max)^2)

    evaluated at the neighbor whose speed is already known in each sweep.
    On a straight (kappa ~ 0) this is the full budget; at a corner already
    at the cornering limit it's zero (all grip is lateral). The effect is
    strictly-lower (more conservative) speeds around corner entry/exit --
    honest about physics the uncoupled version quietly ignores. Pass
    `friction_ellipse=False` for the old lateral/longitudinal-independent
    behavior (useful for comparing profiles).

    This is *not* a time-optimal racing line -- true time-optimality needs
    a nonlinear optimizer over both the path geometry and the speed
    simultaneously (e.g. minimum-curvature QP solvers like TUM's
    global_racetrajectory_optimization). This is a fast, dependency-light,
    easy-to-verify approximation that is nonetheless the same "cornering
    speed + forward/backward smoothing" technique used in mainstream
    racing-line tooling -- it just doesn't also re-optimize the path
    geometry itself. See docs/racing-autonomy.md for more on this
    trade-off and how to go further.
    """
    n = len(curvature)
    eps = 1e-6
    v_corner = np.sqrt(a_lat_max / np.maximum(curvature, eps))
    v = np.minimum(v_corner, v_max).astype(float)

    def long_fraction(speed: float, kappa: float) -> float:
        # Fraction of the longitudinal budget left after cornering at
        # `speed` through curvature `kappa` (friction ellipse). v_corner
        # already caps a_lat at a_lat_max, so the sqrt argument only goes
        # negative through float round-off -- clamp at 0.
        if not friction_ellipse:
            return 1.0
        lat_fraction = (speed * speed * kappa) / a_lat_max
        return math.sqrt(max(0.0, 1.0 - lat_fraction * lat_fraction))

    for _ in range(max(1, smoothing_passes)):
        # Forward / acceleration-limited sweep.
        fwd_order = range(n) if closed else range(1, n)
        for i in fwd_order:
            prev_i = (i - 1) % n
            a_avail = a_accel_max * long_fraction(v[prev_i], curvature[prev_i])
            v_cap = math.sqrt(v[prev_i] ** 2 + 2.0 * a_avail * seg_len[prev_i])
            if v_cap < v[i]:
                v[i] = v_cap

        # Backward / braking-limited sweep.
        bwd_order = range(n - 1, -1, -1) if closed else range(n - 2, -1, -1)
        for i in bwd_order:
            next_i = (i + 1) % n
            a_avail = a_brake_max * long_fraction(v[next_i], curvature[next_i])
            v_cap = math.sqrt(v[next_i] ** 2 + 2.0 * a_avail * seg_len[i])
            if v_cap < v[i]:
                v[i] = v_cap

    return np.clip(v, v_min, v_max)


def estimate_lap_time(seg_len: np.ndarray, speed: np.ndarray, closed: bool = True) -> float:
    """Rough lap/pass time estimate: sum, over every segment, of (segment
    length / average speed across that segment).

    This is a simple kinematic estimate from the speed profile alone -- it
    assumes the car can actually *hit* this speed profile exactly, which a
    real controller only approximates. Treat it as a way to compare two
    tunings against each other, not as a lap-time guarantee.
    """
    n = len(speed)
    count = n if closed else n - 1
    total = 0.0
    for i in range(count):
        j = (i + 1) % n
        avg_speed = 0.5 * (speed[i] + speed[j])
        if avg_speed > 1e-6:
            total += seg_len[i] / avg_speed
    return total


# ============================================================================
# 5. CSV file I/O
# ============================================================================
#
# Two file "shapes" are used across this package:
#   raw waypoints      -- header "x,y"       -- written by waypoint_recorder_node
#   profiled waypoints -- header "x,y,speed" -- written by generate_velocity_profile,
#                                                read by pure_pursuit_node

def load_xy_csv(path: str) -> np.ndarray:
    """Load a raw (x, y) waypoints CSV into an (N, 2) array."""
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            rows.append((float(row[0]), float(row[1])))
    return np.array(rows, dtype=np.float64)


def save_xy_csv(path: str, xy: np.ndarray) -> None:
    """Write an (N, 2) array of (x, y) points as a raw waypoints CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        for x, y in xy:
            writer.writerow([f'{x:.4f}', f'{y:.4f}'])


def load_profiled_csv(path: str):
    """Load a profiled (x, y, speed) waypoints CSV.

    Returns (xy, speed): xy is an (N, 2) array, speed is an (N,) array.
    Raises ValueError if the file doesn't look like a profiled file (i.e.
    is missing the speed column) -- most likely a raw recording that
    hasn't been run through generate_velocity_profile yet.
    """
    rows = []
    speeds = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or len(header) < 3:
            raise ValueError(
                f"'{path}' does not look like a profiled waypoints file "
                f"(expected a header with 3 columns, 'x,y,speed'). "
                f"Run generate_velocity_profile on it first."
            )
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) < 3:
                raise ValueError(
                    f"'{path}' line {line_no}: expected 3 columns (x, y, speed), "
                    f"got {len(row)}")
            rows.append((float(row[0]), float(row[1])))
            speeds.append(float(row[2]))
    xy = np.array(rows, dtype=np.float64).reshape(-1, 2)
    speed = np.array(speeds, dtype=np.float64)
    # float() happily parses 'nan', 'inf' and '-1'. A NaN or inf point
    # poisons nearest-waypoint search; a NaN speed survives np.clip all the
    # way to /drive or raises mid-control-step; a negative speed is a
    # reverse command nothing upstream ever meant. Refuse the file instead.
    bad = ~np.isfinite(xy).all(axis=1) | ~np.isfinite(speed)
    if np.any(bad):
        raise ValueError(
            f"'{path}' has a non-finite value on data row "
            f"{int(np.argmax(bad)) + 1}")
    if np.any(speed < 0.0):
        raise ValueError(
            f"'{path}' has a negative speed on data row "
            f"{int(np.argmax(speed < 0.0)) + 1}; profiled speeds must be >= 0")
    return xy, speed


def save_profiled_csv(path: str, xy: np.ndarray, speed: np.ndarray) -> None:
    """Write (x, y, speed) waypoints -- the file pure_pursuit_node loads."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'speed'])
        for (x, y), v in zip(xy, speed):
            writer.writerow([f'{x:.4f}', f'{y:.4f}', f'{v:.3f}'])


# ============================================================================
# 6. Reactive obstacle avoidance (gap-finding, reused from gap_follow)
# ============================================================================
#
# pure_pursuit_node's racing line has no idea an opponent's car, a
# spun-out car, or anything else not in the map exists. The reactive
# LIDAR safety net catches this at runtime, in two tiers: emergency stop
# if something is very close, or -- what this function is for -- steer
# *around* it if there's room, instead of just stopping. This is the same
# "find the best gap" idea gap_follow uses to drive with no map at all,
# reused here as a temporary override on top of the racing line rather
# than as the whole control strategy.

def find_best_gap(ranges: np.ndarray, min_gap_distance: float):
    """Find the best drivable opening in a window of range readings.

    A "gap" is a contiguous run of readings all farther than
    `min_gap_distance`. Candidates are scored by width * average_depth,
    not width alone: a shallow dead end (e.g. a ~1m doorway alcove) can
    subtend a *wider* angle than a genuinely open corridor that's
    angularly narrower but far deeper. Scoring this way means a gap has
    to actually stay open for a while, not just be wide at its mouth, to
    win -- the same reasoning gap_follow's own gap search uses.

    Returns (start_index, end_index) of the winning run within `ranges`,
    or (None, None) if nothing in the window is farther than
    `min_gap_distance` at all (boxed in on every side).
    """
    free = ranges > min_gap_distance
    candidates = []
    run_start = None
    for i, is_free in enumerate(free):
        if is_free and run_start is None:
            run_start = i
        elif not is_free and run_start is not None:
            candidates.append((run_start, i - 1))
            run_start = None
    if run_start is not None:
        candidates.append((run_start, len(free) - 1))

    if not candidates:
        return None, None

    def score(run):
        start, end = run
        segment = ranges[start:end + 1]
        width = end - start + 1
        avg_depth = float(np.mean(segment))
        return width * avg_depth

    return max(candidates, key=score)


# ============================================================================
# 7. Opponent detection, tracking, and overtaking
# ============================================================================
#
# "Proper racing" against another car needs more than reactive avoidance
# -- it needs to notice the other car is *there*, work out whether it's
# gaining or falling behind relative to it, and -- if it's gaining --
# actually get past, not just follow at a safe distance forever. This
# section is the math behind that, in three steps that mirror how a
# human driver actually thinks about a car ahead:
#
#   1. "Is that a car?"           -- detect_opponent_cluster / cluster_scan_ranges
#   2. "Are they pulling away, or am I catching them?" -- track progress
#      along the racing line itself (compute_cumulative_arc_length /
#      track_progress_gap), not raw x/y -- see the note on why below.
#   3. "Which side has room, and how do I actually steer there?"  --
#      pick_pass_side / lateral_offset_point
#
# None of this needs a second sensor, a neural network, or radio contact
# with the other car -- just the same LIDAR scan and racing line already
# in use everywhere else in this node.

def cluster_scan_ranges(ranges: np.ndarray, max_range: float, cluster_gap_threshold: float = 0.3):
    """Group consecutive LIDAR readings into clusters: contiguous runs of
    "something is there" (a reading noticeably less than max_range, i.e.
    not just open track) where consecutive readings don't jump by more
    than `cluster_gap_threshold`. A big jump between neighbors means a
    *different* object even if both readings are "close" -- e.g. a car
    sitting in front of a wall shows up as one cluster for the car, a
    jump, then a separate cluster for the wall behind it.

    Returns a list of (start_index, end_index) tuples, one per cluster, in
    scan order. Deliberately simple -- no calibration, no learned model,
    just "is there a return here, and is it continuous with its
    neighbor" -- see detect_opponent_cluster for how that's enough to
    tell an isolated, car-sized object apart from a wall.
    """
    is_object = ranges < (max_range * 0.95)
    clusters = []
    start = None
    for i in range(len(ranges)):
        if is_object[i]:
            if start is None:
                start = i
            elif abs(float(ranges[i]) - float(ranges[i - 1])) > cluster_gap_threshold:
                clusters.append((start, i - 1))
                start = i
        else:
            if start is not None:
                clusters.append((start, i - 1))
                start = None
    if start is not None:
        clusters.append((start, len(ranges) - 1))
    return clusters


def cluster_geometry(ranges: np.ndarray, angle_min: float, angle_increment: float,
                      start_idx: int, end_idx: int):
    """Physical size and centroid of a cluster (see cluster_scan_ranges),
    relative to the LIDAR. `physical_width` is the straight-line (chord)
    distance between the cluster's first and last point in Cartesian
    coordinates -- a much better estimate of an object's actual size than
    its angular width alone, which exaggerates anything close and
    shrinks anything far away.

    Returns (physical_width, centroid_range, centroid_angle).
    """
    angles = angle_min + np.arange(start_idx, end_idx + 1) * angle_increment
    segment = ranges[start_idx:end_idx + 1]
    xs = segment * np.cos(angles)
    ys = segment * np.sin(angles)
    physical_width = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    centroid_idx = (start_idx + end_idx) // 2
    centroid_range = float(ranges[centroid_idx])
    centroid_angle = float(angle_min + centroid_idx * angle_increment)
    return physical_width, centroid_range, centroid_angle


def detect_opponent_cluster(ranges: np.ndarray, angle_min: float, angle_increment: float,
                             max_range: float, min_width: float, max_width: float,
                             max_engagement_range: float, cluster_gap_threshold: float = 0.3,
                             open_side_margin: float = 0.5):
    """Look for a single object in the scan shaped and sized like another
    race car, sitting out in the open track rather than flush against a
    wall.

    A car-sized object looks like: a cluster of readings noticeably
    closer than its surroundings (by at least `open_side_margin`),
    spanning a physical width between `min_width` and `max_width` --
    narrow enough it isn't a wall segment (walls produce much longer or
    far more irregular runs), wide enough it isn't sensor noise or a
    thin post. Among every cluster that qualifies, the closest one wins
    (the one most immediately relevant to a racing decision right now).

    This is a geometric heuristic, not object recognition -- no map is
    consulted, no machine learning, and nothing is tracked between calls
    (see OpponentTracker in pure_pursuit_node.py for that part). It's
    exactly the kind of reasoning gap_follow already does for its own gap
    search, just aimed at *finding* an object instead of *avoiding* one.

    Returns (start_idx, end_idx, centroid_range, centroid_angle) for the
    winning cluster, or None if nothing currently in view qualifies.
    """
    best = None
    for start_idx, end_idx in cluster_scan_ranges(ranges, max_range, cluster_gap_threshold):
        width, centroid_range, centroid_angle = cluster_geometry(
            ranges, angle_min, angle_increment, start_idx, end_idx)
        if not (min_width <= width <= max_width):
            continue
        if centroid_range > max_engagement_range:
            continue

        # Confirm open space immediately on both sides -- otherwise this
        # is plausibly just one section of a curving wall, not a
        # discrete object sitting in the middle of the drivable track.
        before = ranges[max(0, start_idx - 5):start_idx]
        after = ranges[end_idx + 1:end_idx + 6]
        if before.size and float(np.mean(before)) < centroid_range + open_side_margin:
            continue
        if after.size and float(np.mean(after)) < centroid_range + open_side_margin:
            continue

        if best is None or centroid_range < best[2]:
            best = (start_idx, end_idx, centroid_range, centroid_angle)

    return best


def lead_speed(speed_profile: np.ndarray, cumulative: np.ndarray, total_length: float,
               nearest_idx: int, measured_speed: float, lead_time: float, closed: bool = True) -> float:
    """The profile's speed for the car at waypoint ``nearest_idx``, accelerating ``lead_time`` early.

    A speed loop answers a command late: car 2's fitted VESC loop waits about 0.1 s, then closes on the
    command at 7/s, so following a profile that rises at a m/s^2 it trails by about a/7 m/s (0.4-0.6 m/s
    out of every corner on the 6 Nov courses). Where the profile rises within the distance the car covers
    in ``lead_time`` (at its measured speed, at least 0.5 m/s), this returns the higher speed found there,
    linearly interpolated, so the speed loop starts climbing on time. It never returns less than the
    profile here: braking is left to the planned profile, whose braking points already move earlier by the
    speed loop's delay (min_time_line._respond_early). lead_time <= 0 returns speed_profile[nearest_idx]
    exactly. Any curvature ceiling or slew stage after this still applies to the result.
    """
    here = float(speed_profile[nearest_idx])
    if not (lead_time > 0.0) or not math.isfinite(lead_time):
        return here
    v = abs(float(measured_speed)) if math.isfinite(float(measured_speed)) else 0.0
    reach = max(v, 0.5) * float(lead_time)
    s_here = float(cumulative[nearest_idx])
    s_ahead = s_here + reach
    n = len(speed_profile)
    if closed and total_length > 0.0:
        xs = np.concatenate([cumulative, [total_length], cumulative + total_length])
        vs = np.concatenate([speed_profile, [speed_profile[0]], speed_profile])
    else:
        xs, vs = np.asarray(cumulative, dtype=float), np.asarray(speed_profile, dtype=float)
        s_ahead = min(s_ahead, float(cumulative[n - 1]))
    ahead = float(np.interp(s_ahead, xs, vs))
    return max(here, ahead)


def compute_cumulative_arc_length(seg_len: np.ndarray) -> np.ndarray:
    """Running total of track distance up to (not including) each
    waypoint: cumulative[i] is how far along the track waypoint i is from
    waypoint 0. Precomputed once at startup (the racing line doesn't
    change during a run) so "how far along the track is index i" is an
    O(1) array read instead of re-summing seg_len from scratch -- needed
    at least once per control tick, for both the ego car and any tracked
    opponent.
    """
    cumulative = np.zeros(len(seg_len))
    cumulative[1:] = np.cumsum(seg_len[:-1])
    return cumulative


def track_progress_gap(ego_arc_length: float, other_arc_length: float, total_length: float) -> float:
    """Signed track-distance from the ego car to another point on a
    closed-loop racing line, expressed as "how far *ahead*" the other
    point is: always in [0, total_length), wrapping the finish line back
    to the start -- on a loop, "ahead" always exists somewhere less than
    one full lap away, which is exactly what predicting an opponent's
    position needs (see OpponentTracker.predicted_arc_length).
    """
    if total_length <= 0.0:
        return 0.0
    return float((other_arc_length - ego_arc_length) % total_length)


def track_lead_distance(ego_arc_length: float, other_arc_length: float,
                        total_length: float) -> float:
    """How far the ego car is *past* another point, as a signed distance.

    Positive means the ego is ahead by that many meters; negative means the
    other point is still ahead of the ego. Unlike track_progress_gap, which
    wraps everything into [0, total_length) and so cannot tell "3 m behind"
    from "all but 3 m ahead", this picks the shorter way round -- which is
    the only sensible reading for two cars racing near each other.

    This distinction caused a real collision. Asking track_progress_gap
    whether a pass is finished invites `gap > total_length - clear_margin`,
    which looks like "at least clear_margin past" and in fact means "at
    most clear_margin past" -- true the instant the ego's nose edges in
    front, while the two cars are still side by side. Compare against this
    function instead: `track_lead_distance(...) >= clear_margin`.
    """
    if total_length <= 0.0:
        return 0.0
    half = total_length / 2.0
    behind = (other_arc_length - ego_arc_length + half) % total_length - half
    return float(-behind)


def side_clearance(ranges: np.ndarray, angle_min: float, angle_increment: float,
                   side: int, half_span: float, range_min: float,
                   range_max: float) -> float:
    """Perpendicular distance to the wall on one side of the car.

    Mirrors gap_logic.side_wall_distance (the two packages deliberately keep
    separate copies of their framework-free helpers). The *minimum* valid
    range in a window centred on the side beam is the correct estimator, not
    a conservative fudge: a beam striking a locally straight wall at angle
    theta off the perpendicular reads d/cos(theta), minimised exactly at the
    perpendicular, so the window recovers the true perpendicular distance
    even with the car yawed relative to the wall.

    ``side`` is +1 for the left wall, -1 for the right. Returns ``inf`` when
    no beam in the window came back usable, which callers must read as "no
    wall seen here", not "very far away".
    """
    values = np.asarray(ranges, dtype=np.float64)
    if not math.isfinite(half_span) or half_span <= 0.0:
        raise ValueError('half_span must be finite and positive')
    angles = angle_min + np.arange(values.size, dtype=np.float64) * angle_increment
    # A reading at exactly range_max is "nothing out there", not a wall; the
    # node clips inf to max_range before this ever sees it.
    valid = np.isfinite(values) & (values > range_min) & (values < range_max)
    inside = valid & (np.abs(angles - side * (math.pi / 2.0)) <= half_span)
    if not np.any(inside):
        return math.inf
    return float(np.min(values[inside]))


def overtake_side_has_room(ranges: np.ndarray, angle_min: float,
                           angle_increment: float, side: int,
                           required_clearance: float, half_span: float,
                           range_min: float, range_max: float) -> bool:
    """Is there physically room to sit ``required_clearance`` off the line?

    pick_pass_side answers "which side is *more* open", and always names one.
    It never asks whether that side is open *enough*. Without this check the
    car commits to a pass, steers overtake_lateral_offset toward a wall with
    the avoidance tier suppressed for the duration, and only reacts at the
    emergency-stop distance -- measured stopping dead with 0.19-0.34 m of
    forward clearance, which is where the traffic scenario deadlocks.

    Seeing no wall at all on that side counts as room: an open side is the
    easy case, not a missing measurement.
    """
    if required_clearance <= 0.0:
        return True
    clearance = side_clearance(ranges, angle_min, angle_increment, side,
                               half_span, range_min, range_max)
    return clearance >= required_clearance


def braking_speed_limit(clearance: float, reserve_distance: float,
                        max_braking_decel: float, max_speed: float) -> float:
    """Speed ceiling that still allows stopping short: v^2 <= 2*a*(d - reserve).

    Mirrors gap_logic.braking_speed_limit. Infinite clearance means nothing
    was seen, so the configured top speed stands.
    """
    if clearance == math.inf:
        return max_speed
    if not all(math.isfinite(value) for value in (
            clearance, reserve_distance, max_braking_decel, max_speed)):
        raise ValueError('braking speed-limit inputs must be finite')
    if reserve_distance < 0.0 or max_braking_decel <= 0.0 or max_speed < 0.0:
        raise ValueError(
            'reserve_distance/max_speed must be non-negative and braking decel positive')
    usable_distance = max(0.0, clearance - reserve_distance)
    return min(max_speed, math.sqrt(2.0 * max_braking_decel * usable_distance))


def lateral_offset_point(xy: np.ndarray, index: int, next_index: int, offset: float):
    """A point `offset` meters to the *left* of waypoint `index` (negative
    offset = right), measured perpendicular to the track's local
    direction of travel (estimated from index -> next_index). Used to
    nudge the Pure Pursuit steering target sideways during an overtake,
    without needing a whole second recorded racing line -- see
    "Overtaking" in docs/racing-autonomy.md.
    """
    dx = xy[next_index][0] - xy[index][0]
    dy = xy[next_index][1] - xy[index][1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return float(xy[index][0]), float(xy[index][1])
    # Perpendicular-left unit vector of the direction of travel (dx, dy):
    # rotate it +90 degrees, matching pick_pass_side's left/right sense.
    perp_x, perp_y = -dy / length, dx / length
    return float(xy[index][0] + perp_x * offset), float(xy[index][1] + perp_y * offset)


def pick_pass_side(ranges: np.ndarray, start_idx: int, end_idx: int, window: int = 20) -> int:
    """Which side of a detected cluster has more open room to pass
    through: the average range in a small window of readings just
    outside the cluster on the left vs. the right (higher scan index is
    further left, per REP-103 -- angle increases with index, and +angle
    means +Y, which is left). Returns +1 (pass on the left) or -1 (pass
    on the right); a tie breaks toward the left arbitrarily -- both
    sides being this close to equal means neither choice is meaningfully
    better.
    """
    left = ranges[end_idx + 1:end_idx + 1 + window]
    right = ranges[max(0, start_idx - window):start_idx]
    left_room = float(np.mean(left)) if left.size else 0.0
    right_room = float(np.mean(right)) if right.size else 0.0
    return 1 if left_room >= right_room else -1


# ----------------------------------------------------------------------------
# Map-subtraction opponent detection
# ----------------------------------------------------------------------------
#
# detect_opponent_cluster above has to *guess* "car vs. wall" from cluster
# shape alone, which is exactly what fools it on curving walls and clipped
# clusters. When a map and a localized pose are available (they are,
# whenever pure_pursuit is racing), there's a categorically better signal:
# ray-cast the scan the LIDAR *should* see from the current pose given
# only the map (see map_subtraction.py for the range_libc wrapper), and
# compare it with the scan it *actually* sees. Any beam that comes back
# meaningfully shorter than the map predicts is, by definition, hitting
# something that is not in the map -- an opponent, wherever it is and
# whatever the wall behind it looks like. The comparison/clustering here
# is kept free of range_libc so it stays unit-testable like everything
# else in this file.

def dynamic_beam_mask(measured: np.ndarray, expected: np.ndarray,
                       margin: float, range_min: float = 0.0) -> np.ndarray:
    """Boolean mask of beams whose measured range is at least `margin`
    shorter than the map-predicted range -- i.e. beams hitting something
    that is not in the map. Non-finite or sub-`range_min` measurements are
    never flagged (an invalid beam is *unknown*, not evidence of an
    object), and neither is anything the map already explains.
    """
    measured = np.asarray(measured, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    valid = np.isfinite(measured) & (measured >= range_min)
    return valid & (measured < (expected - margin))


def detect_dynamic_cluster(measured: np.ndarray, expected: np.ndarray,
                            angle_min: float, angle_increment: float,
                            margin: float, min_width: float, max_width: float,
                            max_engagement_range: float, range_min: float = 0.0,
                            cluster_gap_threshold: float = 0.3):
    """Find the closest car-plausible cluster of *dynamic* beams -- beams
    the map fails to explain (see dynamic_beam_mask). Contiguous flagged
    beams form a cluster, split where consecutive measured ranges jump by
    more than `cluster_gap_threshold` (two unmapped objects at different
    depths must not merge into one). Width and engagement-range gating
    reuse the same limits as the heuristic detector, but note the width
    check here is only guarding against garbage (a flock of noise beams,
    two cars merged), not doing the car-vs-wall discrimination -- the map
    subtraction itself already did that.

    Returns (start_idx, end_idx, centroid_range, centroid_angle) in the
    same shape detect_opponent_cluster returns, or None. Indices are into
    `measured`/`expected` -- if the caller downsampled the scan, it owns
    mapping them back (and must pass the *downsampled* angle_increment).
    """
    mask = dynamic_beam_mask(measured, expected, margin, range_min)

    best = None
    start = None
    n = len(measured)
    for i in range(n + 1):
        in_cluster = i < n and mask[i]
        split = (in_cluster and start is not None
                 and abs(float(measured[i]) - float(measured[i - 1])) > cluster_gap_threshold)
        if in_cluster and start is None:
            start = i
            continue
        if in_cluster and not split:
            continue
        if start is not None:
            end = i - 1
            width, centroid_range, centroid_angle = cluster_geometry(
                measured, angle_min, angle_increment, start, end)
            if (min_width <= width <= max_width
                    and centroid_range <= max_engagement_range
                    and (best is None or centroid_range < best[2])):
                best = (start, end, centroid_range, centroid_angle)
            start = i if split else None
    return best


def body_arc_free_distance(clean: np.ndarray, valid: np.ndarray, angles: np.ndarray,
                           curvature: float, half_width: float, car_length: float,
                           wheelbase: float, laser_offset_x: float,
                           laser_offset_y: float = 0.0,
                           max_length: float = 1.0, step: float = 0.02) -> float:
    """Distance base_link can travel along a constant-curvature arc before any
    valid scan point lies inside the body rectangle. inf if none within
    max_length. 0.0 only if a point is inside the rectangle at the start.

    The rectangle is x in [rear, front], |y| <= half_width in base_link, with
    front = wheelbase/2 + car_length/2 and rear = wheelbase/2 - car_length/2
    (the model ``swept_path_clearance`` uses). Scan points are fixed in the
    starting base_link frame and the body moves. Unlike
    ``swept_path_clearance``, a point alongside the body but outside the
    rectangle is not an obstacle unless the body moves into it. REP-103:
    positive curvature turns left.
    """
    ranges = np.asarray(clean, dtype=np.float64)
    validity = np.asarray(valid, dtype=bool)
    beam_angles = np.asarray(angles, dtype=np.float64)
    if not (ranges.shape == validity.shape == beam_angles.shape):
        raise ValueError('ranges, validity, and angles must have matching shapes')
    for name, value in (('curvature', curvature), ('half_width', half_width),
                        ('car_length', car_length), ('wheelbase', wheelbase),
                        ('laser_offset_x', laser_offset_x),
                        ('laser_offset_y', laser_offset_y),
                        ('max_length', max_length), ('step', step)):
        if not math.isfinite(value):
            raise ValueError(f'{name} must be finite')
    for name, value in (('half_width', half_width), ('car_length', car_length),
                        ('wheelbase', wheelbase), ('max_length', max_length),
                        ('step', step)):
        if value <= 0.0:
            raise ValueError(f'{name} must be positive')

    front = wheelbase / 2.0 + car_length / 2.0
    rear = wheelbase / 2.0 - car_length / 2.0

    usable = validity & (ranges > 0.0) & np.isfinite(ranges) & np.isfinite(beam_angles)
    if not np.any(usable):
        return math.inf
    px = ranges[usable] * np.cos(beam_angles[usable]) + laser_offset_x
    py = ranges[usable] * np.sin(beam_angles[usable]) + laser_offset_y

    travel = np.arange(0.0, max_length + 0.5 * step, step)
    travel = travel[travel <= max_length + 1e-12]
    if abs(curvature) < 1e-6:
        bx = travel
        by = np.zeros_like(travel)
        heading = np.zeros_like(travel)
    else:
        heading = curvature * travel
        bx = np.sin(heading) / curvature
        by = (1.0 - np.cos(heading)) / curvature
    cos_h = np.cos(heading)[:, None]
    sin_h = np.sin(heading)[:, None]
    dx = px[None, :] - bx[:, None]
    dy = py[None, :] - by[:, None]
    local_x = cos_h * dx + sin_h * dy
    local_y = -sin_h * dx + cos_h * dy
    hit = ((local_x >= rear) & (local_x <= front) & (np.abs(local_y) <= half_width)).any(axis=1)
    if not hit.any():
        return math.inf
    return float(travel[int(np.argmax(hit))])


def vehicle_boundary_distances(angles: np.ndarray, car_width: float,
                               car_length: float, wheelbase: float,
                               laser_offset_x: float,
                               laser_offset_y: float = 0.0) -> np.ndarray:
    """Distance from the LiDAR to the rectangular vehicle edge, per beam.

    Subtracting this from a raw LiDAR range converts "how far is that thing
    from the *sensor*" into "how far is it from the *car body*" -- the only
    form that answers whether the car is about to touch something. A forward
    cone minimum cannot: a wall alongside the car reads as a comfortable
    range on a beam pointing sideways while the bodywork is millimetres away.

    ``base_link`` is the rear axle in this workspace; following the F1TENTH
    collision model the rectangle is centred halfway along the wheelbase with
    symmetric overhang, and the LiDAR sits at ``laser_offset_x/y`` from
    ``base_link``. A sensor outside the footprint means the transform or the
    footprint is wrong, which would silently *understate* collision distance,
    so that raises rather than returning a dangerous number.

    Deliberately duplicated from ``gap_follow``'s ``gap_logic.py`` rather
    than imported: packages in this workspace talk over topics only, never
    by direct cross-package import (see CLAUDE.md). Keep the two in sync --
    ``gap_logic.vehicle_boundary_distances`` is the reference implementation.
    """
    beam_angles = np.asarray(angles, dtype=np.float64)
    dimensions = (car_width, car_length, wheelbase)
    if not all(math.isfinite(value) and value > 0.0 for value in dimensions):
        raise ValueError('car_width, car_length, and wheelbase must be finite and positive')
    if not math.isfinite(laser_offset_x) or not math.isfinite(laser_offset_y):
        raise ValueError('LiDAR offsets must be finite')

    half_width = car_width / 2.0
    half_length = car_length / 2.0
    body_center_x = wheelbase / 2.0
    x_min = body_center_x - half_length
    x_max = body_center_x + half_length
    y_min, y_max = -half_width, half_width
    tolerance = 1e-9
    if not (x_min - tolerance <= laser_offset_x <= x_max + tolerance
            and y_min - tolerance <= laser_offset_y <= y_max + tolerance):
        raise ValueError('LiDAR origin must lie inside the configured vehicle footprint')

    direction_x = np.cos(beam_angles)
    direction_y = np.sin(beam_angles)
    epsilon = 1e-12
    distance_x = np.full(beam_angles.shape, np.inf, dtype=np.float64)
    distance_y = np.full(beam_angles.shape, np.inf, dtype=np.float64)

    positive_x = direction_x > epsilon
    negative_x = direction_x < -epsilon
    distance_x[positive_x] = (x_max - laser_offset_x) / direction_x[positive_x]
    distance_x[negative_x] = (x_min - laser_offset_x) / direction_x[negative_x]

    positive_y = direction_y > epsilon
    negative_y = direction_y < -epsilon
    distance_y[positive_y] = (y_max - laser_offset_y) / direction_y[positive_y]
    distance_y[negative_y] = (y_min - laser_offset_y) / direction_y[negative_y]

    return np.minimum(distance_x, distance_y)


def minimum_footprint_clearance(ranges: np.ndarray, valid: np.ndarray,
                                boundary_distances: np.ndarray) -> float:
    """Smallest obstacle clearance measured from the vehicle body, over
    every valid beam in any direction. ``inf`` when nothing is usable."""
    measured = np.asarray(ranges, dtype=np.float64)
    validity = np.asarray(valid, dtype=bool)
    boundaries = np.asarray(boundary_distances, dtype=np.float64)
    if measured.shape != validity.shape or measured.shape != boundaries.shape:
        raise ValueError('ranges, validity, and boundary distances must have matching shapes')
    usable = validity & np.isfinite(boundaries)
    if not np.any(usable):
        return math.inf
    return float(np.min(measured[usable] - boundaries[usable]))


def minimum_footprint_clearance_in_cone(
        ranges: np.ndarray, valid: np.ndarray, angles: np.ndarray,
        boundary_distances: np.ndarray, cone_width_rad: float) -> float:
    """Smallest body clearance inside a forward-centred angular cone.

    The narrower sibling of ``minimum_footprint_clearance`` above: a
    minimum over a strict subset of the same beams, so this always returns
    a value ``>=`` the wider function's result on the same scan -- equality
    (within float slop) means the point driving the wider minimum is also
    inside this narrower cone, i.e. dead ahead rather than at the flank.
    That comparison is the entire directional escape gate both driving
    nodes use to tell a forward contact (escapable by steering toward a
    gap) from a flank one (not -- steering "toward the gap" would drag the
    body along whatever the flank is already touching).

    Deliberately duplicated from ``gap_follow``'s ``gap_logic.py`` rather
    than imported -- see ``vehicle_boundary_distances`` above for why.
    ``gap_logic.minimum_footprint_clearance_in_cone`` is the reference
    implementation; keep the two in sync.
    """
    measured = np.asarray(ranges, dtype=np.float64)
    validity = np.asarray(valid, dtype=bool)
    beam_angles = np.asarray(angles, dtype=np.float64)
    boundaries = np.asarray(boundary_distances, dtype=np.float64)
    if not (measured.shape == validity.shape == beam_angles.shape == boundaries.shape):
        raise ValueError('all forward-clearance inputs must have matching shapes')
    if not math.isfinite(cone_width_rad) or not (
            0.0 < cone_width_rad <= 2.0 * math.pi):
        raise ValueError('cone_width_rad must be finite and in (0, 2*pi]')

    inside_cone = np.abs(beam_angles) <= cone_width_rad / 2.0
    return minimum_footprint_clearance(
        measured[inside_cone], validity[inside_cone], boundaries[inside_cone])


# ============================================================================
# 8. Swept-path clearance -- the arc check behind pure_pursuit_node's Tier 1
# ============================================================================
#
# Deliberately duplicated from gap_follow's gap_logic.py rather than imported
# (see vehicle_boundary_distances above for why). gap_logic.swept_path_clearance
# is the reference implementation: the function below must stay byte-identical
# to it, and test_racing_math.py compares the two files' text to hold that.
# pure_pursuit_node passes the same +/-body_clearance_fov_deg window that
# _footprint_clearance uses, never the whole scan: a return behind the sensor
# can be the car's own chassis, which reads as "alongside the body" (0.0) on
# every tick. See docs/racing-autonomy.md, reactive safety.


def swept_path_clearance(clean: np.ndarray, valid: np.ndarray,
                         angles: np.ndarray, curvature: float,
                         half_width: float, car_length: float,
                         wheelbase: float, laser_offset_x: float,
                         laser_offset_y: float = 0.0,
                         max_length: float = 3.0) -> float:
    """How far the car can drive along a constant-curvature arc before its
    body reaches a scanned obstacle, or ``inf`` if nothing is in the way
    within ``max_length``.

    This is the question the forward-cone clearance was standing in for, and
    on a narrow corridor the cone answers it badly. The cone takes the
    *minimum* range over +/-30deg, and in a 1.5m hallway the beams at +/-30deg
    land on the side walls ~1.25m away -- so "forward clearance" there is
    really side-wall distance, which a car driving straight down the hallway
    will never hit. That number capped the car at ~2.4m/s on straights with
    nothing ahead of it (docs/gap-follow-improvement-plan.md section 3.2).

    Here only obstacles inside the band the body actually sweeps count, and
    the distance is measured *along* the arc. ``base_link`` is the rear axle
    (the bicycle model turns about it), so the arc is the rear axle's path,
    and the band is the region the rectangular body covers as it follows
    it: for a turn of radius R the body spans radii ``R - half_width`` (the
    inner side, alongside the rear axle) out to
    ``hypot(front, R + half_width)`` (the outer front corner). The inner
    edge is what models off-tracking -- the rear of the car cuts inside the
    path the LiDAR traces, which is how a car aimed past an inner corner
    still clips it with its flank.

    ``half_width`` is half the car's width plus whatever margin the caller
    wants (the node passes ``car_width/2 + safety_margin``, the same swept
    width the TTC brake uses). Returns the free distance from the body's
    front edge, so 0.0 means an obstacle is already inside the band
    alongside the body.

    Sign convention REP-103: positive curvature turns left. Invalid beams
    are unknown, not obstacles, as everywhere else in this file.
    """
    ranges = np.asarray(clean, dtype=np.float64)
    validity = np.asarray(valid, dtype=bool)
    beam_angles = np.asarray(angles, dtype=np.float64)
    if not (ranges.shape == validity.shape == beam_angles.shape):
        raise ValueError('ranges, validity, and angles must have matching shapes')
    for name, value in (('curvature', curvature), ('half_width', half_width),
                        ('car_length', car_length), ('wheelbase', wheelbase),
                        ('laser_offset_x', laser_offset_x),
                        ('laser_offset_y', laser_offset_y),
                        ('max_length', max_length)):
        if not math.isfinite(value):
            raise ValueError(f'{name} must be finite')
    if half_width <= 0.0 or car_length <= 0.0 or wheelbase <= 0.0 or max_length <= 0.0:
        raise ValueError('half_width, car_length, wheelbase and max_length must be positive')

    # Body extent along base_link's x axis (same model as
    # vehicle_boundary_distances: centred halfway along the wheelbase).
    front = wheelbase / 2.0 + car_length / 2.0
    rear = wheelbase / 2.0 - car_length / 2.0

    x = ranges * np.cos(beam_angles) + laser_offset_x
    y = ranges * np.sin(beam_angles) + laser_offset_y
    usable = validity & (ranges > 0.0)

    if abs(curvature) < 1e-6:
        in_band = usable & (np.abs(y) <= half_width) & (x >= rear)
        along = x - front
    else:
        radius = 1.0 / curvature
        abs_radius = abs(radius)
        if abs_radius <= half_width:
            raise ValueError('turn radius must exceed half_width')
        rho = np.hypot(x, y - radius)
        inner = abs_radius - half_width
        outer = math.hypot(front, abs_radius + half_width)
        # Angle swept about the turn centre, in the direction of travel,
        # starting from the rear axle (which sits at angle -pi/2 from a
        # left-turn centre, +pi/2 from a right-turn one).
        start = -math.pi / 2.0 if radius > 0.0 else math.pi / 2.0
        theta = np.arctan2(y - radius, x)
        swept = np.mod(math.copysign(1.0, radius) * (theta - start), 2.0 * math.pi)
        # Points just behind the rear axle wrap to ~2*pi; anything beyond
        # the horizon is ignored anyway, so cap the sweep below a full turn.
        max_sweep = min(max_length + front, 1.9 * math.pi * abs_radius) / abs_radius
        in_band = usable & (rho >= inner) & (rho <= outer) & (swept <= max_sweep)
        along = swept * abs_radius - front

    along = np.where(in_band, along, np.inf)
    nearest = float(np.min(along)) if along.size else math.inf
    if nearest > max_length:
        return math.inf
    return max(0.0, nearest)


def tier1_arc_clearance(ranges: np.ndarray, range_min: float, angles: np.ndarray,
                        steerings, half_width: float, car_length: float,
                        wheelbase: float, laser_offset_x: float,
                        laser_offset_y: float, max_length: float) -> float:
    """Tier 1's arc check: the smallest swept_path_clearance over the arcs for
    each steering angle in ``steerings`` (radians), or ``inf`` if all are clear.

    The caller chooses the beams. pure_pursuit_node and the simulator pass the
    +/-body_clearance_fov_deg window that the footprint check uses, never the
    whole scan (see the note above swept_path_clearance). A beam is valid only
    if it is finite, positive and at least ``range_min``; NaN, +/-inf, zero,
    negative and sub-range_min readings are unknown and are never obstacles.

    ``steerings`` is the rack's last commanded angle (the slew basis) and this
    tick's steering before the emergency override. Taking the minimum over both
    covers the rack still being where it was (servo lag) and where it is being
    sent.

    The curvature is clipped to a radius of 1.01 * half_width, because
    swept_path_clearance raises for a tighter turn. No steering the car can take
    comes near that limit, so the clip only keeps a bad value from raising inside
    the control loop. A non-finite steering cannot be evaluated, so it reads as
    blocked (0.0), which makes the caller stop.
    """
    measured = np.asarray(ranges, dtype=np.float64)
    beam_angles = np.asarray(angles, dtype=np.float64)
    if measured.shape != beam_angles.shape:
        raise ValueError('ranges and angles must have matching shapes')
    if not all(math.isfinite(steering) for steering in steerings):
        return 0.0
    valid = np.isfinite(measured) & (measured > 0.0) & (measured >= range_min)
    clean = np.where(valid, measured, 0.0)
    max_curvature = 1.0 / (1.01 * half_width)
    best = math.inf
    for steering in steerings:
        curvature = math.tan(steering) / wheelbase
        curvature = min(max_curvature, max(-max_curvature, curvature))
        best = min(best, swept_path_clearance(
            clean, valid, beam_angles, curvature, half_width, car_length,
            wheelbase, laser_offset_x, laser_offset_y, max_length))
    return best
