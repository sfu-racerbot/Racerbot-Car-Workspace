"""Conservative longitudinal envelope, independent of ROS and command slew."""
import math

import numpy as np


def visible_arc(xy, seg_len, index, pose, ranges, angle_min, angle_step,
                laser_x, laser_y, half_width, horizon=5.0):
    """Lower bound on empty line corridor observed by this scan, in metres.

    A lost car is never advanced by velocity prediction. Instead, visibility
    can prove that its old location is now empty. Check the entire angular
    interval across each lateral slice (not only the centre ray), with 0.05 m
    radial reserve for range error. The 0.65 m longitudinal reserve covers
    station spacing (0.15 m on the shipped lines). Invalid beams
    are unknown, hence blocked. The caller subtracts ego's front extent.
    """
    if (len(xy) < 2 or len(ranges) < 2 or not math.isfinite(angle_step)
            or angle_step <= 0 or not all(math.isfinite(v) for v in pose)):
        return 0.0
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    origin = np.array([x + c * laser_x - s * laser_y,
                       y + s * laser_x + c * laser_y])
    distance = 0.0
    cleared = 0.0
    for step in range(len(xy)):
        i = (index + step) % len(xy)
        distance += float(seg_len[(i - 1) % len(xy)]) if step else 0.0
        if distance > horizon:
            break
        if distance < 0.7:
            continue  # ego's own current body, not forward empty road
        tangent = xy[(i + 1) % len(xy)] - xy[i]
        norm = float(np.linalg.norm(tangent))
        if not math.isfinite(norm) or norm <= 0:
            return cleared
        normal = np.array([-tangent[1], tangent[0]]) / norm
        ends = np.array([xy[i] - half_width * normal, xy[i] + half_width * normal]) - origin
        angles = (np.arctan2(ends[:, 1], ends[:, 0]) - yaw + math.pi) % (2 * math.pi) - math.pi
        lo = math.floor((float(min(angles)) - angle_min) / angle_step)
        hi = math.ceil((float(max(angles)) - angle_min) / angle_step)
        if lo < 0 or hi >= len(ranges):
            return cleared
        window = np.asarray(ranges[lo:hi + 1])
        ray_angles = yaw + angle_min + np.arange(lo, hi + 1) * angle_step
        rays = np.column_stack((np.cos(ray_angles), np.sin(ray_angles)))
        tangent /= norm
        denominator = rays @ tangent
        if np.any(denominator <= 0):
            return cleared
        required = float(np.dot(xy[i] - origin, tangent)) / denominator + 0.05
        if not np.all(np.isfinite(window) & (window >= required)):
            return cleared
        cleared = distance
    return cleared


def project_arc(xy, cumulative, point, index):
    """Continuous local line projection; avoids 0.15 m waypoint speed impulses."""
    if len(xy) < 2 or not np.all(np.isfinite(point)):
        return math.nan
    best = (math.inf, 0.0)
    for i in ((index - 1) % len(xy), index):
        delta = xy[(i + 1) % len(xy)] - xy[i]
        length = float(np.linalg.norm(delta))
        fraction = float(np.clip(np.dot(np.asarray(point) - xy[i], delta)
                                 / max(length * length, 1e-12), 0, 1))
        error = float(np.linalg.norm(np.asarray(point) - xy[i] - fraction * delta))
        if error < best[0]:
            best = error, float(cumulative[i]) + fraction * length
    total = float(cumulative[-1]) + float(np.linalg.norm(xy[0] - xy[-1]))
    return best[1] % total if total > 0 else math.nan


def speed_ceiling(gap, opponent_speed, age, *, brake=3.0,
                  delay=0.45, reserve=0.65, speed_uncertainty=1.0, stop_speed=0.30):
    """Bound ego's delayed stop before the opponent's earliest stop.

    Gap is visible rear surface to ego front, along the line. The reserve
    covers centroid/pose error, curvature and tracking. Opponent may stop
    instantaneously: deliberately give NO stopping-distance credit to its
    speed. Its tracked speed only increases the uncertainty allowance during
    lost sightings; it never creates free space. Thus hard braking cannot
    invalidate an optimistic opponent-velocity prediction.

    Bound the speed target through a 0.45 s response allowance (0.10
    transport + 1/7 speed-loop lag + sensing/tracking budget), then brake
    at 3 m/s². Solve v*t+v²/(2*b) <= gap-reserve. The command cap must
    bypass a falling-target slew floor. This models delay as coasting at
    the capped speed, rather than treating the 8 m/s² command slew as
    physical braking. Acceleration is bounded by the measured plant's
    4 m/s²; it does not increase this speed target.
    """
    values = (gap, opponent_speed, age, brake, delay, reserve, speed_uncertainty, stop_speed)
    if not all(math.isfinite(v) for v in values):
        return 0.0
    if min(age, delay, reserve, speed_uncertainty, stop_speed) < 0 or brake <= 0:
        return 0.0
    available = gap - reserve - age * (abs(opponent_speed) + speed_uncertainty)
    linear = delay * brake
    cap = max(0.0, math.sqrt(max(0.0, linear ** 2 + 2 * brake * available)) - linear)
    # A first-order speed loop asymptotically following tiny falling targets
    # eats the standstill reserve. Command a real zero before that tail.
    return 0.0 if cap < stop_speed else cap
