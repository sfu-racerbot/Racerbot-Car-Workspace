"""
metrics.py -- the numbers a recording is graded on, from plain arrays.

No ROS and no bag reading here (bag_stats.py does that), so every function can
be checked against a closed form. Times are bag receive times in seconds.

scan_change and moved_between came from tools/run_analysis/run_triage.py,
which imports them back from here, so the triage tool and the classifier judge
"carried between holds" the same way. Which code drove a run is algorithms.py.
"""

import math

import numpy as np

STILL_MPS = 0.02  # wheel speed of a parked car
MOVED_M = 0.15  # median change of the scan, parked, that means the car is somewhere else
GRAVITY = 9.80665  # m/s^2 per g


def scan_change(a, b):
    """Median change of the ranges both scans saw, in metres: about zero for the same place."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    both = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return float(np.median(np.abs(a[both] - b[both]))) if both.any() else float('nan')


def moved_between(scan_t, scans, odom, windows, still_mps=STILL_MPS, moved_m=MOVED_M):
    """For each gap between holds: whether the scan changed while the wheels never turned (carried)."""
    odom = np.asarray(odom, dtype=float).reshape(-1, 2)
    flags = []
    for (_, end), (start, _) in zip(windows[:-1], windows[1:]):
        before = int(np.searchsorted(scan_t, end, side='right')) - 1
        after = int(np.searchsorted(scan_t, start))
        if before < 0 or after >= len(scan_t):
            flags.append(False)
            continue
        between = odom[(odom[:, 0] > end + 0.5) & (odom[:, 0] < start)]
        # Carried needs the wheels seen still once the car has coasted to a stop. A release shorter than
        # that 0.5 s shows no such thing (rb1 20261111-153721: a 0.4 s LB blip at 2.7 m/s read as carried).
        wheels_still = len(between) > 0 and bool(np.all(np.abs(between[:, 1]) < still_mps))
        flags.append(wheels_still and scan_change(scans[before], scans[after]) > moved_m)
    return flags


def lb_windows(joy_t, held, min_duration=0.5):
    """[(press, release)] spans with LB held, from /joy times and 0/1 states.

    The same spans tools/run_analysis/runlog.py's Run.lb_windows() finds: a
    hold still open at the last /joy message ends there, and holds shorter
    than min_duration are dropped."""
    windows, start = [], None
    for t, h in zip(joy_t, held):
        if h and start is None:
            start = float(t)
        elif not h and start is not None:
            windows.append((start, float(t)))
            start = None
    if start is not None:
        windows.append((start, float(joy_t[-1])))
    return [w for w in windows if w[1] - w[0] >= min_duration]


def timing(t, start, end, coverage_gap_s):
    """(rate_hz, max_gap_s, coverage) of messages received at times t.

    rate is 1 / the median time between messages, so a few long pauses do not
    hide the rate the topic really runs at; max_gap is the longest pause.
    coverage is the share of [start, end] not inside a silence longer than
    coverage_gap_s, counting the silences before the first message and after
    the last. rate and max_gap are None below two messages."""
    t = np.asarray(t, dtype=float)
    rate = max_gap = None
    if len(t) >= 2:
        dt = np.diff(t)
        median = float(np.median(dt))
        rate = 1.0 / median if median > 0 else None
        max_gap = float(np.max(dt))
    if end is None or start is None or end <= start:
        return rate, max_gap, None
    if not len(t):
        return rate, max_gap, 0.0
    edges = np.concatenate(([start], np.clip(t, start, end), [end]))
    pauses = np.diff(edges)
    silent = float(np.sum(pauses[pauses > coverage_gap_s]))
    return rate, max_gap, min(1.0, max(0.0, 1.0 - silent / (end - start)))


def reversals(x, swing):
    """How many times x turned round after moving at least `swing` from its
    last turning point. Noise smaller than swing never counts.

    A sine of amplitude A >= swing over k whole periods turns 2k times."""
    x = [float(v) for v in x if math.isfinite(v)]
    if len(x) < 2:
        return 0
    lo = hi = x[0]
    direction, count = 0, 0
    for v in x[1:]:
        if direction == 0:
            lo, hi = min(lo, v), max(hi, v)
            if v - lo >= swing:
                direction, hi = 1, v
            elif hi - v >= swing:
                direction, lo = -1, v
        elif direction == 1:
            if v > hi:
                hi = v
            elif hi - v >= swing:
                count, direction, lo = count + 1, -1, v
        else:
            if v < lo:
                lo = v
            elif v - lo >= swing:
                count, direction, hi = count + 1, 1, v
    return count


def motion(t, v, moving_speed_mps, max_gap_s):
    """(distance_m, moving_s, max_speed_mps) from wheel speeds v at times t.

    Each step counts the speed at its end over its length, as run_triage's
    hold_stats does; a step longer than max_gap_s is not counted at all, so a
    gap in the odometry is not guessed across. Reversing counts as distance."""
    t, v = np.asarray(t, dtype=float), np.abs(np.asarray(v, dtype=float))
    if not len(t):
        return None, None, None
    top = float(np.max(v))
    if len(t) < 2:
        return 0.0, 0.0, top
    dt, speed = np.diff(t), v[1:]
    ok = dt <= max_gap_s
    distance = float(np.sum(speed[ok] * dt[ok]))
    moving = float(np.sum(dt[ok & (speed >= moving_speed_mps)]))
    return distance, moving, top


def speed_at(t_query, t, v):
    """|wheel speed| at each t_query by linear interpolation, NaN outside the odometry."""
    t_query = np.asarray(t_query, dtype=float)
    if len(t) < 2:
        return np.full(len(t_query), np.nan)
    out = np.interp(t_query, t, np.abs(np.asarray(v, dtype=float)))
    out[(t_query < t[0]) | (t_query > t[-1])] = np.nan
    return out


def steering_excitation(cmd_t, steering, odom_t, odom_v, moving_speed_mps, swing_rad):
    """(std_rad, reversals) of the commanded steering while the car moved.

    A car that never moved exercised no steering response: (0.0, 0). None
    only when there are no commands or no odometry to tell."""
    if not len(cmd_t) or len(odom_t) < 2:
        return None, None
    speed = speed_at(cmd_t, odom_t, odom_v)
    moving = np.asarray(steering, dtype=float)[speed >= moving_speed_mps]
    if len(moving) < 2:
        return 0.0, 0
    return float(np.std(moving)), reversals(moving, swing_rad)


def finite_fraction_while_fast(t, values, odom_t, odom_v, min_speed_mps):
    """Share of the values that are finite while the wheels turned faster than min_speed_mps."""
    if not len(t) or len(odom_t) < 2:
        return None
    fast = speed_at(t, odom_t, odom_v) >= min_speed_mps
    if not fast.any():
        return None
    return float(np.mean(np.isfinite(np.asarray(values, dtype=float)[fast])))


def yaw_rate(gyro, accel, axes):
    """Yaw rate (rad/s, about up) from gyro rows (n, 3) in rad/s.

    In car axes it is the z rate. In a board's or sensor's own axes, the
    gyro is projected on "up" as the accelerometer measures it, averaged over
    the run (a car's average acceleration is gravity's reaction, pointing up),
    so no mounting calibration is assumed. None if the average is under half
    a g, which no level car gives."""
    gyro = np.asarray(gyro, dtype=float).reshape(-1, 3)
    if axes == 'car':
        return gyro[:, 2]
    accel = np.asarray(accel, dtype=float).reshape(-1, 3)
    if not len(accel):
        return None
    up = np.mean(accel, axis=0)
    norm = float(np.linalg.norm(up))
    if not math.isfinite(norm) or norm < 0.5 * GRAVITY:
        return None
    return gyro @ (up / norm)


def step_lengths(t, max_gap_s):
    """Each sample's share of time: the step to the next sample, zero across a gap."""
    t = np.asarray(t, dtype=float)
    dt = np.diff(t, append=t[-1]) if len(t) else np.zeros(0)
    dt[dt > max_gap_s] = 0.0
    return dt


def lateral_acceleration(imu_t, yaw, odom_t, odom_v, moving_speed_mps, cornering_mps2, max_gap_s):
    """(p95_mps2, cornering_s): |v x yaw rate| while moving, and how long it was over cornering_mps2.

    A car that never moved had no lateral acceleration: (0.0, 0.0). None only
    when there is no yaw rate or no odometry to tell."""
    if yaw is None or not len(imu_t) or len(odom_t) < 2:
        return None, None
    speed = speed_at(imu_t, odom_t, odom_v)
    moving = speed >= moving_speed_mps
    if not moving.any():
        return 0.0, 0.0
    lateral = speed * np.abs(np.asarray(yaw, dtype=float))
    dt = step_lengths(imu_t, max_gap_s)
    cornering = float(np.sum(dt[moving & (lateral >= cornering_mps2)]))
    return float(np.percentile(lateral[moving], 95)), cornering


def measured_lateral_p95(imu_t, accel_y, odom_t, odom_v, moving_speed_mps):
    """p95 of |measured lateral acceleration| while moving, from a car-axis IMU."""
    if not len(imu_t) or len(odom_t) < 2:
        return None
    moving = speed_at(imu_t, odom_t, odom_v) >= moving_speed_mps
    if not moving.any():
        return None
    return float(np.percentile(np.abs(np.asarray(accel_y, dtype=float)[moving]), 95))


def autonomous(odom_t, odom_v, windows, cmd_t, match_s, moving_speed_mps, max_gap_s):
    """(seconds, metres) the car moved with LB held and a driving command
    received within match_s: the stretch an algorithm was driving."""
    odom_t = np.asarray(odom_t, dtype=float)
    if len(odom_t) < 2:
        return 0.0, 0.0
    dt, end = np.diff(odom_t), odom_t[1:]
    speed = np.abs(np.asarray(odom_v, dtype=float))[1:]
    held = np.zeros(len(end), dtype=bool)
    for start, stop in windows:
        held |= (end >= start) & (end <= stop)
    cmd_t = np.asarray(cmd_t, dtype=float)
    last = np.searchsorted(cmd_t, end, side='right') - 1
    recent = (last >= 0) & (end - cmd_t[np.maximum(last, 0)] <= match_s) if len(cmd_t) else np.zeros(len(end), bool)
    driven = (dt <= max_gap_s) & (speed >= moving_speed_mps) & held & recent
    return float(np.sum(dt[driven])), float(np.sum(speed[driven] * dt[driven]))
