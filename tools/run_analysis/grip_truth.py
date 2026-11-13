#!/usr/bin/env python3
"""How much lateral acceleration the car sustains, from two independent signals.

    python3 tools/run_analysis/grip_truth.py data/rb2/racerbot_runs/20261111-152132 \\
        --imu-topic /imu/vesc/data --json /tmp/grip_rb2_vesc.json

Asked is true speed squared times the commanded curvature (what the steering would
deliver with perfect grip). Delivered is measured twice: by the accelerometer
(linear_acceleration.y, bias-corrected and averaged over 0.2 s) and by true speed
times the gyro. In a slide the gyro keeps reading yaw while the path straightens, so
speed times gyro overstates grip and the accelerometer does not; where the two agree
within 15% the grip is usable. Only steady cornering counts: steering and speed
constant over 0.3 s. Needs runs located with loc.py, and a bag with the chosen IMU
topic (our own launches record /imu/data and /imu/vesc/data; team A's launches record
only /sensors/imu/raw, which this tool does not convert). The math is importable
without ROS; only main() reads a bag.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

WHEELBASE = 0.36          # m, docs/hardware-reference.md
STRAIGHT_YAW = 0.05       # rad/s, below this the car is on a straight
BINS = ((0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4),
        (4, 5), (5, 6), (6, 8), (8, 12), (12, 30))
AVERAGE_SEC = 0.2         # accelerometer averaging window
AGREE = 0.15              # accelerometer and gyro agree within this fraction of the gyro value
MIN_BIN_SAMPLES = 50


def asked_lateral_accel(speed, steering, wheelbase: float = WHEELBASE) -> np.ndarray:
    """Kinematic lateral acceleration the steering asks for, m/s^2: v^2 * tan(delta) / L."""
    speed, steering = np.asarray(speed, dtype=float), np.asarray(steering, dtype=float)
    return speed ** 2 * np.abs(np.tan(steering)) / wheelbase


def steady_corners(t, speed, yaw_rate, lateral_accel, steering, *, min_speed: float = 1.5,
                   max_steering_change: float = 0.03, window: float = 0.3,
                   max_long_accel: float = 0.5) -> np.ndarray:
    """Boolean mask of steady-cornering samples.

    A sample counts when, over the ``window`` seconds centred on it, the steering moves by
    at most ``max_steering_change`` rad (max minus min) and the speed changes at no more than
    ``max_long_accel`` m/s^2, and the car is above ``min_speed`` m/s and turning
    (``|yaw_rate|`` at least 0.05 rad/s). Non-finite inputs are never steady.
    """
    t = np.asarray(t, dtype=float)
    speed = np.asarray(speed, dtype=float)
    yaw = np.asarray(yaw_rate, dtype=float)
    lat = np.asarray(lateral_accel, dtype=float)
    steer = np.asarray(steering, dtype=float)
    n = len(t)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    finite = np.isfinite(t) & np.isfinite(speed) & np.isfinite(yaw) & np.isfinite(lat) & np.isfinite(steer)
    candidate = finite & (speed >= min_speed) & (np.abs(yaw) >= STRAIGHT_YAW)
    lo = np.searchsorted(t, t - window / 2.0, side="left")
    hi = np.searchsorted(t, t + window / 2.0, side="right") - 1
    for i in np.nonzero(candidate)[0]:
        a, b = lo[i], hi[i]
        if b <= a:
            continue
        span = t[b] - t[a]
        if span <= 0.0 or not finite[a:b + 1].all():
            continue
        if np.ptp(steer[a:b + 1]) > max_steering_change:
            continue
        if abs(speed[b] - speed[a]) / span > max_long_accel:
            continue
        out[i] = True
    return out


def remove_straight_bias(speed, yaw_rate, lateral_accel, min_speed: float = 1.5):
    """(corrected, bias, count): the accelerometer's lateral reading minus its median over
    straight samples (``|yaw_rate| < 0.05`` rad/s, above ``min_speed``), which carries its
    bias and gravity leak. No straight samples: nothing is removed, bias is NaN, count 0."""
    speed = np.asarray(speed, dtype=float)
    yaw = np.asarray(yaw_rate, dtype=float)
    lat = np.asarray(lateral_accel, dtype=float)
    straight = np.isfinite(speed) & np.isfinite(yaw) & np.isfinite(lat) & (
        speed > min_speed) & (np.abs(yaw) < STRAIGHT_YAW)
    count = int(straight.sum())
    if count == 0:
        return lat.copy(), math.nan, 0
    bias = float(np.median(lat[straight]))
    return lat - bias, bias, count


def curve(asked, delivered, bins=BINS, min_samples: int = 1) -> list:
    """One dict per bin of asked with at least ``min_samples`` finite pairs: low, high, n,
    asked (median), median and p90 of delivered."""
    asked = np.asarray(asked, dtype=float)
    delivered = np.asarray(delivered, dtype=float)
    keep = np.isfinite(asked) & np.isfinite(delivered)
    asked, delivered = asked[keep], delivered[keep]
    rows = []
    for low, high in bins:
        chosen = (asked >= low) & (asked < high)
        if chosen.sum() >= max(1, min_samples):
            rows.append(dict(low=float(low), high=float(high), n=int(chosen.sum()),
                             asked=float(np.median(asked[chosen])),
                             median=float(np.median(delivered[chosen])),
                             p90=float(np.percentile(delivered[chosen], 90))))
    return rows


def sustained_limit(acc_rows, gyro_rows, min_samples: int = MIN_BIN_SAMPLES,
                    agree: float = AGREE) -> float:
    """The accelerometer's median delivered in the highest bin that has ``min_samples`` or more
    samples and where accelerometer and gyro agree within ``agree`` of the gyro value; NaN if
    no bin qualifies."""
    gyro = {(r["low"], r["high"]): r for r in gyro_rows}
    best = None
    for row in acc_rows:
        other = gyro.get((row["low"], row["high"]))
        if other is None or min(row["n"], other["n"]) < min_samples:
            continue
        if abs(row["median"] - other["median"]) <= agree * other["median"] + 1e-9:
            if best is None or row["low"] > best["low"]:
                best = row
    return math.nan if best is None else float(best["median"])


def analyse(t, speed, yaw_rate, lateral_accel, steering, wheelbase: float = WHEELBASE,
            bins=BINS, min_bin_samples: int = MIN_BIN_SAMPLES) -> dict:
    """Curves from one set of aligned samples (accelerometer already bias-corrected)."""
    mask = steady_corners(t, speed, yaw_rate, lateral_accel, steering)
    speed = np.asarray(speed, dtype=float)
    asked = asked_lateral_accel(speed, steering, wheelbase)[mask]
    acc = np.abs(np.asarray(lateral_accel, dtype=float))[mask]
    gyro = np.abs(speed * np.asarray(yaw_rate, dtype=float))[mask]
    return _tables(asked, acc, gyro, bins, min_bin_samples)


def _tables(asked, acc, gyro, bins, min_bin_samples) -> dict:
    acc_rows, gyro_rows = curve(asked, acc, bins), curve(asked, gyro, bins)
    gyro_by = {(r["low"], r["high"]): r for r in gyro_rows}
    rows = []
    for r in acc_rows:
        g = gyro_by[(r["low"], r["high"])]
        rows.append(dict(low=r["low"], high=r["high"], n=r["n"], asked=r["asked"],
                         acc_median=r["median"], acc_p90=r["p90"],
                         gyro_median=g["median"], gyro_p90=g["p90"],
                         ratio=r["median"] / g["median"] if g["median"] > 0 else math.nan))
    return dict(rows=rows, sustained_limit=sustained_limit(acc_rows, gyro_rows, min_bin_samples),
                samples=int(len(asked)))


def moving_average(t, values, window: float = AVERAGE_SEC) -> np.ndarray:
    """Mean of ``values`` over the trailing-centred ``window`` seconds, by cumulative sums."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(values, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(v)])
    lo = np.searchsorted(t, t - window / 2.0, side="left")
    hi = np.searchsorted(t, t + window / 2.0, side="right")
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def _imu_lateral(run, topic):
    from run_analysis.bagread import messages
    ts, ys = [], []
    for _topic, t, msg in messages(run, [topic]):
        ts.append(t)
        ys.append(msg.linear_acceleration.y)
    return np.asarray(ts), np.asarray(ys)


def per_run_samples(run, loc_dir, topic):
    """(speed, yaw, lateral, steering, t, bias, straight count) for one located run."""
    from run_analysis.loc import COL, load_track
    from run_analysis.sectors import MAX_TRUE_SPEED, true_speed
    a = load_track(run, loc_dir)
    ti, yi = _imu_lateral(run, topic)
    if len(ti) < 100:
        raise SystemExit(f"{run}: no usable {topic} in the bag ({len(ti)} messages)")
    t = a[:, COL["t"]]
    v = true_speed(t, a[:, COL["x"]], a[:, COL["y"]])
    lat = np.interp(t, ti, moving_average(ti, yi))
    keep = (a[:, COL["lb"]] > 0) & (a[:, COL["h20"]] > 0.8) & (v > 0.0) & (v < MAX_TRUE_SPEED)
    lat, bias, count = remove_straight_bias(v[keep], a[keep, COL["gyro_z"]], lat[keep])
    return (v[keep], a[keep, COL["gyro_z"]], lat, a[keep, COL["steer_cmd"]], t[keep], bias, count)


def main(argv=None) -> int:
    from run_analysis.loc import DEFAULT_OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories already located with loc.py")
    ap.add_argument("--loc", default=str(DEFAULT_OUT))
    ap.add_argument("--imu-topic", default="/imu/vesc/data")
    ap.add_argument("--car", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    asked, acc, gyro = [], [], []
    for run in args.runs:
        v, yaw, lat, steer, t, bias, count = per_run_samples(run, args.loc, args.imu_topic)
        print(f"{Path(run).name}: removed bias {bias:+.3f} m/s^2 from {count} straight samples")
        mask = steady_corners(t, v, yaw, lat, steer)
        asked.append(asked_lateral_accel(v, steer)[mask])
        acc.append(np.abs(lat)[mask])
        gyro.append(np.abs(v * yaw)[mask])
    if not asked:
        print("no located runs", file=sys.stderr)
        return 1
    asked, acc, gyro = (np.concatenate(x) for x in (asked, acc, gyro))
    result = _tables(asked, acc, gyro, BINS, MIN_BIN_SAMPLES)
    print(f"{result['samples']} steady-cornering samples, topic {args.imu_topic}")
    print(" asked   n   accel med  p90   gyro med  p90   ratio")
    for r in result["rows"]:
        print(f"{r['low']:4.1f}-{r['high']:4.1f} {r['n']:6d}  {r['acc_median']:7.2f} {r['acc_p90']:5.2f}"
              f"  {r['gyro_median']:7.2f} {r['gyro_p90']:5.2f}  {r['ratio']:5.2f}")
    print(f"sustained_limit {result['sustained_limit']:.2f} m/s^2")
    if args.json:
        car = args.car or next((p for p in Path(args.runs[0]).resolve().parts if p in ("rb1", "rb2")), "unknown")
        Path(args.json).write_text(json.dumps(dict(
            car=car, runs=[Path(r).name for r in args.runs], imu_topic=args.imu_topic,
            curve=[dict(asked=r["asked"], delivered=r["acc_median"], n=r["n"]) for r in result["rows"]],
            sustained_limit=result["sustained_limit"] if math.isfinite(result["sustained_limit"]) else None),
            indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
