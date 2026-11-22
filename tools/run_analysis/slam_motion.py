#!/usr/bin/env python3
"""Differentiate loc.py tracks into rear-axle and CG motion, without ROS.

loc.scan_points adds LASER_X=0.26 before matching, so loc's pose is the
rear-axle base_link, NOT the laser (docs/hardware-reference.md, Physical
dimensions). CG lateral velocity is v_rear + l_r * yaw_rate. Geometry comes
from sim_fidelity.calibration.calibration_for_car: wheelbase 0.36 m and
l_r=0.18 m on both cars. Its cog_front_fraction=0.5 is an UNMEASURED midpoint
assumption, not a per-car CG measurement; see CarCalibration's mass properties.

Trust requires h10 >= 0.75. Position/yaw steps above 0.30 m/rad reject both
ends plus 0.20 s; gaps above 0.125 s split segments. Each segment is linearly
resampled at 20 Hz, then a cubic Savitzky-Golay filter (7 points, 0.30 s span)
smooths/differentiates x, y and unwrapped yaw. Half-window edges are discarded;
no interpolation, unwrapping or smoothing crosses a rejected sample. Results
retain the original timestamps, with NaN motion outside the final trust mask.

Speed is the rear-axle ground-speed magnitude. a_y = speed * yaw_rate is a
quasi-steady turning-acceleration proxy, not a differentiated accelerometer
measurement. Likewise curvature = yaw_rate / speed; when slip angle changes,
this differs from the curvature of the velocity-heading path. Slip is only
reported above 0.5 m/s. No LB gate is imposed; a trusted parked pose is valid,
but excluded from moving statistics, as is reverse (body longitudinal <= 0).
Wheel ratios use |wheel speed| / speed.

CAVEAT: loc.py uses wheel speed and raw VESC gyro-z prediction before map
correction. These estimates are not fully independent sensor ground truth;
h10 alone cannot rule out a wrong match in repetitive geometry. gyro_z is the
loc column (raw z converted from deg/s), not mount-corrected or lag-aligned.
The gyro fit is gyro_z = scale * SLAM_yaw_rate + bias; yaw RMS is the UNCORRECTED
gyro-minus-SLAM error. Steering bins compare command, not measured rack angle,
and therefore mix linkage, actuator and tyre effects, not tyre understeer alone.

    python3 tools/run_analysis/slam_motion.py ~/.cache/racerbot/loc/RUN.npz \\
        --car rb2 --json ~/.cache/racerbot/loc/motion.json --samples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/f1tenth_sim')]
from run_analysis.loc import COL, COLUMNS  # noqa: E402
from sim_fidelity.calibration import calibration_for_car  # noqa: E402

DT = 0.05
SMOOTH_S = 0.30
JUMP_MARGIN_S = 0.20
MOVING_MPS = 0.5
ACCEL_EDGES = (0.0, 1.0, 2.0, 3.0, 4.0, np.inf)
STEER_EDGES = (0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, np.inf)
MOTION_KEYS = ('x', 'y', 'yaw', 'world_vx', 'world_vy', 'ground_speed',
               'longitudinal', 'lateral_rear', 'lateral_cg', 'yaw_rate',
               'slip_rear', 'slip_cg', 'lateral_accel', 'curvature')


def _segments(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2])


def extract_motion(track, lr: float) -> dict:
    """Motion arrays on the original clock; angles in radians, velocities in m/s."""
    a = np.asarray(track, dtype=float)
    if a.ndim != 2 or a.shape[1] != len(COLUMNS):
        raise ValueError(f'expected an (N, {len(COLUMNS)}) loc track')
    t = a[:, COL['t']]
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('track timestamps must be finite and strictly increasing')
    if not np.isfinite(lr) or lr < 0:
        raise ValueError('l_r must be finite and nonnegative')
    out = {key: np.full(len(t), np.nan) for key in MOTION_KEYS}
    out.update({key: a[:, COL[key]].copy() for key in
                ('t', 'h10', 'h20', 'wheel_v', 'gyro_z', 'steer_cmd', 'speed_cmd', 'lb')})
    out['trusted'] = np.zeros(len(t), dtype=bool)
    out['segment'] = np.full(len(t), -1, dtype=int)
    pose = a[:, [COL['x'], COL['y'], COL['yaw']]]
    good = np.isfinite(pose).all(axis=1) & np.isfinite(out['h10']) & (out['h10'] >= 0.75)
    steps = np.diff(pose, axis=0)
    yaw_step = np.arctan2(np.sin(steps[:, 2]), np.cos(steps[:, 2]))
    jumps = (np.hypot(steps[:, 0], steps[:, 1]) > 0.30) | (np.abs(yaw_step) > 0.30)
    for i in np.flatnonzero(jumps):
        good[(t >= t[i] - JUMP_MARGIN_S) & (t <= t[i + 1] + JUMP_MARGIN_S)] = False
    breaks = np.r_[True, np.diff(t) > 0.125]
    window = int(round(SMOOTH_S / DT)) | 1
    half = window // 2
    segment_id = 0
    for start, end in _segments(good):
        cuts = [start, *list(np.flatnonzero(breaks[start + 1:end]) + start + 1), end]
        for left, right in zip(cuts[:-1], cuts[1:]):
            if right - left < window:
                continue
            st = t[left:right] - t[left]
            grid = np.arange(int(np.floor((st[-1] + 1e-9) / DT)) + 1) * DT
            if len(grid) < window:
                continue
            p = pose[left:right].copy()
            p[:, 2] = np.unwrap(p[:, 2])
            uniform = np.column_stack([np.interp(grid, st, p[:, j]) for j in range(3)])
            smooth = savgol_filter(uniform, window, 3, axis=0)
            velocity = savgol_filter(uniform, window, 3, deriv=1, delta=DT, axis=0)
            keep = (st >= grid[half]) & (st <= grid[-half - 1])
            indices = np.arange(left, right)[keep]
            if not len(indices):
                continue
            values = np.column_stack([np.interp(st[keep], grid, smooth[:, j]) for j in range(3)])
            rates = np.column_stack([np.interp(st[keep], grid, velocity[:, j]) for j in range(3)])
            x, y, yaw = values.T
            dx, dy, r = rates.T
            u = np.cos(yaw) * dx + np.sin(yaw) * dy
            v = -np.sin(yaw) * dx + np.cos(yaw) * dy
            speed = np.hypot(dx, dy)
            vcg = v + lr * r
            moving = speed >= MOVING_MPS
            beta = np.where(moving, np.arctan2(v, u), np.nan)
            beta_cg = np.where(moving, np.arctan2(vcg, u), np.nan)
            curvature = np.divide(r, speed, out=np.full(len(r), np.nan), where=moving)
            columns = (x, y, yaw, dx, dy, speed, u, v, vcg, r, beta, beta_cg, speed * r, curvature)
            for key, value in zip(MOTION_KEYS, columns):
                out[key][indices] = value
            out['trusted'][indices] = True
            out['segment'][indices] = segment_id
            segment_id += 1
    out['steady'] = steady_mask(out)
    return out


def steady_mask(motion, hold_s: float = 0.30) -> np.ndarray:
    """Within one hold: steer span <=.02 rad, speed <=.3 m/s, yaw rate <=.2 rad/s."""
    t, steer, speed, rate = (motion[k] for k in ('t', 'steer_cmd', 'ground_speed', 'yaw_rate'))
    steady = np.zeros(len(t), dtype=bool)
    for segment in np.unique(motion['segment'][motion['trusted']]):
        ids = np.flatnonzero(motion['segment'] == segment)
        st = t[ids] - t[ids[0]]
        for j, stamp in enumerate(st):
            if stamp < hold_s / 2 or stamp + hold_s / 2 > st[-1]:
                continue
            left = max(0, np.searchsorted(st, stamp - hold_s / 2, side='right') - 1)
            right = min(len(ids), np.searchsorted(st, stamp + hold_s / 2) + 1)
            chosen = ids[left:right]
            if (np.isfinite(steer[chosen]).all() and np.isfinite(speed[chosen]).all()
                    and np.isfinite(rate[chosen]).all() and np.ptp(rate[chosen]) <= 0.20
                    and np.ptp(steer[chosen]) <= 0.02 and np.ptp(speed[chosen]) <= 0.30):
                steady[ids[j]] = True
    return steady


def _distribution(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0, median=None, iqr=None)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return dict(count=len(values), median=float(median), iqr=float(q3 - q1))


def _rms(values):
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else None


def summarize(motion, wheelbase: float) -> dict:
    """Sample-weighted summaries; missing/unidentifiable statistics are JSON null."""
    trusted = motion['trusted']
    speed, rate = motion['ground_speed'], motion['yaw_rate']
    moving = trusted & (speed >= MOVING_MPS) & (motion['longitudinal'] > 0)
    wheel = moving & np.isfinite(motion['wheel_v'])
    wheel_speed = np.abs(motion['wheel_v'][wheel])
    wheel_stats = _distribution(wheel_speed / speed[wheel])
    wheel_stats['speed_rms_mps'] = _rms(wheel_speed - speed[wheel])
    gyro = moving & np.isfinite(motion['gyro_z'])
    reference, measured = rate[gyro], motion['gyro_z'][gyro]
    gyro_stats = dict(count=len(reference), scale=None, bias_rad_s=None,
                      raw_rms_rad_s=_rms(measured - reference), fitted_rms_rad_s=None)
    if len(reference) >= 3 and np.std(reference) >= 0.1:
        scale, bias = np.linalg.lstsq(np.c_[reference, np.ones(len(reference))], measured, rcond=None)[0]
        gyro_stats.update(scale=float(scale), bias_rad_s=float(bias),
                          fitted_rms_rad_s=_rms(measured - scale * reference - bias))
    slips = []
    for low, high in zip(ACCEL_EDGES[:-1], ACCEL_EDGES[1:]):
        mask = moving & (np.abs(motion['lateral_accel']) >= low) & (np.abs(motion['lateral_accel']) < high)
        row = dict(low_mps2=low, high_mps2=high if np.isfinite(high) else None)
        for name in ('rear', 'cg'):
            beta = np.degrees(motion[f'slip_{name}'][mask])
            row[name + '_deg'] = _distribution(beta)
            row[name + '_abs_deg'] = _distribution(np.abs(beta))
        slips.append(row)
    steering = []
    for low, high in zip(STEER_EDGES[:-1], STEER_EDGES[1:]):
        delta = motion['steer_cmd']
        mask = moving & motion['steady'] & (np.abs(delta) >= low) & (np.abs(delta) < high)
        command = np.tan(np.abs(delta[mask])) / wheelbase
        achieved = motion['curvature'][mask] * np.sign(delta[mask])
        steering.append(dict(low_rad=low, high_rad=high if np.isfinite(high) else None,
                             count=int(mask.sum()), ground_speed_mps=_distribution(speed[mask]),
                             commanded_curvature_per_m=_distribution(command),
                             achieved_curvature_per_m=_distribution(achieved),
                             achieved_over_commanded=_distribution(achieved / command)))
    return dict(samples=len(trusted), trusted_samples=int(trusted.sum()),
                trusted_fraction=float(trusted.mean()) if len(trusted) else 0.0,
                moving_samples=int(moving.sum()), wheel_over_ground=wheel_stats,
                gyro_vs_slam=gyro_stats, sideslip_by_abs_lateral_accel=slips,
                steady_curvature_by_abs_command=steering)


def pool_motions(motions):
    """Concatenate already-derived runs; never smooth or form holds across runs."""
    if not motions:
        raise ValueError('at least one motion track is required')
    pooled = {key: np.concatenate([m[key] for m in motions]) for key in motions[0]}
    segments, offset = [], 0
    for motion in motions:
        ids = motion['segment'].copy()
        ids[ids >= 0] += offset
        if np.any(ids >= 0):
            offset = int(ids.max()) + 1
        segments.append(ids)
    pooled['segment'] = np.concatenate(segments)
    return pooled


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tracks', nargs='+', type=Path)
    ap.add_argument('--car', choices=('rb1', 'rb2'), required=True)
    ap.add_argument('--json', required=True, type=Path)
    ap.add_argument('--samples', nargs='?', const=str(Path.home() / '.cache/racerbot/slam_motion'),
                    type=Path, help='save per-sample NPZs below ~/.cache (optional directory)')
    args = ap.parse_args(argv)
    if args.samples is not None:
        cache = (Path.home() / '.cache').resolve()
        args.samples = args.samples.expanduser().resolve()
        if not args.samples.is_relative_to(cache):
            ap.error('--samples must be below ~/.cache')
        args.samples.mkdir(parents=True, exist_ok=True)
    cal = calibration_for_car(args.car)
    results, motions = [], []
    for path in args.tracks:
        with np.load(path.expanduser(), allow_pickle=False) as saved:
            motion = extract_motion(saved['a'], cal.lr)
        summary = summarize(motion, cal.wheelbase)
        results.append(dict(run=path.stem, car=args.car, **summary))
        motions.append(motion)
        if args.samples is not None:
            np.savez_compressed(args.samples / f'{args.car}_{path.stem}.npz', **motion)
        print(f'{args.car} {path.stem}: {summary["trusted_samples"]}/{summary["samples"]} trusted', flush=True)
    report = dict(car=args.car, geometry=dict(wheelbase_m=cal.wheelbase, lr_m=cal.lr,
                  cg_source='CarCalibration.cog_front_fraction: unmeasured midpoint assumption'),
                  method=__doc__, runs=results,
                  pooled_sample_weighted=summarize(pool_motions(motions), cal.wheelbase))
    args.json.expanduser().write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
