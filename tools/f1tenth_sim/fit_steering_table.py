"""Fit and score an offline Car 2 command-response table, without writing to data/.

Requires mcap and mcap-ros2-support in addition to the simulator requirements.
Use wheel speed to identify steady periods, then measure forward speed by
matching LiDAR scans. The raw VESC gyro is in degrees per second.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.extend(map(str, (ROOT / 'tools', ROOT / 'tools/imu_calibration')))
from racer_imu.calibration import ImuCalibration
from racer_imu.frames import load_config
from run_analysis.grip_truth import steady_corners
from run_analysis.scan_motion import scan_points, icp
from sim_fidelity.calibration import FLOORS
from steering_table import fit, predict

FIT = ('135934', '142254')
RESERVED = ('140322', '141227')
FLOOR = 'sfu_course_2026_11_11_rb2'
SCALE = FLOORS[FLOOR].overrides['odom_speed_scale']
IMU_CONFIG = ROOT / 'src/racerbot_car/cars/rb2/imu/car.yaml'


def held(stream, grid, max_age):
    a = np.array(sorted(stream), dtype=float)
    i = np.searchsorted(a[:, 0], grid, side='right') - 1
    return np.where((i >= 0) & (grid - a[i, 0] <= max_age), a[i, 1], np.nan)


def samples(directory, run):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    cfg = load_config(IMU_CONFIG)
    vesc = cfg.imus['vesc']
    bias = ImuCalibration.load(cfg.resolve(vesc.extra['calib'])).eff_gyro_bias()
    topics = {k: [] for k in ('/odom', '/ackermann_cmd', '/sensors/imu/raw')}
    paths = sorted((directory / ('20261111-' + run) / 'bag').glob('*.mcap'))
    for path in paths:
        with path.open('rb') as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, ch, rec, m in reader.iter_decoded_messages(topics=topics.keys()):
                if ch.topic == '/odom':
                    value = m.twist.twist.linear.x / SCALE
                elif ch.topic == '/ackermann_cmd':
                    value = m.drive.steering_angle
                else:
                    g = m.angular_velocity
                    value = vesc.R[2] @ (np.radians([g.x, g.y, g.z]) - bias)
                topics[ch.topic].append((rec.log_time * 1e-9, value))
    if not all(topics.values()):
        raise ValueError(f'{run}: missing odometry, mux command or raw VESC gyro')
    t = np.arange(max(min(s)[0] for s in topics.values()),
                  min(max(s)[0] for s in topics.values()), .05)
    v = held(topics['/odom'], t, .1)
    delay = FLOORS[FLOOR].overrides['steering_delay_ticks'] / 40.0
    cmd = held(topics['/ackermann_cmd'], t - delay, .1)
    yaw = held(topics['/sensors/imu/raw'], t, .1)
    # Use the Nov 11 grip fixture's published steady-window criteria.
    ok = steady_corners(t, v, yaw, v * yaw, cmd, max_steering_change=.1)
    # No partial steady windows at the beginning or end of a drive.
    ok &= (t >= t[0] + .15) & (t <= t[-1] - .15)
    times = t[ok]
    scans = []
    for path in paths:
        with path.open('rb') as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, ch, rec, m in reader.iter_decoded_messages(topics=['/scan']):
                stamp = rec.log_time * 1e-9
                i = np.searchsorted(times, stamp - .1)
                if i < len(times) and times[i] <= stamp + .1:
                    angles = m.angle_min + np.arange(len(m.ranges)) * m.angle_increment
                    scans.append((stamp, scan_points(m.ranges, angles)))
    st = np.array([s[0] for s in scans])
    measured = []
    for j in np.flatnonzero(ok):
        left, right = np.searchsorted(st, [t[j] - .04, t[j] + .02])
        if right >= len(st) or left >= right:
            continue
        dt = st[right] - st[left]
        if not .035 <= dt <= .085:
            continue
        match = icp(scans[left][1], scans[right][1], (yaw[j]*dt, v[j]*dt, 0.0))
        if match is not None and match[3] <= .02 and match[1] / dt >= 1.5:
            measured.append((match[1] / dt, cmd[j], yaw[j]))
    return np.asarray(measured).reshape(-1, 3)


def score(rows, pooled, indexed):
    errors = []
    for v, cmd, yaw in rows:
        ideal = v * np.tan(cmd) / pooled['wheelbase']
        a, b = (predict(model, v, cmd) for model in (pooled, indexed))
        errors.append((abs(ideal - yaw), abs((ideal if a is None else a) - yaw),
                       abs((ideal if b is None else b) - yaw), a is not None, b is not None))
    if not errors:
        return dict(n=0)
    e = np.array(errors)
    common = (e[:, 3] > 0) & (e[:, 4] > 0)
    return dict(n=len(e), common=int(common.sum()), pooled_n=int(e[:, 3].sum()),
                indexed_n=int(e[:, 4].sum()), mae=e[:, :3].mean(axis=0).tolist(),
                common_mae=e[common, :3].mean(axis=0).tolist() if any(common) else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    training = {run: samples(args.runs, run) for run in FIT}
    train = np.vstack(list(training.values()))
    wheelbase = load_config(IMU_CONFIG).car.wheelbase
    pooled, indexed = [fit(train, wheelbase, indexed=i) for i in (False, True)]
    scores = {r: score(samples(args.runs, r), pooled, indexed) for r in RESERVED}
    # Same-support AND all-sample yaw MAE must improve on both whole drives.
    wins = all(s.get('common', 0) and s['common_mae'][2] < s['common_mae'][1]
               and s['mae'][2] < s['mae'][1] for s in scores.values())
    out = dict(indexed=indexed, pooled=pooled, indexed_qualified=bool(wins),
               train_samples={r: len(a) for r, a in training.items()},
               reserved=scores, wheel_over_ground=SCALE)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
