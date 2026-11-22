#!/usr/bin/env python3
"""Find the stretches of a recording where the car was really racing.

    python3 tools/run_analysis/race_segments.py data/<car>/racerbot_runs/<time> [more runs]
    python3 tools/run_analysis/race_segments.py <runs> --yaml out.yaml

A recording made at a race or a test day holds much more than racing: the car parked
between heats, short moves to the start line, minutes stuck against a wall, and people
picking it up and carrying it. Anything that fits a model to the car's motion (the
simulator calibration, grip and lap studies) must use only the racing.

The recording is cut into half-second steps. Each step gets one label from the car's
own sensors, never from where a map says it was:

  racing      the wheels turn at walking pace or faster (0.5 m/s), the LiDAR view
              changes, and LB is held (the driving code is in charge)
  coasting    the same, just after LB was let go
  spinning    the wheels turn but the LiDAR view does not change: the car is held in
              the air or pinned against something
  handled     the wheels are still but the LiDAR view changes, or the car tilts
              (the VESC IMU's roll/pitch rate): someone is moving it by hand
  parked      nothing moves

Racing steps are joined into stretches. A pause of up to 3 s inside a stretch is kept
(traffic, a brief stop); a longer pause, or any spinning or handled step, ends it.
A stretch is a race stretch when it lasts at least 10 s, covers at least one course
length by the wheels (28.17 m on the 2026-11-11 UVic course) and averages 1.5 m/s.
Shorter stretches are short moves (repositioning, launch checks).

Reads the bag with bagread (rosbags without ROS). Writes nothing unless --yaml is given.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STEP_S = 0.5           # length of one labelled step
MOVING_MPS = 0.5       # wheel speed of a car that is driving, not creeping
STILL_MPS = 0.2        # wheel speed below which the wheels are stopped
SCAN_STILL_MPS = 0.05  # median LiDAR range change per second of a scene that is not moving
SCAN_MOVED_MPS = 0.10  # median LiDAR range change per second that means the car moved
TILT_DPS = 8.0         # VESC roll/pitch rate of a car being handled; parked noise is under 2
PAUSE_S = 3.0          # longest pause kept inside one stretch
RACE_MIN_S = 10.0
RACE_MIN_MPS = 1.5
COURSE_M = 28.17       # 2026-11-11 UVic course line (data/rb2/racerbot_auto/20261111-151026-rebuilt)

LABELS = ('racing', 'coasting', 'spinning', 'handled', 'parked')


def step_labels(speed, scan_change, tilt, lb):
    """One label per step from wheel speed (m/s), LiDAR change (m/s), roll/pitch rate (deg/s) and LB."""
    out = []
    for v, c, g, held in zip(np.abs(speed), scan_change, tilt, lb):
        c = 0.0 if not np.isfinite(c) else c
        g = 0.0 if not np.isfinite(g) else g
        if not np.isfinite(v):
            out.append('parked')
        elif v >= MOVING_MPS and c < SCAN_STILL_MPS:
            out.append('spinning')
        elif v >= MOVING_MPS:
            out.append('racing' if held else 'coasting')
        elif v < STILL_MPS and (c >= SCAN_MOVED_MPS or g >= TILT_DPS):
            out.append('handled')
        else:
            out.append('parked')
    return out


def stretches(t, labels, pause_s=PAUSE_S):
    """[(first, last)] step indices of driving runs, split by long pauses and any handling."""
    out, first, last = [], None, None
    for i, label in enumerate(labels):
        if label not in ('racing', 'coasting'):
            continue
        broken = first is not None and (
            t[i] - t[last] > pause_s + STEP_S
            or any(labels[k] in ('spinning', 'handled') for k in range(last + 1, i)))
        if first is None or broken:
            if first is not None:
                out.append((first, last))
            first = i
        last = i
    if first is not None:
        out.append((first, last))
    # Coasting after LB is let go belongs to the stretch it ends; alone it is not driving.
    return [(a, b) for a, b in out if any(labels[k] == 'racing' for k in range(a, b + 1))]


@dataclass
class Stretch:
    start: float      # s from the first scan
    end: float
    metres: float     # by the wheels
    kind: str         # 'race' or 'short'

    @property
    def seconds(self):
        return self.end - self.start


def classify(t, speed, labels, course_m=COURSE_M):
    """Stretch objects for each driving run, sorted by start."""
    out = []
    for a, b in stretches(t, labels):
        seconds = t[b] - t[a] + STEP_S
        metres = float(np.sum(np.abs(speed[a:b + 1])) * STEP_S)
        race = seconds >= RACE_MIN_S and metres >= course_m and metres / seconds >= RACE_MIN_MPS
        out.append(Stretch(float(t[a]), float(t[b] + STEP_S), metres, 'race' if race else 'short'))
    return out


def events(t, labels, kind, merge_s=2.0):
    """[(start, end)] of runs of one label, merging runs closer than merge_s."""
    out = []
    for i, label in enumerate(labels):
        if label != kind:
            continue
        if out and t[i] - out[-1][1] <= merge_s:
            out[-1][1] = float(t[i] + STEP_S)
        else:
            out.append([float(t[i]), float(t[i] + STEP_S)])
    return [tuple(e) for e in out]


def lb_held(joy, grid):
    """Whether LB (button 4, the deadman) was held at each grid time, from (t, button) rows."""
    joy = np.asarray(joy, dtype=float).reshape(-1, 2)
    if not len(joy):
        return np.zeros(len(grid), bool)
    i = np.searchsorted(joy[:, 0], grid, side='right') - 1
    return np.where(i >= 0, joy[np.clip(i, 0, None), 1] > 0.5, False)


def scan_change(scan_t, scans, grid, step=STEP_S):
    """Median |range change| between the scans nearest each pair of grid times, per second."""
    out = np.full(len(grid), np.nan)
    if not len(scan_t):
        return out
    idx = np.clip(np.searchsorted(scan_t, grid), 0, len(scan_t) - 1)
    for k in range(1, len(grid)):
        a, b = scans[idx[k - 1]], scans[idx[k]]
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0.05) & (b > 0.05) & (a < 20) & (b < 20)
        if ok.sum() >= 20:
            out[k] = float(np.median(np.abs(a[ok] - b[ok]))) / step
    return out


def tilt_rate(imu, grid):
    """Largest roll/pitch rate (deg/s) in each step, from (t, gx, gy) rows of /sensors/imu/raw."""
    out = np.zeros(len(grid))
    imu = np.asarray(imu, dtype=float).reshape(-1, 3)
    if not len(imu):
        return out
    idx = np.searchsorted(imu[:, 0], grid)
    for k in range(1, len(grid)):
        rows = imu[idx[k - 1]:idx[k]]
        if len(rows):
            out[k] = float(np.max(np.hypot(rows[:, 1], rows[:, 2])))
    return out


def read_run(run):
    """Arrays from a run's bag: scan times and ranges (every 4th beam), odom (t, v), joy (t, LB), imu (t, gx, gy)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_analysis import bagread
    scan_t, scans, odom, joy, imu = [], [], [], [], []
    for topic, t, m in bagread.messages(run, ['/scan', '/odom', '/joy', '/sensors/imu/raw']):
        if topic == '/scan':
            scan_t.append(t)
            scans.append(np.asarray(m.ranges, dtype=np.float32)[::4])
        elif topic == '/odom':
            odom.append((t, m.twist.twist.linear.x))
        elif topic == '/joy':
            joy.append((t, m.buttons[4] if len(m.buttons) > 4 else 0))
        else:
            imu.append((t, m.angular_velocity.x, m.angular_velocity.y))
    if not scan_t or not odom:
        raise ValueError('the bag needs /scan and /odom')
    return (np.array(scan_t), np.stack(scans), np.array(odom), np.array(joy).reshape(-1, 2),
            np.array(imu).reshape(-1, 3))


def segment(scan_t, scans, odom, joy, imu, course_m=COURSE_M):
    """(step times from the first scan, wheel speed per step, labels, stretches) for one run's arrays."""
    t0 = scan_t[0]
    grid = np.arange(t0, scan_t[-1], STEP_S)
    order = np.argsort(odom[:, 0])
    # Each step's LiDAR change and tilt cover the half second before it; take the wheel speed
    # at the middle of that half second, so a stop is not read as a car moved by hand.
    speed = np.interp(grid - STEP_S / 2, odom[order, 0], np.abs(odom[order, 1]))
    labels = step_labels(speed, scan_change(scan_t, scans, grid), tilt_rate(imu, grid), lb_held(joy, grid))
    rel = grid - t0
    return rel, speed, labels, classify(rel, speed, labels, course_m)


def report(run, course_m=COURSE_M):
    scan_t, scans, odom, joy, imu = read_run(run)
    t, speed, labels, found = segment(scan_t, scans, odom, joy, imu, course_m)
    lb = lb_held(joy, t + scan_t[0])
    stalled = [(a, b) for a, b in events(t, ['stall' if (lab == 'parked' and held) else lab
                                             for lab, held in zip(labels, lb)], 'stall', 0.5)
               if b - a >= PAUSE_S]
    return dict(run=str(run), first_scan_epoch=float(scan_t[0]), seconds=float(t[-1] + STEP_S),
                stretches=found, handled=events(t, labels, 'handled'),
                spinning=events(t, labels, 'spinning'), stalled_with_lb=stalled)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--course-m', type=float, default=COURSE_M,
                        help='course length in metres; a race stretch covers at least this (default %(default)s)')
    parser.add_argument('--yaml', type=Path, help='also write the stretches and events to this YAML file')
    parser.add_argument('--utc-offset-hours', type=float, default=-8.0,
                        help="offset of the clock times in --yaml; the cars' own clocks used -8 in November 2026")
    args = parser.parse_args(argv)
    results = []
    for run in args.runs:
        try:
            r = report(run, args.course_m)
        except (OSError, ValueError) as exc:
            print(f'{run.name}: unreadable: {exc}')
            continue
        results.append(r)
        print(f"{run.name}: {r['seconds']:.0f} s recorded")
        for s in r['stretches']:
            print(f'    {s.kind:5s} {s.start:7.1f} to {s.end:7.1f} s  ({s.seconds:5.1f} s, {s.metres:5.0f} m, '
                  f'about {s.metres / args.course_m:4.1f} course lengths)')
        for name in ('stalled_with_lb', 'handled', 'spinning'):
            if r[name]:
                print(f"    {name.replace('_', ' ')}: " + ', '.join(f'{a:.0f}-{b:.0f} s' for a, b in r[name]))
    if args.yaml:
        import datetime
        import yaml
        zone = datetime.timezone(datetime.timedelta(hours=args.utc_offset_hours))

        def clock(r, s):
            return datetime.datetime.fromtimestamp(r['first_scan_epoch'] + s, zone).strftime('%H:%M:%S')
        doc = [dict(run=r['run'], first_scan_epoch=round(r['first_scan_epoch'], 3), seconds=round(r['seconds'], 1),
                    stretches=[dict(kind=s.kind, start_s=round(s.start, 1), end_s=round(s.end, 1),
                                    metres=round(s.metres, 1), clock=f'{clock(r, s.start)} to {clock(r, s.end)}')
                               for s in r['stretches']],
                    stalled_with_lb=[[round(a, 1), round(b, 1)] for a, b in r['stalled_with_lb']],
                    handled=[[round(a, 1), round(b, 1)] for a, b in r['handled']],
                    spinning=[[round(a, 1), round(b, 1)] for a, b in r['spinning']]) for r in results]
        args.yaml.write_text(yaml.safe_dump(doc, sort_keys=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
