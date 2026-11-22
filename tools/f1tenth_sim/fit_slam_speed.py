#!/usr/bin/env python3
"""Freeze development-only wheel/SLAM speed scales, then evaluate unchanged tyres.

The sample-weighted median uses slam_motion's trusted forward-moving mask.
Command scale = recorded command eRPM gain / abs(odom eRPM gain) / wheel ratio,
as in existing floor profiles. Acceptance is November H=2 AND H=4: speed RMS
mean AND median strictly improve; yaw/slip/position means are <= 1.02 baseline.
October is a reported independent-floor check, not an additional acceptance gate.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import yaml

import fit_slam_tyres as ft
from run_analysis.slam_motion import pool_motions, summarize

ACCEPTANCE = dict(horizons=[2, 4], speed_mean_and_median_strictly_improve=True,
                  other_mean_max_ratio=1.02, acceptance_floor='nov-eval',
                  october='reported independent-floor check only')
GUARDS = ('yaw_rms_rad_s', 'slip_cg_rms_rad', 'position_error_m')


def baseline(car):
    return replace(ft.base_calibration(car), odom_speed_scale=1.0, speed_command_scale=1.0)


def candidate(car, ratio, gain_ratio):
    if not all(np.isfinite(x) and x > 0 for x in (ratio, gain_ratio)):
        raise ValueError('positive finite wheel ratio and command/odom gain ratio required')
    return replace(baseline(car), odom_speed_scale=ratio, speed_command_scale=gain_ratio / ratio)


def recorded_gains(data, car, run):
    params = Path(data) / car / 'racerbot_runs' / run / 'metadata/params'
    gains = []
    for node in ('ackermann_to_vesc_node', 'vesc_to_odom_node'):
        config = yaml.safe_load((params / f'{node}.yaml').read_text())
        values = next(iter(config.values()))['ros__parameters']
        gain = abs(float(values['speed_to_erpm_gain']))
        if not np.isfinite(gain) or gain <= 0 or values['speed_to_erpm_offset'] != 0:
            raise ValueError(f'{run}: positive finite gain and zero offset required')
        gains.append(gain)
    return gains


def freeze(car, loc, data, output):
    cal = baseline(car)
    motions, evidence = [], []
    for run in ft.DEV[car]:
        with np.load(Path(loc) / f'{run}.npz', allow_pickle=False) as saved:
            motion = ft.sr.extract_motion(saved['a'], cal.lr)
        motions.append(motion)
        evidence.append(dict(run=run, wheel_over_ground=summarize(motion, cal.wheelbase)['wheel_over_ground'],
                             command_and_odom_gains=recorded_gains(data, car, run)))
    gains = evidence[0]['command_and_odom_gains']
    if any(row['command_and_odom_gains'] != gains for row in evidence):
        raise ValueError('development runs have inconsistent command/odom gains')
    stats = summarize(pool_motions(motions), cal.wheelbase)['wheel_over_ground']
    ratio = stats['median']
    if ratio is None:
        raise ValueError('no moving development wheel/ground measurements')
    cal = candidate(car, ratio, gains[0] / gains[1])
    payload = dict(car=car, fit_runs=list(ft.DEV[car]), evidence=evidence, pooled_wheel_over_ground=stats,
                   odom_speed_scale=cal.odom_speed_scale, speed_command_scale=cal.speed_command_scale,
                   gain_ratio=gains[0] / gains[1], acceptance=ACCEPTANCE, frozen=True)
    report = dict(sha256=ft.digest(payload), payload=payload)
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('FROZEN', car, report['sha256'], ratio, cal.speed_command_scale, flush=True)
    return report


def load_frozen(path):
    record = json.loads(Path(path).read_text())
    p = record['payload']
    if not p.get('frozen') or record['sha256'] != ft.digest(p):
        raise ValueError('candidate not frozen or fingerprint mismatch')
    if p['car'] not in ft.DEV or p['fit_runs'] != list(ft.DEV[p['car']]) or p['acceptance'] != ACCEPTANCE:
        raise ValueError('development manifest or acceptance protocol mismatch')
    cal = candidate(p['car'], p['odom_speed_scale'], p['gain_ratio'])
    if cal.speed_command_scale != p['speed_command_scale']:
        raise ValueError('command scale does not match frozen ratio')
    return record


def acceptance(default, proposed):
    failures = []
    for h in ('2', '4'):
        a, b = default[h], proposed[h]
        if not a['windows'] or a['windows'] != b['windows']:
            failures.append(f'H{h}: missing or mismatched windows')
            continue
        for metric in ('speed_rms_mps', *GUARDS):
            keys = ('mean', 'median') if metric == 'speed_rms_mps' else ('mean',)
            for key in keys:
                x, y = a['metrics'][metric][key], b['metrics'][metric][key]
                valid = x is not None and y is not None and np.isfinite(x) and np.isfinite(y)
                valid = valid and (y < x if metric == 'speed_rms_mps' else y <= 1.02 * x)
                if not valid:
                    failures.append(f'H{h}: {metric} {key}')
    return dict(accepted=not failures, failures=failures)


def evaluate(frozen, loc, data, output):
    record = load_frozen(frozen)
    p = record['payload']
    car = p['car']
    proposed = candidate(car, p['odom_speed_scale'], p['gain_ratio'])
    reports = {}
    for role, names in {'nov-dev': ft.DEV[car], **ft.EVAL[car]}.items():
        tracks = ft.load_tracks(car, names, loc, data)
        pair = {}
        for label, cal in [('default', baseline(car)), ('candidate', proposed)]:
            runs = ft.score_tracks(tracks, cal, horizons=(2, 4))
            pair[label] = dict(runs=runs, pooled=ft.sr.aggregate(
                [w for run in runs for w in run['windows']], horizons=(2, 4)))
            print(car, role, label, flush=True)
        pair['verdict'] = acceptance(pair['default']['pooled'], pair['candidate']['pooled'])
        reports[role] = pair
    result = dict(car=car, frozen=record, accepted=reports['nov-eval']['verdict']['accepted'], reports=reports)
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('VERDICT', car, result['accepted'], reports['nov-eval']['verdict'], flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('freeze', 'evaluate'):
        parser = sub.add_parser(name)
        parser.add_argument('--loc', type=Path, default=Path.home() / '.cache/racerbot/loc')
        parser.add_argument('--data', type=Path, default=ft.sr.ROOT / 'data')
        parser.add_argument('--out', type=Path, required=True)
        if name == 'freeze':
            parser.add_argument('--car', choices=tuple(ft.DEV), required=True)
        else:
            parser.add_argument('--frozen', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.command == 'freeze':
        return freeze(args.car, args.loc, args.data, args.out)
    return evaluate(args.frozen, args.loc, args.data, args.out)


if __name__ == '__main__':
    main()
