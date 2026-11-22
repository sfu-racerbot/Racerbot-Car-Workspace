#!/usr/bin/env python3
"""Development-only Pacejka search, then frozen-candidate held-out Gym rollouts.

Inspired by On-Track SysID (arXiv:2411.17508), not a reproduction of its force
estimator. The sparse steady data and unmeasured rack angle do not identify
axle forces reliably here. Instead fit five effective tyre parameters directly
against scan-derived motion. Mass, CG, inertia, mu, actuators and speed loop
stay fixed. Peak D is a force/normal-load coefficient; C_S is Gym stiffness.

Protocol fixed before evaluation: objective = mean of yaw_RMS/.30 rad/s,
CG_slip_RMS/.10 rad, position_error/.50 m, equally weighted, at H=2 s.
Twenty seeded coarse candidates plus at most twenty Nelder-Mead calls use six
evenly spaced windows per development run. Rank the best three on ALL eligible
H=2 development windows. Fit uses November development ONLY, never October.
The external steering clamp is disabled for the candidate (FidelityPlant's
normal behaviour with TyreModel). A same-parameters clamp-on ablation uses the
existing grip path, without changing production code, before freezing.

Freeze a SHA256-checked JSON before `evaluate` can load evaluation runs. Accept
only if yaw/slip/position means strictly improve and medians do not worsen at
BOTH H=2 and H=4; speed mean AND median may worsen by at most .02 m/s. rb2 must
pass separately on November evaluation and October independent-floor drives.
No candidate is installed by this tool. Compiled immutable tyre functions are
cached, but every window still constructs a fresh seeded plant/reset stream.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

import slam_rollout as sr
from sim_fidelity import tyres
from sim_fidelity.calibration import calibration_for_car, with_floor
from sim_fidelity.tyres import TyreModel

DEV = {'rb1': ('20261111-153436', '20261111-153721', '20261111-155542'),
       'rb2': ('20261111-135934', '20261111-142254', '20261111-152132')}
EVAL = {'rb1': {'nov-eval': ('20261116-095242',)},
        'rb2': {'nov-eval': tuple('20261111-' + x for x in
                                 ('135453', '140322', '141227', '142036', '142604', '152844', '154204')),
                'oct-held-out': tuple('20261030-' + x for x in
                                      ('184826', '183134', '181003', '181819', '184316'))}}
NAMES = ('cornering_stiffness_front', 'cornering_stiffness_rear', 'peak_front', 'peak_rear', 'shape')
BOUNDS = ((1.2, 8.0), (.8, 7.0), (.35, 1.0), (.35, 1.0), (1.1, 1.9))
SCALES = {'yaw_rms_rad_s': .30, 'slip_cg_rms_rad': .10, 'position_error_m': .50}
ACCEPTANCE = dict(horizons=[2, 4], improve=list(SCALES), medians_not_worse=True,
                  speed_mean_and_median_tolerance_mps=.02, require_each_evaluation_floor=True)
_compile = lru_cache(maxsize=256)(tyres.make_dynamics)


def base_calibration(car):
    """Preserve the pre-fit linear-tyre baseline even after an accepted default changes."""
    base = calibration_for_car(car)
    if car == 'rb2':
        base = with_floor('pre_slam_speed_2026_11_rb2', base)
    return replace(base, cornering_stiffness_front=4.718,
                   cornering_stiffness_rear=5.4562, tyre_model=None, grip_curve=None, floor='')


def candidate_calibration(car, parameters):
    values = dict(zip(NAMES, map(float, parameters)))
    if len(parameters) != 5 or any(not low <= p <= high for p, (low, high) in zip(parameters, BOUNDS)):
        raise ValueError('five tyre parameters within the fixed search bounds required')
    return replace(base_calibration(car),
                   cornering_stiffness_front=values['cornering_stiffness_front'],
                   cornering_stiffness_rear=values['cornering_stiffness_rear'],
                   tyre_model=TyreModel(values['peak_front'], values['peak_rear'], values['shape']))


def plant_factory(calibration, seed=1, clamp=False):
    original = tyres.make_dynamics
    tyres.make_dynamics = _compile
    try:
        plant = sr.make_plant(calibration, seed)
    finally:
        tyres.make_dynamics = original
    if clamp and calibration.tyre_model is not None:
        # The installed dynamics remains saturating; only re-enable the existing clamp gate.
        plant._tyres = None
    return plant


def load_tracks(car, names, loc, data):
    cal = base_calibration(car)
    result = []
    for run in names:
        with np.load(Path(loc) / f'{run}.npz', allow_pickle=False) as saved:
            motion = sr.extract_motion(saved['a'], cal.lr)
        drive, topic = sr.read_commands(Path(data) / car / 'racerbot_runs' / run)
        if not len(drive):
            raise ValueError(f'{run}: no recorded drive commands')
        history = max(cal.speed_delay_ticks, cal.steering_delay_ticks) * sr.CONTROL_DT
        eligible = [w for w in sr.windows(motion) if motion['t'][w[1]] - history >= drive[0, 0]
                    and motion['t'][w[1]] + w[0] <= drive[-1, 0]]
        result.append(dict(run=run, motion=motion, drive=drive, topic=topic, eligible=eligible))
    return result


def score_tracks(tracks, calibration, horizons=(2,), subset=None, clamp=False):
    results = []
    for track in tracks:
        chosen = [w for w in track['eligible'] if w[0] in horizons]
        if subset is not None and len(chosen) > subset:
            chosen = [chosen[i] for i in np.linspace(0, len(chosen) - 1, subset, dtype=int)]
        rows = [sr.replay_window(track['motion'], track['drive'], i, ids, h, calibration,
                                 factory=lambda cal, seed: plant_factory(cal, seed, clamp))
                for h, i, ids in chosen]
        results.append(dict(run=track['run'], command_topic=track['topic'], windows=rows,
                            summary=sr.aggregate(rows, horizons)))
    return results


def objective(results):
    rows = [row for run in results for row in run['windows']]
    if not rows or any(row[k] is None or not np.isfinite(row[k]) for row in rows for k in SCALES):
        raise ValueError('objective requires nonempty finite yaw, slip and position measurements')
    return float(np.mean([np.mean([row[k] / scale for k, scale in SCALES.items()]) for row in rows]))


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fit(car, loc, data, output, coarse=20, maxfev=20):
    from scipy.optimize import minimize
    from scipy.stats import qmc
    if coarse < 1 or maxfev < 1:
        raise ValueError('search budgets must be positive')
    tracks = load_tracks(car, DEV[car], loc, data)
    history = []
    def trial(parameters):
        parameters = list(map(float, parameters))
        try:
            loss = objective(score_tracks(tracks, candidate_calibration(car, parameters), subset=6))
        except (RuntimeError, ValueError):
            loss = 1e6
        history.append(dict(parameters=parameters, subset_loss=loss))
        print(car, 'trial', len(history), f'{loss:.6f}', parameters, flush=True)
        return loss
    anchors = [[4.718, 5.4562, .55, .55, 1.6], [3, 2, .55, .55, 1.4],
               [4, 2, .65, .5, 1.6], [2, 1.5, .55, .55, 1.8]]
    points = (qmc.scale(qmc.LatinHypercube(5, seed=17).random(coarse - 4), *np.array(BOUNDS).T)
              if coarse > 4 else [])
    for point in [*anchors[:coarse], *points]:
        trial(point)
    best = min(history, key=lambda row: row['subset_loss'])
    minimize(trial, best['parameters'], method='Nelder-Mead', bounds=BOUNDS,
             options=dict(maxfev=maxfev, xatol=.01, fatol=.001))
    finalists, seen = [], set()
    for row in sorted(history, key=lambda row: row['subset_loss']):
        key = tuple(row['parameters'])
        if key in seen:
            continue
        seen.add(key)
        scores = score_tracks(tracks, candidate_calibration(car, key))
        finalists.append(dict(parameters=list(key), full_h2_loss=objective(scores), results=scores))
        print(car, 'full H2 finalist', len(finalists), finalists[-1]['full_h2_loss'], flush=True)
        if len(finalists) == 3:
            break
    winner = min(finalists, key=lambda row: row['full_h2_loss'])
    cal = candidate_calibration(car, winner['parameters'])
    ablation = score_tracks(tracks, cal, clamp=True)
    payload = dict(car=car, fit_runs=list(DEV[car]), parameters=winner['parameters'],
                   parameter_names=list(NAMES), bounds=BOUNDS, objective_scales=SCALES,
                   acceptance=ACCEPTANCE, clamp=False,
                   clamp_ablation=dict(off_loss=winner['full_h2_loss'], on_loss=objective(ablation)),
                   fixed=dict(mass=cal.mass, lr=cal.lr, yaw_inertia=cal.yaw_inertia,
                              surface_friction=cal.surface_friction),
                   history=history, finalists=finalists, frozen=True)
    report = dict(sha256=digest(payload), payload=payload)
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('FROZEN', car, report['sha256'], winner['parameters'], payload['clamp_ablation'], flush=True)
    return report


def load_frozen(path):
    report = json.loads(Path(path).read_text())
    payload = report['payload']
    if not payload.get('frozen') or report['sha256'] != digest(payload):
        raise ValueError('candidate is not frozen or fingerprint mismatch')
    if (payload['fit_runs'] != list(DEV[payload['car']]) or payload['acceptance'] != ACCEPTANCE
            or payload['objective_scales'] != SCALES or payload['clamp']):
        raise ValueError('frozen protocol differs from development-only protocol')
    return report


def progressive_curve(tracks, wheelbase):
    """Development-only command/SLAM envelope proxy, not identified axle friction."""
    asked, delivered = [], []
    for track in tracks:
        m = track['motion']
        keep = m['trusted'] & (m['ground_speed'] >= 1) & np.isfinite(m['steer_cmd'])
        asked.extend(np.abs(m['ground_speed'][keep] ** 2 * np.tan(m['steer_cmd'][keep]) / wheelbase))
        delivered.extend(np.abs(m['lateral_accel'][keep]))
    asked, delivered = np.asarray(asked), np.asarray(delivered)
    curve = [(0.0, 0.0)]
    for low in np.arange(0, 10, .5):
        keep = (asked >= low) & (asked < low + .5)
        if keep.sum() >= 10:
            x = float(np.median(asked[keep]))
            if x > curve[-1][0]:
                curve.append((x, min(x, float(np.median(delivered[keep])))))
    if len(curve) < 2:
        raise ValueError('insufficient development data for a progressive envelope')
    return tuple(curve)


def ablate(frozen, loc, data, output):
    """Refreeze unchanged tyre values after a development-only ProgressiveGrip comparison."""
    report = load_frozen(frozen)
    payload = report['payload']
    car = payload['car']
    tracks = load_tracks(car, DEV[car], loc, data)
    cal = candidate_calibration(car, payload['parameters'])
    curve = progressive_curve(tracks, cal.wheelbase)
    with_curve = replace(cal, grip_curve=curve, grip_curve_falls=True)
    score = objective(score_tracks(tracks, with_curve, clamp=True))
    payload['clamp_ablation'].update(progressive_on_loss=score, progressive_curve=curve,
                                     curve_source='November development only; commanded-steer envelope proxy')
    report['sha256'] = digest(payload)
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('REFROZEN', car, report['sha256'], payload['clamp_ablation'], flush=True)
    return report


def acceptance(default, candidate):
    failures = []
    for horizon in ('2', '4'):
        a, b = default[horizon], candidate[horizon]
        if not a['windows'] or a['windows'] != b['windows']:
            failures.append(f'H{horizon}: missing or mismatched windows')
            continue
        for metric in SCALES:
            x, y = a['metrics'][metric], b['metrics'][metric]
            if not (y['mean'] < x['mean'] and y['median'] <= x['median']):
                failures.append(f'H{horizon}: {metric} mean/median')
        x, y = a['metrics']['speed_rms_mps'], b['metrics']['speed_rms_mps']
        if any(y[k] > x[k] + .02 for k in ('mean', 'median')):
            failures.append(f'H{horizon}: speed tolerance')
    return dict(accepted=not failures, failures=failures)


def evaluate(frozen, loc, data, output):
    frozen = load_frozen(frozen)
    car = frozen['payload']['car']
    cal = candidate_calibration(car, frozen['payload']['parameters'])
    reports = {}
    # This is the only dispatch site that opens held-out drives.
    for role, names in {'nov-dev': DEV[car], **EVAL[car]}.items():
        tracks = load_tracks(car, names, loc, data)
        pair = {}
        for label, calibration in [('default', base_calibration(car)), ('candidate', cal)]:
            runs = score_tracks(tracks, calibration, (1, 2, 4))
            pair[label] = dict(runs=runs, pooled=sr.aggregate([w for r in runs for w in r['windows']]))
            print(car, role, label, {h: s['windows'] for h, s in pair[label]['pooled'].items()}, flush=True)
        if role != 'nov-dev':
            pair['verdict'] = acceptance(pair['default']['pooled'], pair['candidate']['pooled'])
        reports[role] = pair
    accepted = all(reports[role]['verdict']['accepted'] for role in EVAL[car])
    result = dict(car=car, frozen_sha256=frozen['sha256'], accepted=accepted, reports=reports)
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('VERDICT', car, accepted, flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('fit', 'ablate', 'evaluate'):
        parser = sub.add_parser(name)
        parser.add_argument('--loc', type=Path, default=Path.home() / '.cache/racerbot/loc')
        parser.add_argument('--data', type=Path, default=sr.ROOT / 'data')
        parser.add_argument('--out', type=Path, required=True)
        if name == 'fit':
            parser.add_argument('--car', choices=('rb1', 'rb2'), required=True)
            parser.add_argument('--coarse', type=int, default=20)
            parser.add_argument('--maxfev', type=int, default=20)
        else:
            parser.add_argument('--frozen', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.command == 'fit':
        fit(args.car, args.loc, args.data, args.out, args.coarse, args.maxfev)
    elif args.command == 'ablate':
        ablate(args.frozen, args.loc, args.data, args.out)
    else:
        evaluate(args.frozen, args.loc, args.data, args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
