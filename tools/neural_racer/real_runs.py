"""Offline real-run imitation experiment. No ROS graph, deployment or model export."""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for package in ('neural_racer', 'gap_follow', 'racerbot_car'):
    sys.path.insert(0, str(HERE.parents[1] / 'src' / package))

from neural_racer import model_store, observation as obs
from neural_racer.policy_numpy import PolicyNumpy
from neural_racer.safety_wrapper import SafetyWrapper, WrapperConfig, WrapperResult, scan_angles
from racerbot_car.layers import effective_config
from real_run_dataset import extract
from neural_racer_train.speed_calibration import SteeringCalibration, calibrated_steer, speed_bucket


def wrapper_for(car, cap):
    params = effective_config('neural_racer.yaml', 'neural_racer_node', car)
    config = {f.name: params[f.name] for f in fields(WrapperConfig)}
    config['max_speed'] = cap
    return SafetyWrapper(WrapperConfig(**config))


@dataclass
class ReplayTick:
    speed: float
    target: np.ndarray
    raw: np.ndarray
    result: WrapperResult
    forward_ms: float


def replay(data, policy, wrapper, *, calibration=None):
    """Recorded commands are targets only; the policy sees its own command history."""
    history = obs.History()
    cap = wrapper.cfg.max_speed
    reset = True
    prev_t = None
    prev = (0., 0.)
    for t, speed, ranges, geom, action, boundary in zip(
            *(data[k] for k in ('time_ns', 'speed', 'ranges', 'geometry', 'action', 'reset'))):
        if reset or boundary:
            policy.reset()
            history.reset()
            prev = (0., 0.)
            prev_t = None
        ang_min, inc, r_min, r_max = map(float, geom)
        encoded, valid = obs.pool_scan(ranges, r_min, r_max)
        history.push_tick(encoded, valid, speed)
        state = obs.build_observation(history, imu_present=False, mode=obs.MODE_REACTIVE, v_cap=cap)
        start = time.perf_counter_ns()
        net = policy.step(state)
        latency_ms = (time.perf_counter_ns() - start) / 1e6
        dt = 0.025 if prev_t is None else (t - prev_t) * 1e-9
        args = (net.ego_path, ranges, scan_angles(ang_min, inc, ranges.size), r_min, r_max, prev[0], dt)
        raw = np.array([net.steering, net.speed])
        prediction = raw if calibration is None else (calibrated_steer(net.steering, speed, calibration), net.speed)
        result = wrapper.apply(*prediction, *args)
        yield ReplayTick(speed, action, raw, result, latency_ms)
        reset = result.stopped
        prev = (result.steering, result.speed)
        history.push_command(*prev)
        prev_t = t


class Metrics:
    def __init__(self):
        self.n = self.turns = self.agree = self.interventions = self.stops = 0
        self.abs_error = np.zeros(2)

    def add(self, prediction, target, intervened=False, stopped=False):
        self.n += 1
        self.abs_error += np.abs(np.asarray(prediction) - target)
        if abs(target[0]) > 0.02:
            self.turns += 1
            self.agree += int(np.sign(prediction[0]) == np.sign(target[0]))
        self.interventions += int(intervened)
        self.stops += int(stopped)

    def report(self):
        return dict(ticks=self.n, steer_mae_rad=self.abs_error[0] / self.n,
                    speed_mae_mps=self.abs_error[1] / self.n, turns=self.turns,
                    turn_sign_accuracy=self.agree / self.turns if self.turns else None,
                    interventions=self.interventions, stops=self.stops)


def evaluate(dataset, model, cap, fit_days):
    model = Path(model)
    if not math.isfinite(cap) or not 0 < cap <= 6.37:
        raise ValueError('offline speed cap must be in (0, 6.37] m/s')
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if manifest['version'] != 1:
        raise ValueError('unsupported real-run dataset version')
    runs = manifest['runs']
    if set(fit_days) - {r['run'].split('-')[0] for r in runs}:
        raise ValueError('a requested fit day is absent')
    weights, card = model_store.load_dir(model)
    model_store.check_modalities(card, camera_enabled=False)
    policy = PolicyNumpy(weights, card)
    cal = SteeringCalibration()
    train, test = [], []
    for run in runs:
        (train if run['run'].split('-')[0] in fit_days else test).append(run)
    if not any(r['file'] for r in test):
        raise ValueError('no eligible evaluation ticks outside the fit days')

    for r in train:
        if not r['file']:
            continue
        with np.load(dataset / r['file']) as data:
            for tick in replay(data, policy, wrapper_for(r['car'], cap)):
                if not tick.result.stopped:
                    cal.add(tick.speed, tick.raw[0], tick.target[0])
    if fit_days and not cal.n:
        raise ValueError('no usable training ticks on requested fit days')

    groups = defaultdict(lambda: defaultdict(Metrics))
    latency = []
    for r in test:
        if not r['file']:
            continue
        run_id = r['car'] + '/' + r['run']
        wrapper = wrapper_for(r['car'], cap)
        with np.load(dataset / r['file']) as archive:
            data = dict(archive)
        candidates = [('base', None)]
        if fit_days:
            candidates.append(('adapted', cal.gains()))
        for name, adapter in candidates:
            for tick in replay(data, policy, wrapper, calibration=adapter):
                res = tick.result
                if name == 'base':
                    latency.append(tick.forward_ms)
                for key in ('all', 'run:' + run_id, 'course:' + r['course'],
                            'traffic:' + r['traffic'], 'speed:' + speed_bucket(tick.speed)):
                    groups[key][name].add((res.steering, res.speed), tick.target, res.intervened, res.stopped)

    return dict(model_hash=model_store.weights_hash(model / 'weights.npz'), cap_mps=cap,
                fit_days=sorted(fit_days), fit_ticks=cal.n,
                steering_gains=cal.gains(),
                fit_runs=[r['car'] + '/' + r['run'] for r in train if r['file']],
                evaluated_runs=[r['car'] + '/' + r['run'] for r in test if r['file']],
                empty_runs=[r['car'] + '/' + r['run'] for r in runs if not r['file']],
                forward_p99_ms=float(np.percentile(latency, 99)),
                groups={k: {p: m.report() for p, m in group.items()} for k, group in groups.items()})


def simulate(model, scorecard, track, grid, seeds, timeout):
    """Paired races: identical track, seed, grid, speed cap and masked IMU."""
    from neural_racer_train import referee
    from racerbot_car.identity import current_car_id

    report = json.loads(scorecard.read_text())
    model_hash = model_store.weights_hash(model / 'weights.npz')
    if report['model_hash'] != model_hash or not report.get('steering_gains'):
        raise ValueError('expected a fitted scorecard for these exact weights')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('simulation timeout must be positive')
    weights, card = model_store.load_dir(model)
    races = []
    for seed in seeds:
        for name, gains in [('base', None), ('adapted', report['steering_gains'])]:
            neural = referee.neural_factory(weights, card, v_cap=report['cap_mps'], imu=False,
                                            steering_gains=gains)
            drivers = {name: neural}
            if grid == 'behind-gap':
                drivers = {'gap': referee.gap_factory(), **drivers}
            elif grid == 'ahead-of-gap':
                drivers['gap'] = referee.gap_factory()
            race = referee.race(track, drivers, seed, timeout_s=timeout)
            print(race.as_dict(), flush=True)
            races.append(race.as_dict())
    return dict(model_hash=model_hash, car=current_car_id(), cap_mps=report['cap_mps'],
                steering_gains=report['steering_gains'], timeout_s=timeout, grid=grid, races=races)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    ex = sub.add_parser('extract', help='align recorded runs; output must be a new directory')
    ex.add_argument('data', type=Path)
    ex.add_argument('out', type=Path)
    ex.add_argument('--context', type=Path, default=HERE / 'real_run_context.csv')
    ev = sub.add_parser('evaluate', help='score an exported model; optionally fit on separate days')
    ev.add_argument('dataset', type=Path)
    ev.add_argument('model', type=Path)
    ev.add_argument('report', type=Path)
    ev.add_argument('--cap', type=float, default=1.0, help='offline policy and wrapper cap in m/s')
    ev.add_argument('--fit-day', action='append', default=[], help='YYYYMMDD; exclude both cars that day from scoring')
    sim = sub.add_parser('simulate', help='compare base and fitted gains in the existing CPU referee')
    sim.add_argument('model', type=Path)
    sim.add_argument('scorecard', type=Path)
    sim.add_argument('report', type=Path)
    sim.add_argument('--track', required=True)
    sim.add_argument('--grid', choices=('solo', 'behind-gap', 'ahead-of-gap'), default='solo')
    sim.add_argument('--seed', type=int, action='append', help='repeat for paired races')
    sim.add_argument('--timeout', type=float, default=60., help='simulated seconds per race')
    args = parser.parse_args()
    if args.command == 'extract':
        extract(args.data, args.out, args.context)
    elif args.command == 'evaluate':
        report = evaluate(args.dataset, args.model, args.cap, args.fit_day)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report['groups']['all'], indent=2))
    else:
        report = simulate(args.model, args.scorecard, args.track, args.grid,
                          args.seed or [12345], args.timeout)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
