"""Fit a reduced-order Gym turn model on the frozen November development drives.

See docs/ai-handoff/turn-response.md. Fit uses only development examples. Replay uses only
commands and initial speed, not gyro or wheel observations.
"""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from sim_fidelity.bootstrap import bootstrap

bootstrap()
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter

from eval_gap import CONTROL_DT, make_plant
from sim_fidelity.calibration import calibration_for_car, with_floor
from sim_fidelity.turn_response import TurnResponse, make_dynamics

DEV = {'rb1': {'20261111-153409', '20261111-153436', '20261111-153721', '20261111-155542'},
       'rb2': {'20261111-135934', '20261111-142254', '20261111-152132'}}
INPUT_HASH = {
    'blocks.csv': 'e364423d3841919b7ff076b2611793a4c58a81f7c1fdc1e077e4e4a0bd8d53bf',
    'ticks.csv': '4dbf56d312158ceb78d7f8cdf902be357d429faa31340ea549aead48f01006bf',
}
INITIAL = TurnResponse(1., 0., 4., .08)


def flat_map(directory):
    """Canonical clear floor: origin is 192 m from the boundary."""
    from PIL import Image
    directory = Path(directory)
    image = np.full((768, 768), 255, dtype=np.uint8)
    image[[0, -1], :] = image[:, [0, -1]] = 0
    Image.fromarray(image).save(directory / 'flat.png')
    (directory / 'flat_map.yaml').write_text(
        'image: flat.png\nresolution: 0.5\norigin: [-192.0, -192.0, 0.0]\n'
        'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
    angles = np.linspace(0, 2*np.pi, 128, endpoint=False)
    np.savetxt(directory / 'flat_centerline.csv', 100*np.c_[np.cos(angles), np.sin(angles)],
               delimiter=',', header='x_m,y_m', comments='')
    return str(directory.resolve() / 'flat')


def blocks(directory, car, fit):
    for name, expected in INPUT_HASH.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'unexpected canonical input: {name}')
    def key(row):
        return row['car'], row['run'], row['b_ns']
    with (directory / 'blocks.csv').open() as f:
        starts = {key(r): float(r['initial_mps']) for r in csv.DictReader(f)
                  if r['car'] == car and r['status'] == 'accepted'
                  and (not fit or (r['role'] == 'development' and r['run'] in DEV[car]))}
    with (directory / 'ticks.csv').open() as f:
        for block, rows in itertools.groupby(csv.DictReader(f), key):
            if block not in starts:
                continue
            yield block, starts.pop(block), list(rows)
    if starts:
        raise ValueError('missing accepted block')


def apply_response(plant, model):
    """Study-only replacement for both stock yaw dynamics and external grip."""
    if model is not None:
        plant.profile = replace(plant.profile, grip_envelope=False)
        plant.env.unwrapped.sim.dynamics_fn = make_dynamics(model)
    return plant


def study_calibration(car):
    cal = calibration_for_car(car)
    return with_floor('sfu_course_2026_11_11_rb2', cal) if car == 'rb2' else cal


def frozen_model(path, car):
    record = json.loads(path.read_text())
    if record['car'] != car or record['development'] != sorted(DEV[car]):
        raise ValueError('fit manifest does not match requested car')
    return TurnResponse(**record['model'])


def replay(track, cal, model, initial, rows):
    plant = make_plant(track, 0, calibration=cal, car_id=cal.car_id, max_laps=1000)
    try:
        apply_response(plant, model)
        plant.reset(np.zeros((1, 3)), speeds=[initial])
        states = [plant.truth(0).copy()]
        if len(rows) != 480:
            raise ValueError('incomplete 12-second command block')
        for step, row in enumerate(rows):
            if step != int(row['step']):
                raise ValueError('unordered replay')
            _, _, done, truncated, info = plant.step([(float(row['steer_rad']), float(row['demand_mps']))])
            if done or truncated or plant.chassis_contact_steps or abs(info['sim_time'] - (step+1)*CONTROL_DT) > 1e-6:
                raise RuntimeError('incomplete or mistimed replay')
            states.append(plant.truth(0).copy())
        states = np.asarray(states)
        if not np.all(np.isfinite(states)):
            raise ValueError('nonfinite replay')
        return states
    finally:
        plant.close()


def fit_response(examples, wheelbase):
    # Steering and speed evolve independently of yaw on this empty floor. Fit
    # using their Gym midpoints and exact first-order holds; evaluation uses RK4.
    steer, speed, observed, weights = [], [], [], []
    for run, states, rows in examples:
        midpoint = (states[1:] + states[:-1]) * .5
        steer.append(midpoint[:, 2])
        speed.append(midpoint[:, 3])
        observed.append([float(r['yaw_rad_s']) if int(r['turn_yaw']) else np.nan for r in rows])
        weights.append(run)
    speed, steer, observed = map(np.asarray, (speed, steer, observed))
    valid = np.isfinite(observed)
    if not valid.any():
        raise ValueError('no eligible development gyro observations')
    ids = np.broadcast_to(np.array(weights)[:, None], valid.shape)[valid]
    # Equal run mass rather than weighting the longest drive most.
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weight = 1 / np.sqrt(counts[inverse])

    def residual(p):
        gain, trim, limit, lag = p
        target = limit / np.maximum(speed, .5) * np.tanh(
            gain * speed**2 * np.tan(steer + trim) / (wheelbase * limit))
        decay = np.exp(-CONTROL_DT / lag)
        rates = lfilter([1-decay], [1., -decay], target, axis=1)
        return (rates[valid] - observed[valid]) * weight

    fit = least_squares(residual, list(asdict(INITIAL).values()),
                        bounds=([.6, -.04, 1.5, .025], [1.4, .04, 10., .35]))
    if not fit.success:
        raise ValueError(fit.message)
    return TurnResponse(*fit.x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['fit', 'replay'])
    ap.add_argument('car', choices=DEV)
    ap.add_argument('frozen', type=Path, help='directory containing canonical v1 tables')
    ap.add_argument('output', type=Path)
    ap.add_argument('--model', type=Path, help='frozen fit JSON (omit for reference Gym)')
    args = ap.parse_args()
    if args.mode == 'fit' and args.model:
        ap.error('fit does not accept a model')
    with TemporaryDirectory() as directory:
        run(args, flat_map(directory))


def run(args, track):
    cal = study_calibration(args.car)
    if args.mode == 'fit':
        examples = [(b[1], replay(track, cal, INITIAL, v, rows), rows)
                    for b, v, rows in blocks(args.frozen, args.car, True)]
        model = fit_response(examples, cal.wheelbase)
        with args.output.open('x') as f:
            json.dump(dict(car=args.car, development=sorted(DEV[args.car]), model=asdict(model)), f, indent=2)
            f.write('\n')
        return
    model = frozen_model(args.model, args.car) if args.model else None
    with args.output.open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('car', 'run', 'b_ns', 'step', 'body_forward_mps', 'body_yaw_rad_s'))
        for block, initial, rows in blocks(args.frozen, args.car, False):
            states = replay(track, cal, model, initial, rows)[1:]
            for step, s in enumerate(states):
                writer.writerow((*block, step, s[3]*np.cos(s[6]), s[5]))


if __name__ == '__main__':
    main()
