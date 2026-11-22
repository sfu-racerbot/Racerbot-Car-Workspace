"""Read-only observer for PR 94's CPU race. Run from the workspace root; see report."""
import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import sys

repo = Path.cwd()
for p in (repo / '.sim/python', repo / 'tools/neural_racer', repo / 'src/racerbot_car'):
    sys.path.insert(0, str(p))

import numpy as np
from neural_racer_train import referee
from neural_racer import model_store, observation


def observe(driver, plant, index, writer):
    """Delegate every operation to the real driver; observe before plant integration."""
    policy_step, apply, command = driver.policy.step, driver.wrapper.apply, driver.command
    row = {}
    line = plant.env.unwrapped.track.raceline
    xy = np.column_stack((line.xs, line.ys))

    def policy(o):
        out = policy_step(o)
        row.update(network_steer_rad=out.steering, network_speed_mps=out.speed)
        return out

    def wrapper(steer, speed, path, scan, angles, lo, hi, prev, dt):
        res = apply(steer, speed, path, scan, angles, lo, hi, prev, dt)
        valid = observation.beam_validity(scan, lo, hi)
        boundary = driver.wrapper.boundary(angles)
        margins = np.where(valid & np.isfinite(boundary), scan - boundary, np.inf)
        closest = np.argmin(margins)
        ahead = np.abs(angles) <= math.radians(10)
        row.update(prewrap_steer_rad=steer, binding=res.binding, reason=res.reason,
                   wrapper_clear_m=res.clearance, invalid_fraction=1-valid.mean(),
                   closest_beam_deg=np.degrees(angles[closest]) if np.isfinite(margins[closest]) else None,
                   forward10_clear_m=np.min(margins[ahead]))
        return res

    def tick(sensors, truth, step):
        row.clear()
        out = command(sensors, truth, step)
        delta = xy - truth[:2]
        i = np.argmin(np.sum(delta**2, axis=1))
        heading_error = truth[4] - line.yaws[i]
        row.update(step=step, time_s=step*.025, x_m=truth[0], y_m=truth[1],
                   yaw_rad=truth[4], actual_steer_rad=truth[2], true_v_mps=truth[3],
                   measured_v_mps=sensors['std_state'][3], command_steer_rad=out[0],
                   command_speed_mps=out[1], map_gap_m=plant.chassis_map_clearance(index),
                   line_point=i, dist_line_point_m=np.linalg.norm(delta[i]),
                   heading_error_deg=np.degrees(np.arctan2(np.sin(heading_error),np.cos(heading_error))))
        writer.writerow(row)
        return out

    driver.policy.step, driver.wrapper.apply, driver.command = policy, wrapper, tick
    return driver


FIELDS = ('step time_s x_m y_m yaw_rad true_v_mps measured_v_mps actual_steer_rad '
          'network_steer_rad prewrap_steer_rad command_steer_rad network_speed_mps '
          'command_speed_mps binding reason wrapper_clear_m closest_beam_deg forward10_clear_m '
          'invalid_fraction map_gap_m line_point dist_line_point_m heading_error_deg').split()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('scorecard', type=Path)
    p.add_argument('model', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('track')
    p.add_argument('seed', type=int)
    p.add_argument('grid', choices=['solo','behind-gap'])
    p.add_argument('--timeout', default=60., type=float)
    p.add_argument('--gap-control', action='store_true')
    args = p.parse_args()
    data = json.loads(args.scorecard.read_text())['date_split']
    assert os.environ['RACERBOT_CAR'] == 'rb2'
    assert model_store.weights_hash(args.model/'weights.npz') == data['model_hash']
    weights, card = model_store.load_dir(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    names = ['gap-control'] if args.gap_control else ['base', 'adapted']
    for name in names:
        stem = f'{args.track}-{args.grid}-{args.seed}-{name}'
        if args.gap_control:
            result = referee.race(args.track, {name:referee.gap_factory()}, args.seed, timeout_s=args.timeout)
        else:
            with gzip.open(args.out/f'{stem}.csv.gz', 'wt') as fp:
                writer = csv.DictWriter(fp, fieldnames=FIELDS)
                writer.writeheader()
                def factory(plant):
                    ego = referee.NeuralDriver(weights, card, v_cap=data['cap_mps'], imu=False,
                                               steering_gains=None if name=='base' else data['steering_gains'])
                    return observe(ego, plant, int(args.grid=='behind-gap'), writer)
                drivers = {'gap':referee.gap_factory()} if args.grid=='behind-gap' else {}
                drivers[name] = factory
                result = referee.race(args.track, drivers, args.seed, timeout_s=args.timeout)
        packet = dict(**result.as_dict(), timeout_s=args.timeout, grid=args.grid)
        (args.out/f'{stem}.json').write_text(json.dumps(packet)+'\n')
        print(stem, result.progress, result.finish_time, result.contact, flush=True)


if __name__ == '__main__':
    main()
