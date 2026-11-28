#!/usr/bin/env python3
"""Run the explorer against gap_follow in the simulator, closed loop, no ROS.

The car_explorer conductor (src/car_explorer/car_explorer/explorer.py) is
called exactly as explorer_node calls it, at 20 Hz, and every setting it
returns is written into the gap_follow mirror's parameters, as the node's
set_parameters call does on the car. gap_follow (gap_mirror.GapController)
drives; the plant (FidelityPlant) is the car.

It answers two questions before any floor time (docs/explorer.md):

1. Does the explorer find the limit and back off, without the car touching a
   wall? Contact is judged by the plant's own chassis-against-map check, never
   by the explorer or gap_follow (TEST_QUALITY_STANDARDS A7).
2. Is what it drives enough to fit the car? ``--record`` saves the commands
   and true motion; ``--fit`` hands them to sim_fidelity.explorer_fit and
   prints how close the fit lands to the numbers the simulated car really has.

    python3 tools/f1tenth_sim/explore_sim.py --car rb2 --track car2_course_2026_11_11 \
        --floor sfu_course_2026_11_11_rb2 --minutes 6 --fit
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))
from sim_fidelity.bootstrap import bootstrap  # noqa: E402

ROOT = bootstrap()
sys.path.insert(0, str(ROOT / 'src' / 'car_explorer'))

import numpy as np  # noqa: E402

import eval_gap  # noqa: E402
from gap_mirror import GapController  # noqa: E402
from sim_fidelity.calibration import (FLOORS, add_car_argument, calibration_for_car,  # noqa: E402
                                      car_from_args, with_floor)
from car_explorer.explorer import (LATERAL, MIN_SPEED, SPEED, Conductor, Fence,  # noqa: E402
                                   Inputs, Rules, path_stop_margin, straight_room)

CONTROL_DT = eval_gap.CONTROL_DT


def run(track, calibration, car_id, *, minutes, seed=1, rules=None, fence=None, geometry=None,
        grip_scale=None, use_pose=True):
    """Drive ``minutes`` of simulated time with the explorer tuning gap_follow.

    Returns (report, recording). The recording holds 40 Hz arrays of the
    commands and the true speed and turn rate, for the fitter.
    """
    plant = eval_gap.make_plant(track, seed, calibration=calibration, car_id=car_id, max_laps=1000)
    line = plant.env.unwrapped.track.raceline
    obs, _ = plant.reset(np.array([[line.xs[0], line.ys[0], line.yaws[0]]], dtype=float))
    if grip_scale is not None:
        plant.set_grip_scale(grip_scale)

    class _Lidar:
        angle_min = float(-eval_gap.components_half_fov(car_id))
        angle_increment = float(eval_gap.components_increment(car_id))
        range_min = float(eval_gap.components_range_min(car_id))
    params = eval_gap.parse_set([], car_id)
    ctrl = GapController(params, _Lidar)

    rules = rules or Rules()
    # gap_follow's own min_speed, as explorer_node reads it through get_parameters.
    conductor = Conductor(fence or Fence(), rules, {MIN_SPEED: float(ctrl.p[MIN_SPEED])})
    for name, value in conductor.start().items():
        ctrl.p[name] = value
    geometry = geometry or dict(wheelbase=calibration.wheelbase, half_width=0.135, lidar_x=0.265,
                                front_reach=0.445, delay=rules.brake_delay)
    steps = int(round(minutes * 60.0 / CONTROL_DT))
    rec = {k: [] for k in ('t', 'steer', 'speed_cmd', 'speed', 'yaw_rate', 'max_speed', 'max_lateral',
                           'min_speed')}
    contact_steps, min_clear, applied = 0, math.inf, []
    info = {'lap_counts': [0]}
    for step in range(steps):
        ego = obs['agent_0']
        truth = plant.truth(0)
        state = np.asarray(ego['std_state'], dtype=float)
        steering, speed_cmd = ctrl.command(ego['scan'], float(state[3]), step,
                                           horizontal_accel=0.0, lb_pressed=True,
                                           fused_speed=float(truth[3]), yaw_rate=float(state[5]),
                                           measured_steering=None)
        now = step * CONTROL_DT
        if step % 2 == 0:  # the node's 20 Hz
            margin = path_stop_margin(ego['scan'], _Lidar.angle_min, _Lidar.angle_increment,
                                      speed=float(state[3]), steering=steering, **geometry)
            room = straight_room(ego['scan'], _Lidar.angle_min, _Lidar.angle_increment,
                                 half_width=geometry['half_width'], lidar_x=geometry['lidar_x'],
                                 front_reach=geometry['front_reach'])
            # The pose SLAM would give on the car: here the plant's true one.
            pose = (float(truth[0]), float(truth[1]), float(truth[4])) if use_pose else None
            out = conductor.update(Inputs(
                now=now, lb_held=True, speed=float(state[3]), steering=steering,
                yaw_rate=float(state[5]), stop_margin=margin, voltage=None,
                safety_state=ctrl.stop_reason if ctrl.stopped else 'gap_follow',
                speed_cmd=float(speed_cmd), pose=pose, straight_room=room))
            if out is not None:
                applied.append({'t': round(now, 2), **out})
                for name, value in out.items():
                    ctrl.p[name] = value
        obs, _r, done, _t, info = plant.step([(steering, speed_cmd)])
        # The motion each command produced: the state after this tick, as the
        # fitter's replay (fit_sim.simulate_commands) reports it.
        after = plant.truth(0)
        for key, value in (('t', now), ('steer', steering), ('speed_cmd', speed_cmd),
                           ('speed', float(after[3])), ('yaw_rate', float(after[5])),
                           ('max_speed', ctrl.p[SPEED]), ('max_lateral', ctrl.p[LATERAL]),
                           ('min_speed', ctrl.p[MIN_SPEED])):
            rec[key].append(value)
        clear = plant.chassis_map_clearance(0)
        if math.isfinite(clear):
            min_clear = min(min_clear, clear)
        if plant.wall_contacts():
            contact_steps += 1
        if done or conductor.done:
            break
    plant.close()
    report = conductor.report()
    sim_seconds = now if conductor.done else steps * CONTROL_DT
    report.update({'track': track, 'seed': seed, 'sim_seconds': round(sim_seconds, 1),
                   'laps': float(info['lap_counts'][0]), 'contact_steps': contact_steps,
                   'min_chassis_clearance_m': round(min_clear, 3) if math.isfinite(min_clear) else None,
                   'applied': applied})
    return report, {k: np.asarray(v, float) for k, v in rec.items()}


FIT_NAMES = ('surface_friction', 'steering_delay_ticks', 'speed_loop_brake_max')


def fit_recording(rec, start_calibration, names=FIT_NAMES, workers=4, maxiter=12):
    """Fit the recording: values, a 90% range for each, the score on held-back
    windows, and whether grip changed between the start and end of the session."""
    from sim_fidelity import explorer_fit as ef
    windows = ef.windows_from_series(rec['t'], rec['steer'], rec['speed_cmd'], rec['speed'],
                                     rec['yaw_rate'], np.ones(len(rec['t']), bool),
                                     horizon=2.0, spacing=3.0, min_speed=1.0)
    result = ef.fit(start_calibration, windows, list(names), maxiter=maxiter, popsize=6,
                    workers=workers, holdout_every=5, with_intervals=True)
    fitted = replace(start_calibration, **result.values)
    try:
        change = ef.drift(fitted, windows, 'surface_friction', workers=workers)
    except ValueError as exc:   # too short a session to split
        change = {'skipped': str(exc)}
    return result, len(windows), change


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_car_argument(parser)
    parser.add_argument('--track', default='car2_course_2026_11_11')
    parser.add_argument('--floor', choices=sorted(FLOORS))
    parser.add_argument('--minutes', type=float, default=6.0, help='simulated minutes')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--segment-sec', type=float, default=Rules().segment_sec)
    parser.add_argument('--friction', type=float,
                        help="the simulated car's true floor grip (default: the calibration's). "
                             'The fit starts from the calibration and must find this')
    parser.add_argument('--fit', action='store_true', help='fit the recording afterwards')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', help='write the JSON report here')
    parser.add_argument('--record', help='write the 40 Hz recording (.npz) here')
    args = parser.parse_args(argv)
    car_id = car_from_args(parser, args)
    base = with_floor(args.floor, calibration_for_car(car_id))
    truth = base if args.friction is None else replace(base, surface_friction=args.friction)
    eval_gap.track_path(args.track)
    report, rec = run(args.track, truth, car_id, minutes=args.minutes, seed=args.seed,
                      rules=Rules(segment_sec=args.segment_sec))
    if args.record:
        np.savez(args.record, **rec)
    if args.fit:
        result, n, change = fit_recording(rec, base, workers=args.workers)
        report['fit'] = {'windows': n, 'values': result.values,
                         'truth': {k: getattr(truth, k) for k in result.values},
                         'range_90': result.intervals,
                         'cost_before': round(result.cost_before, 4),
                         'cost_after': round(result.cost_after, 4),
                         'held_back_before': round(result.holdout_before, 4),
                         'held_back_after': round(result.holdout_after, 4),
                         'sensitivity': {k: round(v, 4) for k, v in result.sensitivity.items()},
                         'grip_change': change}
    text = json.dumps(report, indent=1, default=float)
    if args.output:
        Path(args.output).write_text(text + '\n')
    print(text)
    return 0 if report['contact_steps'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
