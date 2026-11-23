#!/usr/bin/env python3
"""Open-loop, multi-second FidelityPlant benchmark against slam_motion tracks.

Benchmark only: no product/control changes. Each window gets a NEW seeded Gym
ST plant (40 Hz, RK4 at 5 ms), including the car's default actuation, grip and
speed loop; --floor explicitly overlays a floor calibration. Sensor outputs
are not scored. Eight LiDAR beams suffice on fit_sim's empty floor, without
changing vehicle dynamics. A contact aborts rather than contaminating scores.

loc/slam_motion positions and speed refer to the REAR AXLE; Gym ST's x,y,v,
beta refer to the CG. Initialize CG position = rear + lr*heading and CG velocity
= (u_rear, v_rear + lr*r), and set r and beta directly. Real rack angle is not
recorded: initialize Gym's steering state from the in-force command (through
the profile's linkage, when present), not from curvature. Preload transport
FIFOs from recorded pre-window commands, without advancing physics. This is
an approximation to the unobserved actual steering/actuator state.

Scores use rear-axle speed/position, yaw rate, CG slip and wrapped heading.
All poses are in the window-start rear-axle body frame. RMS excludes t=0;
post-step truth at t+dt is compared with measured t+dt, including the horizon.
CG slip RMS ignores measured low-speed NaNs and reports its sample count.
Per-run and split aggregates are unweighted means/medians of WINDOW metrics,
not pooled sample RMS nor equal-run means. Starts are >=1 s apart on original
trusted timestamps, with ground speed >=1 m/s; all horizons use that schedule.
Interpolation never crosses a trusted-segment boundary. High-a_y bias means
|measured speed*yaw_rate| >=3 m/s². Negative turn-aligned yaw bias means less
turning than measured; negative absolute-slip bias means less CG slip magnitude.

CAVEAT: loc predicts with wheel+gyro; scan-derived motion is not independent
ground truth. CG geometry is the same unmeasured midpoint assumption as Step 2.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/f1tenth_sim')]
from sim_fidelity.bootstrap import bootstrap  # noqa: E402
from sim_fidelity.calibration import calibration_for_car, with_floor  # noqa: E402

bootstrap()
from run_analysis import bagread  # noqa: E402
from run_analysis.fit_sim import CONTROL_DT, empty_floor, resample_commands  # noqa: E402
from run_analysis.slam_motion import extract_motion  # noqa: E402

ANCHOR = np.array([0.0, -25.0])
NOV_DEV = {'rb1': {'153409', '153436', '153721', '155542'},
           'rb2': {'135934', '142254', '152132'}}
NOV_EVAL = {'rb1': {'095058', '095242', '095652'},
            'rb2': {'135453', '140322', '141227', '142036', '142604', '152844', '154204'}}
OCT_FIT = {'185036', '184556', '175728', '180355', '181616', '183607'}
OCT_HELD = {'184826', '183134', '181003', '181819', '184316', '175124'}


def split_role(car, run):
    day, _, suffix = run.partition('-')
    if day == '20261111' and suffix in NOV_DEV[car]:
        return 'nov-dev'
    if day == ('20261116' if car == 'rb1' else '20261111') and suffix in NOV_EVAL[car]:
        return 'nov-eval'
    if car == 'rb2' and day == '20261030':
        if suffix in OCT_FIT:
            return 'oct-fit'
        if suffix in OCT_HELD:
            return 'oct-held-out'
    return 'unassigned'


def read_commands(run):
    """Same /drive preference and /ackermann_cmd fallback as Run.commands; no data/ cache."""
    rows = {'/drive': [], '/ackermann_cmd': []}
    for topic, stamp, msg in bagread.messages(run, list(rows)):
        rows[topic].append((stamp, 0.0, msg.drive.steering_angle, msg.drive.speed))
    topic = '/drive' if rows['/drive'] else '/ackermann_cmd'
    drive = np.asarray(rows[topic], dtype=float).reshape(-1, 4)
    if not len(drive):
        raise ValueError(f'{run}: no /drive or /ackermann_cmd')
    if not np.isfinite(drive).all() or np.any(np.diff(drive[:, 0]) < 0):
        raise ValueError(f'{run}: invalid command sequence')
    return drive, topic


def windows(motion, horizons=(1, 2, 4)):
    """Yield (horizon, start index, segment indices), on one shared start schedule."""
    if any(h <= 0 or abs(h / CONTROL_DT - round(h / CONTROL_DT)) > 1e-6 for h in horizons):
        raise ValueError('horizons must be positive multiples of 0.025 s')
    t = motion['t']
    for segment in np.unique(motion['segment'][motion['trusted']]):
        ids = np.flatnonzero(motion['trusted'] & (motion['segment'] == segment))
        next_start = t[ids[0]]
        for index in ids:
            if (t[index] < next_start - 1e-6 or motion['ground_speed'][index] < 1.0
                    or not np.isfinite(motion['slip_cg'][index])):
                continue
            next_start = t[index] + 1.0
            for horizon in horizons:
                if t[index] + horizon <= t[ids[-1]] + 1e-7:
                    yield float(horizon), int(index), ids


def initial_state(motion, index, steering, lr):
    u = motion['longitudinal'][index]
    v = motion['lateral_cg'][index]
    return np.array([ANCHOR[0] + lr, ANCHOR[1], steering, np.hypot(u, v),
                     0.0, motion['yaw_rate'][index], np.arctan2(v, u)])


def make_plant(calibration, seed=1):
    import gymnasium as gym
    import f1tenth_gym  # noqa: F401
    from f1tenth_gym.envs.dynamic_models import DynamicModel
    from f1tenth_gym.envs.env_config import EnvConfig, ObservationConfig, SimulationConfig
    from f1tenth_gym.envs.integrators import IntegratorType
    from f1tenth_gym.envs.observation import ObservationType
    from sim_fidelity.plant import FidelityPlant, PROFILES, env_components

    profile = PROFILES['car']
    comp = env_components(profile, calibration)
    config = EnvConfig(
        seed=seed, map_name=empty_floor(), params=comp['params'], num_agents=1,
        control_config=comp['control_config'],
        simulation_config=SimulationConfig(timestep=CONTROL_DT, integrator_timestep=.005,
                                           integrator=IntegratorType.RK4,
                                           dynamics_model=DynamicModel.ST, max_laps=1000),
        observation_config=ObservationConfig(type=ObservationType.DIRECT),
        lidar_config=replace(comp['lidar_config'], num_beams=8),
        collision_check=comp['collision_check'], render_enabled=False)
    env = gym.make('f1tenth_gym:f1tenth-v0', config=config, render_mode=None)
    return FidelityPlant(env, profile, 1, seed, CONTROL_DT, calibration)


def initialize(plant, state, drive, start):
    """Benchmark adapter for Gym's full ST reset and FidelityPlant's truth cache."""
    plant.reset(state[None, [0, 1, 4]], speeds=[state[3]])
    obs, _ = plant.env.reset(options={'states': state[None, :]})
    plant._observe(obs)
    for key, column in (('steer_delays', 2), ('speed_delays', 3)):
        delay = getattr(plant, key)[0]
        ticks = start + np.arange(-delay.ticks, 0) * CONTROL_DT
        for stamp in ticks:
            j = np.searchsorted(drive[:, 0], stamp, side='right') - 1
            if j < 0:
                raise ValueError('missing pre-window command history')
            delay.push(float(drive[j, column]))


def _wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def score_window(motion, index, ids, states, lr):
    """Score post-step CG states at CONTROL_DT against one trusted measured segment."""
    states = np.asarray(states)
    elapsed = np.arange(1, len(states) + 1) * CONTROL_DT
    time = motion['t'][ids] - motion['t'][index]
    def measured(key, unwrap=False):
        values = motion[key][ids]
        return np.interp(elapsed, time, np.unwrap(values) if unwrap else values)
    yaw0 = motion['yaw'][index]
    dx, dy = measured('x') - motion['x'][index], measured('y') - motion['y'][index]
    actual_xy = np.column_stack((np.cos(yaw0) * dx + np.sin(yaw0) * dy,
                                 -np.sin(yaw0) * dx + np.cos(yaw0) * dy))
    simulated_xy = states[:, :2] - ANCHOR - lr * np.column_stack((np.cos(states[:, 4]),
                                                                 np.sin(states[:, 4])))
    u = states[:, 3] * np.cos(states[:, 6])
    v = states[:, 3] * np.sin(states[:, 6]) - lr * states[:, 5]
    speed_error = np.hypot(u, v) - measured('ground_speed')
    rate = measured('yaw_rate')
    yaw_error = states[:, 5] - rate
    slip = measured('slip_cg')
    slip_error = _wrap(states[:, 6] - slip)
    valid = np.isfinite(slip_error)
    rms = lambda a: float(np.sqrt(np.mean(a ** 2))) if len(a) else None
    high = np.abs(measured('lateral_accel')) >= 3.0
    mean = lambda a: float(np.mean(a)) if len(a) else None
    return dict(speed_rms_mps=rms(speed_error), yaw_rms_rad_s=rms(yaw_error),
                slip_cg_rms_rad=rms(slip_error[valid]), slip_samples=int(valid.sum()),
                position_error_m=float(np.linalg.norm(simulated_xy[-1] - actual_xy[-1])),
                heading_error_rad=float(abs(_wrap(states[-1, 4] - (measured('yaw', True)[-1] - yaw0)))),
                speed_bias_mps=mean(speed_error), yaw_bias_rad_s=mean(yaw_error),
                turn_aligned_yaw_bias_rad_s=mean(yaw_error * np.sign(rate)),
                abs_slip_bias_rad=mean((np.abs(states[:, 6]) - np.abs(slip))[valid]),
                high_ay_samples=int(high.sum()),
                high_ay_turn_yaw_bias_rad_s=mean((yaw_error * np.sign(rate))[high]),
                high_ay_abs_slip_bias_rad=mean((np.abs(states[:, 6]) - np.abs(slip))[high & valid]))


def replay_window(motion, drive, index, ids, horizon, calibration, seed=1, factory=make_plant):
    start = motion['t'][index]
    local_drive = drive.copy()
    local_drive[:, 0] -= start
    _, commands = resample_commands(local_drive, 0.0, horizon)
    steering = drive[max(0, np.searchsorted(drive[:, 0], start, side='right') - 1), 2]
    if calibration.steering_linkage is not None:
        steering = np.interp(steering, *np.array(calibration.steering_linkage).T)
    steering = np.clip(steering, calibration.steering_max_right, calibration.steering_max_left)
    steering = calibration.physical_steering(float(steering))
    state = initial_state(motion, index, steering, calibration.lr)
    plant = factory(calibration, seed)
    try:
        initialize(plant, state, local_drive, 0.0)
        states = []
        for command in commands:
            plant.step([command])
            states.append(plant.truth(0).copy())
            if plant.chassis_contact_steps:
                raise RuntimeError('empty-floor contact during rollout')
        states = np.asarray(states)
        if not np.isfinite(states).all():
            raise RuntimeError('nonfinite simulated state')
        return dict(horizon_s=horizon, start_t=float(start),
                    **score_window(motion, index, ids, states, calibration.lr))
    finally:
        plant.close()


def aggregate(rows, horizons=(1, 2, 4)):
    result = {}
    for horizon in horizons:
        selected = [row for row in rows if row['horizon_s'] == horizon]
        stats = {}
        if selected:
            for key in selected[0]:
                if key in ('horizon_s', 'start_t') or key.endswith('_samples'):
                    continue
                values = [r[key] for r in selected if r[key] is not None]
                stats[key] = dict(count=len(values), mean=float(np.mean(values)) if values else None,
                                  median=float(np.median(values)) if values else None)
        result[str(horizon)] = dict(windows=len(selected), metrics=stats)
    return result


def calibration_metadata(calibration):
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        return value
    return clean(asdict(calibration))


def analyze_run(path, car, loc_dir, calibration, horizons=(1, 2, 4), seed=1, factory=make_plant):
    run = Path(path).name
    with np.load(Path(loc_dir) / f'{run}.npz', allow_pickle=False) as saved:
        motion = extract_motion(saved['a'], calibration.lr)
    drive, topic = read_commands(path)
    history = max(calibration.speed_delay_ticks, calibration.steering_delay_ticks) * CONTROL_DT
    rows, missing = [], 0
    if not (car == 'rb1' and run == '20261111-153409'):
        for horizon, index, ids in windows(motion, horizons):
            if (motion['t'][index] - history < drive[0, 0]
                    or motion['t'][index] + horizon > drive[-1, 0]):
                missing += 1
                continue
            rows.append(replay_window(motion, drive, index, ids, horizon, calibration, seed, factory))
    warnings = []
    if not rows:
        warnings.append('no eligible windows')
    fraction = float(motion['trusted'].mean()) if len(motion['trusted']) else 0.0
    if fraction < .35 or any(sum(r['horizon_s'] == h for r in rows) < 10 for h in horizons):
        warnings.append('sparse trusted coverage/windows; do not interpret as a robust run estimate')
    return dict(car=car, run=run, split=split_role(car, run), command_topic=topic,
                trusted_fraction=fraction, warnings=warnings,
                missing_command_windows=missing, summary=aggregate(rows, horizons), windows=rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--car', required=True, choices=('rb1', 'rb2'))
    ap.add_argument('--floor', default=None)
    ap.add_argument('--loc', type=Path, default=Path.home() / '.cache/racerbot/loc')
    ap.add_argument('--json', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args(argv)
    cal = with_floor(args.floor, calibration_for_car(args.car))
    runs = []
    for path in args.runs:
        result = analyze_run(path.expanduser(), args.car, args.loc.expanduser(), cal, seed=args.seed)
        runs.append(result)
        print(args.car, path.name, {h: s['windows'] for h, s in result['summary'].items()},
              result['warnings'], flush=True)
    pooled = {}
    for role in sorted({run['split'] for run in runs}):
        pooled[role] = aggregate([w for run in runs if run['split'] == role for w in run['windows']])
    report = dict(car=args.car, floor=args.floor, seed=args.seed, calibration=calibration_metadata(cal),
                  method=__doc__, runs=runs, pooled_by_split=pooled)
    args.json.expanduser().write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
