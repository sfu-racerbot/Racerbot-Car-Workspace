"""Paired, ROS-free steering-table trials on a saved SLAM map and synthetic oval.

After fit_steering_table.py, run with RACERBOT_CAR=rb2 and .sim/python on PYTHONPATH:
  python3 tools/f1tenth_sim/compare_steering_table.py MAP_PACKAGE FIT_JSON OUTPUT_DIR

Only the ego tracking-steering request changes; no node, launch or car profile is modified.
The indexed table is tested diagnostically even if it failed qualification.
"""
import argparse
import csv
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np

import race_two as race
from steering_table import compensate
import track_import

FLOOR = 'sfu_course_2026_11_11_rb2'
LAPS = 3


def courses(package, out):
    line = package / 'raceline_profiled.csv'
    xy, speeds = race.racing_math.load_profiled_csv(line)
    spec = track_import.load_map(package / 'map.yaml')
    track = track_import.extract_track(spec, seed_xy=tuple(xy[0]))
    r, c = track_import.world_to_pixel(spec, xy[:, 0], xy[:, 1])
    if not np.all(track.region[r, c]) or min(speeds) <= 0:
        raise ValueError('saved speed profile leaves the connected free grid or stops')
    path = track_import.write_gym_track(track, out, 'nov11').with_suffix('')
    wide = Path(race.eval_gap.track_path('indoor_wide'))
    synth = np.loadtxt(str(wide) + '_centerline.csv', delimiter=',', skiprows=1)[:, :2]
    if np.allclose(synth[0], synth[-1]):
        synth = synth[:-1]
    rm = race.racing_math
    seg = rm.compute_segment_lengths(synth, closed=True)
    params = race.rv._pure_pursuit_params('rb2')
    speed = rm.compute_velocity_profile(seg, rm.estimate_path_curvature(synth, closed=True),
        v_max=min(5., params['max_speed']), v_min=.5, a_lat_max=params['max_lateral_accel'],
        a_accel_max=params['max_acceleration'], a_brake_max=params['max_braking_decel'],
        closed=True, smoothing_passes=8, friction_ellipse=True)
    synth_line = out / 'indoor_wide.csv'
    rm.save_profiled_csv(str(synth_line), synth, speed)
    return dict(nov11=(str(path), line), indoor_wide=(str(wide), synth_line))


def gate_times(positions, times, xy):
    """Truth projected on line segments, unwrapped. Gate is the final CSV station.

    The stock harness stops at the nearest starting waypoint, sometimes before
    the exact crossing. A gate one station earlier is reached by both arms.
    """
    seg = np.roll(xy, -1, axis=0) - xy
    length = np.linalg.norm(seg, axis=1)
    if min(length) <= 0:
        raise ValueError('duplicate station')
    cumulative = np.r_[0., np.cumsum(length)]
    arc, error = [], []
    for p in positions:
        delta = p - xy
        u = np.clip((delta * seg).sum(1) / length**2, 0, 1)
        dist = np.linalg.norm(delta - u[:, None]*seg, axis=1)
        i = np.argmin(dist)
        arc.append(cumulative[i] + u[i]*length[i])
        error.append(dist[i])
    L = cumulative[-1]
    progress = np.cumsum(np.r_[0., (np.diff(arc) + L/2) % L - L/2])
    marks = []
    for n in range(LAPS):
        gate = cumulative[-2] + n*L
        crossed = np.flatnonzero(progress >= gate)
        if len(crossed):
            j = crossed[0]
            marks.append(float(np.interp(gate, progress[j-1:j+1], times[j-1:j+1])))
    return np.diff([0.] + marks).tolist(), np.array(error[1:])


def run(job, settings, fit, out):
    course, seed, arm = job
    track, line = settings[course]
    race.rv.RACELINE_SOURCE = 'plan'
    xy, _ = race.racing_math.load_profiled_csv(line)
    params = race.rv._pure_pursuit_params('rb2')
    if (params['enable_opponent_overtake'] or params['enable_local_pass_planner']
            or not params['enable_deadman'] or not params['enable_lidar_safety']
            or params.get('understeer_curve_asked') or params.get('understeer_curve_delivered')):
        raise ValueError('unexpected passing, deadman, obstacle or grip settings')
    wall = []
    table_used = 0

    def adjust(delta, measured):
        nonlocal table_used
        result = compensate(fit['indexed'], measured / fit['wheel_over_ground'], delta)
        if result is None:
            return delta
        table_used += 1
        return float(np.clip(result, -params['max_steering_angle'], params['max_steering_angle']))

    stats, rows = race.run(track, line, 'solo', seed=seed, laps=LAPS, timeout=90.,
        car='rb2', floor=FLOOR, overrides={'steering_law': 'adaptive_pure_pursuit'},
        ego_steering=adjust if arm == 'indexed' else None,
        observe_step=lambda plant: wall.append(plant.chassis_map_clearance(0)))
    if len(wall) != len(rows) or not np.isfinite(wall).all() or any(r['active'] for r in rows):
        raise RuntimeError('incomplete clearance measurements or passing active')
    pos = np.vstack([xy[0], [[r['ego_x'], r['ego_y']] for r in rows]])
    t = [0.] + [r['time'] for r in rows]
    laps, err = gate_times(pos, t, xy)
    name = f'{course}-{seed}-{arm}.csv.gz'
    with gzip.open(out / name, 'wt') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'x', 'y', 'wall_m', 'xte_m', 'steer', 'wall_contact', 'car_contact'])
        for r, d, e in zip(rows, wall, err):
            writer.writerow([r['time'], r['ego_x'], r['ego_y'], d, e, r['steering'],
                             bool(r['wall_contact']), bool(r['car_contact'])])
    return dict(course=course, seed=seed, arm=arm, lap_s=laps,
                nonfinish=len(laps) < LAPS or stats['race_time_s'] is None,
                failed_start=float(np.linalg.norm(pos[np.asarray(t) <= 5.] - pos[0], axis=1).max()) < .5,
                wall_contacts=stats['wall_contact_ticks'],
                car_contacts=stats['car_contact_ticks'], min_wall_m=min(wall),
                p95_xte_m=float(np.percentile(err, 95)), ticks=len(rows), table_ticks=table_used,
                fallback_ticks=len(rows)-table_used if arm == 'indexed' else None,
                trace=name, position_sha256=hashlib.sha256(pos.tobytes()).hexdigest(),
                line_sha256=stats['line_sha256'], node_sha256=stats['node_sha256'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('package', type=Path)
    p.add_argument('fit', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--seeds', type=int, nargs='+', default=[12345, 12346])
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    settings = courses(args.package, args.out)
    fit = json.loads(args.fit.read_text())
    do = partial(run, settings=settings, fit=fit, out=args.out)
    jobs = product(settings, args.seeds, ('adaptive', 'indexed'))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(do, jobs):
            results.append(result)
            (args.out / 'runs.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
