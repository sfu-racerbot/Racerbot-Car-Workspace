"""Train and judge a sector lookahead schedule without ROS or hardware.

Run with RACERBOT_CAR=rb2 PYTHONPATH="$PWD/.sim/python" after simulator setup:
  python3 tools/f1tenth_sim/sector_trials.py train DATA_RB2/racerbot_auto OUTPUT
  python3 tools/f1tenth_sim/sector_trials.py judge DATA_RB2/racerbot_auto OUTPUT

Training: whole-lap probes at three offsets, two seeds. Judging: frozen
per-map offsets versus the unchanged law, on three disjoint seeds.
"""
import argparse
import csv
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

import race_two as race
from sector_lookahead import SECTORS, OFFSETS, SectorLookahead, select_schedule
import track_import

PACKAGES = dict(nov09='20261109-185146-rebuilt', nov11='20261111-151026-rebuilt',
                nov16='20261116-105215-line')
FLOOR = 'sfu_course_2026_11_11_rb2'
TRAIN = (4101, 4102)
JUDGE = (5101, 5102, 5103)
LAPS = 3


def setup(packages, out):
    rm = race.racing_math
    paths = {}
    for key, name in PACKAGES.items():
        line = packages / name / 'raceline_profiled.csv'
        xy, speed = rm.load_profiled_csv(line)
        spec = track_import.load_map(line.parent / 'map.yaml')
        track = track_import.extract_track(spec, seed_xy=tuple(xy[0]))
        iy, ix = track_import.world_to_pixel(spec, xy[:, 0], xy[:, 1])
        if not np.all(track.region[iy, ix]) or min(speed) <= 0:
            raise ValueError(f'{name}: line outside connected free grid or stopped')
        path = track_import.write_gym_track(track, out, key)
        paths[key] = (str(path.with_suffix('')), line)
    track = race.eval_gap.track_path('indoor_wide')
    xy = np.loadtxt(track + '_centerline.csv', delimiter=',', skiprows=1)[:, :2]
    if np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    seg = rm.compute_segment_lengths(xy, closed=True)
    p = race.rv._pure_pursuit_params('rb2')
    speed = rm.compute_velocity_profile(seg, rm.estimate_path_curvature(xy, closed=True),
        v_max=min(5., p['max_speed']), v_min=.5, a_lat_max=p['max_lateral_accel'],
        a_accel_max=p['max_acceleration'], a_brake_max=p['max_braking_decel'],
        closed=True, smoothing_passes=8, friction_ellipse=True)
    line = out / 'indoor_wide.csv'
    rm.save_profiled_csv(str(line), xy, speed)
    paths['indoor_wide'] = (track, line)
    return paths


def project(xy, positions):
    seg = np.roll(xy, -1, axis=0) - xy
    lengths = np.linalg.norm(seg, axis=1)
    if min(lengths) <= 0:
        raise ValueError('duplicate station')
    cumulative = np.r_[0., np.cumsum(lengths)]
    arc, error = [], []
    for pos in positions:
        dx = pos - xy
        u = np.clip((dx * seg).sum(1) / lengths**2, 0., 1.)
        dist = np.linalg.norm(dx - u[:, None] * seg, axis=1)
        j = np.argmin(dist)
        arc.append(cumulative[j] + u[j]*lengths[j])
        error.append(dist[j])
    length = cumulative[-1]
    progress = np.cumsum(np.r_[0., (np.diff(arc) + length/2) % length - length/2])
    return np.asarray(arc), progress, np.array(error), length, lengths[-1]


def assess(xy, pos, times, wall):
    arc, progress, error, length, closing = project(xy, pos)
    # Last station is a shared gate reached before the stock nearest-index stop.
    def crossing(g):
        idx = np.flatnonzero(progress >= g - 1e-10)
        if not len(idx):
            return None
        j = idx[0]
        return float(np.interp(g, progress[j-1:j+1], times[j-1:j+1]))

    marks = [crossing(n * length - closing) for n in range(1, LAPS+1)]
    laps = np.diff([0.] + [t for t in marks if t is not None]).tolist()
    sectors = []
    for i in range(SECTORS):
        start, end = np.array([i, i+1])*length/SECTORS
        mask = (arc[1:] >= start) & (arc[1:] < end)
        durations = []
        # Use complete sectors only; use the middle flying circuit so every
        # sector has the same opportunity even if the final gate stops early.
        a, b = crossing(length + start), crossing(length + end)
        if a is not None and b is not None and b > a:
            durations.append(b-a)
        sectors.append(dict(times=durations, min_wall=float(np.min(wall[mask])))
                       if durations and mask.any() else None)
    return laps, sectors, error[1:]


def run(job, paths, out):
    course, seed, key, offsets = job
    path, line = paths[course]
    race.rv.RACELINE_SOURCE = 'plan'
    params = race.rv._pure_pursuit_params('rb2')
    if (params['enable_opponent_overtake'] or params['enable_local_pass_planner']
            or not params['enable_deadman'] or not params['enable_lidar_safety']):
        raise RuntimeError('unexpected Car 2 safety or passing configuration')
    wall = []
    stats, rows = race.run(path, line, 'solo', seed=seed, laps=LAPS, timeout=75.,
        floor=FLOOR, car='rb2', overrides={'steering_law': 'adaptive_pure_pursuit'},
        ego_law_factory=None if offsets is None else lambda law: SectorLookahead(law, offsets),
        observe_step=lambda plant: wall.append(plant.chassis_map_clearance(0)))
    xy, _ = race.racing_math.load_profiled_csv(line)
    if not np.isfinite(wall).all() or len(wall) != len(rows) or any(r['active'] for r in rows):
        raise RuntimeError('invalid clearance samples or passing active')
    pos = np.vstack([xy[0], [[r['ego_x'], r['ego_y']] for r in rows]])
    t = np.array([0.] + [r['time'] for r in rows])
    wall = np.asarray(wall)
    laps, sectors, err = assess(xy, pos, t, wall)
    trace = f'{course}-{seed}-{key}.csv.gz'
    with gzip.open(out / trace, 'wt') as f:
        writer = csv.writer(f)
        writer.writerow(('s','x','y','xte_m','wall_m','wall_contact','car_contact'))
        writer.writerows((r['time'], r['ego_x'], r['ego_y'], e, d,
                          bool(r['wall_contact']), bool(r['car_contact']))
                         for r, e, d in zip(rows, err, wall))
    return dict(course=course, seed=seed, arm=key, offsets=offsets,
                offset=offsets[0] if offsets and len(set(offsets)) == 1 else 0.,
                sectors=sectors, laps=laps, p95=float(np.percentile(err, 95)),
                wall=float(min(wall)), contacts=stats['car_contact_ticks'] + stats['wall_contact_ticks'],
                wall_contact=stats['wall_contact_ticks'], car_contact=stats['car_contact_ticks'],
                dnf=len(laps) < LAPS or stats['race_time_s'] is None,
                failed_start=float(np.linalg.norm(pos[t <= 5.] - pos[0], axis=1).max()) < .5,
                elapsed=stats['elapsed_s'], line_sha256=stats['line_sha256'],
                trace=trace, position_sha256=hashlib.sha256(pos.tobytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['train', 'judge'])
    p.add_argument('packages', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    paths = setup(args.packages, args.out)
    frozen = args.out / 'frozen.json'
    if args.stage == 'train':
        jobs = [(c, s, str(o), [o]*SECTORS) for c in paths for s in TRAIN for o in OFFSETS]
    else:
        config = json.loads(frozen.read_text())
        for c, (_, line) in paths.items():
            if config[c]['line_sha256'] != hashlib.sha256(line.read_bytes()).hexdigest():
                raise ValueError(f'{c}: frozen schedule came from a different line')
        jobs = [(c, s, arm, config[c]['offsets'] if arm == 'sector' else None)
                for c in paths for s in JUDGE for arm in ('adaptive', 'sector')]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(partial(run, paths=paths, out=args.out), jobs):
            results.append(result)
            (args.out / (args.stage + '.json')).write_text(json.dumps(results, indent=2) + '\n')
            print(result['course'], result['seed'], result['arm'], result['laps'], result['contacts'], flush=True)
    if args.stage == 'train':
        config = {}
        for c, (_, line) in paths.items():
            schedule = select_schedule([r for r in results if r['course'] == c])
            config[c] = dict(offsets=schedule, line_sha256=hashlib.sha256(line.read_bytes()).hexdigest())
        frozen.write_text(json.dumps(config, indent=2) + '\n')


if __name__ == '__main__':
    main()
