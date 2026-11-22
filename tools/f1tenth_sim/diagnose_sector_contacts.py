"""Passively identify November 9 wall contacts from completed sector trials.

Usage: python3 diagnose_sector_contacts.py DATA_RB2 TRIAL_OUTPUT
Requires MCAP ROS 2 decoders. Reads the bag's /map as a second judge;
the replay still drives against the exact original gym-imported mask.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from PIL import Image
from scipy.ndimage import distance_transform_edt, label

import sector_trials as trial


def published_grid(bag_dir, package):
    cfg = yaml.safe_load((package / 'map.yaml').read_text())
    src = np.asarray(Image.open(package / cfg['image']))
    occupancy = (255 - src.astype(float)) / 255
    saved = np.where(occupancy > cfg['occupied_thresh'], 100,
                     np.where(occupancy < cfg['free_thresh'], 0, -1))
    count = 0
    for bag in sorted(bag_dir.glob('*.mcap')):
        with bag.open('rb') as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, _, _, msg in reader.iter_decoded_messages(topics=['/map']):
                info = msg.info
                grid = np.asarray(msg.data).reshape(info.height, info.width)[::-1]
                pose = info.origin
                if (not np.allclose([pose.position.x, pose.position.y, 0.], cfg['origin'],
                                    rtol=0, atol=1e-7)
                        or abs(pose.orientation.w - 1) > 1e-8
                        or abs(info.resolution - cfg['resolution']) > 1e-8
                        or not np.array_equal(grid, saved)):
                    raise ValueError('bag and saved SLAM grid do not agree')
                count += 1
    if not count:
        raise ValueError('no paired published occupancy grid')
    return grid, src


def diagnose(record, out, data):
    package = data / 'racerbot_auto' / trial.PACKAGES['nov09']
    line = package / 'raceline_profiled.csv'
    xy, _ = trial.race.racing_math.load_profiled_csv(line)
    grid, src = published_grid(data / 'racerbot_runs/20261109-190202/bag', package)
    mask = np.asarray(Image.open(out / 'nov09.png')) == 0
    spec = trial.track_import.load_map(package / 'map.yaml')
    components, _ = label(mask, np.ones((3, 3)))
    occ_dist = spec.resolution * distance_transform_edt(grid != 100)
    nonfree_dist = spec.resolution * distance_transform_edt(grid == 0)
    contacts, nearest = [], []

    def observe(plant):
        corners = plant._chassis_outline(0)
        samples = np.vstack([corners, (corners + np.roll(corners, -1, axis=0)) / 2])
        row, col = trial.track_import.world_to_pixel(spec, *samples.T)
        if (np.any(row < 0) or np.any(row >= mask.shape[0]) or np.any(col < 0)
                or np.any(col >= mask.shape[1]) or plant._contact_dt is not None):
            raise ValueError('outline outside map or nonstandard judge')
        nearest.append(float(occ_dist[row, col].min()))
        hit = np.unique(np.c_[row, col][mask[row, col]], axis=0)
        if bool(len(hit)) != (0 in plant._wall_contact()):
            raise ValueError('passive judge disagrees with gym')
        for y, x in hit:
            contacts.append(dict(image_rc=[int(y), int(x)], published_xy=[int(x),int(len(grid)-1-y)],
                published=int(grid[y,x]), neighbors=grid[y-1:y+2,x-1:x+2].tolist(),
                pgm=int(src[y,x]), obstacle_cells=int(np.count_nonzero(components==components[y,x])),
                nearest_nonfree_from_cell_m=float(nonfree_dist[y,x]),
                nearest_occupied_from_outline_m=nearest[-1]))

    offsets = record['offsets']
    race = trial.race
    race.rv.RACELINE_SOURCE = 'plan'
    stats, rows = race.run(str(out / 'nov09'), line, 'solo', seed=record['seed'],
        car='rb2', floor=trial.FLOOR, laps=trial.LAPS, timeout=75,
        overrides={'steering_law':'adaptive_pure_pursuit'},
        ego_law_factory=None if offsets is None else lambda law: trial.SectorLookahead(law, offsets),
        observe_step=observe)
    pos = np.vstack([xy[0], [[r['ego_x'], r['ego_y']] for r in rows]])
    digest = hashlib.sha256(pos.tobytes()).hexdigest()
    if (digest != record['position_sha256'] or not contacts or not rows[-1]['wall_contact']
            or stats['wall_contact_ticks'] != record['wall_contact']
            or any(r['active'] for r in rows)):
        raise ValueError('not a reproduction of the original run')
    keep = ('time', 'ego_progress', 'ego_speed', 'ego_command', 'steering',
            'desired_steering', 'safety', 'safety_detail', 'arc_clearance', 'profile_speed')
    approach = [{k: (None if isinstance(r[k], float) and not np.isfinite(r[k]) else r[k])
                 for k in keep} for r in rows[-32:]]
    for r in approach:
        r['command_time'] = r['time'] - race.rv.CONTROL_DT
    return dict(seed=record['seed'], arm=record['arm'], waypoint=int(np.argmin(np.linalg.norm(xy-pos[-1],axis=1))),
        hits=contacts, min_published_occupied_m=min(nearest),
        published_contact_ticks=int(np.count_nonzero(np.array(nearest)==0)),
        approach=approach, position_sha256=digest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path)
    p.add_argument('out',type=Path)
    args = p.parse_args()
    evidence = []
    for stage in ('train', 'judge'):
        for r in json.loads((args.out / (stage + '.json')).read_text()):
            if r['course'] == 'nov09' and r['wall_contact']:
                evidence.append(dict(stage=stage, **diagnose(r, args.out, args.data)))
    (args.out / 'wall-replay.json').write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
