#!/usr/bin/env python3
"""Offline dashboard apexes -> a new, checked pure-pursuit run directory."""
import argparse
import csv
import datetime
import json
import math
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np
import yaml

from pure_pursuit import racing_math, raceline_optimizer as optimizer, run_commands
from pure_pursuit.occupancy_map import OccupancyMap
from pure_pursuit.map_distance import MapDistance
import hashlib


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            digest.update(block)
    return digest.hexdigest()


def default_config():
    # Installed config, including in non-symlink colcon builds. Source fallback
    # keeps this math-only module usable in the framework-independent tests.
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory('pure_pursuit')) / 'config/auto_map_race.yaml'
    except (ImportError, LookupError):
        return Path(__file__).resolve().parents[1] / 'config/auto_map_race.yaml'


CONFIG = default_config()
TOLERANCE = 0.02


def fingerprint(xy):
    """Match apex.js, including JS's ties-toward-positive-infinity rounding."""
    h = 0x811c9dc5
    for x, y in xy:
        for c in f'{math.floor(x * 1000 + .5)},{math.floor(y * 1000 + .5)};':
            h = ((h ^ ord(c)) * 0x01000193) & 0xffffffff
    return dict(points=len(xy), closed=True, fnv1a=f'{h:08x}')


def read_inputs(run, apex_file):
    rows = []
    with (run / 'raceline_profiled.csv').open() as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith('#') or row == ['x', 'y', 'speed']:
                continue
            rows.append([float(v) for v in row])
    table = np.asarray(rows)
    if (table.ndim != 2 or table.shape[1] != 3 or len(table) < 4
            or not np.isfinite(table).all() or np.any(table[:, 2] < 0)):
        raise ValueError('CSV needs at least four finite x,y,speed rows')
    xy = table[:, :2]
    doc = json.loads(apex_file.read_text())
    if not isinstance(doc, dict) or (doc.get('format'), doc.get('version'), doc.get('frame')) != (
            'racerbot-apexes', 1, 'map'):
        raise ValueError('expected map-frame racerbot-apexes version 1')
    if doc.get('line') != fingerprint(xy):
        raise ValueError('line fingerprint mismatch: export apexes against this exact CSV')
    apexes = doc.get('apexes')
    if not isinstance(apexes, list) or not apexes:
        raise ValueError('at least one associated apex is required')
    for a in apexes:
        if not isinstance(a, dict):
            raise ValueError('invalid apex')
        i = a.get('waypoint_index')
        if type(i) is not int or not 0 <= i < len(xy):
            raise ValueError('apex waypoint_index must reference the source CSV')
        for key in ('x', 'y', 's_m', 'distance_to_line_m'):
            v = a.get(key)
            if type(v) not in (int, float) or not math.isfinite(v):
                raise ValueError(f'apex {key} must be finite')
        if i != int(np.argmin(np.linalg.norm(xy - [a['x'], a['y']], axis=1))):
            raise ValueError('apex waypoint association does not match source line')
    return xy, doc


def validate_line(xy, pins, occ, params):
    if not np.isfinite(xy).all():
        raise ValueError('non-finite optimized line')
    curvature = racing_math.estimate_path_curvature(xy, closed=True)
    limit = optimizer.curvature_limit(params['profile_max_steering_angle'],
                                     params['profile_wheelbase'])
    if not np.isfinite(curvature).all() or np.max(np.abs(curvature)) > limit:
        raise ValueError('edited line exceeds steering curvature limit')
    segments = np.roll(xy, -1, axis=0) - xy
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths < 1e-6):
        raise ValueError('duplicate line points')
    distances = []
    for pin in pins:
        t = np.clip(np.sum((pin - xy) * segments, axis=1) / lengths**2, 0, 1)
        distances.append(float(np.min(np.linalg.norm(xy + t[:, None] * segments - pin, axis=1))))
    if max(distances) > TOLERANCE:
        raise ValueError('edited line misses an apex tolerance')
    # Use ROS occupancy semantics: unknown is blocked. Pad to make map edges walls.
    pixels = np.where(np.flipud(occ.grid) == 0, 255, 0).astype(np.uint8)
    field = MapDistance(np.pad(pixels, 1), occ.resolution,
                        (occ.origin_x - occ.resolution, occ.origin_y - occ.resolution))
    body = np.array([[-.2675, -.15], [.2675, -.15], [.2675, .15], [-.2675, .15]])
    body[:, 0] += .18
    minimum = .5
    # Sample every <=1cm along each segment, allowing another half-step of
    # clearance for translation between samples (fixed segment heading).
    for p, delta, length in zip(xy, segments, lengths):
        yaw = math.atan2(delta[1], delta[0])
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        for t in np.linspace(0, 1, math.ceil(length / .01) + 1):
            corners = body @ rotation.T + p + t * delta
            if (np.any(corners < [occ.origin_x, occ.origin_y]) or
                    np.any(corners > [occ.origin_x + occ.width * occ.resolution,
                                      occ.origin_y + occ.height * occ.resolution])):
                raise ValueError('chassis leaves map bounds')
            minimum = min(minimum, field.body_clearance(corners))
    if minimum < .105:
        raise ValueError(f'chassis clearance {minimum:.3f}m is below 0.105m')
    return dict(min_body_clearance_m=minimum, max_curvature=float(np.max(np.abs(curvature))),
                curvature_limit=limit, apex_distances_m=distances)


def _velocity_profile(line, params):
    return racing_math.compute_velocity_profile(
        racing_math.compute_segment_lengths(line, closed=True),
        racing_math.estimate_path_curvature(line, closed=True),
        v_max=params['profile_max_speed'], v_min=params['profile_min_speed'],
        a_lat_max=params['profile_max_lateral_accel'], a_accel_max=params['profile_max_accel'],
        a_brake_max=params['profile_max_brake'], closed=True,
        smoothing_passes=params['profile_smoothing_passes'])


def edit(run, apex_file, out, config=CONFIG):
    run, apex_file, out = Path(run).resolve(), Path(apex_file).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('output directory already exists; never overwrite a run')
    xy, doc = read_inputs(run, apex_file)
    if run_commands.read_start_pose(run) is None:
        raise ValueError('a valid start_pose.yaml is required for direct pure-pursuit launch')
    from racerbot_car.layers import effective_values, layered
    params = effective_values(layered(str(config), 'auto_map_race.yaml'), 'auto_map_race_node')
    occ = OccupancyMap.from_yaml(str(run / 'map.yaml'))
    pins = np.array([[a['x'], a['y']] for a in doc['apexes']])
    center, left, right = optimizer.refine_centerline(
        occ, xy, params['optimize_spacing'], params['optimize_max_track_width'],
        params['optimize_centerline_passes'], params['optimize_centerline_smoothing'])
    result = optimizer.optimize_minimum_curvature(
        center, left, right, params['optimize_car_width'] / 2,
        # A rear-axle point needs room for the forward chassis corner,
        # not just half-width. Never relax the configured keep-out.
        max(params['optimize_safety_margin'],
            math.hypot(.4475, .15) + .105 - params['optimize_car_width'] / 2),
        params['optimize_spacing'],
        iterations=params['optimize_iterations'], smoothing_weight=params['optimize_smoothing_weight'],
        trust_region=params['optimize_trust_region'], apex_points=pins,
        apex_tolerance=TOLERANCE / 2)
    line = optimizer.resample_closed_path(result['line'], params['optimize_output_spacing'])
    checks = validate_line(line, pins, occ, params)
    speed = _velocity_profile(line, params)
    if not np.isfinite(speed).all() or np.any(np.asarray(speed) <= 0):
        raise ValueError('invalid velocity profile')
    meta = yaml.safe_load((run / 'map.yaml').read_text())
    image = (run / meta['image']).resolve()
    inputs = [run / 'raceline_profiled.csv', run / 'map.yaml', image,
              run / 'start_pose.yaml', apex_file, Path(config)]
    report = dict(inputs_sha256={str(p): sha256_file(p) for p in inputs},
                  apexes=doc, checks=checks, parameters=params,
                  validation='static geometry only; driven simulation still required')
    # Exclusive mkdir protects every original, even when --out names the source.
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(image, out / ('map' + image.suffix))
    meta['image'] = 'map' + image.suffix
    (out / 'map.yaml').write_text(yaml.safe_dump(meta))
    shutil.copyfile(run / 'start_pose.yaml', out / 'start_pose.yaml')
    np.savetxt(out / 'raceline_profiled.csv', np.column_stack((line, speed)),
               delimiter=',', header='x,y,speed', fmt='%.10f')
    (out / 'apex_edit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    command = ('ros2 launch racerbot_launch race_launch.py '
               + shlex.quote('run_dir:=' + str(out)) + ' steering_law:=pure_pursuit')
    (out / 'commands.txt').write_text(command + '\n')
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--apexes', required=True, type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    out = args.out or args.run / ('apex_edit_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    try:
        print(edit(args.run, args.apexes, out))
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.exit(2, f'Apex edit refused: {exc}\n')


if __name__ == '__main__':
    main()
