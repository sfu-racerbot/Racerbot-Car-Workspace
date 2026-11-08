#!/usr/bin/env python3
"""Replay pure_pursuit's Tier 1 trigger over recorded runs: the old cone rule against the
arc-aware check (2026-11-07). Counts the scans on which each rule would stop the car, and
classifies the scans where only the new check trips, so that a noisy return cannot pass for a
real wall.

    rb python3 tools/run_analysis/tier1_replay.py <run> [<run> ...] [--car rb2] [--map-dir DIR]

<run> is a racerbot_runs directory. Its bag is read and nothing is written. Geometry, thresholds
and the window come from the car's effective pure_pursuit config, so the replay uses the numbers
the node uses.

Steering. The bag records the command the car was given (/ackermann_cmd), not the controller's
pre-override steering. The basis is the last command before a scan (the rack's angle), and the
desired steering is the first command after it (the controller's next request). This is a proxy,
and the output says so.

Tier 0 scans (body contact within emergency_stop_clearance) are excluded from both counts: both
rules leave them to the same code path.

Classification of a new-only trip, by the return that binds it:
  mapped wall -- the run has a pose (/slam_pose) and --map-dir is given, and the return lies within
                 0.10 m of an occupied cell of that map.
  persistent  -- the same return (within 1 deg and 0.15 m) is in at least PERSIST_SCANS
                 consecutive scans, this one included (75 ms at 40 Hz). A surface the car drives
                 past enters and leaves the view over several scans, so it counts.
  transient   -- it is in one or two consecutive scans only.
  isolated    -- no other return within two beams and 0.05 m in its own scan (a lone beam).
Transient and isolated returns are the noise-like classes.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src' / 'pure_pursuit'))
sys.path.insert(0, str(REPO / 'src' / 'racerbot_car'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pure_pursuit import racing_math  # noqa: E402
from racerbot_car.layers import effective_config  # noqa: E402
import bagread  # noqa: E402

WALL_RADIUS_M = 0.10   # a return this close to an occupied cell is a mapped wall
PERSIST_DEG = 1.0      # a persistent return is within this bearing ...
PERSIST_M = 0.15       # ... and this range of its neighbour scan
ISOLATED_M = 0.05      # a lone beam: nothing within two beams and this range
PERSIST_SCANS = 3      # a return in this many consecutive scans is structure, not a spike


def load_geometry(car):
    node = effective_config('pure_pursuit.yaml', 'pure_pursuit_node', car)
    return {
        'car_width': node['car_width'], 'car_length': node['car_length'],
        'wheelbase': node['wheelbase'], 'lx': node['laser_offset_x'], 'ly': node['laser_offset_y'],
        'horizon': node['emergency_arc_length'], 'arc_trip': node['emergency_arc_clearance'],
        'backstop': node['emergency_hard_stop_distance'], 'old_stop': node['emergency_stop_distance'],
        'contact': node['emergency_stop_clearance'], 'body_fov': node['body_clearance_fov_deg'],
        'cone_fov': node['safety_fov_deg'],
    }


def fov_indices(amin, amax, ainc, n, fov_deg):
    """The node's _fov_indices: beam bounds of a forward window of fov_deg, centred ahead."""
    half = math.radians(fov_deg) / 2.0
    lo = int((max(amin, -half) - amin) / ainc)
    hi = int((min(amax, half) - amin) / ainc)
    return max(0, min(lo, n - 1)), max(0, min(hi, n - 1))


def load(run):
    scans, cmds, poses = [], [], []
    for topic, t, msg in bagread.messages(run, ['/scan', '/ackermann_cmd', '/slam_pose']):
        if topic == '/scan':
            scans.append((t, np.asarray(msg.ranges, dtype=np.float64), msg.angle_min,
                          msg.angle_max, msg.angle_increment, msg.range_min))
        elif topic == '/ackermann_cmd':
            cmds.append((t, msg.drive.steering_angle))
        else:
            q = msg.pose.orientation
            poses.append((t, msg.pose.position.x, msg.pose.position.y, 2.0 * math.atan2(q.z, q.w)))
    scans.sort(key=lambda s: s[0])
    cmds.sort(key=lambda c: c[0])
    poses.sort(key=lambda p: p[0])
    return scans, cmds, poses


def steering_at(cmd_t, cmd_s, t):
    """(basis, desired): the last command at or before t, and the first one after it."""
    if not len(cmd_t):
        return 0.0, 0.0
    k = int(np.searchsorted(cmd_t, t, side='right'))
    basis = float(cmd_s[k - 1]) if k >= 1 else float(cmd_s[0])
    desired = float(cmd_s[k]) if k < len(cmd_s) else basis
    return basis, desired


def evaluate(scans, cmds, g):
    cmd_t = np.array([c[0] for c in cmds])
    cmd_s = np.array([c[1] for c in cmds])
    rows, boundary_key, boundary = [], None, None
    for index, (t, ranges, amin, amax, ainc, rmin) in enumerate(scans):
        n = len(ranges)
        angles = amin + np.arange(n) * ainc
        key = (n, amin, ainc)
        if key != boundary_key:
            boundary = racing_math.vehicle_boundary_distances(
                angles, g['car_width'], g['car_length'], g['wheelbase'], g['lx'], g['ly'])
            boundary_key = key
        valid = np.isfinite(ranges) & (ranges > 0.0) & (ranges >= rmin)
        lo, hi = fov_indices(amin, amax, ainc, n, g['body_fov'])
        body = racing_math.minimum_footprint_clearance(
            ranges[lo:hi + 1], valid[lo:hi + 1], boundary[lo:hi + 1])
        if body <= g['contact']:
            continue  # Tier 0: both rules leave this to the same code path
        basis, desired = steering_at(cmd_t, cmd_s, t)
        arc = racing_math.tier1_arc_clearance(
            ranges[lo:hi + 1], rmin, angles[lo:hi + 1], (basis, desired),
            g['car_width'] / 2.0, g['car_length'], g['wheelbase'], g['lx'], g['ly'], g['horizon'])
        clo, chi = fov_indices(amin, amax, ainc, n, g['cone_fov'])
        cone_valid = valid[clo:chi + 1]
        cone = float(np.min(ranges[clo:chi + 1][cone_valid])) if cone_valid.any() else math.inf
        arc_trip = arc < g['arc_trip']
        rows.append({
            'index': index, 't': t, 'body': body, 'arc': arc, 'cone': cone,
            'basis': basis, 'desired': desired,
            'old': cone < g['old_stop'],
            'new': arc_trip or cone < g['backstop'],
            'arc_trip': arc_trip, 'lo': lo, 'hi': hi, 'clo': clo, 'chi': chi,
        })
    return rows


def binding_return(scan, row, g):
    """The return that binds a new trip. For an arc trip it is the window beam whose own swept
    free distance is least; for a backstop trip it is the nearest valid return in the cone.
    Returns (bearing_deg, range_m, free_m, kind, beam_index), or None."""
    t, ranges, amin, amax, ainc, rmin = scan
    n = len(ranges)
    angles = amin + np.arange(n) * ainc
    valid = np.isfinite(ranges) & (ranges > 0.0) & (ranges >= rmin)
    if row['arc_trip']:
        best = None
        for k in range(row['lo'], row['hi'] + 1):
            if not valid[k]:
                continue
            free = racing_math.tier1_arc_clearance(
                ranges[k:k + 1], rmin, angles[k:k + 1], (row['basis'], row['desired']),
                g['car_width'] / 2.0, g['car_length'], g['wheelbase'], g['lx'], g['ly'],
                g['horizon'])
            if best is None or free < best[2]:
                best = (k, float(ranges[k]), float(free))
        kind = 'arc'
    else:
        candidates = [k for k in range(row['clo'], row['chi'] + 1) if valid[k]]
        if not candidates:
            return None
        k = min(candidates, key=lambda j: ranges[j])
        best = (k, float(ranges[k]), float(ranges[k]))
        kind = 'backstop'
    if best is None:
        return None
    k, rng, free = best
    return math.degrees(angles[k]), rng, free, kind, k


def isolated(ranges, valid, k):
    lo, hi = max(0, k - 2), min(len(ranges) - 1, k + 2)
    return not any(j != k and valid[j] and abs(ranges[j] - ranges[k]) <= ISOLATED_M
                   for j in range(lo, hi + 1))


def neighbour_has(scan, bearing_rad, rng):
    """Whether this scan has a valid return within PERSIST_DEG and PERSIST_M of a return."""
    _, ranges, amin, _, ainc, rmin = scan
    n = len(ranges)
    k = int(round((bearing_rad - amin) / ainc))
    half = int(round(math.radians(PERSIST_DEG) / ainc))
    for j in range(max(0, k - half), min(n - 1, k + half) + 1):
        r = ranges[j]
        if np.isfinite(r) and r > 0.0 and r >= rmin and abs(r - rng) <= PERSIST_M:
            return True
    return False


def run_length(scans, index, bearing_rad, rng, limit=PERSIST_SCANS * 4):
    """Consecutive scans around ``index``, this one included, with a matching return."""
    length = 1
    for step in (-1, 1):
        j = index + step
        taken = 0
        while 0 <= j < len(scans) and taken < limit and neighbour_has(scans[j], bearing_rad, rng):
            length += 1
            taken += 1
            j += step
    return length


def read_pgm(path):
    with open(path, 'rb') as handle:
        assert handle.readline().strip() == b'P5', 'not a binary PGM'
        tokens = []
        while len(tokens) < 3:
            line = handle.readline()
            if not line.startswith(b'#'):
                tokens += line.split()
        width, height, maxval = (int(x) for x in tokens[:3])
        data = np.frombuffer(handle.read(width * height), dtype=np.uint8)
    return data.reshape(height, width).astype(np.float64), maxval


def load_map(map_dir):
    meta = {}
    for line in (Path(map_dir) / 'map.yaml').read_text().splitlines():
        if ':' in line and not line.strip().startswith('#'):
            key, value = line.split(':', 1)
            meta[key.strip()] = value.strip()
    image, maxval = read_pgm(Path(map_dir) / meta['image'])
    if meta.get('negate', '0') != '0' or meta.get('mode', 'trinary') != 'trinary':
        raise SystemExit('only trinary, non-negated maps are read here')
    occupancy = (maxval - image) / maxval  # trinary: dark is occupied
    origin = [float(v) for v in meta['origin'].strip('[]').split(',')[:2]]
    return {
        'occupied': occupancy > float(meta['occupied_thresh']),
        'res': float(meta['resolution']), 'ox': origin[0], 'oy': origin[1],
    }


def on_occupied(grid, x, y, radius=WALL_RADIUS_M):
    res = grid['res']
    rows, cols = grid['occupied'].shape
    reach = int(math.ceil(radius / res))
    cx = int(math.floor((x - grid['ox']) / res))
    cy = rows - 1 - int(math.floor((y - grid['oy']) / res))
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if dx * dx + dy * dy > (radius / res) ** 2 + 1e-9:
                continue
            col, row = cx + dx, cy + dy
            if 0 <= row < rows and 0 <= col < cols and grid['occupied'][row, col]:
                return True
    return False


def pose_at(poses, t, tolerance=0.2):
    if not poses:
        return None
    times = np.array([p[0] for p in poses])
    k = int(np.clip(np.searchsorted(times, t), 1, len(poses) - 1))
    if abs(times[k - 1] - t) < abs(times[k] - t):
        k -= 1
    return poses[k] if abs(times[k] - t) <= tolerance else None


def classify(scans, poses, grid, row, bound, g):
    bearing_deg, rng, free, kind, k = bound
    scan = scans[row['index']]
    if grid is not None:
        pose = pose_at(poses, row['t'])
        if pose is not None:
            _, px, py, yaw = pose
            bearing = math.radians(bearing_deg)
            bx = g['lx'] + rng * math.cos(bearing)   # the return in base_link
            by = g['ly'] + rng * math.sin(bearing)
            mx = px + math.cos(yaw) * bx - math.sin(yaw) * by
            my = py + math.sin(yaw) * bx + math.cos(yaw) * by
            if on_occupied(grid, mx, my):
                return 'mapped wall', kind, free
    bearing_rad = math.radians(bearing_deg)
    ranges = scan[1]
    valid = np.isfinite(ranges) & (ranges > 0.0) & (ranges >= scan[5])
    if isolated(ranges, valid, k):
        return 'isolated', kind, free
    if run_length(scans, row['index'], bearing_rad, rng) >= PERSIST_SCANS:
        return 'persistent', kind, free
    return 'transient', kind, free


def replay(run, g, grid, show):
    scans, cmds, poses = load(run)
    rows = evaluate(scans, cmds, g)
    old_rows = [r for r in rows if r['old']]
    new_rows = [r for r in rows if r['new']]
    new_only = [r for r in rows if r['new'] and not r['old']]
    old_only = [r for r in rows if r['old'] and not r['new']]
    print(f'RUN {run}')
    print(f'  scans {len(scans)}, evaluated (not Tier 0) {len(rows)}, poses {len(poses)}, '
          f'commands {len(cmds)}, map {"yes" if grid is not None else "no"}')
    print(f'  old cone rule (< {g["old_stop"]:.2f} m raw in +/-{g["cone_fov"] / 2:.0f} deg): '
          f'{len(old_rows)}')
    print(f'  new arc-aware Tier 1 (arc < {g["arc_trip"]:.2f} m, or cone < {g["backstop"]:.2f} m): '
          f'{len(new_rows)}  (arc {sum(r["arc_trip"] for r in rows)}, backstop '
          f'{sum(r["new"] and not r["arc_trip"] for r in rows)} alone)')
    print(f'  both {len(old_rows) - len(old_only)}, new only {len(new_only)}, old only {len(old_only)}')
    classes = Counter()
    listed = []
    for r in new_only:
        bound = binding_return(scans[r['index']], r, g)
        if bound is None:
            classes['no valid return'] += 1
            continue
        label, kind, free = classify(scans, poses, grid, r, bound, g)
        classes[label] += 1
        if len(listed) < show:
            listed.append((r['t'], kind, bound[0], bound[1], free, label, r['basis'], r['desired']))
    print('  new-only classes: ' + (', '.join(f'{k} {v}' for k, v in sorted(classes.items()))
                                    or 'none'))
    for t, kind, bearing, rng, free, label, basis, desired in listed:
        print(f'    t={t:.2f} {kind:8s} bearing {bearing:+7.2f} deg range {rng:.3f} m '
              f'free {free:.3f} m  {label}  (basis {basis:+.3f}, desired {desired:+.3f})')
    if old_only:
        cones = sorted(r['cone'] for r in old_only)
        print(f'  old-only cone distances: min {cones[0]:.3f} m, median '
              f'{cones[len(cones) // 2]:.3f} m, max {cones[-1]:.3f} m; arc clear on all of them')
    return len(new_only), dict(classes)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('runs', nargs='+', help='racerbot_runs directories')
    parser.add_argument('--car', default='rb2')
    parser.add_argument('--map-dir', default=None,
                        help='the run\'s rebuilt map (map.yaml, map.pgm); used only with a pose')
    parser.add_argument('--show', type=int, default=6, help='new-only trips to list per run')
    args = parser.parse_args()
    g = load_geometry(args.car)
    grid = load_map(args.map_dir) if args.map_dir else None
    for run in args.runs:
        replay(run, g, grid, args.show)


if __name__ == '__main__':
    main()
