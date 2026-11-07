#!/usr/bin/env python3
"""Replay a recorded map-racer run's opponent detections through OpponentGate.

pure_pursuit_node's map-subtraction detector, re-run offline on the run's real /scan and the particle filter's
/pf/viz/inferred_pose, against the map that run raced on. A numpy ray caster stands in for range_libc, so this
runs on any machine. Each forward-cone detection goes through OpponentGate with the shipped defaults (or the
overrides below), and is printed with its distance to the nearest mapped wall.

On a solo run every detection is false, so "0 confirmed" is the result to want. Car 2's 2026-11-06 runs:
164401, 175300, 180114 and 180341 all confirm none; 175300 confirms 70 with --min-wall-distance 0.

    python3 tools/run_analysis/replay_opponent_gate.py data/rb2/racerbot_runs/20261106-175300 \\
        data/rb2/racerbot_auto/20261106-164002-rebuilt [--min-wall-distance M] [--dump fixture.csv]

--dump writes the detections as the CSV test_opponent_gate.py reads
(src/pure_pursuit/test/data/phantom_20261106-175300.csv was made this way). Detector settings are
pure_pursuit.yaml's: every 4th beam, 0.4 m margin, 0.15-0.7 m wide, 5 m range, 0.3 m cluster gap, 60 deg cone.
"""

import argparse
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src' / 'pure_pursuit'), str(ROOT / 'tools')]
from pure_pursuit import racing_math  # noqa: E402
from pure_pursuit.occupancy_map import FREE, OccupancyMap  # noqa: E402
from pure_pursuit.opponent_gate import OpponentGate  # noqa: E402
from run_analysis.bagread import messages  # noqa: E402

BEAM_STEP, MARGIN, MIN_W, MAX_W, RANGE, GAP = 4, 0.4, 0.15, 0.7, 5.0, 0.3
HALF_FOV = math.radians(60.0) / 2.0


def expected_ranges(blocked, grid, lx, ly, yaw, angles, max_range=10.0, step=0.02):
    """Ray-march the map from the LiDAR: distance to the first non-free cell (off-map counts as wall)."""
    d = np.arange(0.0, max_range, step)
    px = lx + np.cos(yaw + angles)[:, None] * d
    py = ly + np.sin(yaw + angles)[:, None] * d
    c = np.floor((px - grid.origin_x) / grid.resolution).astype(int)
    r = np.floor((py - grid.origin_y) / grid.resolution).astype(int)
    inside = (c >= 0) & (c < blocked.shape[1]) & (r >= 0) & (r < blocked.shape[0])
    hit = np.ones_like(inside)
    hit[inside] = blocked[r[inside], c[inside]]
    return d[np.where(hit.any(axis=1), hit.argmax(axis=1), len(d) - 1)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', help='recorded run directory (with bag/)')
    parser.add_argument('map_dir', help='the racerbot_auto directory that run raced on (map.yaml, raceline_raw.csv)')
    parser.add_argument('--min-wall-distance', type=float, default=0.30)
    parser.add_argument('--dump', help='write the detections to this CSV')
    parser.add_argument('--quiet', action='store_true', help='print only the summary')
    args = parser.parse_args(argv)

    grid = OccupancyMap.from_yaml(str(Path(args.map_dir) / 'map.yaml'), inflate_cells=0)
    blocked = grid.grid != FREE
    field = grid.clearance_field()
    line = racing_math.load_xy_csv(str(Path(args.map_dir) / 'raceline_raw.csv'))
    seg = racing_math.compute_segment_lengths(line, closed=True)
    arc = racing_math.compute_cumulative_arc_length(seg)
    total = float(seg.sum())
    gate = OpponentGate(min_wall_distance=args.min_wall_distance)

    start, pf, rows, confirmed, scans = None, None, [], 0, 0
    for topic, t, msg in messages(args.run, ['/pf/viz/inferred_pose', '/scan']):
        start = t if start is None else start
        if topic == '/pf/viz/inferred_pose':
            q = msg.pose.orientation
            pf = (msg.pose.position.x, msg.pose.position.y, 2.0 * math.atan2(q.z, q.w))
            continue
        if pf is None:
            continue
        scans += 1
        lx, ly, yaw = pf  # the filter's pose is the LiDAR's pose here (its laser frame)
        ranges = np.asarray(msg.ranges, float)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        cand = racing_math.detect_dynamic_cluster(
            ranges[::BEAM_STEP], expected_ranges(blocked, grid, lx, ly, yaw, angles[::BEAM_STEP]),
            msg.angle_min, msg.angle_increment * BEAM_STEP, MARGIN, MIN_W, MAX_W, RANGE,
            range_min=msg.range_min, cluster_gap_threshold=GAP)
        if cand is None or abs(cand[3]) > HALF_FOV:
            continue
        _, _, cr, ca = cand
        wx, wy = lx + cr * math.cos(yaw + ca), ly + cr * math.sin(yaw + ca)
        i = int(np.argmin(np.hypot(line[:, 0] - wx, line[:, 1] - wy)))
        wall = float(grid.clearance_at(np.array([wx]), np.array([wy]), field)[0])
        verdict = gate.observe(t - start, wx, wy, float(arc[i]), total, wall_distance=wall)
        confirmed += verdict.confirmed
        rows.append((t - start, wx, wy, float(arc[i]), wall))
        if not args.quiet:
            print(f't={t - start:7.2f}  {cr:.2f} m at {math.degrees(ca):+.0f} deg -> ({wx:.2f}, {wy:.2f})  '
                  f'wall {wall:.2f} m  {"CONFIRMED" if verdict.confirmed else verdict.reason}')

    walls = np.array([r[4] for r in rows]) if rows else np.zeros(1)
    print(f'{Path(args.run).name}: {scans} scans, {len(rows)} forward detections, {confirmed} confirmed; '
          f'wall distance p50/p90/max {np.percentile(walls, 50):.2f}/{np.percentile(walls, 90):.2f}/'
          f'{walls.max():.2f} m')
    if args.dump:
        with open(args.dump, 'w') as handle:
            handle.write(f'# Forward-cone map-subtraction detections replayed from {Path(args.run).name} on map '
                         f'{Path(args.map_dir).name};\n# particle-filter poses + real scans, detector settings of '
                         f'pure_pursuit.yaml. tools/run_analysis/replay_opponent_gate.py.\n'
                         f'# total_length {total:.4f}\nt,x,y,arc_length,wall_distance\n')
            for row in rows:
                handle.write(','.join(f'{v:.4f}' for v in row) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
