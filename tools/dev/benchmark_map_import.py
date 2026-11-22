#!/usr/bin/env python3
"""Compare two map importers with unchanged headless rb2 controller runners.

Run from the workspace root after tools/f1tenth_sim/setup.sh, for example:
    PYTHONPATH=.sim/python:.sim/f1tenth_gym:src/gap_follow:src/pure_pursuit:tools/f1tenth_sim:tools \
      python3 tools/dev/benchmark_map_import.py --car rb2 --output ~/work/nov9-import

The old importer is loaded verbatim from git, not reimplemented. Both imports
use the recorded map for the same centreline. Each runner constructs a fresh
plant and controller per seed. Instrumentation observes resets/steps without
altering commands, observations, collision response or termination conditions.
Contact verdicts use independent rectangle/cell separating-axis geometry, not
Gym's collision flag. Coordinates are world metres, image row zero is at top.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
IMPORTER = 'tools/f1tenth_sim/track_import.py'
MAP = 'data/rb2/racerbot_auto/20261109-185146-rebuilt/map.yaml'
LINE_MAP = 'data/rb2/racerbot_auto/20261109-185306/map.yaml'
CANDIDATES = np.array([[4.113, 4.205], [.513, -.795]])


@dataclass
class GridGeometry:
    free: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float

    def pixel(self, xy):
        col = math.floor((xy[0] - self.origin_x) / self.resolution)
        row = self.free.shape[0] - 1 - math.floor((xy[1] - self.origin_y) / self.resolution)
        return row, col

    def is_free(self, xy):
        row, col = self.pixel(xy)
        h, w = self.free.shape
        return 0 <= row < h and 0 <= col < w and bool(self.free[row, col])

    def centers(self, cells):
        cells = np.asarray(cells, dtype=int).reshape(-1, 2)
        return np.column_stack((
            self.origin_x + (cells[:, 1] + .5) * self.resolution,
            self.origin_y + (self.free.shape[0] - cells[:, 0] - .5) * self.resolution))

    def contacts(self, corners):
        """Occupied cell indices overlapping a closed chassis rectangle (SAT)."""
        corners = np.asarray(corners, dtype=float)
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        top, left = self.pixel([lo[0], hi[1]])
        bottom, right = self.pixel([hi[0], lo[1]])
        h, w = self.free.shape
        # Include adjacent cells when a rectangle lies exactly on a cell edge.
        top, left = max(0, top - 1), max(0, left - 1)
        bottom, right = min(h - 1, bottom + 1), min(w - 1, right + 1)
        if top > bottom or left > right:
            return np.empty((0, 2), dtype=int)
        rr, cc = np.nonzero(~self.free[top:bottom + 1, left:right + 1])
        cells = np.column_stack((rr + top, cc + left))
        centers = self.centers(cells)
        half = self.resolution / 2
        keep = np.ones(len(cells), dtype=bool)
        axes = [np.array([1., 0.]), np.array([0., 1.])]
        for edge in (corners[1] - corners[0], corners[3] - corners[0]):
            axes.append(edge / np.linalg.norm(edge))
        for axis in axes:
            body = corners @ axis
            cell = centers @ axis
            radius = half * np.abs(axis).sum()
            keep &= (cell + radius >= body.min()) & (cell - radius <= body.max())
        return cells[keep]


class EpisodeAudit:
    def __init__(self, active, old):
        self.active, self.old = active, old
        self.poses = []
        self.contacts = []

    def record(self, time_s, pose, corners, plant_contact):
        cells = self.active.contacts(corners)
        centers = self.active.centers(cells)
        outside = not self.old.is_free(pose[:2])
        self.poses.append([float(time_s), float(pose[0]), float(pose[1]),
                           float(pose[4]), outside])
        h, w = self.active.free.shape
        off_map = any(not (0 <= r < h and 0 <= c < w)
                      for r, c in (self.active.pixel(xy) for xy in corners))
        if len(cells) or off_map or plant_contact:
            near_cells = (np.linalg.norm(centers[:, None, :] - CANDIDATES, axis=2) <= .15).any(axis=0)
            self.contacts.append(dict(
                time_s=float(time_s), pose_xy=np.asarray(pose[:2]).tolist(),
                occupied_cells_rc=cells.tolist(), occupied_cell_centers_xy=centers.tolist(),
                near_candidate_cells=near_cells.tolist(),
                pose_near_candidates=(np.linalg.norm(np.asarray(pose[:2]) - CANDIDATES, axis=1) <= .15).tolist(),
                off_map=off_map, plant_wall_contact=bool(plant_contact)))

    def summary(self):
        outside = [p for p in self.poses if p[4]]
        independent = [c for c in self.contacts if c['occupied_cells_rc'] or c['off_map']]
        return dict(
            audited_poses=len(self.poses),
            independent_contact_ticks=len(independent),
            independent_collision=bool(independent),
            near_candidate_contact_ticks=[sum(c['near_candidate_cells'][i] for c in independent) for i in range(2)],
            outside_old_region_poses=len(outside),
            first_outside_old_region=outside[0] if outside else None,
            first_contact=independent[0] if independent else None)


@contextmanager
def instrument_factory(module, audit):
    factory = module.make_plant

    def observed_factory(*args, **kwargs):
        plant = factory(*args, **kwargs)
        reset, step = plant.reset, plant.step

        def record(result):
            audit.record(result[1].get('sim_time', 0.) if len(result) == 2 else result[4]['sim_time'],
                         plant.truth(0).copy(), plant._chassis_outline(0), bool(plant._wall_contact()))
            return result

        def observed_reset(*args, **kwargs):
            return record(reset(*args, **kwargs))

        def observed_step(*args, **kwargs):
            return record(step(*args, **kwargs))

        plant.reset, plant.step = observed_reset, observed_step
        return plant

    module.make_plant = observed_factory
    try:
        yield
    finally:
        module.make_plant = factory


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def load_importer(source, name):
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / IMPORTER)
    sys.modules[name] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return number


def map_hashes(*yaml_paths):
    hashes = {}
    for path in yaml_paths:
        path = Path(path).expanduser().absolute()
        metadata = yaml.safe_load(path.read_text())
        image = metadata.get('image') if isinstance(metadata, dict) else None
        if not isinstance(image, str) or not image.strip():
            raise ValueError(f'{path}: image must be a nonempty path')
        for source in (path, (path.parent / image).absolute()):
            key = str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source)
            hashes[key] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--car', required=True, choices=['rb1', 'rb2'])
    ap.add_argument('--before-ref', default='5f55aad^')
    ap.add_argument('--map-yaml', type=Path, default=ROOT / MAP)
    ap.add_argument('--centerline-from', type=Path, default=ROOT / LINE_MAP)
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(12345, 12350)))
    ap.add_argument('--laps', type=int, default=3)
    ap.add_argument('--timeout', type=positive, default=180.)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.laps < 1 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        ap.error('laps must be positive and seeds must be unique nonnegative integers')
    args.map_yaml = args.map_yaml.expanduser().absolute()
    args.centerline_from = args.centerline_from.expanduser().absolute()
    try:
        hashes = map_hashes(args.map_yaml, args.centerline_from)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        ap.error(str(exc))
    output = args.output.expanduser().resolve()
    sys.path[:0] = [str(ROOT / 'tools/f1tenth_sim'), str(ROOT / 'tools')]
    import eval_gap
    import run_validation as rv
    from gap_mirror import node_params
    from sim_fidelity import calibration_for_car

    rv.select_profile('car', args.car)
    before_sha = git('rev-parse', args.before_ref)
    sources = dict(before=git('show', f'{before_sha}:{IMPORTER}'), after=(ROOT / IMPORTER).read_text())
    modules = {label: load_importer(source, f'benchmark_import_{label}') for label, source in sources.items()}
    # Fixed even if an importer change affects centreline classification.
    fixed_line_map = modules['before'].load_map(args.centerline_from)
    tracks, paths = {}, {}
    for label, module in modules.items():
        spec = module.load_map(args.map_yaml)
        tracks[label] = module.extract_track(spec, centerline_from=fixed_line_map)
    if not np.array_equal(tracks['before'].centerline, tracks['after'].centerline):
        raise RuntimeError('comparison requires identical centrelines')
    output.mkdir(parents=True, exist_ok=False)
    for label, module in modules.items():
        paths[label] = module.write_gym_track(tracks[label], output / label, 'nov9').with_suffix('')
    spec = tracks['before'].spec
    old = GridGeometry(tracks['before'].region, spec.resolution, spec.origin_x, spec.origin_y)
    calibration = calibration_for_car(args.car)
    params = node_params(car_id=args.car)
    metadata = dict(car=args.car, before_ref=before_sha, after_ref=git('rev-parse', 'HEAD'),
                    git_status=git('status', '--short'), seeds=args.seeds, laps=args.laps,
                    timeout_s=args.timeout, calibration=asdict(calibration), gap_params=params,
                    pure_params=rv.PURE_PURSUIT_PARAMS,
                    hashes={**hashes, **{str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (ROOT / 'tools/f1tenth_sim/eval_gap.py', ROOT / 'tools/f1tenth_sim/gap_mirror.py',
                                      ROOT / 'tools/f1tenth_sim/run_validation.py', ROOT / 'tools/f1tenth_sim/sim_fidelity/plant.py')}},
                    importer_hashes={k: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()})
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    results = []
    for label, track in tracks.items():
        active = GridGeometry(track.region, spec.resolution, spec.origin_x, spec.origin_y)
        for controller in ('gap_follow', 'pure_pursuit'):
            for seed in args.seeds:
                audit = EpisodeAudit(active, old)
                module = eval_gap if controller == 'gap_follow' else rv
                with instrument_factory(module, audit):
                    if controller == 'gap_follow':
                        result = eval_gap.run_track(str(paths[label]), params.copy(), seed, args.timeout,
                                                    car_id=args.car, calibration=calibration, max_laps=args.laps)
                    else:
                        result = rv.run_pure_solo(str(paths[label]), seed, args.timeout,
                                                  car_id=args.car, calibration=calibration, laps=args.laps)
                row = {**result, 'importer': label, 'controller': controller, 'seed': seed, **audit.summary()}
                name = f'{label}-{controller}-{seed}'
                np.savez_compressed(output / f'{name}-poses.npz',
                                    columns=['time_s', 'x', 'y', 'yaw', 'outside_old_region'], poses=audit.poses)
                (output / f'{name}-contacts.json').write_text(json.dumps(audit.contacts, indent=2) + '\n')
                results.append(row)
                (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
                print(json.dumps(row), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
