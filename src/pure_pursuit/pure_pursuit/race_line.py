"""Build and compare racing lines against the same final-waypoint limits."""

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path

import numpy as np

from pure_pursuit import recorded_path
from pure_pursuit.min_time_line import (
    CarPhysics, cone_clearance, footprint_clearance, lap_time, optimize_min_time,
    speed_profile,
)
from pure_pursuit.occupancy_map import OccupancyMap
from pure_pursuit.racing_math import load_xy_csv
from pure_pursuit.raceline_optimizer import (
    optimize_minimum_curvature, refine_centerline, resample_closed_path,
    signed_curvature,
)


@dataclass(frozen=True)
class RaceLineConfig:
    physics: CarPhysics = CarPhysics()
    mode: str = 'fastest'
    spacing: float = 0.15
    optimize_spacing: float = 0.25
    wall_margin: float = 0.15
    cone_margin: float = 0.10
    emergency_stop_distance: float = 0.35
    body_floor: float = 0.08
    half_width: float = 0.155
    laser_x: float = 0.26
    max_track_width: float = 6.0
    min_feature_wavelength: float = 1.5
    curvature_margin: float = 1.0
    max_deviation: float = 0.35

    def __post_init__(self):
        if self.mode not in ('fastest', 'recorded', 'min_curvature', 'min_time'):
            raise ValueError(f'unknown mode: {self.mode}')
        for name in ('spacing', 'optimize_spacing', 'half_width', 'max_track_width',
                     'min_feature_wavelength', 'curvature_margin'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('wall_margin', 'cone_margin', 'emergency_stop_distance',
                     'body_floor', 'laser_x', 'max_deviation'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and non-negative')


@dataclass(frozen=True)
class Candidate:
    source: str
    xy: np.ndarray | None
    speed: np.ndarray | None
    est_lap_s: float
    min_cone_m: float
    min_body_m: float
    max_curvature: float
    ok: bool
    reason: str


@dataclass(frozen=True)
class RaceLine:
    ok: bool
    source: str | None
    xy: np.ndarray | None
    speed: np.ndarray | None
    est_lap_s: float
    candidates: tuple
    reason: str


def check_line(field_at, xy, cfg: RaceLineConfig) -> tuple[bool, float, float, float, str]:
    """Check the given waypoints, including the closing station."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 4 or not np.isfinite(xy).all():
        raise ValueError('need at least 4 finite xy points')
    if np.any(np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1) <= 0):
        raise ValueError('duplicate consecutive points')
    cone = float(np.min(cone_clearance(field_at, xy, laser_x=cfg.laser_x)))
    body = float(np.min(footprint_clearance(field_at, xy, half_width=cfg.half_width)))
    curvature = float(np.max(np.abs(signed_curvature(xy))))
    limits = (('forward cone', cone, cfg.emergency_stop_distance + cfg.cone_margin, True),
              ('body', body, cfg.body_floor, True),
              ('curvature', curvature, cfg.physics.curvature_limit + 0.02, False))
    failures = []
    for name, value, limit, minimum in limits:
        if not math.isfinite(value) or (value < limit if minimum else value > limit):
            failures.append(f'{name} {value:.4f}, limit {limit:.4f}, '
                            f'{"shortfall" if minimum else "excess"} '
                            f'{(limit - value if minimum else value - limit):.4f}')
    values = f'forward cone {cone:.4f} m, body {body:.4f} m, curvature {curvature:.4f}/m'
    return not failures, cone, body, curvature, '; '.join(failures) if failures else values


def _failed(source, reason):
    return Candidate(source, None, None, math.inf, math.nan, math.nan, math.nan, False, str(reason))


def _candidate(source, xy, field_at, cfg):
    ok, cone, body, curvature, reason = check_line(field_at, xy, cfg)
    speed = speed_profile(xy, cfg.physics)[0]
    return Candidate(source, xy, speed, lap_time(xy, cfg.physics), cone, body, curvature, ok, reason)


def build_race_line(grid: OccupancyMap, seed_xy, cfg: RaceLineConfig) -> RaceLine:
    field = grid.clearance_field()

    def field_at(x, y):
        return grid.clearance_at(x, y, field)

    prepared = None
    candidates = []
    try:
        prepared = recorded_path.prepare(
            seed_xy, spacing=cfg.spacing, max_steering_angle=cfg.physics.max_steering_angle,
            wheelbase=cfg.physics.wheelbase, min_feature_wavelength=cfg.min_feature_wavelength,
            curvature_margin=cfg.curvature_margin, max_deviation=cfg.max_deviation,
            clearance_fn=field_at, required_clearance=cfg.half_width + cfg.body_floor)
        candidates.append(_candidate('recorded', prepared.xy, field_at, cfg))
    except Exception as exc:
        candidates.append(_failed('recorded', exc))

    mc = None
    corridor_error = None
    try:
        center, left, right = refine_centerline(
            grid, prepared.xy if prepared is not None else seed_xy,
            cfg.optimize_spacing, cfg.max_track_width)
        width = np.asarray(left) + np.asarray(right)
        needed = 2 * (cfg.half_width + cfg.wall_margin)
        if np.any(width < needed):
            raise ValueError(f'corridor too narrow: {width.min():.3f} m, needs {needed:.3f} m')
        mc = optimize_minimum_curvature(
            center, left, right, vehicle_half_width=cfg.half_width,
            safety_margin=cfg.wall_margin, spacing=cfg.optimize_spacing)
        candidates.append(_candidate('min_curvature', resample_closed_path(mc['line'], cfg.spacing),
                                     field_at, cfg))
    except Exception as exc:
        corridor_error = str(exc)
        candidates.append(_failed('min_curvature', exc))

    try:
        if mc is None:
            raise ValueError(f'minimum-curvature corridor unavailable: {corridor_error}')
        for run in range(3):
            result = optimize_min_time(
                mc['line'], mc['remaining_left'], mc['remaining_right'], cfg.physics, field_at,
                half_width=cfg.half_width, wall_margin=cfg.wall_margin,
                emergency_stop_distance=cfg.emergency_stop_distance,
                cone_margin=cfg.cone_margin + 0.03 * run, body_floor=cfg.body_floor,
                laser_x=cfg.laser_x)
            candidate = _candidate('min_time', resample_closed_path(result.xy, cfg.spacing), field_at, cfg)
            if candidate.ok:
                break
        candidates.append(candidate)
    except Exception as exc:
        candidates.append(_failed('min_time', exc))

    eligible = [c for c in candidates if c.ok and (cfg.mode == 'fastest' or c.source == cfg.mode)]
    if eligible:
        best = min(eligible, key=lambda c: c.est_lap_s)
        return RaceLine(True, best.source, best.xy, best.speed, best.est_lap_s,
                        tuple(candidates), f'chosen {best.source}: {best.est_lap_s:.3f} s')
    reason = '; '.join(f'{c.source}: {c.reason}' for c in candidates)
    return RaceLine(False, None, None, None, math.inf, tuple(candidates), reason)


def write_race_line(out_dir, line: RaceLine, floor_test_speed: float = 1.5) -> dict:
    """Write the selected waypoints without rounding away the checked geometry."""
    if not line.ok:
        raise ValueError(f'cannot write a failed race line: {line.reason}')
    if not math.isfinite(floor_test_speed) or floor_test_speed < 0:
        raise ValueError('floor_test_speed must be finite and non-negative')
    xy, speed = np.asarray(line.xy, float), np.asarray(line.speed, float)
    if (xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 4 or speed.shape != (len(xy),)
            or not np.isfinite(xy).all() or not np.isfinite(speed).all() or np.any(speed < 0)):
        raise ValueError('invalid selected waypoints or speeds')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    floor_speed = np.minimum(speed, floor_test_speed)
    for name, velocities in (('raceline_profiled.csv', speed),
                             ('raceline_floor_test.csv', floor_speed)):
        np.savetxt(out / name, np.column_stack((xy, velocities)), delimiter=',',
                   header='x,y,speed', comments='', fmt='%.17g')

    def finite(value):
        return value if math.isfinite(value) else None

    report = {
        'choice': line.source, 'ok': line.ok, 'est_lap_s': finite(line.est_lap_s),
        'reason': line.reason,
        'candidates': [dict(source=c.source, ok=c.ok, est_lap_s=finite(c.est_lap_s),
                            min_cone_m=finite(c.min_cone_m), min_body_m=finite(c.min_body_m),
                            max_curvature=finite(c.max_curvature), reason=c.reason)
                       for c in line.candidates],
    }
    (out / 'race_line.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return {'raceline_profiled.csv': float(speed.max()),
            'raceline_floor_test.csv': float(floor_speed.max())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--mode', default='fastest',
                        choices=('fastest', 'recorded', 'min_curvature', 'min_time'))
    defaults = CarPhysics()
    for name in ('a_lat_left', 'a_lat_right', 'a_brake', 'a_accel', 'v_max', 'max_steering_angle'):
        parser.add_argument('--' + name.replace('_', '-'), type=float, default=getattr(defaults, name))
    parser.add_argument('--wall-margin', type=float, default=0.15)
    args = parser.parse_args(argv)
    physics = CarPhysics(a_lat_left=args.a_lat_left, a_lat_right=args.a_lat_right,
                         a_brake=args.a_brake, a_accel=args.a_accel, v_max=args.v_max,
                         max_steering_angle=args.max_steering_angle)
    cfg = RaceLineConfig(physics=physics, mode=args.mode, wall_margin=args.wall_margin)
    grid = OccupancyMap.from_yaml(args.run_dir / 'map.yaml', inflate_cells=0)
    seed = load_xy_csv(args.run_dir / 'raceline_raw.csv')
    line = build_race_line(grid, seed, cfg)
    for candidate in line.candidates:
        print(f'{candidate.source}: {"pass" if candidate.ok else "fail"}; '
              f'{candidate.est_lap_s:.3f} s; {candidate.reason}')
    print(f'choice: {line.source if line.ok else "none"}; {line.reason}')
    if not line.ok:
        return 1
    write_race_line(args.run_dir, line)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
