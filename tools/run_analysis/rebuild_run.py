#!/usr/bin/env python3
"""Turn a recorded mapping run into a folder race_launch.py can use, or say why not.

    python3 tools/run_analysis/rebuild_run.py data/<car>/racerbot_runs/<ts> [--out-root DIR] [--max-speed V]

map_course runs this after every run, so a run stopped early still keeps what it mapped. It takes the last
/map in the bag, matches every scan against that map, and judges the map from the result. A line recorded from
SLAM's live pose can run through the walls of the final map (docs/superpowers/specs/2026-10-30-slam-reliability.md),
so the racing line here comes from the car's path on the map itself, through the same cleanup and limits a live
mapping run uses (recorded_path.prepare, auto_map_race.yaml).

Writes <out-root>/<run name>-rebuilt/, never over an existing folder, with commands.txt last, so a rebuild that
dies part-way leaves no folder that looks finished. A map refused by the verdict gets no raceline_profiled.csv,
so race_launch.py refuses that folder too.

Reads files and writes files. It starts no node and publishes nothing. The first read of a bag needs ROS sourced;
after that the arrays come from the run's analysis/ cache.

Exit code: 0 a folder to race from; 1 a map refused, or no line; 2 nothing to work from.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field as dataclass_field
import json
from pathlib import Path
import sys

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO_ROOT / 'src' / 'pure_pursuit'))
from pure_pursuit import map_localize, occupancy_map, racing_math, recorded_path, run_commands  # noqa: E402
from racerbot_car.layers import effective_config  # noqa: E402
from run_analysis import runlog, slam_bag  # noqa: E402

CONFIG_PATH = REPO_ROOT / 'src' / 'pure_pursuit' / 'config' / 'auto_map_race.yaml'
DEFAULT_MAX_SPEED = 1.5  # m/s, the speed docs/writing-your-own-node.md's "low speed" first floor test means
FASTER_LINES = ((2.5, 'raceline_2p5.csv'), (4.0, 'raceline_4p0.csv'))
SCAN_STRIDE = 2  # of the 40 Hz scans, the same rate slam_toolbox's throttle_scans takes
SEE_THROUGH_STRIDE = 5  # of the tracked scans: see-through is the slow part, and varies slowly
SAME_FIT_MARGIN = 0.03  # fits closer than this are a tie: a few beams of noise
MIN_START_FIT = 0.8  # below this, a place found for the car on the map is not a place at all
MIN_ODOM_ROWS = 10
OFF_LINE_WARNING_M = 0.5


class NothingToRebuild(RuntimeError):
    """The run has no map or no scans: no bag, or stopped before SLAM started."""


@dataclass
class RunInputs:
    grid: np.ndarray       # int8, row 0 at the lowest y: -1 unknown, 0 free, 100 occupied
    resolution: float
    origin_x: float
    origin_y: float
    times: np.ndarray      # (N,) scan receive times
    scans: np.ndarray      # (N, beams)
    angle_min: float
    angle_increment: float
    odom: np.ndarray       # (K, 4): t, x, y, yaw, on the same clock as times
    odom_source: str
    # (P, 4): t, x, y, yaw of SLAM's live pose, on the same clock; empty if the run has none
    slam_pose: np.ndarray = dataclass_field(default_factory=lambda: np.zeros((0, 4)))


@dataclass
class RebuildResult:
    out_dir: Path
    verdict: map_localize.Verdict
    lap_count: int
    raceline_ok: bool
    message: str


def load_inputs(run_dir) -> RunInputs:
    """A run's last map, scans and best odometry, from its bag or analysis/ cache."""
    run_dir = Path(run_dir)
    try:
        slam = slam_bag.load(run_dir)
    except slam_bag.NoSlamMapping as exc:
        raise NothingToRebuild(f'SLAM never mapped this run: {exc}') from exc
    except (FileNotFoundError, OSError) as exc:
        raise NothingToRebuild(f'no readable bag in {run_dir}: {exc}') from exc
    if slam['map'] is None:
        raise NothingToRebuild(f'{run_dir} has no /map message: it stopped before SLAM published one')
    run = runlog.load_run(run_dir)
    if len(run.scan_t) == 0:
        raise NothingToRebuild(f'{run_dir} has no scans')
    use_gyro = len(slam['odom_imu']) >= MIN_ODOM_ROWS
    resolution, origin_x, origin_y, grid = slam['map']
    return RunInputs(grid=grid, resolution=resolution, origin_x=origin_x, origin_y=origin_y,
                     times=run.scan_t, scans=run.scan, angle_min=run.angle_min,
                     angle_increment=run.angle_increment,
                     odom=slam['odom_imu'] if use_gyro else slam['odom'],
                     odom_source='/odom/imu' if use_gyro else '/odom',
                     slam_pose=np.asarray(slam.get('slam_pose', np.zeros((0, 4))), dtype=float).reshape(-1, 4))


def load_config(car_id=None) -> dict:
    return effective_config('auto_map_race.yaml', 'auto_map_race_node', car_id)


def _write_map(out_dir: Path, inputs: RunInputs, max_speckle_cells: int) -> None:
    """The same trinary files slam_toolbox's map saver writes, then the same cleanup a mapping run gives them."""
    image = np.full(inputs.grid.shape, 205, np.uint8)
    image[inputs.grid == 0] = 254
    image[inputs.grid >= 50] = 0
    height, width = image.shape
    with open(out_dir / 'map.pgm', 'wb') as handle:
        handle.write(f'P5\n{width} {height}\n255\n'.encode())
        handle.write(image[::-1].tobytes())  # a PGM's first row is the top of the picture, the highest y
    (out_dir / 'map.yaml').write_text(
        f'image: map.pgm\nmode: trinary\nresolution: {inputs.resolution}\n'
        f'origin: [{inputs.origin_x}, {inputs.origin_y}, 0]\n'
        'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
    occupancy_map.despeckle_map_file(str(out_dir / 'map.yaml'), max_cells=max_speckle_cells)


def _slam_pose_at(inputs: RunInputs, t):
    """SLAM's live pose at time t (the nearest end outside its rows: the fit test decides), or None."""
    rows = inputs.slam_pose
    if len(rows) < 2:
        return None
    yaw = np.unwrap(rows[:, 3])
    return (float(np.interp(t, rows[:, 0], rows[:, 1])), float(np.interp(t, rows[:, 0], rows[:, 2])),
            float(np.interp(t, rows[:, 0], yaw)))


def _place_car(field, inputs: RunInputs):
    """(pose, index of that scan, where it came from) for the earliest scan the map can place, or None.

    SLAM's own pose, refined against the map, when it fits as well as the best place a search of the whole map
    finds: that map was built in SLAM's frame, and on a course that looks alike in two places (the simulator's
    oval looks the same turned round) the search can land on the twin, which fits just as well. A SLAM pose
    that fits worse than the search's best is wrong, as a live pose can be on old settings.
    """
    count = len(inputs.times)
    one_second_in = int(np.searchsorted(inputs.times, inputs.times[0] + 1.0))
    for index in (one_second_in, count // 4, count // 2, 3 * count // 4):
        index = min(index, count - 1)
        beams = map_localize.scan_points(inputs.scans[index], inputs.angle_min, inputs.angle_increment)
        found, found_fit = map_localize.find_start(field, *beams)
        guess = _slam_pose_at(inputs, inputs.times[index])
        if guess is not None:
            pose = map_localize.match_scan(field, *beams, guess)
            fit = map_localize.scan_fit(field, *beams, pose)
            # Against the search's best, not only MIN_START_FIT: inside a solid wall or 3 m along a straight a
            # pose can still fit 0.88, where the right one fits about 1.0.
            if fit >= MIN_START_FIT and fit >= found_fit - SAME_FIT_MARGIN:
                return pose, index, 'SLAM pose'
        if found_fit >= MIN_START_FIT:
            return found, index, 'map search'
    return None


def _best_line(track, laps, grid, config):
    """(line, raw lap) of the accepted lap with the most room to the walls, or (None, None)."""
    field = grid.clearance_field()

    def clearance(xs, ys):
        return grid.clearance_at(xs, ys, field)
    best = (-1.0, None, None)
    for lap in laps:
        raw = track.poses[lap.start:lap.end + 1, :2]
        prepared = recorded_path.prepare(
            raw, spacing=config['waypoint_spacing'],
            max_steering_angle=config['profile_max_steering_angle'], wheelbase=config['profile_wheelbase'],
            min_feature_wavelength=config['profile_min_feature_wavelength'],
            curvature_margin=config['profile_curvature_margin'], max_deviation=config['profile_max_deviation'],
            reject_ratio=config['profile_reject_ratio'], reject_fraction=config['profile_reject_fraction'],
            clearance_fn=clearance, required_clearance=config['profile_wall_clearance'])
        if not prepared.acceptable:
            continue
        room = float(np.min(clearance(prepared.xy[:, 0], prepared.xy[:, 1])))
        if room > best[0]:
            best = (room, prepared.xy, raw)
    return best[1], best[2]


def _write_lines(out_dir: Path, line, raw, max_speed: float, config: dict) -> dict:
    """The profile files, and {file: top speed} of what each one asks of the car."""
    racing_math.save_xy_csv(str(out_dir / 'raceline_raw.csv'), raw)
    seg_len = racing_math.compute_segment_lengths(line, closed=True)
    curvature = racing_math.estimate_path_curvature(line, closed=True)
    tops = {}
    for speed_cap, name in ((max_speed, 'raceline_profiled.csv'), *FASTER_LINES):
        speeds = racing_math.compute_velocity_profile(
            seg_len, curvature, v_max=min(speed_cap, config['profile_max_speed']),
            v_min=min(config['profile_min_speed'], speed_cap),
            a_lat_max=config['profile_max_lateral_accel'], a_accel_max=config['profile_max_accel'],
            a_brake_max=config['profile_max_brake'], closed=True,
            smoothing_passes=config['profile_smoothing_passes'])
        racing_math.save_profiled_csv(str(out_dir / name), line, speeds)
        tops[name] = float(speeds.max())
    return tops


def _commands(out_dir: Path, run_name: str, verdict, message: str, raceable: bool, max_speed: float) -> str:
    rule = run_commands.RULE
    if not raceable:
        return '\n'.join([
            rule, f'NO RACING FOLDER FROM RUN {run_name}.', rule, '', f'  Why: {message}', '',
            f'  The map is kept in {out_dir} to look at. Map again:',
            '    ros2 run racerbot_launch map_course', rule, '']) + '\n'
    lines = [f'Rebuilt from run {run_name}: {message}', '',
             f'FIRST TRY: raceline_profiled.csv is capped at {max_speed:g} m/s. Wheels off the ground first, then the',
             'floor. To go faster later, stop racing and copy a faster line over it:']
    lines += [f'  cp {out_dir / name} {out_dir / "raceline_profiled.csv"}    # {cap:g} m/s'
              for cap, name in FASTER_LINES]
    return '\n'.join(lines) + '\n\n' + run_commands.commands_text(out_dir, posegraph=False)


def rebuild(run_dir, out_root, max_speed: float = DEFAULT_MAX_SPEED, load=load_inputs) -> RebuildResult:
    """See the module docstring. Raises NothingToRebuild, and FileExistsError for an existing folder."""
    run_dir = Path(run_dir)
    out_dir = Path(out_root) / f'{run_dir.name}-rebuilt'
    if out_dir.exists():
        raise FileExistsError(f'{out_dir} already exists; rebuilds are never overwritten')
    inputs = load(run_dir)
    config = load_config()
    out_dir.mkdir(parents=True)
    _write_map(out_dir, inputs, config['map_despeckle_max_cells'])

    grid = occupancy_map.OccupancyMap.from_yaml(str(out_dir / 'map.yaml'))
    report = {'run': str(run_dir), 'odom_source': inputs.odom_source, 'scans': int(len(inputs.times))}

    def finish(verdict, laps, raceable, message):
        report.update(verdict=asdict(verdict), laps=laps, raceline_ok=raceable, message=message)
        (out_dir / 'rebuild.json').write_text(json.dumps(report, indent=2) + '\n')
        (out_dir / 'commands.txt').write_text(
            _commands(out_dir, run_dir.name, verdict, message, raceable, max_speed))
        return RebuildResult(out_dir, verdict, laps, raceable, message)

    try:
        field = map_localize.WallField.from_occupancy(grid)
    except ValueError as exc:
        return finish(map_localize.Verdict(False, str(exc)), 0, False, str(exc))
    placed = _place_car(field, inputs)
    if placed is None:
        reason = (f'the map has no place for this run\'s scans (fit under {MIN_START_FIT}): '
                  'it is doubled, or of somewhere else')
        return finish(map_localize.Verdict(False, reason), 0, False, reason)
    start, first, report['start_from'] = placed
    # Always ends on the last scan: that is the pose the car is parked at.
    picked = np.unique(np.r_[np.arange(first, len(inputs.times), SCAN_STRIDE), len(inputs.times) - 1])
    times, scans = inputs.times[picked], inputs.scans[picked]
    track =map_localize.track_scans(field, times, scans, inputs.odom, inputs.angle_min,
                                     inputs.angle_increment, start)
    through = map_localize.see_through_rate(
        field, track.poses[::SEE_THROUGH_STRIDE], scans[::SEE_THROUGH_STRIDE],
        inputs.angle_min, inputs.angle_increment)
    verdict = map_localize.judge(track.fit, through)
    report['tracked_path_m'] = float(np.hypot(*np.diff(track.poses[:, :2], axis=0).T).sum())
    weaker = ('' if inputs.odom_source != '/odom' else
              ' Mapped on /odom with no gyro, the weaker source: leave out imu:=false and gyro_odom:=false.')
    if not verdict.ok:
        return finish(verdict, 0, False, verdict.reason + weaker)
    laps = map_localize.find_laps(track.poses)
    report['lap_lengths_m'] = [lap.length for lap in laps]
    if not laps:
        return finish(verdict, 0, False, 'the map is good, but less than one whole lap was driven, so there '
                                         'is no racing line.' + weaker)
    line, raw = _best_line(track, laps, grid, config)
    if line is None:
        return finish(verdict, len(laps), False, f'the map is good, but none of its {len(laps)} lap(s) gave a line '
                                                 'the car fits and can steer.' + weaker)
    report['speeds'] = _write_lines(out_dir, line, raw, max_speed, config)
    stop = track.poses[-1]
    run_commands.write_start_pose(out_dir, stop)
    off_line = float(np.min(np.hypot(line[:, 0] - stop[0], line[:, 1] - stop[1])))
    report['start_off_line_m'] = off_line
    message = (f'map OK (fit {verdict.mean_fit:.2f}, see-through {verdict.see_through * 100:.0f}%), '
               f'{len(laps)} lap(s).' + weaker)
    if off_line > OFF_LINE_WARNING_M:
        message += f' The car stopped {off_line:.1f} m off the line: start it nearer the line.'
    return finish(verdict, len(laps), True, message)


def default_out_root(run_dir: Path):
    """data/<car>/racerbot_auto for a run in data/<car>/racerbot_runs/, so a rebuilt race
    folder lands with its own car's maps; None when the run is anywhere else."""
    run_dir = Path(run_dir)
    if run_dir.parent.name == 'racerbot_runs' and run_dir.parent.parent.parent.name == 'data':
        return run_dir.parent.parent / 'racerbot_auto'
    return None


def main(argv=None, load=load_inputs) -> int:
    parser = argparse.ArgumentParser(description='Turn a recorded mapping run into a folder to race from.')
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--out-root', type=Path,
                        help='where the race folder goes (default: the racerbot_auto folder of the '
                             "run's own car, data/<car>/racerbot_auto beside data/<car>/racerbot_runs)")
    parser.add_argument('--max-speed', type=float, default=DEFAULT_MAX_SPEED,
                        help=f'top speed of raceline_profiled.csv in m/s (default {DEFAULT_MAX_SPEED})')
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser()
    out_root = args.out_root.expanduser() if args.out_root is not None else default_out_root(run_dir)
    if out_root is None:
        parser.error(f'{run_dir} is not in a data/<car>/racerbot_runs folder, so its car is unknown: '
                     'pass --out-root')
    try:
        result = rebuild(run_dir, out_root, args.max_speed, load=load)
    except NothingToRebuild as exc:
        print(f'NOTHING TO REBUILD: {exc}')
        return 2
    except FileExistsError as exc:
        print(f'NOTHING DONE: {exc}')
        return 2
    print(('GO: ' if result.raceline_ok else 'NO-GO: ') + result.message)
    print(f'Saved in: {result.out_dir}')
    return 0 if result.raceline_ok else 1


if __name__ == '__main__':
    sys.exit(main())
