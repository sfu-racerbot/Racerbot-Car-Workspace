"""ROS-free builder tests, run by pytest and the pure_pursuit package runner."""

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml

from pure_pursuit.min_time_line import cone_clearance, footprint_clearance
from pure_pursuit.occupancy_map import FREE, OCCUPIED, OccupancyMap
from pure_pursuit.race_line import RaceLineConfig, build_race_line, check_line, write_race_line
from pure_pursuit.raceline_optimizer import signed_curvature
from pure_pursuit.racing_math import load_profiled_csv, load_xy_csv


DATA = Path(__file__).parent / 'data' / 'course_20261106a'
PACKAGE = Path(__file__).parent.parent
OUTPUTS = {'raceline_profiled.csv', 'raceline_floor_test.csv', 'race_line.json'}


@pytest.fixture(scope='module')
def course():
    grid = OccupancyMap.from_yaml(DATA / 'map.yaml', inflate_cells=0)
    seed = load_xy_csv(DATA / 'raceline_raw.csv')
    cfg = RaceLineConfig()
    return grid, seed, cfg, build_race_line(grid, seed, cfg)


def circle(radius, n=100):
    theta = np.arange(n) * (2 * np.pi / n)
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))


def test_fixture_selection_and_independent_safety(course):
    grid, _, cfg, line = course
    # oracle: invariant, fastest among those passing, and safety recomputed on output.
    assert line.ok
    passing = [c for c in line.candidates if c.ok]
    assert line.source == min(passing, key=lambda c: c.est_lap_s).source
    field = grid.clearance_field()
    at = lambda x, y: grid.clearance_at(x, y, field)
    assert cone_clearance(at, line.xy, laser_x=cfg.laser_x).min() >= cfg.emergency_stop_distance + cfg.cone_margin
    assert footprint_clearance(at, line.xy, half_width=cfg.half_width).min() >= cfg.body_floor
    assert np.abs(signed_curvature(line.xy)).max() <= cfg.physics.curvature_limit + 0.02
    by_name = {c.source: c for c in line.candidates}
    # oracle: measured car 2 course 2026-11-06, curvature line cone 0.41 m.
    assert not by_name['min_curvature'].ok
    assert 'cone' in by_name['min_curvature'].reason
    assert by_name['min_time'].est_lap_s < by_name['recorded'].est_lap_s


def test_forced_modes(course):
    grid, seed, cfg, _ = course
    # oracle: spec, a named mode cannot fall back to a different candidate.
    recorded = build_race_line(grid, seed, replace(cfg, mode='recorded'))
    rejected = build_race_line(grid, seed, replace(cfg, mode='min_curvature'))
    assert recorded.ok and recorded.source == 'recorded'
    assert not rejected.ok and rejected.source is None
    assert 'cone' in rejected.reason


def test_check_line_clear_wall_and_curvature():
    cfg = RaceLineConfig()
    # oracle: closed-form circle curvature 1/r; a zero field is a wall everywhere.
    clear = lambda x, y: np.full(np.broadcast(x, y).shape, 10.0)
    wall = lambda x, y: np.zeros(np.broadcast(x, y).shape)
    ok, _, _, curvature, _ = check_line(clear, circle(3), cfg)
    assert ok and curvature == pytest.approx(1 / 3, abs=1e-10)  # 1/m
    ok, _, body, _, reason = check_line(wall, circle(3), cfg)
    assert not ok and body == 0 and 'body' in reason
    ok, _, _, curvature, reason = check_line(clear, circle(0.5), cfg)
    assert not ok and curvature > cfg.physics.curvature_limit + 0.02 and 'curvature' in reason


def test_invalid_line_and_configuration():
    # oracle: spec, malformed geometry cannot be a safe closed path.
    at = lambda x, y: np.ones(np.broadcast(x, y).shape)
    for xy in (np.empty((0, 2)), np.zeros((1, 2)), np.full((4, 2), np.nan), np.zeros((4, 2))):
        with pytest.raises(ValueError, match='points'):
            check_line(at, xy, RaceLineConfig())
    with pytest.raises(ValueError, match='bogus'):
        RaceLineConfig(mode='bogus')


def test_narrow_track_reports_all_failures():
    # oracle: geometry: a 0.20 m wide annulus cannot fit a 0.31 m wide car.
    resolution = 0.05
    coords = (np.arange(160) + 0.5) * resolution - 4
    xx, yy = np.meshgrid(coords, coords)
    grid = OccupancyMap(np.where(np.abs(np.hypot(xx, yy) - 2) < 0.10, FREE, OCCUPIED),
                        resolution, -4, -4, inflate_cells=0)
    line = build_race_line(grid, circle(2), RaceLineConfig())
    assert not line.ok
    assert len(line.candidates) == 3
    assert all(not c.ok and c.reason and c.reason in line.reason for c in line.candidates)


def test_write_preserves_waypoints_and_caps_speed(course, tmp_path):
    _, _, _, line = course
    # oracle: spec, output is precisely the selected path and its speed profile.
    tops = write_race_line(tmp_path, line)
    assert {p.name for p in tmp_path.iterdir()} == OUTPUTS
    xy, speed = load_profiled_csv(tmp_path / 'raceline_profiled.csv')
    floor_xy, floor_speed = load_profiled_csv(tmp_path / 'raceline_floor_test.csv')
    np.testing.assert_allclose(xy, line.xy, atol=1e-12, rtol=0)
    np.testing.assert_allclose(floor_xy, line.xy, atol=1e-12, rtol=0)
    np.testing.assert_allclose(speed, line.speed, atol=1e-6, rtol=0)
    np.testing.assert_allclose(floor_speed, np.minimum(line.speed, 1.5), atol=1e-6, rtol=0)
    assert tops['raceline_profiled.csv'] == pytest.approx(max(line.speed), abs=1e-9)  # m/s
    assert tops['raceline_floor_test.csv'] == pytest.approx(max(floor_speed), abs=1e-9)  # m/s
    report = json.loads((tmp_path / 'race_line.json').read_text())
    assert report['choice'] == line.source
    assert {c['source'] for c in report['candidates']} == {'recorded', 'min_curvature', 'min_time'}
    refused = replace(line, ok=False, reason='refused')
    empty = tmp_path / 'empty'
    with pytest.raises(ValueError, match='refused'):
        write_race_line(empty, refused)
    assert not empty.exists()


@pytest.mark.parametrize('args,exit_code,choice', [
    ([], 0, 'min_time'),
    (['--mode', 'min_curvature'], 1, None),
    (['--wall-margin', '5.0'], 0, 'recorded'),
])
def test_cli_fixture(tmp_path, args, exit_code, choice):
    # oracle: car 2 fixture measurement, and spec that named mode refuses fallback.
    # Explicit --car must win over the current rb1 used by the test environment.
    for name in ('map.yaml', 'map.pgm', 'raceline_raw.csv'):
        shutil.copy(DATA / name, tmp_path / name)
    env = dict(os.environ, RACERBOT_CAR='rb1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    if choice == 'recorded':
        # The fallback case needs a recorded line that meets the floor. The fixture's sits
        # 0.20 m from the walls, under the shipped 0.22-0.30 m floor (2026-11-07), so this case
        # runs on a copy of the shipped supervisor config with a 0.15 m floor: what it tests is
        # that the CLI falls back to the recorded line when no optimized line fits, not the floor.
        base = yaml.safe_load((PACKAGE / 'config' / 'auto_map_race.yaml').read_text())
        block = base['auto_map_race_node']['ros__parameters']
        block['race_line_body_floor'] = 0.15
        block['race_line_body_floor_tight'] = 0.15
        config = tmp_path / 'supervisor.yaml'
        config.write_text(yaml.safe_dump(base))
        args = [*args, '--config', str(config)]
    proc = subprocess.run([sys.executable, '-m', 'pure_pursuit.race_line', '--run-dir',
                           str(tmp_path), '--car', 'rb2', *args],
                          env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == exit_code, proc.stdout + proc.stderr
    assert OUTPUTS.intersection(p.name for p in tmp_path.iterdir()) == (OUTPUTS if exit_code == 0 else set())
    if choice is not None:
        report = json.loads((tmp_path / 'race_line.json').read_text())
        assert report['choice'] == choice
        if args:
            for c in report['candidates'][1:]:
                assert not c['ok'] and 'too narrow' in c['reason']
    else:
        assert 'cone' in proc.stdout


def test_check_line_judges_each_station_against_its_own_floor():
    # Invariant: check_line fails iff some station's body clearance is under that station's floor.
    # A constant clearance field gives the same body clearance (0.26 m) at every station. On a
    # stadium with fully tight corners the floor is 0.22 m on straights and 0.30 m in corners:
    # 0.26 fails only because of the corners, so a tight floor of None or 0.25 passes.
    radius, straight, spacing = 2.0, 6.0, 0.1
    x = np.arange(0.0, straight, spacing)
    angle = np.linspace(-np.pi / 2, np.pi / 2, 64, endpoint=False)
    xy = np.vstack([
        np.column_stack([x, np.full_like(x, -radius)]),
        np.column_stack([straight + radius * np.cos(angle), radius * np.sin(angle)]),
        np.column_stack([straight - x, np.full_like(x, radius)]),
        np.column_stack([-radius * np.cos(angle), -radius * np.sin(angle)]),
    ])

    def field_at(px, py):
        return np.full(np.shape(px), 0.26)

    # curvature limit 0.5 / 0.9: the 2 m corners (0.5/m) are fully tight; cone and curvature pass
    physics = replace(RaceLineConfig().physics, max_steering_angle=float(np.arctan(0.36 * 0.5 / 0.9 / 0.99)))
    base = RaceLineConfig(physics=physics, body_floor=0.22, emergency_stop_distance=0.0, cone_margin=0.0)
    for tight, ok in ((None, True), (0.25, True), (0.30, False)):
        passed, _, body, _, reason = check_line(field_at, xy, replace(base, body_floor_tight=tight))
        assert body == pytest.approx(0.26, abs=1e-12)
        assert passed is ok, reason
        if not ok:
            assert 'body 0.2600, limit 0.3000' in reason


def test_race_line_config_refuses_a_tight_floor_below_the_base_floor():
    with pytest.raises(ValueError, match='body_floor_tight'):
        RaceLineConfig(body_floor=0.22, body_floor_tight=0.20)
