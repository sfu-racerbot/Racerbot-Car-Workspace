"""rebuild_run.py, from a recorded run to a folder race_launch.py can use, or a stated refusal.

Runner: python3 -m pytest tools/run_analysis/tests/ (no sourcing needed: arrays come through a fake loader).

The oracle is a course built here with its own ray-caster, so true poses and the true lap length are known in
closed form. Reading a real bag is exercised by running the tool on real runs: results in
docs/superpowers/specs/2026-10-30-slam-reliability.md.
"""
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_analysis import rebuild_run as rb  # noqa: E402
from pure_pursuit import occupancy_map, racing_math, run_commands  # noqa: E402

RES = 0.05
ANGLE_MIN = -math.radians(135.0)
ANGLE_INC = math.radians(0.25)
BEAMS = 1081
WEST, EAST, SOUTH, NORTH, RADIUS = 3.5, 20.5, 2.0, 12.0, 2.0
LAP_M = 2 * (EAST - WEST - 2 * RADIUS) + 2 * (NORTH - SOUTH - 2 * RADIUS) + 2 * math.pi * RADIUS


def _course():
    """A walled ring with uneven bumps along its outer walls, clear of the car's path."""
    g = np.zeros((int(14 / RES), int(24 / RES)), dtype=bool)
    g[:2, :] = g[-2:, :] = True
    g[:, :2] = g[:, -2:] = True
    g[int(4 / RES):int(10 / RES), int(5 / RES):int(19 / RES)] = True
    for index, x in enumerate(np.arange(3.0, 22.0, 2.5)):
        depth = 0.3 + 0.25 * (index % 3)
        width = int((0.3 + 0.2 * (index % 2)) / RES)
        g[2:int(depth / RES), int(x / RES):int(x / RES) + width] = True
        g[-int((depth + 0.2) / RES):, int((x + 1) / RES):int((x + 1) / RES) + width] = True
    for index, y in enumerate(np.arange(3.0, 12.0, 2.5)):
        depth = 0.3 + 0.2 * (index % 3)
        g[int(y / RES):int((y + 0.4) / RES), 2:int(depth / RES)] = True
        g[int((y + 1) / RES):int((y + 1.4) / RES), -int(depth / RES):] = True
    return g


def _ray_cast(pose, walls):
    x, y, yaw = pose
    lx, ly = x + 0.26 * math.cos(yaw), y + 0.26 * math.sin(yaw)
    angles = yaw + ANGLE_MIN + ANGLE_INC * np.arange(BEAMS)
    distances = np.arange(0.1, 12.0, RES / 2)
    cols = ((lx + np.outer(np.cos(angles), distances)) / RES).astype(int)
    rows = ((ly + np.outer(np.sin(angles), distances)) / RES).astype(int)
    inside = (rows >= 0) & (rows < walls.shape[0]) & (cols >= 0) & (cols < walls.shape[1])
    hit = np.zeros(cols.shape, dtype=bool)
    hit[inside] = walls[rows[inside], cols[inside]]
    return np.where(hit.any(axis=1), distances[hit.argmax(axis=1)], np.inf).astype(np.float32)


def _pose_at(s):
    """Anticlockwise on a rounded rectangle centred in the corridor."""
    legs = [EAST - WEST - 2 * RADIUS, NORTH - SOUTH - 2 * RADIUS] * 2
    arc = math.pi / 2 * RADIUS
    s = s % LAP_M
    corners = [(WEST + RADIUS, SOUTH), (EAST, SOUTH + RADIUS), (EAST - RADIUS, NORTH), (WEST, NORTH - RADIUS)]
    for leg, (cx, cy), heading in zip(legs, corners, (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)):
        if s <= leg:
            return cx + s * math.cos(heading), cy + s * math.sin(heading), heading
        s -= leg
        if s <= arc:
            t = s / RADIUS
            ex, ey = cx + leg * math.cos(heading), cy + leg * math.sin(heading)
            ox, oy = ex - RADIUS * math.sin(heading), ey + RADIUS * math.cos(heading)
            return ox + RADIUS * math.sin(heading + t), oy - RADIUS * math.cos(heading + t), heading + t
        s -= arc
    raise AssertionError('past the lap')


COURSE = _course()


def _plain_ring():
    """The same ring with no bumps: it looks the same from a pose and from that pose turned 180 deg about the
    centre, (24 - x, 14 - y, yaw + pi). A scan alone cannot tell the two apart."""
    g = np.zeros((int(14 / RES), int(24 / RES)), dtype=bool)
    g[:2, :] = g[-2:, :] = True
    g[:, :2] = g[:, -2:] = True
    g[int(4 / RES):int(10 / RES), int(5 / RES):int(19 / RES)] = True
    return g


def _twin(pose):
    return 24.0 - pose[0], 14.0 - pose[1], pose[2] + math.pi


def _yaw_gap(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def _inputs(laps=2.3, map_walls=COURSE, spacing=0.4, odom_source='/odom/imu', world=COURSE, slam_pose=None):
    """The arrays a recorded run yields: true scans of `world`, and a grid that may or may not match them.

    slam_pose: 'truth' gives SLAM's live pose as the true pose; an array is used as given.
    """
    distances = np.arange(0.0, laps * LAP_M, spacing)
    times = 100.0 + distances / 2.5
    poses = np.array([_pose_at(s) for s in distances])
    poses[:, 2] = np.unwrap(poses[:, 2])
    scans = np.array([_ray_cast(pose, world) for pose in poses])
    grid = np.zeros(map_walls.shape, dtype=np.int8)
    grid[map_walls] = 100
    if isinstance(slam_pose, str) and slam_pose == 'truth':
        slam_pose = np.c_[times, poses]
    extra = {} if slam_pose is None else {'slam_pose': slam_pose}
    return rb.RunInputs(grid=grid, resolution=RES, origin_x=0.0, origin_y=0.0, times=times, scans=scans,
                        angle_min=ANGLE_MIN, angle_increment=ANGLE_INC, odom=np.c_[times, poses],
                        odom_source=odom_source, **extra)


def _field(walls):
    """The field the rebuild matches against: walls, and everything else free, as _inputs' grid says."""
    return rb.map_localize.WallField(walls, ~walls, RES, 0.0, 0.0)


def test_a_course_that_looks_the_same_both_ways_starts_where_slam_put_the_car():
    # oracle: closed form. On the plain ring the true pose and its 180 deg twin fit the first scan equally, so a
    # whole-map search alone can land on the twin, 4 m or more away, and the check then loses the car on the
    # first stretch that differs (the simulator's oval did this). SLAM's own pose there is the true one.
    ring = _plain_ring()
    inputs = _inputs(map_walls=ring, world=ring, slam_pose='truth')
    pose, index, source = rb._place_car(_field(ring), inputs)
    truth = inputs.odom[index, 1:]
    assert source == 'SLAM pose'
    assert math.hypot(pose[0] - truth[0], pose[1] - truth[1]) < 0.10  # m
    assert _yaw_gap(pose[2], truth[2]) < 0.05  # rad, 3 deg


def test_the_search_alone_cannot_tell_that_course_apart():
    # Why the SLAM pose is needed, and the test above is not passing by luck: both poses fit the scan.
    ring = _plain_ring()
    inputs = _inputs(map_walls=ring, world=ring)
    field = _field(ring)
    index = int(np.searchsorted(inputs.times, inputs.times[0] + 1.0))
    bx, by = rb.map_localize.scan_points(inputs.scans[index], ANGLE_MIN, ANGLE_INC)
    truth = inputs.odom[index, 1:]
    assert rb.map_localize.scan_fit(field, bx, by, truth) > 0.95
    assert rb.map_localize.scan_fit(field, bx, by, _twin(truth)) > 0.95


def test_a_slam_pose_that_does_not_fit_is_not_used():
    # A live pose can sit 2 m from the final map on old settings: then the search must decide. 3 m along a
    # straight still fits 0.88 there, so fitting at all is not enough: it has to fit as well as the best.
    inputs = _inputs(slam_pose='truth')
    wrong = inputs.slam_pose.copy()
    wrong[:, 1] += 3.0  # m, along the bottom straight
    inputs = _inputs(slam_pose=wrong)
    pose, index, source = rb._place_car(_field(COURSE), inputs)
    truth = inputs.odom[index, 1:]
    assert source == 'map search'
    assert math.hypot(pose[0] - truth[0], pose[1] - truth[1]) < 0.10  # m


def test_a_slam_pose_inside_a_wall_is_not_used():
    inputs = _inputs(slam_pose='truth')
    wrong = inputs.slam_pose.copy()
    wrong[:, 2] += 3.0  # m, from the bottom straight into the solid inner block
    inputs = _inputs(slam_pose=wrong)
    _, _, source = rb._place_car(_field(COURSE), inputs)
    assert source == 'map search'


def test_with_no_slam_pose_the_search_places_the_car():
    inputs = _inputs()
    pose, index, source = rb._place_car(_field(COURSE), inputs)
    truth = inputs.odom[index, 1:]
    assert source == 'map search'
    assert math.hypot(pose[0] - truth[0], pose[1] - truth[1]) < 0.10  # m


def test_the_report_says_where_the_start_came_from(tmp_path):
    result = _rebuild(tmp_path, _inputs(slam_pose='truth'))
    assert json.loads((result.out_dir / 'rebuild.json').read_text())['start_from'] == 'SLAM pose'


@pytest.fixture(scope='module')
def good_inputs():
    return _inputs()


def _rebuild(tmp_path, inputs, **kwargs):
    run = tmp_path / '20261030-120000'
    run.mkdir(exist_ok=True)
    return rb.rebuild(run, tmp_path / 'auto', load=lambda _: inputs,
                      supervisor_config=_recorded_base(tmp_path), **kwargs)


def _recorded_base(tmp_path):
    """Make optimized corridors ineligible, so the known driven geometry is the oracle.

    The real builder still runs and checks all three candidates. A 20 m wall
    margin cannot fit this 2-4 m corridor, so only the recorded candidate can
    win. Minimum-time selection is covered on the measured fixture separately.
    """
    config = yaml.safe_load(rb.CONFIG_PATH.read_text())
    config['auto_map_race_node']['ros__parameters']['race_line_wall_margin'] = 20.0
    path = tmp_path / 'recorded_base.yaml'
    path.write_text(yaml.safe_dump(config))
    return path


def _doubled():
    bad = COURSE.copy()
    bad[int(3.2 / RES):int(3.3 / RES), int(5 / RES):int(19 / RES)] = True  # a false wall across open corridor
    bad[int(1.0 / RES):int(1.1 / RES), int(3 / RES):int(21 / RES)] = True
    bad[int(11.0 / RES):int(11.1 / RES), int(3 / RES):int(21 / RES)] = True
    bad[int(3 / RES):int(11 / RES), int(3.0 / RES):int(3.1 / RES)] = True
    bad[int(3 / RES):int(11 / RES), int(21.0 / RES):int(21.1 / RES)] = True
    return bad


# --- an accepted run ------------------------------------------------------------------------------

def test_a_good_run_is_accepted_with_every_file(tmp_path, good_inputs):
    result = _rebuild(tmp_path, good_inputs)
    assert result.verdict.ok
    assert result.raceline_ok
    assert result.out_dir == tmp_path / 'auto' / '20261030-120000-rebuilt'
    # oracle: spec, one selected line, capped copy, and candidate report replace FASTER_LINES.
    for name in ('map.yaml', 'map.pgm', 'raceline_raw.csv', 'raceline_profiled.csv',
                 'raceline_floor_test.csv', 'race_line.json', 'start_pose.yaml', 'commands.txt', 'rebuild.json'):
        assert (result.out_dir / name).is_file(), name


def test_the_saved_map_loads_and_matches_the_input(tmp_path, good_inputs):
    # oracle: invariant. What goes to disk is the same grid, the right way up: row 0 at the lowest y.
    result = _rebuild(tmp_path, good_inputs)
    saved = occupancy_map.OccupancyMap.from_yaml(str(result.out_dir / 'map.yaml'), inflate_cells=0)
    assert saved.resolution == pytest.approx(RES, abs=1e-9)
    walls = saved.grid == occupancy_map.OCCUPIED
    assert walls.shape == COURSE.shape
    assert (walls == COURSE).mean() > 0.999


def test_the_line_is_the_course(tmp_path, good_inputs):
    # oracle: closed form. The car drove one known rounded rectangle, so the line must be that long and
    # stay on that path, here within 0.25 m where cleaning rounds the corners.
    result = _rebuild(tmp_path, good_inputs)
    xy, _ = racing_math.load_profiled_csv(str(result.out_dir / 'raceline_profiled.csv'))[:2]
    length = racing_math.compute_segment_lengths(xy, closed=True).sum()
    assert length == pytest.approx(LAP_M, abs=1.0)  # m, 2% of a lap
    truth = np.array([_pose_at(s)[:2] for s in np.arange(0.0, LAP_M, 0.05)])
    nearest = np.min(np.hypot(xy[:, 0, None] - truth[:, 0], xy[:, 1, None] - truth[:, 1]), axis=1)
    assert nearest.max() < 0.25  # m


def test_the_line_is_the_cleaned_one_not_the_raw_lap(tmp_path, good_inputs):
    # oracle: cited file. recorded_path.prepare resamples to waypoint_spacing (auto_map_race.yaml, 0.15 m),
    # while this fixture's raw lap has a pose every 0.4 m, so which of the two was saved shows in the spacing.
    result = _rebuild(tmp_path, good_inputs)
    xy, _ = racing_math.load_profiled_csv(str(result.out_dir / 'raceline_profiled.csv'))[:2]
    spacing = racing_math.compute_segment_lengths(xy, closed=True)
    assert np.median(spacing) == pytest.approx(rb.load_config()['waypoint_spacing'], abs=0.02)  # m
    raw = racing_math.load_xy_csv(str(result.out_dir / 'raceline_raw.csv'))
    assert np.median(np.hypot(*np.diff(raw, axis=0).T)) > 0.3  # m: the raw lap really is the coarser one


def test_the_first_line_is_capped_and_the_faster_ones_are_not(tmp_path, good_inputs):
    # oracle: spec, default line uses car physics; only the separate floor-test copy is capped.
    result = _rebuild(tmp_path, good_inputs)
    _, speed = racing_math.load_profiled_csv(str(result.out_dir / 'raceline_profiled.csv'))[:2]
    xy, floor = racing_math.load_profiled_csv(str(result.out_dir / 'raceline_floor_test.csv'))[:2]
    assert speed.max() > 2.0  # m/s, this course has 13 m straights
    np.testing.assert_allclose(floor, np.minimum(speed, 1.5), atol=1e-12, rtol=0)  # m/s
    selected = racing_math.load_xy_csv(str(result.out_dir / 'raceline_profiled.csv'))
    np.testing.assert_allclose(xy, selected, atol=1e-12, rtol=0)  # m


def test_max_speed_is_honoured_when_changed(tmp_path, good_inputs):
    result = _rebuild(tmp_path, good_inputs, max_speed=1.0)
    _, speed = racing_math.load_profiled_csv(str(result.out_dir / 'raceline_profiled.csv'))[:2]
    assert speed.max() == pytest.approx(1.0, abs=1e-3)  # m/s


def test_the_start_pose_is_where_the_car_stopped(tmp_path, good_inputs):
    # oracle: the last true pose. That is where the car is parked now, so it is where racing starts.
    result = _rebuild(tmp_path, good_inputs)
    x, y, yaw = run_commands.read_start_pose(result.out_dir)
    truth = good_inputs.odom[-1, 1:]
    assert math.hypot(x - truth[0], y - truth[1]) < 0.15  # m
    assert abs(math.remainder(yaw - truth[2], 2 * math.pi)) < math.radians(3.0)


def test_the_report_records_what_was_measured(tmp_path, good_inputs):
    result = _rebuild(tmp_path, good_inputs)
    report = json.loads((result.out_dir / 'rebuild.json').read_text())
    assert report['verdict']['ok'] is True
    assert report['odom_source'] == '/odom/imu'
    assert report['laps'] == result.lap_count == 2
    assert report['verdict']['mean_fit'] > 0.9
    # oracle: spec, all three candidate verdicts survive into rebuild.json.
    assert report['choice'] == 'recorded'
    candidates = {c['source']: c for c in report['candidates']}
    assert set(candidates) == {'recorded', 'min_curvature', 'min_time'}
    assert candidates['recorded']['ok'] is True
    assert candidates['recorded']['min_cone_m'] > report['parameters']['emergency_stop_distance']
    assert candidates['recorded']['min_body_m'] >= report['parameters']['race_line_body_floor']
    assert candidates['min_curvature']['ok'] is False
    assert 'too narrow' in candidates['min_curvature']['reason']


def test_commands_say_how_to_race_and_at_what_speed(tmp_path, good_inputs):
    result = _rebuild(tmp_path, good_inputs)
    text = (result.out_dir / 'commands.txt').read_text()
    assert 'posegraph' not in text  # a rebuilt folder has none, so it must not claim one
    assert not list(result.out_dir.glob('posegraph*'))
    assert f'race_launch.py run_dir:={result.out_dir}' in text
    assert '1.5 m/s' in text
    assert 'profile:=raceline_floor_test.csv' in text
    assert text.index('profile:=raceline_floor_test.csv') < text.rindex('race_launch.py run_dir:=')


# --- refusals: review focus 1-3 --------------------------------------------------------------------

def test_a_doubled_map_is_refused_with_no_line(tmp_path):
    result = _rebuild(tmp_path, _inputs(map_walls=_doubled()))
    assert not result.verdict.ok
    assert not result.raceline_ok
    assert 'see-through' in result.message
    assert not (result.out_dir / 'raceline_profiled.csv').exists()  # so race_launch.py refuses the folder


def test_a_refused_map_gives_no_race_command(tmp_path):
    result = _rebuild(tmp_path, _inputs(map_walls=_doubled()))
    text = (result.out_dir / 'commands.txt').read_text()
    assert 'race_launch.py' not in text
    assert 'see-through' in text
    assert 'Map again' in text


def test_a_map_of_somewhere_else_is_refused(tmp_path):
    # oracle: invariant. Walls that are not the ones the scans saw: nowhere to put the car.
    other = np.zeros_like(COURSE)
    other[100:102, 40:300] = True
    other[200:202, 100:200] = True
    result = _rebuild(tmp_path, _inputs(laps=0.4, map_walls=other))
    assert not result.verdict.ok
    assert not result.raceline_ok
    assert 'no place' in result.message  # stopped at placing the car, before tracking it anywhere


def test_a_map_with_no_walls_is_refused_not_a_crash(tmp_path):
    result = _rebuild(tmp_path, _inputs(laps=0.2, map_walls=np.zeros_like(COURSE)))
    assert not result.verdict.ok
    assert 'no walls' in result.message


def test_less_than_a_lap_keeps_the_map_and_says_there_is_no_line(tmp_path):
    result = _rebuild(tmp_path, _inputs(laps=0.7))
    assert result.verdict.ok
    assert not result.raceline_ok
    assert result.lap_count == 0
    assert (result.out_dir / 'map.yaml').is_file()
    assert not (result.out_dir / 'raceline_profiled.csv').exists()
    assert 'less than one whole lap' in result.message
    assert 'race_launch.py' not in (result.out_dir / 'commands.txt').read_text()


def test_an_existing_folder_is_never_overwritten(tmp_path, good_inputs):
    _rebuild(tmp_path, good_inputs)
    with pytest.raises(FileExistsError, match='already exists'):
        _rebuild(tmp_path, good_inputs)


def test_commands_are_written_last(tmp_path, good_inputs, monkeypatch):
    # map_course's report picks the newest folder with a commands.txt, so a rebuild that dies part-way must
    # leave none, and be reported as no map rather than half of one.
    import builtins
    real_open = builtins.open

    def fail_profile_write(path, mode='r', *args, **kwargs):
        if str(path).endswith('raceline_profiled.csv') and 'w' in mode:
            raise OSError('died part-way')
        return real_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', fail_profile_write)
    with pytest.raises(OSError, match='died part-way'):
        _rebuild(tmp_path, good_inputs)
    assert not (tmp_path / 'auto' / '20261030-120000-rebuilt' / 'commands.txt').exists()


# --- the weaker odometry: review focus 4 -----------------------------------------------------------

def test_steering_odometry_is_named_as_the_weaker_source(tmp_path):
    result = _rebuild(tmp_path, _inputs(odom_source='/odom'))
    assert result.verdict.ok
    assert '/odom' in result.message
    assert 'gyro' in result.message


# --- the loader ------------------------------------------------------------------------------------

def _fake_run(tmp_path, monkeypatch, slam=None, slam_error=None, run_error=None):
    def load_slam(_):
        if slam_error is not None:
            raise slam_error
        return slam

    def load_run(_):
        if run_error is not None:
            raise run_error
        return type('Run', (), dict(scan_t=np.arange(3.0), scan=np.ones((3, 5)), angle_min=-1.0, angle_increment=0.5))
    monkeypatch.setattr(rb.slam_bag, 'load', load_slam)
    monkeypatch.setattr(rb.runlog, 'load_run', load_run)
    return tmp_path


def _slam_arrays(odom_imu_rows=20):
    rows = np.zeros((20, 4))
    rows[:, 0] = np.arange(20.0)
    return dict(map=(0.05, -1.0, -2.0, np.zeros((4, 4), np.int8)), odom=rows,
                odom_imu=rows[:odom_imu_rows] if odom_imu_rows else np.zeros((0, 4)))


def test_loader_prefers_the_gyro(tmp_path, monkeypatch):
    inputs = rb.load_inputs(_fake_run(tmp_path, monkeypatch, slam=_slam_arrays()))
    assert inputs.odom_source == '/odom/imu'
    assert (inputs.resolution, inputs.origin_x, inputs.origin_y) == (0.05, -1.0, -2.0)


def test_loader_reads_slams_live_pose(tmp_path, monkeypatch):
    slam = _slam_arrays()
    slam['slam_pose'] = slam['odom'] + 1.0
    inputs = rb.load_inputs(_fake_run(tmp_path, monkeypatch, slam=slam))
    assert np.array_equal(inputs.slam_pose, slam['slam_pose'])


def test_loader_without_a_live_pose_gives_none_rows(tmp_path, monkeypatch):
    inputs = rb.load_inputs(_fake_run(tmp_path, monkeypatch, slam=_slam_arrays()))
    assert inputs.slam_pose.shape == (0, 4)


def test_loader_falls_back_to_wheels_without_a_gyro(tmp_path, monkeypatch):
    inputs = rb.load_inputs(_fake_run(tmp_path, monkeypatch, slam=_slam_arrays(odom_imu_rows=0)))
    assert inputs.odom_source == '/odom'


def test_loader_says_there_is_nothing_when_slam_never_mapped(tmp_path, monkeypatch):
    run = _fake_run(tmp_path, monkeypatch, slam_error=rb.slam_bag.NoSlamMapping('no map -> odom'))
    with pytest.raises(rb.NothingToRebuild, match='SLAM'):
        rb.load_inputs(run)


def test_loader_says_there_is_nothing_with_no_map_message(tmp_path, monkeypatch):
    slam = _slam_arrays()
    slam['map'] = None
    with pytest.raises(rb.NothingToRebuild, match='/map'):
        rb.load_inputs(_fake_run(tmp_path, monkeypatch, slam=slam))


def test_loader_says_there_is_nothing_with_no_bag(tmp_path, monkeypatch):
    run = _fake_run(tmp_path, monkeypatch, slam=_slam_arrays(), slam_error=FileNotFoundError('no bag'))
    with pytest.raises(rb.NothingToRebuild, match='bag'):
        rb.load_inputs(run)


def test_loader_says_there_is_nothing_with_no_scans(tmp_path, monkeypatch):
    def empty_run(_):
        return type('Run', (), dict(scan_t=np.zeros(0), scan=np.zeros((0, 0)), angle_min=0.0, angle_increment=0.1))
    monkeypatch.setattr(rb.slam_bag, 'load', lambda _: _slam_arrays())
    monkeypatch.setattr(rb.runlog, 'load_run', empty_run)
    with pytest.raises(rb.NothingToRebuild, match='scans'):
        rb.load_inputs(tmp_path)


# --- the command line -----------------------------------------------------------------------------

def test_exit_zero_for_a_racing_folder(tmp_path, good_inputs, capsys):
    run = tmp_path / '20261030-120000'
    run.mkdir()
    code = rb.main([str(run), '--out-root', str(tmp_path / 'auto'),
                    '--config', str(_recorded_base(tmp_path))], load=lambda _: good_inputs)
    assert code == 0
    assert 'GO' in capsys.readouterr().out


def test_exit_one_for_a_refused_map(tmp_path, capsys):
    run = tmp_path / '20261030-120000'
    run.mkdir()
    inputs = _inputs(map_walls=_doubled())
    code = rb.main([str(run), '--out-root', str(tmp_path / 'auto')], load=lambda _: inputs)
    assert code == 1
    assert 'NO-GO' in capsys.readouterr().out


def test_exit_two_for_nothing_to_work_from(tmp_path, capsys):
    run = tmp_path / '20261030-120000'
    run.mkdir()

    def nothing(_):
        raise rb.NothingToRebuild('this run has no /map message')
    code = rb.main([str(run), '--out-root', str(tmp_path / 'auto')], load=nothing)
    assert code == 2
    assert 'no /map' in capsys.readouterr().out
    assert not (tmp_path / 'auto' / '20261030-120000-rebuilt').exists()


def test_the_default_out_root_is_the_runs_own_car(tmp_path):
    # oracle: the data layout (data/README.md): a car's runs are data/<car>/racerbot_runs/<time>,
    # its maps data/<car>/racerbot_auto/<time>.
    for car in ('rb1', 'rb2'):
        run = tmp_path / 'data' / car / 'racerbot_runs' / '20261030-185036'
        assert rb.default_out_root(run) == tmp_path / 'data' / car / 'racerbot_auto'


@pytest.mark.parametrize('where', [('runs', '20261030-185036'),
                                   ('racerbot_runs', '20261030-185036'),
                                   ('data', 'racerbot_runs', '20261030-185036')])
def test_a_run_outside_a_cars_data_folder_has_no_default_out_root(tmp_path, where):
    # The last is the pre-merge layout: runs straight under data/, with no car folder.
    assert rb.default_out_root(tmp_path.joinpath(*where)) is None


def test_main_saves_into_the_runs_own_car_by_default(tmp_path, good_inputs, capsys):
    run = tmp_path / 'data' / 'rb2' / 'racerbot_runs' / '20261030-120000'
    run.mkdir(parents=True)
    code = rb.main([str(run), '--config', str(_recorded_base(tmp_path))], load=lambda _: good_inputs)
    assert code == 0
    saved = capsys.readouterr().out.split('Saved in: ')[1].strip()
    assert Path(saved).parent == tmp_path / 'data' / 'rb2' / 'racerbot_auto'


def test_main_refuses_a_run_with_no_car_and_no_out_root(tmp_path, capsys):
    run = tmp_path / '20261030-120000'
    run.mkdir()

    def never(_):
        raise AssertionError('nothing may be read before the output folder is known')
    with pytest.raises(SystemExit) as exc:
        rb.main([str(run)], load=never)
    assert exc.value.code == 2
    assert 'pass --out-root' in capsys.readouterr().err


def test_config_is_the_shipped_mapping_config():
    # oracle: spec, effective car calibration plus the unchanged controller stop distance.
    from racerbot_car.layers import effective_config
    for car, angle in [('rb1', .30), ('rb2', .38)]:
        config = rb.load_config(car)
        supervisor = effective_config('auto_map_race.yaml', 'auto_map_race_node', car)
        controller = effective_config('pure_pursuit.yaml', 'pure_pursuit_node', car)
        assert config == dict(supervisor, emergency_stop_distance=controller['emergency_stop_distance'])
        assert config['profile_max_steering_angle'] == angle
        assert config['profile_wall_clearance'] > 0.0


def test_default_builder_selects_fastest_safe_candidate_and_writes_capped_copy(tmp_path):
    # oracle: measured car 2 fixture; winner must have the minimum time among
    # acceptable candidates, and the floor file must have identical geometry.
    data = Path(__file__).resolve().parents[3] / 'src/pure_pursuit/test/data/course_20261106a'
    grid = occupancy_map.OccupancyMap.from_yaml(data / 'map.yaml', inflate_cells=0)
    raw = racing_math.load_xy_csv(str(data / 'raceline_raw.csv'))
    built, tops = rb._write_lines(tmp_path, raw, grid, rb.load_config('rb2'))
    assert built.ok, built.reason
    eligible = [c for c in built.candidates if c.ok]
    assert built.est_lap_s == min(c.est_lap_s for c in eligible)
    assert built.source == 'min_time'
    xy, speed = racing_math.load_profiled_csv(str(tmp_path / 'raceline_profiled.csv'))[:2]
    floor_xy, floor = racing_math.load_profiled_csv(str(tmp_path / 'raceline_floor_test.csv'))[:2]
    np.testing.assert_allclose(xy, built.xy, atol=1e-12, rtol=0)  # m
    np.testing.assert_allclose(speed, built.speed, atol=1e-12, rtol=0)  # m/s
    np.testing.assert_allclose(floor_xy, xy, atol=1e-12, rtol=0)  # m
    np.testing.assert_allclose(floor, np.minimum(speed, 1.5), atol=1e-12, rtol=0)  # m/s
    assert 1.5 < tops['raceline_profiled.csv'] <= 5.0


def test_builder_refusal_never_writes_a_raceable_profile(tmp_path):
    # oracle: geometry, an entirely occupied grid cannot contain a safe line.
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    raw = np.column_stack((3 + 2 * np.cos(angles), 3 + 2 * np.sin(angles)))
    grid = occupancy_map.OccupancyMap(np.full((120, 120), 100, np.int8), .05, 0., 0.)
    built, tops = rb._write_lines(tmp_path, raw, grid, rb.load_config('rb1'))
    assert not built.ok
    assert tops == {}
    assert not (tmp_path / 'raceline_profiled.csv').exists()
    assert not (tmp_path / 'raceline_floor_test.csv').exists()
    assert all(not candidate.ok for candidate in built.candidates)


def test_run_car_identity_uses_recorded_car_before_current_car(tmp_path, monkeypatch):
    # oracle: spec, rebuilding rb2's recording on rb1 uses rb2's physics.
    monkeypatch.setenv('RACERBOT_CAR', 'rb1')
    run = tmp_path / 'data/rb2/racerbot_runs/time'
    run.mkdir(parents=True)
    assert rb.run_car_id(run) == 'rb2'
    metadata = run / 'metadata'
    metadata.mkdir()
    (metadata / 'car.yaml').write_text('car_id: rb1\n')
    assert rb.run_car_id(run) == 'rb1'
