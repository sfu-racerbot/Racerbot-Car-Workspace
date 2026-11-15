"""race_line --from-run-dir: racing one car on another car's map (docs/racing-autonomy.md#racing-on-another-cars-map).

Oracles: the copied files are byte-identical to the source's; nothing the source built for its own car is copied; the
line built for each car uses that car's layered limits (load_params), so the two cars' profiles differ exactly where
their limits do. Fixture: src/pure_pursuit/test/data/course_20261111 (car 2's race-day map and recorded path).
Run by `python3 -m pytest src/pure_pursuit/test/`.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from pure_pursuit import race_line

SOURCE = Path(__file__).resolve().parent / 'data' / 'course_20261111'


@pytest.fixture
def source(tmp_path):
    """A copy of the fixture run with a line car 2 'built' (must not be adopted) and a start pose."""
    src = tmp_path / 'car2_source_run'
    shutil.copytree(SOURCE, src)
    (src / 'raceline_profiled.csv').write_text('x,y,speed\n0,0,9.9\n')
    (src / 'start_pose.yaml').write_text('x: 1.0\ny: 2.0\nyaw: 0.5\n')
    return src


def test_it_copies_the_map_path_and_start_pose_and_nothing_built(source, tmp_path):
    target = tmp_path / 'rb1_run'
    copied = race_line.adopt_run_dir(source, target, 'rb1')
    image = yaml.safe_load((source / 'map.yaml').read_text())['image']
    assert sorted(copied) == sorted(['map.yaml', 'raceline_raw.csv', image, 'start_pose.yaml'])
    for name in copied:
        assert (target / name).read_bytes() == (source / name).read_bytes()
    assert not (target / 'raceline_profiled.csv').exists()
    meta = yaml.safe_load((target / 'adopted_from.yaml').read_text())
    assert meta == dict(source=str(source.resolve()), car='rb1', files=copied)


def test_it_never_writes_into_a_non_empty_directory(source, tmp_path):
    target = tmp_path / 'busy'
    target.mkdir()
    (target / 'raceline_profiled.csv').write_text('keep me\n')
    with pytest.raises(FileExistsError, match='not empty'):
        race_line.adopt_run_dir(source, target, 'rb1')
    assert (target / 'raceline_profiled.csv').read_text() == 'keep me\n'


def test_it_refuses_the_same_directory_and_a_directory_that_is_not_a_run(source, tmp_path):
    with pytest.raises(ValueError, match='same directory'):
        race_line.adopt_run_dir(source, source, 'rb1')
    (source / 'raceline_raw.csv').unlink()
    with pytest.raises(FileNotFoundError, match='raceline_raw.csv'):
        race_line.adopt_run_dir(source, tmp_path / 'new', 'rb1')


def test_a_map_yaml_naming_a_missing_or_outside_image_is_refused(source, tmp_path):
    meta = yaml.safe_load((source / 'map.yaml').read_text())
    (source.parent / 'outside.pgm').write_bytes((source / meta['image']).read_bytes())   # exists, but outside the run
    for bad in ('missing.pgm', '../outside.pgm'):
        meta['image'] = bad
        (source / 'map.yaml').write_text(yaml.safe_dump(meta))
        with pytest.raises(FileNotFoundError, match='image'):
            race_line.adopt_run_dir(source, tmp_path / f'new_{len(bad)}', 'rb1')


def test_the_cli_builds_the_line_with_the_adopting_cars_limits(source, tmp_path, monkeypatch):
    # The same course built for each car: each line's speeds respect that car's own planning limits.
    lines = {}
    for car in ('rb1', 'rb2'):
        monkeypatch.setenv('RACERBOT_CAR', car)
        target = tmp_path / f'{car}_run'
        assert race_line.main(['--from-run-dir', str(source), '--run-dir', str(target), '--car', car]) == 0
        data = np.loadtxt(target / 'raceline_profiled.csv', delimiter=',', skiprows=1)
        lines[car] = data
        cfg = race_line.RaceLineConfig.from_params(race_line.load_params(car))
        assert data[:, 2].max() <= cfg.physics.v_max + 1e-9
        assert yaml.safe_load((target / 'adopted_from.yaml').read_text())['car'] == car
    assert not np.array_equal(lines['rb1'], lines['rb2'])          # different limits, different line
