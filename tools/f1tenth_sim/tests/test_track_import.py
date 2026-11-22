"""Tests for tools.f1tenth_sim.track_import.

Oracle notes: every expected value is either computed closed form in-test,
or follows directly from map_server conventions / the formulas in the module.
"""
import math
import os
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from track_import import (
    SPECK_AREA_M2,
    driveable_region,
    drop_narrow_gaps,
    extract_track,
    load_map,
    pixel_to_world,
    signed_area,
    track_summary,
    world_to_pixel,
    write_gym_track,
)


def write_map(tmp_path, image_uint8, resolution, origin=(0.0, 0.0), free_thresh=0.196, negate=0, name="map"):
    """Write <name>.png via Pillow plus a map_server yaml."""
    # oracle: closed form from map_server format conventions
    image_path = tmp_path / f"{name}.png"
    Image.fromarray(image_uint8.astype(np.uint8), mode="L").save(image_path)
    yaml_path = tmp_path / f"{name}.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "image": f"{name}.png",
                "resolution": float(resolution),
                "origin": [float(origin[0]), float(origin[1]), 0.0],
                "negate": int(negate),
                "occupied_thresh": 0.65,
                "free_thresh": float(free_thresh),
            }
        )
    )
    return yaml_path


def test_load_map_thresholds(tmp_path):
    """Oracle: Nav2 Jazzy p=(255-v)/255; free iff p <= YAML free_thresh."""
    # 1x5 image with pixel values [0, 205, 214, 250, 254]
    img = np.array([[0, 205, 214, 250, 254]], dtype=np.uint8)
    yaml_path = write_map(tmp_path, img, resolution=0.05, origin=(0.0, 0.0), free_thresh=0.196, negate=0)
    spec = load_map(yaml_path)
    assert spec.free[0].tolist() == [False, False, True, True, True]

    # negate: 1 case: value 0 is free (occupancy = image/255 = 0)
    img_neg = np.array([[0]], dtype=np.uint8)
    yaml_path_neg = write_map(tmp_path, img_neg, resolution=0.05, origin=(0.0, 0.0), free_thresh=0.196, negate=1, name="map_neg")
    spec_neg = load_map(yaml_path_neg)
    assert bool(spec_neg.free[0, 0]) is True


@pytest.mark.parametrize("negate", [0, 1])
def test_load_map_jazzy_boundaries(tmp_path, negate):
    """Oracle: navigation2/jazzy/nav2_map_server/src/map_io.cpp loadMapFromFile.

    The occupied/free masks use >= occupied_thresh and <= free_thresh.
    """
    probabilities = np.array([[0, 50, 51, 52, 152, 153, 154, 255]], dtype=np.uint8)
    pixels = probabilities if negate else 255 - probabilities
    path = write_map(tmp_path, pixels, 0.05, free_thresh=51 / 255, negate=negate)
    meta = yaml.safe_load(path.read_text())
    meta['occupied_thresh'] = 153 / 255
    path.write_text(yaml.safe_dump(meta))
    spec = load_map(path)
    assert spec.occupancy.tolist() == [[0, 0, 0, -1, -1, 100, 100, 100]]
    assert spec.free.tolist() == [[True, True, True, False, False, False, False, False]]


@pytest.mark.parametrize("mode", ['scale', 'raw'])
@pytest.mark.parametrize("negate", [0, 1])
def test_load_map_cost_modes(tmp_path, mode, negate):
    """Oracle: map_server raw 0..100 or scaled costs; live ROS grid blocks >=50."""
    costs = np.array([[0, 1, 49, 50, 51, 99, 100]], dtype=np.uint8)
    if mode == 'raw':
        pixels = np.concatenate([costs, [[101, 255]]], axis=1).astype(np.uint8)
        expected = [[0, 1, 49, 50, 51, 99, 100, -1, -1]]
    else:
        # Thresholds 0..1 make p*100 the decoded cost, rounded to nearest integer.
        pixels = np.rint(costs.astype(float) * 255 / 100).astype(np.uint8)
        pixels = pixels if negate else 255 - pixels
        expected = costs.tolist()
    path = write_map(tmp_path, pixels, .05, free_thresh=0, negate=negate)
    meta = yaml.safe_load(path.read_text())
    meta.update(mode=mode, occupied_thresh=1)
    path.write_text(yaml.safe_dump(meta))
    spec = load_map(path)
    assert spec.occupancy.tolist() == expected
    assert spec.free.tolist() == [[v >= 0 and v < 50 for v in expected[0]]]


@pytest.mark.parametrize('negate', [0, 1])
def test_load_map_scale_uses_yaml_interval(tmp_path, negate):
    """Oracle: scale maps [free_thresh, occupied_thresh] linearly to [0,100]."""
    levels = np.array([[0, 51, 102, 153, 255]], dtype=np.uint8)
    path = write_map(tmp_path, levels if negate else 255 - levels, .05,
                     free_thresh=.2, negate=negate)
    meta = yaml.safe_load(path.read_text())
    meta.update(mode='scale', occupied_thresh=.6)
    path.write_text(yaml.safe_dump(meta))
    spec = load_map(path)
    assert spec.occupancy.tolist() == [[0, 0, 50, 100, 100]]
    assert spec.free.tolist() == [[True, True, False, False, False]]


@pytest.mark.parametrize('mode', ['trinary', 'scale'])
def test_load_map_transparency_is_unknown(tmp_path, mode):
    """Oracle: map_server marks nonopaque pixels unknown, including white."""
    path = write_map(tmp_path, np.array([[255, 255, 0]]), .05)
    Image.fromarray(np.array([[[255, 255, 255, 0], [255, 255, 255, 254],
                               [0, 0, 0, 255]]], dtype=np.uint8), 'RGBA').save(path.with_suffix('.png'))
    meta = yaml.safe_load(path.read_text())
    meta['mode'] = mode
    path.write_text(yaml.safe_dump(meta))
    spec = load_map(path)
    assert spec.occupancy.tolist() == [[-1, -1, 100]]
    assert not spec.free.any()


@pytest.mark.parametrize('key,value,error', [
    ('mode', 'typo', 'unsupported map mode'),
    ('negate', 2, 'negate must be'),
    ('negate', -1, 'negate must be'),
    ('free_thresh', -0.1, 'require 0 <='),
    ('free_thresh', .65, 'require 0 <='),
    ('free_thresh', float('nan'), 'require 0 <='),
    ('occupied_thresh', float('inf'), 'require 0 <='),
    ('occupied_thresh', 1.01, 'require 0 <='),
])
def test_load_map_invalid_semantics(tmp_path, key, value, error):
    """Oracle: malformed metadata must not silently invent drivable cells."""
    path = write_map(tmp_path, np.array([[255]]), .05)
    meta = yaml.safe_load(path.read_text())
    meta[key] = value
    path.write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError, match=error):
        load_map(path)


def test_nov9_map_gray_free_cells(tmp_path):
    """Measurement: Nov 9 rb2 /map in runs 190202 and 190837 is 0 in both 3x3s.

    The rebuilt map's YAML free_thresh=.25 classifies 205 (p=50/255) as free.
    Exercise the exported Gym image too, not just the parser's internal mask.
    """
    source = Path(__file__).resolve().parents[3] / 'data/rb2/racerbot_auto/20261109-185146-rebuilt/map.yaml'
    spec = load_map(source)
    # The rebuilt YAML also frees the gray background; use the recorded map
    # for loop extraction, without changing the map Gym will drive on.
    line_map = source.parent.parent / '20261109-185306/map.yaml'
    track = extract_track(spec, centerline_from=load_map(line_map))
    target = write_gym_track(track, tmp_path, 'nov9')
    with Image.open(target.with_suffix('.png')) as handle:
        exported = np.array(handle)
    for row, col in [(207, 188), (307, 116)]:
        assert spec.free[row-1:row+2, col-1:col+2].all()
        assert (exported[row-1:row+2, col-1:col+2] == 255).all()


def test_driveable_region_prefers_enclosed_loop():
    """Oracle: a rectangular corridor, with larger free exterior and interior."""
    free = np.ones((60, 60), dtype=bool)
    free[10:50, 10:50] = False
    free[12:48, 12:48] = True
    free[14:46, 14:46] = False
    free[18:42, 18:42] = True
    expected = np.zeros_like(free)
    expected[12:48, 12:48] = True
    expected[14:46, 14:46] = False
    np.testing.assert_array_equal(driveable_region(free), expected)
    selected = driveable_region(free, (0, 0))
    assert selected[0, 0] and not selected[12, 12] and not selected[18, 18]


def test_driveable_region_fallback_and_errors():
    """Oracle: without an enclosed loop, retain the largest component."""
    np.testing.assert_array_equal(driveable_region(np.ones((1, 1), bool)), [[True]])
    free = np.zeros((5, 5), bool)
    free[0, :3] = True
    free[3, 3] = True
    expected = np.zeros_like(free)
    expected[0, :3] = True
    np.testing.assert_array_equal(driveable_region(free), expected)
    with pytest.raises(ValueError, match='no free space'):
        driveable_region(np.zeros((1, 1), bool))
    with pytest.raises(ValueError, match='not free space'):
        driveable_region(free, (2, 2))


def test_cropped_corridor_survives_false_enclosed_loop(tmp_path):
    """Oracle: wide rectangular race ring at the border, thin false ring inside."""
    image = np.zeros((100, 100), np.uint8)
    image[:90, :90] = 255
    image[15:75, 15:75] = 0
    image[30:39, 30:39] = 255
    image[32:37, 32:37] = 0
    spec = load_map(write_map(tmp_path, image, .05))
    expected = image == 255
    expected[30:39, 30:39] = False
    track = extract_track(spec)
    np.testing.assert_array_equal(track.region, expected)
    rows, cols = world_to_pixel(spec, *track.centerline.T)
    assert expected[rows.astype(int), cols.astype(int)].all()
    # Explicitly selecting the false loop must not silently switch corridors.
    seed = pixel_to_world(spec, 30, 30)
    with pytest.raises(ValueError, match='no free space|not a loop|usable race loop'):
        extract_track(spec, seed_xy=seed)


@pytest.fixture
def blocked_loop_maps(tmp_path):
    """A large rectangular corridor cut by a barrier, with a smaller free exterior."""
    image = np.full((100, 100), 255, np.uint8)
    image[4:96, 4:96] = 0
    image[7:93, 7:93] = 255
    image[32:68, 32:68] = 0
    line = load_map(write_map(tmp_path, image, .05, name='line'))
    image[7:32, 48:52] = 0
    driving = load_map(write_map(tmp_path, image, .05, name='driving'))
    expected = np.zeros((100, 100), bool)
    expected[7:93, 7:93] = True
    expected[32:68, 32:68] = False
    expected[7:32, 48:52] = False
    return driving, line, expected


@pytest.mark.parametrize('seed_rc', [None, (60, 5)])
def test_narrowing_tries_cropped_loop_after_larger_side_room(tmp_path, seed_rc):
    """Oracle: a cropped ring and larger filled room joined by a 15 cm neck."""
    image = np.zeros((120, 180), np.uint8)
    image[10:110, :60] = 255
    image[25:95, 15:45] = 0
    image[10:110, 80:170] = 255
    image[58:61, 60:80] = 255
    spec = load_map(write_map(tmp_path, image, .05))
    seed = None if seed_rc is None else pixel_to_world(spec, *seed_rc)
    track = extract_track(spec, seed_xy=seed)
    rows, cols = world_to_pixel(spec, *track.centerline.T)
    assert ((rows >= 10) & (rows < 110) & (cols >= 0) & (cols < 60)).all()
    np.testing.assert_array_equal(track.region, image == 255)
    narrowed = drop_narrow_gaps(spec.free, .05, .155)
    assert narrowed[60, 5] and narrowed[60, 100]
    assert not narrowed[59, 70]


def test_seeded_room_cannot_switch_to_reference_loop(tmp_path):
    """Oracle: a seeded room has no loop once its narrow connection is closed."""
    image = np.zeros((120, 180), np.uint8)
    image[10:110, :60] = 255
    image[25:95, 15:45] = 0
    image[10:110, 80:170] = 255
    image[58:61, 60:80] = 255
    line = load_map(write_map(tmp_path, image, .05, name='line'))
    image[58:61, 60:80] = 0
    driving = load_map(write_map(tmp_path, image, .05, name='driving'))
    seed = pixel_to_world(driving, 60, 100)
    with pytest.raises(ValueError, match='seed.*no usable race loop'):
        extract_track(driving, seed_xy=seed, centerline_from=line)
    with pytest.raises(ValueError, match='seed.*no usable race loop'):
        extract_track(line, seed_xy=seed)
    loop_seed = pixel_to_world(driving, 60, 5)
    track = extract_track(driving, seed_xy=loop_seed, centerline_from=line)
    rows, cols = world_to_pixel(driving, *track.centerline.T)
    assert track.region[rows, cols].all()
    assert not track.region[60, 100]


def test_clipped_free_seed_projects_to_surviving_loop(tmp_path):
    """Oracle: opening rounds a rectangular ring corner, not the connected loop."""
    image = np.zeros((100, 100), np.uint8)
    image[10:90, 10:90] = 255
    image[30:70, 30:70] = 0
    spec = load_map(write_map(tmp_path, image, .05))
    assert spec.free[10, 10]
    assert not drop_narrow_gaps(spec.free, .05, .155)[10, 10]
    seed = pixel_to_world(spec, 10, 10)
    track = extract_track(spec, seed_xy=seed, centerline_from=spec)
    rows, cols = world_to_pixel(spec, *track.centerline.T)
    assert track.region[rows, cols].all()
    np.testing.assert_array_equal(track.region, image == 255)
    thin = np.zeros((100, 100), np.uint8)
    thin[50:52, 10:90] = 255
    narrow = load_map(write_map(tmp_path, thin, .05, name='thin'))
    with pytest.raises(ValueError, match='seed.*no usable race loop'):
        extract_track(narrow, seed_xy=pixel_to_world(narrow, 50, 20))


def test_open_corridor_beats_smaller_border_loop(blocked_loop_maps):
    """Oracle: the barrier opens the corridor, not permission to select the exterior."""
    driving, _, expected = blocked_loop_maps
    np.testing.assert_array_equal(driveable_region(driving.free), expected)


def test_reference_centerline_selects_blocked_corridor(tmp_path, blocked_loop_maps):
    """Oracle: export the rectangular corridor, retaining the real barrier as wall."""
    driving, line, expected = blocked_loop_maps
    track = extract_track(driving, centerline_from=line)
    write_gym_track(track, tmp_path, 'blocked')
    np.testing.assert_array_equal(track.region, expected)
    exported = np.asarray(Image.open(tmp_path / 'blocked.png'))
    np.testing.assert_array_equal(exported, np.where(expected, 255, 0))


def test_reference_votes_and_explicit_seed():
    """Oracle: three reference points in the small patch, one in the large patch."""
    free = np.zeros((30, 30), bool)
    free[1:20, 1:20] = True
    free[22:28, 22:28] = True
    reference = (np.array([24, 25, 26, 2]), np.array([24, 25, 26, 2]))
    expected = np.zeros_like(free)
    expected[22:28, 22:28] = True
    np.testing.assert_array_equal(driveable_region(free, reference_rc=reference), expected)
    np.testing.assert_array_equal(driveable_region(free, (2, 2), reference_rc=reference), free & ~expected)
    with pytest.raises(ValueError, match='centreline does not intersect free space'):
        driveable_region(free, reference_rc=(np.array([], int), np.array([], int)))
    with pytest.raises(ValueError, match='seed component does not intersect'):
        driveable_region(free, (2, 2), reference_rc=(np.array([24]), np.array([24])))


@pytest.mark.parametrize('name', ['car1_course_2026_10_26', 'car2_course_2026_11_11',
                                 'circular_obs', 'course_2026_10_30',
                                 'levine_blocked', 'levine_obs', 'room_2026_07_27'])
def test_catalog_regeneration_keeps_corridor(tmp_path, name):
    """Invariant: regenerating available catalog maps must not wall off old free cells."""
    root = Path(__file__).resolve().parents[3]
    maps = root / 'tools/f1tenth_sim/maps'
    args = shlex.split(yaml.safe_load((maps / 'catalog.yaml').read_text())[name]['import_command'])
    source = Path(args[2])
    if name == 'car1_course_2026_10_26':
        source = Path('data/rb1/racerbot_auto/20261026-194514-clean/map.yaml')
    elif name == 'course_2026_10_30':
        source = Path('data/rb2/racerbot_auto/20261030-185036-fixed/map.yaml')
    other = load_map(root / args[args.index('--centerline-from') + 1]) if '--centerline-from' in args else None
    track = extract_track(load_map(root / source), centerline_from=other)
    write_gym_track(track, tmp_path, name)
    old_free = np.asarray(Image.open(maps / f'{name}.png')) == 255
    exported = np.asarray(Image.open(tmp_path / f'{name}.png')) == 255
    assert old_free.any()
    assert exported[old_free].all()


def test_pixel_to_world_world_to_pixel_roundtrip(tmp_path):
    """oracle: closed form per track_import formulas; image row 0 is top (largest y)."""
    # 10 rows x 20 cols, res 0.1, origin (-1.0, -2.0)
    res = 0.1
    origin = (-1.0, -2.0)
    h, w = 10, 20
    img = np.full((h, w), 255, dtype=np.uint8)
    yaml_path = write_map(tmp_path, img, resolution=res, origin=origin)
    spec = load_map(yaml_path)

    # pixel (row 9, col 0) centre is (-0.95, -1.95)
    x, y = pixel_to_world(spec, 9, 0)
    assert float(x) == pytest.approx(-0.95, abs=1e-12)
    assert float(y) == pytest.approx(-1.95, abs=1e-12)

    # pixel (row 0, col 19) centre is (0.95, -1.05)
    x2, y2 = pixel_to_world(spec, 0, 19)
    assert float(x2) == pytest.approx(0.95, abs=1e-12)
    assert float(y2) == pytest.approx(-1.05, abs=1e-12)

    # round trip of 50 random in-bounds pixels is exact
    rng = np.random.RandomState(12345)
    rows = rng.randint(0, h, size=50)
    cols = rng.randint(0, w, size=50)
    xw, yw = pixel_to_world(spec, rows, cols)
    rr, cc = world_to_pixel(spec, xw, yw)
    assert np.array_equal(rr.astype(int), rows.astype(int))
    assert np.array_equal(cc.astype(int), cols.astype(int))


def test_ring_circle_extract_track(tmp_path):
    """oracle: closed form - equidistant from both walls gives radius = average of inner/outer radii = 2.5; circumference 2π*2.5."""
    res = 0.05
    origin = (-5.0, -5.0)
    # Make image large enough for ring radius ~3.0 m
    size = int(12.0 / res)
    img = np.zeros((size, size), dtype=np.uint8)  # everything wall by default
    # draw free annulus between r=2.0 and r=3.0 about (0,0)
    # easier to work in world coords
    # fill free where distance from origin is between 2.0 and 3.0
    for r in range(size):
        for c in range(size):
            x = origin[0] + (c + 0.5) * res
            y = origin[1] + (size - r - 0.5) * res  # world y for this pixel
            dist = math.hypot(x, y)
            if 2.0 <= dist <= 3.0:
                img[r, c] = 255  # free
            else:
                img[r, c] = 0  # wall
    yaml_path = write_map(tmp_path, img, resolution=res, origin=origin, free_thresh=0.196)
    spec = load_map(yaml_path)
    track = extract_track(spec, seed_xy=(2.5, 0.0))
    # every centreline point's radius within 0.05 m of 2.5
    radii = np.hypot(track.centerline[:, 0], track.centerline[:, 1])
    assert np.max(np.abs(radii - 2.5)) < 0.05
    # lap length within 2% of 2π·2.5
    summary = track_summary(track)
    assert summary["lap_length_m"] == pytest.approx(2.0 * math.pi * 2.5, rel=0.02)
    # signed_area(centerline) > 0 (counter-clockwise)
    assert signed_area(track.centerline) > 0
    # median width within 0.1 m of 1.0 (annulus width is 1.0 m)
    widths_pos = track.widths[track.widths > 0]
    assert np.median(widths_pos) == pytest.approx(1.0, abs=0.1)
    # consecutive centreline points ~0.10 m apart (within 0.02)
    steps = np.hypot(*np.diff(np.vstack([track.centerline, track.centerline[:1]]), axis=0).T)
    assert np.all(np.abs(steps - 0.10) < 0.02)


def test_unknown_is_wall_separates_outer_free_area(tmp_path):
    """oracle: unknown pixels (205) count as wall and separate regions; seed must pick the inner ring."""
    res = 0.05
    origin = (-5.0, -5.0)
    size = int(12.0 / res)
    img = np.zeros((size, size), dtype=np.uint8)  # everything wall
    for r in range(size):
        for c in range(size):
            x = origin[0] + (c + 0.5) * res
            y = origin[1] + (size - r - 0.5) * res
            dist = math.hypot(x, y)
            if 2.0 <= dist <= 3.0:
                img[r, c] = 255  # free ring
            elif 3.0 < dist <= 3.5:
                img[r, c] = 205  # unknown - should be treated as wall
            elif dist > 3.5:
                img[r, c] = 254  # free beyond, but separated by unknown wall band
            else:
                img[r, c] = 0
    yaml_path = write_map(tmp_path, img, resolution=res, origin=origin, free_thresh=0.196)
    spec = load_map(yaml_path)
    track = extract_track(spec, seed_xy=(2.5, 0.0))
    # must give the same ring result as the simple ring
    radii = np.hypot(track.centerline[:, 0], track.centerline[:, 1])
    assert np.max(np.abs(radii - 2.5)) < 0.05
    summary = track_summary(track)
    assert summary["lap_length_m"] == pytest.approx(2.0 * math.pi * 2.5, rel=0.02)


def test_speck_ignored_real_obstacle_not(tmp_path):
    """oracle: wall blobs under SPECK_AREA_M2 = 0.01 m^2 are erased by the speck
    fill in drop_narrow_gaps -- the 1 px speck is 0.0025 m^2, so the line stays
    on radius 2.5; bigger obstacles are kept and bend it (the 6x6 block is
    0.09 m^2). The opening does not remove the speck: a morphological opening
    never turns wall pixels free, and with the fill disabled the speck bends
    the centreline to |r-2.5| ~ 0.28 m, so this half does test the fill
    threshold."""
    res = 0.05
    # closed-form areas vs that threshold: one pixel (res^2 = 0.0025 m^2) is
    # below it, the 6x6 block (36 * res^2 = 0.09 m^2) is above it
    assert res * res < SPECK_AREA_M2
    assert 6 * 6 * res * res > SPECK_AREA_M2
    origin = (-5.0, -5.0)
    size = int(12.0 / res)
    # base annulus (narrow)
    def make_annulus(extra_fn=None):
        img = np.zeros((size, size), dtype=np.uint8)
        for r in range(size):
            for c in range(size):
                x = origin[0] + (c + 0.5) * res
                y = origin[1] + (size - r - 0.5) * res
                dist = math.hypot(x, y)
                if 2.0 <= dist <= 3.0:
                    img[r, c] = 255
                else:
                    img[r, c] = 0
        if extra_fn:
            extra_fn(img)
        return img

    # add ONE wall pixel at (2.5, 0) - a single pixel speck. It is erased by
    # drop_narrow_gaps' speck fill (area below SPECK_AREA_M2), not by the
    # opening, which can never turn a wall pixel free.
    def add_speck(img):
        x0, y0 = 2.5, 0.0
        c = int((x0 - origin[0]) / res - 0.5)
        r = int(size - 1 - (y0 - origin[1]) / res - 0.5)
        if 0 <= r < size and 0 <= c < size:
            img[r, c] = 0  # wall pixel in the middle of free ring

    img1 = make_annulus(add_speck)
    yaml_path1 = write_map(tmp_path, img1, resolution=res, origin=origin, free_thresh=0.196, name="map1")
    spec1 = load_map(yaml_path1)
    track1 = extract_track(spec1, seed_xy=(2.5, 0.0))
    radii1 = np.hypot(track1.centerline[:, 0], track1.centerline[:, 1])
    # max |radius - 2.5| over the centreline still < 0.06
    assert np.max(np.abs(radii1 - 2.5)) < 0.06

    # wider ring: free between radii 1.5 and 3.5 (2.0 m wide)
    def make_annulus_wide(extra_fn=None):
        img = np.zeros((size, size), dtype=np.uint8)
        for r in range(size):
            for c in range(size):
                x = origin[0] + (c + 0.5) * res
                y = origin[1] + (size - r - 0.5) * res
                dist = math.hypot(x, y)
                if 1.5 <= dist <= 3.5:
                    img[r, c] = 255
                else:
                    img[r, c] = 0
        if extra_fn:
            extra_fn(img)
        return img

    # control: same wide ring with NO block gives max |radius - 2.5| < 0.06 (oracle: closed-form centre radius 2.5 of annulus 1.5..3.5)
    img_ctrl = make_annulus_wide()
    yaml_path_ctrl = write_map(tmp_path, img_ctrl, resolution=res, origin=origin, free_thresh=0.196, name="map_ctrl")
    spec_ctrl = load_map(yaml_path_ctrl)
    track_ctrl = extract_track(spec_ctrl, seed_xy=(0.0, 2.5))
    radii_ctrl = np.hypot(track_ctrl.centerline[:, 0], track_ctrl.centerline[:, 1])
    assert np.max(np.abs(radii_ctrl - 2.5)) < 0.06

    # add a 0.3 x 0.3 m wall block (6 x 6 pixels = 0.30 m) centred at (2.5, 0.0)
    def add_block(img):
        x0, y0 = 2.5, 0.0
        for dr in range(-3, 3):  # 6 pixels
            for dc in range(-3, 3):  # 6 pixels
                c = int((x0 - origin[0]) / res - 0.5) + dc
                r = int(size - 1 - (y0 - origin[1]) / res - 0.5) + dr
                if 0 <= r < size and 0 <= c < size:
                    img[r, c] = 0

    img2 = make_annulus_wide(add_block)
    yaml_path2 = write_map(tmp_path, img2, resolution=res, origin=origin, free_thresh=0.196, name="map2")
    spec2 = load_map(yaml_path2)
    track2 = extract_track(spec2, seed_xy=(0.0, 2.5))
    radii2 = np.hypot(track2.centerline[:, 0], track2.centerline[:, 1])
    # max |radius - 2.5| > 0.10 (the line bends round it)
    assert np.max(np.abs(radii2 - 2.5)) > 0.10


def test_drop_narrow_gaps(tmp_path):
    """oracle: morphological opening removes free space narrower than 2*half_width;
    a 0.20 m wide slot closes at half_width 0.155 (opening diameter 0.31 m) and
    stays open at 0.05 (opening diameter 0.10 m)."""
    res = 0.01  # fine resolution for precise gap
    w = 300
    h = 300
    region = np.zeros((h, w), dtype=bool)
    # corridor: vertical strip down the middle, width ~1.0 m
    corridor_start = int((w / 2.0 - 0.5 / res))
    corridor_end = int((w / 2.0 + 0.5 / res))
    region[10:290, corridor_start:corridor_end] = True  # corridor
    # slot: horizontal branch off the corridor, 0.20 m wide (20 px) and
    # 0.60 m long (60 px) -- wide enough to hold an elliptical opening kernel
    slot_r0 = 140
    slot_r1 = int(140 + 0.20 / res)              # 160
    slot_c0 = corridor_end                       # mouth where it joins
    slot_c1 = int(corridor_end + 0.60 / res)     # 260
    region[slot_r0:slot_r1, slot_c0:slot_c1] = True
    # slot BODY: exclude the first 0.2 m (20 px of columns) next to the mouth,
    # so corridor-mouth spill -- which stays free because it is reachable from
    # the 1.0 m corridor -- can never count as a surviving slot. The body is
    # 20 rows (0.20 m) x 40 cols (0.40 m) = 800 px, closed form.
    body_c0 = slot_c0 + int(0.20 / res)          # 220
    body_px = (slot_r1 - slot_r0) * (slot_c1 - body_c0)
    assert body_px == 800

    # half_width 0.155: opening diameter 0.31 m > 0.20 m slot width, so the
    # slot closes; the body lies 0.2 m past the mouth, farther than the
    # opening's 0.155 m dilation can reach back out of the corridor.
    opened155 = drop_narrow_gaps(region, res, 0.155)
    slot_body155 = int(opened155[slot_r0:slot_r1, body_c0:slot_c1].sum())
    assert slot_body155 == 0  # not one slot-body pixel survives
    # the 1.0 m corridor centre is wider than the opening, so it stays free
    assert opened155[150, w // 2]

    # half_width 0.05: opening diameter 0.10 m < 0.20 m slot width, so the
    # body stays open -- 784 of the 800 closed-form pixels (the opening rounds
    # the far-end corners); a majority must survive.
    opened05 = drop_narrow_gaps(region, res, 0.05)
    slot_body05 = int(opened05[slot_r0:slot_r1, body_c0:slot_c1].sum())
    assert slot_body05 > body_px // 2  # slot body survives


def test_not_a_loop_raises(tmp_path):
    """oracle: find_island raises ValueError 'not a loop' if no enclosed wall region."""
    res = 0.05
    origin = (-2.0, -2.0)
    size = int(5.0 / res)
    img = np.zeros((size, size), dtype=np.uint8)
    # filled disc (no island) - free space is the disc, walls are outside; no enclosed wall inside the free region
    for r in range(size):
        for c in range(size):
            x = origin[0] + (c + 0.5) * res
            y = origin[1] + (size - r - 0.5) * res
            if math.hypot(x, y) < 1.0:
                img[r, c] = 255
            else:
                img[r, c] = 0
    yaml_path = write_map(tmp_path, img, resolution=res, origin=origin, free_thresh=0.196)
    spec = load_map(yaml_path)
    with pytest.raises(ValueError, match="not a loop"):
        extract_track(spec, seed_xy=(0.0, 0.0))


def test_write_gym_track_ring(tmp_path):
    """oracle: written files follow gym triplet format as specified."""
    res = 0.05
    origin = (-5.0, -5.0)
    size = int(12.0 / res)
    img = np.zeros((size, size), dtype=np.uint8)
    for r in range(size):
        for c in range(size):
            x = origin[0] + (c + 0.5) * res
            y = origin[1] + (size - r - 0.5) * res
            dist = math.hypot(x, y)
            if 2.0 <= dist <= 3.0:
                img[r, c] = 255
            else:
                img[r, c] = 0
    yaml_path = write_map(tmp_path, img, resolution=res, origin=origin, free_thresh=0.196)
    spec = load_map(yaml_path)
    track = extract_track(spec, seed_xy=(2.5, 0.0))
    out_dir = tmp_path / "out"
    ret = write_gym_track(track, out_dir, "testrack")
    # png contains only values {0, 255}
    png = np.array(Image.open(out_dir / "testrack.png"))
    unique_vals = set(np.unique(png).tolist())
    assert unique_vals == {0, 255}
    # <name>_map.yaml has the input resolution and origin
    map_yaml = yaml.safe_load((out_dir / "testrack_map.yaml").read_text())
    assert map_yaml["resolution"] == res
    assert map_yaml["origin"][0] == origin[0]
    assert map_yaml["origin"][1] == origin[1]
    # <name>_centerline.csv header is `x_m,y_m` and its row count equals len(centerline)
    csv_lines = (out_dir / "testrack_centerline.csv").read_text().strip().splitlines()
    assert csv_lines[0] == "x_m,y_m"
    assert len(csv_lines) - 1 == len(track.centerline)
    # returns <dir>/<name>.yaml
    assert ret == out_dir / "testrack.yaml"
