"""The real course maps as pool tracks (training plan step 3c): they fit the
GPU canvas at 5 cm, the centreline runs through free space, the measured
widths and lap lengths agree with the import catalog, and the held-out race
runs on them.

Runner: PYTHONPATH=<torch> python3 -m pytest tools/neural_racer/tests/test_realmaps.py -v   (CPU)
"""
import conftest  # noqa: F401
import numpy as np
import pytest
import yaml

from neural_racer_train import realmaps, teacher_eval

CATALOG = yaml.safe_load((realmaps.MAP_DIR / 'catalog.yaml').read_text())
CLEAR = {'room_2026_07_27': 0.99, 'levine_blocked': 0.99, 'levine_obs': 0.80, 'circular_obs': 0.80,
         'car1_course_2026_10_26': 0.99, 'car2_course_2026_10_30': 0.99,
         'car2_course_2026_11_06a_clean': 0.99, 'car2_course_2026_11_06b_clean': 0.99}


@pytest.mark.parametrize('name', teacher_eval.REAL_MAPS)
def test_real_map_is_a_pool_track(name):
    r = realmaps.load(name)
    assert r.resolution == 0.05 and max(r.sdf.shape) <= 1024
    c = r.spec.centre
    col = ((c[:, 0] - r.origin[0]) / r.resolution).astype(int)
    row = ((c[:, 1] - r.origin[1]) / r.resolution).astype(int)
    assert np.mean(r.sdf[row, col] > 0) >= CLEAR[name]            # boxes sit on the obstacle maps' line
    assert np.mean(np.minimum(r.spec.w_left, r.spec.w_right) > 0.2) >= CLEAR[name]
    if 'lap_length_m' in CATALOG.get(name, {}):
        assert r.lap_length == pytest.approx(CATALOG[name]['lap_length_m'], rel=0.1)


def test_heldout_race_includes_the_real_maps():
    ev = teacher_eval.Evaluator(starts=1, limit=1, max_seconds=1.0, raceline=False, real=True)
    res = ev.run(teacher_eval.pursue_policy(0.8))
    n = 1 + len(teacher_eval.REAL_MAPS)
    assert res['episodes'] == n and sorted(res['real_maps']) == sorted(teacher_eval.REAL_MAPS)
    for k in range(1, n):                                          # no start inside a box
        lv = ev._level(k, teacher_eval.FIXED_LEVEL_SEED + k, k)
        r = ev.rasters[k]
        col = ((lv.poses[:, 0] - r.origin[0]) / r.resolution).astype(int)
        row = ((lv.poses[:, 1] - r.origin[1]) / r.resolution).astype(int)
        assert (r.sdf[row, col] >= 0.18).all()


def test_a_real_map_varies_as_a_whole():
    """Mirrored and reversed copies of car 1's course keep the real walls round the moved
    centreline: the free space at each centreline point is the original's."""
    r = realmaps.load('car1_course_2026_10_26')

    def sdf_at(raster, pts):
        col = ((pts[:, 0] - raster.origin[0]) / raster.resolution).astype(int)
        row = ((pts[:, 1] - raster.origin[1]) / raster.resolution).astype(int)
        return raster.sdf[row, col]

    m = realmaps.variant(r, mirror=True)
    assert np.allclose(sdf_at(m, m.spec.centre), sdf_at(r, r.spec.centre), atol=0.08)
    assert np.allclose(m.spec.w_left, r.spec.w_right) and np.allclose(m.spec.w_right, r.spec.w_left)
    rv = realmaps.variant(r, reverse=True)
    assert np.allclose(rv.spec.centre[::-1], r.spec.centre) and np.allclose(rv.spec.w_left[::-1], r.spec.w_right)
    big = realmaps.variant(r, scale=1.08)
    assert big.lap_length == pytest.approx(r.lap_length * 1.08, rel=0.01)
    vs = realmaps.variants('car1_course_2026_10_26')
    assert len(vs) == 12 and len({v.digest for v in vs}) == 12


def test_free_space_no_car_can_enter_becomes_wall():
    """A 1 m corridor, a 0.1 m sliver off it through the wall, and a pocket behind a thin wall:
    reachable_free keeps the corridor and drops the other two (car 2's scanned course has both)."""
    occ = np.ones((60, 100), dtype=bool)
    occ[20:40, 5:95] = False                     # the corridor, 1 m wide at 5 cm
    occ[40:55, 50:52] = False                    # a 0.1 m sliver out of it
    occ[5:18, 20:40] = False                     # a pocket behind a 0.1 m wall (rows 18-19)
    occ[18:20, 30] = False                       # with a one-cell gap in that wall
    centre = np.stack([np.linspace(0.5, 4.5, 50), np.full(50, 1.5)], axis=1)
    keep = realmaps.reachable_free(occ, centre, (0.0, 0.0), 0.05)
    assert keep[20:40, 10:90].all()
    # what is kept reaches its radius (15 cm) into a sliver's mouth or through a gap, and no further
    assert not keep[44:55, 50:52].any() and not keep[5:16, 20:40].any()


@pytest.mark.parametrize('name', teacher_eval.OBSTACLE_FREE_REAL)
def test_real_centreline_has_no_jogs(name):
    """The CSV centrelines jog (car 2's course: a 0.32 m radius over 5 cm); the teacher reads the
    curvature ahead, so load() smooths them (CENTRE_SMOOTH): no point a 1/m sharper than its neighbours."""
    from scipy import ndimage
    from neural_racer_train import tracks
    k = np.abs(tracks.signed_curvature(realmaps.load(name).spec.centre))
    assert (k - ndimage.gaussian_filter1d(k, 3, mode='wrap')).max() < 1.0
