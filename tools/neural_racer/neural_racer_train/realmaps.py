"""The real course maps (tools/f1tenth_sim/maps, heldout.REAL) as GPU
pool tracks, for held-out evaluation only (training plan step 3c): "any
track" was otherwise tested on the generator's own style alone.

Each map is a ROS occupancy image, its map yaml, and a centreline CSV of
x, y only. The image becomes the pool's 5 cm grid -- resampled when its
resolution differs, a cell free only where every source pixel under it is
free; unknown counts as wall -- and is cropped to the free space plus a
margin. The SDF is tracks.rasterize's. The widths, which the CSV lacks, are
measured along each centreline normal to the first wall, so a box in the
corridor narrows the track where it stands. Obstacles stay in the grid (the
student's LiDAR sees them; the teacher's privileged input does not).
"""
import hashlib
import math

import numpy as np
import yaml
from scipy import ndimage

from neural_racer_train import common, tracks

MAP_DIR = common.REPO / 'tools' / 'f1tenth_sim' / 'maps'
CROP_MARGIN = 1.0           # m of wall kept around the free space
MAX_WIDTH = 5.0             # m: a normal that meets no wall within this is open space
# The CSV centrelines jog: every real map has a few points where the curvature spikes 1-2.5 1/m above
# its neighbours' (car 2's course 3.17 1/m, a 0.32 m radius, at a 5 cm jog). The teacher reads the
# centreline's curvature ahead, and on car 2's course turned into a wall with 1.5 m clear straight ahead
# (2026-11-01). Smoothed over this many metres (Gaussian sigma) it moves <= 8 cm and keeps its clearance.
CENTRE_SMOOTH = 0.25


def _grid(name: str):
    """(occupied (H, W) bool, row 0 at origin y; origin (x, y); resolution) as the map files say."""
    from PIL import Image
    meta = yaml.safe_load((MAP_DIR / f'{name}_map.yaml').read_text())
    img = np.asarray(Image.open(MAP_DIR / meta['image']).convert('L'), dtype=np.float64) / 255.0
    occ_p = img if meta.get('negate', 0) else 1.0 - img
    free = occ_p < float(meta.get('free_thresh', 0.196))
    return ~free[::-1], (float(meta['origin'][0]), float(meta['origin'][1])), float(meta['resolution'])


def _resample(occ, origin, res, target: float):
    """Occupancy on a `target` grid over the same extent; a target cell is
    wall if any source pixel it covers is."""
    if abs(res - target) < 1e-9:
        return occ
    h, w = occ.shape
    H, W = int(math.ceil(h * res / target)), int(math.ceil(w * res / target))
    rows = np.minimum((np.arange(H + 1) * target / res).astype(int), h)
    cols = np.minimum((np.arange(W + 1) * target / res).astype(int), w)
    ii = np.add.reduceat(occ.astype(np.int32), rows[:-1], axis=0)
    out = np.add.reduceat(ii, cols[:-1], axis=1) > 0
    return out


def _sdf(occ, resolution: float) -> np.ndarray:
    free = ~occ
    outside = ndimage.distance_transform_edt(free) * resolution - resolution / 2
    inside = ndimage.distance_transform_edt(occ) * resolution - resolution / 2
    return np.where(free, outside, -inside).astype(np.float32)


def variant(r: tracks.RasterTrack, mirror: bool = False, reverse: bool = False, scale: float = 1.0) -> tracks.RasterTrack:
    """A real map changed the ways a car could meet its shape elsewhere, for
    training on it (2026-11-01: car 1's course, whose hairpin no generated
    track had): mirrored (left turns become right), driven the other way round,
    and scaled about the world origin. The real walls stay the real walls; only
    the grid, the centreline and its widths move."""
    occ, lo, res = r.occupancy, r.origin, r.resolution
    c, wl, wr = r.spec.centre.copy(), r.spec.w_left.copy(), r.spec.w_right.copy()
    if scale != 1.0:
        occ = ndimage.zoom(occ.astype(np.uint8), scale, order=0) > 0
        lo = (lo[0] * scale, lo[1] * scale)
        c, wl, wr = c * scale, wl * scale, wr * scale
    if mirror:                                     # y -> -y: the grid's rows flip, left and right swap
        lo = (lo[0], -(lo[1] + occ.shape[0] * res))
        occ = occ[::-1].copy()
        c = c * np.array([1.0, -1.0])
        wl, wr = wr, wl
    if reverse:
        c, wl, wr = c[::-1].copy(), wr[::-1].copy(), wl[::-1].copy()
    tag = f'm{int(mirror)}r{int(reverse)}s{scale:g}'
    spec = tracks.TrackSpec(f'{r.spec.kind}/{tag}', 0, c, wl, wr)
    digest = r.digest + '-' + tag
    return tracks.RasterTrack(spec, occ, _sdf(occ, res), (float(lo[0]), float(lo[1])), res, tracks.lap_length(c), digest)


def variants(name: str, scales=(0.92, 1.0, 1.08)) -> list:
    """Every mirror x direction x scale of a real map: 12 by default."""
    r = load(name)
    return [variant(r, m, rv, s) for m in (False, True) for rv in (False, True) for s in scales]


def load(name: str, resolution: float = tracks.RESOLUTION) -> tracks.RasterTrack:
    occ, origin, res = _grid(name)
    occ = _resample(occ, origin, res, resolution)
    free_rc = np.argwhere(~occ)
    m = int(math.ceil(CROP_MARGIN / resolution))
    r0, c0 = free_rc.min(axis=0) - m                  # may run past the image: the padding is wall
    r1, c1 = free_rc.max(axis=0) + m + 1
    occ = np.pad(occ, m, constant_values=True)[r0 + m:r1 + m, c0 + m:c1 + m]
    lo = (origin[0] + c0 * resolution, origin[1] + r0 * resolution)
    sdf = _sdf(occ, resolution)

    raw = np.loadtxt(MAP_DIR / f'{name}_centerline.csv', delimiter=',', skiprows=1)
    centre = tracks.resample_closed(raw[:, :2])
    centre = tracks.resample_closed(ndimage.gaussian_filter1d(centre, CENTRE_SMOOTH / tracks.SPACING, axis=0,
                                                              mode='wrap'))
    nrm = tracks.normals(centre)
    steps = np.arange(0.0, MAX_WIDTH, resolution / 2)

    def width(sign):
        p = centre[:, None, :] + sign * nrm[:, None, :] * steps[None, :, None]
        col = np.floor((p[..., 0] - lo[0]) / resolution).astype(int)
        row = np.floor((p[..., 1] - lo[1]) / resolution).astype(int)
        inside_grid = (row >= 0) & (row < occ.shape[0]) & (col >= 0) & (col < occ.shape[1])
        wall = ~inside_grid | occ[np.clip(row, 0, occ.shape[0] - 1), np.clip(col, 0, occ.shape[1] - 1)]
        first = np.where(wall.any(axis=1), wall.argmax(axis=1), len(steps) - 1)
        return np.maximum(steps[first] - resolution / 2, 0.0)

    spec = tracks.TrackSpec(f'real:{name}', 0, centre, width(1.0), width(-1.0))
    digest = 'real-' + hashlib.sha256(occ.tobytes() + np.array([resolution, *lo]).tobytes()
                                      + centre.astype(np.float32).tobytes()).hexdigest()[:11]
    return tracks.RasterTrack(spec, occ, sdf, (float(lo[0]), float(lo[1])), resolution,
                              tracks.lap_length(centre), digest)


def reachable_free(occ: np.ndarray, centre: np.ndarray, origin, resolution: float, radius: float = 0.15) -> np.ndarray:
    """The free cells a disk of `radius` can sweep while connected to the
    centreline: free space opened by the disk, its pieces under the
    centreline, grown back by the radius through free space. Drops what no
    car body can enter -- slivers through gaps in a scanned wall, and pockets
    behind an island's thin outline -- which LiDAR beams in the sim would
    otherwise pass into."""
    r = max(int(round(radius / resolution)), 1)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = xx ** 2 + yy ** 2 <= (r + 0.5) ** 2     # not r**2: its one-cell tips slip through one-cell gaps
    free = ~occ
    lab, _ = ndimage.label(ndimage.binary_opening(free, structure=disk))
    col = np.clip(((centre[:, 0] - origin[0]) / resolution).astype(int), 0, occ.shape[1] - 1)
    row = np.clip(((centre[:, 1] - origin[1]) / resolution).astype(int), 0, occ.shape[0] - 1)
    ids = np.unique(lab[row, col])
    keep = np.isin(lab, ids[ids > 0])
    for _ in range(r):          # grown back through free space only: a plain dilation reaches across thin walls
        keep = ndimage.binary_dilation(keep, structure=np.ones((3, 3), bool)) & free
    return keep


def write_clean(name: str, out: str, source: str, radius: float = 0.15, fill: str = '',
                fill_erode: float = 0.25) -> dict:
    """tools/f1tenth_sim/maps/<out>.png/_map.yaml/_centerline.csv: map `name`
    with every free cell outside reachable_free() made wall, on the source's
    own grid; the centreline is copied. `fill`: another map on the same grid
    whose walls (after its own reachable_free), shrunk by `fill_erode`, are
    added first -- to close a map whose walls are accurate but gappy (car 2's
    `_scans` map) with one whose outside is filled but whose walls stand in
    the corridor. Returns the catalog entry."""
    from PIL import Image
    occ, origin, res = _grid(name)
    raw = np.loadtxt(MAP_DIR / f'{name}_centerline.csv', delimiter=',', skiprows=1)
    if fill:
        focc, forigin, fres = _grid(fill)
        assert focc.shape == occ.shape and np.allclose(forigin, origin) and abs(fres - res) < 1e-9, fill
        # the fill's own unreachable pockets first, so an island it outlines thinly is solid before shrinking
        focc = ~reachable_free(focc, raw[:, :2], origin, res, radius)
        n = max(int(round(fill_erode / res)), 1)
        yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
        occ = occ | ndimage.binary_erosion(focc, structure=xx ** 2 + yy ** 2 <= (n + 0.5) ** 2, border_value=1)
    keep = reachable_free(occ, raw[:, :2], origin, res, radius)
    Image.fromarray(np.where(keep, 254, 0).astype(np.uint8)[::-1]).save(MAP_DIR / f'{out}.png')
    meta = yaml.safe_load((MAP_DIR / f'{name}_map.yaml').read_text())
    meta['image'] = f'{out}.png'
    (MAP_DIR / f'{out}_map.yaml').write_text(yaml.safe_dump(meta, sort_keys=True))
    (MAP_DIR / f'{out}_centerline.csv').write_text((MAP_DIR / f'{name}_centerline.csv').read_text())
    r = load(out)
    w = r.spec.w_left + r.spec.w_right
    filled = f' --fill {fill} --fill-erode {fill_erode}' if fill else ''
    closed = f', closed by {fill} shrunk {fill_erode} m,' if fill else ''
    return {'import_command': f"python3 -m neural_racer_train.realmaps {name} {out} --radius {radius}{filled}",
            'source': f'{source}: {name}{closed} with the free space no {2 * radius:.1f} m disk can reach from '
                      f'the centreline made wall (realmaps.reachable_free)',
            'lap_length_m': round(r.lap_length, 2), 'width_median_m': round(float(np.median(w)), 2),
            'width_min_m': round(float(w.min()), 2), 'width_p10_m': round(float(np.percentile(w, 10)), 2)}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='write a cleaned copy of a real map (realmaps.write_clean)')
    ap.add_argument('name')
    ap.add_argument('out')
    ap.add_argument('--radius', type=float, default=0.15)
    ap.add_argument('--source', default='')
    ap.add_argument('--fill', default='', help='a map on the same grid whose shrunk walls close this one')
    ap.add_argument('--fill-erode', type=float, default=0.25)
    a = ap.parse_args()
    print(yaml.safe_dump({a.out: write_clean(a.name, a.out, a.source or a.name, a.radius, a.fill, a.fill_erode)},
                         sort_keys=False))
