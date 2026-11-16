"""Procedural tracks: generation, validity checks, mutation and rasterization
(plan, Phase 4). NumPy/SciPy only -- runs in CPU worker processes.

Two generators, because smooth loops alone are blob-shaped, with few long
straights and no real hairpins, which is where races are won:

    smooth_loop       random control points on a wobbly circle, periodic cubic spline
    filleted_polygon  a random polygon with dents, each corner rounded by an arc

Every track must pass check(): corner radius >= 1.3 m (the car's minimum at
0.33 rad is 1.05 m), no part of the track within a wall of another part, and
at least 0.8 m of free width everywhere.
"""
import hashlib
import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

SPACING = 0.10            # m between centreline points
RESOLUTION = 0.05         # m per grid cell, as the GPU pool stores it
MARGIN = 2.0              # m of wall around the track's bounding box
WALL_GAP = 0.20           # m: two parts of a track must be at least this far apart, wall to wall


@dataclass(frozen=True)
class TrackGenConfig:
    lap_min: float = 15.0
    lap_max: float = 120.0
    width_min: float = 0.9
    width_max: float = 3.0
    min_radius: float = 1.3
    # m: a corner whose centreline radius R is under this needs R + its outside
    # width >= turn_room, so the car's turning circle fits by swinging wide
    # (the outer front corner sweeps ~1.34 m at full lock). 0 = off: every
    # corner then keeps min_radius (1.3 m, over the car's 1.05 m minimum)
    turn_room: float = 0.0
    hairpin_share: float = 0.0           # share of hairpin_loop tracks (the rest split as `shares`)
    max_obstacles: int = 4
    min_free: float = 0.8
    shares: tuple = (0.4, 0.4, 0.2)      # smooth_loop, filleted_polygon, imported
    max_extent: float = 50.0             # m: the GPU pool's canvas edge


@dataclass
class TrackSpec:
    """The editable description: what mutation changes and rasterize draws."""
    kind: str
    seed: int
    centre: np.ndarray                   # (N, 2) closed, ~SPACING apart, CCW not required
    w_left: np.ndarray                   # (N,)
    w_right: np.ndarray                  # (N,)
    obstacles: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # x, y, half_w, half_h (axis-aligned)


@dataclass
class RasterTrack:
    spec: TrackSpec
    occupancy: np.ndarray                # (H, W) bool, True = blocked; row 0 at origin_y
    sdf: np.ndarray                      # (H, W) float32 m: + in free space, - inside walls
    origin: tuple                        # (x, y) of cell (0, 0)'s corner
    resolution: float
    lap_length: float
    digest: str


# --- geometry helpers ---------------------------------------------------------

def resample_closed(points: np.ndarray, spacing: float = SPACING) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    seg = np.hypot(*(np.roll(p, -1, axis=0) - p).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(round(s[-1] / spacing)), 8)
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    closed = np.vstack([p, p[:1]])
    return np.stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])], axis=1)


def lap_length(centre: np.ndarray) -> float:
    return float(np.hypot(*(np.roll(centre, -1, axis=0) - centre).T).sum())


def signed_curvature(centre: np.ndarray) -> np.ndarray:
    """Menger curvature with sign (+ = left turn), on a closed polyline."""
    a, b, c = np.roll(centre, 1, axis=0), centre, np.roll(centre, -1, axis=0)
    ab, bc, ca = b - a, c - b, a - c
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    denom = np.hypot(*ab.T) * np.hypot(*bc.T) * np.hypot(*ca.T)
    return np.where(denom > 1e-12, 2.0 * cross / np.maximum(denom, 1e-12), 0.0)


def normals(centre: np.ndarray) -> np.ndarray:
    t = np.roll(centre, -1, axis=0) - np.roll(centre, 1, axis=0)
    t /= np.maximum(np.hypot(*t.T), 1e-12)[:, None]
    return np.stack([-t[:, 1], t[:, 0]], axis=1)       # left of travel


def smooth_periodic(n: int, rng, lo: float, hi: float, terms: int = 4) -> np.ndarray:
    """A smooth periodic profile within [lo, hi]."""
    u = np.linspace(0, 2 * math.pi, n, endpoint=False)
    v = np.zeros(n)
    for j in range(1, terms + 1):
        v += rng.normal(0, 1.0 / j) * np.sin(j * u + rng.uniform(0, 2 * math.pi))
    v = (v - v.min()) / max(v.max() - v.min(), 1e-9)
    mid = rng.uniform(lo, hi)
    span = rng.uniform(0.0, hi - lo)
    return np.clip(mid + (v - 0.5) * span, lo, hi)


def _widths(n, rng, cfg):
    total = smooth_periodic(n, rng, cfg.width_min, cfg.width_max)
    split = smooth_periodic(n, rng, 0.35, 0.65, terms=2)
    return total * split, total * (1.0 - split)


# --- generators ------------------------------------------------------------------

def smooth_loop(seed: int, cfg: TrackGenConfig = TrackGenConfig()) -> TrackSpec:
    rng = np.random.default_rng(seed)
    k = int(rng.integers(8, 17))
    ang = (np.arange(k) + rng.uniform(-0.3, 0.3, k)) * 2 * math.pi / k
    ang.sort()
    r = rng.uniform(0.6, 1.4, k)
    pts = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    t = np.concatenate([ang, [ang[0] + 2 * math.pi]])
    spline = CubicSpline(t, np.vstack([pts, pts[:1]]), bc_type='periodic')
    dense = spline(np.linspace(ang[0], ang[0] + 2 * math.pi, 2000, endpoint=False))
    target = rng.uniform(cfg.lap_min, cfg.lap_max)
    dense *= target / lap_length(dense)
    centre = resample_closed(dense)
    wl, wr = _widths(len(centre), rng, cfg)
    return TrackSpec('smooth_loop', seed, centre, wl, wr)


def hairpin_loop(seed: int, cfg: TrackGenConfig = TrackGenConfig()) -> TrackSpec:
    """A long loop around a thin divider, as car 1's course is: two straights
    joined by U-turns round the divider's ends. Each U-turn has a bulb (room on
    the outside mid-turn) and then, just past its exit, a pinch where the outside
    wall comes in close -- car 1's left hairpin, where a car that starts its exit
    turn late ends up nose-on to that wall (2026-11-01). Mirrored at random, so
    U-turns go both ways."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(1.1, 1.8)                      # corridor width
    d = rng.uniform(0.25, 0.8)                     # divider thickness
    rc = (w + d) / 2                               # centreline radius round the divider's ends
    ls = rng.uniform(4.0, 14.0)                    # straight length
    seg = []
    n_s = max(int(ls / SPACING), 2)
    n_c = max(int(math.pi * rc / SPACING), 8)
    seg.append(np.stack([np.linspace(0, ls, n_s, endpoint=False), np.zeros(n_s)], 1))
    a = np.linspace(-math.pi / 2, math.pi / 2, n_c, endpoint=False)
    seg.append(np.stack([ls + rc * np.cos(a), rc + rc * np.sin(a)], 1))
    seg.append(np.stack([np.linspace(ls, 0, n_s, endpoint=False), np.full(n_s, 2 * rc)], 1))
    a = np.linspace(math.pi / 2, 3 * math.pi / 2, n_c, endpoint=False)
    seg.append(np.stack([rc * np.cos(a), rc + rc * np.sin(a)], 1))
    raw = np.vstack(seg)
    lens = [len(x) for x in seg]
    # the outside (right, the loop running anticlockwise) widens mid-turn and pinches after each exit
    outside = np.full(len(raw), w / 2)
    start = 0
    for k, n in enumerate(lens):
        if k % 2 == 1:                             # a U-turn
            u = np.linspace(0, 1, n)
            outside[start:start + n] += rng.uniform(0.0, 1.0) * np.sin(math.pi * u)
            pinch = rng.uniform(0.45, 0.95) * w / 2
            n_p = int(rng.uniform(0.6, 1.6) / SPACING)
            after = (start + n + np.arange(n_p)) % len(raw)
            outside[after] = np.minimum(outside[after], w / 2 - (w / 2 - pinch) * np.sin(math.pi * np.linspace(0, 1, n_p)))
        start += n
    inside = np.full(len(raw), w / 2)
    centre = resample_closed(raw)
    idx = np.linspace(0, len(raw), len(centre), endpoint=False)
    wr = np.interp(idx, np.arange(len(raw)), outside, period=len(raw))
    wl = np.interp(idx, np.arange(len(raw)), inside, period=len(raw))
    if rng.random() < 0.5:                         # clockwise: mirror, and the outside is now the left
        centre = centre * np.array([1.0, -1.0])
        wl, wr = wr, wl
    th = rng.uniform(0, 2 * math.pi)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return TrackSpec('hairpin_loop', seed, centre @ rot.T, wl, wr)


def _fillet_path(verts: np.ndarray, radii: np.ndarray):
    """Round each polygon corner with a tangent arc; None if a radius doesn't fit."""
    n = len(verts)
    pieces = []
    for i in range(n):
        prev, v, nxt = verts[i - 1], verts[i], verts[(i + 1) % n]
        d_in = (v - prev) / np.linalg.norm(v - prev)
        d_out = (nxt - v) / np.linalg.norm(nxt - v)
        cross = d_in[0] * d_out[1] - d_in[1] * d_out[0]
        phi = math.atan2(cross, float(d_in @ d_out))          # signed turn angle
        if abs(phi) < 1e-3:
            pieces.append(np.array([v]))
            continue
        r = radii[i]
        t = r * math.tan(abs(phi) / 2)
        lim = 0.5 * min(np.linalg.norm(v - prev), np.linalg.norm(nxt - v))
        if t > lim:
            return None
        p1 = v - d_in * t
        left = np.array([-d_in[1], d_in[0]])
        centre = p1 + left * r * math.copysign(1.0, phi)
        a0 = math.atan2(*(p1 - centre)[::-1])
        m = max(int(abs(phi) * r / 0.05), 4)
        a = a0 + np.linspace(0.0, phi, m)
        pieces.append(centre + r * np.stack([np.cos(a), np.sin(a)], axis=1))
    return np.vstack(pieces)


def filleted_polygon(seed: int, cfg: TrackGenConfig = TrackGenConfig()) -> TrackSpec:
    rng = np.random.default_rng(seed)
    for _ in range(50):
        k = int(rng.integers(5, 11))
        ang = np.sort(rng.uniform(0, 2 * math.pi, k))
        verts = np.stack([np.cos(ang), np.sin(ang)], axis=1) * rng.uniform(0.85, 1.15, (k, 1))
        for i in rng.choice(k, size=int(rng.integers(1, 4)), replace=False):
            verts[i] *= rng.uniform(0.4, 0.7)                   # dents: hairpins and chicanes
        perimeter = lap_length(verts)
        target = rng.uniform(cfg.lap_min, cfg.lap_max)
        verts *= target / perimeter
        path = _fillet_path(verts, rng.uniform(cfg.min_radius, 6.0, k))
        if path is None:
            continue
        centre = resample_closed(path)
        wl, wr = _widths(len(centre), rng, cfg)
        return TrackSpec('filleted_polygon', seed, centre, wl, wr)
    raise RuntimeError(f'filleted_polygon: no fillet fit after 50 tries (seed {seed})')


def add_obstacles(spec: TrackSpec, rng, cfg: TrackGenConfig) -> TrackSpec:
    n = int(rng.integers(0, cfg.max_obstacles + 1))
    obs = []
    nrm = normals(spec.centre)
    for _ in range(n):
        i = int(rng.integers(len(spec.centre)))
        hw = rng.uniform(0.15, 0.30)
        width = spec.w_left[i] + spec.w_right[i]
        if width < cfg.min_free + 2 * hw + 0.1:
            continue
        side = rng.choice([-1.0, 1.0])
        # leave min_free (+ margin) on the other side of the box
        lateral = side * (min(spec.w_left[i] if side > 0 else spec.w_right[i],
                              width - cfg.min_free - 0.1) - hw)
        c = spec.centre[i] + nrm[i] * lateral
        obs.append([c[0], c[1], hw, hw])
    return replace(spec, obstacles=np.array(obs).reshape(-1, 4))


# --- validity ---------------------------------------------------------------------

def check(spec: TrackSpec, cfg: TrackGenConfig = TrackGenConfig()) -> list:
    """Reasons this track is unusable; empty if it is valid."""
    reasons = []
    c, wl, wr = spec.centre, spec.w_left, spec.w_right
    if len(c) < 20 or not np.all(np.isfinite(c)):
        return ['degenerate centreline']
    L = lap_length(c)
    if not (cfg.lap_min - 1e-6 <= L <= cfg.lap_max + 1e-6):
        reasons.append(f'lap length {L:.1f} m outside [{cfg.lap_min}, {cfg.lap_max}]')
    # curvature on a 0.5 m baseline: point-to-point noise at 0.1 m spacing
    # would otherwise read as curvature
    step = max(int(round(0.5 / SPACING)), 1)
    coarse = c[::step]
    kappa = np.abs(signed_curvature(coarse))
    if np.any(kappa > 1.0 / cfg.min_radius + 1e-9):
        reasons.append(f'radius {1.0 / kappa.max():.2f} m under {cfg.min_radius} m')
    if cfg.turn_room > 0:
        # signed_curvature is + for a left turn, whose outside is the right
        signed = signed_curvature(coarse)
        radius = 1.0 / np.maximum(np.abs(signed), 1e-9)
        outside = np.where(signed > 0, wr[::step], wl[::step])
        if np.any((radius < cfg.turn_room) & (radius + outside < cfg.turn_room - 1e-9)):
            reasons.append(f'a corner too tight for its width: radius + outside width under {cfg.turn_room} m')
    if np.any(wl + wr < cfg.min_free - 1e-9):
        reasons.append('free width under min_free')
    # self-contact: any two centreline points far apart along the lap must be
    # further apart in space than their half-widths plus a wall
    half = np.maximum(wl, wr)
    tree = cKDTree(c)
    s = np.arange(len(c)) * (L / len(c))
    reach = 2 * half.max() + WALL_GAP
    for i, js in enumerate(tree.query_ball_point(c, reach)):
        for j in js:
            along = abs(s[i] - s[j])
            along = min(along, L - along)
            need = half[i] + half[j] + WALL_GAP
            if along > need * 1.5 + 0.5 and np.hypot(*(c[i] - c[j])) < need:
                reasons.append('track touches itself (self-intersection or walls too close)')
                return reasons
    lo, hi = c.min(axis=0) - half.max() - MARGIN, c.max(axis=0) + half.max() + MARGIN
    if np.any(hi - lo > cfg.max_extent):
        reasons.append(f'extent {float((hi - lo).max()):.1f} m over the {cfg.max_extent} m canvas')
    return reasons


def generate(seed: int, cfg: TrackGenConfig = TrackGenConfig(), kind: str | None = None) -> TrackSpec:
    """A valid track from `seed` (retrying with derived seeds)."""
    rng = np.random.default_rng(seed)
    if kind is None:
        # no draw at share 0, so v1 and v2 tracks stay what their seeds always made
        if cfg.hairpin_share > 0 and rng.random() < cfg.hairpin_share:
            kind = 'hairpin_loop'
        else:
            kind = rng.choice(['smooth_loop', 'filleted_polygon'],
                              p=np.array(cfg.shares[:2]) / sum(cfg.shares[:2]))
    gen = {'smooth_loop': smooth_loop, 'filleted_polygon': filleted_polygon, 'hairpin_loop': hairpin_loop}[kind]
    for attempt in range(200):
        sub = int(rng.integers(2 ** 31))
        try:
            spec = add_obstacles(gen(sub, cfg), np.random.default_rng(sub + 1), cfg)
        except RuntimeError:
            continue
        if not check(spec, cfg):
            return replace(spec, seed=seed)
    raise RuntimeError(f'no valid {kind} track after 200 attempts (seed {seed})')


def mutate(spec: TrackSpec, rng, cfg: TrackGenConfig = TrackGenConfig(), tries: int = 20) -> TrackSpec:
    """ACCEL-style edit (plan, Phase 4): move a stretch of the centreline,
    change a width, or add/remove an obstacle. Redrawn until check() passes;
    returns the input unchanged if no valid mutation is found."""
    for _ in range(tries):
        kind = rng.integers(3)
        c, wl, wr, obs = spec.centre.copy(), spec.w_left.copy(), spec.w_right.copy(), spec.obstacles.copy()
        n = len(c)
        if kind == 0:
            i = int(rng.integers(n))
            span = int(rng.integers(n // 20 + 2, n // 6 + 3))
            idx = (i + np.arange(-span, span + 1)) % n
            bump = np.cos(np.linspace(-math.pi / 2, math.pi / 2, len(idx)))[:, None]
            c[idx] += bump * rng.normal(0, 0.5, 2)
            c = resample_closed(c)
            wl = np.interp(np.linspace(0, 1, len(c)), np.linspace(0, 1, n), wl)
            wr = np.interp(np.linspace(0, 1, len(c)), np.linspace(0, 1, n), wr)
        elif kind == 1:
            i = int(rng.integers(n))
            span = int(rng.integers(5, n // 4 + 6))
            idx = (i + np.arange(-span, span + 1)) % n
            delta = rng.uniform(-0.2, 0.2) * np.cos(np.linspace(-math.pi / 2, math.pi / 2, len(idx)))
            side = wl if rng.random() < 0.5 else wr
            side[idx] = np.clip(side[idx] + delta, cfg.width_min / 2, cfg.width_max / 2)
        else:
            if len(obs) and rng.random() < 0.5:
                obs = np.delete(obs, int(rng.integers(len(obs))), axis=0)
            else:
                more = add_obstacles(replace(spec, obstacles=np.zeros((0, 4))), rng, replace(cfg, max_obstacles=1))
                obs = np.vstack([obs, more.obstacles])
        cand = TrackSpec(spec.kind, spec.seed, c, wl, wr, obs)
        if not check(cand, cfg):
            return cand
    return spec


# --- rasterization --------------------------------------------------------------------

def rasterize(spec: TrackSpec, resolution: float = RESOLUTION) -> RasterTrack:
    c, wl, wr = spec.centre, spec.w_left, spec.w_right
    half = max(wl.max(), wr.max())
    lo = c.min(axis=0) - half - MARGIN
    hi = c.max(axis=0) + half + MARGIN
    w = int(math.ceil((hi[0] - lo[0]) / resolution))
    h = int(math.ceil((hi[1] - lo[1]) / resolution))
    xs = lo[0] + (np.arange(w) + 0.5) * resolution
    ys = lo[1] + (np.arange(h) + 0.5) * resolution
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    # dense centreline so the nearest point is close to the true projection
    dense_n = len(c) * 4
    tt = np.linspace(0, len(c), dense_n, endpoint=False)
    i0 = np.floor(tt).astype(int) % len(c)
    i1 = (i0 + 1) % len(c)
    f = (tt - np.floor(tt))[:, None]
    dense = c[i0] * (1 - f) + c[i1] * f
    dwl = wl[i0] * (1 - f[:, 0]) + wl[i1] * f[:, 0]
    dwr = wr[i0] * (1 - f[:, 0]) + wr[i1] * f[:, 0]
    nrm = normals(dense)
    # Nearest centreline point per cell, via one distance transform over the
    # drawn centreline (linear in cells; a KD-tree query was 2.9 s a track).
    label = np.zeros((h, w), dtype=np.int64)
    col = np.clip(((dense[:, 0] - lo[0]) / resolution).astype(int), 0, w - 1)
    row = np.clip(((dense[:, 1] - lo[1]) / resolution).astype(int), 0, h - 1)
    label[row, col] = np.arange(len(dense)) + 1
    _, (ri, ci) = ndimage.distance_transform_edt(label == 0, return_indices=True)
    j = (label[ri, ci] - 1).ravel()
    lateral = ((pts - dense[j]) * nrm[j]).sum(axis=1)
    free = (lateral <= dwl[j]) & (lateral >= -dwr[j])
    free = free.reshape(h, w)
    for ox, oy, hw, hh in spec.obstacles:
        c0, c1 = int((ox - hw - lo[0]) / resolution), int(math.ceil((ox + hw - lo[0]) / resolution))
        r0, r1 = int((oy - hh - lo[1]) / resolution), int(math.ceil((oy + hh - lo[1]) / resolution))
        free[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)] = False
    occ = ~free
    # Cell-centre distances overstate the distance to a wall's *edge* by half
    # a cell; subtracting it puts the zero crossing on the cell boundary, which
    # is where the occupancy grid (and the LiDAR's oracle) says the wall is.
    outside = ndimage.distance_transform_edt(free) * resolution - resolution / 2
    inside = ndimage.distance_transform_edt(occ) * resolution - resolution / 2
    sdf = np.where(free, outside, -inside).astype(np.float32)
    digest = hashlib.sha256(occ.tobytes() + np.array([resolution, *lo], np.float64).tobytes()
                            + np.array(occ.shape, np.int64).tobytes()).hexdigest()[:16]
    return RasterTrack(spec, occ, sdf, (float(lo[0]), float(lo[1])), resolution, lap_length(c), digest)


def start_grid(spec: TrackSpec, n_cars: int, rng, stagger: float = 1.0, order=None):
    """Grid start (team decision 2026-10-23): cars in a line along the lap,
    `stagger` m apart (a 0.535 m body plus a gap: 0.6 m left bodies
    overlapping on bends, found by the referee 2026-10-23), alternating sides, random grid order --
    or `order`, car i's grid slot (0 = front). Returns
    (poses (n, 3) x, y, yaw at the rear axle, start indices)."""
    c = spec.centre
    L = lap_length(c)
    k0 = int(rng.integers(len(c)))
    nrm = normals(c)
    tangent = np.stack([nrm[:, 1], -nrm[:, 0]], axis=1)
    poses, idx = [], []
    for slot in (rng.permutation(n_cars) if order is None else order):
        back = slot * stagger
        k = (k0 - int(round(back / (L / len(c))))) % len(c)
        half = 0.25 * min(spec.w_left[k], spec.w_right[k])
        side = (1 if slot % 2 == 0 else -1) * min(half, 0.3)
        p = c[k] + nrm[k] * side
        poses.append([p[0], p[1], math.atan2(tangent[k, 1], tangent[k, 0])])
        idx.append(k)
    return np.array(poses), np.array(idx)
