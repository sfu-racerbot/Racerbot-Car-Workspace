"""Prototype used only to verify the plan's math and produce reference numbers. Not the deliverable package."""
import numpy as np

# ---------- geometry ----------
def reflect_dir(u, n):
    return u - 2.0 * (u @ n) * n

def ray_segments(o, d, A, B):
    """Ray o + t d against segments A[k]->B[k]. Returns t (inf if miss) per segment and s in [0,1]."""
    e = B - A
    den = d[0] * e[:, 1] - d[1] * e[:, 0]
    w = A - o
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / den
        s = (w[:, 0] * d[1] - w[:, 1] * d[0]) / den
    ok = (np.abs(den) > 1e-12) & (t > 1e-6) & (s >= 0) & (s <= 1)
    return np.where(ok, t, np.inf), s

def rect_segments(c, yaw, L, Wd):
    hx, hy = L / 2, Wd / 2
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    pts = (R @ np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]]).T).T + c
    return pts, np.roll(pts, -1, axis=0)

# ---------- scene: L corner, corridor A along +x (y in [0,W]), corridor B along +y (x in [0,W]) ----------
class Scene:
    def __init__(self, W=1.5, c=None, panel_len=0.80, far=10.0,
                 pattern=((0.05, 0), (0.20, 1), (0.05, 0), (0.20, 1), (0.05, 0), (0.20, 1), (0.05, 0))):
        self.W = W
        c = panel_len / np.sqrt(2) if c is None else c
        self.P0 = np.array([W - c, 0.0]); self.P1 = np.array([W, c])     # chamfer panel endpoints
        t = (self.P1 - self.P0); self.L = np.linalg.norm(t); self.t = t / self.L
        self.n = np.array([-self.t[1], self.t[0]])                         # normal into the track (-1,1)/sqrt2
        A = [(-far, 0), (-far, W), (W, c), (0, W)]
        B = [(W - c, 0), (0, W), (W, far), (0, far)]
        self.A = np.array(A, float); self.B = np.array(B, float)           # diffuse walls
        edges = np.cumsum([0.0] + [p[0] for p in pattern]); self.edges = edges
        self.kind = np.array([p[1] for p in pattern])                      # 1 mirror, 0 matte
        assert abs(edges[-1] - self.L) < 1e-9, (edges[-1], self.L)

    def panel_kind(self, s_m, half_fp=0.0):
        """0 matte / 1 mirror for a hit at arclength s_m along the panel. A beam footprint of half-width
        half_fp (measured along the panel) that touches matte returns the matte (first-return assumption)."""
        i = np.clip(np.searchsorted(self.edges, s_m, side='right') - 1, 0, len(self.kind) - 1)
        if self.kind[i] == 0:
            return 0
        if s_m - half_fp < self.edges[i] or s_m + half_fp > self.edges[i + 1]:
            return 0
        return 1

def trace_scan(sc, pose, angles, car=None, sigma=0.005, rng=None, fp_k=0.0, max_range=30.0, no_return=65.533, max_ghost_path=1e9):
    """pose = (x, y, yaw) of the LASER. car = (cx, cy, yaw, L, W) or None. One specular bounce off panel mirror cells."""
    rng = rng or np.random.default_rng(0)
    o = np.array(pose[:2]); out = np.full(len(angles), no_return); truth = np.full(len(angles), 5)  # 0 wall,1 matte stripe,2 ghost-static,3 ghost-car,4 direct car,5 no return
    A, B = sc.A, sc.B
    if car is not None:
        ca, cb = rect_segments(np.array(car[:2]), car[2], car[3], car[4])
        A2 = np.vstack([A, ca]); B2 = np.vstack([B, cb]); ncar = 4
    else:
        A2, B2, ncar = A, B, 0
    PA = sc.P0[None]; PB = sc.P1[None]
    for i, a in enumerate(angles):
        d = np.array([np.cos(pose[2] + a), np.sin(pose[2] + a)])
        t, _ = ray_segments(o, d, A2, B2); k = int(np.argmin(t)); tw = t[k]
        tp, sp = ray_segments(o, d, PA, PB); tp = tp[0]
        if tp < tw and (d @ sc.n) < 0:                       # hits the panel from the front
            s_m = sp[0] * sc.L
            cosi = -(d @ sc.n)
            half_fp = 0.5 * fp_k * tp / max(cosi, 1e-3)
            if sc.panel_kind(s_m, half_fp) == 0:
                out[i] = tp; truth[i] = 1
            else:
                w = o + tp * d; d2 = reflect_dir(d, sc.n)
                t2, _ = ray_segments(w, d2, A2, B2); k2 = int(np.argmin(t2))
                if np.isfinite(t2[k2]) and tp + t2[k2] <= min(max_range, max_ghost_path):
                    out[i] = tp + t2[k2]; truth[i] = 3 if (ncar and k2 >= len(A)) else 2
        elif tp < tw:                                        # panel seen from behind: a plain wall
            out[i] = tp; truth[i] = 0
        elif np.isfinite(tw) and tw <= max_range:
            out[i] = tw; truth[i] = 4 if (ncar and k >= len(A)) else 0
    ok = out < max_range
    out[ok] += rng.normal(0, sigma, ok.sum())
    return out, truth

# ---------- detector (scan-local) ----------
def fit_line_ransac(P, tol=0.02, iters=300, rng=None, t_prior=None, max_angle=np.radians(20)):
    rng = rng or np.random.default_rng(1)
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(P), 2, replace=False)
        t = P[j] - P[i]; L = np.linalg.norm(t)
        if L < 0.15: continue
        t /= L; n = np.array([-t[1], t[0]])
        if t_prior is not None and abs(t @ t_prior) < np.cos(max_angle): continue
        dist = np.abs((P - P[i]) @ n); inl = dist < tol
        if best is None or inl.sum() > best.sum(): best = inl
    if best is None: return None
    for tol_k in (tol, 0.5 * tol, 0.5 * tol):                # refit: total least squares, tightening
        Q = P[best]; c = Q.mean(0); _, _, Vt = np.linalg.svd(Q - c); t = Vt[0]; n = np.array([-t[1], t[0]])
        d = np.abs((P - c) @ n); s_all = (P - c) @ t
        s_in = s_all[best]
        new = (d < tol_k) & (s_all >= s_in.min() - 0.02) & (s_all <= s_in.max() + 0.02)
        if new.sum() < 4: break
        best = new
    Q = P[best]; c = Q.mean(0); _, _, Vt = np.linalg.svd(Q - c); t = Vt[0]; n = np.array([-t[1], t[0]])
    return c, t, n, best

def unfold_scan(ranges, angles, prior_p0, prior_p1, gate_perp=0.5, margin_ang=0.15, tol=0.02, tau_g=0.10, max_range=30.0, rng=None):
    """Everything in the LASER frame. prior_p0/p1 = predicted panel endpoints (from pose + panel registry)."""
    pts = np.c_[ranges * np.cos(angles), ranges * np.sin(angles)]
    a0, a1 = sorted([np.arctan2(prior_p0[1], prior_p0[0]), np.arctan2(prior_p1[1], prior_p1[0])])
    sel = (angles > a0 - margin_ang) & (angles < a1 + margin_ang) & (ranges < max_range)
    tp = prior_p1 - prior_p0; Lp = np.linalg.norm(tp); tp /= Lp; npr = np.array([-tp[1], tp[0]])
    perp = (pts - prior_p0) @ npr; along = (pts - prior_p0) @ tp
    near = sel & (np.abs(perp) < gate_perp) & (along > -gate_perp) & (along < Lp + gate_perp)
    if near.sum() < 6: return None
    idx = np.flatnonzero(near)
    fit = fit_line_ransac(pts[idx], tol=tol, rng=rng, t_prior=tp)
    if fit is None: return None
    c, t, n, inl = fit
    if n @ (-c) < 0: n = -n                                  # normal faces the sensor
    if t @ tp < 0: t = -t
    s_in = (pts[idx][inl] - c) @ t
    s_lo, s_hi = s_in.min(), s_in.max()                      # fitted extent (end stripes)
    if s_hi - s_lo < 0.5 * Lp or inl.sum() < 6: return None
    u = np.c_[np.cos(angles), np.sin(angles)]
    den = u @ n
    with np.errstate(divide='ignore', invalid='ignore'):
        rhat = (c @ n) / den                                 # beam-line intersection range
    w = rhat[:, None] * u
    s_w = (w - c) @ t
    on_panel = (den < -0.05) & (rhat > 0) & (s_w > s_lo) & (s_w < s_hi)
    e = ranges - rhat
    ghost = on_panel & (e > tau_g) & (ranges < max_range)
    u2 = u - 2.0 * den[:, None] * n[None, :]
    p = w + e[:, None] * u2
    return dict(c=c, t=t, n=n, s_lo=s_lo, s_hi=s_hi, ghost=ghost, p=p, e=e, rhat=rhat, on_panel=on_panel, ninl=int(inl.sum()))

def to_panel_frame(res, P):
    """Panel frame: origin at fitted low end, x along t, y along n (into the track; hidden points have y>0)."""
    o = res['c'] + res['s_lo'] * res['t']
    return np.c_[(P - o) @ res['t'], (P - o) @ res['n']]

def cluster(P, eps=0.25, min_pts=3):
    n = len(P); lab = -np.ones(n, int); cur = 0
    for i in range(n):
        if lab[i] >= 0: continue
        stack = [i]; lab[i] = cur
        while stack:
            j = stack.pop(); nb = np.flatnonzero((np.linalg.norm(P - P[j], axis=1) < eps) & (lab < 0))
            lab[nb] = cur; stack.extend(nb.tolist())
        cur += 1
    return [np.flatnonzero(lab == k) for k in range(cur) if (lab == k).sum() >= min_pts]

class Background:
    def __init__(self, cell=0.05, size=8.0):
        self.cell = cell; self.n = int(size / cell); self.cnt = np.zeros((self.n, self.n), int); self.size = size
    def _ij(self, P):
        ij = np.floor((P + np.array([self.size / 2, 0.5])) / self.cell).astype(int)   # x in [-4,4), y in [-0.5,7.5)
        ok = (ij >= 0).all(1) & (ij < self.n).all(1); return ij, ok
    def add(self, P):
        ij, ok = self._ij(P); np.add.at(self.cnt, (ij[ok, 0], ij[ok, 1]), 1)
    def finalize(self, min_hits=2, dilate=2):
        from scipy.ndimage import binary_dilation
        self.static = binary_dilation(self.cnt >= min_hits, iterations=dilate)
    def is_static(self, P):
        ij, ok = self._ij(P); out = np.ones(len(P), bool); out[ok] = self.static[ij[ok, 0], ij[ok, 1]]; return out

def outline_distance(P, car):
    a, b = rect_segments(np.array(car[:2]), car[2], car[3], car[4])
    d = np.full(len(P), np.inf)
    for k in range(4):
        e = b[k] - a[k]; s = np.clip(((P - a[k]) @ e) / (e @ e), 0, 1)
        d = np.minimum(d, np.linalg.norm(P - (a[k] + s[:, None] * e), axis=1))
    return d
