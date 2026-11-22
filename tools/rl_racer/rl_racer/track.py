"""Periodic Hermite centerlines with chord arc-length coordinates; n points left."""
from pathlib import Path

import numpy as np


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class Track:
    def __init__(self, points, width=1.0):
        """Points: x,y[,w_right,w_left]; width fallback is FULL track width."""
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] not in (2, 4) or len(p) < 3:
            raise ValueError('centerline needs at least three rows of 2 or 4 columns')
        if not np.isfinite(p).all() or not np.isfinite(width) or width <= 0:
            raise ValueError('centerline and positive width must be finite')
        if np.linalg.norm(p[0, :2] - p[-1, :2]) < 1e-9:
            p = p[:-1]
        self.xy = p[:, :2].copy()
        self.delta = np.roll(self.xy, -1, axis=0) - self.xy
        self.lengths = np.linalg.norm(self.delta, axis=1)
        if len(p) < 3 or np.any(self.lengths < 1e-9):
            raise ValueError('centerline has degenerate segments')
        self.s = np.r_[0., np.cumsum(self.lengths)[:-1]]
        self.length = self.lengths.sum()
        self.widths = p[:, 2:].copy() if p.shape[1] == 4 else np.full((len(p), 2), width / 2)
        if np.any(self.widths <= 0):
            raise ValueError('half widths must be positive')
        headings = np.arctan2(self.delta[:, 1], self.delta[:, 0])
        self.curvature = wrap(headings - np.roll(headings, 1)) / (
            (self.lengths + np.roll(self.lengths, 1)) / 2)
        units = self.delta / self.lengths[:, None]
        self.tangent = units + np.roll(units, 1, axis=0)
        norms = np.linalg.norm(self.tangent, axis=1, keepdims=True)
        self.tangent = np.divide(self.tangent, norms, out=units.copy(), where=norms > 1e-9)

    def _geometry(self, i, t):
        j = (i + 1) % len(self.xy)
        h = self.lengths[i, None]
        t = t[..., None]
        a, b = self.xy[i], self.xy[j]
        u, v = self.tangent[i] * h, self.tangent[j] * h
        xy = (2*t**3 - 3*t**2 + 1)*a + (t**3 - 2*t**2 + t)*u + (-2*t**3 + 3*t**2)*b + (t**3 - t**2)*v
        derivative = (6*t**2 - 6*t)*a + (3*t**2 - 4*t + 1)*u + (-6*t**2 + 6*t)*b + (3*t**2 - 2*t)*v
        return xy, derivative / h

    @classmethod
    def from_csv(cls, path: str | Path, width=1.0):
        """Read headered x_m,y_m or x,y[,w_right,w_left] CSV files."""
        data = np.genfromtxt(path, delimiter=',', names=True, encoding='utf-8')
        names = data.dtype.names
        if names not in (('x_m', 'y_m'), ('x', 'y'), ('x', 'y', 'w_right', 'w_left')):
            raise ValueError('unsupported centerline CSV columns')
        return cls(np.column_stack([np.atleast_1d(data[n]) for n in names]), width)

    def sample(self, s):
        s = np.asarray(s, dtype=float)
        if not np.isfinite(s).all():
            raise ValueError('arc lengths must be finite')
        q = s % self.length
        i = np.searchsorted(self.s, q, side='right') - 1
        t = (q - self.s[i]) / self.lengths[i]
        j = (i + 1) % len(self.xy)
        xy, derivative = self._geometry(i, t)
        yaw = np.arctan2(derivative[..., 1], derivative[..., 0])
        curvature = (1 - t) * self.curvature[i] + t * self.curvature[j]
        widths = (1 - t[..., None]) * self.widths[i] + t[..., None] * self.widths[j]
        return xy, yaw, curvature, widths

    def project(self, poses):
        """Project (...,3) x,y,yaw onto nearest segment; return s,n,heading error."""
        p = np.asarray(poses, dtype=float)
        if p.ndim < 1 or p.shape[-1] != 3 or not np.isfinite(p).all():
            raise ValueError('poses must be finite (...,3)')
        d = p[..., None, :2] - self.xy
        t = np.clip(np.sum(d * self.delta, axis=-1) / self.lengths**2, 0, 1)
        residual = d - t[..., None] * self.delta
        i = np.argmin(np.sum(residual**2, axis=-1), axis=-1)
        chosen_t = np.take_along_axis(t, i[..., None], axis=-1)[..., 0]
        s = self.s[i] + chosen_t * self.lengths[i]
        for _ in range(8):
            q = s % self.length
            i = np.searchsorted(self.s, q, side='right') - 1
            chosen_t = (q - self.s[i]) / self.lengths[i]
            foot, tangent = self._geometry(i, chosen_t)
            s += np.sum((p[..., :2] - foot) * tangent, axis=-1) / np.sum(tangent**2, axis=-1)
        xy, yaw, _, _ = self.sample(s)
        foot = xy
        unit = np.stack((np.cos(yaw), np.sin(yaw)), axis=-1)
        offset = p[..., :2] - foot
        n = unit[..., 0] * offset[..., 1] - unit[..., 1] * offset[..., 0]
        yaw = np.arctan2(unit[..., 1], unit[..., 0])
        return s % self.length, n, wrap(p[..., 2] - yaw)

    def preview(self, s, k, spacing=0.3):
        """Return (...,K) curvature and FULL width, starting one spacing ahead."""
        if not isinstance(k, int) or k < 1 or not np.isfinite(spacing) or spacing <= 0:
            raise ValueError('preview needs positive integer k and spacing')
        _, _, c, w = self.sample(np.asarray(s)[..., None] + spacing * np.arange(1, k + 1))
        return c, w.sum(axis=-1)
