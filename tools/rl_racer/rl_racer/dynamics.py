"""Single-track dynamics with wheel speed expressed as tyre surface speed (m/s).

Simplified Magic Formula B,C,D,E, static axle loads and equal AWD slip demand.
Radial friction-ellipse scaling couples longitudinal and lateral forces. Unlike
MF6.1 this has no load/camber dependence, aligning moment or transient slip.
Below 0.5 m/s a kinematic relaxation avoids singular/stiff slip dynamics.
"""
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'vehicle_rb1.yaml'


@dataclass(frozen=True)
class VehicleParams:
    mass: float
    wheelbase: float
    inertia: float
    front_fraction: float
    friction: float
    lateral_b_front: float
    lateral_b_rear: float
    lateral_c: float
    lateral_e: float
    longitudinal_b: float
    longitudinal_c: float
    longitudinal_e: float
    steer_min: float
    steer_max: float
    steer_tau: float
    steer_rate: float
    wheel_tau: float
    wheel_rate: float
    v_max: float
    slip_floor: float
    substep: float

    def __post_init__(self):
        values = np.array([getattr(self, f.name) for f in fields(self)])
        if not np.isfinite(values).all():
            raise ValueError('vehicle parameters must be finite')
        signed = {'steer_min', 'lateral_e', 'longitudinal_e'}
        if any(getattr(self, f.name) <= 0 for f in fields(self) if f.name not in signed):
            raise ValueError('vehicle scales must be positive')
        if not 0 < self.front_fraction < 1 or not self.steer_min < 0 < self.steer_max < np.pi / 2:
            raise ValueError('invalid geometry or steering limits')

    @classmethod
    def from_yaml(cls, path=CONFIG):
        with open(path, encoding='utf-8') as stream:
            return cls(**yaml.safe_load(stream)['vehicle'])


def magic(slip, b, c, d, e):
    z = b * slip
    return d * np.sin(c * np.arctan(z - e * (z - np.arctan(z))))


class SingleTrack:
    """State columns: x,y,yaw,vx,vy,r,steer,wheel_surface_speed."""
    def __init__(self, params=None):
        self.params = params or VehicleParams.from_yaml()

    def derivative(self, state, steering_ref, wheel_ref, friction, parameters=None):
        p = SimpleNamespace(**(vars(self.params) | (parameters or {})))
        x, y, yaw, vx, vy, r, steer, wheel = state.T
        lf = p.wheelbase * p.front_fraction
        lr = p.wheelbase - lf
        denom = np.maximum(np.abs(vx), p.slip_floor)
        alpha_f = steer - np.arctan2(vy + lf * r, denom)
        alpha_r = -np.arctan2(vy - lr * r, denom)
        slip = (wheel - vx) / denom
        forces = []
        for load, alpha, b in ((p.mass * 9.81 * lr / p.wheelbase, alpha_f, p.lateral_b_front),
                               (p.mass * 9.81 * lf / p.wheelbase, alpha_r, p.lateral_b_rear)):
            peak = friction * load
            fx = magic(slip, p.longitudinal_b, p.longitudinal_c, peak, p.longitudinal_e)
            fy = magic(alpha, b, p.lateral_c, peak, p.lateral_e)
            scale = np.maximum(1., np.hypot(fx, fy) / peak)
            forces.append((fx / scale, fy / scale))
        (fxf, fyf), (fxr, fyr) = forces
        fx = fxf * np.cos(steer) - fyf * np.sin(steer) + fxr
        fy_front = fyf * np.cos(steer) + fxf * np.sin(steer)
        ax = fx / p.mass + r * vy
        ay = (fy_front + fyr) / p.mass - r * vx
        ar = (lf * fy_front - lr * fyr) / p.inertia
        blend = np.clip((np.abs(vx) - 0.5) / 0.5, 0, 1)
        target_r = vx * np.tan(steer) / p.wheelbase
        ay = blend * ay + (1 - blend) * (lr * target_r - vy) / 0.05
        ar = blend * ar + (1 - blend) * (target_r - r) / 0.05
        out = np.column_stack((vx * np.cos(yaw) - vy * np.sin(yaw),
                               vx * np.sin(yaw) + vy * np.cos(yaw), r, ax, ay, ar,
                               np.clip((steering_ref - steer) / p.steer_tau, -p.steer_rate, p.steer_rate),
                               np.clip((wheel_ref - wheel) / p.wheel_tau, -p.wheel_rate, p.wheel_rate)))
        return out

    def step(self, state, steering_ref, wheel_ref, dt=0.05, friction=None, parameters=None):
        state = np.asarray(state, dtype=float)
        if state.ndim != 2 or state.shape[1] != 8 or not np.isfinite(state).all():
            raise ValueError('state must be finite (N,8)')
        p = self.params
        refs = [np.broadcast_to(np.asarray(v, dtype=float), (len(state),))
                for v in (steering_ref, wheel_ref, p.friction if friction is None else friction)]
        if not all(np.isfinite(v).all() for v in refs) or np.any(refs[2] <= 0) or not np.isfinite(dt) or dt <= 0:
            raise ValueError('references, positive friction and dt must be finite')
        sr = np.clip(refs[0], p.steer_min, p.steer_max)
        wr = np.clip(refs[1], 0, p.v_max)
        count = int(np.ceil(dt / p.substep))
        h = dt / count
        z = state.copy()
        for _ in range(count):
            mid = z + h / 2 * self.derivative(z, sr, wr, refs[2], parameters)
            z += h * self.derivative(mid, sr, wr, refs[2], parameters)
            z[:, 6] = np.clip(z[:, 6], p.steer_min, p.steer_max)
            z[:, 7] = np.clip(z[:, 7], 0, p.v_max)
        z[:, 2] = (z[:, 2] + np.pi) % (2 * np.pi) - np.pi
        z[:, 6] = np.clip(z[:, 6], state[:, 6] - p.steer_rate * dt, state[:, 6] + p.steer_rate * dt)
        z[:, 7] = np.clip(z[:, 7], state[:, 7] - p.wheel_rate * dt, state[:, 7] + p.wheel_rate * dt)
        return z
