"""Batched safety wrapper -- the training side of neural_racer/safety_wrapper.py.

Training runs the same wrapper as the car, and every tick it overrides the
network costs reward, so the policy learns not to need it (plan A6). The
braking cap's clearance is gap_logic.swept_path_clearance, batched;
tests/test_observation_parity.py holds this to the NumPy wrapper.
"""
import math

import numpy as np
import torch

from neural_racer import observation as obs
from neural_racer.safety_wrapper import WrapperConfig, forward_window


class SafetyTorch:
    def __init__(self, cfg: WrapperConfig, angles: np.ndarray, device='cpu'):
        self.cfg = cfg
        angles = np.asarray(angles, dtype=np.float64)
        idx = np.flatnonzero(forward_window(angles, cfg))
        self.window = slice(int(idx[0]), int(idx[-1]) + 1)      # contiguous: the scan is ordered
        a = torch.as_tensor(angles[self.window], dtype=torch.float32, device=device)
        self.cos, self.sin = torch.cos(a), torch.sin(a)
        # gap_follow's escape creep (gap_follow_node.py, escape_creep_speed): inside the reserve, up to
        # this speed on the braking formula with half the reserve. 0 = off, as the car's wrapper is;
        # set only by experiments asking whether the car's wrapper should have it (2026-11-02)
        self.creep = 0.0

    def swept_clearance(self, ranges, valid, steer):
        """gap_logic.swept_path_clearance with gap_follow's settings, batched:
        ranges and valid (E, C, B), steer (E, C) -> (E, C) free length along
        the arc `steer` drives, inf when nothing is in it within the max length."""
        c = self.cfg
        hw = c.car_width / 2.0 + c.swept_clearance_margin
        front = c.wheelbase / 2.0 + c.car_length / 2.0
        rear = c.wheelbase / 2.0 - c.car_length / 2.0
        r = ranges[..., self.window]
        x = r * self.cos + c.laser_offset_x
        y = r * self.sin
        usable = valid[..., self.window] & (r > 0.0)
        k = (torch.tan(steer) / c.wheelbase).unsqueeze(-1)
        straight = k.abs() < 1e-6
        radius = 1.0 / torch.where(straight, torch.ones_like(k), k)
        ar = radius.abs()
        rho = torch.hypot(x, y - radius)
        outer = torch.hypot(torch.full_like(ar, front), ar + hw)
        # the angle swept about the turn centre from the rear axle, in the direction of travel
        start = torch.where(radius > 0.0, -math.pi / 2.0, math.pi / 2.0)
        swept = torch.remainder(torch.sign(radius) * (torch.atan2(y - radius, x) - start), 2.0 * math.pi)
        max_sweep = torch.minimum(torch.full_like(ar, c.swept_clearance_max_length + front),
                                  1.9 * math.pi * ar) / ar
        in_arc = (rho >= ar - hw) & (rho <= outer) & (swept <= max_sweep)
        in_line = (y.abs() <= hw) & (x >= rear)
        in_band = usable & torch.where(straight, in_line, in_arc)
        along = torch.where(straight, x - front, swept * ar - front)
        nearest = torch.where(in_band, along, torch.full_like(along, math.inf)).amin(dim=-1)
        return torch.where(nearest > c.swept_clearance_max_length, torch.full_like(nearest, math.inf),
                           nearest.clamp(min=0.0))

    def braking_limit(self, clearance, max_speed):
        c = self.cfg
        at = c.max_braking_decel * c.braking_latency_sec

        def formula(reserve):
            usable = (clearance - reserve).clamp(min=0.0)
            return -at + torch.sqrt(at * at + 2.0 * c.max_braking_decel * usable)

        v = formula(c.reserve_distance)
        if self.creep > 0.0:
            v = torch.maximum(v, formula(c.reserve_distance / 2.0).clamp(max=self.creep))
        v = torch.where(torch.isinf(clearance), max_speed, v)
        return torch.minimum(v, max_speed)

    def apply(self, net_steer, net_speed, ranges, range_min, range_max, prev_steer, dt, max_speed):
        """All (E, C) except ranges (E, C, B). Returns (steer, speed, stopped, intervened)."""
        return self.apply_detailed(net_steer, net_speed, ranges, range_min, range_max, prev_steer, dt,
                                   max_speed)[:4]

    def apply_detailed(self, net_steer, net_speed, ranges, range_min, range_max, prev_steer, dt, max_speed,
                       max_lateral_accel=None):
        """apply(), plus `safety_stop` (the LiDAR braking cap bound, or the wrapper
        stopped the car) and `cut` (m/s the braking cap removed). Steering slew and the curvature cap are actuator and
        grip limits the car enforces every tick; they count as `intervened`
        (as on the car) but are not what the training penalty is for.
        max_lateral_accel: (E, C) per-race cornering caps, or None for the config's."""
        c = self.cfg
        valid = torch.isfinite(ranges) & (ranges > range_min) & (ranges < range_max)
        invalid_frac = 1.0 - valid.float().mean(dim=-1)
        bad = (invalid_frac > c.max_invalid_fraction) | ~torch.isfinite(net_steer) | ~torch.isfinite(net_speed)
        asked = net_steer.clamp(-c.max_steering_angle, c.max_steering_angle)
        speed = torch.minimum(net_speed.clamp(min=0.0), max_speed)
        step = c.max_steering_rate * min(dt, c.command_slew_max_dt)
        steer = torch.minimum(torch.maximum(asked, prev_steer - step), prev_steer + step)
        clearance = torch.minimum(self.swept_clearance(ranges, valid, asked), self.swept_clearance(ranges, valid, steer))
        brake = self.braking_limit(clearance, max_speed)
        a_lat = c.max_lateral_accel if max_lateral_accel is None else max_lateral_accel
        unbraked = torch.minimum(speed, curvature_speed(steer, c.wheelbase, a_lat))
        v = torch.minimum(unbraked, brake)
        intervened = (v < speed - 1e-6) | ((steer - net_steer).abs() > 1e-6) | bad
        safety_stop = (brake < unbraked - 1e-6) | bad
        cut = (unbraked - brake).clamp(min=0.0)
        steer = torch.where(bad, torch.zeros_like(steer), steer)
        v = torch.where(bad, torch.zeros_like(v), v)
        return steer, v, bad, intervened, safety_stop, cut


def curvature_speed(steer, wheelbase: float, max_lateral_accel):
    """The wrapper's curvature cap: the fastest speed at which `steer` keeps
    the kinematic lateral acceleration within max_lateral_accel (inf when
    straight)."""
    kappa = torch.tan(steer).abs() / wheelbase
    curve = torch.sqrt(max_lateral_accel / kappa.clamp(min=1e-9))
    return torch.where(kappa > 1e-9, curve, torch.full_like(curve, math.inf))


def limits_only(cfg: WrapperConfig, net_steer, net_speed, prev_steer, dt, max_speed):
    """The actuator limits alone (steering clamp and slew, speed in [0, cap]),
    with apply_detailed()'s return shape: nothing stops, brakes or intervenes.
    The teacher trains this way; the student and the car keep the wrapper."""
    steer = net_steer.clamp(-cfg.max_steering_angle, cfg.max_steering_angle)
    step = cfg.max_steering_rate * min(dt, cfg.command_slew_max_dt)
    steer = torch.minimum(torch.maximum(steer, prev_steer - step), prev_steer + step)
    speed = torch.minimum(net_speed.clamp(min=0.0), max_speed)
    no = torch.zeros_like(steer, dtype=torch.bool)
    return steer, speed, no, no, no, torch.zeros_like(speed)


def nominal_angles():
    """The car's beam angles: UST-30LX, 1081 beams over 270 degrees."""
    return np.linspace(-math.radians(135.0), math.radians(135.0), obs.N_BEAMS)
