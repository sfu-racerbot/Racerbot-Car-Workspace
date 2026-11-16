"""Batched LiDAR: sphere tracing on each track's signed-distance field, plus
exact ray-versus-rectangle hits on the other cars.

Beams that reach `cast_limit` without a hit read as a valid return at that
range (indoors the real UST-30LX nearly always hits something within its
30 m; the encoding saturates at 10 m anyway). Only dropout makes a beam
invalid, as a dark or glancing surface does on the car.
"""
import math
import os

import torch

from neural_racer_train.gpu_sim import lidar_triton

# auto: the Triton kernel on a GPU that has it, else this file's PyTorch cast;
# torch: always the PyTorch cast (the reference the kernel is tested against)
BACKEND = os.environ.get('NR_LIDAR', 'auto')


def beam_angles(n: int = 1081, fov: float = math.radians(270.0), device='cpu'):
    return torch.linspace(-fov / 2, fov / 2, n, device=device)


def ray_box(ox, oy, dx, dy, cx, cy, cyaw, hx, hy):
    """Distance along unit rays (o, d) to oriented boxes (centre c, yaw, half
    sizes); inf on a miss. Broadcasts over leading dims."""
    c, s = torch.cos(cyaw), torch.sin(cyaw)
    px, py = ox - cx, oy - cy
    lx, ly = c * px + s * py, -s * px + c * py          # origin in the box frame
    ddx, ddy = c * dx + s * dy, -s * dx + c * dy
    big = torch.full_like(lx, 1e9)
    inv_x = torch.where(ddx.abs() > 1e-12, 1.0 / ddx, big * torch.sign(ddx + 1e-30))
    inv_y = torch.where(ddy.abs() > 1e-12, 1.0 / ddy, big * torch.sign(ddy + 1e-30))
    tx1, tx2 = (-hx - lx) * inv_x, (hx - lx) * inv_x
    ty1, ty2 = (-hy - ly) * inv_y, (hy - ly) * inv_y
    t_near = torch.maximum(torch.minimum(tx1, tx2), torch.minimum(ty1, ty2))
    t_far = torch.minimum(torch.maximum(tx1, tx2), torch.maximum(ty1, ty2))
    hit = (t_near <= t_far) & (t_far > 0)
    return torch.where(hit, t_near.clamp(min=0.0), torch.full_like(t_near, math.inf))


def cast(pool, slot, ox, oy, yaw, angles, cars=None, cast_limit: float = 12.0,
         iters: int = 64, eps: float = 0.005, refine: int = 5):
    """Ranges (E, C, B) for LiDARs at (ox, oy, yaw) of shape (E, C).

    cars: optional (cx, cy, cyaw, hx, hy, active) each (E, C) -- every car's
    body; a car never sees its own.
    """
    if BACKEND != 'torch' and lidar_triton.available(ox.device):
        return lidar_triton.cast(pool, slot, ox, oy, yaw, angles, cars, cast_limit, iters, eps, refine)
    return cast_reference(pool, slot, ox, oy, yaw, angles, cars, cast_limit, iters, eps, refine)


def cast_reference(pool, slot, ox, oy, yaw, angles, cars=None, cast_limit: float = 12.0,
                   iters: int = 64, eps: float = 0.005, refine: int = 5):
    """cast() in PyTorch: every beam takes all `iters` steps (torch.where
    cannot stop one early). The reference for lidar_triton."""
    E, C = ox.shape
    B = angles.shape[0]
    a = yaw.unsqueeze(-1) + angles                      # (E, C, B)
    dx, dy = torch.cos(a), torch.sin(a)
    x0, y0 = ox.unsqueeze(-1).expand_as(a), oy.unsqueeze(-1).expand_as(a)
    sl = slot.view(E, 1, 1).expand_as(a)
    t = torch.zeros_like(a)
    last = torch.zeros_like(a)
    hit = torch.zeros_like(a, dtype=torch.bool)
    for _ in range(iters):
        d = pool.sample(sl, x0 + t * dx, y0 + t * dy)
        hit = hit | (d < eps)
        # relaxed sphere tracing: a small floor on the step keeps grazing rays
        # moving; the overshoot it allows is bounded by the floor itself, and
        # the bisection below takes most of it back
        step = torch.maximum(d, 0.005 + 0.002 * t)
        moving = ~(hit | (t >= cast_limit))
        last = torch.where(moving, step, last)
        t = torch.where(moving, t + step, t)
    # bisect each hit between the last free point and where it stopped
    lo, hi = (t - last).clamp(min=0.0), t
    for _ in range(refine):
        mid = 0.5 * (lo + hi)
        inside = pool.sample(sl, x0 + mid * dx, y0 + mid * dy) < 0.0
        hi = torch.where(hit & inside, mid, hi)
        lo = torch.where(hit & ~inside, mid, lo)
    t = torch.where(hit, hi, t)
    walls = torch.where(hit, t, torch.full_like(t, cast_limit)).clamp(max=cast_limit)
    if cars is None:
        return walls
    cx, cy, cyaw, hx, hy, active = cars
    best = walls
    for j in range(C):
        tj = ray_box(x0, y0, dx, dy, cx[:, j].view(E, 1, 1), cy[:, j].view(E, 1, 1),
                     cyaw[:, j].view(E, 1, 1), hx[:, j].view(E, 1, 1), hy[:, j].view(E, 1, 1))
        own = torch.zeros(C, dtype=torch.bool, device=ox.device)
        own[j] = True
        usable = active[:, j].view(E, 1, 1) & ~own.view(1, C, 1)
        best = torch.where(usable, torch.minimum(best, tj), best)
    return best


def line_of_sight(pool, slot, ox, oy, px, py, iters: int = 64, eps: float = 0.005):
    """True where the segment from (ox, oy) to (px, py) crosses no wall, by
    sphere tracing toward the point (all broadcast together; slot broadcasts
    too). It errs toward blocked: the step floor (2 cm + 0.4%) can step into
    a wall a segment only grazes, and a segment the march has not finished in
    `iters` steps -- a long one along a wall -- counts as blocked."""
    dx, dy = px - ox, py - oy
    dist = torch.sqrt(dx * dx + dy * dy)
    ux, uy = dx / dist.clamp(min=1e-6), dy / dist.clamp(min=1e-6)
    shape = dist.shape
    x0, y0 = ox.expand(shape), oy.expand(shape)
    sl = slot.expand(shape)
    t = torch.zeros(shape, device=dist.device)
    blocked = torch.zeros(shape, dtype=torch.bool, device=dist.device)
    for _ in range(iters):
        d = pool.sample(sl, x0 + t * ux, y0 + t * uy)
        blocked = blocked | ((d < eps) & (t < dist))
        t = torch.where(blocked, t, torch.minimum(t + torch.maximum(d, 0.02 + 0.004 * t), dist))
    return ~blocked & (t >= dist)


def add_noise(ranges, noise_std, dropout, gen, range_max_invalid: float, also_drop=None):
    """Gaussian range noise, and dropped beams set to range_max_invalid (which
    the observation reads as 'no return'): each beam alone with probability
    `dropout`, and wherever also_drop (same shape, bool) is set."""
    n = torch.randn(ranges.shape, generator=gen, device=gen.device).to(ranges.device)
    r = ranges + n * noise_std.view(-1, *([1] * (ranges.dim() - 1)))
    u = torch.rand(ranges.shape, generator=gen, device=gen.device).to(ranges.device)
    drop = u < dropout.view(-1, *([1] * (ranges.dim() - 1)))
    if also_drop is not None:
        drop = drop | also_drop
    return torch.where(drop, torch.full_like(r, range_max_invalid), r.clamp(min=0.05))


SECTOR_WIDTH = (4.0, 60.0)      # beams, drawn log-uniform: car 1's runs had a median of 9 and a p90 of 56
SECTOR_CANDIDATES = 3


def sector_dropout(shape, sectors, persist, persist_deg, persist_width, angles, gen):
    """(E, C, B) bool: beams lost in a dropout sector, as car 1's LiDAR loses
    them -- in runs, not one by one. Transient: SECTOR_CANDIDATES sectors a
    scan, each there with probability sectors / SECTOR_CANDIDATES (so `sectors`
    on average), anywhere, SECTOR_WIDTH wide. Persistent: one sector per
    episode at persist_deg, persist_width beams wide, there in each scan with
    probability `persist` (<= 0: never). sectors, persist, persist_deg and
    persist_width are (E,). Built as +1/-1 at each sector's ends and a cumsum,
    not a (E, C, K, B) mask."""
    E, C, B = shape
    dev = angles.device
    K = SECTOR_CANDIDATES
    u = torch.rand(E, C, K + 1, 3, generator=gen, device=gen.device).to(dev)
    lo_w, hi_w = SECTOR_WIDTH
    centre = u[..., 1] * B
    width = lo_w * (hi_w / lo_w) ** u[..., 2]
    present = u[..., 0] < (sectors.view(-1, 1, 1) / K)
    step = float(angles[1] - angles[0])
    p_centre = (torch.deg2rad(persist_deg) - float(angles[0])) / step
    centre[..., K] = p_centre.view(-1, 1)
    width[..., K] = persist_width.view(-1, 1)
    present[..., K] = u[..., K, 0] < persist.view(-1, 1)
    lo = (centre - width / 2).round().clamp(0, B).long()
    hi = (centre + width / 2).round().clamp(0, B).long()
    lo = torch.where(present, lo, torch.full_like(lo, B))
    hi = torch.where(present, hi, torch.full_like(hi, B))
    diff = torch.zeros(E, C, B + 1, device=dev, dtype=torch.int32)
    diff.scatter_add_(2, lo, torch.ones_like(lo, dtype=torch.int32))
    diff.scatter_add_(2, hi, -torch.ones_like(hi, dtype=torch.int32))
    return diff.cumsum(-1)[..., :B] > 0
