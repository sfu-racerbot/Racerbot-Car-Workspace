"""lidar.cast as one Triton kernel (plan item 1.3): each beam marches in
registers and a block of beams stops as soon as none of them moved, then
bisects and tests the other cars' boxes in the same launch. Training only
(the car never runs it); lidar.cast is the reference it is tested against.

The march is lidar.cast's step for step: the same relaxed sphere-tracing
step, the same hit test, the same bisection. A block keeps going one
iteration after its last beam stopped, because lidar.cast's fixed loop
re-tests a beam that stopped at the cast limit there (it can still hit), and
every later iteration of that loop changes nothing.
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:        # noqa: E722 -- no Triton (a CPU node, the car): lidar.cast falls back
    triton = None

BLOCK = 128


if triton is not None:
    @triton.jit
    def _sdf(sdf_ptr, base, ox, oy, x, y, inv_res, H: tl.constexpr, W: tl.constexpr, mask):
        """TrackPool.sample: bilinear SDF, outside the canvas is wall (-1)."""
        u = (x - ox) * inv_res - 0.5
        v = (y - oy) * inv_res - 0.5
        c0f = tl.floor(u)
        r0f = tl.floor(v)
        fu = u - c0f
        fv = v - r0f
        c0 = c0f.to(tl.int32)
        r0 = r0f.to(tl.int32)
        c1 = c0 + 1
        r1 = r0 + 1
        in_c0 = (c0 >= 0) & (c0 < W)
        in_c1 = (c1 >= 0) & (c1 < W)
        in_r0 = (r0 >= 0) & (r0 < H)
        in_r1 = (r1 >= 0) & (r1 < H)
        cc0 = tl.minimum(tl.maximum(c0, 0), W - 1)
        cc1 = tl.minimum(tl.maximum(c1, 0), W - 1)
        rr0 = tl.minimum(tl.maximum(r0, 0), H - 1)
        rr1 = tl.minimum(tl.maximum(r1, 0), H - 1)
        a = tl.load(sdf_ptr + base + rr0 * W + cc0, mask=mask, other=-1.0).to(tl.float32)
        b = tl.load(sdf_ptr + base + rr0 * W + cc1, mask=mask, other=-1.0).to(tl.float32)
        c = tl.load(sdf_ptr + base + rr1 * W + cc0, mask=mask, other=-1.0).to(tl.float32)
        d = tl.load(sdf_ptr + base + rr1 * W + cc1, mask=mask, other=-1.0).to(tl.float32)
        a = tl.where(in_r0 & in_c0, a, -1.0)
        b = tl.where(in_r0 & in_c1, b, -1.0)
        c = tl.where(in_r1 & in_c0, c, -1.0)
        d = tl.where(in_r1 & in_c1, d, -1.0)
        return a * (1 - fu) * (1 - fv) + b * fu * (1 - fv) + c * (1 - fu) * fv + d * fu * fv

    @triton.jit
    def _cast_kernel(out_ptr, sdf_ptr, origin_ptr, slot_ptr, ox_ptr, oy_ptr, yaw_ptr, ang_ptr,
                     cx_ptr, cy_ptr, cyaw_ptr, hx_ptr, hy_ptr, act_ptr,
                     inv_res, cast_limit, eps, B,
                     C: tl.constexpr, H: tl.constexpr, W: tl.constexpr, ITERS: tl.constexpr,
                     REFINE: tl.constexpr, CARS: tl.constexpr, BLOCK_B: tl.constexpr):
        pid = tl.program_id(0)                  # env * C + car
        e = pid // C
        k = pid % C
        offs = tl.program_id(1) * BLOCK_B + tl.arange(0, BLOCK_B)
        mask = offs < B
        slot = tl.load(slot_ptr + e).to(tl.int64)
        base = slot * (H * W)
        gx = tl.load(origin_ptr + slot * 2)
        gy = tl.load(origin_ptr + slot * 2 + 1)
        x0 = tl.load(ox_ptr + pid)
        y0 = tl.load(oy_ptr + pid)
        a = tl.load(yaw_ptr + pid) + tl.load(ang_ptr + offs, mask=mask, other=0.0)
        dx = tl.cos(a)
        dy = tl.sin(a)
        t = tl.zeros([BLOCK_B], dtype=tl.float32)
        last = tl.zeros([BLOCK_B], dtype=tl.float32)
        hit = tl.zeros([BLOCK_B], dtype=tl.int1)
        it = pid * 0                            # a scalar int32 tensor, so it can be loop-carried
        go = it < ITERS
        while go:
            d = _sdf(sdf_ptr, base, gx, gy, x0 + t * dx, y0 + t * dy, inv_res, H, W, mask)
            hit = hit | (d < eps)
            step = tl.maximum(d, 0.005 + 0.002 * t)
            moving = (~(hit | (t >= cast_limit))) & mask
            last = tl.where(moving, step, last)
            t = tl.where(moving, t + step, t)
            it += 1
            go = (it < ITERS) & (tl.max(moving.to(tl.int32), axis=0) > 0)
        lo = tl.maximum(t - last, 0.0)
        hi = t
        for _ in tl.static_range(REFINE):
            mid = 0.5 * (lo + hi)
            inside = _sdf(sdf_ptr, base, gx, gy, x0 + mid * dx, y0 + mid * dy, inv_res, H, W, mask) < 0.0
            hi = tl.where(hit & inside, mid, hi)
            lo = tl.where(hit & (~inside), mid, lo)
        t = tl.where(hit, hi, t)
        best = tl.minimum(tl.where(hit, t, cast_limit), cast_limit)
        if CARS:
            for j in tl.static_range(C):
                q = e * C + j
                usable = (tl.load(act_ptr + q) != 0) & (j != k)
                cyaw = tl.load(cyaw_ptr + q)
                c = tl.cos(cyaw)
                s = tl.sin(cyaw)
                px = x0 - tl.load(cx_ptr + q)
                py = y0 - tl.load(cy_ptr + q)
                lx = c * px + s * py
                ly = -s * px + c * py
                ddx = c * dx + s * dy
                ddy = -s * dx + c * dy
                hx = tl.load(hx_ptr + q)
                hy = tl.load(hy_ptr + q)
                sgx = tl.where(ddx + 1e-30 >= 0, 1.0, -1.0)
                sgy = tl.where(ddy + 1e-30 >= 0, 1.0, -1.0)
                inv_x = tl.where(tl.abs(ddx) > 1e-12, 1.0 / ddx, 1e9 * sgx)
                inv_y = tl.where(tl.abs(ddy) > 1e-12, 1.0 / ddy, 1e9 * sgy)
                tx1 = (-hx - lx) * inv_x
                tx2 = (hx - lx) * inv_x
                ty1 = (-hy - ly) * inv_y
                ty2 = (hy - ly) * inv_y
                t_near = tl.maximum(tl.minimum(tx1, tx2), tl.minimum(ty1, ty2))
                t_far = tl.minimum(tl.maximum(tx1, tx2), tl.maximum(ty1, ty2))
                box = (t_near <= t_far) & (t_far > 0) & usable
                best = tl.where(box, tl.minimum(best, tl.maximum(t_near, 0.0)), best)
        tl.store(out_ptr + pid * B + offs, best, mask=mask)


def available(device) -> bool:
    return triton is not None and str(device).startswith('cuda')


def cast(pool, slot, ox, oy, yaw, angles, cars=None, cast_limit: float = 12.0, iters: int = 64,
         eps: float = 0.005, refine: int = 5):
    """lidar.cast's contract and results, in one launch."""
    E, C = ox.shape
    B = angles.shape[0]
    out = torch.empty(E, C, B, device=ox.device, dtype=torch.float32)
    f = lambda v: v.contiguous().float()             # noqa: E731
    use_cars = cars is not None
    if cars is None:
        z = torch.zeros(E, C, device=ox.device)
        cars = (z, z, z, z, z, torch.zeros(E, C, dtype=torch.bool, device=ox.device))
    cx, cy, cyaw, hx, hy, active = cars
    grid = (E * C, triton.cdiv(B, BLOCK))
    _cast_kernel[grid](out, pool.sdf, f(pool.origin), slot.contiguous(), f(ox), f(oy), f(yaw), f(angles),
                       f(cx), f(cy), f(cyaw), f(hx), f(hy), active.contiguous().to(torch.int8),
                       1.0 / pool.res, float(cast_limit), float(eps), B,
                       C=C, H=pool.H, W=pool.W, ITERS=iters, REFINE=refine, CARS=use_cars,
                       BLOCK_B=BLOCK)
    return out
