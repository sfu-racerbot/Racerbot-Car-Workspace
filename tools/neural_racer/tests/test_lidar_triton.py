"""The Triton LiDAR kernel (plan item 1.3) against the PyTorch cast it
replaces, on generated tracks with four cars. GPU only: skipped elsewhere.

Runner: PYTHONPATH=<torch> python3 -m pytest tools/neural_racer/tests/test_lidar_triton.py -v   (GPU)
"""
import math

import conftest  # noqa: F401
import pytest
import torch

from neural_racer_train import learn
from neural_racer_train.gpu_sim import lidar, lidar_triton
from neural_racer_train.track_pool import TrackPool

pytestmark = pytest.mark.skipif(not lidar_triton.available('cuda' if torch.cuda.is_available() else 'cpu'),
                                reason='needs a CUDA device and Triton')


def _scene(E=64, C=4, seed=0):
    dev = 'cuda'
    pool = TrackPool(3, device=dev)
    rasters = [learn._generate(s) for s in (11, 12, 13)]
    for k, r in enumerate(rasters):
        pool.insert(r, k)
    g = torch.Generator().manual_seed(seed)
    slot = torch.randint(0, 3, (E,), generator=g)
    pts = []
    for e in range(E):
        c = rasters[int(slot[e])].spec.centre
        idx = torch.randint(0, len(c), (C,), generator=g)
        pts.append(torch.as_tensor(c[idx.numpy()], dtype=torch.float32))
    xy = torch.stack(pts) + 0.3 * torch.randn(E, C, 2, generator=g)
    yaw = (torch.rand(E, C, generator=g) * 2 - 1) * math.pi
    active = torch.rand(E, C, generator=g) > 0.2
    to = lambda v: v.to(dev)      # noqa: E731
    ox, oy, yaw, slot, active = to(xy[..., 0]), to(xy[..., 1]), to(yaw), to(slot), to(active)
    hx, hy = torch.full_like(ox, 0.29), torch.full_like(ox, 0.155)
    cars = (ox + 0.1 * torch.cos(yaw), oy + 0.1 * torch.sin(yaw), yaw, hx, hy, active)
    return pool, slot, ox, oy, yaw, cars


@pytest.mark.parametrize('with_cars', [True, False])
def test_kernel_matches_the_pytorch_cast(with_cars):
    pool, slot, ox, oy, yaw, cars = _scene()
    angles = lidar.beam_angles(1081, device='cuda')
    kw = dict(cars=cars if with_cars else None, cast_limit=12.0, iters=48)
    ref = lidar.cast_reference(pool, slot, ox, oy, yaw, angles, **kw)
    got = lidar_triton.cast(pool, slot, ox, oy, yaw, angles, **kw)
    diff = (got - ref).abs()
    assert float((diff < 2e-3).float().mean()) > 0.999, float((diff < 2e-3).float().mean())
    assert float(diff.max()) < 0.06, float(diff.max())
    assert float((got - ref).mean().abs()) < 1e-4                  # no bias


def test_kernel_is_what_cast_uses_on_the_gpu():
    pool, slot, ox, oy, yaw, cars = _scene(E=8)
    angles = lidar.beam_angles(1081, device='cuda')
    assert torch.equal(lidar.cast(pool, slot, ox, oy, yaw, angles, cars),
                       lidar_triton.cast(pool, slot, ox, oy, yaw, angles, cars))
