"""Performance gate: the on-car tick fits its budget. ROS-free.

Runner: python3 -m pytest src/neural_racer/test/test_latency.py -v   (unsourced, on the Jetson)

Budget: p99 of observation + forward pass + safety wrapper <= 5 ms, 20% of
the 25 ms tick (docs/superpowers/specs/2026-10-23-neural-racer-plan.md,
Phase 2). Measured reference: the forward pass alone at 3.50 ms p99 on car
1's Jetson, one core, 2026-10-23. This is a timing gate, not a behaviour
test; Phase 6 repeats it with the whole race stack running, and that number
decides between scan_width full and half.
"""
import time

import nr_testutil
import numpy as np

from neural_racer import observation as obs
from neural_racer.policy_numpy import PolicyNumpy
from neural_racer.safety_wrapper import SafetyWrapper, WrapperConfig, scan_angles

BUDGET_P99_MS = 5.0


def _tick_times(width, n=1500, scans=None, bindings=None):
    policy = PolicyNumpy(nr_testutil.random_weights(0, width=width))
    wrapper = SafetyWrapper(WrapperConfig())
    angles = scan_angles(nr_testutil.UST_ANGLE_MIN, nr_testutil.UST_INCREMENT, 1081)
    history = obs.History()
    rng = np.random.default_rng(0)
    if scans is None:
        scans = rng.uniform(0.3, 12.0, (64, 1081))
    times = []
    for i in range(n + 100):
        r = scans[i % 64]
        t0 = time.perf_counter()
        enc, valid = obs.pool_scan(r, 0.02, 30.0)
        history.push_tick(enc, valid, 1.0, (0.1, 0.0, 0.2))
        o = obs.build_observation(history, imu_present=True, mode=0, v_cap=1.0)
        out = policy.step(o)
        res = wrapper.apply(out.steering, out.speed, out.ego_path, r, angles, 0.02, 30.0, 0.0, 0.025)
        history.push_command(res.steering, res.speed)
        if i >= 100:
            times.append(time.perf_counter() - t0)
            if bindings is not None:
                bindings.append(res.binding)
    return np.array(times) * 1e3


def _beside_a_wall(n=64):
    """A 1.2 m corridor, the car 2.5 cm off its right wall (0.18 m from the LiDAR), with 2 mm of
    range noise: the braking cap reads zero, and every return of the right wall within reach
    of the body's arc is traced. (The test network steers slightly left, away from it.)"""
    rng = np.random.default_rng(1)
    a = scan_angles(nr_testutil.UST_ANGLE_MIN, nr_testutil.UST_INCREMENT, 1081)
    with np.errstate(divide='ignore'):
        wall = np.where(a < 0.0, 0.18 / np.abs(np.sin(a)), 1.02 / np.abs(np.sin(a)))
    return np.minimum(wall, 10.0) + rng.normal(0.0, 0.002, (n, 1081))


def test_full_width_tick_within_budget():
    t = _tick_times(nr_testutil.FULL)
    p99 = float(np.percentile(t, 99))
    print(f'full width: median {np.median(t):.2f} ms, p99 {p99:.2f} ms')
    assert p99 <= BUDGET_P99_MS, f'p99 {p99:.2f} ms over the {BUDGET_P99_MS} ms budget'


def test_latched_tick_within_budget():
    """The slowest tick: beside a wall, where the flank creep traces the body's arc every
    tick (safety_wrapper.SafetyWrapper.body_arc_free)."""
    bindings = []
    t = _tick_times(nr_testutil.FULL, n=600, scans=_beside_a_wall(), bindings=bindings)
    p99 = float(np.percentile(t, 99))
    print(f'beside a wall: median {np.median(t):.2f} ms, p99 {p99:.2f} ms, {bindings.count("flank_creep")} creep ticks')
    assert bindings.count("flank_creep") >= len(bindings) // 4      # the creep's path really ran (316 of 600)
    assert p99 <= BUDGET_P99_MS, f'p99 {p99:.2f} ms over the {BUDGET_P99_MS} ms budget'


def _forward_times(width, n=400):
    policy = PolicyNumpy(nr_testutil.random_weights(0, width=width))
    history = obs.History()
    rng = np.random.default_rng(0)
    times = []
    for i in range(n + 50):
        enc, valid = obs.pool_scan(rng.uniform(0.3, 12.0, 1081), 0.02, 30.0)
        history.push_tick(enc, valid, 1.0, (0.1, 0.0, 0.2))
        o = obs.build_observation(history, imu_present=True, mode=0, v_cap=1.0)
        t0 = time.perf_counter()
        out = policy.step(o)
        if i >= 50:
            times.append(time.perf_counter() - t0)
        history.push_command(out.steering, out.speed)
    return np.array(times) * 1e3


def test_half_width_is_faster():
    # The forward pass alone: the wrapper's swept-path clearance traces one arc or two
    # depending on what the network asks for, so whole ticks of two random networks
    # differ by their outputs (0.22 against 0.36 ms on the pit PC), not by their widths.
    full = np.median(_forward_times(nr_testutil.FULL))
    half = np.median(_forward_times(nr_testutil.HALF))
    assert half < full
