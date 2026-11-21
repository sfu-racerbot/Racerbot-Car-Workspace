"""Measure serial, sensor-state rollouts of the pinned stock MuSHR one_car scene."""

import argparse
import json
import statistics
import time

import mujoco
import numpy as np


def benchmark(model, steps=10_000):
    data = mujoco.MjData(model)
    steer = data.actuator('buddy_steering_pos').ctrl
    speed = data.actuator('buddy_throttle_velocity').ctrl
    velocity = data.sensor('buddy_velocimeter').data
    gyro = data.sensor('buddy_gyro').data

    def rollout(seed, count):
        mujoco.mj_resetData(model, data)
        rng = np.random.default_rng(seed)
        commands = rng.uniform([-.18, .5], [.18, 3.], (count, 2))
        trace = np.empty((count, 2))
        start = time.perf_counter()
        for i, (angle, target) in enumerate(commands):
            steer[0], speed[0] = angle, target
            mujoco.mj_step(model, data)
            # mj_step integrates qpos last; update sensors at the resulting state.
            mujoco.mj_forward(model, data)
            trace[i] = velocity[0], gyro[2]
        return trace, time.perf_counter() - start

    rollout(0, 500)
    first, elapsed = rollout(17, steps)
    times = [elapsed]
    for _ in range(2):
        again, elapsed = rollout(17, steps)
        if not np.array_equal(first, again) or not np.isfinite(again).all():
            raise RuntimeError('non-deterministic reset or non-finite sensor readings')
        times.append(elapsed)
    other, _ = rollout(18, steps)
    if np.array_equal(first, other):
        raise RuntimeError('sensor trace did not respond to different actions')
    start = time.perf_counter()
    for _ in range(1000):
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
    reset_us = (time.perf_counter() - start) * 1000
    rate = steps / statistics.median(times)
    return dict(mujoco=mujoco.__version__, steps=steps, seconds=times, steps_per_s=rate,
                timestep_s=model.opt.timestep, realtime_factor=rate * model.opt.timestep,
                reset_forward_us=reset_us, same_seed_identical=True, changed_seed_differs=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    args = parser.parse_args()
    print(json.dumps(benchmark(mujoco.MjModel.from_xml_path(args.model)), indent=2))
