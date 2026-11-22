"""MuSHR open-loop worker for frozen prequalified blocks.

Read CSV car,run,b_ns,step,speed_cmd,steer_cmd,initial_speed (only on step 0).
Emit all 480 post-step body speeds/yaw rates per complete input block. No
real observations after initialization are inputs. The common scorer owns
eligibility masks, missing-coverage accounting and per-run aggregation.
"""

import argparse
import csv
import os
from itertools import groupby
from pathlib import Path
from tempfile import NamedTemporaryFile

import mujoco
import numpy as np

STEPS = 480
DT = 0.025


class FlatMuSHR:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        # Five equal solver steps hit every 25 ms boundary without rounding.
        self.model.opt.timestep = DT / 5
        self.data = mujoco.MjData(self.model)
        self.body_id = self.model.body('buddy').id
        self.wheels = [self.model.joint(f'buddy_wheel_{w}_throttle').dofadr[0]
                       for w in ('fl', 'fr', 'bl', 'br')]
        # Nominal axle height above the model's flat floor, before compression.
        self.nominal_radius = self.model.body('buddy_wheel_bl').pos[2]
        self.steer = self.data.actuator('buddy_steering_pos').ctrl
        self.speed = self.data.actuator('buddy_throttle_velocity').ctrl
        self.twist = np.empty(6)

    def observe(self):
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,
                                 self.body_id, self.twist, 1)
        return self.twist[3], self.twist[2]  # Body-origin linear X and angular Z.

    def reset(self, v0):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:2] = 0.
        self.data.qvel[0] = v0
        self.data.qvel[self.wheels] = v0 / self.nominal_radius
        mujoco.mj_forward(self.model, self.data)

    def step(self, speed, steer):
        self.speed[0], self.steer[0] = speed, steer
        mujoco.mj_step(self.model, self.data, nstep=5)
        mujoco.mj_forward(self.model, self.data)
        return self.observe()


def rollout(plant, rows):
    if [int(r['step']) for r in rows] != list(range(STEPS)):
        raise ValueError('missing, duplicate or unordered step in 480-tick block')
    v0 = float(rows[0]['initial_speed'])
    commands = np.array([(float(r['speed_cmd']), float(r['steer_cmd'])) for r in rows])
    if not np.isfinite(v0) or not np.isfinite(commands).all():
        raise ValueError('non-finite input in prequalified block')
    plant.reset(v0)
    predicted = np.array([plant.step(speed, steer) for speed, steer in commands])
    if (not np.isfinite(predicted).all() or np.any(plant.data.warning.number)
            or not np.isclose(plant.data.time, STEPS * DT, rtol=0., atol=1e-10)):
        raise RuntimeError('non-finite rollout, solver warning or wrong final physics time')
    return predicted


def run(plant, reader, writer):
    writer.writerow(('car', 'run', 'b_ns', 'step', 'body_speed_mps', 'body_yaw_radps'))
    seen = set()
    for key, group in groupby(reader, lambda r: (r['car'], r['run'], int(r['b_ns']))):
        if key in seen:
            raise ValueError(f'duplicate block: {key}')
        seen.add(key)
        try:
            output = rollout(plant, list(group))
        except (KeyError, RuntimeError, ValueError) as exc:
            raise ValueError(f'incomplete block {key}: {exc}') from exc
        writer.writerows((*key, step, *pair) for step, pair in enumerate(output))
    return len(seen)


def write_predictions(plant, source, path):
    # Publish no partial CSV if a later block fails. Never overwrite an output.
    with NamedTemporaryFile(mode='w', newline='', dir=path.parent) as tmp:
        count = run(plant, csv.DictReader(source), csv.writer(tmp))
        tmp.flush()
        os.link(tmp.name, path)
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    plant = FlatMuSHR(args.model)
    with args.inputs.open(newline='') as source:
        count = write_predictions(plant, source, args.output)
    print(f'{count} blocks, {count * STEPS} post-step observations')
