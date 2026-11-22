"""Run the stock MuSHR plant on the November 11 published scalar speed-loop task."""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from run_analysis import fit_speed_loop as fit


class StraightLine:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = fit.DT / 5
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data, nstep=200)
        self.rest = self.data.qpos.copy()
        self.wheels = [self.model.joint(f'buddy_wheel_{w}_throttle').dofadr[0]
                       for w in ('fl', 'fr', 'bl', 'br')]
        self.radius = self.model.body('buddy_wheel_bl').pos[2]
        self.motor = self.data.actuator('buddy_throttle_velocity').ctrl

    def horizon(self, commands, v0):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.rest
        self.data.qvel[0] = v0
        self.data.qvel[self.wheels] = v0 / self.radius
        wheel = [v0]
        for target in commands:
            self.motor[0] = target
            mujoco.mj_step(self.model, self.data, nstep=5)
            wheel.append(np.mean(self.data.qvel[self.wheels]) * self.radius)
        return np.array(wheel)


def speed_errors(plant, grids):
    errors = []
    horizon = round(fit.HORIZON / fit.DT)
    for _, cmd, v in grids:
        for start in range(0, len(v) - horizon, round(fit.RESTART / fit.DT)):
            end = start + horizon
            pred = plant.horizon(cmd[start:end], v[start])
            valid = cmd[start:end + 1] >= fit.MIN_CMD
            errors.extend((pred - v[start:end + 1])[valid])
    return np.array(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    args = parser.parse_args()
    plant = StraightLine(args.model)
    pooled = []
    for stamp in ('20261111-155542', '20261111-153721'):
        folder = TOOLS.parent / 'data' / 'rb1' / 'racerbot_runs' / stamp
        grids, _ = fit.load_grids([folder])
        err = speed_errors(plant, grids)
        pooled.extend(err)
        print(stamp, 'samples', len(err), 'wheel RMSE m/s', np.sqrt(np.mean(err**2)))
    print('pooled samples', len(pooled), 'wheel RMSE m/s', np.sqrt(np.mean(np.square(pooled))))
