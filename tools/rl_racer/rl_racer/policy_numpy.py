"""Torch-free deterministic actor and stateful wheel-command integrator."""
import json

import numpy as np


class PolicyNumpy:
    def __init__(self, path, num_envs=1):
        with np.load(path, allow_pickle=False) as archive:
            self.metadata = json.loads(str(archive['metadata']))
            if self.metadata['format_version'] != 1:
                raise ValueError('unsupported policy format')
            self.scales = archive['observation_scale'].copy()
            self.action_scale = archive['action_scale'].copy()
            self.limits = archive['steering_limits'].copy()
            self.v_max = float(archive['v_max'])
            keys = sorted((key for key in archive if key.startswith('actor.') and key.endswith('.weight')),
                          key=lambda key: int(key.split('.')[1]))
            self.layers = [(archive[key].copy(), archive[key.replace('.weight', '.bias')].copy()) for key in keys]
        if not self.layers or len(self.scales) != self.layers[0][0].shape[1]:
            raise ValueError('invalid actor dimensions')
        self.steer_cmd = np.zeros(num_envs)
        self.wheel_cmd = np.zeros(num_envs)
        self.accel_prev = np.zeros(num_envs)

    def act(self, observations, normalized=True):
        x = np.asarray(observations, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(self.scales) or not np.isfinite(x).all():
            raise ValueError('observations must be finite (N,obs_dim)')
        if not normalized:
            x = (x / self.scales).astype(np.float32)
        for i, (weight, bias) in enumerate(self.layers):
            x = x @ weight.T + bias
            if i != len(self.layers) - 1:
                x = np.where(x >= 0, x, self.metadata['negative_slope'] * x)
        return np.tanh(x)

    def reset(self, speed=0., indices=None):
        ids = np.arange(len(self.wheel_cmd)) if indices is None else indices
        self.wheel_cmd[ids] = np.clip(speed, 0, self.v_max)
        self.steer_cmd[ids] = 0
        self.accel_prev[ids] = 0

    def commands(self, actions):
        a = np.asarray(actions, dtype=float)
        if a.shape != (len(self.wheel_cmd), 2) or not np.isfinite(a).all():
            raise ValueError('actions must be finite (N,2)')
        a = np.clip(a, -1, 1)
        self.steer_cmd = np.clip(a[:, 0] * self.action_scale[0], *self.limits)
        self.accel_prev = a[:, 1] * self.action_scale[1]
        self.wheel_cmd = np.clip(self.wheel_cmd + self.metadata['dt'] * self.accel_prev, 0, self.v_max)
        return np.column_stack((self.steer_cmd, self.wheel_cmd))
