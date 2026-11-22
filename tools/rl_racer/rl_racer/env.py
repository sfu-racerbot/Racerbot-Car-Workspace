"""Batched paper-policy environment. Step auto-resets terminated/truncated rows.

Returned observations are post-reset; info['final_observation'] contains the
pre-reset observation for every row (use done/truncated masks when bootstrapping).
Width previews are full widths. Boundary checks use the local side's half width.
"""
import numpy as np
import yaml

from .dynamics import CONFIG, SingleTrack, VehicleParams


class RacingEnv:
    RANDOMIZABLE = frozenset({'friction', 'mass', 'inertia', 'wheelbase',
                             'lateral_b_front', 'lateral_b_rear', 'longitudinal_b',
                             'steer_tau', 'wheel_tau'})

    def __init__(self, track, num_envs=1, seed=None, config=CONFIG, **overrides):
        if isinstance(config, dict):
            document = config
        else:
            with open(config, encoding='utf-8') as stream:
                document = yaml.safe_load(stream)
        settings = dict(document['environment'])
        if set(overrides) - set(settings):
            raise ValueError('unknown environment setting')
        settings.update(overrides)
        self.settings = settings
        if not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError('num_envs must be positive integer')
        self.num_envs, self.track = num_envs, track
        self.model = SingleTrack(VehicleParams(**document['vehicle']))
        self.rng = np.random.default_rng(seed)
        self.k = settings['preview_points']
        self.dt, self.limit = settings['dt'], settings['time_limit']
        if not isinstance(self.k, int) or self.k < 1 or not isinstance(self.limit, int) or self.limit < 1:
            raise ValueError('preview_points and time_limit must be positive integers')
        scales = np.asarray(settings['observation_scale'], dtype=float)
        scalar_scales = np.array([self.dt, settings['curvature_scale'], settings['width_scale']])
        if scales.shape != (10,) or not np.isfinite(scales).all() or np.any(scales <= 0) or not np.isfinite(scalar_scales).all() or np.any(scalar_scales <= 0):
            raise ValueError('normalization and dt must be finite and positive')
        if not np.isfinite(settings['dr_sigma']) or settings['dr_sigma'] < 0:
            raise ValueError('dr_sigma must be finite and nonnegative')
        if not np.isfinite(settings['reset_speed']) or not 0 <= settings['reset_speed'] <= self.model.params.v_max:
            raise ValueError('reset_speed outside vehicle speed envelope')
        if not set(settings['dr_parameters']) <= self.RANDOMIZABLE:
            raise ValueError('unsupported domain randomization parameter')
        self.scales = np.r_[scales, np.full(self.k, settings['curvature_scale']),
                            np.full(self.k, settings['width_scale'])]
        self.state = np.zeros((num_envs, 8))
        self.steer_cmd = np.zeros(num_envs)
        self.wheel_cmd = np.zeros(num_envs)
        self.accel_prev = np.zeros(num_envs)
        self.steps = np.zeros(num_envs, dtype=int)
        self.parameters = {key: np.full(num_envs, getattr(self.model.params, key))
                           for key in settings['dr_parameters']}
        self.reset()

    def reset(self, indices=None):
        ids = np.arange(self.num_envs) if indices is None else np.asarray(indices, dtype=int)
        s = self.rng.uniform(0, self.track.length, size=len(ids))
        xy, yaw, _, _ = self.track.sample(s)
        self.state[ids] = 0
        self.state[ids, :2] = xy
        self.state[ids, 2] = yaw
        self.state[ids, 3] = self.settings['reset_speed']
        self.state[ids, 7] = self.settings['reset_speed']
        self.steer_cmd[ids] = 0
        self.wheel_cmd[ids] = self.settings['reset_speed']
        self.accel_prev[ids] = 0
        self.steps[ids] = 0
        for key, values in self.parameters.items():
            factor = self.rng.normal(1., self.settings['dr_sigma'], len(ids))
            # Rejection preserves the Gaussian conditioned on physical positivity.
            while np.any(factor <= 0):
                bad = factor <= 0
                factor[bad] = self.rng.normal(1., self.settings['dr_sigma'], bad.sum())
            values[ids] = getattr(self.model.params, key) * factor
        return self.observe()

    def observe(self):
        z = self.state
        s, n, heading = self.track.project(z[:, :3])
        c, w = self.track.preview(s, self.k)
        raw = np.column_stack((z[:, 3], z[:, 4], z[:, 5], heading, n, z[:, 6],
                               self.steer_cmd, z[:, 7], self.wheel_cmd, self.accel_prev, c, w))
        return raw / self.scales

    def step(self, actions):
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.num_envs, 2) or not np.isfinite(actions).all():
            raise ValueError('actions must be finite (N,2)')
        a = np.clip(actions, -1, 1)
        before, _, _ = self.track.project(self.state[:, :3])
        p = self.model.params
        self.steer_cmd = np.clip(0.5 * a[:, 0], p.steer_min, p.steer_max)
        self.accel_prev = 5. * a[:, 1]
        self.wheel_cmd = np.clip(self.wheel_cmd + self.dt * self.accel_prev, 0, p.v_max)
        friction = self.parameters.get('friction', p.friction)
        self.state = self.model.step(self.state, self.steer_cmd, self.wheel_cmd,
                                     self.dt, friction, self.parameters)
        s, n, _ = self.track.project(self.state[:, :3])
        widths = self.track.sample(s)[3]
        done = (n < -widths[:, 0]) | (n > widths[:, 1])
        progress = (s - before + self.track.length / 2) % self.track.length - self.track.length / 2
        reward = np.where(done, -1., progress)
        self.steps += 1
        truncated = (self.steps >= self.limit) & ~done
        final = self.observe()
        ids = np.flatnonzero(done | truncated)
        if len(ids):
            self.reset(ids)
        return self.observe(), reward, done, truncated, {'final_observation': final, 'progress': progress}
