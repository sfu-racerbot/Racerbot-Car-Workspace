"""Offline lap metrics for NumPy policies; optional read-only FidelityPlant adapter."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .env import RacingEnv
from .policy_numpy import PolicyNumpy
from .track import Track


class LapMetrics:
    def __init__(self, length):
        self.length = length
        self.elapsed = self.progress = self.lap_elapsed = self.lap_offtrack = self.offtrack = 0.
        self.laps = []
        self.resets = 0

    def add(self, progress, n, widths, dt, reset=False):
        excess = max(0., n - widths[1], -n - widths[0]) * dt
        self.elapsed += dt
        self.offtrack += excess
        self.lap_offtrack += excess
        self.lap_elapsed += dt
        self.progress += progress
        if reset:
            self.progress = self.lap_elapsed = self.lap_offtrack = 0.
            self.resets += 1
        elif self.progress >= self.length - 1e-10:
            fraction_after = (self.progress - self.length) / max(progress, 1e-12)
            leftover_time = np.clip(fraction_after, 0, 1) * dt
            leftover_error = np.clip(fraction_after, 0, 1) * excess
            self.laps.append({'lap': len(self.laps) + 1,
                              'lap_time_s': float(self.lap_elapsed - leftover_time),
                              'offtrack_integral_m_s': float(self.lap_offtrack - leftover_error)})
            self.progress = max(0., self.progress - self.length)
            self.lap_elapsed, self.lap_offtrack = leftover_time, leftover_error

    def report(self):
        times = [lap['lap_time_s'] for lap in self.laps]
        return {'elapsed_s': float(self.elapsed), 'laps_completed': len(self.laps),
                'lap_time_s': None if not times else float(np.mean(times)),
                'fastest_lap_s': None if not times else min(times),
                'offtrack_integral_m_s': float(self.offtrack), 'resets': self.resets,
                'per_lap': self.laps}


def make_fidelity_plant(centerline, dt, seed):
    """Construct without importing the existing controller/training harnesses."""
    root = Path(__file__).resolve().parents[3]
    for directory in (root / '.sim/f1tenth_gym', root / 'tools/f1tenth_sim'):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import gymnasium as gym
    import f1tenth_gym  # noqa: F401 - environment registration
    from f1tenth_gym.envs.dynamic_models import DynamicModel
    from f1tenth_gym.envs.env_config import EnvConfig, ObservationConfig, SimulationConfig
    from f1tenth_gym.envs.integrators import IntegratorType
    from f1tenth_gym.envs.observation import ObservationType
    from sim_fidelity.calibration import calibration_for_car
    from sim_fidelity.plant import FidelityPlant, PROFILES, env_components

    path = Path(centerline)
    if not path.name.endswith('_centerline.csv'):
        raise ValueError('FidelityPlant requires a *_centerline.csv with matching map files')
    map_path = path.with_name(path.name.removesuffix('_centerline.csv') + '.yaml')
    calibration = calibration_for_car('rb1')
    components = env_components(PROFILES['car'], calibration)
    config = EnvConfig(seed=seed, map_name=str(map_path), num_agents=1,
                       simulation_config=SimulationConfig(timestep=dt, integrator_timestep=.002,
                                                          integrator=IntegratorType.RK4,
                                                          dynamics_model=DynamicModel.ST, max_laps=None),
                       observation_config=ObservationConfig(type=ObservationType.DIRECT),
                       render_enabled=False, **components)
    gym_env = gym.make('f1tenth_gym:f1tenth-v0', config=config, render_mode=None)
    return FidelityPlant(gym_env, PROFILES['car'], 1, seed, dt, calibration)


def map_fidelity_state(observation):
    """std_state is x,y,steer,speed,yaw,yaw_rate,slip_angle.

    Steering is available in simulated state, not a real servo measurement.
    Wheel speed is unavailable: estimate it as body longitudinal ground speed
    (zero longitudinal-slip assumption). Body velocities use speed/slip angle.
    """
    x, y, steer, speed, yaw, rate, slip = np.asarray(observation['agent_0']['std_state'])
    vx, vy = speed * np.cos(slip), speed * np.sin(slip)
    return np.array([[x, y, yaw, vx, vy, rate, steer, vx]])


def evaluate(policy_path, *, track_path=None, max_steps=2000, laps=3, seed=42, fidelity=False):
    if not isinstance(max_steps, int) or max_steps < 1 or not isinstance(laps, int) or laps < 1:
        raise ValueError('max_steps and laps must be positive integers')
    policy = PolicyNumpy(policy_path)
    metadata = policy.metadata
    path = Path(track_path or metadata['config']['track'])
    track = Track.from_csv(path, metadata['config']['track_width'])
    env = RacingEnv(track, seed=seed, config=metadata['vehicle'], dt=metadata['dt'],
                    preview_points=metadata['preview_points'], dr_parameters=[], dr_sigma=0.,
                    time_limit=max_steps + 1)
    if not np.array_equal(env.scales, policy.scales):
        raise ValueError('policy and evaluation observation normalization differ')
    xy, yaw, _, _ = track.sample(np.array([0.]))
    pose = np.column_stack((xy, yaw))
    env.state[:, :2], env.state[:, 2] = xy, yaw
    policy.reset(env.settings['reset_speed'])
    metrics = LapMetrics(track.length)
    plant = None
    mapping = 'native single-track state; no measurement substitution'
    try:
        if fidelity:
            plant = make_fidelity_plant(path, env.dt, seed)
            observation, _ = plant.reset(pose, speeds=[env.settings['reset_speed']])
            env.state = map_fidelity_state(observation)
            mapping = ('FidelityPlant car profile: estimated pose/speed, simulated steering and yaw/slip; '
                       'wheel speed estimated as vx (no longitudinal-slip measurement); '
                       'previous commands tracked locally; no OptiTrack')
        for _ in range(max_steps):
            before = track.project(env.state[:, :3])[0][0]
            actions = policy.act(env.observe())
            if plant is None:
                _, _, done, trunc, info = env.step(actions)
                progress = float(info['progress'][0])
                s = (before + progress) % track.length
                n = float(info['final_observation'][0, 4] * env.scales[4])
                reset = bool(done[0] or trunc[0])
            else:
                commands = policy.commands(actions)
                observation, _, _, _, _ = plant.step(commands)
                env.state = map_fidelity_state(observation)
                env.steer_cmd = policy.steer_cmd.copy()
                env.wheel_cmd = policy.wheel_cmd.copy()
                env.accel_prev = policy.accel_prev.copy()
                s_arr, n_arr, _ = track.project(env.state[:, :3])
                s, n = float(s_arr[0]), float(n_arr[0])
                progress = (s - before + track.length / 2) % track.length - track.length / 2
                widths = track.sample(s)[3]
                reset = bool(n < -widths[0] or n > widths[1] or observation['agent_0']['collision'])
            metrics.add(progress, n, track.sample(s)[3], env.dt, reset)
            if plant is not None and reset:
                observation, _ = plant.reset(pose, speeds=[env.settings['reset_speed']])
                env.state = map_fidelity_state(observation)
                policy.reset(env.settings['reset_speed'])
                env.steer_cmd = policy.steer_cmd.copy()
                env.wheel_cmd = policy.wheel_cmd.copy()
                env.accel_prev = policy.accel_prev.copy()
            if len(metrics.laps) >= laps:
                break
    finally:
        if plant is not None:
            plant.close()
    return dict(metrics.report(), backend='FidelityPlant' if fidelity else 'RacingEnv',
                config_hash=metadata['config_hash'], seed=seed, state_mapping=mapping,
                track=str(path), policy_smoke=metadata['smoke'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('policy', type=Path)
    parser.add_argument('--track', type=Path)
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--laps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fidelity', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(args.policy, track_path=args.track, max_steps=args.max_steps,
                      laps=args.laps, seed=args.seed, fidelity=args.fidelity)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
