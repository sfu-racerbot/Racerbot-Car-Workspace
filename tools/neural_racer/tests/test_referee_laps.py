"""The referee counts laps itself, on each car's progress along the line, never on the
gym's info['lap_counts'] (2026-11-11). ROS- and gym-free: a fake plant moves the cars
round a circle and its gym lap counter never counts, as the real one failed to with two
cars on car 1's course (its Frenet s jumped to the neighbouring straight and ran backwards).

Runner: python3 -m pytest tools/neural_racer/tests/test_referee_laps.py -v
Oracle: the circle's arc length -- a car at speed v finishes `laps` laps plus its grid
offset behind the start line in (laps * L + offset) / v seconds.
"""
import math
from types import SimpleNamespace

import conftest  # noqa: F401
import numpy as np
import pytest

from neural_racer_train import referee

R, N, DT = 5.0, 400, 0.025          # circle radius m, line points, control period s
L = 2 * math.pi * R


class FakePlant:
    def __init__(self, n, speeds):
        a = np.linspace(0.0, 2 * math.pi, N, endpoint=False)
        line = SimpleNamespace(xs=R * np.cos(a), ys=R * np.sin(a), yaws=a + math.pi / 2)
        self.env = SimpleNamespace(unwrapped=SimpleNamespace(track=SimpleNamespace(raceline=line)))
        self.n, self.speeds, self.t = n, speeds, 0.0
        self.theta = np.zeros(n)

    def reset(self, poses):
        self.theta = np.arctan2(poses[:, 1], poses[:, 0])
        return self._obs(), None

    def _obs(self):
        return {f'agent_{i}': {'collision': False} for i in range(self.n)}

    def truth(self, i):
        th = self.theta[i]
        return np.array([R * math.cos(th), R * math.sin(th), th + math.pi / 2, self.speeds[i], 0.0, 0.0, 0.0])

    def step(self, cmds):
        self.t += DT
        for i, (_steer, speed) in enumerate(cmds):
            self.theta[i] += speed * DT / R
        # the gym's lap counter, as it failed: never counts
        return self._obs(), 0.0, False, False, {'sim_time': self.t, 'lap_counts': np.zeros(self.n)}

    def close(self):
        pass


class Cruise:
    def __init__(self, speed):
        self.speed = speed

    def command(self, agent_obs, truth, step):
        return 0.0, self.speed


@pytest.fixture
def fake_rv(monkeypatch):
    def nearest(xy, p, closed=True, prev_index=None, search_window=80):
        d = np.hypot(xy[:, 0] - p[0], xy[:, 1] - p[1])
        k = int(np.argmin(d))
        return k, float(d[k])

    def make(speeds):
        rv = SimpleNamespace(
            make_plant=lambda track, n, seed: FakePlant(n, speeds),
            initial_pose=lambda line, i: np.array([line.xs[i], line.ys[i], line.yaws[i]]),
            racing_math=SimpleNamespace(find_nearest_index=nearest))
        monkeypatch.setattr(referee, '_rv', lambda: rv)
        monkeypatch.setattr(referee, 'track_map_name', lambda t: t)
    return make


def test_laps_counted_on_progress_when_the_gym_counter_fails(fake_rv):
    fake_rv([2.0, 2.0])
    r = referee.race('circle', {'a': lambda p: Cruise(2.0), 'b': lambda p: Cruise(2.0)},
                     seed=0, laps=3, timeout_s=60.0)
    spacing = L / N
    back = round(referee.GRID_STAGGER / spacing)
    assert r.finish_time[0] == pytest.approx(3 * L / 2.0, abs=0.1)
    assert r.finish_time[1] == pytest.approx((3 * L + back * spacing) / 2.0, abs=0.1)
    assert r.laps == [3.0, 3.0] and r.order == ['a', 'b']


def test_a_car_short_of_the_distance_does_not_finish(fake_rv):
    fake_rv([2.0, 0.5])
    r = referee.race('circle', {'fast': lambda p: Cruise(2.0), 'slow': lambda p: Cruise(0.5)},
                     seed=0, laps=2, timeout_s=40.0)
    assert r.finish_time[0] == pytest.approx(2 * L / 2.0, abs=0.1)
    assert r.finish_time[1] is None and r.laps[1] == 0.0      # 20 m of a 63 m race
    assert referee.finish_outcome(r.names, r.finish_time, 'fast', 'slow') == 'win'
