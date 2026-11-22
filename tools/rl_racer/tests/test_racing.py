from pathlib import Path

import numpy as np
import pytest

from rl_racer import RacingEnv, SingleTrack, Track, VehicleParams


def circle(radius=5., count=1000, width=1.):
    theta = np.arange(count) * 2 * np.pi / count
    return Track(np.column_stack((radius * np.cos(theta), radius * np.sin(theta))), width)


def test_circle_projection_curvature():
    track = circle()
    theta = np.array([0.21, 1.03, 3.17, 6.2])
    poses = np.column_stack((4.8 * np.cos(theta), 4.8 * np.sin(theta), theta + np.pi / 2))
    s, n, heading = track.project(poses)
    # Circle closed forms; polygon discretization below 0.1 mm and 0.004 rad.
    np.testing.assert_allclose(s, 5 * theta, atol=1e-4, rtol=0)
    np.testing.assert_allclose(n, .2, atol=1e-4, rtol=0)
    np.testing.assert_allclose(heading, 0, atol=.004, rtol=0)
    np.testing.assert_allclose(track.curvature, 1 / 5, atol=2e-6, rtol=0)
    c, w = track.preview(s, 4)
    np.testing.assert_allclose(c, .2, atol=2e-6, rtol=0)
    np.testing.assert_allclose(w, 1., atol=1e-12, rtol=0)


def test_wrap_and_widths():
    track = circle()
    s = np.array([track.length - .01, .01])
    xy, yaw, _, _ = track.sample(s)
    actual, n, h = track.project(np.column_stack((xy, yaw + 2 * np.pi)))
    np.testing.assert_allclose(actual, s, atol=1e-12, rtol=0)
    np.testing.assert_allclose(n, 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(h, 0, atol=1e-12, rtol=0)
    c1, w1 = track.preview(s, 3)
    c2, w2 = track.preview(s + track.length, 3)
    np.testing.assert_allclose(c1, c2, atol=1e-12, rtol=0)
    np.testing.assert_allclose(w1, w2, atol=1e-12, rtol=0)
    p = np.column_stack((track.xy, np.full(len(track.xy), .3), np.full(len(track.xy), .7)))
    np.testing.assert_allclose(Track(p).sample(s)[3], [[.3, .7], [.3, .7]], atol=1e-12, rtol=0)


def test_checked_in_csv():
    root = Path(__file__).resolve().parents[3]
    paths = list((root / 'tools/f1tenth_sim/maps').glob('*centerline*.csv'))
    assert paths
    for path in paths:
        track = Track.from_csv(path, width=1.4)
        raw = np.loadtxt(path, delimiter=',', skiprows=1)
        # Independent polyline perimeter oracle, including closing segment.
        length = np.linalg.norm(np.diff(np.vstack((raw, raw[0])), axis=0), axis=1).sum()
        assert abs(track.length - length) < 1e-8  # metres
        np.testing.assert_allclose(track.widths, .7, atol=1e-12, rtol=0)


@pytest.mark.parametrize('points', [[], [[0, 0]], [[0, 0], [1, 0], [1, 0]],
                                     [[0, 0], [1, 0], [np.nan, 1]]])
def test_bad_track(points):
    with pytest.raises(ValueError, match='centerline'):
        Track(points)


def test_bad_csv(tmp_path):
    path = tmp_path / 'bad.csv'
    path.write_text('x,z\n0,1\n1,2\n2,3\n')
    with pytest.raises(ValueError, match='columns'):
        Track.from_csv(path)


def test_straight_acceleration():
    model = SingleTrack()
    z = np.zeros((2, 8))
    for _ in range(80):
        z = model.step(z, [0, 0], [2., 1.])
    assert z[0, 0] > z[1, 0] > 1.
    np.testing.assert_allclose(z[:, 3], [2., 1.], atol=.03, rtol=0)  # m/s
    np.testing.assert_allclose(z[:, [1, 2, 4, 5, 6]], 0, atol=1e-12, rtol=0)


def test_steady_circle():
    model = SingleTrack()
    z = np.zeros((2, 8))
    speed, radius = .35, 4.
    steer = np.arctan(model.params.wheelbase / radius) * np.array([1., -1.])
    z[:, 3] = z[:, 7] = speed
    for _ in range(120):
        z = model.step(z, steer, speed)
    # Bicycle-model low-speed oracle; 3% allows tyre slip and relaxation.
    np.testing.assert_allclose(z[:, 5], speed / radius * np.array([1., -1.]), atol=.003, rtol=0)
    np.testing.assert_allclose(z[:, 6], steer, atol=1e-10, rtol=0)


def test_zero_and_actuator_limits():
    model = SingleTrack()
    z = np.zeros((1, 8))
    np.testing.assert_array_equal(model.step(z, 0, 0), z)
    moved = model.step(z, 10, 100)
    assert 0 < moved[0, 6] <= model.params.steer_rate * .05
    assert 0 < moved[0, 7] <= model.params.wheel_rate * .05
    assert np.isfinite(moved).all()
    empty = model.step(np.empty((0, 8)), [], [])
    assert empty.shape == (0, 8)
    with pytest.raises(ValueError, match='references'):
        model.step(z, np.nan, 1)


def test_observation_and_progress():
    env = RacingEnv(circle(), 2, seed=4, preview_points=7, dr_sigma=0)
    xy, yaw, _, _ = env.track.sample(np.array([env.track.length - .002, 3.]))
    env.state[:, :2], env.state[:, 2] = xy, yaw
    env.state[:, 3] = env.state[:, 7] = env.wheel_cmd[:] = .2
    obs, reward, done, truncated, info = env.step(np.zeros((2, 2)))
    assert obs.shape == (2, 24)
    assert np.all(reward > 0)
    np.testing.assert_allclose(reward, .2 * .05, atol=2e-4, rtol=0)  # metres
    np.testing.assert_allclose(obs[:, 10:17], .2 / 2, atol=1e-6, rtol=0)
    np.testing.assert_allclose(obs[:, 17:], 1 / 2, atol=1e-12, rtol=0)
    assert not done.any() and not truncated.any()
    np.testing.assert_allclose(info['progress'], reward, atol=1e-12, rtol=0)
    env.state[:, 3] = -.2
    assert np.all(env.step(np.zeros((2, 2)))[1] < 0)


def test_offtrack_reset_and_truncation():
    env = RacingEnv(circle(), 2, seed=7, time_limit=1)
    env.state[0, :2] = [7., 0.]
    obs, reward, done, truncated, info = env.step(np.zeros((2, 2)))
    assert reward[0] == -1.
    np.testing.assert_array_equal(done, [True, False])
    np.testing.assert_array_equal(truncated, [False, True])
    _, n, _ = env.track.project(env.state[:, :3])
    np.testing.assert_allclose(n, 0, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(env.steps, [0, 0])
    assert abs(info['final_observation'][0, 4]) > .5 / env.scales[4]
    assert abs(obs[0, 4]) < 1e-12


def test_seeded_randomization():
    a = RacingEnv(circle(), 5, seed=23, dr_parameters=['friction', 'mass'])
    b = RacingEnv(circle(), 5, seed=23, dr_parameters=['friction', 'mass'])
    c = RacingEnv(circle(), 5, seed=24, dr_parameters=['friction', 'mass'])
    for key in ('friction', 'mass'):
        np.testing.assert_array_equal(a.parameters[key], b.parameters[key])
        assert not np.array_equal(a.parameters[key], c.parameters[key])
        assert np.ptp(a.parameters[key]) > 0
    initial = a.parameters['friction'].copy()
    actions = np.tile([.2, .3], (5, 1))
    np.testing.assert_array_equal(a.step(actions)[0], b.step(actions)[0])
    np.testing.assert_array_equal(a.parameters['friction'], initial)
    a.reset([0])
    assert a.parameters['friction'][0] != initial[0]
    np.testing.assert_array_equal(a.parameters['friction'][1:], initial[1:])


def test_action_limits_and_errors():
    env = RacingEnv(circle(), 2, seed=1)
    env.step([[2, 2], [-2, -2]])
    np.testing.assert_allclose(env.steer_cmd, [env.model.params.steer_max, env.model.params.steer_min], atol=1e-12, rtol=0)
    np.testing.assert_allclose(env.wheel_cmd, [.45, 0], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(env.accel_prev, [5, -5])
    for invalid in ([], [[0, 0]], [[np.inf, 0], [0, 0]]):
        with pytest.raises(ValueError, match='actions'):
            env.step(invalid)
    with pytest.raises(ValueError, match='unsupported'):
        RacingEnv(circle(), dr_parameters=['unknown'])
    with pytest.raises(ValueError, match='num_envs'):
        RacingEnv(circle(), 0)
    assert VehicleParams.from_yaml().mass > 0
