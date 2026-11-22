import numpy as np
import pytest

from rl_racer import RacingEnv, Track


def test_side_boundaries_and_command_ceiling():
    # Segment x=1..2 is straight: its endpoint tangents are horizontal.
    xy = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [3., 3.], [0., 3.]])
    track = Track(np.column_stack((xy, np.full(6, .3), np.full(6, .7))))
    env = RacingEnv(track, 4, seed=1, reset_speed=0.)
    env.state[:, :2] = np.column_stack((np.full(4, 1.5), [-.3, -.300001, .7, .700001]))
    env.state[:, 2:] = 0
    _, reward, done, _, _ = env.step(np.zeros((4, 2)))
    np.testing.assert_array_equal(done, [False, True, False, True])
    np.testing.assert_array_equal(reward, [0., -1., 0., -1.])
    env.wheel_cmd[:] = env.model.params.v_max
    env.step(np.tile([0., 1.], (4, 1)))
    np.testing.assert_array_equal(env.wheel_cmd, np.full(4, env.model.params.v_max))


def test_preview_spacing_and_empty_projection():
    track = Track([[0., 0.], [3., 0.], [4., 2.], [0., 3.]])
    s = np.array([.1, track.length - .2])
    actual, widths = track.preview(s, 5)
    query = (s[:, None] + .3 * np.arange(1, 6)) % track.length
    expected = np.interp(query, np.r_[track.s, track.length],
                         np.r_[track.curvature, track.curvature[0]])
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(widths, 1., atol=1e-12, rtol=0)
    assert all(v.shape == (0,) for v in track.project(np.empty((0, 3))))
    with pytest.raises(ValueError, match='poses'):
        track.project([[np.nan, 0, 0]])
    with pytest.raises(ValueError, match='arc lengths'):
        track.sample(np.inf)
    with pytest.raises(ValueError, match='preview'):
        track.preview([0.], 0)
