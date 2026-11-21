"""Closed-form geometry and parameter contracts for opt-in adaptive pursuit."""
from dataclasses import fields, replace
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_pursuit import racing_math, tracking_laws as laws


LIMITS = laws.VehicleLimits(0.36, 0.42, 1.0, 0.025)


def context(xy, curvature, closed=False):
    xy = np.asarray(xy, dtype=float)
    segments = racing_math.compute_segment_lengths(xy, closed)
    return laws.PathContext(xy, segments, racing_math.compute_cumulative_arc_length(segments),
                            np.ones(len(xy)), np.full(len(xy), curvature), closed)


def straight():
    return context(np.column_stack((np.linspace(0, 20, 101), np.zeros(101))), 0)


def test_straight_zero_steering():
    law = laws.make_law('adaptive_pure_pursuit', LIMITS, {})
    assert law.steer(laws.TrackingInput(0, 0, 0, 6, 0, 0), straight()).steering == 0


@pytest.mark.parametrize('sign', [-1, 1])
def test_circle_steady_state(sign):
    radius = 4
    theta = sign * np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    path = context(np.column_stack((radius * np.cos(theta), radius * np.sin(theta))),
                   sign / radius, True)
    expected = math.atan(sign * LIMITS.wheelbase / radius)
    inp = laws.TrackingInput(radius, 0, sign * math.pi / 2, 5, 0, expected)
    law = laws.make_law('adaptive_pure_pursuit', LIMITS, {})
    assert law.steer(inp, path).steering == pytest.approx(expected, abs=2e-5)


def test_curvature_shortens_lookahead_and_respects_bounds():
    law = laws.make_law('adaptive_pure_pursuit', LIMITS, {})
    path = straight()
    curved = replace(path, curvature=np.ones(len(path.xy)))
    assert law.params.min_lookahead <= law.lookahead(6, 0, curved) < law.lookahead(6, 0, path)
    assert law.lookahead(100, 0, path) == law.params.max_lookahead
    assert law.lookahead(0, 0, curved) == law.params.min_lookahead


def test_zero_delay_is_no_prediction():
    inp = laws.TrackingInput(2, .1, .02, 4, 10, .05)
    law = laws.make_law('adaptive_pure_pursuit', LIMITS,
                        {'adaptive_pure_pursuit.delay_compensation': 0})
    distance = law.params.min_lookahead + inp.speed * law.params.lookahead_speed_gain
    xb = math.cos(inp.yaw) * distance - math.sin(inp.yaw) * inp.y
    yb = -math.sin(inp.yaw) * distance - math.cos(inp.yaw) * inp.y
    expected = math.atan(LIMITS.wheelbase * 2 * yb / (xb * xb + yb * yb))
    assert law.steer(inp, straight()).steering == pytest.approx(expected, abs=1e-12)


def test_nonzero_delay_uses_predicted_geometry():
    path = straight()
    inp = laws.TrackingInput(2, .1, .02, 4, 10, .05)
    delay = laws.AdaptivePurePursuitParams().delay_compensation
    predicted = laws.predict_input(inp, path, delay, LIMITS.wheelbase)
    delayed = laws.make_law('adaptive_pure_pursuit', LIMITS, {})
    instant = laws.make_law('adaptive_pure_pursuit', LIMITS,
                            {'adaptive_pure_pursuit.delay_compensation': 0})
    assert delayed.steer(inp, path) == instant.steer(predicted, path)
    assert delayed.steer(inp, path).steering != instant.steer(inp, path).steering


def test_interpolated_target_on_sparse_open_path():
    path = context([[0, 0], [10, 0]], 0)
    law = laws.make_law('adaptive_pure_pursuit', LIMITS, {
        'adaptive_pure_pursuit.delay_compensation': 0,
        'adaptive_pure_pursuit.min_lookahead': 1,
        'adaptive_pure_pursuit.max_lookahead': 1})
    result = law.steer(laws.TrackingInput(0, .1, 0, 0, 0, 0), path)
    assert result.steering == pytest.approx(math.atan(.36 * -.2 / 1.01))


@pytest.mark.parametrize('key,value', [
    ('min_lookahead', 0), ('min_lookahead', -1), ('max_lookahead', .1),
    ('lookahead_speed_gain', -1), ('curvature_gain', -1),
    ('delay_compensation', -.1), ('delay_compensation', .6),
] + [(f.name, value) for f in fields(laws.AdaptivePurePursuitParams)
     for value in (float('nan'), float('inf'))])
def test_invalid_params(key, value):
    with pytest.raises(ValueError, match='adaptive_pure_pursuit.' + key):
        laws.make_law('adaptive_pure_pursuit', LIMITS, {'adaptive_pure_pursuit.' + key: value})


@pytest.mark.parametrize('xy', [[], [[0, 0]], [[0, 0], [0, 0]]])
def test_empty_or_degenerate_path_fails(xy):
    points = np.asarray(xy, dtype=float).reshape(-1, 2)
    path = laws.PathContext(points, np.zeros(len(points)), np.zeros(len(points)),
                            np.zeros(len(points)), np.zeros(len(points)), False)
    with pytest.raises(laws.TrackingLawError, match='nondegenerate'):
        laws.make_law('adaptive_pure_pursuit', LIMITS, {}).steer(
            laws.TrackingInput(0, 0, 0, 1, 0, 0), path)


def test_yaml_and_node_defaults_match():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / 'config/pure_pursuit.yaml').read_text())['pure_pursuit_node']['ros__parameters']
    node = (root / 'pure_pursuit/pure_pursuit_node.py').read_text()
    defaults = laws.AdaptivePurePursuitParams()
    for field in fields(defaults):
        key = 'adaptive_pure_pursuit.' + field.name
        value = getattr(defaults, field.name)
        assert config[key] == value
        assert f"self.declare_parameter('{key}', {value})" in node


@pytest.mark.parametrize('field', ['x', 'y', 'yaw', 'speed', 'previous_steering'])
@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_input_fails(field, value):
    inp = replace(laws.TrackingInput(0, 0, 0, 1, 0, 0), **{field: value})
    with pytest.raises(laws.TrackingLawError):
        laws.make_law('adaptive_pure_pursuit', LIMITS, {}).steer(inp, straight())


def test_end_of_open_path_clamps_target():
    path = context([[0, 0], [1, 0]], 0)
    inp = laws.TrackingInput(.9, .01, 0, 0, 1, 0)
    law = laws.make_law('adaptive_pure_pursuit', LIMITS, {})
    expected = math.atan(.36 * -.02 / (.1 ** 2 + .01 ** 2))
    assert law.steer(inp, path).steering == pytest.approx(np.clip(expected, -.42, .42), abs=1e-12)


def test_steering_clip_and_reset():
    law = laws.make_law('adaptive_pure_pursuit', LIMITS, {})
    inp = laws.TrackingInput(0, 1, 0, 0, 0, 0)
    result = law.steer(inp, straight())
    assert result.unclipped < -LIMITS.max_steering_angle
    assert result.steering == -LIMITS.max_steering_angle
    law.reset()
    assert law.steer(inp, straight()) == result


def test_curvature_preview_wraps_closing_segment():
    path = context([[0, 0], [10, 0], [10, 10], [0, 10]], 0, True)
    law = laws.make_law('adaptive_pure_pursuit', LIMITS, {})
    curve = replace(path, curvature=np.array([2., 0., 0., 0.]))
    assert law.lookahead(5, 39.5, curve) < law.lookahead(5, 39.5, path)
