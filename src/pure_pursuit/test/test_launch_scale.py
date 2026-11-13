"""LaunchScale: the pure arithmetic of the launch-lap speed scale (no rclpy). The node wiring is tested in
test_launch_scale_node.py. Run by `python3 -m pytest src/pure_pursuit/test/` and the package runner."""
import math

import pytest

from pure_pursuit.launch_scale import LaunchScale

LINE = 41.5     # m, a stand-in line length


def armed(scale=0.85, distance=12.0):
    ls = LaunchScale(scale, distance)
    ls.on_deadman(True, 0.0, LINE)
    return ls


@pytest.mark.parametrize('scale', [0.5, 0.0, -1.0, 1.0001, float('nan'), float('inf')])
def test_a_scale_outside_a_half_to_one_is_refused(scale):
    with pytest.raises(ValueError, match='launch_speed_scale'):
        LaunchScale(scale, 0.0)


@pytest.mark.parametrize('scale', [0.5001, 0.85, 1.0])
def test_a_scale_in_the_open_half_to_one_is_accepted(scale):
    assert LaunchScale(scale, 0.0).factor == 1.0            # nothing armed yet


@pytest.mark.parametrize('distance', [-0.001, float('nan'), float('inf'), float('-inf')])
def test_a_bad_distance_is_refused(distance):
    with pytest.raises(ValueError, match='launch_scale_distance'):
        LaunchScale(0.9, distance)


def test_a_press_from_standstill_arms_the_distance_and_the_scale():
    # oracle: spec (2026-11-13 brief): arm on the deadman rising edge at measured speed < 0.5 m/s.
    ls = armed(0.85, 12.0)
    assert ls.remaining == 12.0 and ls.factor == 0.85


def test_distance_zero_means_one_line_length():
    ls = LaunchScale(0.85, 0.0)
    ls.on_deadman(True, 0.0, LINE)
    assert ls.remaining == LINE


def test_a_press_at_the_arming_speed_boundary():
    # oracle: spec, armed only below 0.5 m/s: 0.5 itself does not arm, the next step below does.
    at = LaunchScale(0.85, 12.0)
    at.on_deadman(True, 0.5, LINE)
    assert at.remaining == 0.0 and at.factor == 1.0
    below = LaunchScale(0.85, 12.0)
    below.on_deadman(True, 0.4999, LINE)
    assert below.remaining == 12.0


def test_a_press_while_rolling_does_not_arm():
    ls = LaunchScale(0.85, 12.0)
    ls.on_deadman(True, 3.0, LINE)
    assert ls.remaining == 0.0 and ls.factor == 1.0


def test_reverse_speed_counts_by_magnitude_when_arming():
    ls = LaunchScale(0.85, 12.0)
    ls.on_deadman(True, -3.0, LINE)
    assert ls.remaining == 0.0


def test_stale_odometry_counts_as_standing_when_arming():
    # oracle: spec "stale odometry = 0".
    ls = LaunchScale(0.85, 12.0)
    ls.on_deadman(True, None, LINE)
    assert ls.remaining == 12.0


def test_holding_does_not_rearm_and_a_rolling_press_does_not_clear_the_remainder():
    ls = armed(0.85, 12.0)
    ls.step(2.0, 1.0)                                    # 2 m driven
    assert ls.remaining == pytest.approx(10.0, abs=1e-12)  # m
    ls.on_deadman(True, 0.0, LINE)                       # still held: no edge, even at standstill
    assert ls.remaining == pytest.approx(10.0, abs=1e-12)
    ls.on_deadman(False, 2.0, LINE)
    ls.on_deadman(True, 2.0, LINE)                       # a fresh press while rolling: neither arms nor clears
    assert ls.remaining == pytest.approx(10.0, abs=1e-12)
    assert ls.factor == 0.85


def test_a_fresh_press_from_standstill_rearms_the_full_distance():
    ls = armed(0.85, 12.0)
    ls.step(2.0, 1.0)
    ls.on_deadman(False, 0.0, LINE)
    ls.on_deadman(True, 0.0, LINE)
    assert ls.remaining == 12.0


def test_the_count_down_is_speed_times_dt_by_magnitude_and_stops_at_zero():
    ls = armed(0.9, 1.0)
    ls.step(2.0, 0.1)
    assert ls.remaining == pytest.approx(0.8, abs=1e-12)     # m
    ls.step(-2.0, 0.1)
    assert ls.remaining == pytest.approx(0.6, abs=1e-12)
    ls.step(100.0, 1.0)                                      # overshoots: never below 0
    assert ls.remaining == 0.0


def test_the_factor_returns_to_one_when_the_distance_is_used_up():
    ls = armed(0.8, 0.5)
    assert ls.factor == 0.8
    ls.step(1.0, 0.5)
    assert ls.remaining == 0.0 and ls.factor == 1.0
    ls.step(1.0, 0.5)
    assert ls.factor == 1.0


@pytest.mark.parametrize('speed,dt', [(None, 0.1), (float('nan'), 0.1), (2.0, float('nan')), (2.0, -0.1), (2.0, 0.0),
                                      (float('inf'), 0.1)])
def test_a_stale_or_bad_tick_does_not_count(speed, dt):
    # oracle: spec, stale odometry does not count down (a standing car's remainder must not run out).
    ls = armed(0.85, 12.0)
    ls.step(speed, dt)
    assert ls.remaining == 12.0


def test_a_scale_of_one_never_arms():
    ls = LaunchScale(1.0, 12.0)
    ls.on_deadman(True, 0.0, LINE)
    assert ls.remaining == 0.0 and ls.factor == 1.0


def test_a_bad_line_length_arms_nothing_when_the_distance_is_the_line():
    for bad in (0.0, -3.0, float('nan'), float('inf')):
        ls = LaunchScale(0.85, 0.0)
        ls.on_deadman(True, 0.0, bad)
        assert ls.remaining == 0.0 and ls.factor == 1.0, bad
