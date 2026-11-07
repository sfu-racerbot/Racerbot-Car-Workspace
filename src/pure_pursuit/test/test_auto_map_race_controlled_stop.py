"""The end of mapping brakes the car to a stop under gap_follow's steering.

On 2026-11-06 (car 2, runs 20261106-164002, -175408, -175837, -181002, -184753)
auto_map_race_node answered the last lap's closure with an all-zero command:
speed 0 *and steering 0*, from 3+ m/s, often in a bend. In 4 of 5 runs
gap_follow then logged the car's body inside a wall about a second later
(clearance -0.009 to -0.146 m) while asking for -0.28 to -0.35 rad. These
tests pin the replacement: when mapping ends, gap_follow's command keeps
going out with its steering unchanged and its speed capped by a ramp
v0 - mapping_stop_decel * t, then the node holds a full stop as before.

Runner: rclpy, so ROS sourced and the test domain isolated (CLAUDE.md):

    export ROS_DOMAIN_ID=79 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
    python3 -m pytest src/pure_pursuit/test/test_auto_map_race_controlled_stop.py -v

Every command is read from /test_only/drive. Time is injected exactly as in
test_auto_map_race_deadman.py, whose rig this reuses. Only what needs SLAM or
TF is stubbed: the recorder's lap detection, the pose lookups, the profile
writer and the map save.
"""

import types

from ackermann_msgs.msg import AckermannDriveStamped
import pytest
import rclpy

from test_auto_map_race_deadman import FLOAT32_TOL, _Rig, _spin_until  # noqa: F401
from pure_pursuit.auto_map_race_node import AutoMapRaceNode

TICK = 0.025               # s, the node's 40 Hz control period
DECEL = 2.0                # m/s^2, mapping_stop_decel set below
HOLD = 0.5                 # s, mapping_stop_hold_sec set below
CLOSING_SPEED = 3.0        # m/s, what gap_follow asks as the last lap closes
CLOSING_STEER = -0.30      # rad, in a right-hand bend (2026-11-06: -0.28 to -0.35)


def _stub_recorder(close_on_call):
    """A LapRecorder stand-in that reports one closed lap on call number close_on_call."""
    calls = {'n': 0}

    def update(*_args):
        calls['n'] += 1
        return calls['n'] == close_on_call

    return types.SimpleNamespace(
        update=update, reset=lambda: None, lap_points=lambda: [], points=[],
        close_at_closest_approach=False, distance=32.0, elapsed=12.0, turn=6.3,
        revolutions=1.0, closest_approach=0.05, effective_closure_distance=1.5,
        closure_distance=1.5, max_closure_distance=4.0, reanchor_count=0,
        start=(0.0, 0.0, 0.0), min_turn_rad=5.2, closure_heading_rad=0.52,
        distance_from_start=0.05, heading_error=0.0, min_distance=5.0,
        required_duration_sec=4.0, departed=True)


class _StopRig(_Rig):
    def __init__(self, node, write_profile=lambda: '/tmp/raceline_profiled.csv'):
        super().__init__(node)
        node._lookup_and_publish_pose = lambda: (1.0, 2.0, 0.3)
        node._lookup_odom_pose = lambda: (0.3, (1.0, 2.0))
        node._update_pf_handover = lambda *_a: None
        node.completed_mapping_laps = node.mapping_laps - 1
        node._lap_quality_ok = lambda: (True, [])
        node._write_profile = write_profile
        node.recorder = _stub_recorder(close_on_call=2)
        self.saves = []
        node._try_save_map = lambda now_sec=None: self.saves.append(now_sec)
        node._map_save_settled = lambda now_sec: False

    def command(self, speed, steering):
        msg = AckermannDriveStamped()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.now += 1e-4
        self.cmd_pub.publish(msg)
        assert _spin_until(self.node, lambda: self.node.mapping_cmd_time == self.now), \
            'the node never took the mapping command'

    def step(self, speed=CLOSING_SPEED, steering=CLOSING_STEER, lb=True):
        """One tick TICK after the last: fresh /joy and gap_follow command, then the tick."""
        self.now += TICK
        self.feed_joy(lb)
        if speed is not None:
            self.command(speed, steering)
        return self.tick()[-1].drive

    def close_last_lap(self):
        """One mapping tick, then the tick on which the last lap closes. Returns its time."""
        self.step()
        assert self.node.state == 'mapping'
        out = self.step()
        assert self.node.state != 'mapping', 'the stubbed recorder should have closed the lap'
        return self.now, out


def _node(finish_after_mapping=True):
    rclpy.init(args=['--ros-args',
                     '-p', 'enable_deadman:=true',
                     '-p', 'control_rate_hz:=0.01',
                     '-p', 'decision_log_period_sec:=0.0',
                     '-p', f'finish_after_mapping:={str(finish_after_mapping).lower()}',
                     '-p', f'mapping_stop_decel:={DECEL}',
                     '-p', f'mapping_stop_hold_sec:={HOLD}',
                     '-p', 'drive_topic:=/test_only/drive'])
    return AutoMapRaceNode()


@pytest.fixture
def rig():
    node = _node()
    try:
        assert node.drive_pub.topic_name == '/test_only/drive'
        yield _StopRig(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_the_closing_tick_keeps_gap_follows_steering(rig):
    """The tick the last lap closes on still steers where gap_follow asks.

    Fails on the old code, whose closing tick went out as steering 0.
    """
    _t0, out = rig.close_last_lap()
    # oracle: closed form; the ramp starts at v0 = the closing command's speed.
    assert out.steering_angle == pytest.approx(CLOSING_STEER, abs=FLOAT32_TOL)
    assert out.speed == pytest.approx(CLOSING_SPEED, abs=FLOAT32_TOL)


def test_speed_follows_the_ramp_to_zero_with_steering_kept(rig):
    t0, _ = rig.close_last_lap()
    ramp_end = CLOSING_SPEED / DECEL          # 1.5 s, closed form
    previous = CLOSING_SPEED
    while rig.now - t0 < ramp_end + HOLD - 2 * TICK:
        out = rig.step()
        t = rig.now - t0
        # oracle: closed form v(t) = max(0, v0 - decel * t), gap_follow asking more.
        expected = max(0.0, CLOSING_SPEED - DECEL * t)
        assert out.speed == pytest.approx(expected, abs=1e-3), f't={t:.3f}s'
        assert out.speed <= previous + FLOAT32_TOL
        assert out.steering_angle == pytest.approx(CLOSING_STEER, abs=FLOAT32_TOL), f't={t:.3f}s'
        previous = out.speed
    assert previous == 0.0


def test_after_the_ramp_and_hold_the_node_holds_a_full_stop(rig):
    t0, _ = rig.close_last_lap()
    while rig.now - t0 < CLOSING_SPEED / DECEL + HOLD + 2 * TICK:
        rig.step()
    for _ in range(3):
        out = rig.step()
        # oracle: spec, loading_profile publishes the default all-zero message.
        assert out.speed == 0.0
        assert out.steering_angle == 0.0


def test_the_ramp_never_asks_for_more_than_gap_follow(rig):
    """gap_follow's own stop (its emergency tiers) wins mid-ramp."""
    rig.close_last_lap()
    rig.step()
    out = rig.step(speed=0.0, steering=0.20)
    # oracle: closed form, min(gap_follow 0.0, ramp > 0) = 0, its steering kept.
    assert out.speed == 0.0
    assert out.steering_angle == pytest.approx(0.20, abs=FLOAT32_TOL)
    out = rig.step(speed=1.0, steering=0.10)
    assert out.speed == pytest.approx(1.0, abs=FLOAT32_TOL)


def test_lb_released_during_the_ramp_is_a_full_stop_at_once(rig):
    rig.close_last_lap()
    rig.step()
    out = rig.step(lb=False)
    # oracle: spec docs/architecture.md, no non-zero command without LB.
    assert out.speed == 0.0
    assert out.steering_angle == 0.0


def test_a_stale_gap_follow_command_during_the_ramp_is_a_full_stop(rig):
    rig.close_last_lap()
    rig.now += rig.node.command_timeout_sec + 0.05
    rig.feed_joy(True)
    out = rig.tick()[-1].drive
    # oracle: spec, a missing or stale selected command publishes stop (_command_status).
    assert out.speed == 0.0
    assert out.steering_angle == 0.0


def test_the_map_save_waits_for_the_car_to_stop(rig):
    t0, _ = rig.close_last_lap()
    while rig.now - t0 < CLOSING_SPEED / DECEL + HOLD - TICK:
        rig.step()
        assert rig.saves == [], f'save started while the car was still braking at t={rig.now - t0:.3f}s'
    while rig.now - t0 < CLOSING_SPEED / DECEL + HOLD + 2 * TICK:
        rig.step()
    assert rig.saves, 'the save never started once the car had stopped'


def test_a_refused_profile_in_race_mode_also_brakes_with_steering():
    """Map-then-race: a profile that can't be generated goes to 'error', which
    used to stop with steering 0 as well."""
    node = _node(finish_after_mapping=False)
    try:
        def refuse():
            raise ValueError('the cleaned racing line asks for more steering than the car has')
        rig = _StopRig(node, write_profile=refuse)
        t0, _ = rig.close_last_lap()
        assert node.state == 'error'
        out = rig.step()
        assert out.steering_angle == pytest.approx(CLOSING_STEER, abs=FLOAT32_TOL)
        # oracle: closed form, v0 - decel * t one tick into the ramp.
        assert out.speed == pytest.approx(CLOSING_SPEED - DECEL * (rig.now - t0), abs=1e-3)
        assert out.speed < CLOSING_SPEED
    finally:
        node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize('bad', ['0.0', '-1.0', 'nan'])
def test_a_non_positive_decel_is_refused_at_start(bad):
    rclpy.init(args=['--ros-args', '-p', f'mapping_stop_decel:={bad}',
                     '-p', 'drive_topic:=/test_only/drive'])
    try:
        with pytest.raises(ValueError, match='mapping_stop_decel'):
            AutoMapRaceNode()
    finally:
        rclpy.shutdown()
