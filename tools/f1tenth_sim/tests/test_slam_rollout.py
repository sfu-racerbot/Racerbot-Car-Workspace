"""Analytic rigid-body oracles plus real Gym near-kinematic regression cases."""

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import slam_rollout as sr  # noqa: E402
from sim_fidelity.actuation import TransportDelay  # noqa: E402
from sim_fidelity.calibration import calibration_for_car  # noqa: E402


def analytic(rate=0.0, beta=0.0, speed=2.0, duration=7.0, yaw0=2.8):
    # CG to rear axle: the selected car's own (car 2's from its measured axle weights).
    lr = calibration_for_car().lr
    t = np.arange(round(duration / .05) + 1) * .05
    yaw = yaw0 + rate * t
    if rate:
        dx = speed / rate * (np.sin(yaw + beta) - np.sin(yaw0 + beta))
        dy = speed / rate * (np.cos(yaw0 + beta) - np.cos(yaw + beta))
    else:
        dx, dy = speed * t * np.cos(yaw0 + beta), speed * t * np.sin(yaw0 + beta)
    x = 7 + dx + lr * (np.cos(yaw0) - np.cos(yaw))
    y = -3 + dy + lr * (np.sin(yaw0) - np.sin(yaw))
    u, vcg = speed * np.cos(beta), speed * np.sin(beta)
    rear_speed = np.hypot(u, vcg - lr * rate)
    steer = np.arctan(.36 * rate / u)
    motion = dict(t=t, x=x, y=y, yaw=yaw, longitudinal=np.full(len(t), u),
                  lateral_cg=np.full(len(t), vcg), ground_speed=np.full(len(t), rear_speed),
                  yaw_rate=np.full(len(t), rate), slip_cg=np.full(len(t), beta),
                  lateral_accel=np.full(len(t), rear_speed * rate),
                  trusted=np.ones(len(t), dtype=bool), segment=np.zeros(len(t), dtype=int))
    drive = np.array([[-1, 0, steer, speed], [duration + 1, 0, steer, speed]])
    return motion, drive


class KinematicPlant:
    def __init__(self, cal, seed):
        self.cal = cal
        self.closed = False
        self.chassis_contact_steps = 0
        self.steer_delays = [TransportDelay(cal.steering_delay_ticks)]
        self.speed_delays = [TransportDelay(cal.speed_delay_ticks)]
        self.env = SimpleNamespace(reset=self.full_reset)

    def reset(self, poses, speeds):
        self.steer_delays[0].reset()
        self.speed_delays[0].reset()

    def full_reset(self, options):
        self.state = options['states'][0].copy()
        return {}, {}

    def _observe(self, obs):
        return obs

    def truth(self, index):
        return self.state

    def step(self, commands):
        delta = self.steer_delays[0].push(commands[0][0])
        speed = self.speed_delays[0].push(commands[0][1])
        x, y, _, _, yaw, _, beta = self.state
        rate = speed * np.cos(beta) * np.tan(delta) / self.cal.wheelbase
        end = yaw + rate * sr.CONTROL_DT
        if abs(rate) > 1e-12:
            x += speed / rate * (np.sin(end + beta) - np.sin(yaw + beta))
            y += speed / rate * (np.cos(yaw + beta) - np.cos(end + beta))
        else:
            x += speed * np.cos(yaw + beta) * sr.CONTROL_DT
            y += speed * np.sin(yaw + beta) * sr.CONTROL_DT
        self.state = np.array([x, y, delta, speed, end, rate, beta])

    def close(self):
        self.closed = True


@pytest.mark.parametrize('rate,beta', [(0, 0), (.7, -.08), (-.7, .08)])
def test_exact_straight_and_drifting_turn_replay_matches_analytic_motion(rate, beta):
    motion, drive = analytic(rate, beta)
    cal = calibration_for_car()
    made = []
    def factory(c, seed):
        plant = KinematicPlant(c, seed)
        made.append(plant)
        return plant
    for h in (1, 2, 4):
        result = sr.replay_window(motion, drive, 20, np.arange(len(motion['t'])), h, cal, factory=factory)
        for key in ('speed_rms_mps', 'yaw_rms_rad_s', 'slip_cg_rms_rad',
                    'position_error_m', 'heading_error_rad'):
            assert result[key] < 1e-10
        assert result['slip_samples'] == h * 40
    assert len({id(p) for p in made}) == 3
    assert all(p.closed for p in made)


@pytest.mark.parametrize('delta', [0.0, .015, -.015])
def test_actual_gym_matches_low_force_kinematic_straight_and_turn(delta):
    cal = replace(calibration_for_car(), speed_command_scale=1.0)
    beta = np.arctan(cal.lr / cal.wheelbase * np.tan(delta))
    speed = 1.1
    rate = speed * np.cos(beta) * np.tan(delta) / cal.wheelbase
    motion, drive = analytic(rate, beta, speed)
    result = sr.replay_window(motion, drive, 20, np.arange(len(motion['t'])), 4, cal)
    # Linear tyres depart slightly from ideal no-slip kinematics even at low force.
    assert result['speed_rms_mps'] < .008
    assert result['yaw_rms_rad_s'] < .006
    assert result['slip_cg_rms_rad'] < .006
    assert result['position_error_m'] < .04
    assert result['heading_error_rad'] < .02


def test_initial_state_and_delay_history_are_set_without_warmup():
    motion, _ = analytic(.6, -.07)
    cal = calibration_for_car()
    state = sr.initial_state(motion, 20, .17, cal.lr)
    assert state == pytest.approx([cal.lr, -25, .17, 2, 0, .6, -.07])
    plant = KinematicPlant(cal, 1)
    times = np.arange(-10, 2) * sr.CONTROL_DT
    drive = np.c_[times, times * 0, times + .2, times + 3]
    sr.initialize(plant, state, drive, 0)
    assert plant.truth(0) == pytest.approx(state)
    for delay, column in ((plant.steer_delays[0], 2), (plant.speed_delays[0], 3)):
        assert delay.push(99) == pytest.approx(drive[10 - delay.ticks, column])


def test_windows_respect_breaks_speed_threshold_stride_and_empty_input():
    motion, _ = analytic(duration=6)
    motion['trusted'][60] = False
    motion['segment'][60] = -1
    motion['segment'][61:] = 1
    result = list(sr.windows(motion))
    assert result
    assert not any(h == 4 for h, _, _ in result)
    assert [motion['t'][i] for h, i, ids in result if h == 1] == pytest.approx([0, 1, 3.05, 4.05])
    for h, i, ids in result:
        assert len(np.unique(motion['segment'][ids])) == 1
        assert motion['t'][i] + h <= motion['t'][ids[-1]] + 1e-7
    motion['ground_speed'][:] = .999
    assert list(sr.windows(motion)) == []
    motion['ground_speed'][:] = 1
    assert list(sr.windows(motion))
    motion['trusted'][:] = False
    assert list(sr.windows(motion)) == []
    with pytest.raises(ValueError, match='multiples'):
        list(sr.windows(motion, (.01,)))


def test_command_reader_prefers_drive_and_falls_back_or_refuses(monkeypatch):
    def msg(steer, speed):
        return SimpleNamespace(drive=SimpleNamespace(steering_angle=steer, speed=speed))
    rows = [('/ackermann_cmd', 1, msg(.1, 2)), ('/drive', 2, msg(.2, 3))]
    monkeypatch.setattr(sr.bagread, 'messages', lambda *args: iter(rows))
    drive, topic = sr.read_commands('unused')
    assert topic == '/drive'
    assert drive.tolist() == [[2, 0, .2, 3]]
    rows.pop()
    assert sr.read_commands('unused')[1] == '/ackermann_cmd'
    rows.clear()
    with pytest.raises(ValueError, match='no /drive'):
        sr.read_commands('unused')


def test_scoring_biases_slip_nans_and_aggregation_have_known_oracles():
    motion, _ = analytic(.7, -.08)
    cal = calibration_for_car()
    state = sr.initial_state(motion, 0, .1, cal.lr)
    states = np.tile(state, (40, 1))
    states[:, 3] += .2
    states[:, 5] += .3
    states[:, 6] -= .04
    motion['lateral_accel'][:] = 4
    score = sr.score_window(motion, 0, np.arange(len(motion['t'])), states, cal.lr)
    assert score['yaw_rms_rad_s'] == pytest.approx(.3)
    assert score['slip_cg_rms_rad'] == pytest.approx(.04)
    assert score['turn_aligned_yaw_bias_rad_s'] == pytest.approx(.3)
    assert score['high_ay_abs_slip_bias_rad'] == pytest.approx(.04)
    motion['slip_cg'][:] = np.nan
    assert sr.score_window(motion, 0, np.arange(len(motion['t'])), states, cal.lr)['slip_cg_rms_rad'] is None
    rows = [dict(horizon_s=1, start_t=0, speed_rms_mps=v) for v in (1, 2, 9)]
    summary = sr.aggregate(rows)
    assert summary['1']['metrics']['speed_rms_mps'] == dict(count=3, mean=4, median=2)
    assert summary['4']['windows'] == 0


def test_replay_closes_on_missing_history_and_is_seed_repeatable():
    motion, drive = analytic(.2, .01)
    cal = calibration_for_car()
    made = []
    def factory(c, seed):
        made.append(KinematicPlant(c, seed))
        return made[-1]
    with pytest.raises(ValueError, match='history'):
        sr.replay_window(motion, drive[1:], 0, np.arange(len(motion['t'])), 1, cal, factory=factory)
    assert made[0].closed
    args = motion, drive, 20, np.arange(len(motion['t'])), 1, cal
    first = sr.replay_window(*args)
    sr.replay_window(motion, drive, 30, args[3], 2, cal, seed=42)
    assert sr.replay_window(*args) == first


def test_cli_floor_selection_split_pooling_and_strict_metadata(tmp_path, monkeypatch):
    captured = []
    def analyze(path, car, loc, cal, **kwargs):
        captured.append(cal)
        return dict(split=sr.split_role(car, path.name), windows=[dict(horizon_s=1, start_t=0, speed_rms_mps=2)],
                    summary={'1': dict(windows=1)}, warnings=[])
    monkeypatch.setattr(sr, 'analyze_run', analyze)
    output = tmp_path / 'scores.json'
    assert sr.main(['20261111-135934', '--car', 'rb2', '--floor', 'sfu_course_2026_10_30',
                    '--json', str(output)]) == 0
    data = json.loads(output.read_text(), parse_constant=lambda value: pytest.fail(value))
    assert data['car'] == 'rb2'
    assert data['pooled_by_split']['nov-dev']['1']['windows'] == 1
    assert captured[0] == sr.with_floor('sfu_course_2026_10_30', calibration_for_car('rb2'))
    assert sr.split_role('rb1', '20261116-095242') == 'nov-eval'
    assert sr.split_role('rb2', '20261111-135453') == 'nov-eval'
    assert sr.split_role('rb2', '20261030-185036') == 'oct-fit'
    assert sr.split_role('rb2', '20261030-184826') == 'oct-held-out'
    assert sr.split_role('rb1', '20261030-185036') == 'unassigned'


def test_run_analysis_normal_empty_and_missing_commands(tmp_path, monkeypatch):
    motion, drive = analytic(yaw0=0)
    a = np.zeros((len(motion['t']), 11))
    a[:, :4] = np.c_[motion['t'], motion['x'], motion['y'], motion['yaw']]
    a[:, 4:6] = 1
    a[:, 6] = 2
    a[:, 9] = 2
    path = tmp_path / '20261111-135934'
    saved = tmp_path / (path.name + '.npz')
    np.savez(saved, a=a)
    monkeypatch.setattr(sr, 'read_commands', lambda path: (drive, '/drive'))
    cal = calibration_for_car('rb2')
    result = sr.analyze_run(path, 'rb2', tmp_path, cal, factory=KinematicPlant)
    assert result['split'] == 'nov-dev'
    assert [s['windows'] for s in result['summary'].values()] == [6, 5, 3]
    assert result['summary']['4']['metrics']['speed_rms_mps']['mean'] < 1e-10
    a[:, 4] = 0
    np.savez(saved, a=a)
    empty = sr.analyze_run(path, 'rb2', tmp_path, cal, factory=KinematicPlant)
    assert empty['windows'] == []
    assert empty['trusted_fraction'] == 0
    assert 'no eligible windows' in empty['warnings']
    def missing(path):
        raise ValueError('no commands')
    monkeypatch.setattr(sr, 'read_commands', missing)
    with pytest.raises(ValueError, match='no commands'):
        sr.analyze_run(path, 'rb2', tmp_path, cal, factory=KinematicPlant)
