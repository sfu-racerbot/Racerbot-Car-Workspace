"""The calibrated retraining setup: narrow randomization around the selected
car's fit, club SLAM courses in the training pool with a declared holdout,
the sign-off gate, the scripted-opponent share, and the club-map diagnosis.

Runner: PYTHONPATH=<torch> python3 -m pytest tools/neural_racer/tests/test_retrain_setup.py -v   (CPU)

Oracles: the CPU referee's own calibration (centres), the map's own occupancy
image (club tracks), heldout.yaml (holdout), closed-form dwell times (diagnosis).
"""
import math
from dataclasses import replace

import conftest  # noqa: F401
import numpy as np
import pytest
import torch
import yaml

from neural_racer_train import diagnose, matchmaking, retrain, tracks
from neural_racer_train.gpu_sim.params import (FITTED_RANDOMIZATION, RANDOMIZATION, BatchParams, CarParams,
                                             from_calibration, max_delay_ticks)
from neural_racer_train.gpu_sim.sim import BatchedRaceSim, Level
from neural_racer_train.learn import CLUB_MAP_DIR, Runner, heldout_real
from neural_racer_train.track_pool import TrackPool

torch.set_num_threads(4)
SETUP = conftest.ROOT + '/tools/neural_racer/retrain_calibrated.yaml'


# --- narrow randomization around the fit -------------------------------------------------------

@pytest.mark.parametrize('car_id', ['rb1', 'rb2'])
def test_fitted_spreads_are_centred_on_each_cars_fit(car_id):
    p = from_calibration(car_id)
    b = BatchParams.randomized(p, 4096, torch.Generator().manual_seed(0), profile='fitted')
    for name, spread in (('mu', 0.10), ('C_Sf', 0.10), ('C_Sr', 0.10), ('servo_gain', 0.05)):
        v, nominal = getattr(b, name), getattr(p, name)
        assert float(v.min()) >= nominal * (1 - spread) - 1e-6
        assert float(v.max()) <= nominal * (1 + spread) + 1e-6
        assert abs(float(v.mean()) - nominal) < 0.01 * nominal
    for name in ('steer_delay_ticks', 'speed_delay_ticks'):
        nominal = getattr(p, name)
        assert set(getattr(b, name).long().tolist()) == {nominal - 1, nominal, nominal + 1}
    assert float((b.steer_gain - 1).abs().max()) <= 0.03 + 1e-6


def test_fitted_is_narrower_than_wide_and_wide_is_unchanged():
    p = CarParams()
    wide = BatchParams.randomized(p, 4096, torch.Generator().manual_seed(0))
    assert float(wide.mu.min()) < 0.45 and float(wide.mu.max()) > 1.0     # STOCK-grade range, as before
    assert RANDOMIZATION['mu'] == ('range', (0.40, 1.10))
    for name, (kind, spread) in FITTED_RANDOMIZATION.items():
        wk, ws = RANDOMIZATION[name]
        if kind == 'rel' and wk == 'rel':
            assert spread <= ws, name


def test_delay_lines_hold_the_longest_fitted_delay():
    spec = _ring()
    r = tracks.rasterize(spec)
    pool = TrackPool(1, canvas=r.sdf.shape, max_points=len(spec.centre) + 1, dtype=torch.float32)
    pool.insert(r, 0)
    p = replace(CarParams(), steer_delay_ticks=5)      # beyond the wide line's fixed 4 ticks
    assert max_delay_ticks(p, 'steer_delay_ticks', 'fitted') == 6
    sim = BatchedRaceSim(64, 1, pool, car=p, seed=2, randomization='fitted')
    level = Level(slot=0, poses=np.array([[*spec.centre[0], math.pi / 2]]), start_idx=np.array([0]))
    sim.reset_envs(range(64), [level] * 64)
    assert set(sim.steer_delay.delay.flatten().tolist()) == {4, 5, 6}


def test_unknown_randomization_profile_is_refused():
    spec = _ring()
    pool = TrackPool(1, canvas=(8, 8), max_points=len(spec.centre) + 1)
    with pytest.raises(ValueError, match='randomization'):
        BatchedRaceSim(1, 1, pool, car=CarParams(), randomization='huge')


def _ring(radius=12.0, half_width=4.0):
    t = np.linspace(0, 2 * math.pi, int(2 * math.pi * radius / tracks.SPACING), endpoint=False)
    c = np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)
    w = np.full(len(c), half_width)
    return tracks.TrackSpec('ring', 0, c, w, w)


# --- club courses in the pool --------------------------------------------------------------------

@pytest.mark.parametrize('name', ['car1_course_2026_10_26', 'car2_course_2026_11_06b'])
def test_club_course_keeps_its_slam_walls(name):
    from PIL import Image
    r = tracks.from_gym_map(CLUB_MAP_DIR, name)
    meta = yaml.safe_load((CLUB_MAP_DIR / f'{name}_map.yaml').read_text())
    img = np.asarray(Image.open(CLUB_MAP_DIR / meta['image']).convert('L'))[::-1]
    # every cell the raster calls free is free in the map image itself
    col0 = int(round((r.origin[0] - meta['origin'][0]) / r.resolution))
    row0 = int(round((r.origin[1] - meta['origin'][1]) / r.resolution))
    src = img[row0:row0 + r.occupancy.shape[0], col0:col0 + r.occupancy.shape[1]]
    assert np.all((1.0 - src[~r.occupancy] / 255.0) < meta['free_thresh'])
    # the line runs in free space, with walls on both sides within 3 m
    c = r.spec.centre
    rows = ((c[:, 1] - r.origin[1]) / r.resolution).astype(int)
    cols = ((c[:, 0] - r.origin[0]) / r.resolution).astype(int)
    assert np.mean(r.sdf[rows, cols] > 0) > 0.98
    assert np.median(r.spec.w_left) < 3.0 and np.median(r.spec.w_right) < 3.0
    raw = np.loadtxt(CLUB_MAP_DIR / f'{name}_centerline.csv', delimiter=',', skiprows=1)
    assert r.lap_length == pytest.approx(tracks.lap_length(raw), rel=0.02)


def test_the_race_day_course_is_held_out():
    assert 'car2_course_2026_11_11' in heldout_real()
    with pytest.raises(RuntimeError, match='held-out'):
        Runner(2, 1, 'cpu', 0, pool_slots=2, initial_tracks=0, car_id='rb2',
               club_tracks=('car2_course_2026_11_11',))


def test_club_courses_stay_pinned_in_the_pool():
    runner = Runner(2, 1, 'cpu', 0, pool_slots=2, initial_tracks=0, car_id='rb2',
                    club_tracks=('car2_course_2026_11_06b',))
    club = {s: runner.slot_digest[s] for s in runner.pinned}
    assert len(club) == 1
    for seed in range(3):                       # the pool is full: each insert must evict a procedural slot
        runner._insert(tracks.rasterize(tracks.generate(seed)))
    assert all(runner.slot_digest[s] == d for s, d in club.items())
    assert runner.maker.specs[next(iter(club))].kind == 'club:car2_course_2026_11_06b'


# --- the setup file and its sign-off gate ---------------------------------------------------------

class _Args:
    def __init__(self, path):
        self.retrain_config = path


def test_committed_setup_is_valid_and_unsigned():
    cfg = retrain.load(SETUP)
    assert cfg['randomization'] == 'fitted'
    assert not retrain.signed_off(cfg)
    assert not set(cfg['club_tracks']) & heldout_real()


def test_training_stages_refuse_an_unsigned_setup():
    with pytest.raises(SystemExit, match='not signed off'):
        retrain.runner_kwargs(_Args(SETUP))
    with pytest.raises(SystemExit, match='not signed off'):
        retrain.scripted_share(_Args(SETUP))


def test_a_signed_setup_reaches_the_runner(tmp_path):
    cfg = yaml.safe_load(open(SETUP).read())
    cfg['signed_off_by'] = 'Vera'
    p = tmp_path / 'setup.yaml'
    p.write_text(yaml.safe_dump(cfg))
    kw = retrain.runner_kwargs(_Args(p))
    assert kw == {'car_id': cfg['car'], 'randomization': 'fitted', 'club_tracks': tuple(cfg['club_tracks'])}
    assert retrain.scripted_share(_Args(p)) == (0.30, 0.10)
    assert retrain.runner_kwargs(_Args(None)) == {}


@pytest.mark.parametrize('change, match', [
    ({'club_tracks': ['car2_course_2026_11_11']}, 'held-out'),
    ({'club_tracks': ['no_such_course']}, 'no map'),
    ({'car': 'rb9'}, 'car must be'),
    ({'randomization': 'huge'}, 'randomization'),
    ({'scripted_share': {'base': 0.7, 'floor': 0.1}}, 'scripted_share'),
    ({'surprise': 1}, 'unknown keys'),
])
def test_bad_setups_are_refused(change, match):
    cfg = yaml.safe_load(open(SETUP).read())
    cfg.update(change)
    with pytest.raises(retrain.SetupError, match=match):
        retrain.check(cfg)


# --- scripted opponents -----------------------------------------------------------------------------

def test_scripted_share_reaches_the_main_mix():
    mix = matchmaking.role_mix('main0', 0.0, (0.30, 0.10))
    assert mix['scripted'] == pytest.approx(0.30)
    assert sum(mix.values()) == pytest.approx(1.0)
    assert matchmaking.role_mix('main0', 1.0, (0.30, 0.10))['scripted'] == pytest.approx(0.10)
    assert matchmaking.role_mix('main0', 0.0) == {'self': 0.35, 'pfsp': 0.40, 'scripted': 0.15, 'uniform': 0.10}


# --- the club-map diagnosis ---------------------------------------------------------------------------

def _run(driver, speed, L=10.0, laps=2, finish=True, binding='network'):
    ticks, s, t = [], 0.0, 0.0
    while s < L * laps:
        t += diagnose.DT
        s += speed(s % L) * diagnose.DT
        ticks.append({'t': t, 'x': s, 'y': 0.0, 'v': speed(s % L), 's': s, 'lap': float(s // L),
                      'binding': binding, 'net_speed': 2.0, 'cmd_speed': 1.5, 'clearance': 1.0})
    return {'course': 'c', 'car': 'rb1', 'driver': driver, 'seed': 0, 'laps': laps,
            'finish_time': t if finish else None, 'contact': not finish, 'progress': s, 'ticks': ticks}


def test_diagnosis_finds_the_slow_segment():
    slow = lambda s: 1.0 if 6.0 <= s < 7.0 else 2.0      # noqa: E731
    runs = [_run('neural', slow, binding='curvature'), _run('gap_follow', lambda s: 2.0)]
    c = diagnose.summarize(runs)['c']
    worst = c['worst_segments'][0]
    assert worst['segment_m'] == [6.0, 7.0]
    assert worst['loss_s'] == pytest.approx(0.5, abs=0.06)       # 1 m at 1 m/s against 1 m at 2 m/s
    assert worst['binding'] == 'curvature'
    assert c['drivers']['neural']['mean_speed_cut_by_wrapper'] == pytest.approx(0.5)


def test_the_finish_tick_does_not_count_as_another_lap():
    L, laps, step = 10.0, 2, 2.0 * diagnose.DT
    ticks = [{'s': round(k * step, 6), 'lap': float(round(k * step, 6) // L)}
             for k in range(1, int(round(L * laps / step)) + 1)]
    assert ticks[-1]['s'] == L * laps and ticks[-1]['lap'] == laps    # the referee logs the finish tick
    d = diagnose.dwell({'laps': laps, 'ticks': ticks}, L)
    assert d[0] == pytest.approx(d[5], abs=0.03)


def test_diagnosis_reports_where_a_run_failed():
    runs = [_run('neural', lambda s: 2.0, finish=False), _run('gap_follow', lambda s: 2.0)]
    f = diagnose.summarize(runs)['c']['drivers']['neural']['failures']
    assert len(f) == 1 and f[0]['kind'] == 'contact' and f[0]['s'] == pytest.approx(20.0, abs=0.1)


def test_the_cli_gate_fir_run_uses(tmp_path, capsys):
    assert retrain.main(['--config', SETUP, '--check-only']) == 0
    assert retrain.main(['--config', SETUP, '--check-only', '--require-signoff']) == 3
    cfg = yaml.safe_load(open(SETUP).read())
    cfg['signed_off_by'] = 'Vera'
    p = tmp_path / 'setup.yaml'
    p.write_text(yaml.safe_dump(cfg))
    assert retrain.main(['--config', str(p), '--check-only', '--require-signoff', '--car', cfg['car']]) == 0
    other = 'rb1' if cfg['car'] == 'rb2' else 'rb2'
    assert retrain.main(['--config', str(p), '--check-only', '--require-signoff', '--car', other]) == 2
    assert 'RACERBOT_CAR' in capsys.readouterr().err
