"""A resumed Runner rebuilds the tracks its saved levels point at, and the
safety penalty ignores steering slew.

Runner: PYTHONPATH=<torch> python3 -m pytest tools/neural_racer/tests/test_resume.py -v   (CPU)
"""
import math

import torch

from neural_racer.safety_wrapper import WrapperConfig
from neural_racer_train.gpu_sim.safety_torch import SafetyTorch, nominal_angles
from neural_racer_train.learn import Runner


def _runner(seed):
    return Runner(4, 1, 'cpu', seed, pool_slots=8, initial_tracks=3, solo=True, warmup_levels=0)


def test_resume_rebuilds_tracks():
    a = _runner(0)
    a.end_episodes(torch.ones(4, dtype=torch.bool))
    st = a.state()
    b = _runner(1)                         # a fresh process: different initial tracks
    b.load(st['plr'], st['slot_seed'])
    assert b.levels.levels, 'every saved level was dropped'
    for r in b.levels.levels.values():
        assert b.slot_digest[r.meta['slot']] == r.track_digest
    for _ in range(5):                     # the KeyError of job 505481 came from here
        b.end_episodes(torch.ones(4, dtype=torch.bool))


def test_resume_without_slot_seeds_drops_unknown_levels():
    a = _runner(0)
    st = a.state()
    b = _runner(1)
    b.load(st['plr'], None)
    for r in b.levels.levels.values():
        assert b.slot_digest.get(r.meta['slot']) == r.track_digest
    b.end_episodes(torch.ones(4, dtype=torch.bool))


def test_steering_slew_is_not_a_safety_stop():
    st = SafetyTorch(WrapperConfig(), nominal_angles())
    ranges = torch.full((1, 2, len(nominal_angles())), 20.0)
    steer = torch.tensor([[0.3, 0.0]])                 # car 0 jumps 0.3 rad in one tick
    speed = torch.tensor([[1.0, 1.0]])
    _, _, _, intervened, stop, _ = st.apply_detailed(steer, speed, ranges, 0.05, 30.0, torch.zeros(1, 2), 0.025,
                                                  torch.full((1, 2), 6.0))
    assert bool(intervened[0, 0]) and not bool(stop[0, 0])
    near = torch.full((1, 2, len(nominal_angles())), 0.3)  # a wall 0.3 m away at 1 m/s brakes
    _, v, _, _, stop, _ = st.apply_detailed(torch.zeros(1, 2), speed, near, 0.05, 30.0, torch.zeros(1, 2), 0.025,
                                         torch.full((1, 2), 6.0))
    assert bool(stop.all()) and float(v.max()) < 1.0 - 1e-6 and math.isfinite(float(v.max()))
