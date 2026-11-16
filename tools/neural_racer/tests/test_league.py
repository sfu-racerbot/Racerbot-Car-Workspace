"""The league learner (Stage 3) on the shared PPO update: a tiny run from a
distilled student with the teacher's kickstarting term trains, freezes the
actor while the critic warms up, and resumes.

Runner: PYTHONPATH=<torch> python3 -m pytest tools/neural_racer/tests/test_league.py -v   (CPU)
"""
import json

import conftest  # noqa: F401
import pytest
import torch

from neural_racer_train import league_learner
from neural_racer_train.model import HALF, PolicyNet, TeacherNet

torch.set_num_threads(4)


def test_kickstart_weight_fades():
    assert league_learner.kickstart_weight(1.0, 0.1, 0, 1e9) == 1.0
    assert league_learner.kickstart_weight(1.0, 0.1, 0.05e9, 1e9) == pytest.approx(0.5)
    assert league_learner.kickstart_weight(1.0, 0.1, 0.2e9, 1e9) == 0.0


def test_per_sample_kl_averages_to_the_batch_kl():
    from neural_racer_train.ppo import gaussian_kl
    m0, m1 = torch.randn(20, 2), torch.randn(20, 2)
    s0, s1 = torch.tensor([-1.0, -0.5]), torch.tensor([-0.8, -0.7])
    assert float(league_learner.gaussian_kl_each(m0, m1, s0, s1).mean()) == pytest.approx(
        float(gaussian_kl(m0, m1, s0, s1)), rel=1e-6)


def test_a_tiny_league_role_trains_with_the_critic_warming_up(tmp_path):
    torch.save({'policy': PolicyNet(HALF).state_dict(), 'policy_config': {'width': HALF, 'recurrent': True}},
               tmp_path / 'student.pt')
    torch.save({'actor': TeacherNet().state_dict(), 'squash': 'capped'}, tmp_path / 'teacher.pt')
    run = tmp_path / 'league' / 'main0'
    argv = ['--run-dir', str(run), '--archive', str(tmp_path / 'league' / 'archive'), '--role', 'main0',
            '--init-from', str(tmp_path / 'student.pt'), '--teacher', str(tmp_path / 'teacher.pt'), '--device', 'cpu',
            '--num-envs', '2', '--rollout', '4', '--pool-slots', '2', '--max-cars', '2', '--opponents', '2',
            '--critic-warmup', '2', '--micro', '1', '--log-every', '1', '--snapshot-every', '2', '--kickstart-frac', '1.0',
            '--eval-every', '2', '--eval-starts', '1', '--eval-tracks', '1', '--eval-seconds', '1.0']
    assert league_learner.main(argv + ['--steps', '24']) == 0
    rows = [json.loads(l) for l in (run / 'metrics.jsonl').read_text().splitlines()]
    lines = [r for r in rows if 'kind' not in r]
    assert [l['actor_trained'] for l in lines] == [False, False, True]
    assert all(l['kickstart'] > 0 for l in lines)
    assert {r['update'] for r in rows if r.get('kind') == 'eval'} == {2, 3}     # every 2 updates, and at the end
    assert (run / 'checkpoints' / 'best.pt').exists()
    assert league_learner.main(argv + ['--steps', '32', '--resume']) == 0
    assert torch.load(run / 'checkpoints' / 'latest.pt', weights_only=False)['steps'] == 32


def test_opponent_table_keeps_each_race_s_drivers():
    """A refresh waits until no racing car still uses the other half: drivers
    never change mid-race (they did every rollout before)."""
    from neural_racer_train.learn import SeatAssignment
    from neural_racer_train.matchmaking import OpponentRef
    E, C = 6, 3
    gen = torch.Generator().manual_seed(0)
    seats = SeatAssignment(E, C, 'cpu')
    alive = torch.ones(E, C, dtype=torch.bool)
    table = league_learner.OpponentTable([OpponentRef('scripted', 'gap_follow')] * 2)
    seats.assign(list(range(E)), table.n, gen, table.offset)
    assert bool((seats.source < 2).all())
    first = seats.source.clone()
    assert table.refresh([OpponentRef('self', 'self')] * 2, seats, alive)        # half 1 was free
    assert table.live == 1 and torch.equal(seats.source, first)                 # racing cars keep theirs
    assert not table.refresh([OpponentRef('scripted', 'parked')] * 2, seats, alive)   # half 0 still racing
    seats.assign([0, 1, 2], table.n, gen, table.offset)                         # three races restart
    assert bool((seats.source[:3] >= 2).all())
    alive[3:] = False                                                            # the rest end
    assert table.refresh([OpponentRef('scripted', 'parked')] * 2, seats, alive) and table.live == 0


def test_an_mlp_opponent_drives_as_it_was_trained():
    from neural_racer_train import export
    from neural_racer_train.opponents import StackedOpponents
    torch.manual_seed(0)
    for recurrent in (False, True):
        own = PolicyNet(recurrent=recurrent).eval()
        stacked = StackedOpponents([own.state_dict()], recurrent=[recurrent])
        x = export.synthetic_inputs(seqs=1, steps=6)
        h_own, h_st = torch.zeros(1, 128), torch.zeros(1, 1, 128)
        with torch.no_grad():
            for t in range(6):
                o = {k: torch.as_tensor(v[0, t:t + 1]) for k, v in x.items()}
                a_own, h_own, _, _ = own(o, h_own)
                a_st, h_st = stacked.act({k: v.unsqueeze(0) for k, v in o.items()}, h_st)
                assert torch.allclose(a_own, a_st[0], atol=1e-5)


def test_a_crashed_car_loses_to_one_that_did_not():
    import numpy as np
    place = np.array([1.0, 2.0, 3.0])
    assert league_learner.won(place, np.array([False, False, False]), 1)
    assert not league_learner.won(place, np.array([True, False, False]), 2)      # ahead, but crashed
    assert league_learner.won(np.array([3.0, 2.0, 1.0]), np.array([False, True, True]), 2)
    assert league_learner.won(place, np.array([True, True, True]), 2)            # both out: the place


def test_the_teacher_s_critic_gets_the_other_cars_as_new_inputs():
    """Its normaliser only ever saw zeros there (clipping every real car); its
    weights there never trained. Afterwards it computes what it did."""
    from neural_racer_train.gpu_sim.sim import PRIV_DIM, PRIV_OTHERS, PRIV_PLACE
    from neural_racer_train.model import Critic, fresh_opponent_inputs, solo_view
    torch.manual_seed(0)
    critic = Critic().eval()
    priv = torch.randn(64, PRIV_DIM)
    solo = solo_view(priv)
    assert bool((solo[:, PRIV_OTHERS] == 0).all()) and bool((solo[:, PRIV_PLACE] == 0.25).all())
    with torch.no_grad():
        critic.enc.norm.update(solo[:, 256:PRIV_DIM - 64])
        fresh_opponent_inputs(critic.enc, priv)
        assert torch.allclose(critic(priv), critic(solo), atol=1e-5)
    base = 4 * 64
    std = critic.enc.norm.std()[PRIV_OTHERS.start - base:PRIV_OTHERS.stop - base]
    assert bool((std > 0.5).all())                    # the sample's spread, not the 0.05 floor


def test_referee_workers_sharing_an_archive_race_each_snapshot_once(tmp_path):
    """eval_worker.claim: the first worker takes a snapshot, a second skips it, and a claim left by a
    worker that died is taken over once it is older than CLAIM_TTL."""
    import os
    import time
    from neural_racer_train import eval_worker
    lock = tmp_path / 'eval' / 'main0' / '00000200.claim'
    assert eval_worker.claim(lock) and not eval_worker.claim(lock)
    old = time.time() - eval_worker.CLAIM_TTL - 10
    os.utime(lock, (old, old))
    assert eval_worker.claim(lock)
