"""Distillation (Stage 2): the batched sequence update equals the per-step
one, its losses equal the old ones, and a tiny run trains, evaluates and
writes best.pt.

Runner: PYTHONPATH=<torch> python3 -m pytest tools/neural_racer/tests/test_distill.py -v   (CPU)
"""
import conftest  # noqa: F401
import pytest
import torch

from neural_racer import observation as obs
from neural_racer_train import distill, learn
from neural_racer_train.model import HALF, PolicyNet, TeacherNet

torch.set_num_threads(4)


def _obs(T, N, g):
    return {'scan': torch.rand(T, N, 4, 270, generator=g), 'ego': torch.rand(T, N, obs.EGO_DIM, generator=g),
            'preview': torch.randn(T, N, 4, obs.PREVIEW_POINTS, generator=g),
            'race': torch.rand(T, N, obs.RACE_DIM, generator=g), 'camera': torch.zeros(T, N, obs.CAMERA_DIM)}


@pytest.mark.parametrize('recurrent', [True, False])
def test_unroll_equals_stepping_the_whole_network(recurrent):
    g = torch.Generator().manual_seed(0)
    T, N = 6, 3
    o = _obs(T, N, g)
    o['ego'][..., obs.EGO['mode']] = (torch.rand(T, N, generator=g) > 0.5).float()
    start = torch.rand(T, N, generator=g) < 0.3
    h0 = torch.randn(N, 128, generator=g)
    net = PolicyNet(HALF, recurrent=recurrent)
    a, path, opp, H = net.unroll(o, h0, start)
    loss_b = a.square().sum() + path.square().sum() + opp.square().sum()
    grads_b = torch.autograd.grad(loss_b, list(net.parameters()), allow_unused=True)
    h, loss_s = h0, 0.0
    for t in range(T):
        at, h, pt, ot_ = net({k: v[t] for k, v in o.items()}, h * (~start[t]).float().unsqueeze(1))
        assert torch.allclose(at, a[t], atol=1e-5) and torch.allclose(h, H[t], atol=1e-5)
        assert torch.allclose(pt, path[t], atol=1e-5) and torch.allclose(ot_, opp[t], atol=1e-5)
        loss_s = loss_s + at.square().sum() + pt.square().sum() + ot_.square().sum()
    grads_s = torch.autograd.grad(loss_s, list(net.parameters()), allow_unused=True)
    for gb, gs in zip(grads_b, grads_s):
        assert (gb is None) == (gs is None)
        if gb is not None:
            assert torch.allclose(gb, gs, atol=1e-4, rtol=1e-4)


def test_per_sample_losses_match_the_batch_losses():
    g = torch.Generator().manual_seed(1)
    n = 50
    path, ego_t = torch.randn(n, obs.AUX_PATH_POINTS, 2, generator=g), torch.randn(n, obs.AUX_PATH_POINTS, 2, generator=g)
    opp = torch.randn(n, obs.OPP_SLOTS, 13, generator=g)
    opp_t = torch.randn(n, obs.OPP_SLOTS, 13, generator=g)
    opp_t[..., 0] = (torch.rand(n, obs.OPP_SLOTS, generator=g) > 0.5).float()
    lp, bce, pos, cnt = distill.aux_errors(path, opp, ego_t, opp_t)
    ref_p, ref_o = learn.aux_loss(path, opp, ego_t, opp_t)
    assert float(lp.mean()) == pytest.approx(float(ref_p), rel=1e-5)
    assert float(bce.mean() + pos.sum() / cnt.sum()) == pytest.approx(float(ref_o), rel=1e-5)


def test_a_tiny_distillation_trains_and_keeps_its_best(tmp_path):
    teacher = TeacherNet()
    torch.save({'actor': teacher.state_dict(), 'squash': 'capped'}, tmp_path / 'teacher.pt')
    run = tmp_path / 'run'
    code = distill.main(['--run-dir', str(run), '--teacher', str(tmp_path / 'teacher.pt'), '--device', 'cpu',
                         '--num-envs', '2', '--seq', '4', '--steps', '16', '--pool-slots', '2', '--max-cars', '2',
                         '--width', 'half', '--latent', '0.1', '--micro', '1', '--eval-every', '1',
                         '--eval-starts', '1', '--eval-tracks', '1', '--eval-seconds', '1.0', '--log-every', '1',
                         '--min-teacher-finish', '0', '--max-teacher-stop', '1'])     # an untrained teacher
    assert code == 0
    ck = torch.load(run / 'checkpoints' / 'best.pt', weights_only=False)
    assert ck['policy_config']['width'] == HALF and 'latent' in ck and ck['eval']['all']['episodes'] == 3
    lines = (run / 'metrics.jsonl').read_text().splitlines()
    assert any('"loss_latent"' in l for l in lines) and any('"kind": "reference"' in l for l in lines)


@pytest.mark.parametrize('finish, stop, ok', [(0.97, 0.01, True), (0.0, 0.995, False), (0.5, 0.01, False),
                                              (0.97, 0.5, False)])
def test_a_teacher_that_cannot_drive_the_student_races_stops_the_run(finish, stop, ok):
    r = {'finish': finish, 'safety_stop': stop}
    if ok:
        distill.check_reference(r, 0.8, 0.2)
    else:
        with pytest.raises(SystemExit, match='not training'):
            distill.check_reference(r, 0.8, 0.2)
