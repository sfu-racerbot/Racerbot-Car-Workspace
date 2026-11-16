"""PyTorch training network vs the car's NumPy forward pass, through export.

Runner: PYTHONPATH=<torch> python3 -m pytest tools/neural_racer/tests/test_export_golden.py -v

Oracle: PyTorch's own forward pass -- an independent implementation from
policy_numpy.py's im2col version. Then the real approve_model, unmodified,
must accept the exported directory (it re-runs the golden check itself).
"""
import json

import conftest  # noqa: F401
import numpy as np
import pytest
import torch

from neural_racer import approve_model, model_store, observation as obs
from neural_racer.policy_numpy import PolicyNumpy
from neural_racer_train import export
from neural_racer_train.model import FULL, HALF, PolicyNet


def _randomized(width=FULL, seed=0, recurrent=True):
    torch.manual_seed(seed)
    p = PolicyNet(width, recurrent)
    with torch.no_grad():
        for name, v in p.named_parameters():       # move off PyTorch's init so biases matter
            if 'camera' not in name:
                v.add_(torch.randn_like(v) * 0.05)
        ln = p.fuse['ln'].normalized_shape[0]
        p.fuse['fc'].weight[:, ln:] = 0.0
        p.ego['fc0'].weight[:, 21:23] = 0.0
    return p


@pytest.mark.parametrize('width,recurrent', [(FULL, True), (HALF, True), (FULL, False)])
def test_numpy_reproduces_pytorch(width, recurrent):
    """recurrent False: the MLP ablation, whose card the NumPy network must read (2026-11-06)."""
    policy = _randomized(width, recurrent=recurrent)
    g = export.golden(policy, export.synthetic_inputs())
    w = export.weights_of(policy)
    err = 0.0
    numpy_policy = PolicyNumpy(w, {'recurrent': recurrent})
    for s in range(g['action'].shape[0]):
        numpy_policy.reset()
        for t in range(g['action'].shape[1]):
            out = numpy_policy.step({k: g[k][s, t] for k in ('scan', 'ego', 'preview', 'race', 'camera')})
            err = max(err, abs(out.steering - g['action'][s, t, 0]), abs(out.speed - g['action'][s, t, 1]),
                      float(np.abs(out.ego_path - g['ego_path'][s, t]).max()))
    assert err <= 1e-5, err


def test_state_dict_names_are_policy_numpy_names():
    w = export.weights_of(PolicyNet())
    assert 'log_std' not in w
    PolicyNumpy(w)                                    # raises PolicyError on any missing name


def test_camera_weights_start_at_zero():
    p = PolicyNet()
    ln = p.fuse['ln'].normalized_shape[0]
    assert torch.all(p.fuse['fc'].weight[:, ln:] == 0)
    assert torch.all(p.ego['fc0'].weight[:, obs.EGO['camera_present']:obs.EGO['camera_age'] + 1] == 0)


def test_mlp_ablation_ignores_memory():
    p = _randomized(recurrent=False)
    o = {k: torch.as_tensor(v[0, :1]) for k, v in export.synthetic_inputs().items()}
    a1, h, _, _ = p(o, torch.zeros(1, 128))
    a2, _, _, _ = p(o, h)                            # a non-zero hidden state, ignored
    assert torch.equal(a1, a2)


@pytest.mark.parametrize('recurrent', [True, False])
def test_exported_dir_passes_approve_model(tmp_path, recurrent):
    policy = _randomized(recurrent=recurrent)
    d = export.export(policy, tmp_path / 'exports', run_name='test', snapshot_id='0')
    for name in ('selection_report.json', 'referee_report.json'):
        (d / name).write_text(json.dumps({'pass': True}))
    h = approve_model.approve(d, tmp_path / 'models', shadow_only=False)
    loaded, card, approval = model_store.load_approved(h, models_dir=tmp_path / 'models')
    assert approval == 'full' and card['parameters'] == 646_338      # spec v2: 64 more for the cornering cap


def test_spec1_export_folds_the_cornering_cap(tmp_path):
    """--car-spec 1 (2026-11-11): main's car code reads 23 ego values, no cornering cap.
    Oracle: the trained network told a_lat (export.golden), against the folded weights
    run through the car's NumPy pass -- with a zero cap column, which is the v1 network."""
    import yaml
    policy = _randomized(seed=4)
    d = export.export(policy, tmp_path, run_name='t', snapshot_id='0', car_spec=1, a_lat=2.5)
    card = yaml.safe_load((d / 'model_card.yaml').read_text())
    assert card['obs_spec_version'] == 1 and card['a_lat_folded'] == 2.5
    assert card['parameters'] == 646_338 - 64
    with np.load(d / 'weights.npz') as z:
        w = {k: z[k] for k in z.files}
    with np.load(d / 'golden.npz') as z:
        g = {k: z[k] for k in z.files}
    assert w['ego.fc0.weight'].shape == (64, obs.EGO_DIM - 1) and g['ego'].shape[-1] == obs.EGO_DIM - 1
    w['ego.fc0.weight'] = np.concatenate([w['ego.fc0.weight'], np.zeros((64, 1), np.float32)], axis=1)
    net = PolicyNumpy(w, {'recurrent': True})
    worst = 0.0
    for s in range(g['action'].shape[0]):
        net.reset()
        for t in range(g['action'].shape[1]):
            o = {k: g[k][s, t] for k in ('scan', 'preview', 'race', 'camera')}
            o['ego'] = np.append(g['ego'][s, t], np.float32(0.0))
            out = net.step(o)
            worst = max(worst, abs(out.steering - g['action'][s, t, 0]), abs(out.speed - g['action'][s, t, 1]))
    assert worst < 1e-5
    # and the cap matters: told 5.0 instead, the trained network answers differently
    inputs = export.synthetic_inputs()
    inputs['ego'][..., obs.EGO['a_lat']] = 5.0 / obs.A_LAT_SCALE
    assert np.abs(export.golden(policy, inputs)['action'] - g['action']).max() > 1e-3


def test_spec1_from_export_matches_direct_export(tmp_path):
    policy = _randomized(seed=5)
    v2 = export.export(policy, tmp_path / 'v2', run_name='t', snapshot_id='0')
    direct = export.export(policy, tmp_path / 'direct', run_name='t', snapshot_id='0', car_spec=1, a_lat=2.0)
    assert export.main([str(tmp_path / 'redo'), '--from-export', str(v2), '--car-spec', '1', '--a-lat', '2.0']) == 0
    assert (tmp_path / 'redo' / direct.name).is_dir()           # same weights, same hash
    with pytest.raises(ValueError):
        export.export(policy, tmp_path / 'bad', run_name='t', snapshot_id='0', car_spec=1)   # no cap given


def test_manual_gru_equals_grucell():
    """Oracle: PyTorch's own nn.GRUCell (the manual form exists only so vmap batches it)."""
    from neural_racer_train.model import gru_cell
    torch.manual_seed(0)
    cell = torch.nn.GRUCell(256, 128)
    x, h = torch.randn(7, 256), torch.randn(7, 128)
    assert torch.allclose(gru_cell(x, h, cell), cell(x, h), atol=1e-6)


def test_stacked_opponents_match_individual_policies():
    """16 stacked opponents in one vmap call give each policy's own actions."""
    from neural_racer_train.opponents import StackedOpponents
    pols = [_randomized(seed=s) for s in range(4)]
    stacked = StackedOpponents([p.state_dict() for p in pols])
    inputs = export.synthetic_inputs(seqs=3, steps=1)
    o = {k: torch.as_tensor(v[:, 0]) for k, v in inputs.items()}           # B = 3
    ob = {k: v.unsqueeze(0).expand(4, *v.shape) for k, v in o.items()}
    a, _ = stacked.act(ob, torch.zeros(4, 3, 128))
    for m, p in enumerate(pols):
        ref, _, _, _ = p(o, torch.zeros(3, 128))
        assert torch.allclose(a[m], ref, atol=1e-5)
