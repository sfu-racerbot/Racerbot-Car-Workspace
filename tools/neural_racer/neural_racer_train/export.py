"""export: a trained checkpoint -> the model directory the car loads (plan B0).

    python -m neural_racer_train.export <checkpoint.pt> <out_dir> [--car-spec 1 --a-lat 2.0]
    python -m neural_racer_train.export --from-export <model_dir> <out_dir> --car-spec 1 --a-lat 2.0

Writes <out_dir>/<hash>/ with weights.npz (float32, policy_numpy's names),
model_card.yaml and golden.npz: 4 sequences x 16 ticks of inputs, with
PyTorch's own outputs for them, which approve_model on the car must
reproduce to 1e-5 before the model can load.

--car-spec 1 exports for car code still on observation spec v1 (main, 2026-11-11),
whose ego block lacks v2's last value, the cornering cap. That input is folded into
ego.fc0's bias at --a-lat, the car's max_lateral_accel, so the v1 model computes
exactly what the v2 network does at that cap; the golden outputs come from the v2
network itself. The model then assumes that cap: re-export when the car's changes.
"""
import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from neural_racer_train import common  # noqa: F401  (paths)
from neural_racer import model_store, observation as obs
from neural_racer_train.model import PolicyNet, squash

EXCLUDE = ('log_std',)


def weights_of(policy: PolicyNet) -> dict:
    return {k: v.detach().cpu().float().numpy() for k, v in policy.state_dict().items() if k not in EXCLUDE}


def synthetic_inputs(seqs: int = 4, steps: int = 16, seed: int = 0) -> dict:
    """Plausible observation sequences when no recorded rollout is given:
    valid ranges, both modes, IMU on and off, speed caps across the ladder."""
    g = np.random.default_rng(seed)
    out = {'scan': g.random((seqs, steps, 4, 270)).astype(np.float32),
           'ego': (g.random((seqs, steps, obs.EGO_DIM)) * 2 - 1).astype(np.float32),
           'preview': g.normal(0, 0.5, (seqs, steps, 4, obs.PREVIEW_POINTS)).astype(np.float32),
           'race': g.random((seqs, steps, obs.RACE_DIM)).astype(np.float32),
           'camera': np.zeros((seqs, steps, obs.CAMERA_DIM), np.float32)}
    out['scan'][:, :, 3] = (out['scan'][:, :, 3] > 0.1)
    e = out['ego']
    e[..., obs.EGO['mode']] = np.arange(seqs)[:, None] % 2
    e[..., obs.EGO['imu_present']] = (np.arange(seqs)[:, None] // 2) % 2 == 0
    e[..., obs.EGO['v_cap']] = np.linspace(0.2, 1.27, seqs)[:, None]
    e[..., obs.EGO['camera_present']] = 0.0
    e[..., obs.EGO['camera_age']] = 0.0
    e[..., obs.EGO['a_lat']] = np.linspace(2.0, 5.0, seqs)[:, None] / obs.A_LAT_SCALE
    return out


@torch.no_grad()
def golden(policy: PolicyNet, inputs: dict) -> dict:
    """PyTorch's own outputs over the sequences, GRU state carried, float32."""
    policy = policy.cpu().float().eval()
    seqs, steps = inputs['ego'].shape[:2]
    action = np.zeros((seqs, steps, 2), np.float32)
    path = np.zeros((seqs, steps, obs.AUX_PATH_POINTS, 2), np.float32)
    for s in range(seqs):
        h = torch.zeros(1, 128)
        for t in range(steps):
            o = {k: torch.as_tensor(v[s, t:t + 1]) for k, v in inputs.items()}
            a, h, p, _ = policy(o, h)
            v_cap = o['ego'][:, obs.EGO['v_cap']] * obs.V_SCALE
            action[s, t] = squash(a, v_cap)[0].numpy()
            path[s, t] = p[0].numpy()
    return dict(inputs, action=action, ego_path=path)


def _commit() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                              cwd=common.REPO, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


A_LAT_COL = obs.EGO['a_lat']          # spec v2's last ego value; v1 ends just before it


def fold_a_lat(weights: dict, a_lat: float) -> dict:
    """Spec-v2 weights -> spec-v1 weights for a car whose cornering cap is a_lat (m/s^2):
    ego.fc0's cornering-cap column moves into its bias at that value."""
    assert A_LAT_COL == obs.EGO_DIM - 1, 'the cornering cap must be the last ego value'
    w = dict(weights)
    W, b = weights['ego.fc0.weight'], weights['ego.fc0.bias']
    w['ego.fc0.weight'] = np.ascontiguousarray(W[:, :A_LAT_COL])
    w['ego.fc0.bias'] = (b.astype(np.float64) + W[:, A_LAT_COL].astype(np.float64) * (a_lat / obs.A_LAT_SCALE)
                         ).astype(np.float32)
    return w


def export(policy: PolicyNet, out_root: Path, *, run_name: str, snapshot_id: str,
           modalities=('scan', 'ego', 'imu', 'preview', 'race'), inputs=None,
           car_spec: int = obs.OBS_SPEC_VERSION, a_lat: float | None = None) -> Path:
    """car_spec 1 (with a_lat): a model for spec-v1 car code, cornering cap folded in."""
    if car_spec not in (1, obs.OBS_SPEC_VERSION):
        raise ValueError(f'car_spec {car_spec}: this code exports spec 1 or {obs.OBS_SPEC_VERSION}')
    if car_spec == 1 and not (a_lat and a_lat > 0):
        raise ValueError('car_spec 1 needs the car\'s cornering cap: a_lat, m/s^2')
    out_root = Path(out_root)
    tmp = out_root / '_export_tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    weights = weights_of(policy)
    inputs = inputs or synthetic_inputs()
    if car_spec == 1:
        weights = fold_a_lat(weights, a_lat)
        inputs = dict(inputs, ego=inputs['ego'].copy())
        inputs['ego'][..., A_LAT_COL] = a_lat / obs.A_LAT_SCALE     # what the v2 network is told
    np.savez(tmp / 'weights.npz', **weights)
    (c0, c1, c2, c3), fc = policy.width
    card = {'obs_spec_version': car_spec, 'scan_channels': [c0, c1, c2, c3], 'scan_fc': fc,
            'recurrent': policy.recurrent, 'modalities': list(modalities),
            'parameters': int(sum(v.size for v in weights.values())),
            'run_name': run_name, 'snapshot_id': snapshot_id, 'training_commit': _commit(),
            'exported': datetime.datetime.now().isoformat(timespec='seconds')}
    if car_spec == 1:
        card['a_lat_folded'] = float(a_lat)
    (tmp / 'model_card.yaml').write_text(yaml.safe_dump(card, sort_keys=False))
    g = golden(policy, inputs)
    if car_spec == 1:
        g['ego'] = np.ascontiguousarray(g['ego'][..., :A_LAT_COL])   # the v1 car's inputs
    np.savez(tmp / 'golden.npz', **g)
    h = model_store.weights_hash(tmp / 'weights.npz')
    final = out_root / h
    if final.exists():
        import shutil
        shutil.rmtree(final)
    os.replace(tmp, final)
    return final


def load_policy(checkpoint: Path) -> tuple:
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = state.get('policy_config', {})
    width = cfg.get('width', ((32, 64, 64, 64), 256))
    # stage checkpoints carry policy_config; league archive entries carry meta
    recurrent = cfg.get('recurrent', state.get('meta', {}).get('recurrent', True))
    policy = PolicyNet(width=(tuple(int(c) for c in width[0]), int(width[1])), recurrent=recurrent)
    policy.load_state_dict(state['policy'])
    return policy, state


def load_export(model_dir: Path) -> tuple:
    """(PolicyNet, card) from an exported spec-v2 model directory: the network again,
    from weights.npz, for re-exporting it (for example to a spec-v1 car)."""
    model_dir = Path(model_dir)
    card = yaml.safe_load((model_dir / 'model_card.yaml').read_text()) or {}
    if card.get('obs_spec_version') != obs.OBS_SPEC_VERSION:
        raise ValueError(f"{model_dir}: obs_spec_version {card.get('obs_spec_version')}; "
                         f'only spec {obs.OBS_SPEC_VERSION} exports can be re-exported')
    width = (tuple(int(c) for c in card['scan_channels']), int(card['scan_fc']))
    policy = PolicyNet(width=width, recurrent=bool(card.get('recurrent', True)))
    with np.load(model_dir / 'weights.npz') as z:
        sd = {k: torch.from_numpy(z[k].copy()) for k in z.files}
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if unexpected or set(missing) - set(EXCLUDE):
        raise ValueError(f'{model_dir}: weights do not match PolicyNet (missing {missing}, extra {unexpected})')
    return policy, card


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('checkpoint', type=Path, nargs='?')
    ap.add_argument('out_dir', type=Path)
    ap.add_argument('--from-export', type=Path, help='re-export this exported model directory instead')
    ap.add_argument('--car-spec', type=int, default=obs.OBS_SPEC_VERSION,
                    help='observation spec of the car code it is for (1: main, 2026-11-11)')
    ap.add_argument('--a-lat', type=float, help='with --car-spec 1: the car\'s max_lateral_accel, m/s^2')
    ap.add_argument('--run-name', help='the card\'s run_name, when the source does not carry it')
    args = ap.parse_args(argv)
    if (args.checkpoint is None) == (args.from_export is None):
        ap.error('give a checkpoint or --from-export, not both')
    if args.from_export:
        policy, card = load_export(args.from_export)
        run, snap = card.get('run_name', '?'), str(card.get('snapshot_id', '?'))
    else:
        policy, state = load_policy(args.checkpoint)
        run, snap = state.get('run_name', '?'), str(state.get('snapshot_id', args.checkpoint.stem))
    run = args.run_name or run
    path = export(policy, args.out_dir, run_name=run, snapshot_id=snap, car_spec=args.car_spec, a_lat=args.a_lat)
    print(f'exported {path.name} -> {path} (car spec {args.car_spec})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
