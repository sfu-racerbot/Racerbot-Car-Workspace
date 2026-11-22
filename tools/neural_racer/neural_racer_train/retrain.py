"""The calibrated retraining setup (retrain_calibrated.yaml) and its sign-off gate.

    python -m neural_racer_train.retrain --config ../retrain_calibrated.yaml

Without --run-dir this is a dry run: it checks the setup, builds the batched
simulator on the selected car's calibration with the setup's randomization and
club courses, steps it a few ticks with neutral commands and prints what a
training run would use. It computes no gradient and writes nothing.

Training stages take the same file with --retrain-config and refuse it until
signed_off_by names who approved the run.
"""
import argparse
import sys
from pathlib import Path

import yaml

KEYS = {'car', 'randomization', 'club_tracks', 'scripted_share', 'signed_off_by'}


class SetupError(ValueError):
    pass


def load(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    check(cfg)
    return cfg


def check(cfg: dict) -> None:
    from neural_racer_train.fidelity import calibration_module
    from neural_racer_train.gpu_sim.params import RANDOMIZATION_PROFILES
    from neural_racer_train.learn import CLUB_MAP_DIR, heldout_real
    unknown = set(cfg) - KEYS
    if unknown:
        raise SetupError(f'unknown keys {sorted(unknown)}')
    known = calibration_module().known_car_ids(calibration_module().cars_dir())
    if cfg.get('car') not in known:
        raise SetupError(f"car must be one of {sorted(known)}, got {cfg.get('car')!r}")
    if cfg.get('randomization') not in RANDOMIZATION_PROFILES:
        raise SetupError(f"randomization must be one of {sorted(RANDOMIZATION_PROFILES)}")
    tracks = list(cfg.get('club_tracks') or [])
    held = sorted(set(tracks) & heldout_real())
    if held:
        raise SetupError(f'held-out courses in club_tracks: {held}')
    missing = [t for t in tracks if not (CLUB_MAP_DIR / f'{t}_map.yaml').exists()]
    if missing:
        raise SetupError(f'no map for {missing} in {CLUB_MAP_DIR}')
    sc = cfg.get('scripted_share') or {}
    base, floor = float(sc.get('base', 0.15)), float(sc.get('floor', 0.03))
    if not 0.0 <= floor <= base <= 0.55:
        raise SetupError('scripted_share needs 0 <= floor <= base <= 0.55 (the share pfsp gives up)')


def signed_off(cfg: dict) -> bool:
    return bool(str(cfg.get('signed_off_by') or '').strip())


def _cfg(args):
    path = getattr(args, 'retrain_config', None)
    if path is None:
        return None
    cfg = load(path)
    if not signed_off(cfg):
        raise SystemExit(f'{path}: not signed off (signed_off_by is empty); refusing to train')
    return cfg


def runner_kwargs(args) -> dict:
    """Runner keywords for a training stage: {} without --retrain-config."""
    cfg = _cfg(args)
    if cfg is None:
        return {}
    return {'car_id': cfg['car'], 'randomization': cfg['randomization'],
            'club_tracks': tuple(cfg.get('club_tracks') or ())}


def scripted_share(args) -> tuple:
    cfg = _cfg(args)
    sc = (cfg or {}).get('scripted_share') or {}
    return float(sc.get('base', 0.15)), float(sc.get('floor', 0.03))


def dry_run(cfg: dict, num_envs: int = 4, ticks: int = 20) -> dict:
    import torch
    from neural_racer_train.gpu_sim.params import RANDOMIZATION_PROFILES
    from neural_racer_train.learn import Runner
    runner = Runner(num_envs, 2, 'cpu', 0, pool_slots=len(cfg.get('club_tracks') or ()) + 2,
                    initial_tracks=1, car_id=cfg['car'], randomization=cfg['randomization'],
                    club_tracks=tuple(cfg.get('club_tracks') or ()))
    sim = runner.sim
    for _ in range(ticks):
        sim.step(torch.zeros(num_envs, 2, 2))
    spreads, gain = RANDOMIZATION_PROFILES[cfg['randomization']]
    return {'car': cfg['car'], 'mu': round(float(sim.car.mu), 3),
            'steer_delay_ticks': int(sim.car.steer_delay_ticks),
            'speed_delay_ticks': int(sim.car.speed_delay_ticks),
            'randomization': cfg['randomization'], 'steer_gain_error': gain,
            'spreads': {k: v[1] for k, v in spreads.items()},
            'club_tracks': list(cfg.get('club_tracks') or ()),
            'pinned_slots': len(runner.pinned), 'scripted_share': cfg.get('scripted_share'),
            'signed_off': signed_off(cfg), 'ticks_stepped': ticks}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load(args.config)
    print(yaml.safe_dump(dry_run(cfg), sort_keys=False))
    if not signed_off(cfg):
        print('not signed off: training stages will refuse this setup')
    return 0


if __name__ == '__main__':
    sys.exit(main())
