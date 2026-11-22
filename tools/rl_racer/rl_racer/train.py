"""Paper-style PPO training. No long run may bypass validate_signoff()."""
import argparse
import copy
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
import yaml

from .env import RacingEnv
from .ppo import ActorCritic, advantages, update
from .track import Track


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'config/train_paper.yaml'
SMOKE_BUDGET = 200_000
SCRATCH = Path.home() / '.cache/rl_racer/smoke'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_signoff(config_path, total_steps, signoff=None):
    digest = sha256(config_path)
    if signoff is None:
        if total_steps > SMOKE_BUDGET:
            raise ValueError('REFUSED: runs above 200000 steps require --signoff with reviewer approval')
        return digest
    try:
        approval = yaml.safe_load(Path(signoff).read_text(encoding='utf-8'))
        if not isinstance(approval, dict) or not isinstance(approval.get('approved_by'), str) or not approval['approved_by'].strip():
            raise ValueError('approved_by must name the reviewer')
        date.fromisoformat(str(approval['date']))
        if approval.get('sha256') != digest:
            raise ValueError('sha256 does not match the exact train config')
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f'REFUSED: invalid signoff: {exc}') from exc
    return digest


def load_config(path, smoke=False):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    for key in ('num_envs', 'rollout_steps', 'minibatch_size', 'total_steps', 'epochs', 'checkpoint_every'):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if smoke:
        config.update(num_envs=2, rollout_steps=32, minibatch_size=64, total_steps=256, epochs=2)
    for key in ('track', 'vehicle'):
        config[key] = str((path.parent / config[key]).resolve())
    return config


def make_env(config):
    return RacingEnv(Track.from_csv(config['track'], config['track_width']),
                     config['num_envs'], seed=config['seed'], config=config['vehicle'], dt=config['dt'])


def env_state(env):
    return {key: copy.deepcopy(getattr(env, key)) for key in
            ('state', 'steer_cmd', 'wheel_cmd', 'accel_prev', 'steps', 'parameters')}


def train(config_path=DEFAULT_CONFIG, *, smoke=False, signoff=None, run_dir=None,
          resume=None, stop_after_updates=None):
    config = load_config(config_path, smoke)
    digest = validate_signoff(config_path, config['total_steps'], signoff)
    is_smoke = config['total_steps'] <= SMOKE_BUDGET
    if is_smoke:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        if run_dir is not None:
            raise ValueError('smoke runs choose their own scratch directory; omit --run-dir')
        directory = Path(resume).resolve().parent if resume else Path(tempfile.mkdtemp(prefix='run-', dir=SCRATCH))
        if not directory.is_relative_to(SCRATCH.resolve()):
            raise ValueError('smoke resume must stay in the scratch directory')
    else:
        if resume:
            directory = Path(resume).resolve().parent
            if run_dir and Path(run_dir).resolve() != directory:
                raise ValueError('resume and run-dir disagree')
        elif run_dir:
            directory = Path(run_dir).resolve()
        else:
            raise ValueError('approved runs require --run-dir')
    directory.mkdir(parents=True, exist_ok=True)
    if not resume and (directory / 'latest.pt').exists():
        raise ValueError('checkpoint already exists: use --resume')
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = 'cuda' if config['device'] == 'auto' and torch.cuda.is_available() else config['device']
    if device == 'auto':
        device = 'cpu'
    env = make_env(config)
    model = ActorCritic(len(env.scales), config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_start'])
    assets = {key: sha256(config[key]) for key in ('track', 'vehicle')}
    completed = iteration = 0
    if resume:
        # Only load checkpoints you trust: optimizer/RNG state uses Python pickle.
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['config_hash'] != digest or saved['config'] != config or saved['assets'] != assets:
            raise ValueError('resume config or track/vehicle hash mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        for key, value in saved['env'].items():
            setattr(env, key, value)
        env.rng.bit_generator.state = saved['env_rng']
        torch.set_rng_state(saved['torch_rng'])
        if saved['cuda_rng'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng'])
        completed, iteration = saved['steps'], saved['iteration']
    metadata = {'config_hash': digest, 'config': config, 'assets': assets, 'smoke': is_smoke,
                'signoff': None if signoff is None else str(Path(signoff).resolve())}
    (directory / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    last = {}

    def checkpoint():
        saved = dict(metadata, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     steps=completed, iteration=iteration, env=env_state(env),
                     env_rng=env.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                     numpy_rng=np.random.get_state())
        target = directory / 'latest.pt'
        torch.save(saved, directory / 'latest.pending')
        (directory / 'latest.pending').replace(target)

    while completed < config['total_steps']:
        remaining = config['total_steps'] - completed
        # Batch size N means the last update can undershoot the budget by N-1.
        horizon = min(config['rollout_steps'], remaining // env.num_envs)
        if horizon == 0:
            break
        rollout_size = config['rollout_steps'] * env.num_envs
        last_start = ((config['total_steps'] - 1) // rollout_size) * rollout_size
        lr = config['lr_start'] + (config['lr_end'] - config['lr_start']) * completed / max(1, last_start)
        for group in optimizer.param_groups:
            group['lr'] = lr
        records = []
        for _ in range(horizon):
            observation = env.observe()
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, latent, logp, value = model.sample(tensor)
            _, reward, done, truncated, info = env.step(action.cpu().numpy())
            with torch.no_grad():
                next_value = model.value(torch.as_tensor(info['final_observation'], dtype=torch.float32, device=device))
            records.append((observation, latent.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy(),
                            reward, done, truncated, next_value.cpu().numpy()))
        obs, latent, logp, values, rewards, done, trunc, next_values = map(np.stack, zip(*records))
        adv, returns = advantages(rewards, values, next_values, done, trunc, config['gamma'], config['gae_lambda'])
        batch = [torch.as_tensor(v.reshape((-1,) + v.shape[2:]), dtype=torch.float32, device=device)
                 for v in (obs, latent, logp, adv, returns)]
        last = update(model, optimizer, batch, config)
        completed += horizon * env.num_envs
        iteration += 1
        last.update(steps=completed, iteration=iteration, lr=lr, reward=float(rewards.mean()), offtracks=int(done.sum()))
        metrics_path = directory / 'metrics.csv'
        exists = metrics_path.exists()
        with metrics_path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(last))
            if not exists:
                writer.writeheader()
            writer.writerow(last)
        with (directory / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(last, allow_nan=False) + '\n')
        if iteration % config['checkpoint_every'] == 0:
            checkpoint()
        if stop_after_updates is not None and iteration >= stop_after_updates:
            break
    checkpoint()
    print(json.dumps({'run_dir': str(directory), 'steps': completed, 'metrics': last}, allow_nan=False))
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--smoke', action='store_true', help='2 environments, 256 steps, scratch output only')
    parser.add_argument('--signoff', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args(argv)
    try:
        train(args.config, smoke=args.smoke, signoff=args.signoff, run_dir=args.run_dir, resume=args.resume)
    except ValueError as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
