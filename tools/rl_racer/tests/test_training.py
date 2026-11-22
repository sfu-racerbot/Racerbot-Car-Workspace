import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from rl_racer.evaluate import LapMetrics, evaluate, map_fidelity_state
from rl_racer.export import export
from rl_racer.policy_numpy import PolicyNumpy
from rl_racer.ppo import ActorCritic, advantages
from rl_racer.train import DEFAULT_CONFIG, SCRATCH, load_config, make_env, train, validate_signoff


@pytest.fixture(scope='module')
def smoke_run():
    # One thread avoids BLAS oversubscription for these deliberately tiny batches.
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    directory = train(DEFAULT_CONFIG, smoke=True, stop_after_updates=2)
    yield directory
    torch.set_num_threads(before)


@pytest.fixture
def untrained_export(tmp_path):
    config = load_config(DEFAULT_CONFIG, smoke=True)
    env = make_env(config)
    torch.manual_seed(10)
    model = ActorCritic(len(env.scales), config)
    checkpoint = tmp_path / 'untrained.pt'
    torch.save({'model': model.state_dict(), 'config': config,
                'config_hash': hashlib.sha256(DEFAULT_CONFIG.read_bytes()).hexdigest(),
                'assets': {key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
                           for key in ('track', 'vehicle')}, 'smoke': True, 'steps': 0}, checkpoint)
    return export(checkpoint, tmp_path / 'untrained.npz'), model, env


def test_signoff_gate(tmp_path):
    config = tmp_path / 'train.yaml'
    config.write_bytes(DEFAULT_CONFIG.read_bytes())
    expected = hashlib.sha256(config.read_bytes()).hexdigest()
    assert validate_signoff(config, 200_000) == expected
    with pytest.raises(ValueError, match='require --signoff'):
        validate_signoff(config, 200_001)
    with pytest.raises(ValueError, match='invalid signoff'):
        validate_signoff(config, 120_000_000, tmp_path / 'absent.yaml')
    approval = tmp_path / 'signoff.yaml'
    approval.write_text(yaml.safe_dump({'approved_by': 'test reviewer (fixture only)',
                                      'date': '2026-11-21', 'sha256': 'wrong'}))
    with pytest.raises(ValueError, match='sha256'):
        validate_signoff(config, 120_000_000, approval)
    approval.write_text(yaml.safe_dump({'approved_by': 'test reviewer (fixture only)',
                                      'date': '2026-11-21', 'sha256': expected}))
    assert validate_signoff(config, 120_000_000, approval) == expected
    config.write_bytes(config.read_bytes() + b'\n# changed\n')
    with pytest.raises(ValueError, match='sha256'):
        validate_signoff(config, 120_000_000, approval)


def test_gate_runs_before_output_creation(tmp_path):
    output = tmp_path / 'must-not-exist'
    config = load_config(DEFAULT_CONFIG)
    config.update(num_envs=2, rollout_steps=2, total_steps=200_001)
    path = tmp_path / 'bounded-gate-test.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='require --signoff'):
        # A regressed gate can execute at most four steps, never a full run.
        train(path, run_dir=output, stop_after_updates=1)
    assert not output.exists()


def test_gae_bootstrap_and_reset():
    # gamma=lambda=1: truncation bootstraps 10 but must not carry next episode's 100.
    rewards = np.array([[1.], [100.], [3.]])
    zeros = np.zeros_like(rewards)
    terminal = np.array([[False], [True], [False]])
    trunc = np.array([[True], [False], [False]])
    adv, returns = advantages(rewards, zeros, np.full_like(zeros, 10.), terminal, trunc, 1., 1.)
    np.testing.assert_array_equal(adv[:, 0], [11., 100., 13.])
    np.testing.assert_array_equal(returns, adv)


def test_smoke_checkpoint_resume(smoke_run):
    checkpoint = smoke_run / 'latest.pt'
    before = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert before['steps'] == 128
    assert smoke_run.is_relative_to(SCRATCH)
    directory = train(DEFAULT_CONFIG, smoke=True, resume=checkpoint)
    after = torch.load(directory / 'latest.pt', map_location='cpu', weights_only=False)
    assert after['steps'] == 256
    assert after['iteration'] == 4
    assert any(not torch.equal(before['model'][key], after['model'][key]) for key in before['model'])
    with (directory / 'metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row['steps']) for row in rows] == [64, 128, 192, 256]
    assert all(np.isfinite(float(row['loss'])) for row in rows)
    assert all(np.isfinite(float(row['value_loss'])) for row in rows)
    assert json.loads((directory / 'run.json').read_text())['smoke'] is True
    with pytest.raises(ValueError, match='scratch'):
        train(DEFAULT_CONFIG, smoke=True, run_dir=Path('/approved_models/no'))


def test_export_parity_and_integrator(untrained_export):
    path, model, env = untrained_export
    policy = PolicyNumpy(path, 3)
    # Independent randomized inputs plus real observations exercise all layer signs.
    obs = np.random.default_rng(34).normal(0, 1, (19, len(env.scales))).astype(np.float32)
    obs = np.vstack((obs, env.observe()))
    with torch.no_grad():
        expected = model.actor(torch.tensor(obs, dtype=torch.float32)).tanh().numpy()
    np.testing.assert_allclose(policy.act(obs), expected, atol=1e-6, rtol=0)
    np.testing.assert_allclose(policy.act(obs * env.scales, normalized=False), expected, atol=1e-6, rtol=0)
    policy.reset(speed=.2)
    commands = policy.commands([[1, 1], [-1, -1], [0, .5]])
    np.testing.assert_allclose(commands[:, 0], [env.model.params.steer_max, env.model.params.steer_min, 0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(commands[:, 1], [.45, 0, .325], atol=1e-12, rtol=0)
    assert policy.metadata['config_hash'] == hashlib.sha256(DEFAULT_CONFIG.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='observations'):
        policy.act([[np.nan] * len(env.scales)])
    with pytest.raises(ValueError, match='actions'):
        policy.commands([])


def test_evaluate_untrained_without_torch(untrained_export, tmp_path):
    path, _, _ = untrained_export
    result = evaluate(path, max_steps=5)
    assert result['backend'] == 'RacingEnv'
    assert abs(result['elapsed_s'] - 5 * .05) < 1e-12  # seconds
    assert result['laps_completed'] == len(result['per_lap']) == 0
    assert result['lap_time_s'] is None
    assert np.isfinite(result['offtrack_integral_m_s'])
    json.dumps(result, allow_nan=False)
    # Block torch imports, then run the real CLI in a fresh interpreter.
    script = (
        "import sys; sys.modules['torch'] = None; "
        "from rl_racer.evaluate import main; "
        f"main([{str(path)!r}, '--max-steps', '2', '--output', {str(tmp_path / 'eval.json')!r}])"
    )
    import os
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    subprocess.run([sys.executable, '-c', script], env=env, check=True, capture_output=True, text=True)
    assert json.loads((tmp_path / 'eval.json').read_text())['elapsed_s'] == .1


def test_lap_metric_closed_form():
    metrics = LapMetrics(1.)
    # Constant 2 m/s covers two 1 m laps in one second.
    for _ in range(20):
        metrics.add(.1, .6, [.5, .5], .05)
    report = metrics.report()
    assert report['laps_completed'] == 2
    np.testing.assert_allclose([lap['lap_time_s'] for lap in report['per_lap']], [.5, .5], atol=1e-12, rtol=0)
    assert abs(report['offtrack_integral_m_s'] - .1) < 1e-12  # m*s
    metrics.add(.1, 0, [.5, .5], .05, reset=True)
    assert metrics.report()['resets'] == 1


def test_fidelity_state_mapping():
    mapped = map_fidelity_state({'agent_0': {'std_state': [1, 2, .1, 3, .2, .4, .3]}})
    expected = [[1, 2, .2, 3 * np.cos(.3), 3 * np.sin(.3), .4, .1, 3 * np.cos(.3)]]
    np.testing.assert_allclose(mapped, expected, atol=1e-12, rtol=0)


def test_fidelity_crosscheck(untrained_export):
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / '.sim/f1tenth_gym'))
    pytest.importorskip('f1tenth_gym')
    result = evaluate(untrained_export[0], max_steps=3, fidelity=True)
    assert result['backend'] == 'FidelityPlant'
    assert abs(result['elapsed_s'] - .15) < 1e-12  # seconds
    assert 'wheel speed estimated' in result['state_mapping']
    assert np.isfinite(result['offtrack_integral_m_s'])
    json.dumps(result, allow_nan=False)
