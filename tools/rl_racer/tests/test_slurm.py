import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
SLURM = ROOT / 'tools/rl_racer/slurm'


def test_fir_scripts_parse():
    subprocess.run(['bash', '-n', *(str(SLURM / name) for name in
                   ('fir_setup.sh', 'fir_env.sh', 'train_paper.sbatch'))], check=True)


def test_fir_requires_signoff():
    result = subprocess.run(['bash', str(SLURM / 'train_paper.sbatch')], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'signoff YAML is mandatory' in result.stderr


@pytest.mark.parametrize('resume', [False, True])
def test_fir_training_command(tmp_path, resume):
    venv = tmp_path / 'venv'
    (venv / 'bin').mkdir(parents=True)
    calls = tmp_path / 'python-args'
    python = venv / 'bin/python'
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CALLS"\n')
    python.chmod(0o755)
    (venv / 'bin/activate').write_text(f'export PATH="{venv}/bin:$PATH"\n')
    shell_init = tmp_path / 'bash-init'
    shell_init.write_text('module() { :; }\n')
    signoff = tmp_path / 'signoff.yaml'
    signoff.write_text('test fixture; python is mocked, no training occurs\n')
    runs = tmp_path / 'runs'
    directory = runs / 'test-run'
    directory.mkdir(parents=True)
    if resume:
        (directory / 'latest.pt').touch()
    env = dict(os.environ, BASH_ENV=str(shell_init), CALLS=str(calls),
               RL_RACER_REPO=str(ROOT), RL_RACER_VENV=str(venv),
               RL_RACER_RUNS_ROOT=str(runs), SLURM_JOB_ID='123',
               RL_RACER_CONFIG=str(ROOT / 'tools/rl_racer/config/train_paper.yaml'))
    subprocess.run(['bash', str(SLURM / 'train_paper.sbatch'), str(signoff), 'test-run'],
                   env=env, check=True, capture_output=True, text=True)
    args = calls.read_text().splitlines()
    assert args[:4] == ['-u', '-m', 'rl_racer.train', '--config']
    assert args[args.index('--signoff') + 1] == str(signoff)
    assert args[args.index('--run-dir') + 1] == str(directory)
    assert ('--resume' in args) == resume
    if resume:
        assert args[args.index('--resume') + 1] == str(directory / 'latest.pt')
