"""Shared plumbing for every training stage: paths, devices, checkpoints, and
the trainer contract the FIR scripts rely on (tools/neural_racer/slurm/README.md):

    --run-dir DIR --archive DIR --role NAME --cpu-workers N --resume
    SIGUSR1            -> checkpoint, then exit 85
    exit 0             -> the run's whole budget is done
"""
import argparse
import hashlib
import io
import json
import os
import random
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for rel in ('src/neural_racer', 'src/gap_follow', 'src/drive_intent', 'src/pure_pursuit'):
    p = str(REPO / rel)
    if p not in sys.path:
        sys.path.insert(0, p)
# fidelity.NR_SIM_ROOT: the referee races another workspace's gap_follow and pure_pursuit, so theirs come
# first -- one gap_follow per process, the baseline's (the car's wrapper uses only gap_logic's braking
# geometry from it)
if os.environ.get('NR_SIM_ROOT'):
    for rel in ('src/pure_pursuit', 'src/gap_follow'):
        p = str(Path(os.environ['NR_SIM_ROOT']) / rel)
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

EXIT_DONE = 0
EXIT_CHECKPOINTED = 85


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--archive', type=Path, default=None)
    ap.add_argument('--role', default='solo')
    ap.add_argument('--cpu-workers', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--num-envs', type=int, default=16384)
    ap.add_argument('--log-every', type=int, default=10, help='updates between metric copies to the host')
    ap.add_argument('--checkpoint-every', type=int, default=50)
    return ap


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def configure_torch(device: str) -> None:
    if device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class StopFlag:
    """Set by SIGUSR1 (the time limit is near); checked between updates."""

    def __init__(self):
        self.requested = False
        signal.signal(signal.SIGUSR1, self._handle)

    def _handle(self, *_):
        self.requested = True


def save_checkpoint(run_dir: Path, state: dict, name: str = 'latest') -> Path:
    """Atomic: written to a temp file, then renamed over the old one."""
    ckpt_dir = Path(run_dir) / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f'{name}.pt'
    tmp = path.with_suffix('.pt.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(run_dir: Path, name: str = 'latest', map_location='cpu'):
    path = Path(run_dir) / 'checkpoints' / f'{name}.pt'
    if not path.exists():
        return None
    return torch.load(path, map_location=map_location, weights_only=False)


class MetricLog:
    """Append-only JSON lines; one per logged update."""

    def __init__(self, run_dir: Path):
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(run_dir) / 'metrics.jsonl'

    def write(self, **fields) -> None:
        fields.setdefault('time', time.time())
        with open(self.path, 'a') as f:
            f.write(json.dumps(fields, default=float) + '\n')


def _git(*cmd):
    try:
        return subprocess.run(['git', '-C', str(REPO), *cmd], capture_output=True, timeout=60, check=True).stdout
    except Exception:     # noqa: BLE001 -- no git, or not a checkout: the record says so
        return None


def write_run_record(args, max_file_bytes: int = 2_000_000) -> dict:
    """One line of run.jsonl per job segment (plan item 1.8): the arguments,
    the commit, and a hash of the uncommitted changes (tracked diff plus
    untracked files); the changed files themselves go to code-<job>.tar.gz,
    since FIR runs code that is not committed."""
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    job = os.environ.get('SLURM_JOB_ID', 'local')
    head, diff = _git('rev-parse', 'HEAD'), _git('diff', 'HEAD')
    changed = [f for f in (_git('ls-files', '-m', '-o', '--exclude-standard') or b'').decode().splitlines() if f]
    h = hashlib.sha256(diff or b'')
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for f in sorted(set(changed)):
            p = REPO / f
            if p.is_file() and p.stat().st_size <= max_file_bytes:
                data = p.read_bytes()
                h.update(f.encode() + b'\0' + data)
                info = tarfile.TarInfo(f)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    (run_dir / f'code-{job}.tar.gz').write_bytes(buf.getvalue())
    rec = {'time': time.time(), 'job': job, 'host': os.uname().nodename, 'argv': sys.argv,
           'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           'commit': head.decode().strip() if head else None, 'changes_sha256': h.hexdigest(),
           'changed_files': len(changed), 'torch': torch.__version__}
    with open(run_dir / 'run.jsonl', 'a') as f:
        f.write(json.dumps(rec, default=str) + '\n')
    return rec


def run_stage(train_fn, args) -> int:
    """Wrap a stage: resume, train until done or SIGUSR1, checkpoint, exit code."""
    configure_torch(args.device)
    write_run_record(args)
    stop = StopFlag()
    finished = train_fn(args, stop)
    return EXIT_DONE if finished else EXIT_CHECKPOINTED
