"""eval_worker: score archive snapshots in the referee, on the job's spare CPU cores.

    python -m neural_racer_train.eval_worker --archive DIR --workers N [--once]

For every snapshot without a result, against each panel member, it races
2-car matches on held-out tracks (grid order alternating) in batches of 10
until the sequential test settles the matchup or 200 races pass (plan A8,
Phase 9). A panel member a role has beaten above 95% for 10 snapshots in a
row is re-checked only every 10th snapshot; one failed re-check restores it.

Writes <archive>/eval/<role>/<update>.json and appends cross-play lines
({"a", "b", "wins_a", "races", "fault_a", "fault_b"}) to
<archive>/eval/crossplay.jsonl for selection.py.

Several workers can share an archive -- the GPU jobs' spare cores and a CPU
job beside them (2026-11-06): a worker claims a snapshot with an exclusive
<entry>.claim file before racing it, and skips one another worker claimed
less than CLAIM_TTL ago. --newest-first races the latest snapshots first.
"""
import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path

from neural_racer_train import common  # noqa: F401  (paths)
from neural_racer_train import heldout, selection
from neural_racer_train.archive import Archive

PANEL = ('gap_follow', 'pure_pursuit')
# The release crossplay races at the car's top speed with the cornering cap out of the way (2026-11-08, the
# team: speeds as high as we can, the racers find what works), into <archive>/eval-max/. The 2.5 m/s
# crossplay before that is in <archive>/eval/ (--tag '' --v-cap 2.5 --a-lat 2.0 reproduces it)
# A win is on finishing order since 2026-11-08 (referee.finish_outcome); <archive>/eval-max-distance/ holds
# the crossplay before that, scored on distance covered
RELEASE_V_CAP, RELEASE_A_LAT, RELEASE_TAG = 5.05, 5.0, 'max'
CLAIM_TTL = 4 * 3600.0      # s: a claim older than this is a worker that died mid-snapshot


def claim(path: Path) -> bool:
    """Take the snapshot for this worker: True if it created the claim file (or replaced a stale one)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.exists() and time.time() - path.stat().st_mtime > CLAIM_TTL:
            path.unlink()
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    except FileNotFoundError:       # another worker removed a stale claim between the two calls
        return claim(path)
    with os.fdopen(fd, 'w') as f:
        f.write(f'{os.uname().nodename} {os.getpid()} {time.time():.0f}\n')
    return True


def _race_job(job):
    """One referee race, in a worker process: (snapshot, recurrent, width, panel, track, seed, neural first?, v_cap).
    Any failure comes back as {'error'}: an exception escaping a pool worker,
    or the gym's bootstrap exiting it, left Pool.map waiting forever."""
    try:
        return _race(*job)
    except BaseException as exc:  # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}'}


def _race(entry_path, recurrent, width, panel, track, seed, neural_first, v_cap, a_lat=None):
    import torch
    from neural_racer_train import referee
    from neural_racer_train.export import weights_of
    from neural_racer_train.model import FULL, PolicyNet
    width = width or FULL
    policy = PolicyNet((tuple(width[0]), int(width[1])), recurrent=recurrent)
    policy.load_state_dict(torch.load(entry_path, map_location='cpu', weights_only=False)['policy'])
    w = weights_of(policy)
    kw = {'v_cap': v_cap, **({'a_lat': a_lat} if a_lat else {})}
    drivers = {'neural': referee.neural_factory(w, {'recurrent': bool(recurrent)}, **kw),
               'panel': referee.gap_factory() if panel == 'gap_follow' else referee.pursuit_factory()}
    if not neural_first:
        drivers = {'panel': drivers['panel'], 'neural': drivers['neural']}
    try:
        r = referee.race(track, drivers, seed=seed, laps=1, timeout_s=150.0)
    except RuntimeError as exc:
        return {'error': str(exc)}
    i_n, i_p = r.names.index('neural'), r.names.index('panel')
    # finishing order (referee.finish_outcome, 2026-11-08): a crash never wins, two non-finishers is no win
    return {'won': referee.finish_outcome(r.names, r.finish_time, 'neural', 'panel') == 'win',
            'fault_n': int(r.at_fault[i_n]), 'fault_p': int(r.at_fault[i_p]), 'track': track, 'seed': seed}


def evaluate(entry, pool, tracks, retire: dict, snapshot_index: int, v_cap: float, a_lat=None) -> dict:
    out = {'entry': entry.entry_id, 'matchups': {}}
    for panel in PANEL:
        key = f'{entry.role}|{panel}'
        ret = selection.Retirement(**retire.get(key, {}))
        if not ret.should_check(snapshot_index):
            out['matchups'][panel] = {'skipped': 'retired'}
            continue
        wins = races = fn = fp = 0
        k = 0
        verdict, errors = 'continue', []
        while verdict == 'continue':
            jobs = []
            for _ in range(selection.BATCH):
                jobs.append((entry.path, entry.meta.get('recurrent', True), entry.meta.get('width'), panel,
                             tracks[k % len(tracks)], k, k % 2 == 0, v_cap, a_lat))
                k += 1
            ok = 0
            for res in pool.map(_race_job, jobs):
                if 'error' in res:
                    errors.append(res['error'])
                    continue
                ok += 1
                wins += int(res['won'])
                races += 1
                fn += res['fault_n']
                fp += res['fault_p']
            verdict = selection.sequential_verdict(wins, races)
            if ok == 0:
                # a whole batch failed (no gym on this node, a broken snapshot): never loop on it
                verdict = 'error'
                print(f'[eval_worker] {entry.entry_id} vs {panel}: every race failed: {errors[-1]}', flush=True)
        if verdict != 'error':
            ret.record(snapshot_index, wins / max(races, 1))
            retire[key] = vars(ret)
        out['matchups'][panel] = {'wins': wins, 'races': races, 'verdict': verdict,
                                  'fault_neural': fn, 'fault_panel': fp, 'errors': len(errors)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--v-cap', type=float, default=RELEASE_V_CAP)
    ap.add_argument('--once', action='store_true', help='evaluate what is there, then exit')
    ap.add_argument('--poll-sec', type=float, default=60.0)
    ap.add_argument('--newest-first', action='store_true', help='race the latest snapshots first')
    ap.add_argument('--a-lat', type=float, default=RELEASE_A_LAT, help="the neural driver's cornering cap, m/s^2")
    ap.add_argument('--tag', default=RELEASE_TAG, help="results in <archive>/eval-<tag>/ ('' = eval/): one folder per "
                    'set of caps')
    ap.add_argument('--only', default='', help='comma-separated entry ids to race (default: all)')
    args = ap.parse_args(argv)
    archive = Archive(args.archive)
    eval_dir = args.archive / (f'eval-{args.tag}' if args.tag else 'eval')
    eval_dir.mkdir(parents=True, exist_ok=True)
    retire_path = eval_dir / 'retirement.json'
    retire = json.loads(retire_path.read_text()) if retire_path.exists() else {}
    tracks = heldout.names()
    heldout.ensure_maps()
    stop = {'now': False}

    def _term(*_):
        # at once: a snapshot's matchups can take an hour, and the job script
        # waits on this process before it reports (the GPU sat idle meanwhile)
        stop['now'] = True
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _term)
    with mp.get_context('spawn').Pool(max(args.workers, 1)) as pool:
        while not stop['now']:
            todo = [e for e in archive.entries() if not (eval_dir / f'{e.entry_id}.json').exists()]
            if args.only:
                todo = [e for e in todo if e.entry_id in args.only.split(',')]
            if args.newest_first:
                todo.sort(key=lambda e: -e.update)
            per_role = {}
            for e in todo:
                if stop['now']:
                    break
                lock = eval_dir / f'{e.entry_id}.claim'
                if (eval_dir / f'{e.entry_id}.json').exists() or not claim(lock):
                    continue        # done or being raced by another worker
                idx = per_role.setdefault(e.role, len([x for x in archive.entries() if x.role == e.role
                                                       and (eval_dir / f'{x.entry_id}.json').exists()]))
                try:
                    res = evaluate(e, pool, tracks, retire, idx, args.v_cap, args.a_lat)
                except BaseException:
                    lock.unlink(missing_ok=True)
                    raise
                if any(m.get('verdict') == 'error' for m in res['matchups'].values()):
                    lock.unlink(missing_ok=True)
                    continue        # no result: it is raced again on the next pass
                per_role[e.role] += 1
                path = eval_dir / f'{e.entry_id}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix('.json.tmp')
                tmp.write_text(json.dumps(res, indent=2))
                tmp.replace(path)
                with open(eval_dir / 'crossplay.jsonl', 'a') as f:
                    for panel, m in res['matchups'].items():
                        if 'races' in m:
                            f.write(json.dumps({'a': e.entry_id, 'b': panel, 'wins_a': m['wins'], 'races': m['races'],
                                                'fault_a': m['fault_neural'], 'fault_b': m['fault_panel']}) + '\n')
                retire_path.write_text(json.dumps(retire))
                lock.unlink(missing_ok=True)
            if args.once:
                break
            time.sleep(args.poll_sec)
    return 0


if __name__ == '__main__':
    sys.exit(main())
