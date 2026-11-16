"""A library of pre-generated training tracks on disk (plan Phase 4), so a
run fills its GPU pool at start-up instead of waiting on generator workers:
the teacher-v2 smoke run reached only 304 tracks in 20 minutes and spent
66% of its replays on 10 of them.

    python -m neural_racer_train.tracklib --out DIR --count 8192 --workers 32

One compressed .npz per track, named by seed: the spec, the fp16 signed
distance field the GPU pool stores, the occupancy, origin and digest.
Building is resumable (existing files are kept) and the library records the
generator config it was built with, so a run cannot mix in tracks made under
another one. Seeds come from below learn.HELDOUT_SEED_BASE only.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from neural_racer_train import tracks

META = 'library.json'


def save(raster: tracks.RasterTrack, path: Path) -> None:
    s = raster.spec
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(
        tmp, kind=s.kind, seed=s.seed, centre=s.centre, w_left=s.w_left, w_right=s.w_right,
        obstacles=s.obstacles, sdf=raster.sdf.astype(np.float16), occupancy=np.packbits(raster.occupancy),
        shape=np.array(raster.occupancy.shape), origin=np.array(raster.origin), resolution=raster.resolution,
        lap_length=raster.lap_length, digest=raster.digest)
    os.replace(tmp, path)


def load(path) -> tracks.RasterTrack:
    z = np.load(path)
    shape = tuple(int(v) for v in z['shape'])
    spec = tracks.TrackSpec(str(z['kind']), int(z['seed']), z['centre'], z['w_left'], z['w_right'], z['obstacles'])
    occ = np.unpackbits(z['occupancy'])[:shape[0] * shape[1]].reshape(shape).astype(bool)
    return tracks.RasterTrack(spec, occ, z['sdf'].astype(np.float32), tuple(float(v) for v in z['origin']),
                              float(z['resolution']), float(z['lap_length']), str(z['digest']))


class Library:
    """The .npz files of one directory, in a fixed shuffled order."""

    def __init__(self, directory, cfg: tracks.TrackGenConfig | None = None, seed: int = 0):
        """cfg: the config the library must have been built with; None takes
        whichever of learn.TRAIN_CONFIGS it was (self.cfg)."""
        from neural_racer_train import learn
        self.dir = Path(directory)
        meta = json.loads((self.dir / META).read_text())
        # a key added to TrackGenConfig after a library was built takes its default
        built = {**_cfg_dict(tracks.TrackGenConfig()), **meta['config']}
        if cfg is None:
            cfg = next((c for c in learn.TRAIN_CONFIGS.values() if _cfg_dict(c) == built), None)
            if cfg is None:
                raise ValueError(f'{self.dir} was built with {meta["config"]}, none of learn.TRAIN_CONFIGS')
        elif built != _cfg_dict(cfg):
            raise ValueError(f'{self.dir} was built with {meta["config"]}, not {_cfg_dict(cfg)}')
        self.cfg = cfg
        self.files = sorted(p for p in self.dir.glob('*.npz') if '.tmp' not in p.name)
        np.random.default_rng(seed).shuffle(self.files)
        self.by_seed = {int(p.stem): p for p in self.files}
        self.next = 0

    def __len__(self):
        return len(self.files)

    def take(self, n: int) -> list:
        """The next n paths, or fewer when the library runs out."""
        out = self.files[self.next:self.next + n]
        self.next += len(out)
        return out


def _cfg_dict(cfg: tracks.TrackGenConfig) -> dict:
    return json.loads(json.dumps(asdict(cfg)))


def _build_one(args):
    seed, out, cfg = args
    path = Path(out) / f'{seed}.npz'
    if path.exists():
        return seed, 'kept'
    from neural_racer_train import learn
    try:
        save(learn._generate(seed, cfg), path)
    except Exception as e:      # noqa: BLE001 -- an invalid seed is skipped, the count is a target
        return seed, f'failed: {e}'
    return seed, 'made'


REAL_SEED_BASE = 999_000_000        # library seeds of real-map variants: below learn.HELDOUT_SEED_BASE


def add_real(out, names: list, copies: int, refresh: bool = False) -> int:
    """Every variant (realmaps.variants) of each real map, `copies` times each
    (distinct seeds, so distinct pool slots): the track a car will race, in
    its training pool. Seeds count up from REAL_SEED_BASE, stepping over any a
    generated track already holds (build() draws below HELDOUT_SEED_BASE, so
    it can); a real track already there is from an earlier call and is kept,
    or with `refresh` written over (realmaps changed). Returns the real-map
    tracks in the library."""
    from neural_racer_train import realmaps
    out, k, seed = Path(out), 0, REAL_SEED_BASE
    for name in names:
        for v in realmaps.variants(name):
            for _ in range(copies):
                while True:
                    path = out / f'{seed}.npz'
                    seed += 1
                    if not path.exists():
                        save(v, path)
                        break
                    if load(path).spec.kind.startswith('real:'):
                        if refresh:
                            save(v, path)
                        break
                k += 1
    return k


def build(out, count: int, workers: int, seed: int = 0, config: str = 'v1') -> None:
    from neural_racer_train import learn
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    gen_cfg = learn.TRAIN_CONFIGS[config]
    cfg = _cfg_dict(gen_cfg)
    meta_path = out / META
    if meta_path.exists() and json.loads(meta_path.read_text())['config'] != cfg:
        sys.exit(f'{out} holds tracks from another generator config; use a new directory')
    meta_path.write_text(json.dumps({'config': cfg, 'seed': seed}, indent=1))
    seeds = np.random.default_rng(seed).integers(learn.HELDOUT_SEED_BASE, size=count).tolist()
    t0, made, failed = time.time(), 0, 0
    with mp.get_context('spawn').Pool(workers) as pool:
        jobs = [(s, str(out), gen_cfg) for s in seeds]
        for k, (s, status) in enumerate(pool.imap_unordered(_build_one, jobs, chunksize=2)):
            made += status == 'made'
            failed += status.startswith('failed')
            if (k + 1) % 200 == 0 or k + 1 == count:
                rate = made / max(time.time() - t0, 1e-9)
                print(f'[tracklib] {k + 1}/{count} done ({made} made, {failed} failed), {rate:.2f} tracks/s', flush=True)
    print(f'[tracklib] {len([p for p in out.glob("*.npz") if ".tmp" not in p.name])} tracks in {out}', flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--count', type=int, default=8192)
    ap.add_argument('--workers', type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--config', choices=('v1', 'v2', 'v3'), default='v1',
                    help='learn.TRAIN_CONFIGS: v2 adds tight corners the car must swing wide for, v3 hairpin loops')
    ap.add_argument('--real', default='', help='real maps (comma-separated) to add, every variant of each')
    ap.add_argument('--real-refresh', action='store_true', help='write the real-map tracks over those there')
    ap.add_argument('--real-copies', type=int, default=40,
                    help='copies of each real-map variant: 40 x 12 is ~5.5%% of an 8192-track library')
    a = ap.parse_args(argv)
    build(a.out, a.count, a.workers, a.seed, a.config)
    if a.real:
        n = add_real(a.out, [x for x in a.real.split(',') if x], a.real_copies, a.real_refresh)
        print(f'[tracklib] {n} real-map tracks ({a.real}) added to {a.out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
