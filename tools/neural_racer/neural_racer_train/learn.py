"""The machinery every training stage shares: tracks and levels, the track
feeder, episode resets through PLR, and the auxiliary-head targets.

Seat 0 of every environment is the learner's car. Seats 1..C-1 are opponents
(scripted drivers, archived policies, or copies of the learner).
"""
import math
import multiprocessing as mp
import os
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

from neural_racer import observation as obs
from neural_racer_train import plr as plr_mod, tracklib, tracks
from neural_racer_train.gpu_sim.sim import (EPISODE_HANDOVER, EPISODE_MAP, EPISODE_REACTIVE,
                                            BatchedRaceSim, Level)
from neural_racer_train.track_pool import TrackPool

HELDOUT_FILE = Path(__file__).resolve().parents[1] / 'heldout.yaml'
HELDOUT_SEED_BASE = 1_000_000_000          # seeds at or above this are never trained on
EPISODE_MIX = ((EPISODE_REACTIVE, 0.3), (EPISODE_MAP, 0.3), (EPISODE_HANDOVER, 0.4))
V_CAP_RANGE = (1.0, 10.0)                  # up to gpu_sim.params.TRAIN_SPEED_MAX (the VESC's configured limit is 5.05)
# The wrapper's cornering cap, per race (decision D7): the car runs 2.0 m/s^2
# today and raises it only by measurement, so a share of races sits exactly
# there and the rest spread up to where the simulated car held its grip
A_LAT_RANGE = (2.0, 5.0)
# 2026-11-08, the team: "bump speeds as high as we can; the neural racers will figure out what works best".
# So most races sit near the top: A_LAT_CAR_SHARE (exactly 2.0 m/s^2, the car's cap until then) was 0.4,
# and a V_CAP_HIGH_SHARE of races draw their speed cap from V_CAP_HIGH up to the firmware cap
A_LAT_CAR_SHARE = 0.1
V_CAP_HIGH, V_CAP_HIGH_SHARE = 4.0, 0.7
# Overtaking (plan workstream 2): in the Nov 7 round robin and the Fir benches grid order decided the
# head-to-heads -- nobody passed. In this share of multi-car levels the learner (seat 0) starts at the
# back of the grid, behind opponents held to OVERTAKE_SCALE of its speed cap; RewardConfig.place pays
# each place gained
OVERTAKE_SHARE = 0.3
OVERTAKE_SCALE = (0.7, 0.9)
IMU_DROPOUT_EPISODES = 0.05
# Training episodes, bootstrapped at the limit: teacher.py's since teacher-v3. At
# 90 s, 40% of teacher v5's 3-lap races were still running (2026-10-30 audit)
MAX_SECONDS = 180.0
# Training tracks have no obstacles until after the league (team decision
# 2026-10-27): the teacher cannot see them, and they cost 12-25% of episodes.
TRAIN_TRACKS = tracks.TrackGenConfig(max_obstacles=0)
# The generator configs a training library may be built with (tracklib --config).
# v2 adds corners down to 0.45 m centreline radius where the width lets the car
# swing wide: every real map has them (car 1's course 0.71 m, levine_blocked
# 0.50 m, each with >= 1.55 m of radius plus outside width), v1 never did,
# and teacher v5 and its students stalled in every race on car 1's course
# (2026-10-31). A run takes its library's config.
# v3 adds hairpin_loop tracks (a loop round a thin divider, its U-turn exits pinched), car 1's
# course's shape; its library also holds car 1's course itself (tracklib --real, 2026-11-01)
_V2 = replace(TRAIN_TRACKS, min_radius=0.45, turn_room=1.4)
TRAIN_CONFIGS = {'v1': TRAIN_TRACKS, 'v2': _V2, 'v3': replace(_V2, hairpin_share=0.3)}


def heldout_digests() -> set:
    if not HELDOUT_FILE.exists():
        return set()
    data = yaml.safe_load(HELDOUT_FILE.read_text()) or {}
    return {t['digest'] for t in data.get('procedural', []) if t.get('digest')}


def _generate(seed: int, cfg: tracks.TrackGenConfig = TRAIN_TRACKS):
    """Worker entry: one training track, rasterized (seeds below the held-out base only)."""
    if seed >= HELDOUT_SEED_BASE:
        raise ValueError('held-out seed requested for training')
    return tracks.rasterize(tracks.generate(seed, cfg))


def _generate_or_none(seed: int, cfg: tracks.TrackGenConfig = TRAIN_TRACKS):
    try:
        return _generate(seed, cfg)
    except Exception:     # noqa: BLE001 -- its levels are dropped instead
        return None


def _load_or_none(path):
    try:
        return tracklib.load(path)
    except Exception:     # noqa: BLE001 -- a damaged file is skipped
        return None


def _call(fn, arg):
    return fn(arg)


class TrackFeeder:
    """Background CPU processes producing rasterized training tracks: from
    the library while it lasts (a file load), then freshly generated."""

    def __init__(self, workers: int, seed: int, library: 'tracklib.Library | None' = None):
        self.rng = np.random.default_rng(seed)
        self.workers = workers
        self.library = library
        self.cfg = library.cfg if library else TRAIN_TRACKS     # what fresh tracks are generated with
        self.pool = mp.get_context('spawn').Pool(workers) if workers > 0 else None
        self.pending = []

    def request(self, n: int):
        paths = self.library.take(n) if self.library else []
        for p in paths:
            s = int(p.stem)
            self.pending.append((s, self.pool.apply_async(tracklib.load, (p,)) if self.pool else tracklib.load(p)))
        for _ in range(n - len(paths)):
            s = int(self.rng.integers(HELDOUT_SEED_BASE))
            self.pending.append((s, self.pool.apply_async(_generate, (s, self.cfg)) if self.pool
                                 else _generate(s, self.cfg)))

    def initial(self, n: int) -> list:
        """[(seed, raster)]: n library tracks at once, on every core the job has."""
        paths = self.library.take(n) if self.library else []
        if not paths:
            return []
        k = min(len(os.sched_getaffinity(0)), len(paths))
        if k > 1:
            with mp.get_context('spawn').Pool(k) as pool:
                rasters = pool.map(_load_or_none, paths, chunksize=16)
        else:
            rasters = [_load_or_none(p) for p in paths]
        return [(int(p.stem), r) for p, r in zip(paths, rasters) if r is not None]

    def ready(self) -> list:
        """[(seed, raster)] for the tracks finished since the last call."""
        done, keep = [], []
        for s, p in self.pending:
            if not self.pool:
                done.append((s, p))
            elif p.ready():
                try:
                    done.append((s, p.get()))
                except Exception:     # noqa: BLE001 -- a failed track is skipped, never fatal
                    pass
            else:
                keep.append((s, p))
        self.pending = keep
        return done

    def regenerate(self, seeds) -> list:
        """[(seed, raster or None)]: loaded from the library where it has the
        seed, else generated. A resume is not training yet, so this uses every
        core the job has, not just the feeder's workers."""
        lib = self.library.by_seed if self.library else {}
        gen = partial(_generate_or_none, cfg=self.cfg)
        jobs = [(_load_or_none, lib[s]) if s in lib else (gen, s) for s in seeds]
        n = min(len(os.sched_getaffinity(0)), len(seeds))
        if self.workers > 0 and n > 1:
            with mp.get_context('spawn').Pool(n) as pool:
                out = pool.starmap(_call, jobs, chunksize=4)
        else:
            out = [fn(arg) for fn, arg in jobs]
        return list(zip(seeds, out))

    def close(self):
        if self.pool:
            self.pool.terminate()


class LevelMaker:
    """A Level, deterministically from (slot, seed): replaying a level replays its start."""

    def __init__(self, max_cars: int, solo: bool = False, a_lat_range=A_LAT_RANGE):
        self.max_cars, self.solo = max_cars, solo
        self.a_lat_range = a_lat_range
        self.specs = {}

    def make(self, slot: int, seed: int, level_id: int) -> Level:
        spec = self.specs[slot]
        rng = np.random.default_rng(seed)
        n_cars = 1 if self.solo else int(rng.integers(1, self.max_cars + 1))   # 1-4, team decision
        poses, idx = tracks.start_grid(spec, n_cars, rng)
        types, probs = zip(*EPISODE_MIX)
        ep = int(rng.choice(types, p=probs))
        amp = rng.uniform(0.0, [0.3, 0.15, 0.08])
        lv = Level(slot=slot, poses=poses, start_idx=idx, episode_type=ep,
                   v_cap=float(rng.uniform(*V_CAP_RANGE)),
                   imu_present=bool(rng.random() >= IMU_DROPOUT_EPISODES),
                   ref_offset=np.stack([amp, rng.uniform(0, 2 * math.pi, 3)], axis=1),
                   slam_drift=float(rng.uniform(-0.05, 0.05)),
                   slam_heading=float(rng.uniform(-math.radians(1), math.radians(1))),
                   level_id=level_id)
        # drawn last, so every earlier draw of a level is what it was before
        lo, hi = self.a_lat_range
        if hi > lo and rng.random() >= A_LAT_CAR_SHARE:
            lv.a_lat_cap = float(rng.uniform(lo, hi))
        else:
            lv.a_lat_cap = float(lo)
        # the speed bump and the overtaking grid draw from their own stream, so the draws above stay put
        extra = np.random.default_rng([seed, 7])
        if extra.random() < V_CAP_HIGH_SHARE:
            lv.v_cap = float(extra.uniform(V_CAP_HIGH, V_CAP_RANGE[1]))
        if n_cars >= 2 and extra.random() < OVERTAKE_SHARE:
            order = [n_cars - 1] + list(range(n_cars - 1))          # seat 0 last on the grid
            lv.poses, lv.start_idx = tracks.start_grid(spec, n_cars, extra, order=order)
            lv.v_cap_scale = np.concatenate([[1.0], extra.uniform(*OVERTAKE_SCALE, n_cars - 1)])
        return lv


class Runner:
    """Owns the sim, the track pool, the level buffer and episode resets."""

    def __init__(self, num_envs: int, max_cars: int, device, seed: int, *, pool_slots: int = 64,
                 canvas=(1024, 1024), cpu_workers: int = 0, solo: bool = False, initial_tracks: int = 8,
                 warmup_levels: int = 64, library=None, sensors: bool = True, reward_cfg=None,
                 max_seconds: float = MAX_SECONDS, level_cap: int | None = None, a_lat_range=A_LAT_RANGE,
                 end_with_learner: bool = True):
        """library: a tracklib directory; when given, the whole pool is filled
        from it at start-up. level_cap: at most this many envs on one level
        at a time (default 1 per 1024 envs, at least 2) -- with thousands of
        envs drawing from one PLR distribution, the top level otherwise takes
        a large share of every batch."""
        self.E, self.C, self.device = num_envs, max_cars, device
        self.warmup_levels = warmup_levels
        self.level_cap = level_cap or max(2, num_envs // 1024)
        self.rng = np.random.default_rng(seed)
        self.pool = TrackPool(pool_slots, canvas=canvas, device=device)
        self.levels = plr_mod.LevelBuffer(capacity=max(pool_slots * 8, num_envs * 2), seed=seed)
        self.maker = LevelMaker(max_cars, solo, a_lat_range)
        self.held = heldout_digests()
        lib = tracklib.Library(library, seed=seed) if library else None      # its own config: v1 or v2
        self.feeder = TrackFeeder(cpu_workers, seed + 1, lib)
        self.slot_digest = {}
        self.slot_seed = {}       # slot -> track seed: what a checkpoint keeps of the pool
        self.free_slots = list(range(pool_slots))
        self.env_level = [None] * num_envs
        self.level_envs = {}      # level id -> envs on it now
        self.level_t = torch.full((num_envs,), -1, dtype=torch.long, device=device)   # env -> level id
        self.score_acc = {}       # level id -> [sum of positive advantages, steps]
        self.ended = set()        # levels with an episode ended since the last add_scores
        for s, r in (self.feeder.initial(pool_slots) if lib else []):
            self._insert(r, s)
        for k in range(max(min(initial_tracks, pool_slots) - len(self.slot_digest), 0)):
            s = int(self.rng.integers(HELDOUT_SEED_BASE))
            self._insert(_generate(s, self.feeder.cfg), s)
        if lib:
            print(f'[runner] {len(self.slot_digest)} tracks from the library ({len(lib)} in {library})', flush=True)
        kw = {} if reward_cfg is None else {'reward_cfg': reward_cfg}
        self.sim = BatchedRaceSim(num_envs, max_cars, self.pool, device=device, seed=seed, sensors=sensors,
                                  max_seconds=max_seconds, **kw)
        # the learner's episode is the env's: after it finishes or crashes, the
        # others' race is wasted ticks (a parked opponent never finishes)
        self.sim.end_with_seat0 = end_with_learner
        self.env_replay = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.reset(list(range(num_envs)))

    def _insert(self, raster, seed: int) -> bool:
        if raster.digest in self.held:
            raise RuntimeError(f'held-out track {raster.digest} reached the training pool')
        if raster.sdf.shape[0] > self.pool.H or raster.sdf.shape[1] > self.pool.W:
            return False
        if self.free_slots:
            slot = self.free_slots.pop()
        else:
            # evict the idle track whose best level has the least left to learn
            # (one pass over the levels: with a full 4096-slot pool, a scan per
            # slot was 10^8 steps an insert)
            busy = {self.levels.levels[i].meta['slot'] for i in self.level_envs if i in self.levels.levels}
            best = {}
            for r in self.levels.levels.values():
                s = r.meta.get('slot')
                best[s] = max(best.get(s, 0.0), r.score)
            cand = [s for s in self.slot_digest if s not in busy]
            if not cand:
                return False
            slot = min(cand, key=lambda s: best.get(s, 0.0))
            for i in [i for i, r in self.levels.levels.items() if r.meta.get('slot') == slot]:
                del self.levels.levels[i]
        self._place(raster, seed, slot)
        return True

    def _place(self, raster, seed: int, slot: int):
        self.pool.insert(raster, slot)
        self.maker.specs[slot] = raster.spec
        self.slot_digest[slot] = raster.digest
        self.slot_seed[slot] = seed

    def state(self) -> dict:
        return {'plr': self.levels.state(), 'slot_seed': dict(self.slot_seed)}

    def load(self, plr_state: dict, slot_seed: dict | None) -> None:
        """Restore the level buffer and rebuild the tracks its levels point at.
        A checkpoint without slot seeds (before this was saved) keeps only the
        levels whose slot this pool already holds."""
        self.levels.load(plr_state)
        if slot_seed:
            for slot, (seed, raster) in zip(slot_seed, self.feeder.regenerate(list(slot_seed.values()))):
                slot = int(slot)
                if raster is None or raster.digest in self.held or slot >= self.pool.slots:
                    continue
                self._place(raster, seed, slot)
            self.free_slots = [s for s in range(self.pool.slots) if s not in self.slot_digest]
        lost = [i for i, r in self.levels.levels.items()
                if self.slot_digest.get(r.meta.get('slot')) != r.track_digest]
        for i in lost:
            del self.levels.levels[i]
        print(f'[runner] resumed {len(self.levels.levels)} levels on {len(self.slot_digest)} tracks'
              f' ({len(lost)} levels dropped)', flush=True)
        self.reset(list(range(self.E)))

    def refresh_tracks(self, want: int | None = None):
        """Insert the tracks the feeder has ready and keep 2 requests per
        worker in flight. Without workers, nothing: generating in-process
        would stall training for seconds a track."""
        if self.feeder.workers == 0:
            return
        for s, r in self.feeder.ready():
            self._insert(r, s)
        want = want or 2 * self.feeder.workers
        if len(self.feeder.pending) < want:
            self.feeder.request(want - len(self.feeder.pending))

    def reset(self, env_ids):
        for e in env_ids:
            old = self.env_level[e]
            if old is not None:
                left = self.level_envs.get(old, 1) - 1
                if left > 0:
                    self.level_envs[old] = left
                else:
                    self.level_envs.pop(old, None)
        draws = self.levels.sample(len(env_ids))
        lv, ids = [], []
        for e, (level_id, replay) in zip(env_ids, draws):
            busy = level_id is not None and self.level_envs.get(level_id, 0) >= self.level_cap
            if level_id is None or level_id not in self.levels.levels or busy:
                slot = int(self.rng.choice(list(self.slot_digest)))
                level_id = self.levels.new_id()
                self.levels.add(plr_mod.LevelRecord(level_id, self.slot_digest[slot],
                                                    int(self.rng.integers(2 ** 31)), meta={'slot': slot}))
                replay = False
            rec = self.levels.levels[level_id]
            lv.append(self.maker.make(rec.meta['slot'], rec.seed, level_id))
            self.env_level[e] = level_id
            self.level_envs[level_id] = self.level_envs.get(level_id, 0) + 1
            ids.append(level_id)
            # robust PLR trains only on replayed levels -- once there are
            # enough levels to replay; until then every episode trains
            self.env_replay[e] = replay or len(self.levels.levels) < self.warmup_levels
        self.level_t[torch.as_tensor(env_ids, device=self.device)] = torch.as_tensor(ids, device=self.device)
        self.sim.reset_envs(env_ids, lv)

    def add_scores(self, adv, level_ids, mask=None):
        """Robust PLR's score, the mean positive advantage of a level's
        episode, credited step by step: adv and level_ids are (T, E) for
        seat 0, level_ids as level_t read at each step; mask (T, E), the
        steps to count (the learner's live ones). A level whose episode
        ended in this rollout gets its score now."""
        a = adv.detach().clamp(min=0).reshape(-1)
        lid = level_ids.reshape(-1)
        if mask is not None:
            keep = mask.reshape(-1)
            a, lid = a[keep], lid[keep]
        uniq, inv = torch.unique(lid, return_inverse=True)
        sums = torch.zeros(len(uniq), device=a.device).index_add_(0, inv, a)
        counts = torch.bincount(inv, minlength=len(uniq))
        for i, s, n in zip(uniq.tolist(), sums.tolist(), counts.tolist()):
            if i < 0:
                continue
            acc = self.score_acc.setdefault(i, [0.0, 0])
            acc[0] += s
            acc[1] += n
        for i in self.ended:
            acc = self.score_acc.pop(i, None)
            if acc and acc[1] > 0:
                self.levels.update_score(i, acc[0] / acc[1])
        self.ended.clear()
        self.score_acc = {i: v for i, v in self.score_acc.items() if i in self.levels.levels}

    def end_episodes(self, done_mask):
        ids = done_mask.nonzero().flatten().tolist()
        if len(self.ended) > self.levels.capacity:
            self.ended.clear()      # a stage that never scores (distill, benchmark)
        for e in ids:
            self.ended.add(self.env_level[e])
        if ids:
            self.reset(ids)
        return ids


def aux_targets(sim: BatchedRaceSim):
    """Targets for the auxiliary heads, from the simulator's true state:

      ego path: the car's next 1.5 s under its current speed and yaw rate
      (constant-turn extrapolation), in its own rear-axle frame;
      opponents: up to 4 others by distance, present flag, and their next
      1.5 s at constant velocity in the same frame.
    v1 simplification: extrapolated from the true state rather than read from
    later frames of the rollout (plan, Phase 6 note).
    """
    E, C = sim.E, sim.C
    t = obs.AUX_DT * torch.arange(1, obs.AUX_PATH_POINTS + 1, device=sim.device).view(1, 1, -1)
    v, w = sim.x[..., 3].unsqueeze(-1), sim.x[..., 5].unsqueeze(-1)
    small = w.abs() < 1e-3
    ws = torch.where(small, torch.ones_like(w), w)
    px = torch.where(small, v * t, v * torch.sin(w * t) / ws)
    py = torch.where(small, 0.5 * v * w * t * t, v * (1 - torch.cos(w * t)) / ws)
    ego = torch.stack([px, py], -1)                                       # (E, C, 6, 2)
    bx, by, yaw = sim._rear_axle()
    vx = sim.x[..., 3] * torch.cos(yaw + sim.x[..., 6])
    vy = sim.x[..., 3] * torch.sin(yaw + sim.x[..., 6])
    d = torch.cdist(torch.stack([bx, by], -1), torch.stack([bx, by], -1))
    d = d + torch.eye(C, device=sim.device) * 1e6
    d = torch.where(sim.active.unsqueeze(1), d, torch.full_like(d, 1e6))
    order = d.argsort(-1)
    opp = torch.zeros(E, C, obs.OPP_SLOTS, 1 + 2 * obs.AUX_PATH_POINTS, device=sim.device)
    c, s = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
    for slot in range(min(obs.OPP_SLOTS, C - 1)):
        k = order[..., slot]
        present = torch.gather(d, 2, k.unsqueeze(-1)).squeeze(-1) < 1e5
        ox = torch.gather(bx, 1, k).unsqueeze(-1) + torch.gather(vx, 1, k).unsqueeze(-1) * t - bx.unsqueeze(-1)
        oy = torch.gather(by, 1, k).unsqueeze(-1) + torch.gather(vy, 1, k).unsqueeze(-1) * t - by.unsqueeze(-1)
        lx, ly = c * ox + s * oy, -s * ox + c * oy
        opp[..., slot, 0] = present.float()
        opp[..., slot, 1::2] = lx * present.unsqueeze(-1)
        opp[..., slot, 2::2] = ly * present.unsqueeze(-1)
    return ego, opp


def aux_loss(path, opp, ego_t, opp_t):
    """MSE on the ego path; BCE on presence and masked MSE on opponent positions."""
    l_path = ((path - ego_t) ** 2).mean()
    present = opp_t[..., 0]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(opp[..., 0], present)
    pos_err = ((opp[..., 1:] * obs.AUX_POS_SCALE - opp_t[..., 1:]) ** 2).mean(-1)
    l_pos = (pos_err * present).sum() / present.sum().clamp(min=1.0)
    return l_path, bce + l_pos


# --- opponent seats --------------------------------------------------------------------------

class SeatAssignment:
    """What drives each opponent seat (E, C): an index into `sources`, plus
    per-seat random draws for scripted drivers. Seat 0 is always the learner."""

    def __init__(self, E: int, C: int, device):
        self.source = torch.zeros(E, C, dtype=torch.long, device=device)
        self.params = torch.zeros(E, C, 3, device=device)
        self.h = torch.zeros(E, C, 128, device=device)

    def assign(self, env_ids, n_sources: int, gen, offset: int = 0):
        """New drivers for the envs in env_ids, from sources offset .. offset + n_sources - 1."""
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=self.source.device)
        C = self.source.shape[1]
        self.source[ids] = offset + torch.randint(0, n_sources, (len(ids), C), generator=gen,
                                                  device=gen.device).to(self.source.device)
        self.params[ids] = torch.rand(len(ids), C, 3, generator=gen, device=gen.device).to(self.params.device)
        self.h[ids] = 0.0


def opponent_actions(sim, o: dict, priv, seats: SeatAssignment, sources: list, stacked=None):
    """Squashed actions (E, C, 2) for seats 1..C-1.

    sources[i] is ('scripted', name) or ('policy', g): group g of `stacked`.
    Every policy group's seats are gathered into one padded (M, B, ...) batch
    and run in a single vmap call -- M opponents for the price of one pass.
    """
    from neural_racer_train import scripted
    from neural_racer_train.model import squash
    E, C = sim.E, sim.C
    out = torch.zeros(E, C, 2, device=sim.device)
    t = float(sim.t.float().mean()) * 0.025
    v_cap = sim.v_cap.view(-1, 1).expand(E, C)
    racing = sim.active.clone()
    racing[:, 0] = False
    groups = {}
    for i, (kind, name) in enumerate(sources):
        m = (seats.source == i) & racing
        if not m.any():
            continue
        if kind == 'scripted':
            out[m] = scripted.act(name, priv[m], o['scan'][m], v_cap[m], seats.params[m], t)
        else:
            groups[int(name)] = m
    if groups:
        M = stacked.M
        idx = {g: m.nonzero(as_tuple=True) for g, m in groups.items()}
        B = max(len(ix[0]) for ix in idx.values())
        ob = {k: torch.zeros(M, B, *v.shape[2:], device=sim.device) for k, v in o.items()}
        hh = torch.zeros(M, B, 128, device=sim.device)
        for g, (ei, ci) in idx.items():
            n = len(ei)
            for k, v in o.items():
                ob[k][g, :n] = v[ei, ci]
            hh[g, :n] = seats.h[ei, ci]
        a, h2 = stacked.act(ob, hh)
        for g, (ei, ci) in idx.items():
            n = len(ei)
            out[ei, ci] = squash(a[g, :n], v_cap[ei, ci])
            seats.h[ei, ci] = h2[g, :n]
    return out
