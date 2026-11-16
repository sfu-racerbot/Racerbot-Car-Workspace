"""Held-out evaluation of a teacher: the race we train for -- 3 laps on a
procedural track, without a crash -- on the 20 held-out tracks (plan Phase 4),
obstacles removed while training has none (learn.TRAIN_TRACKS). Actions are
the mean. The speed is held to the car wrapper's curvature cap at the race's
cornering cap (2.0 m/s^2 unless asked), as the car will hold it.

    python -m neural_racer_train.teacher_eval CKPT [CKPT ...] [--starts 8] [--fresh 8] [--a-lat 2.0]
                                              [--pursue] [--no-cap]

Fixed starts (level seeds from 777) are the same for every snapshot, so
snapshots compare; fresh ones (--fresh, a new draw each time) say how much of
a result is luck. `finish_lo`/`finish_hi` are the Wilson 95% interval;
`best.pt` is chosen on its lower bound. `feasible` is the share of races
where 3 laps fit in the limit even at the grip- and cap-limited speed along
the centreline, `hazard` crashes per car-tick, `ref_ratio` the median race
time over the minimum-curvature line's at the same caps (gate G5's measure;
raceline_ref), `curv_cut` the share of live ticks the curvature cap cut the
command, `alat_p95` the 95th percentile of v * yaw rate over live ticks.
"""
import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from neural_racer_train import raceline_ref, realmaps, scripted, tracks
from neural_racer_train.gpu_sim.params import CONTROL_DT, G
from neural_racer_train.gpu_sim.sim import BatchedRaceSim
from neural_racer_train.learn import HELDOUT_FILE, LevelMaker
from neural_racer_train.model import TeacherNet, load_expanded, mask_track, teacher_squash
from neural_racer_train.track_pool import TrackPool

LAPS, MAX_SECONDS = 3, 300.0
REAL_MAPS = ('room_2026_07_27', 'levine_blocked', 'levine_obs', 'circular_obs',
             'car1_course_2026_10_26', 'car2_course_2026_10_30',
             'car2_course_2026_11_06a_clean', 'car2_course_2026_11_06b_clean')     # heldout.REAL; the last three are car 2's course
# the real maps a teacher can be judged on: it never sees obstacles (levine_obs and circular_obs have boxes)
OBSTACLE_FREE_REAL = ('room_2026_07_27', 'levine_blocked', 'car1_course_2026_10_26', 'car2_course_2026_10_30',
                      'car2_course_2026_11_06a_clean', 'car2_course_2026_11_06b_clean')
FIXED_LEVEL_SEED, FIXED_SIM_SEED = 777, 12345
FRESH_SEED = 500_000            # + 10_000 x the draw: never a fixed seed
VCAP_BINS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.4, 8.0, 10.0)
CUT_EPS = 1e-3                  # m/s: a smaller curvature cut is rounding, not a cut
ALAT_BINS, ALAT_MAX = 400, 20.0


def heldout_specs(obstacles: bool = False, limit: int | None = None) -> list:
    held = yaml.safe_load(HELDOUT_FILE.read_text())['procedural'][:limit]
    out = []
    for h in held:
        s = tracks.generate(int(h['seed']), kind=h['kind'])
        out.append(s if obstacles else replace(s, obstacles=np.zeros((0, 4))))
    return out


def _min_time(spec, v_cap: float, mu: float) -> float:
    """Lower bound on LAPS laps: the centreline at min(v_cap, sqrt(0.9 mu g / kappa)), no acceleration limit."""
    c = spec.centre
    a, b = np.roll(c, 5, axis=0), np.roll(c, -5, axis=0)
    ab, bc, ca = c - a, b - c, a - b
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    den = np.hypot(*ab.T) * np.hypot(*bc.T) * np.hypot(*ca.T)
    kappa = np.abs(np.where(den > 1e-12, 2 * cross / np.maximum(den, 1e-12), 0.0))
    v = np.minimum(v_cap, np.sqrt(0.9 * mu * G / np.maximum(kappa, 1e-3)))
    return LAPS * float(np.sum(tracks.SPACING / v))


def wilson(k: float, n: float, z: float = 1.96):
    """Wilson score interval for k successes in n trials."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


class Evaluator:
    """Built once (the tracks and their racing lines take a few seconds); run() any number of policies."""

    def __init__(self, starts: int = 8, device='cpu', obstacles: bool = False, limit: int | None = None,
                 max_seconds: float = MAX_SECONDS, seed: int = FIXED_SIM_SEED, raceline: bool = True,
                 real: bool | tuple = False):
        """real: also the real course maps (realmaps), after the procedural
        tracks: True for all of REAL_MAPS, or a tuple of their names."""
        specs = heldout_specs(obstacles, limit)
        real_names = REAL_MAPS if real is True else tuple(real or ())
        self.rasters = [tracks.rasterize(s) for s in specs] + [realmaps.load(n) for n in real_names]
        self.specs = [r.spec for r in self.rasters]
        self.names = [f'heldout_{k:02d}' for k in range(len(specs))] + list(real_names)
        self.N, self.starts, self.device, self.seed = len(self.specs), starts, device, seed
        self.E = self.N * starts
        self.max_seconds = max_seconds
        self.pool = TrackPool(self.N, canvas=(1024, 1024), device=device)
        self.maker = LevelMaker(1, solo=True)
        for k, r in enumerate(self.rasters):
            self.pool.insert(r, k)
            self.maker.specs[k] = r.spec
        self.lines = [raceline_ref.optimize(s) for s in self.specs] if raceline else None

    def _level(self, k: int, seed: int, level_id: int):
        """maker.make; on a real map, re-drawn (deterministically) while a
        car's rear axle or a point 40 cm ahead of it is within 18 cm of a
        wall -- a box on the centreline. Procedural levels are never re-drawn."""
        r = self.rasters[k]
        lv = self.maker.make(k, seed, level_id)
        if not r.spec.kind.startswith('real:'):
            return lv
        for j in range(1, 32):
            yaw = lv.poses[:, 2]
            px = np.concatenate([lv.poses[:, 0], lv.poses[:, 0] + 0.4 * np.cos(yaw)])
            py = np.concatenate([lv.poses[:, 1], lv.poses[:, 1] + 0.4 * np.sin(yaw)])
            col = ((px - r.origin[0]) / r.resolution).astype(int).clip(0, r.sdf.shape[1] - 1)
            row = ((py - r.origin[1]) / r.resolution).astype(int).clip(0, r.sdf.shape[0] - 1)
            if (r.sdf[row, col] >= 0.18).all():
                break
            lv = self.maker.make(k, seed + 7919 * j, level_id)
        return lv

    def races(self, policy, *, starts: int, level_seed: int, sim_seed: int, a_lat=2.0, cap: bool = True,
              wrapper: bool = False, episode_type=None, mask: bool = False) -> dict:
        """Per-race results of one batch: starts x every track. a_lat None keeps each level's own draw.
        wrapper: the car's whole safety wrapper, which reads the LiDAR, so the
        sim casts it (a student's race); else the actuator limits and, with
        cap, the curvature cap only. episode_type: every race reactive, map or
        handover (sim.EPISODE_*), else each level's own draw; a list of them
        races the same starts once per type in the one batch (split() takes
        a type's races back out; the car draws differ between the copies). A policy with
        `uses_sim` is called with the sim (it builds its own inputs, and
        `reset(E)` first clears its memory); else with (priv, v_cap) -- with
        mask, priv hides what the LiDAR cannot see in reactive mode (model.mask_track)."""
        N, dev = self.N, self.device
        types = list(episode_type) if isinstance(episode_type, (list, tuple)) else [episode_type]
        B = N * starts                      # one copy of the races
        E = B * len(types)
        sim = BatchedRaceSim(E, 1, self.pool, device=dev, seed=sim_seed, laps=LAPS,
                             max_seconds=self.max_seconds, sensors=wrapper)
        sim.safety_enabled = wrapper
        sim.curvature_cap = cap
        levels = [self._level(e % N, level_seed + e % B, e) for e in range(E)]
        for e, lv in enumerate(levels):
            if a_lat is not None:
                lv.a_lat_cap = float(a_lat)
            if types[e // B] is not None:
                lv.episode_type = int(types[e // B])
        sim.reset_envs(list(range(E)), levels)
        uses_sim = getattr(policy, 'uses_sim', False)
        if hasattr(policy, 'reset'):
            policy.reset(E)
        interv, stops, cut_ticks = (torch.zeros(E, device=dev) for _ in range(3))     # per race
        mu, vcap, alat, ep = (t.cpu().numpy().copy() for t in (sim.p.mu, sim.v_cap, sim.a_lat, sim.ep_type))
        done = torch.zeros(E, dtype=torch.bool, device=dev)
        finished, crashed = torch.zeros_like(done), torch.zeros_like(done)
        t_end = torch.full((E,), float(sim.max_ticks), device=dev)
        alive_ticks, speed_sum = torch.zeros(E, device=dev), torch.zeros(E, device=dev)
        hist = torch.zeros(E, ALAT_BINS, device=dev)
        with torch.no_grad():
            for t in range(sim.max_ticks):
                if uses_sim:
                    act = policy(sim)
                else:
                    priv = sim.privileged()[:, 0]
                    if mask:
                        priv = mask_track(priv, sim.track_visibility()[:, 0], sim.mode()[:, 0] == 0)
                    act = policy(priv, sim.v_cap)
                out = sim.step(act.unsqueeze(1))
                live = out['alive'][:, 0] & ~done
                interv += (out['intervened'][:, 0] & live).float()
                stops += (out['safety_stop'][:, 0] & live).float()
                alive_ticks += live.float()
                speed_sum += torch.where(live, out['speed'][:, 0], torch.zeros_like(speed_sum))
                cut_ticks += ((out['curv_cut'][:, 0] > CUT_EPS) & live).float()
                if t % 4 == 0:      # lateral acceleration v * yaw rate, sampled every 0.1 s (histc's bins)
                    a = (sim.x[:, 0, 3] * sim.x[:, 0, 5]).abs()
                    b = (a * (ALAT_BINS / ALAT_MAX)).long().clamp(0, ALAT_BINS - 1)
                    hist.scatter_add_(1, b.unsqueeze(1), (live & (a <= ALAT_MAX)).float().unsqueeze(1))
                fin = out['finished'][:, 0] & ~done
                crash = out['terminal'][:, 0] & ~out['finished'][:, 0] & ~done
                new = fin | crash
                finished |= fin
                crashed |= crash
                t_end = torch.where(new, torch.full_like(t_end, float(t + 1)), t_end)
                done |= new
                if t % 200 == 0 and bool(done.all()):
                    break
        laps = (sim.race.total[:, 0] / self.pool.length[sim.slot]).clamp(max=LAPS)
        track = np.arange(E) % N
        ref = np.full(E, np.nan)
        if self.lines:
            ref = np.array([raceline_ref.race_time(self.lines[track[e]], levels[e].poses[0], vcap[e], alat[e], mu[e],
                                                   LAPS) for e in range(E)])
        return {'finish': finished.cpu().numpy(), 'crash': crashed.cpu().numpy(), 'timeout': (~done).cpu().numpy(),
                'time': (t_end * CONTROL_DT).cpu().numpy(), 'laps': laps.cpu().numpy(),
                'alive_ticks': alive_ticks.cpu().numpy(), 'speed_sum': speed_sum.cpu().numpy(),
                'feasible': np.array([_min_time(self.specs[track[e]], vcap[e], mu[e]) <= self.max_seconds
                                      for e in range(E)]),
                'ref': ref, 'v_cap': vcap, 'a_lat': alat, 'track': track, 'episode_type': ep,
                'cut_ticks': cut_ticks.cpu().numpy(), 'interv_ticks': interv.cpu().numpy(),
                'stop_ticks': stops.cpu().numpy(), 'alat_hist': hist.cpu().numpy()}

    def run(self, policy, a_lat=2.0, cap: bool = True, starts: int | None = None, fresh: int = 0,
            draw: int = 0, **kw) -> dict:
        """policy(priv (E, PRIV_DIM), v_cap (E,)) -> (steering, speed) (E, 2). `starts` fixed
        starts per track, plus `fresh` ones from fresh draw number `draw`; kw goes to races()."""
        recs = [self.races(policy, starts=starts or self.starts, level_seed=FIXED_LEVEL_SEED,
                           sim_seed=self.seed, a_lat=a_lat, cap=cap, **kw)]
        if fresh:
            s = FRESH_SEED + 10_000 * draw
            recs.append(self.races(policy, starts=fresh, level_seed=s, sim_seed=s, a_lat=a_lat, cap=cap, **kw))
        return summarize(recs, self.N, self.names)


def split(rec: dict, mask) -> dict:
    """The races of one races() result where mask (one per race) holds."""
    return {k: v[mask] for k, v in rec.items()}


def summarize(recs: list, n_tracks: int, names=None) -> dict:
    r = {k: np.concatenate([x[k] for x in recs]) for k in recs[0]}
    fin, cr, E = r['finish'].astype(bool), r['crash'].astype(bool), len(r['finish'])
    lo, hi = wilson(fin.sum(), E)
    live = float(r['alive_ticks'].sum())
    hist = r['alat_hist'].sum(0)
    cdf = np.cumsum(hist) / max(hist.sum(), 1.0)
    pct = lambda q: float(np.searchsorted(cdf, q) + 0.5) * ALAT_MAX / ALAT_BINS     # noqa: E731
    ratio = r['time'][fin] / r['ref'][fin]
    ratio = ratio[np.isfinite(ratio)]
    feas = r['feasible'].astype(bool)
    by_vcap = {}
    for a, b in zip(VCAP_BINS[:-1], VCAP_BINS[1:]):
        m = (r['v_cap'] >= a) & (r['v_cap'] < b)
        if m.any():
            by_vcap[f'{a:.0f}-{b:.0f}'] = [int(m.sum()), float(fin[m].mean()), float(cr[m].mean())]
    per_track = np.bincount(r['track'], weights=cr.astype(float), minlength=n_tracks)
    worst = np.argsort(-per_track)[:5]
    real = {}
    for k, name in enumerate(names or []):
        m = r['track'] == k
        if name in REAL_MAPS and m.any():
            real[name] = [int(m.sum()), float(fin[m].mean()), float(cr[m].mean())]
    return {'finish': float(fin.mean()), 'finish_lo': lo, 'finish_hi': hi, 'crash': float(cr.mean()),
            'timeout': float(r['timeout'].mean()), 'feasible': float(feas.mean()),
            'finish_feasible': float(fin[feas].mean()) if feas.any() else 0.0,
            'hazard': float(cr.sum()) / max(live, 1.0), 'laps': float(r['laps'].mean()),
            'speed': float((r['speed_sum'] / np.maximum(r['alive_ticks'], 1)).mean()),
            'race_time': float(r['time'][fin].mean()) if fin.any() else float('nan'),
            'ref_ratio': float(np.median(ratio)) if len(ratio) else float('nan'),
            'curv_cut': float(r['cut_ticks'].sum()) / max(live, 1.0),
            'intervened': float(r['interv_ticks'].sum()) / max(live, 1.0),
            'safety_stop': float(r['stop_ticks'].sum()) / max(live, 1.0),
            'alat_p50': pct(0.5), 'alat_p95': pct(0.95), 'alat_p99': pct(0.99),
            'a_lat': float(np.mean(r['a_lat'])), 'by_vcap': by_vcap,
            'worst_tracks': {int(k): int(per_track[k]) for k in worst if per_track[k] > 0},
            'tracks_crashed': int((per_track > 0).sum()), 'real_maps': real, 'episodes': E}


def teacher_policy(actor: TeacherNet, mode: str = 'plain'):
    sq = teacher_squash(mode)
    return lambda priv, v_cap: sq(actor(priv), v_cap, priv)


def pursue_policy(pace: float = 0.8):
    def act(priv, v_cap):
        n = priv.shape[0]
        return scripted.pursue(priv, v_cap, torch.zeros(n, device=priv.device), torch.full((n,), pace, device=priv.device))
    return act


def actor_from(state: dict, device='cpu') -> TeacherNet:
    """A TeacherNet from a state dict, widened if it predates the cornering-cap input."""
    actor = TeacherNet().to(device)
    load_expanded(actor, {k: v.to(device) for k, v in state.items()})
    actor.eval()
    return actor


def load_actor(path, device='cpu', key: str = 'actor'):
    """(actor, squash mode) from a checkpoint or snapshot."""
    ck = torch.load(path, map_location=device, weights_only=False)
    return actor_from(ck[key], device), ck.get('squash', 'plain')


_EVALUATORS = {}


def _evaluator(device='cpu', real=()) -> Evaluator:
    """One per worker process: the racing lines take a few seconds a track."""
    key = (device, tuple(real))
    if key not in _EVALUATORS:
        _EVALUATORS[key] = Evaluator(device=device, real=tuple(real))
    return _EVALUATORS[key]


def evaluate_file(path, plan, threads: int = 4, device='cpu', real=()) -> list:
    """A snapshot raced once per entry of plan -- dicts of {key ('actor' or
    'ema'), a_lat, starts, fresh, cap, wrapper} -- each result tagged with them. On the
    CPU in a worker process while training runs; on the GPU once it is over,
    and throughout for a teacher trained inside the wrapper (its races cast the LiDAR)."""
    if device == 'cpu':
        torch.set_num_threads(threads)
    ck = torch.load(path, map_location=device, weights_only=False)
    mode = ck.get('squash', 'plain')
    out = []
    for p in plan:
        if p['key'] not in ck:
            continue
        ev = _evaluator(device, real)
        res = ev.run(teacher_policy(actor_from(ck[p['key']], device), mode), a_lat=p['a_lat'], cap=p.get('cap', True),
                     starts=p['starts'], fresh=p.get('fresh', 0), draw=int(ck.get('update', 0)),
                     mask=p.get('mask', False), episode_type=p.get('episode_type'), wrapper=p.get('wrapper', False))
        res.update(steps=ck.get('steps', 0), update=ck.get('update', 0), path=str(path), variant=p['key'],
                   squash=mode, plan_a_lat=p['a_lat'], mask=p.get('mask', False), wrapper=p.get('wrapper', False))
        out.append(res)
    return out


def line(name: str, r: dict) -> str:
    return (f'EVAL {name:<24} a_lat={r.get("a_lat", float("nan")):.1f} n={r["episodes"]} '
            f'finish={r["finish"]:6.1%} [{r.get("finish_lo", 0):.1%}, {r.get("finish_hi", 1):.1%}] '
            f'(of feasible {r["finish_feasible"]:6.1%}) crash={r["crash"]:6.1%} hazard={r["hazard"]:.2e}/tick '
            f'laps={r["laps"]:4.2f} speed={r["speed"]:4.2f}m/s race={r["race_time"]:5.1f}s '
            f'ref_ratio={r.get("ref_ratio", float("nan")):.3f} curv_cut={r.get("curv_cut", 0):.4f} '
            f'alat_p95={r.get("alat_p95", float("nan")):.2f}')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('ckpts', nargs='*')
    ap.add_argument('--starts', type=int, default=8)
    ap.add_argument('--fresh', type=int, default=0, help='fresh starts per track on top of the fixed ones')
    ap.add_argument('--draw', type=int, default=0, help='which fresh draw')
    ap.add_argument('--a-lat', type=float, default=2.0, help='cornering cap, m/s^2; 0 = each level draws its own')
    ap.add_argument('--no-cap', action='store_true', help='no curvature cap (as teacher-v1..v3 trained)')
    ap.add_argument('--key', default='actor', help="'actor' or 'ema'")
    ap.add_argument('--mask', action='store_true', help='reactive races, the track hidden where the LiDAR cannot see')
    ap.add_argument('--pursue', action='store_true', help='also score scripted pursue at pace 0.8')
    ap.add_argument('--obstacles', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args(argv)
    ev = Evaluator(starts=a.starts, device=a.device, obstacles=a.obstacles)
    a_lat = a.a_lat or None
    print(f'[eval] {ev.N} held-out tracks x ({a.starts} fixed + {a.fresh} fresh) starts, obstacles '
          f'{"on" if a.obstacles else "off"}, curvature cap {"off" if a.no_cap else a_lat or "per level"}, '
          f'{ev.max_seconds:.0f} s', flush=True)
    kw = {'a_lat': a_lat, 'cap': not a.no_cap, 'fresh': a.fresh, 'draw': a.draw}
    if a.mask:
        kw.update(mask=True, episode_type=0)
    if a.pursue:
        print(line('pursue pace=0.8', ev.run(pursue_policy(0.8), **kw)), flush=True)
    for p in a.ckpts:
        actor, mode = load_actor(p, a.device, a.key)
        res = ev.run(teacher_policy(actor, mode), **kw)
        print(line(f'{Path(p).stem} ({mode})', res), flush=True)
        print(f'  by v_cap [n, finish, crash]: {res["by_vcap"]}  worst tracks: {res["worst_tracks"]}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
