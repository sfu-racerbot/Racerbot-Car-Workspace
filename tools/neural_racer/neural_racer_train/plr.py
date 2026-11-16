"""Prioritized level replay (robust PLR) with ACCEL-style mutation (plan A8).

A level is a track plus how an episode on it starts. 70% of episodes replay
high-scoring levels and are trained on; 30% are fresh levels, scored but not
trained on (robust PLR's rule). Score = mean positive value loss, the
estimate of how much there is left to learn on the level.

Temperature 1 (P ~ 1/rank), not the paper's 0.1-0.3: those were tuned for 64
parallel envs, and with thousands drawing from one distribution 0.3 gave the
top-ranked level 87% of the score-weighted draws (teacher-v2: 10 of 304
tracks took 66% of the replays). Runner also caps envs per level.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LevelRecord:
    level_id: int
    track_digest: str
    seed: int
    score: float = 0.0
    last_seen: int = 0
    plays: int = 0
    meta: dict = field(default_factory=dict)


class LevelBuffer:

    def __init__(self, capacity: int = 4000, replay_prob: float = 0.7,
                 staleness: float = 0.3, temperature: float = 1.0, seed: int = 0, refresh: int = 16):
        """refresh: draws between recomputing the sampling distribution (a
        Python pass over every level; per draw it was a third of a tick)."""
        self.capacity, self.replay_prob = capacity, replay_prob
        self.staleness, self.temperature = staleness, temperature
        self.refresh = refresh
        self.levels: dict[int, LevelRecord] = {}
        self.clock = 0
        self.next_id = 0
        self.rng = np.random.default_rng(seed)
        self._cache = None                 # (clock, ids, probabilities)

    def new_id(self) -> int:
        self.next_id += 1
        return self.next_id - 1

    def add(self, rec: LevelRecord) -> None:
        if len(self.levels) >= self.capacity and rec.level_id not in self.levels:
            # evict the lowest-scoring 5% at once, not one level per add
            recs = list(self.levels.values())
            k = max(1, self.capacity // 20)
            scores = np.array([r.score for r in recs])
            for j in np.argpartition(scores, k - 1)[:k]:
                del self.levels[recs[j].level_id]
            self._cache = None
        self.levels[rec.level_id] = rec

    def probabilities(self, ids) -> np.ndarray:
        """Rank-based score prioritisation mixed with staleness, as robust PLR."""
        recs = [self.levels[i] for i in ids]
        scores = np.array([r.score for r in recs])
        ranks = np.empty(len(recs))
        ranks[np.argsort(-scores, kind='stable')] = np.arange(1, len(recs) + 1)
        h = (1.0 / ranks) ** (1.0 / self.temperature)
        p_score = h / h.sum()
        stale = np.array([self.clock - r.last_seen for r in recs], dtype=np.float64)
        p_stale = stale / stale.sum() if stale.sum() > 0 else np.full(len(recs), 1.0 / len(recs))
        return (1 - self.staleness) * p_score + self.staleness * p_stale

    def sample(self, n: int):
        """n draws: (level_id or None for 'make a fresh level', is_replay)."""
        self.clock += 1
        if not self.levels:
            return [(None, False)] * n
        # one probability vector for many draws: computing it per draw made a
        # reset O(n x buffer), the slowdown of the teacher-v2 smoke run
        if self._cache is None or self.clock - self._cache[0] >= self.refresh:
            ids = list(self.levels)
            self._cache = (self.clock, ids, self.probabilities(ids))
        _, ids, p = self._cache
        replay = self.rng.random(n) < self.replay_prob
        picks = iter(self.rng.choice(ids, size=int(replay.sum()), p=p).tolist())
        out = []
        for r in replay:
            i = int(next(picks)) if r else None
            if i is not None and i in self.levels:     # the cache may hold a level evicted since
                self.levels[i].last_seen = self.clock
                self.levels[i].plays += 1
                out.append((i, True))
            else:
                out.append((None, False))
        return out

    def update_score(self, level_id: int, score: float) -> None:
        if level_id in self.levels:
            self.levels[level_id].score = float(score)

    def state(self) -> dict:
        return {'levels': {k: vars(v) for k, v in self.levels.items()}, 'clock': self.clock,
                'next_id': self.next_id, 'rng': self.rng.bit_generator.state}

    def load(self, st: dict) -> None:
        self.levels = {int(k): LevelRecord(**v) for k, v in st['levels'].items()}
        self.clock, self.next_id = st['clock'], st['next_id']
        self.rng.bit_generator.state = st['rng']


def positive_value_loss(adv) -> float:
    """Robust PLR's level score: mean of max(advantage, 0) over the episode."""
    a = np.asarray(adv, dtype=np.float64)
    return float(np.maximum(a, 0.0).mean()) if a.size else 0.0
