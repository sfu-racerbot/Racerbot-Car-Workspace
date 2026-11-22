"""Simulation-only lookahead schedule; no change to adaptive pursuit's steering geometry."""
import numpy as np
from pure_pursuit.tracking_laws import AdaptivePurePursuitLaw

SECTORS = 6
OFFSETS = (0., -.10, .10)  # metres, always within the original lookahead limits


class SectorLookahead(AdaptivePurePursuitLaw):
    def __init__(self, original, offsets):
        super().__init__(original.limits, original.params)
        self.offsets = np.asarray(offsets, dtype=float)
        if self.offsets.shape != (SECTORS,) or not np.isin(self.offsets, OFFSETS).all():
            raise ValueError('need six offsets from the fixed 0 / +/-0.10 m grid')

    def lookahead(self, speed, arc, path):
        base = super().lookahead(speed, arc, path)
        length = path.cumulative[-1] + path.seg_len[-1]
        # Blend for 10% of a sector either side of its boundary, including the seam.
        station = arc / length * SECTORS + .1
        i = int(np.floor(station))
        u = min(1., (station - i) / .2)
        delta = (1-u)*self.offsets[(i-1) % SECTORS] + u*self.offsets[i % SECTORS]
        return float(np.clip(base + delta, self.params.min_lookahead, self.params.max_lookahead))


def select_schedule(training):
    """Reject incomplete/contact probes. Preserve baseline sector clearance per seed.

    Choose lowest median flying-sector time, breaking ties in favour of zero.
    No admissible probe or missing paired baseline sector: leave that sector unchanged.
    """
    by = {(r['seed'], r['offset']): r for r in training}
    seeds = sorted({r['seed'] for r in training})
    if not seeds or any((s, o) not in by for s in seeds for o in OFFSETS):
        raise ValueError('incomplete training grid')
    selected = []
    for j in range(SECTORS):
        eligible = []
        for o in OFFSETS:
            times = []
            for seed in seeds:
                run = by[seed, o]
                before = by[seed, 0.]['sectors'][j]
                after = run['sectors'][j]
                if (run['dnf'] or run['contacts'] or before is None or after is None
                        or after['min_wall'] + 1e-9 < before['min_wall']):
                    break
                times.extend(after['times'])
            else:
                eligible.append((np.median(times), abs(o), o))
        selected.append(min(eligible)[2] if eligible else 0.)
    return selected
