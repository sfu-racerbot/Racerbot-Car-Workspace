"""Lookahead-only experiment: circular interpolation and refusal of unsafe winners."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.extend(map(str, (ROOT / 'tools/f1tenth_sim', ROOT / 'src/pure_pursuit')))
from pure_pursuit.tracking_laws import AdaptivePurePursuitParams, AdaptivePurePursuitLaw
from sector_lookahead import OFFSETS, SectorLookahead, select_schedule


def test_schedule_blends_across_lap_seam_and_preserves_original_bounds():
    p = AdaptivePurePursuitParams()
    base = AdaptivePurePursuitLaw(None, p)
    model = SectorLookahead(base, [-.1, 0., 0., 0., 0., .1])
    # Zero curvature gives geometric base 0.8 + 0.1*speed, per configured law.
    path = SimpleNamespace(cumulative=np.arange(6.), seg_len=np.ones(6),
                           curvature=np.zeros(6), closed=True)
    for arc, expected in [(0., .95), (.5, .85), (5.5, 1.05), (6., .95)]:
        assert model.lookahead(1.5, arc, path) == pytest.approx(expected, abs=1e-12)
    assert model.lookahead(0., .5, path) == p.min_lookahead
    assert model.lookahead(10., 5.5, path) == p.max_lookahead
    with pytest.raises(ValueError, match='six offsets'):
        SectorLookahead(base, [float('nan')] * 6)


def test_selection_rejects_contact_dnf_and_clearance_loss_before_sector_time():
    rows = []
    for seed in (41, 42):
        for o in OFFSETS:
            secs = [dict(min_wall=.1 - (.01 if o > 0 and j == 1 else 0.),
                         times=[2. - o * (1 if j < 2 else -1)]) for j in range(6)]
            rows.append(dict(seed=seed, offset=o, contacts=0, dnf=False,
                             failed_start=False, sectors=secs))
    assert select_schedule(rows) == [.1, 0., -.1, -.1, -.1, -.1]
    rows[-1]['failed_start'] = True
    assert select_schedule(rows) == [0., 0., -.1, -.1, -.1, -.1]
    rows[-1]['failed_start'] = False
    rows[-1]['contacts'] = 1   # fastest probe hit a wall on the second seed
    rows[-2]['dnf'] = True    # other nonzero probe didn't finish
    assert select_schedule(rows) == [0.] * 6
    with pytest.raises(ValueError, match='incomplete training grid'):
        select_schedule(rows[:-1])
