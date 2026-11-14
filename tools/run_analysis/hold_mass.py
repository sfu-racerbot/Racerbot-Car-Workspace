"""Hold and memory tick mass per track, by the mover_ticks the track had (opponent_follow_hold_min_mover_ticks).

``collect(ctrl)`` after each ``ctrl.command``: (track id, kind, mover_ticks) for every track whose stopped-car hold (kind
"held": the track is in the 1 s stopped_release hold, relevant or not) is active this tick, plus the track whose cap the
FollowCapMemory is holding past its own tick (kind "memory"). Run with the gate on and follow_hold_min_mover_ticks 0, so
the hold and the memory are unrestricted and the table says what an N would remove: ``removed(events, N)``.
"""
from __future__ import annotations

import math


def collect(ctrl, last):
    """``last`` (a dict kept by the caller) remembers each live track's mover_ticks, for a memory whose track is gone."""
    o = ctrl.opponents
    if o is None:
        return []
    out = []
    for t in o.tracker.tracks:
        last[t.track_id] = t.mover_ticks
    for t in o.tracker.tracks:
        if (not t.moving and t.confirmed_car and t.stopped_sec <= o.params.stopped_release_sec + 1e-12):
            bx, by = o.ego.to_body(t.x, t.y)
            if not (bx <= -o.params.radius or abs(by) > o.params.relevant_lateral
                    or math.hypot(bx, by) > o.params.max_range):
                out.append((t.track_id, "held", t.mover_ticks))
    m = o.follow_memory
    if m._cap_t is not None and m._last_t is not None and m._last_t > m._cap_t and m._source is not None:
        out.append((m._source, "memory", last.get(m._source, -1)))
    return out


def removed(events, n):
    """Fraction of the (tick, track, kind) events an N would remove (mover_ticks < N), and the counts."""
    total = len(events)
    gone = sum(1 for _tid, _k, mv in events if mv < n)
    return (gone / total if total else math.nan), gone, total
