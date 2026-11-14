"""Memory-bound ticks by the state of the track that set the FollowCapMemory's cap (rule M, docs/gap-follow.md#static-confirmation).

``collect(ctrl)`` after each ``ctrl.command`` returns (track id, class) when the memory is holding a cap past the tick it was
stored on, with class
  G gone: no track with that id;  C coasting: the track exists, misses > 0;
  S seen-not-moving: misses == 0 and the track is not in moving_body();
  X seen-static-confirmed: misses == 0, moving, and static-confirmed.
Rule M clears the memory on S and X. Run with the gate on and rule M off (the study's shipped code before M) to measure.
"""
from __future__ import annotations


def collect(ctrl):
    o = ctrl.opponents
    if o is None:
        return []
    m = o.follow_memory
    if not (m._cap_t is not None and m._last_t is not None and m._last_t > m._cap_t and m._source is not None):
        return []
    track = next((t for t in o.tracker.tracks if t.track_id == m._source), None)
    if track is None:
        return [(m._source, "G")]
    if track.misses > 0:
        return [(m._source, "C")]
    if track.moving and track.static_confirmed:
        return [(m._source, "X")]
    moving_ids = {o._track_id_of(b) for b in o.moving_body()}
    return [(m._source, "S" if track.track_id not in moving_ids else "M")]    # M: seen and a live mover (memory not needed)


def table(events):
    """{class: count}."""
    out = {}
    for _tid, cls in events:
        out[cls] = out.get(cls, 0) + 1
    return out


def removed_fraction(events):
    """Fraction of the memory-bound ticks rule M would clear (classes S and X)."""
    total = len(events)
    return (sum(c in ("S", "X") for _t, c in events) / total) if total else float("nan")
