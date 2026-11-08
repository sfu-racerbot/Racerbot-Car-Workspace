"""Measured pass-clearance contract, approved by the tech lead 2026-11-08.

No ROS or dead reckoning. A visible surface may be anywhere on the opponent:
extend it by a whole opponent length before declaring ego's rear clear.
"""
import math


def measured_body_lead(ego_arc, surface_arc, total_length, car_length,
                       wheelbase, uncertainty):
    """Conservative rear-of-ego to front-of-opponent lead along the line."""
    values = (ego_arc, surface_arc, total_length, car_length, wheelbase, uncertainty)
    if not all(math.isfinite(v) for v in values):
        return -math.inf
    if min(total_length, uncertainty) < 0 or min(car_length, wheelbase) <= 0:
        return -math.inf
    lead = ego_arc - surface_arc
    if total_length > 0:
        lead = (lead + total_length / 2) % total_length - total_length / 2
    rear_offset = (wheelbase - car_length) / 2
    return lead + rear_offset - car_length - uncertainty


class CompletionGate:
    """Arm above margin+hysteresis, hold above margin with live measurements."""

    def __init__(self, clear_margin=1.0, hold_sec=0.5, hysteresis=0.1):
        if (not all(math.isfinite(v) for v in (clear_margin, hold_sec, hysteresis))
                or min(clear_margin, hysteresis) < 0 or hold_sec <= 0):
            raise ValueError('completion margins must be nonnegative and hold time positive and finite')
        self.clear_margin = clear_margin
        self.hold_sec = hold_sec
        self.hysteresis = hysteresis
        self.since = None
        self.last_time = None

    def observe(self, now, body_lead, measured):
        if (measured is not True or not math.isfinite(now) or not math.isfinite(body_lead)
                or (self.last_time is not None and now < self.last_time)
                or body_lead < self.clear_margin):
            self.since = None
            self.last_time = now if math.isfinite(now) else None
            return False
        self.last_time = now
        if self.since is None and body_lead >= self.clear_margin + self.hysteresis:
            self.since = now
        # Match ROS clock resolution without making a shorter hold meaningful.
        return self.since is not None and now - self.since >= self.hold_sec - 1e-9
