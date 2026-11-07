"""ROS-free temporal and motion checks for opponent detections."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GateVerdict:
    confirmed: bool
    reason: str


class OpponentGate:
    """Confirms a detected cluster is a car before anything races against it."""

    def __init__(self, confirm_count=6, window_sec=0.3, max_jump_m=0.5,
                 max_reverse_speed=0.5, max_speed=8.0):
        for name, value in locals().copy().items():
            if name == 'self':
                continue
            if (not math.isfinite(value)
                    or (value < 0 if name == 'max_reverse_speed' else value <= 0)
                    or (name == 'confirm_count' and int(value) != value)):
                raise ValueError(f'{name} has an invalid value')
            setattr(self, name, value)
        self.reset()

    def reset(self):
        self._history = []

    def observe(self, now_sec, x, y, arc_length, ego_arc_length, total_length):
        if not all(math.isfinite(v) for v in
                   (now_sec, x, y, arc_length, ego_arc_length, total_length)):
            self.reset()
            return GateVerdict(False, 'non-finite opponent observation')
        if self._history:
            previous = self._history[-1]
            if (now_sec < previous[0]
                    or math.hypot(x - previous[1], y - previous[2]) > self.max_jump_m):
                self.reset()
        self._history = [obs for obs in self._history
                         if now_sec - obs[0] <= self.window_sec]
        self._history.append((now_sec, x, y, arc_length))
        first = self._history[0]
        span = now_sec - first[0]
        if span > 0:
            delta = arc_length - first[3]
            if total_length > 0:
                delta = (delta + total_length / 2) % total_length - total_length / 2
            speed = -delta / span
            if speed < -self.max_reverse_speed:
                self.reset()
                return GateVerdict(False, f'moved backwards along the track at {-speed:.1f} m/s')
            if speed > self.max_speed:
                self.reset()
                return GateVerdict(False, f'moved along the track too fast at {speed:.1f} m/s')
        if len(self._history) < self.confirm_count:
            return GateVerdict(False, f'seen {len(self._history)} of {self.confirm_count} needed')
        if span < 0.5 * self.window_sec:
            return GateVerdict(False, f'observed for {span:.3f}s; need {0.5 * self.window_sec:.3f}s')
        return GateVerdict(True, 'opponent confirmed')
