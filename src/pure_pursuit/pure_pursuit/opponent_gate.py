"""Is this detected cluster really a car? Checks over time, before anything races against it. No ROS.

Why it exists: on car 2's solo run 20261106-175300, pure_pursuit_node's map subtraction reported a "car" about
3 m ahead and the node committed to an overtake on the spot. Nothing was there: the particle filter's pose was
0.8-2.2 m wrong along a straight, so map subtraction ray-cast the walls from the wrong place and read a real
wall as something closer than the map. The pass never ended (the phantom kept reappearing ahead), the car held
its sideways offset toward the wall, and body_contact latched it until LB was released.

Replaying that run's scans and filter poses showed what separates it from a car: the phantom is a *fixed spot
in the map, 0.05-0.25 m from a mapped wall*. A car's centre on a track this car can race sits further out
(half a car's width, 0.155 m, plus the gap it keeps). So a detection has to be:

    away from the walls   at least min_wall_distance from the nearest mapped wall
    seen consistently     confirm_count sightings within window_sec, spanning min_span_sec, none jumping max_jump_m
    moving like a car     along the track, between -max_reverse_speed and max_speed

max_reverse_speed is 1.5 m/s rather than a car's true zero because the along-track position comes from the
nearest waypoint of a line spaced 0.15 m apart: one index of jitter over the 0.15 s minimum span reads 1 m/s.

This only decides what the node may *overtake*. The reactive safety net (avoidance, emergency stop, body contact)
reads the scan directly and is not behind this gate, so an unconfirmed object is still never driven into blind.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GateVerdict:
    confirmed: bool
    reason: str


class OpponentGate:
    """Confirms a detected cluster is a car before anything races against it."""

    def __init__(self, confirm_count=6, window_sec=0.3, min_span_sec=0.15, max_jump_m=0.5,
                 max_reverse_speed=1.5, max_speed=8.0, min_wall_distance=0.30):
        for name, value in (('confirm_count', confirm_count), ('window_sec', window_sec),
                            ('min_span_sec', min_span_sec), ('max_jump_m', max_jump_m),
                            ('max_reverse_speed', max_reverse_speed), ('max_speed', max_speed),
                            ('min_wall_distance', min_wall_distance)):
            may_be_zero = name in ('min_span_sec', 'max_reverse_speed', 'min_wall_distance')
            if (not math.isfinite(value) or value < 0.0 or (value == 0.0 and not may_be_zero)
                    or (name == 'confirm_count' and int(value) != value)):
                raise ValueError(f'{name} must be a positive finite number, got {value!r}')
        if min_span_sec > window_sec:
            raise ValueError(f'min_span_sec ({min_span_sec}) cannot exceed window_sec ({window_sec})')
        self.confirm_count = int(confirm_count)
        self.window_sec = float(window_sec)
        self.min_span_sec = float(min_span_sec)
        self.max_jump_m = float(max_jump_m)
        self.max_reverse_speed = float(max_reverse_speed)
        self.max_speed = float(max_speed)
        self.min_wall_distance = float(min_wall_distance)
        self.reset()

    def reset(self):
        self._history = []

    def observe(self, now_sec, x, y, arc_length, total_length, wall_distance=math.inf) -> GateVerdict:
        """One detection, map frame. `wall_distance` is metres to the nearest mapped wall (inf with no map)."""
        if not all(math.isfinite(v) for v in (now_sec, x, y, arc_length, total_length)) or math.isnan(wall_distance):
            self.reset()
            return GateVerdict(False, 'non-finite opponent observation')
        if wall_distance < self.min_wall_distance:
            self.reset()
            return GateVerdict(False, f'{wall_distance:.2f} m from a mapped wall, under the '
                                      f'{self.min_wall_distance:.2f} m a car would be')
        if self._history:
            previous = self._history[-1]
            if now_sec < previous[0] or math.hypot(x - previous[1], y - previous[2]) > self.max_jump_m:
                self.reset()
        self._history = [obs for obs in self._history if now_sec - obs[0] <= self.window_sec]
        self._history.append((now_sec, x, y, arc_length))
        first = self._history[0]
        span = now_sec - first[0]
        if len(self._history) < self.confirm_count:
            return GateVerdict(False, f'seen {len(self._history)} of {self.confirm_count} needed')
        # Only judged over the full span: one 25 ms step of centroid noise reads as metres per second.
        if span < self.min_span_sec:
            return GateVerdict(False, f'observed for {span:.3f} s, need {self.min_span_sec:.3f} s')
        delta = arc_length - first[3]
        if total_length > 0.0:
            delta = (delta + total_length / 2.0) % total_length - total_length / 2.0
        speed = delta / span if span > 0.0 else 0.0
        if speed < -self.max_reverse_speed:
            self.reset()
            return GateVerdict(False, f'moved backwards along the track at {-speed:.1f} m/s')
        if speed > self.max_speed:
            self.reset()
            return GateVerdict(False, f'moved along the track too fast at {speed:.1f} m/s')
        return GateVerdict(True, 'opponent confirmed')


class WallDistance:
    """Metres from a map point to the nearest mapped wall, from a nav_msgs/OccupancyGrid's fields.

    Unknown cells count as wall, as everywhere else in this package (occupancy_map.from_grid_message), and so
    does anything off the map: a point there gets 0.0, which no detection passes.
    """

    def __init__(self, data, width, height, resolution, origin_x, origin_y, origin_yaw=0.0):
        from pure_pursuit.occupancy_map import OccupancyMap
        if not math.isfinite(origin_yaw) or abs(origin_yaw) > 1e-9:
            raise ValueError(f'map origin yaw must be 0, got {origin_yaw!r}')
        self._map = OccupancyMap.from_grid_message(
            data, width, height, resolution, origin_x, origin_y, inflate_cells=0)
        self._field = self._map.clearance_field()

    def at(self, x, y) -> float:
        if not (math.isfinite(x) and math.isfinite(y)):
            return 0.0
        return float(self._map.clearance_at([x], [y], self._field)[0])

    def at_many(self, xs, ys):
        """Vectorised at(): one clearance per point, 0.0 for any non-finite point."""
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        out = np.zeros(xs.shape, float)
        if np.any(ok):
            out[ok] = np.asarray(self._map.clearance_at(xs[ok], ys[ok], self._field), float)
        return out
