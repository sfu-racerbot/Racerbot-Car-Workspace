"""The explorer's decisions, with no ROS: what gap follow setting to try next,
when to back off, and what the session has covered so far.

The explorer never drives. It only changes two of gap_follow's live-tunable
settings, ``max_speed`` and ``max_lateral_accel``, the same way the web
dashboard's tuning sliders do. gap_follow keeps every one of its own safety
layers (LB deadman, TTC braking, swept clearance, emergency stop), so a
crashed or confused explorer leaves the car exactly as safe as plain
gap_follow with the last setting applied.

What it does (docs/explorer.md):

* **Steps up gently.** It starts low, and after each stretch of clean driving
  raises one setting by one notch, alternating speed and cornering. A raise
  is applied only when the car is driving straight.
* **No cornering or speed cap of its own.** It keeps raising until the car
  shows the limit. The only ends are gap_follow's own slider maxima (5.0 m/s,
  4.0 m/s^2), which the real car has never reached (VESC top speed ~5 m/s;
  achieved cornering peaked near 3.7 m/s^2 on 2026-10-30).
* **Backs off on its own** when the car shows the limit: it starts to slide,
  gap_follow has to emergency-brake, or the speed-aware wall check says the
  car is within its reaction distance (speed x braking delay) plus
  ``wall_margin`` of something on its current path. Backing off is applied
  at once, never waits for a straight.
* **Learns the course first.** With a pose (SLAM's ``map`` frame on the car,
  the true pose in the simulator) it drives one gentle lap and remembers how
  much straight room there is at each place and heading (``CourseMap``).
  Stop tests then go only where the map *and* the live scan both show room,
  and each place is used once, so the tests spread round the course.
* **Collects what the fit needs, and keeps going until it has it.** A ledger
  counts the seconds of driving that teach each number: cornering near the
  limit (grip), braking harder than the speed loop's linear range
  (braking), and fast steering (steering delay). A stop test runs whenever
  braking is the number short of data. The session ends only when both
  settings have found their limit *and* every number has its seconds.
* **Notices a floor that changes.** A second slide below the limit already
  found lowers cornering again and is logged as a possible floor change.
* **Records what it covered**, so the fitter knows which speeds and corner
  loads the session filled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

# gap_follow's own slider ranges (src/gap_follow/gap_follow/live_tuning.py).
# The explorer refuses anything outside them rather than send a value
# gap_follow would refuse mid-session.
GAP_FOLLOW_MAX_SPEED = (0.3, 5.0)
GAP_FOLLOW_MAX_LATERAL = (0.3, 4.0)

# gap_follow /drive_intent states that mean "the car got too close and a
# safety layer took over". Any of them during a stretch means the setting
# was too much.
SAFETY_STATES = frozenset({'ttc_brake', 'ttc_escape', 'emergency_clearance'})

SPEED, LATERAL, MIN_SPEED = 'max_speed', 'max_lateral_accel', 'min_speed'

# What each ledger entry teaches the fit (docs/explorer.md).
INFO = ('grip', 'braking', 'steering')


class ConfigError(ValueError):
    pass


def _require(ok, message):
    if not ok:
        raise ConfigError(message)


def _finite(*values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


@dataclass(frozen=True)
class Fence:
    """Where the ladder starts and how big each step is.

    ``*_top`` default to gap_follow's own maxima: the explorer adds no cap
    (club decision 2026-11-28). The car's limit is what it is measuring.
    """
    speed_start: float = 2.0
    speed_step: float = 0.5
    speed_top: float = GAP_FOLLOW_MAX_SPEED[1]
    lateral_start: float = 1.5
    lateral_step: float = 0.25
    lateral_top: float = GAP_FOLLOW_MAX_LATERAL[1]
    back_off_steps: int = 2

    def __post_init__(self):
        _require(_finite(self.speed_start, self.speed_step, self.speed_top,
                         self.lateral_start, self.lateral_step, self.lateral_top),
                 'every fence value must be a finite number')
        for name, start, step, top, (low, high) in (
                ('speed', self.speed_start, self.speed_step, self.speed_top, GAP_FOLLOW_MAX_SPEED),
                ('lateral', self.lateral_start, self.lateral_step, self.lateral_top,
                 GAP_FOLLOW_MAX_LATERAL)):
            _require(step > 0, f'{name}_step must be positive, got {step}')
            _require(low <= start <= top <= high,
                     f'{name}: need {low} <= start ({start}) <= top ({top}) <= {high}, '
                     f"gap_follow's own range")
        _require(isinstance(self.back_off_steps, int) and self.back_off_steps >= 1,
                 f'back_off_steps must be a whole number >= 1, got {self.back_off_steps}')


@dataclass
class Ladder:
    """Two settings, each on its own ladder of notches."""
    fence: Fence = field(default_factory=Fence)
    speed_level: int = 0
    lateral_level: int = 0
    speed_frozen: bool = False
    lateral_frozen: bool = False
    last_raised: str | None = None
    limits: dict = field(default_factory=dict)  # axis -> value at which it failed

    def _value(self, axis, level):
        f = self.fence
        start, step, top = ((f.speed_start, f.speed_step, f.speed_top) if axis == SPEED
                            else (f.lateral_start, f.lateral_step, f.lateral_top))
        return round(min(start + level * step, top), 6)

    def _top_level(self, axis):
        f = self.fence
        start, step, top = ((f.speed_start, f.speed_step, f.speed_top) if axis == SPEED
                            else (f.lateral_start, f.lateral_step, f.lateral_top))
        return int(math.ceil((top - start) / step - 1e-9))

    def setting(self) -> dict:
        return {SPEED: self._value(SPEED, self.speed_level),
                LATERAL: self._value(LATERAL, self.lateral_level)}

    def _can_raise(self, axis):
        level, frozen = ((self.speed_level, self.speed_frozen) if axis == SPEED
                         else (self.lateral_level, self.lateral_frozen))
        return not frozen and level < self._top_level(axis)

    @property
    def finished(self) -> bool:
        return not self._can_raise(SPEED) and not self._can_raise(LATERAL)

    def raise_one(self) -> dict | None:
        """Raise one notch, alternating axes; None when neither can rise."""
        order = (LATERAL, SPEED) if self.last_raised == SPEED else (SPEED, LATERAL)
        for axis in order:
            if self._can_raise(axis):
                if axis == SPEED:
                    self.speed_level += 1
                else:
                    self.lateral_level += 1
                self.last_raised = axis
                return self.setting()
        return None

    def back_off(self, axis: str | None) -> dict:
        """The car found a limit on ``axis`` (None: the last one raised, or both
        if nothing was raised yet). Step it down and never raise it again."""
        axes = [axis] if axis else ([self.last_raised] if self.last_raised else [SPEED, LATERAL])
        for a in axes:
            self.limits.setdefault(a, self._value(a, self.speed_level if a == SPEED
                                                  else self.lateral_level))
            if a == SPEED:
                self.speed_level = max(0, self.speed_level - self.fence.back_off_steps)
                self.speed_frozen = True
            else:
                self.lateral_level = max(0, self.lateral_level - self.fence.back_off_steps)
                self.lateral_frozen = True
        self.last_raised = None
        return self.setting()


def check_status_topic(topic: str) -> None:
    """The one topic this node publishes must never be mistaken for a drive input."""
    if not (isinstance(topic, str) and topic.startswith('/') and len(topic) > 1):
        raise ConfigError(f'status_topic must be an absolute topic name, got {topic!r}')
    last = topic.rstrip('/').rsplit('/', 1)[-1]
    if last in ('drive', 'teleop') or topic == '/ackermann_cmd' or topic.startswith('/commands/'):
        raise ConfigError(f'status_topic {topic!r} is named like a drive command topic')


# --- the speed-aware wall check -------------------------------------------------------

def stopping_distance(speed: float, decel: float, delay: float) -> float:
    """Distance to stop from ``speed``: ``delay`` seconds before the brakes bite,
    then a constant ``decel``."""
    speed = max(0.0, speed)
    return speed * delay + speed * speed / (2.0 * decel)


def path_stop_margin(ranges, angle_min, angle_increment, *, speed, steering, wheelbase,
                     half_width, lidar_x, front_reach, delay, max_range=30.0) -> float:
    """Room left on the car's current path after it covers its reaction distance.

    Each beam's hit is placed in the rear-axle frame; a hit within
    ``half_width`` of the arc the car is on counts, at its distance along the
    arc minus ``front_reach`` (the body ahead of the rear axle). The result is
    the smallest of those distances minus ``speed * delay``, the distance the
    car covers before any braking bites: under ``wall_margin`` means it is
    about to touch whatever gap_follow does next. ``inf`` means nothing is on
    its path. Non-finite, zero, negative and out-of-range beams are ignored.

    Why not the full stopping distance: gap_follow turns before a wall rather
    than stopping for it. Measured in the simulator on the 2026-11-11 course
    (explore_sim.py), a full-stop check backed off at 2.5 m/s on ordinary
    corner approaches, so the explorer could never look for the limit.
    """
    _require(_finite(angle_min, angle_increment, speed, steering, wheelbase, half_width,
                     lidar_x, front_reach, delay),
             'path_stop_margin needs finite numbers')
    _require(wheelbase > 0 and delay >= 0 and half_width > 0,
             'wheelbase and half_width must be positive, delay non-negative')
    reaction = max(0.0, speed) * delay
    k = math.tan(steering) / wheelbase
    flip = -1.0 if k < 0 else 1.0
    k = abs(k)
    nearest = math.inf
    for i, r in enumerate(ranges):
        if not (isinstance(r, (int, float)) and math.isfinite(r)) or r <= 0.0 or r > max_range:
            continue
        a = angle_min + i * angle_increment
        x = lidar_x + r * math.cos(a)
        y = flip * r * math.sin(a)
        if k < 1e-6:
            if x <= 0 or abs(y) > half_width:
                continue
            along = x
        else:
            radius = 1.0 / k
            if abs(math.hypot(x, y - radius) - radius) > half_width:
                continue
            theta = math.atan2(x, radius - y)
            if theta <= 0:
                continue
            along = radius * theta
        nearest = min(nearest, along - front_reach)
    return nearest - reaction


def straight_room(ranges, angle_min, angle_increment, *, half_width, lidar_x, front_reach,
                  max_range=30.0) -> float:
    """Room straight ahead of the body, with no reaction distance: what the
    course map remembers for each place."""
    return path_stop_margin(ranges, angle_min, angle_increment, speed=0.0, steering=0.0,
                            wheelbase=1.0, half_width=half_width, lidar_x=lidar_x,
                            front_reach=front_reach, delay=0.0, max_range=max_range)


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass
class CourseMap:
    """What the explorer learns about the course: the least straight room it
    has seen ahead at each place and heading, and how many laps it has done.

    Places are ``cell`` m squares, headings ``sectors`` slices of a turn, in
    whatever fixed frame the pose comes in (SLAM's ``map`` on the car). A lap
    closes when the car comes back within ``close_radius`` of where it
    started, heading the same way, after ``min_lap_m`` of driving away.
    """
    cell: float = 0.5
    sectors: int = 8
    close_radius: float = 1.0
    min_lap_m: float = 10.0
    room: dict = field(default_factory=dict)
    tested: set = field(default_factory=set)
    laps: int = 0
    _start: tuple | None = None
    _last: tuple | None = None
    _travelled: float = 0.0
    _away: bool = False

    def key(self, x, y, yaw):
        sector = int(((yaw % (2 * math.pi)) / (2 * math.pi)) * self.sectors) % self.sectors
        return (math.floor(x / self.cell), math.floor(y / self.cell), sector)

    @property
    def learned(self) -> bool:
        return self.laps >= 1

    def add(self, pose, room) -> None:
        if pose is None or len(pose) != 3 or not _finite(*pose):
            return
        x, y, yaw = (float(v) for v in pose)
        if self._start is None:
            self._start = (x, y, yaw)
        if self._last is not None:
            self._travelled += math.hypot(x - self._last[0], y - self._last[1])
        self._last = (x, y)
        if isinstance(room, (int, float)) and not math.isnan(room):
            k = self.key(x, y, yaw)
            self.room[k] = min(self.room.get(k, math.inf), float(room))
        sx, sy, syaw = self._start
        near = (math.hypot(x - sx, y - sy) <= self.close_radius
                and abs(_wrap(yaw - syaw)) < math.pi / 4)
        if not near and self._travelled >= self.min_lap_m:
            self._away = True
        elif near and self._away:
            self.laps += 1
            self._away = False
            self._travelled = 0.0

    def room_at(self, pose):
        """The least straight room seen here, or None for a place never seen."""
        if pose is None or len(pose) != 3 or not _finite(*pose):
            return None
        return self.room.get(self.key(*pose))

    def report(self) -> dict:
        return {'laps': self.laps, 'learned': self.learned, 'places': len(self.room),
                'places_tested': len(self.tested)}


@dataclass
class Ledger:
    """Seconds of driving that teach each number the fit looks for."""
    need: dict
    seconds: dict = field(default_factory=lambda: {k: 0.0 for k in INFO})

    def add(self, name, dt):
        self.seconds[name] += dt

    def short(self) -> list:
        """Numbers still short of data, the furthest behind first."""
        behind = [(self.seconds[k] / self.need[k], k) for k in INFO
                  if self.need[k] > 0 and self.seconds[k] < self.need[k]]
        return [k for _, k in sorted(behind)]

    def report(self) -> dict:
        return {k: {'seconds': round(self.seconds[k], 2), 'need': self.need[k]} for k in INFO}


# --- the conductor ----------------------------------------------------------------------

@dataclass(frozen=True)
class Rules:
    segment_sec: float = 15.0         # clean LB-held driving before the next raise
    straight_steer: float = 0.05      # rad: a raise waits until |steering| is below this
    moving_speed: float = 0.3         # m/s: below this the car counts as stopped
    wall_margin: float = 0.03         # m left after a predicted stop; the user's tight setting
    slide_ratio: float = 0.7          # turning less than this share of what the steering asks
    slide_min_speed: float = 1.5      # m/s
    slide_min_steer: float = 0.08     # rad
    slide_hold_sec: float = 0.2       # how long the shortfall must last
    slide_steer_window: float = 0.3   # s: steering must have been held this long (servo lag)
    event_cooldown_sec: float = 3.0   # s after a back-off before another limit sign counts
    wheelbase: float = 0.325          # m, measured on car 2 2026-11-23 (hub to hub)
    min_voltage: float = 10.5         # V: 3S pack; below it acceleration changes, end the session
    brake_delay: float = 0.24         # s before braking bites (car 2, braking_lag.py, run 20261111-152132)
    brake_min_speed: float = 2.5      # m/s needed to make a stop test worth having
    brake_clear: float = 1.0          # m of spare room needed beyond the braking distance
    brake_decel: float = 2.9          # m/s^2, the braking assumed when checking that room
    brake_speed: float = 0.5          # m/s: max_speed and min_speed during a stop test
    brake_hold_sec: float = 3.0       # s: a stop test ends by then at the latest
    brake_gap_sec: float = 20.0       # s of driving between stop tests
    learn_timeout_sec: float = 90.0   # s: no lap closed by then, carry on with the scan alone
    grip_info_ratio: float = 0.8      # cornering at this share of the limit teaches grip
    brake_info_excess: float = 1.0    # m/s above the command teaches braking
    steer_info_rate: float = 1.0      # rad/s of steering change teaches steering delay
    need_grip_sec: float = 20.0       # seconds of each the fit needs (explore_sim.py sweep)
    need_braking_sec: float = 4.0
    need_steering_sec: float = 15.0
    max_session_sec: float = 900.0    # LB-held driving time

    def __post_init__(self):
        values = [getattr(self, n) for n in self.__dataclass_fields__]
        _require(_finite(*values), 'every rule must be a finite number')
        _require(self.segment_sec > 0 and self.max_session_sec > 0, 'times must be positive')
        _require(self.wall_margin >= 0, 'wall_margin must be >= 0')
        _require(0 < self.slide_ratio < 1, 'slide_ratio must be between 0 and 1')
        _require(self.wheelbase > 0, 'wheelbase must be positive')
        _require(self.brake_decel > 0, 'brake_decel must be positive')
        _require(self.brake_delay >= 0, 'brake_delay must be >= 0')
        _require(GAP_FOLLOW_MAX_SPEED[0] <= self.brake_speed < self.brake_min_speed,
                 'brake_speed must be inside gap_follow range and below brake_min_speed')
        _require(self.brake_hold_sec > 0 and self.brake_gap_sec >= 0 and self.learn_timeout_sec >= 0,
                 'stop-test and learning times must be positive')
        _require(0 < self.grip_info_ratio <= 1, 'grip_info_ratio must be in (0, 1]')
        _require(min(self.need_grip_sec, self.need_braking_sec, self.need_steering_sec) >= 0,
                 'the seconds each number needs must be >= 0')


@dataclass
class Inputs:
    """One look at the car. ``None`` means that input has not arrived (or is stale)."""
    now: float
    lb_held: bool
    speed: float | None = None
    steering: float | None = None
    yaw_rate: float | None = None
    stop_margin: float | None = None       # path_stop_margin of the latest scan
    voltage: float | None = None
    safety_state: str | None = None        # latest gap_follow /drive_intent state
    speed_cmd: float | None = None         # gap_follow's latest speed command
    pose: tuple | None = None              # (x, y, yaw) in a fixed frame (SLAM's map)
    straight_room: float | None = None     # straight_room() of the latest scan


@dataclass
class Coverage:
    """Seconds of driving per (speed, cornering) bin, and the tests done."""
    speed_bin: float = 0.5
    lateral_bin: float = 0.5
    seconds: dict = field(default_factory=dict)
    brake_tests: int = 0
    events: list = field(default_factory=list)

    def add(self, speed, yaw_rate, dt):
        if speed is None or yaw_rate is None or not _finite(speed, yaw_rate, dt):
            return
        key = (int(abs(speed) // self.speed_bin), int(abs(speed * yaw_rate) // self.lateral_bin))
        self.seconds[key] = self.seconds.get(key, 0.0) + dt

    def report(self) -> dict:
        return {
            'bins': [{'speed_from': s * self.speed_bin, 'lateral_from': a * self.lateral_bin,
                      'seconds': round(sec, 3)} for (s, a), sec in sorted(self.seconds.items())],
            'brake_tests': self.brake_tests,
            'events': list(self.events),
        }


class Conductor:
    """Call ``update`` at a steady rate; apply every setting it returns.

    ``baseline`` holds gap_follow's own values of any other setting a test
    touches (``min_speed``); without it there are no stop tests, because
    gap_follow refuses a ``max_speed`` below its ``min_speed``.
    """

    def __init__(self, fence: Fence | None = None, rules: Rules | None = None,
                 baseline: dict | None = None):
        self.ladder = Ladder(fence or Fence())
        self.rules = rules or Rules()
        self.coverage = Coverage()
        self.course = CourseMap()
        r = self.rules
        self.ledger = Ledger({'grip': r.need_grip_sec, 'braking': r.need_braking_sec,
                              'steering': r.need_steering_sec})
        self.baseline = dict(baseline or {})
        if MIN_SPEED in self.baseline:
            _require(_finite(self.baseline[MIN_SPEED]) and self.baseline[MIN_SPEED] >= 0,
                     'baseline min_speed must be a finite number >= 0')
        self.applied: dict | None = None
        self.pending: dict | None = None
        self.driving_sec = 0.0
        self.segment_sec = 0.0
        self.raises = 0
        self.brake_until: float | None = None
        self.since_brake = 0.0
        self.done_reason: str | None = None
        self._last_now: float | None = None
        self._last_steer: tuple | None = None
        self._slide_since: float | None = None
        self._steer_history: list = []   # (now, steering) over the last slide_steer_window
        self._quiet_until = -math.inf
        self._pose_seen = False

    @property
    def done(self) -> bool:
        return self.done_reason is not None

    @property
    def can_stop_test(self) -> bool:
        return MIN_SPEED in self.baseline

    def _out(self, setting: dict, testing: bool = False) -> dict:
        """A setting to send: the ladder's two values, and min_speed put back
        to gap_follow's own (or lowered for a stop test)."""
        out = dict(setting)
        if self.can_stop_test:
            out[MIN_SPEED] = (min(self.baseline[MIN_SPEED], self.rules.brake_speed) if testing
                              else self.baseline[MIN_SPEED])
        return out

    def start(self) -> dict:
        """The first setting, to apply before the car moves."""
        self.applied = self.ladder.setting()
        return self._out(self.applied)

    def _event(self, now, kind, axis, detail):
        self._quiet_until = now + self.rules.event_cooldown_sec
        self._slide_since = None
        if axis is not None and axis in self.ladder.limits:
            detail += (f'; below the limit already found ({self.ladder.limits[axis]}): '
                       'the floor may have changed')
        self.coverage.events.append({'t': round(now, 3), 'kind': kind, 'detail': detail,
                                     'setting': dict(self.applied or {})})
        backed = self.ladder.back_off(axis)
        self.segment_sec = 0.0
        self.brake_until = None
        self.pending = None
        self.applied = backed
        return self._out(backed)

    def _slide(self, now, detail):
        """Blame a slide on what has been rising since the last one. The first
        slide finds the cornering limit; once cornering is held, only speed is
        still being raised, so a later slide lowers speed. Sliding with both at
        their lowest notch ends the session: there is nothing left to lower."""
        lad = self.ladder
        if not lad.lateral_frozen:
            axis = LATERAL
        elif lad.speed_level > 0:
            axis = SPEED
        elif lad.lateral_level > 0:
            axis = LATERAL
        else:
            self.done_reason = 'still sliding at the gentlest settings'
            self.coverage.events.append({'t': round(now, 3), 'kind': 'slide',
                                         'detail': f'{detail}; {self.done_reason}',
                                         'setting': dict(self.applied)})
            if self.brake_until is not None:        # put gap_follow's own min_speed back
                self.brake_until = None
                self.applied = lad.setting()
                return self._out(self.applied)
            return None
        if axis == SPEED:
            detail += '; cornering is already held, so speed comes down'
        return self._event(now, 'slide', axis, detail)

    @property
    def learning(self) -> bool:
        """Driving the first, gentle lap that maps the course."""
        return (self._pose_seen and not self.course.learned
                and self.driving_sec < self.rules.learn_timeout_sec)

    def _count_information(self, x: Inputs, dt) -> None:
        r = self.rules
        steer_rate = None
        if x.steering is not None and _finite(x.steering):
            if self._last_steer is not None and x.now > self._last_steer[0]:
                steer_rate = abs(x.steering - self._last_steer[1]) / (x.now - self._last_steer[0])
            self._last_steer = (x.now, x.steering)
        else:
            self._last_steer = None
        lateral_ref = self.ladder.limits.get(LATERAL, self.applied[LATERAL])
        if (x.yaw_rate is not None and _finite(x.yaw_rate) and abs(x.speed) >= 1.0
                and abs(x.speed * x.yaw_rate) >= r.grip_info_ratio * lateral_ref):
            self.ledger.add('grip', dt)
        if (x.speed_cmd is not None and _finite(x.speed_cmd)
                and x.speed - x.speed_cmd >= r.brake_info_excess):
            self.ledger.add('braking', dt)
        if steer_rate is not None and steer_rate >= r.steer_info_rate and abs(x.speed) >= 1.5:
            self.ledger.add('steering', dt)

    def _stop_test_room(self, x: Inputs) -> bool:
        """Both the live scan and, once learned, the course map show room to
        stop from here, and this place has not had a stop test yet."""
        r = self.rules
        need = x.speed ** 2 / (2 * r.brake_decel) + r.brake_clear
        if x.stop_margin is None or not _finite(x.stop_margin) or x.stop_margin < need:
            return False
        if not self.course.learned:
            return True
        # The map remembers room with no reaction distance taken off.
        mapped = self.course.room_at(x.pose)
        if mapped is None or mapped - x.speed * r.brake_delay < need:
            return False
        return self.course.key(*x.pose) not in self.course.tested

    def update(self, x: Inputs) -> dict | None:
        """The setting to apply now, or None to leave gap_follow as it is."""
        if self.applied is None:
            raise RuntimeError('call start() first')
        dt = 0.0 if self._last_now is None else max(0.0, x.now - self._last_now)
        self._last_now = x.now
        if self.done:
            return None
        r = self.rules

        if x.voltage is not None and _finite(x.voltage) and x.voltage < r.min_voltage:
            self.done_reason = f'battery {x.voltage:.2f} V below {r.min_voltage} V'
            self.coverage.events.append({'t': round(x.now, 3), 'kind': 'battery',
                                         'detail': self.done_reason, 'setting': dict(self.applied)})
            was_testing = self.brake_until is not None
            self.brake_until = None
            if was_testing or self.applied[SPEED] != self.ladder.setting()[SPEED]:
                self.applied = self.ladder.setting()
                return self._out(self.applied)
            return None

        moving = (x.lb_held and x.speed is not None and _finite(x.speed)
                  and abs(x.speed) >= r.moving_speed)

        # The limit signs, only while the explorer is actually driving the car
        # (LB held, moving). Each backs off at once, wherever the car is.
        self._remember_steering(x)
        if moving and x.now >= self._quiet_until:
            if x.safety_state in SAFETY_STATES:
                return self._event(x.now, 'gap_follow_safety', None, x.safety_state)
            if (x.stop_margin is not None and _finite(x.stop_margin)
                    and x.stop_margin < r.wall_margin):
                return self._event(x.now, 'wall', None,
                                   f'{x.stop_margin:.3f} m left after the reaction distance, '
                                   f'under {r.wall_margin} m')
            if self._sliding(x):
                if self._slide_since is None:
                    self._slide_since = x.now
                if x.now - self._slide_since >= r.slide_hold_sec:
                    self._slide_since = None
                    return self._slide(x.now, f'turning under {r.slide_ratio:.0%} '
                                              'of what the steering asks')
            else:
                self._slide_since = None

        if not moving:
            self._slide_since = None
            self._last_steer = None
            # Stopped or LB released: a waiting raise can go in now, nothing else moves on.
            if self.pending is not None and not x.lb_held:
                self.applied, self.pending, self.segment_sec = self.pending, None, 0.0
                return self._out(self.applied)
            return None

        if x.pose is not None:
            self._pose_seen = True
            self.course.add(x.pose, x.straight_room)
        learning = self.learning
        self.driving_sec += dt
        self.since_brake += dt
        if not learning:
            self.segment_sec += dt
        self.coverage.add(x.speed, x.yaw_rate, dt)
        self._count_information(x, dt)

        straight = x.steering is not None and _finite(x.steering) and abs(x.steering) < r.straight_steer

        if self.brake_until is not None:
            slow = x.speed <= r.brake_speed + 0.3
            if x.now >= self.brake_until or slow:
                self.brake_until = None
                self.since_brake = 0.0
                self.coverage.brake_tests += 1
                return self._out(self.applied)   # back to the ladder's speed and min_speed
            return None

        if self.driving_sec >= r.max_session_sec:
            self.done_reason = f'session time {r.max_session_sec:.0f} s reached'
            return None
        if learning:
            return None

        short = self.ledger.short()
        if (self.can_stop_test and short and short[0] == 'braking'
                and self.since_brake >= r.brake_gap_sec and straight
                and x.speed >= r.brake_min_speed and self._stop_test_room(x)):
            self.brake_until = x.now + r.brake_hold_sec
            if self.course.learned:
                self.course.tested.add(self.course.key(*x.pose))
            return self._out({SPEED: r.brake_speed, LATERAL: self.applied[LATERAL]}, testing=True)

        if self.pending is None and self.segment_sec >= r.segment_sec:
            self.segment_sec = 0.0
            nxt = self.ladder.raise_one()
            if nxt is None:
                if self.ladder.finished and not short:
                    self.done_reason = 'both settings at their limit and every number has its data'
                return None
            self.raises += 1
            self.pending = nxt

        if self.pending is not None and straight:
            # The clean stretch for the next raise counts from when this one bites.
            self.applied, self.pending, self.segment_sec = self.pending, None, 0.0
            return self._out(self.applied)
        return None

    def _remember_steering(self, x: Inputs) -> None:
        if x.steering is None or not _finite(x.steering):
            self._steer_history.clear()
            return
        self._steer_history.append((x.now, x.steering))
        cutoff = x.now - self.rules.slide_steer_window - 1e-6  # keep the sample exactly at the edge
        while self._steer_history and self._steer_history[0][0] < cutoff:
            self._steer_history.pop(0)

    def _held_steering(self) -> float | None:
        """The steering the wheels have certainly reached: the smallest angle
        asked for over the last ``slide_steer_window``, if it kept one sign.
        The servo lags the command, so a fresh swing is not yet a turn."""
        h = self._steer_history
        if not h or h[-1][0] - h[0][0] < self.rules.slide_steer_window - 1e-9:
            return None
        values = [s for _, s in h]
        if min(values) > 0:
            return min(values)
        if max(values) < 0:
            return max(values)
        return None

    def _sliding(self, x: Inputs) -> bool:
        r = self.rules
        if x.yaw_rate is None or x.speed is None or not _finite(x.yaw_rate, x.speed):
            return False
        held = self._held_steering()
        if held is None or abs(x.speed) < r.slide_min_speed or abs(held) < r.slide_min_steer:
            return False
        expected = x.speed * math.tan(held) / r.wheelbase
        # Turning the wrong way, or much less than asked, is a slide.
        return x.yaw_rate * expected <= 0 or abs(x.yaw_rate) < r.slide_ratio * abs(expected)

    def report(self) -> dict:
        out = self.coverage.report()
        out.update({'setting': dict(self.applied or {}), 'limits': dict(self.ladder.limits),
                    'driving_sec': round(self.driving_sec, 3), 'raises': self.raises,
                    'done': self.done_reason, 'course': self.course.report(),
                    'information': self.ledger.report(), 'stop_tests': self.can_stop_test})
        return out
