"""Reverse escape: back out of a nose-in stop over ground the car itself just drove across (pure, no rclpy).

A car stopped with something close ahead, free sides and no forward arc that clears it, commands zero forever
(rb2 20261111-142254, 104.6-118.2 s: 13.6 s parked 0.22 m from a return). No controller here commands negative
speed. The LiDAR sees +-135 degrees: the rear quarters, not the 90 degrees directly behind, so this reverses only
over ground the car has just driven, slowly, a short fixed distance, straight, and stops the moment it cannot see
that the rear quarters are clear or cannot measure how far it has gone.

``step`` is called once per scan tick and returns the speed to command (negative while reversing, 0.0 while
settling) or None for "not mine, let the normal controller command". It never returns a positive speed or a
magnitude above ``speed``. The deadman is an input: released at any moment it returns None and goes idle, so the
node's own deadman path commands zero. Rules (docs/gap-follow.md#reverse-escape):

1. idle -> waiting when blocked ahead, the deadman is held, the measured speed is fresh and |speed| < 0.1 m/s; back
   to idle the moment it is not blocked, the deadman is released or the car moves.
2. waiting -> reversing after ``wait_sec`` of continuous waiting, only if the rear quarters are clear, at least 0.20 m
   was driven forward recently and attempts < ``max_attempts``. Target = min(``max_distance``, ground driven).
3. reversing: -``speed``; the measured |speed| * dt is integrated; it ends (-> settling, 0.0 that tick) at the
   target, after ``max_time``, when the rear quarters stop being clear, or when the speed is stale or not finite.
4. Deadman released at any time: None, idle; the attempt still counts.
5. settling: 0.0 for ``settle_sec``, then cooldown: None for ``cooldown_sec`` (normal driving, no new reverse).
6. At most ``max_attempts`` reverses until ``note_forward`` has been fed ``rearm_progress`` metres; then the count
   resets. After the last attempt the state is 'exhausted': a car that cannot get out stays stopped.
7. Constructor refuses a speed outside (0, 0.5], max_distance outside (0, 0.6], wait_sec < 0.5, max_attempts
   outside 1..3 and the other bounds below.
"""

import math

MIN_RECENT_FORWARD_M = 0.20     # m driven forward recently before a reverse is allowed (rule 2)
STOPPED_SPEED = 0.1             # m/s (rule 1)


def _finite(x):
    return x is not None and math.isfinite(x)


class ReverseEscape:
    def __init__(self, *, wait_sec=1.0, speed=0.30, max_distance=0.35, max_time=1.6, settle_sec=0.3,
                 cooldown_sec=4.0, max_attempts=2, rearm_progress=3.0, rear_quarter_clearance=0.10):
        if not (math.isfinite(speed) and 0.0 < speed <= 0.5):
            raise ValueError(f'speed must be finite and in (0, 0.5] m/s, got {speed!r}')
        if not (math.isfinite(max_distance) and 0.0 < max_distance <= 0.6):
            raise ValueError(f'max_distance must be finite and in (0, 0.6] m, got {max_distance!r}')
        if not (math.isfinite(wait_sec) and wait_sec >= 0.5):
            raise ValueError(f'wait_sec must be finite and at least 0.5 s, got {wait_sec!r}')
        if not (isinstance(max_attempts, int) and 1 <= max_attempts <= 3):
            raise ValueError(f'max_attempts must be an integer from 1 to 3, got {max_attempts!r}')
        if not (math.isfinite(max_time) and max_time > 0.0):
            raise ValueError(f'max_time must be finite and positive, got {max_time!r}')
        for name, value in (('settle_sec', settle_sec), ('cooldown_sec', cooldown_sec),
                            ('rear_quarter_clearance', rear_quarter_clearance)):
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f'{name} must be finite and >= 0, got {value!r}')
        if not (math.isfinite(rearm_progress) and rearm_progress > 0.0):
            raise ValueError(f'rearm_progress must be finite and positive, got {rearm_progress!r}')
        self.wait_sec, self.speed, self.max_distance, self.max_time = wait_sec, speed, max_distance, max_time
        self.settle_sec, self.cooldown_sec, self.max_attempts = settle_sec, cooldown_sec, max_attempts
        self.rearm_progress, self.rear_quarter_clearance = rearm_progress, rear_quarter_clearance
        self.state = 'idle'
        self.attempts = 0
        self.target = 0.0
        self.remaining = 0.0
        self._timer = 0.0
        self._elapsed = 0.0
        self._progress = 0.0

    def note_forward(self, metres):
        """Forward progress made since the last call (metres, >= 0); enough of it re-arms the escape."""
        if not (math.isfinite(metres) and metres > 0.0):
            return
        self._progress += metres
        if self._progress >= self.rearm_progress:
            self.attempts = 0
            self._progress = 0.0
            if self.state == 'exhausted':
                self.state = 'idle'

    def _end_reverse(self):
        self.state = 'settling'
        self._timer = 0.0

    def step(self, *, dt, deadman, blocked_ahead, measured_speed, rear_quarter_clear, recent_forward_m):
        dt = dt if _finite(dt) and dt > 0.0 else 0.0
        fresh = _finite(measured_speed)
        if not deadman:
            if self.state != 'exhausted':
                self.state = 'idle'
            return None
        if self.state == 'exhausted':
            return None
        if self.state == 'idle':
            if blocked_ahead and fresh and abs(measured_speed) < STOPPED_SPEED:
                self.state = 'waiting'
                self._timer = 0.0
            return None
        if self.state == 'waiting':
            if not blocked_ahead or not fresh or abs(measured_speed) >= STOPPED_SPEED:
                self.state = 'idle'
                return None
            self._timer += dt
            if self.attempts >= self.max_attempts:
                self.state = 'exhausted'
                return None
            if (self._timer >= self.wait_sec and rear_quarter_clear and _finite(recent_forward_m)
                    and recent_forward_m >= MIN_RECENT_FORWARD_M):
                self.attempts += 1
                self._progress = 0.0
                self.target = min(self.max_distance, float(recent_forward_m))
                self.remaining = self.target
                self._elapsed = 0.0
                self.state = 'reversing'
                return -self.speed
            return None
        if self.state == 'reversing':
            if not fresh:
                self._end_reverse()
                return 0.0
            self._elapsed += dt
            self.remaining -= abs(measured_speed) * dt
            if not rear_quarter_clear or self.remaining <= 0.0 or self._elapsed >= self.max_time:
                self._end_reverse()
                return 0.0
            return -self.speed
        if self.state == 'settling':
            self._timer += dt
            if self._timer >= self.settle_sec:
                self.state = 'cooldown'
                self._timer = 0.0
                return None
            return 0.0
        # cooldown
        self._timer += dt
        if self._timer >= self.cooldown_sec:
            self.state = 'exhausted' if self.attempts >= self.max_attempts else 'idle'
        return None
