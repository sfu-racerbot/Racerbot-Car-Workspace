"""pure_course.py with experiment overrides chosen by environment variables.

LEAD=<s>   the follower's speed_lead_time (racing_math.lead_speed, as pure_pursuit_node), instead of the
           car's configured value. Unset or empty: the configured value.
"""
import os
import sys
from pathlib import Path

W = Path.cwd() if (Path.cwd() / "tools" / "f1tenth_sim").is_dir() else \
    Path("/home/bryan/racerbot-ws/.claude/worktrees/map-racer-faster-than-gap")
sys.path[:0] = [str(W / "tools" / "f1tenth_sim"), str(W / "tools"), str(W / "src" / "pure_pursuit")]
import pure_course  # noqa: E402
import run_validation as rv  # noqa: E402

if os.environ.get("LEAD", "") not in ("",):
    LEAD = float(os.environ["LEAD"])
    orig_init = rv.PathFollower.__init__

    def init(self, *a, **k):
        k.setdefault("speed_lead_time", LEAD)
        orig_init(self, *a, **k)
    rv.PathFollower.__init__ = init

# Start on the line being raced, not on the track's own centreline start: on the 2.2 m F1TENTH circuits the
# built line starts up to 0.9 m off the centreline, and a standing-start merge at full profile speed is
# not how the car joins its line (auto_map_race switches to it while driving, PR #22).
rv.RACELINE_SOURCE = "plan"

VMAX = os.environ.get("VMAX", "")
if VMAX:
    # A raised VESC speed ceiling (vesc.yaml speed_max): the plant's speed clamp. Acceleration and braking
    # keep their fitted limits (4 / 3 m/s^2), traction-limited well below the motor's power limit.
    from dataclasses import replace as _replace
    import sim_fidelity
    _v = float(VMAX)
    _orig_with_floor = pure_course.with_floor

    def _with_floor(*a, **k):
        cal = _orig_with_floor(*a, **k)
        return _replace(cal, speed_max=_v, speed_min=-_v)
    pure_course.with_floor = _with_floor

POS_LEAD = os.environ.get("POS_LEAD", "")
if POS_LEAD:
    # Speed looked up where the car is now, not where the delayed pose estimate puts it: the plan's speed at
    # s + v * POS_LEAD (both directions), then the accelerating lead (speed_lead_time) on top of that.
    import numpy as _np2
    from pure_pursuit import racing_math as _rm2
    _tp = float(POS_LEAD)
    _orig_pl = rv.PathFollower.command

    def _pl(self, state, speed=None, *a, **k):
        out = _orig_pl(self, state, speed, *a, **k)
        if speed is not None or out[1] <= 0.0:
            return out
        if not hasattr(self, "_plc"):
            self._plc = _rm2.compute_cumulative_arc_length(self.plan.seg_len)
            self._plt = float(_np2.sum(self.plan.seg_len))
        v = abs(float(state[3]))
        s = (self._plc[out[2]] + max(v, 0.5) * _tp) % self._plt
        xs = _np2.concatenate([self._plc, [self._plt]])
        vs = _np2.concatenate([self.plan.speed, self.plan.speed[:1]])
        base = float(_np2.interp(s, xs, vs))
        if self.speed_lead_time > 0.0:
            ahead = float(_np2.interp((s + max(v, 0.5) * self.speed_lead_time) % self._plt, xs, vs))
            base = max(base, ahead)
        return (out[0], base, *out[2:])
    rv.PathFollower.command = _pl

CEIL_PATH = os.environ.get("CEIL_PATH", "")
if CEIL_PATH:
    # Online curvature ceiling from the planned line's curvature at the car (times CEIL_PATH, e.g. 1.15)
    # instead of max(|desired|, |commanded|) steering curvature, which a tracking law's corrections inflate.
    # A copy of CommandShaper.command (run_validation) with only online_curvature changed.
    import math as _m
    import numpy as _np
    from pure_pursuit import racing_math as _rm
    from pure_pursuit.raceline_optimizer import signed_curvature as _sk
    _factor = float(CEIL_PATH)
    _state = {}
    _orig_fc = rv.PathFollower.command

    def _fc(self, state, *a, **k):
        out = _orig_fc(self, state, *a, **k)
        if not hasattr(self, "_kap"):
            self._kap = _np.abs(_sk(_np.asarray(self.plan.xy, float)))
        _state["kappa"] = float(self._kap[out[2]])
        return out
    rv.PathFollower.command = _fc

    def _sc(self, desired_steering, desired_speed, hard_speed_cap=None, measured_speed=None):
        params = self.params
        if hard_speed_cap is None:
            hard_speed_cap = self.max_speed
        if desired_speed <= 0.0 or hard_speed_cap <= 0.0:
            self.previous_steering = desired_steering
            self.previous_speed = 0.0
            return desired_steering, 0.0
        steering = _rm.slew_rate_limit(desired_steering, self.previous_steering, self.control_dt,
                                       float(params['max_steering_rate']))
        steering = float(_np.clip(steering, -rv.STEERING_LIMIT, rv.STEERING_LIMIT))
        steer_k = max(abs(_m.tan(desired_steering)), abs(_m.tan(steering))) / self.wheelbase
        online_curvature = max(_state.get("kappa", steer_k) * _factor,
                               steer_k * float(os.environ.get("CEIL_MIX", "0") or 0))
        curve_speed = _rm.curvature_speed_limit(online_curvature, self.max_lateral_accel, self.max_speed)
        speed_target = min(desired_speed, curve_speed, hard_speed_cap)
        ramp_basis = self.previous_speed
        if measured_speed is not None:
            ramp_basis = max(ramp_basis, min(abs(float(measured_speed)), self.max_speed, speed_target))
        speed = _rm.slew_rate_limit(speed_target, ramp_basis, self.control_dt,
                                    float(params['max_acceleration']), float(params['max_braking_decel']))
        speed = min(speed, curve_speed, hard_speed_cap)
        self.previous_steering = steering
        self.previous_speed = max(0.0, speed)
        return steering, self.previous_speed
    rv.CommandShaper.command = _sc

FF_TAU = os.environ.get("FF_TAU", "")
if FF_TAU:
    # Speed feedforward through a first-order speed loop: u = v_p(s') + a_p(s') / k, s' = s + v * tau,
    # a_p = v dv/ds of the plan. Replaces the follower's speed when it reads the plan itself.
    import numpy as np
    from pure_pursuit import racing_math
    TAU, GAIN = float(FF_TAU), float(os.environ.get("FF_GAIN", "7.0"))
    orig_cmd = rv.PathFollower.command

    def command(self, state, speed=None, *a, **k):
        out = orig_cmd(self, state, speed, *a, **k)
        if speed is not None or out[1] <= 0.0:
            return out
        plan = self.plan
        if not hasattr(self, "_ffc"):
            self._ffc = np.concatenate([[0.0], np.cumsum(plan.seg_len)])
            self._ffv = np.concatenate([plan.speed, plan.speed[:1]])
        total = self._ffc[-1]
        v = abs(float(state[3]))
        s1 = (self._ffc[out[2]] + max(v, 0.5) * TAU) % total
        ds = 0.15
        va = float(np.interp(s1, self._ffc, self._ffv))
        vb = float(np.interp((s1 + ds) % total, self._ffc, self._ffv))
        acc = va * (vb - va) / ds
        u = max(0.0, va + acc / GAIN)
        return (out[0], u, *out[2:])
    rv.PathFollower.command = command

if __name__ == "__main__":
    raise SystemExit(pure_course.main())
