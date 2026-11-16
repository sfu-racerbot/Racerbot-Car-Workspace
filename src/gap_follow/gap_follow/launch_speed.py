"""The ``max_speed:=`` launch argument: a different top speed for one run.

No ROS imports, so it is testable on its own (test/test_launch_speed.py).

The argument may set the top speed anywhere up to MAX_SPEED_CEILING (5.0 m/s,
raised from "gap_follow.yaml's max_speed" on 2026-10-26 on request); asking
for more is refused rather than clamped, so a typo never quietly runs the car
at a speed nobody asked for. 5.0 m/s is 23000 erpm at car 1's 4600 gain (23070 at
car 2's 4614), inside the shared 23250 speed_max (test_speed_limits_fit_vesc.py).

Lowering: the node refuses to start unless ``min_speed <= max_speed`` and
``corner_speed <= corner_speed_wide <= max_speed`` (gap_follow_node.py), so
the speeds above the new cap are lowered to it as well. Raising: only
max_speed changes. That already satisfies both checks, and the corner and
curvature caps still bind below it, so it is a faster straight, not a faster
car everywhere. Every other speed in the YAML is left alone: each is either a
ceiling below max_speed already or, like narrow_corridor_speed, turns itself
off at or above max_speed.
"""

import math

import yaml

# The most max_speed:= accepts (m/s). Not a tuned value: the requested limit.
MAX_SPEED_CEILING = 5.0

# Speeds the node requires to be no higher than max_speed.
CAPPED_WITH_MAX = ('min_speed', 'corner_speed', 'corner_speed_wide')


def yaml_parameters(config_path):
    """gap_follow_node's ros__parameters from a gap_follow.yaml."""
    with open(config_path, encoding='utf-8') as handle:
        return yaml.safe_load(handle)['gap_follow_node']['ros__parameters']


def speed_cap_overrides(yaml_params, requested):
    """Parameter overrides for ``max_speed:=<requested>``.

    ``requested`` is the launch argument's text; empty means "use the YAML",
    and returns no overrides. Raises ValueError for anything that is not a
    positive, finite speed at or below MAX_SPEED_CEILING.
    """
    text = str(requested).strip()
    if text == '':
        return {}
    try:
        cap = float(text)
    except ValueError:
        raise ValueError(f'max_speed:={requested!r} is not a number') from None
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError(f'max_speed:={requested!r} must be a positive, finite speed (m/s)')
    if cap > MAX_SPEED_CEILING:
        raise ValueError(
            f'max_speed:={cap} is above the {MAX_SPEED_CEILING} m/s ceiling '
            '(gap_follow/launch_speed.py MAX_SPEED_CEILING)')
    overrides = {'max_speed': cap}
    for key in CAPPED_WITH_MAX:
        if float(yaml_params[key]) > cap:
            overrides[key] = cap
    return overrides


#: gap_follow_launch.py's time_trial:=true: one car against the clock. Faster than
#: the shipped values on the course of 2026-10-30, and worse with another car on the
#: track, which is why opponent awareness goes off with it and why it is not the
#: default (docs/gap-follow.md#speed-on-the-real-course-2026-10-30).
TIME_TRIAL = {'aim_depth_bias': 1.0, 'max_lateral_accel': 3.0,
              'enable_opponent_awareness': False}


#: Per-car additions to TIME_TRIAL. None today. Car 2's steering_gain 0.8 / max_steering_rate 4.0
#: (2026-11-14, #71) was 2-7% faster on its own suite but failed the floor matrix (2026-11-15,
#: tools/f1tenth_sim/floor_matrix.py, 12 courses x {own, race-day floor} x grip 0.85/1.0/1.15 x 3 seeds):
#: new contacts on both 10-30 courses and new stalls on 06a/06b on the race-day floor, clearance down
#: 0.15 -> 0.10 m on the race-day course; tools/run_analysis/results/2026-11-14-gap-follow-pace.md.
TIME_TRIAL_BY_CAR = {}


def time_trial_overrides(car_id):
    """time_trial:=true's parameter overrides for ``car_id``: TIME_TRIAL plus that car's additions. The time
    trial's refusal of opponent awareness always wins: a time trial is one car alone."""
    return {**TIME_TRIAL, **TIME_TRIAL_BY_CAR.get(car_id, {}), 'enable_opponent_awareness': False}
