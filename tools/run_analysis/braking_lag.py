"""How fast wheel speed follows the /drive speed command, and how hard the car brakes.

Two plain measurements, taken the same way on a real run and on a plant replaying that run's commands
(tools/f1tenth_sim/tests/test_floor_2026_11_11_rb2.py):

* ``command_lag``: the delay that best aligns the command with wheel speed, over the samples where the
  command is moving at more than 1 m/s^2 in the chosen direction (and the half second after each such
  sample, which is when the wheels answer), by least squares with a free gain.
* ``hardest_zones`` / ``zone_decels``: stretches where the 0.25 s smoothed deceleration stays above
  1.5 m/s^2 for at least 0.4 s without ending in a stop, ranked by mean deceleration.
"""

from __future__ import annotations

import numpy as np

DT = 0.025
RAMP = 1.0            # m/s^2: a command moving faster than this is a ramp
RESPONSE_S = 0.5      # the wheels' answer to a ramp is looked for this long after it
LAGS = np.arange(0.0, 0.6001, 0.0125)
HARD = 1.5            # m/s^2
MIN_ZONE_S = 0.4
SMOOTH_S = 0.25


def command_lag(segments, sign: int) -> tuple:
    """(best lag in s, ramp samples) over ``segments`` of (t, command, wheel speed); sign +1 rising, -1 falling."""
    n = int(RESPONSE_S / DT)
    prepared, ramps = [], 0
    for t, cmd, wheel in segments:
        t, cmd, wheel = np.asarray(t, float), np.asarray(cmd, float), np.asarray(wheel, float)
        moving = sign * np.gradient(cmd, t) > RAMP
        mask = np.zeros(len(t), bool)
        for i in np.flatnonzero(moving):
            mask[i:i + n] = True
        prepared.append((t, cmd, wheel, mask))
        ramps += int(moving.sum())
    best = (float("nan"), float("inf"))
    for lag in LAGS:
        x = np.concatenate([np.interp(t - lag, t, cmd)[mask] for t, cmd, _, mask in prepared])
        y = np.concatenate([wheel[mask] for _, _, wheel, mask in prepared])
        if not len(x) or not float(x @ x):
            continue
        gain = float(x @ y) / float(x @ x)
        rms = float(np.sqrt(np.mean((gain * x - y) ** 2)))
        if rms < best[1]:
            best = (float(lag), rms)
    return best[0], ramps


def _decel(t, wheel):
    k = max(1, int(round(SMOOTH_S / DT)))
    return -np.gradient(np.convolve(np.asarray(wheel, float), np.ones(k) / k, "same"), np.asarray(t, float))


def hardest_zones(segments, n: int = 3) -> list:
    """The ``n`` hardest braking zones over ``segments`` of (t, wheel speed), as dicts
    (segment, i, j, decel, duration, v_from, v_to); decel is the zone's mean smoothed deceleration."""
    zones = []
    for index, (t, wheel) in enumerate(segments):
        t, wheel = np.asarray(t, float), np.asarray(wheel, float)
        hard = _decel(t, wheel) > HARD
        i = 0
        while i < len(t):
            if not hard[i]:
                i += 1
                continue
            j = i
            while j < len(t) and hard[j]:
                j += 1
            duration = float(t[j - 1] - t[i])
            if duration >= MIN_ZONE_S and wheel[j - 1] >= 1.0:
                zones.append(dict(segment=index, i=i, j=j, decel=float(np.mean(_decel(t, wheel)[i:j])),
                                  duration=duration, v_from=float(wheel[i]), v_to=float(wheel[j - 1])))
            i = j
    zones.sort(key=lambda z: -z["decel"] * min(1.0, z["duration"]))
    return zones[:n]


def zone_decels(segments, zones) -> list:
    """Mean smoothed deceleration of another speed trace over the same zones."""
    return [float(np.mean(_decel(*segments[z["segment"]])[z["i"]:z["j"]])) for z in zones]
