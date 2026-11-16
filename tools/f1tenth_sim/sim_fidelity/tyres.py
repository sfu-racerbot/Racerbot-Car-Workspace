"""Tyres that let go: a single-track model whose axle forces saturate and fall, so the simulated car can spin.

Gym's single-track model (``f1tenth_gym/envs/dynamic_models/single_track.py``, CommonRoad section 7) uses a
linear tyre per axle, ``F_y = mu * C_S * F_z * alpha``, with no upper bound. ``grip.py`` caps the cornering
from outside by clamping the steering angle, which reproduces the envelope but never loss of control: at the
limit the car runs wide along it. Car 1's U-turn hit on 2026-11-11 (run 20261111-155542, about 87.3 s) was a
spin instead: yaw rate jumped from 1.4 to 4-5 rad/s while the steering straightened
(tools/run_analysis/results/2026-11-13-corner-entry-study.md). No clamp on the steering can produce that,
because it needs the rear axle to saturate while the front still has force.

This module replaces gym's dynamics function with the same model and the same state, but with a
saturating per-axle force:

    F_y = F_z * D * sin(C * atan(B * alpha)),   B * C * D = mu * C_S

i.e. Pacejka's magic formula without the curvature term, with its slope at zero slip equal to gym's linear
tyre (so below the limit nothing changes), a peak ``D`` per axle (a friction coefficient), and a shape ``C``
that sets how far the force falls past the peak: it slides at ``D * sin(C * pi / 2)``. Slip angles are the
exact ``atan`` forms rather than gym's small-angle ones, since a spinning car's slip is not small.
Longitudinal force shares each axle's grip by a friction ellipse scaled by ``coupling`` (0: none).

The yaw and slip equations reduce to gym's exactly when the forces are linear and the angles small; the
test ``test_tyres.py`` checks that against gym's own function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 9.81
# Gym's own switch to a kinematic model (single_track.py: ``if V < 0.5``); slip is not modelled below it.
KINEMATIC_SPEED = 0.5


@dataclass(frozen=True)
class TyreModel:
    """A saturating tyre per axle.

    Attributes:
        peak_front, peak_rear: friction coefficient at the force peak, per axle (lateral force / normal load).
        shape: magic-formula C, in (1, 2). Sliding force over peak is sin(C * pi / 2): 1.0 at C = 1, 0 at C = 2.
        coupling: share of the longitudinal acceleration that uses each axle's grip, 0 to 1.
    """

    peak_front: float
    peak_rear: float
    shape: float = 1.6
    coupling: float = 0.0

    def __post_init__(self):
        for name in ("peak_front", "peak_rear"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {value!r}")
        if not (math.isfinite(self.shape) and 1.0 < self.shape < 2.0):
            raise ValueError(f"shape must be in (1, 2), got {self.shape!r}")
        if not (math.isfinite(self.coupling) and 0.0 <= self.coupling <= 1.0):
            raise ValueError(f"coupling must be in [0, 1], got {self.coupling!r}")

    @property
    def slide_ratio(self) -> float:
        """Force when fully sliding over the peak force."""
        return math.sin(self.shape * math.pi / 2.0)


def axle_force_ratio(alpha: float, slope: float, peak: float, shape: float) -> float:
    """Lateral force over normal load for slip angle ``alpha`` (rad): ``peak * sin(shape * atan(B * alpha))``.

    ``slope`` is the linear tyre's force per unit load per radian at zero slip (gym's ``mu * C_S``).
    Pure Python twin of the compiled function, for tests and analysis.
    """
    b = slope / (shape * peak)
    return peak * math.sin(shape * math.atan(b * alpha))


def make_dynamics(model: TyreModel):
    """A numba-compiled drop-in for gym's ``vehicle_dynamics_st`` with ``model``'s tyres.

    Same signature ``f(x, u_init, params)``, same state (x, y, steering, speed, yaw, yaw rate, slip) and the
    same params array (gym's ``VehicleParameters`` order), so ``Simulator.dynamics_fn`` can be swapped.
    """
    from numba import njit
    from f1tenth_gym.envs.dynamic_models.utils import accl_constraints, steering_constraint

    peak_f = float(model.peak_front)
    peak_r = float(model.peak_rear)
    shape = float(model.shape)
    coupling = float(model.coupling)
    kinematic = KINEMATIC_SPEED

    @njit(cache=False)
    def ratio(alpha, slope, peak, budget):
        # The ellipse: what longitudinal force leaves of the peak; the slope at zero slip is unchanged.
        top = peak * budget
        if top <= 1e-6:
            return 0.0
        b = slope / (shape * top)
        return top * np.sin(shape * np.arctan(b * alpha))

    @njit(cache=False)
    def dynamics(x, u_init, params):
        DELTA = x[2]
        V = x[3]
        PSI = x[4]
        PSI_DOT = x[5]
        BETA = x[6]
        mu = params[0]
        C_Sf = params[1]
        C_Sr = params[2]
        lf = params[3]
        lr = params[4]
        h = params[5]
        m = params[6]
        I = params[7]
        STEER_VEL = steering_constraint(DELTA, u_init[0], params[8], params[9], params[10], params[11])
        ACCL = accl_constraints(V, u_init[1], params[12], params[13], params[14], params[15])
        lwb = lf + lr
        if V < kinematic:
            # Gym's kinematic branch, unchanged.
            BETA_HAT = np.arctan(np.tan(DELTA) * lr / lwb)
            BETA_DOT = ((1 / (1 + (np.tan(DELTA) * (lr / lwb)) ** 2)) * (lr / (lwb * np.cos(DELTA) ** 2)) * STEER_VEL)
            return np.array([
                V * np.cos(PSI + BETA_HAT),
                V * np.sin(PSI + BETA_HAT),
                STEER_VEL,
                ACCL,
                V * np.cos(BETA_HAT) * np.tan(DELTA) / lwb,
                (1 / lwb) * (ACCL * np.cos(BETA) * np.tan(DELTA) - V * np.sin(BETA) * np.tan(DELTA) * BETA_DOT
                             + ((V * np.cos(BETA) * STEER_VEL) / (np.cos(DELTA) ** 2))),
                BETA_DOT,
            ])
        # Normal loads with gym's longitudinal transfer (glr, glf times m / L).
        fz_f = m * (G * lr - ACCL * h) / lwb
        fz_r = m * (G * lf + ACCL * h) / lwb
        vx = V * np.cos(BETA)
        vy = V * np.sin(BETA)
        alpha_f = DELTA - np.arctan2(vy + lf * PSI_DOT, vx)
        alpha_r = -np.arctan2(vy - lr * PSI_DOT, vx)
        used = coupling * ACCL / G
        budget_f = 1.0
        budget_r = 1.0
        if used != 0.0:
            share_f = used / peak_f
            share_r = used / peak_r
            budget_f = np.sqrt(max(0.0, 1.0 - share_f * share_f))
            budget_r = np.sqrt(max(0.0, 1.0 - share_r * share_r))
        fy_f = fz_f * ratio(alpha_f, mu * C_Sf, peak_f, budget_f)
        fy_r = fz_r * ratio(alpha_r, mu * C_Sr, peak_r, budget_r)
        return np.array([
            V * np.cos(PSI + BETA),
            V * np.sin(PSI + BETA),
            STEER_VEL,
            ACCL,
            PSI_DOT,
            (lf * fy_f * np.cos(DELTA) - lr * fy_r) / I,
            (fy_f * np.cos(DELTA) + fy_r) / (m * V) - PSI_DOT,
        ])

    return dynamics
