# Adaptive grip for the map racer (plan Phase 5, Task 17): simulator prototype says no lap-time gain to adapt for

Prototype only (a scratch loop around `run_validation`'s follower, map safety and command shaper; nothing shipped):
gap follow's own `AdaptiveGrip`, `SteeringLag` and `LowPass` (rise 0.2 m/s² per s, cut 0.85, hold 0.95, slip 0.9,
lag 0.15 s, ceiling 3.4) drove car 2's online lateral cap and scaled its speed profile by √(limit/base), followed by a
3.0 m/s² backward braking pass. Car 2's race-day line, floor `sfu_course_2026_11_11_rb2` scaled with #74's grip
schedule, `delay_lqr`, map safety, 3 seeds × 3 laps.

| floor | fixed limits | adaptive, floor at base (plan's bound) | adaptive, floor 0.8 × base (advisor's) |
|---|---|---|---|
| calibrated | 9.94 s flying, xt 0.150 m | 9.96 s, xt 0.150; limit 3.10-3.12 | 10.24 s, xt 0.081; limit fell to 2.48 |
| 15% grippier | 9.35 s, xt 0.109 | 9.35 s, xt 0.110; limit rose to 3.34 | 9.70 s, xt 0.072 |
| 15% slicker | 10.80 s, xt 0.215 | 10.80 s, xt 0.216 | 11.04 s, xt 0.122 |
| cut to 0.85 at 12 s | 10.73 s, xt 0.190 | 10.71 s, xt 0.190 | 11.00 s, xt 0.107 |

Zero contact in every cell, fixed limits included.

**Task 17's acceptance fails, and not for want of adaptation.** A fixed profile scale of 1.04 (what a fully adapted
3.34 m/s² limit asks) laps the grippier floor in 9.35 s too: on this line the lap is not limited by the lateral cap or
the profile's corner speeds, so there is no time for an adaptive limit to win. The slicker floors make no contact with
fixed limits either, so there is no contact for it to remove. Allowing cuts below the base (advisor's 0.8) halves the
cross-track error but costs 3% everywhere, because the cut fires on ordinary ratio noise.

No node work was started (pure pursuit has no IMU input today; it would need `/imu/data` and `/imu/status` gated as
gap follow's `_imu_gate`). Revisit if a course or floor appears where the map racer's corners, not its straights or
speed loop, set the lap.
