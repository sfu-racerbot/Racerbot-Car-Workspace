"""Batched actuation: the chain FidelityPlant.step applies before gym, in the
same order (tools/f1tenth_sim/sim_fidelity/plant.py):

    command delays -> speed clamp -> measured speed loop -> grip clamp on
    steering -> servo velocity -> gym's pid_accl once per tick -> RK4 substeps
"""
import math

import torch

from neural_racer_train.gpu_sim.dynamics import accl_constraints, pid_accl

KINEMATIC_SPEED = 0.5      # grip.KINEMATIC_SPEED: no grip limit below this


class DelayLine:
    """Per-environment FIFO delay of 0..max_ticks control ticks, on the GPU.

    push(x) stores x and returns the value pushed `delay` ticks ago (the
    initial value until the line has filled), like TransportDelay.push.
    """

    def __init__(self, shape, max_ticks: int, delay: torch.Tensor, initial: float = 0.0, device='cpu'):
        self.max = int(max_ticks)
        self.buf = torch.full((self.max + 1, *shape), float(initial), device=device)
        self.head = 0
        self.delay = delay.long().clamp(0, self.max)

    def reset(self, env_ids, value: float = 0.0):
        self.buf[:, env_ids] = value

    def push(self, x):
        self.head = (self.head + 1) % (self.max + 1)
        self.buf[self.head] = x
        idx = (self.head - self.delay) % (self.max + 1)
        while idx.dim() < x.dim():
            idx = idx.unsqueeze(-1)
        idx = idx.expand_as(x)
        return torch.gather(self.buf, 0, idx.unsqueeze(0)).squeeze(0)

    def recent(self, k: int):
        """The last k values pushed, newest first: (k, *shape). With a delay
        of d ticks, the newest d of them have not reached the actuator yet."""
        idx = [(self.head - j) % (self.max + 1) for j in range(k)]
        return self.buf[idx]


def measured_speed_loop(command, measured, p):
    """FidelityPlant._measured_speed_loop: the gym speed command that makes
    gym's pid_accl produce this car's fitted acceleration for `command`."""
    diff = command - measured
    accel_up = torch.minimum(p.speed_loop_accel_gain * diff, p.speed_loop_accel_max)
    accel_dn = torch.maximum(p.speed_loop_brake_gain * diff, -p.speed_loop_brake_max)
    gain_up = 10.0 * p.a_max / p.gym_v_max
    gain_dn = 10.0 * p.a_max / (-p.gym_v_min)
    out = measured + torch.where(diff >= 0.0, accel_up / gain_up, accel_dn / gain_dn)
    return torch.where((measured <= 0.0) & (command <= 0.0), command, out)


def longitudinal_accel(speed_cmd, measured, p):
    """grip.longitudinal_accel: what gym's longitudinal controller is about to do."""
    demand = pid_accl(speed_cmd, measured, p.a_max, p.gym_v_max, p.gym_v_min)
    return accl_constraints(measured, demand, p.v_switch, p.a_max, p.gym_v_min, p.gym_v_max)


# The floor's grip as measured (main's sim_fidelity ProgressiveGrip, 2026-11-07): the lateral acceleration
# the rack asks for (m/s^2) against what the tyres deliver. The car follows the request to 2.0 m/s^2, then
# gains ever more slowly: there is no one friction number. GRIP_DELIVERED is the mean of the two measured
# floors at GRIP_ASKED:
#   sfu_course_2026_10_30  0, 2.0, 2.25, 2.737, 3.013, 3.209, 3.45, 3.841, 4.255, 4.715, 5.175
#   sfu_course_2026_11_06  0, 2.0, 2.226, 2.563, 2.726, 3.014, 3.204, 3.597, 3.987, 4.888, 4.888
#                          (its x1.05 replay scale; flat past 10 m/s^2)
# CarParams.grip_scale k stretches it per race, delivered = k * curve(asked / k), drawn from 0.85-1.3:
# replaying the cars' /drive commands through this sim put car 2 near 1.0 and car 1 near 1.25 (2026-11-08).
# Braking costs it almost nothing (2.7 m/s^2 achieved from 3 m/s^2 of braking through 1.5 of acceleration),
# so unlike the circle it takes no share for longitudinal force.
GRIP_ASKED = (0.0, 2.0, 2.25, 2.75, 3.25, 3.75, 4.5, 5.5, 7.0, 10.0, 20.0)
GRIP_DELIVERED = (0.0, 2.0, 2.238, 2.65, 2.869, 3.111, 3.327, 3.719, 4.121, 4.802, 5.032)


def progressive_lateral(asked, k):
    """What the tyres deliver for `asked` m/s^2 at curve scale k (piecewise linear, flat past the table)."""
    xs = torch.tensor(GRIP_ASKED, device=asked.device, dtype=asked.dtype)
    ys = torch.tensor(GRIP_DELIVERED, device=asked.device, dtype=asked.dtype)
    x = (asked / k).clamp(min=0.0, max=xs[-1])
    i = torch.searchsorted(xs, x.contiguous(), right=True).clamp(1, len(xs) - 1)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return k * (y0 + (y1 - y0) * (x - x0) / (x1 - x0))


def grip_clamp(steering, speed, accel, p):
    """The wheel angle the tyres will actually turn the car by. p.grip_model 1 (default): the floor's
    measured progressive curve (ProgressiveGrip.apply); 0: GripEnvelope.apply's friction circle
    (mu * g, longitudinal force taking its share), the CPU referee's plain calibration."""
    v2 = (speed * speed).clamp(min=1e-12)
    no_limit = speed < KINEMATIC_SPEED
    # friction circle
    limit = p.a_max                      # friction_accel_limit = mu * g
    used = accel * p.longitudinal_grip_coupling
    remaining = limit * limit - used * used
    budget = torch.sqrt(remaining.clamp(min=0.0))
    max_steer = torch.atan(p.wheelbase * budget / v2)
    circle = torch.where(steering.abs() <= max_steer, steering, torch.copysign(max_steer, steering))
    # progressive curve
    demanded = torch.tan(steering).abs() * v2 / p.wheelbase
    allowed = progressive_lateral(demanded, p.grip_scale * torch.ones_like(demanded))
    progressive = torch.copysign(torch.atan(p.wheelbase * allowed / v2), steering)
    out = torch.where(p.grip_model > 0.5, progressive, circle)
    return torch.where(no_limit, steering, out)


def servo_velocity(target, current, p):
    """SteeringServo.velocity: proportional position loop with a slew ceiling."""
    t = torch.clamp(target, p.steering_max_right, p.steering_max_left)
    demand = p.servo_gain * (t - current)
    return torch.clamp(demand, -p.steering_rate_max, p.steering_rate_max)


def actuate(steer_cmd_delayed, speed_cmd_delayed, measured_speed, current_steer, p):
    """The (steering velocity, acceleration) control applied for the next tick."""
    speed = torch.clamp(speed_cmd_delayed, p.speed_min, p.speed_max)
    speed = measured_speed_loop(speed, measured_speed, p)
    accel_est = longitudinal_accel(speed, measured_speed, p)
    steering = grip_clamp(steer_cmd_delayed * p.steer_gain, measured_speed, accel_est, p)
    sv = servo_velocity(steering, current_steer, p)
    accl = pid_accl(speed, measured_speed, p.a_max, p.gym_v_max, p.gym_v_min)
    return torch.stack([sv, accl], dim=-1)


def wrap_angle(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi
