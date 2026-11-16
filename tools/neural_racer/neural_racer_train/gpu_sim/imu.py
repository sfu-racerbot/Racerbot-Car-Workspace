"""The simulated IMU: yaw rate and specific force at the IMU's mount point.

From the single-track state x and its derivative f (dynamics.st_derivative)
at the centre of gravity, in the body frame:

    v_x = V cos(beta),  v_y = V sin(beta)
    a_x = d/dt(v_x) - w v_y,     a_y = d/dt(v_y) + w v_x
    at a point d ahead of the CoG:  a_x,P = a_x - w^2 d,   a_y,P = a_y + w_dot d

The mount point is measured from the rear axle (base_link); the CoG sits lr
ahead of it, so d = imu_offset_x - lr. Car 1's BNO085 at the front axle (0.36 m)
is 0.18 m ahead: in a 2 rad/s turn the IMU reads ~0.7 m/s^2 more than the CoG
(plan A3). Simulating the sensor where it sits means neither side corrects.
"""
import torch

# Vibration noise once the car moves (speed > MOVING), per sample: std = c0 + c1 * speed, for gyro z
# (rad/s), accel x and accel y (m/s^2). Fitted to car 1's BNO085 while driving (runs
# 20261030-175701, gap_follow at 5 m/s, and 20261026-194913): gyro 0.023-0.034 rad/s, accel x
# 0.4-1.0 and accel y 0.6-1.9 m/s^2 between 1 and 4.5 m/s, around a 0.08 s mean.
VIBRATION = ((0.012, 0.004), (0.25, 0.15), (0.25, 0.35))
MOVING = 0.2


def noise_std(rest_gyro, rest_accel, vibration, speed, gyro_vibration=None):
    """(..., 3) per-sample noise std of (gyro z, accel x, accel y): the rest noise,
    plus VIBRATION's curve once moving, scaled by gyro_vibration (the gyro; default
    `vibration`) and vibration (the accelerometers). Inputs broadcast."""
    moving = (speed.abs() > MOVING).to(speed.dtype)
    v = speed.abs()
    scale = [vibration if gyro_vibration is None else gyro_vibration, vibration, vibration]
    vib = [moving * s * (c0 + c1 * v) for s, (c0, c1) in zip(scale, VIBRATION)]
    rest = [rest_gyro, rest_accel, rest_accel]
    return torch.stack([torch.broadcast_to(r, v.shape) + d for r, d in zip(rest, vib)], dim=-1)


def specific_force(x, f, d):
    """(gyro_z, a_x, a_y) at a point d metres ahead of the CoG; all (...,)."""
    v, beta = x[..., 3], x[..., 6]
    v_dot, w, w_dot, beta_dot = f[..., 3], f[..., 4], f[..., 5], f[..., 6]
    vx, vy = v * torch.cos(beta), v * torch.sin(beta)
    vx_dot = v_dot * torch.cos(beta) - v * torch.sin(beta) * beta_dot
    vy_dot = v_dot * torch.sin(beta) + v * torch.cos(beta) * beta_dot
    ax = vx_dot - w * vy - w * w * d
    ay = vy_dot + w * vx + w_dot * d
    return w, ax, ay
