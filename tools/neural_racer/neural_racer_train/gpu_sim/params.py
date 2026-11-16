"""Vehicle and sensor parameters, batched per environment, with randomization
keyed to how well each number is known (plan A7).

Values come from tools/f1tenth_sim/sim_fidelity/calibration.py -- the same
numbers the CPU referee uses -- with car 1's own measurements applied on top
(CAR1_OVERRIDES), because that calibration still describes car 2 in places
(docs/car1-vs-car2.md: "Simulator vehicle model: not car 1's"). Overriding
here, rather than editing calibration.py, keeps this branch from touching
existing code; Phase 1 of the plan moves the overrides into calibration.py.
"""
import math
from dataclasses import dataclass, fields, replace

import torch

G = 9.81                   # gym's and calibration.py's gravity
CONTROL_DT = 0.025         # 40 Hz, run_validation.CONTROL_DT
INTEGRATOR_DT = 0.005      # run_validation.INTEGRATOR_DT, RK4

# Car 1 measurements that differ from calibration.py (car 2's numbers).
CAR1_OVERRIDES = {
    # vesc.yaml speed_max / car 1's command gain: 4600 since 2026-11-02 (main ea5f114; at 3650 car 1
    # delivered 0.79 of every command), so 5.05 m/s. Was 6.37 at 3650; car 2's 4614 gives 5.04
    'speed_max': 23250.0 / 4600.0,
    'speed_min': -23250.0 / 4600.0,
    'steering_max_left': 0.401,        # car 1 circles at full stick, 2026-10-21 (hardware-reference.md)
    'steering_max_right': -0.388,
}
# The motor's top speed in training (2026-11-08). The VESC's configured limit is CAR1_OVERRIDES' 5.05 m/s,
# but the car itself can reach about 27 m/s (97 km/h, the team). Training lets the policy use up to 10 m/s:
# the most any course we race can use (levine_blocked's fastest line peaks at 9.8 m/s; car 1's course 7.6,
# car 2's 6.5-8.2, the room 5.0, with the measured 4.5 m/s^2 acceleration, 4.0 braking and grip). Nothing
# above about 5.4 m/s has been measured, so the motor's power limit (v_switch) is drawn +/-30% (RANDOMIZATION).
# The CPU referee and the car keep the VESC's limit until the team raises it.
TRAIN_SPEED_MAX = 10.0


@dataclass(frozen=True)
class CarParams:
    # gym single-track parameter vector (calibration.vehicle_parameters)
    mu: float = 0.55
    C_Sf: float = 4.718
    C_Sr: float = 5.4562
    wheelbase: float = 0.36
    cog_front_fraction: float = 0.5
    h: float = 0.074
    m: float = 3.74
    I: float = 0.04712
    steering_max_left: float = CAR1_OVERRIDES['steering_max_left']
    steering_max_right: float = CAR1_OVERRIDES['steering_max_right']
    steering_rate_max: float = 3.2
    v_switch: float = 7.319            # gym stock, not overridden by calibration
    gym_v_min: float = -5.0            # gym stock; sets pid_accl's gain, see calibration.py
    gym_v_max: float = 20.0
    # FidelityPlant's actuation layer
    speed_max: float = TRAIN_SPEED_MAX
    speed_min: float = CAR1_OVERRIDES['speed_min']
    servo_gain: float = 15.0
    steer_delay_ticks: int = 3
    speed_delay_ticks: int = 2
    speed_loop_accel_gain: float = 5.0
    speed_loop_accel_max: float = 4.5
    speed_loop_brake_gain: float = 4.0
    speed_loop_brake_max: float = 4.0
    longitudinal_grip_coupling: float = 1.0
    # the floor's grip (actuation.grip_clamp): 1 = the measured progressive curve, stretched by grip_scale;
    # 0 = the friction circle at mu * g (the CPU referee's plain calibration; the dynamics parity tests)
    grip_model: float = 1.0
    grip_scale: float = 1.0
    # bodies
    body_width: float = 0.30
    body_length: float = 0.535
    envelope_width: float = 0.31
    envelope_length: float = 0.58
    lidar_offset_x: float = 0.26
    # sensors
    lidar_noise_std: float = 0.01
    lidar_dropout_rate: float = 0.002
    # Car 1's invalid beams come in runs (2026-11-01, runs 20261030-175701 and 20261026-194913): 95%
    # of them in runs of >= 4 beams, median 9, p90 56; and a room can hold a sector that drops out in
    # half the scans (+16 deg, a dark or glass surface). lidar.sector_dropout draws both: transient
    # sectors (lidar_sectors per scan, on average) and one persistent sector at lidar_persist_deg,
    # lidar_persist_width beams wide, present in a scan with probability lidar_persist (<= 0: none)
    lidar_sectors: float = 0.3
    lidar_persist: float = 0.0
    lidar_persist_deg: float = 0.0
    lidar_persist_width: float = 10.0
    lidar_delay_ticks: int = 1
    odom_speed_noise: float = 0.05     # fractional
    # /imu/data is the BNO085 (imu_bridge's primary), rotated into car axes but not moved: car 1
    # mounts it at the front axle's centre, 0.18 m ahead of the wheelbase's midpoint
    # (tools/imu_calibration/config/car.yaml, measured), so 0.36 m ahead of the rear axle.
    # Was 0.45, the front bumper (team's estimate, 2026-10-23)
    imu_offset_x: float = 0.36
    # Per-sample IMU noise as the node sees it: /imu/data's latest message each tick, no averaging.
    # At rest (car 1's BNO085): ~0.001-0.008 rad/s, 0.03-0.09 m/s^2. Once the car moves, vibration
    # adds imu.VIBRATION's speed-dependent std, scaled by imu_vibration (2026-11-01, measured: the
    # 0.005 / 0.15 estimates were 5-13x under what the car reads while driving)
    imu_gyro_noise: float = 0.004
    imu_accel_noise: float = 0.06
    imu_vibration: float = 1.0
    # the gyro's own vibration scale: car 2's gyro shakes up to 2x car 1's while its accelerometers
    # are quieter (runs 20261030-184556 and -185036: gyro 0.020-0.048 rad/s, accel x 0.27-0.48,
    # accel y 0.28-0.39 m/s^2), so the two scales are drawn apart
    imu_gyro_vibration: float = 1.0
    imu_delay_ticks: int = 1           # ESTIMATED until P0.5
    imu_gyro_bias: float = 0.005

    @property
    def lf(self) -> float:
        return self.wheelbase * self.cog_front_fraction

    @property
    def lr(self) -> float:
        return self.wheelbase * (1.0 - self.cog_front_fraction)

    @property
    def a_max(self) -> float:
        return self.mu * G                       # calibration: friction_accel_limit


def from_calibration() -> CarParams:
    """CarParams built from sim_fidelity's CALIBRATION plus car 1's overrides.
    Falls back to the defaults above when the simulator checkout is absent."""
    try:
        from neural_racer_train.fidelity import calibration_module
        cal = calibration_module().CALIBRATION
    except Exception:  # noqa: BLE001 -- no checkout: the defaults mirror it
        return CarParams()
    return CarParams(
        mu=cal.surface_friction, C_Sf=cal.cornering_stiffness_front,
        C_Sr=cal.cornering_stiffness_rear, wheelbase=cal.wheelbase,
        cog_front_fraction=cal.cog_front_fraction, h=cal.cog_height, m=cal.mass,
        I=cal.yaw_inertia, steering_rate_max=cal.steering_rate_max,
        servo_gain=cal.steering_servo_gain, steer_delay_ticks=cal.steering_delay_ticks,
        speed_delay_ticks=cal.speed_delay_ticks,
        speed_loop_accel_gain=cal.speed_loop_accel_gain, speed_loop_accel_max=cal.speed_loop_accel_max,
        speed_loop_brake_gain=cal.speed_loop_brake_gain, speed_loop_brake_max=cal.speed_loop_brake_max,
        longitudinal_grip_coupling=cal.longitudinal_grip_coupling,
        body_width=cal.body_width, body_length=cal.body_length,
        envelope_width=cal.width, envelope_length=cal.length, lidar_offset_x=cal.lidar_offset_x,
        lidar_noise_std=cal.lidar_noise_std, lidar_dropout_rate=cal.lidar_dropout_rate,
        lidar_delay_ticks=cal.lidar_delay_ticks, odom_speed_noise=cal.odom_speed_noise,
    )


# Randomization by provenance (plan A7): (kind, spread). 'rel' = +/- fraction,
# 'abs' = +/- absolute, 'ticks' = integer range [lo, hi], 'range' = [lo, hi].
RANDOMIZATION = {
    'speed_loop_accel_gain': ('rel', 0.05), 'speed_loop_accel_max': ('rel', 0.05),
    'speed_loop_brake_gain': ('rel', 0.05), 'speed_loop_brake_max': ('rel', 0.05),
    'lidar_noise_std': ('rel', 0.10), 'lidar_dropout_rate': ('rel', 0.30),
    # the cars' recorded commands replayed through this sim's dynamics match their gyros at mu ~0.5-0.55
    # (car 1) and ~0.4 (car 2) (2026-11-03): 0.40-1.10 put both floors at its bottom end
    'mu': ('range', (0.30, 0.80)),
    'v_switch': ('rel', 0.30),                 # the motor's power limit: unmeasured above ~5.4 m/s
    'grip_scale': ('range', (0.85, 1.3)),      # actuation.GRIP_DELIVERED: both cars' replays inside (car 1 ~1.25)
    'C_Sf': ('rel', 0.30), 'C_Sr': ('rel', 0.30), 'h': ('rel', 0.30), 'I': ('rel', 0.30),
    'servo_gain': ('rel', 0.30),
    'steer_delay_ticks': ('ticks', (2, 4)), 'speed_delay_ticks': ('ticks', (1, 3)),
    'lidar_offset_x': ('abs', 0.02), 'imu_offset_x': ('abs', 0.02),
    'imu_gyro_noise': ('rel', 0.30), 'imu_accel_noise': ('rel', 0.30),
    'imu_vibration': ('range', (0.3, 1.5)),       # accelerometers: car 2's quietest to car 1's noisiest runs
    'imu_gyro_vibration': ('range', (0.4, 2.3)),  # gyro: car 1's quietest to car 2's noisiest
    'lidar_sectors': ('range', (0.0, 1.5)),       # car 2 drops ~0.9-1.0 runs of beams a scan (car 1 0.1-0.6)
    'lidar_persist': ('range', (-0.6, 0.8)),      # <= 0 in ~43% of episodes: no persistent sector
    'lidar_persist_deg': ('range', (-135.0, 135.0)),
    'lidar_persist_width': ('range', (4.0, 60.0)),
    'imu_delay_ticks': ('ticks', (0, 2)),
}
# Steering-gain error of +/-10% (car 1's gain was 30% off until 2026-10-21):
STEER_GAIN_ERROR = 0.10


class BatchParams:
    """Every CarParams field as an (E,) tensor, plus derived fields."""

    def __init__(self, values: dict, device):
        self.device = device
        for k, v in values.items():
            setattr(self, k, v)

    @classmethod
    def constant(cls, p: CarParams, n: int, device='cpu', dtype=torch.float32):
        vals = {f.name: torch.full((n,), float(getattr(p, f.name)), device=device, dtype=dtype)
                for f in fields(CarParams)}
        vals['steer_gain'] = torch.ones(n, device=device, dtype=dtype)
        out = cls(vals, device)
        out._derive()
        return out

    @classmethod
    def randomized(cls, p: CarParams, n: int, gen: torch.Generator, device='cpu'):
        out = cls.constant(p, n, device)
        for name, (kind, spread) in RANDOMIZATION.items():
            base = getattr(out, name)
            u = torch.rand(n, generator=gen, device=gen.device).to(device)
            if kind == 'rel':
                val = base * (1.0 + spread * (2 * u - 1))
            elif kind == 'abs':
                val = base + spread * (2 * u - 1)
            elif kind == 'range':
                val = spread[0] + (spread[1] - spread[0]) * u
            else:
                lo, hi = spread
                val = torch.floor(lo + (hi - lo + 1) * u).clamp(lo, hi)
            setattr(out, name, val)
        out.steer_gain = 1.0 + STEER_GAIN_ERROR * (2 * torch.rand(n, generator=gen, device=gen.device).to(device) - 1)
        out._derive()
        return out

    def _derive(self):
        self.lf = self.wheelbase * self.cog_front_fraction
        self.lr = self.wheelbase * (1.0 - self.cog_front_fraction)
        self.a_max = self.mu * G

    def index(self, idx):
        vals = {k: v[idx] for k, v in vars(self).items() if isinstance(v, torch.Tensor)}
        out = BatchParams(vals, self.device)
        return out

    def assign(self, idx, other: 'BatchParams'):
        for k, v in vars(other).items():
            if isinstance(v, torch.Tensor):
                getattr(self, k)[idx] = v


def with_overrides(p: CarParams, **kw) -> CarParams:
    return replace(p, **kw)
