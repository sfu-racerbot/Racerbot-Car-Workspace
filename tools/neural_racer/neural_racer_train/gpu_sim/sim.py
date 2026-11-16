"""BatchedRaceSim: E environments x up to C cars, every tensor on one device.

One step() is one 25 ms control tick, in FidelityPlant's order:

    safety wrapper -> command delays -> actuation (speed loop, grip, servo,
    pid_accl) -> 5 x RK4 at 5 ms (IMU sampled every 20 ms) -> contacts ->
    race progress and reward -> sensors (scan one tick stale) -> history

Car state is gym's single-track state at the centre of gravity. Poses handed
in and out (levels, base_link for the preview) are the rear axle, as on the
car. Nothing in a rollout leaves the device.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from neural_racer import observation as obs
from neural_racer.safety_wrapper import WrapperConfig
from neural_racer_train.gpu_sim import actuation, contact, imu as imu_mod, lidar, race as race_mod
from neural_racer_train.gpu_sim import observation_torch as ot
from neural_racer_train.gpu_sim.dynamics import rk4, st_derivative
from neural_racer_train.gpu_sim.params import CONTROL_DT, INTEGRATOR_DT, BatchParams, CarParams
from neural_racer_train.gpu_sim.safety_torch import SafetyTorch, curvature_speed, limits_only, nominal_angles

SUBSTEPS = int(round(CONTROL_DT / INTEGRATOR_DT))       # 5
IMU_EVERY = 4                                           # substeps: 20 ms, 50 Hz
RANGE_MIN, RANGE_MAX = 0.05, 30.0                       # sim scan validity, as the car's driver reports
# privileged layout: track 256, ego 8, others 18, race 4 (PRIV_BASE, the
# scripted drivers index into it), then the commands still in the actuator
# delay lines (4 steering, 3 speed), 5 unobserved car parameters, the race's
# cornering cap, and a flag per track point, 1 = known (model.mask_track
# zeroes the points a student could not know). The last two were added for
# teacher v4: a checkpoint from before them loads with model.load_expanded.
PRIV_BASE = 256 + 8 + 18 + 4
PRIV_OTHERS = slice(256 + 8, 256 + 8 + 18)              # 3 other cars x 6, nearest seats first
PRIV_PLACE = 256 + 8 + 18 + 1                           # place / 4
STEER_HIST, SPEED_HIST, PRIV_PARAMS = 4, 3, 5
PRIV_A_LAT = PRIV_BASE + STEER_HIST + SPEED_HIST + PRIV_PARAMS
PRIV_VIS = PRIV_A_LAT + 1
PRIV_DIM = PRIV_VIS + 64
STALL_SPEED, STALL_TICKS = 0.1, 40                      # m/s, ticks (1 s): a parked car has crashed
IDLE_TICKS = 120                                        # 3 s below STALL_SPEED for any reason: stalled too
EPISODE_REACTIVE, EPISODE_MAP, EPISODE_HANDOVER = 0, 1, 2


@dataclass
class Level:
    """One environment's episode: a track slot, a grid, a mode, a speed cap."""
    slot: int
    poses: np.ndarray                      # (n_cars, 3) rear-axle x, y, yaw
    start_idx: np.ndarray                  # (n_cars,) centreline indices
    episode_type: int = EPISODE_REACTIVE
    v_cap: float = 1.0
    v_cap_scale: np.ndarray | None = None  # (n_cars,) per-car share of v_cap (overtaking levels); None = all 1
    a_lat_cap: float = 2.0                 # m/s^2, the wrapper's curvature cap (WrapperConfig.max_lateral_accel)
    imu_present: bool = True
    ref_offset: np.ndarray = field(default_factory=lambda: np.zeros((3, 2)))  # harmonics: amplitude, phase
    slam_drift: float = 0.0                # m, constant lateral error of the reference
    slam_heading: float = 0.0              # rad, heading error of the reference
    level_id: int = -1

    @property
    def n_cars(self) -> int:
        return len(self.poses)


class BatchedRaceSim:

    def __init__(self, num_envs: int, max_cars: int, pool, car: CarParams = CarParams(),
                 device='cpu', seed: int = 0, randomize: bool = True, laps: int = 3,
                 max_seconds: float = 90.0, cast_limit: float = 12.0, cast_iters: int = 48,
                 reward_cfg: race_mod.RewardConfig = race_mod.RewardConfig(),
                 wrapper_cfg: WrapperConfig = WrapperConfig(max_speed=6.37), sensors: bool = True):
        E, C = num_envs, max_cars
        self.E, self.C, self.pool, self.car, self.device = E, C, pool, car, device
        # False: no LiDAR, IMU or observation history -- for a consumer of
        # privileged() alone (the teacher). The cast was 92% of a tick.
        self.sensors = sensors
        self.gen = torch.Generator(device=device).manual_seed(seed)   # on-device: no host copies per tick
        self.randomize = randomize
        self.laps, self.max_ticks = laps, int(round(max_seconds / CONTROL_DT))
        self.cast_limit, self.cast_iters = cast_limit, cast_iters
        self.reward_cfg = reward_cfg
        self.p = BatchParams.constant(car, E, device)
        self.angles = lidar.beam_angles(obs.N_BEAMS, device=device)
        self.safety = SafetyTorch(wrapper_cfg, nominal_angles(), device)
        self.safety_enabled = True      # False: actuator limits only (the teacher's default)
        # With the wrapper off, still hold the speed to the wrapper's
        # curvature cap: it needs only the steering, not the LiDAR (teacher v4)
        self.curvature_cap = False
        # True: an env's episode ends with seat 0's (the learner's) -- else it
        # runs until every car is done, and a parked opponent never is
        self.end_with_seat0 = False

        z = lambda *s, dt=torch.float32: torch.zeros(*s, device=device, dtype=dt)  # noqa: E731
        self.x = z(E, C, 7)
        self.active = z(E, C, dt=torch.bool)      # in this race
        self.alive = z(E, C, dt=torch.bool)       # still racing: not crashed, not finished
        self.slot = z(E, dt=torch.long)
        self.ep_type = z(E, dt=torch.long)
        self.v_cap = z(E)
        self.v_cap_scale = torch.ones(E, C, device=device)
        self.a_lat = torch.full((E,), 2.0, device=device)
        self.imu_on = z(E, dt=torch.bool)
        self.ref_offset = z(E, 3, 2)
        self.slam_drift, self.slam_heading = z(E), z(E)
        self.level_id = torch.full((E,), -1, dtype=torch.long, device=device)
        self.t = z(E, dt=torch.long)
        self.substep = z(E, dt=torch.long)
        self.prev_steer = z(E, C)
        self.stall_ticks = z(E, C, dt=torch.long)
        self.idle_ticks = z(E, C, dt=torch.long)
        self.touching = z(E, C, dt=torch.bool)    # in contact with a racing car last tick
        self.crashed = z(E, C, dt=torch.bool)     # out of the race by a crash (not by finishing)
        self.imu_held = z(E, C, 3)
        self.gyro_bias = z(E, C)
        self.scan_now = torch.full((E, C, obs.N_BEAMS), RANGE_MAX, device=device)
        self.scan_next = self.scan_now.clone()
        self.hist = ot.ObsHistory(E, C, device)
        self.race = race_mod.RaceState(E, C, device)
        self._make_delays()

    # --- parameters, per car -----------------------------------------------------------

    def _make_delays(self):
        E, C, d = self.E, self.C, self.device
        per_car = lambda v: v.view(E, 1).expand(E, C)     # noqa: E731
        self.steer_delay = actuation.DelayLine((E, C), 4, per_car(self.p.steer_delay_ticks), device=d)
        self.speed_delay = actuation.DelayLine((E, C), 3, per_car(self.p.speed_delay_ticks), device=d)
        self.imu_delay = actuation.DelayLine((E, C, 3), 2, per_car(self.p.imu_delay_ticks), device=d)

    def _pc(self):
        """BatchParams viewed (E, 1), to broadcast against (E, C)."""
        vals = {k: v.view(-1, 1) for k, v in vars(self.p).items() if isinstance(v, torch.Tensor)}
        return BatchParams(vals, self.device)

    # --- reset ---------------------------------------------------------------------------------

    def reset_envs(self, env_ids, levels):
        """Start a new episode in each env in env_ids, from its Level."""
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        n = len(levels)
        if self.randomize:
            fresh = BatchParams.randomized(self.car, n, self.gen, self.device)
        else:
            fresh = BatchParams.constant(self.car, n, self.device)
        self.p.assign(ids, fresh)
        mask = torch.zeros(self.E, dtype=torch.bool, device=self.device)
        mask[ids] = True
        idx0 = torch.zeros(self.E, self.C, dtype=torch.long, device=self.device)
        for k, (e, lv) in enumerate(zip(ids.tolist(), levels)):
            if lv.n_cars > self.C:
                raise ValueError(f'level has {lv.n_cars} cars; the sim holds {self.C}')
            self.slot[e] = lv.slot
            self.ep_type[e] = lv.episode_type
            self.v_cap[e] = lv.v_cap
            self.v_cap_scale[e] = 1.0
            if lv.v_cap_scale is not None:
                self.v_cap_scale[e, :lv.n_cars] = torch.as_tensor(lv.v_cap_scale, dtype=torch.float32)
            self.a_lat[e] = lv.a_lat_cap
            self.imu_on[e] = lv.imu_present
            self.ref_offset[e] = torch.as_tensor(lv.ref_offset, dtype=torch.float32)
            self.slam_drift[e], self.slam_heading[e] = lv.slam_drift, lv.slam_heading
            self.level_id[e] = lv.level_id
            self.active[e] = False
            self.active[e, :lv.n_cars] = True
            self.x[e] = 0.0
            lr = float(self.p.lr[e])
            for c in range(lv.n_cars):
                px, py, yaw = (float(v) for v in lv.poses[c])
                self.x[e, c, 0] = px + lr * math.cos(yaw)       # rear axle -> CoG
                self.x[e, c, 1] = py + lr * math.sin(yaw)
                self.x[e, c, 4] = yaw
                idx0[e, c] = int(lv.start_idx[c])
        self.alive[ids] = self.active[ids]
        self.t[ids] = 0
        self.substep[ids] = 0
        self.prev_steer[ids] = 0.0
        self.stall_ticks[ids] = 0
        self.idle_ticks[ids] = 0
        self.touching[ids] = False
        self.crashed[ids] = False
        self.imu_held[ids] = 0.0
        g = torch.randn(n, self.C, generator=self.gen, device=self.device)
        self.gyro_bias[ids] = g * self.p.imu_gyro_bias[ids].view(-1, 1)
        self._make_delays_for(ids)
        car_mask = mask.view(-1, 1).expand(self.E, self.C)
        self.hist.reset(car_mask)
        cx, cy = self.x[..., 0], self.x[..., 1]
        idx, s, _ = race_mod.project(self.pool, self.slot, cx, cy, idx0, window=5)
        self.race.reset(mask, idx, s)
        if self.sensors:
            ranges = self._scan(ids)            # only the envs being reset
            self.scan_now[ids], self.scan_next[ids] = ranges, ranges
            self._push_sensors(mask)

    def _make_delays_for(self, ids):
        for line, field_name in ((self.steer_delay, 'steer_delay_ticks'), (self.speed_delay, 'speed_delay_ticks'),
                                 (self.imu_delay, 'imu_delay_ticks')):
            d = getattr(self.p, field_name).long().clamp(0, line.max)
            line.delay[ids] = d[ids].view(-1, 1).expand(-1, self.C)
            line.reset(ids)

    # --- geometry --------------------------------------------------------------------------

    def _rear_axle(self):
        lr = self.p.lr.view(-1, 1)
        yaw = self.x[..., 4]
        return self.x[..., 0] - lr * torch.cos(yaw), self.x[..., 1] - lr * torch.sin(yaw), yaw

    def _scan(self, ids=None):
        """Ranges (E, C, B), or (len(ids), C, B) for those envs only."""
        x = self.x if ids is None else self.x[ids]
        slot = self.slot if ids is None else self.slot[ids]
        active = self.active if ids is None else self.active[ids]
        p = self.p if ids is None else self.p.index(ids)
        yaw = x[..., 4]
        fwd = (p.lidar_offset_x - p.lr).view(-1, 1)
        ox = x[..., 0] + fwd * torch.cos(yaw)
        oy = x[..., 1] + fwd * torch.sin(yaw)
        hx = torch.full_like(ox, self.car.body_length / 2)
        hy = torch.full_like(ox, self.car.body_width / 2)
        cars = (x[..., 0], x[..., 1], yaw, hx, hy, active)
        r = lidar.cast(self.pool, slot, ox, oy, yaw, self.angles, cars,
                       cast_limit=self.cast_limit, iters=self.cast_iters)
        lost = lidar.sector_dropout(r.shape, p.lidar_sectors, p.lidar_persist, p.lidar_persist_deg,
                                    p.lidar_persist_width, self.angles, self.gen)
        return lidar.add_noise(r, p.lidar_noise_std, p.lidar_dropout_rate, self.gen, RANGE_MAX, lost)

    def _push_sensors(self, mask=None):
        enc, valid = ot.pool_scan(self.scan_now, RANGE_MIN, RANGE_MAX)
        n = torch.randn(self.E, self.C, generator=self.gen, device=self.device)
        speed = self.x[..., 3] * (1.0 + n * self.p.odom_speed_noise.view(-1, 1))
        # The delay line is shared by every env: only a real tick may advance it.
        # A reset env's first entry is its (zeroed) held sample, undelayed.
        imu = self.imu_delay.push(self.imu_held) if mask is None else self.imu_held
        if mask is None:
            self.hist.push_tick(enc, valid, speed, imu)
        else:
            # only the envs being reset get a first entry
            saved = {k: getattr(self.hist, k).clone() for k in ('scans', 'valid', 'speeds', 'imu', 'count')}
            self.hist.push_tick(enc, valid, speed, imu)
            keep = ~mask
            for k, v in saved.items():
                getattr(self.hist, k)[keep] = v[keep]

    # --- what the policies see -----------------------------------------------------------

    def mode(self):
        laps_done = torch.floor(self.race.total / self.pool.length[self.slot].view(-1, 1))
        handover = (self.ep_type == EPISODE_HANDOVER).view(-1, 1) & (laps_done >= 1)
        return ((self.ep_type == EPISODE_MAP).view(-1, 1) | handover).long().expand(self.E, self.C)

    def _centre_at(self, s):
        """Interpolated centreline (x, y, wl, wr, nx, ny) at arc lengths s (E, C, K)."""
        E = self.E
        n = self.pool.n_points[self.slot].view(E, 1, 1).float()
        L = self.pool.length[self.slot].view(E, 1, 1)
        u = torch.remainder(s, L) / L * n
        i0 = torch.floor(u).long()
        f = (u - i0).unsqueeze(-1)
        i0 = torch.remainder(i0, n.long())
        i1 = torch.remainder(i0 + 1, n.long())
        tab = self.pool.centre[self.slot]
        g = lambda i: torch.gather(tab, 1, i.reshape(E, -1, 1).expand(-1, -1, 5)).reshape(*s.shape, 5)  # noqa: E731
        a, b = g(i0), g(i1)
        p = a * (1 - f) + b * f
        t = b[..., :2] - a[..., :2]
        t = t / torch.linalg.norm(t, dim=-1, keepdim=True).clamp(min=1e-9)
        return p[..., 0], p[..., 1], p[..., 3], p[..., 4], -t[..., 1], t[..., 0]

    def _preview_and_race(self):
        E, C = self.E, self.C
        L = self.pool.length[self.slot].view(E, 1, 1)
        s = self.race.s.unsqueeze(-1) + obs.PREVIEW_SPACING * torch.arange(
            1, obs.PREVIEW_POINTS + 1, device=self.device).view(1, 1, -1)
        cx, cy, wl, wr, nx, ny = self._centre_at(s)
        off = torch.zeros_like(cx)
        for k in range(3):
            amp, ph = self.ref_offset[:, k, 0].view(E, 1, 1), self.ref_offset[:, k, 1].view(E, 1, 1)
            off = off + amp * torch.sin(2 * math.pi * (k + 1) * s / L + ph)
        off = torch.maximum(torch.minimum(off, 0.4 * torch.minimum(wl, wr)), -0.4 * torch.minimum(wl, wr))
        off = off + self.slam_drift.view(E, 1, 1)
        px, py = cx + nx * off, cy + ny * off
        u = torch.rand(cx.shape, generator=self.gen, device=self.device)
        wl2, wr2 = (wl - off) * (1 - 0.1 * u), (wr + off) * (1 - 0.1 * u)
        bx, by, yaw = self._rear_axle()
        yaw = yaw + self.slam_heading.view(E, 1)
        dx, dy = px - bx.unsqueeze(-1), py - by.unsqueeze(-1)
        c, sn = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
        prev = torch.stack([(c * dx + sn * dy) / obs.PREVIEW_XY_SCALE, (-sn * dx + c * dy) / obs.PREVIEW_XY_SCALE,
                            wl2.clamp(min=0) / obs.PREVIEW_WIDTH_SCALE, wr2.clamp(min=0) / obs.PREVIEW_WIDTH_SCALE], dim=2)
        race = torch.zeros(E, C, obs.RACE_DIM, device=self.device)
        race[..., 0] = torch.remainder(self.race.s, L.view(E, 1)) / L.view(E, 1)
        race[..., 1] = 0.25          # v1: the car's node sends place 1 of 4, no gaps (plan B0 note)
        return prev, race

    def car_v_cap(self):
        """(E, C) each car's speed cap: the race's, times its share on an overtaking level."""
        return self.v_cap.view(-1, 1) * self.v_cap_scale

    def observe(self):
        prev, race = self._preview_and_race()
        imu_present = self.imu_on.view(-1, 1).expand(self.E, self.C)
        v_cap = self.car_v_cap()
        a_lat = self.a_lat.view(-1, 1).expand(self.E, self.C)
        return ot.build(self.hist, imu_present, self.mode(), v_cap, prev, race, a_lat=a_lat)

    def privileged(self):
        E, C = self.E, self.C
        s = self.race.s.unsqueeze(-1) + 0.25 * torch.arange(1, 65, device=self.device).view(1, 1, -1)
        cx, cy, wl, wr, _, _ = self._centre_at(s)
        bx, by, yaw = self._rear_axle()
        dx, dy = cx - bx.unsqueeze(-1), cy - by.unsqueeze(-1)
        c, sn = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
        track = torch.cat([(c * dx + sn * dy) / 16, (-sn * dx + c * dy) / 16, wl / 2, wr / 2], dim=-1)
        ego = torch.stack([
            self.x[..., 3] / 5, self.x[..., 6], self.x[..., 5] / 3, self.x[..., 2] / 0.33,
            self.p.mu.view(-1, 1).expand(E, C), self.p.steer_delay_ticks.view(-1, 1).expand(E, C) / 4,
            self.p.speed_delay_ticks.view(-1, 1).expand(E, C) / 4, self.car_v_cap() / 5], dim=-1)
        others = []
        for j in range(1, C):
            k = (torch.arange(C, device=self.device) + j) % C
            ox, oy = bx[:, k] - bx, by[:, k] - by
            oyaw = yaw[:, k] - yaw
            others.append(torch.stack([
                (torch.cos(yaw) * ox + torch.sin(yaw) * oy) / 10, (-torch.sin(yaw) * ox + torch.cos(yaw) * oy) / 10,
                torch.sin(oyaw), torch.cos(oyaw), self.x[:, k, 3] / 5, self.active[:, k].float()], dim=-1))
        while len(others) < 3:
            others.append(torch.zeros(E, C, 6, device=self.device))
        L = self.pool.length[self.slot].view(E, 1)
        racef = torch.stack([torch.remainder(self.race.s, L) / L, self.race.place(self.active).float() / 4,
                             (self.race.total / L) / self.laps,
                             (1 - self.t.float() / self.max_ticks).view(-1, 1).expand(E, C)], dim=-1)
        # Commands not yet applied: with 2-4 ticks of steering delay and 1-3 of
        # speed delay, the state without them is not Markov for a memoryless net.
        steer_hist = self.steer_delay.recent(STEER_HIST).permute(1, 2, 0) / obs.STEER_LIMIT
        speed_hist = self.speed_delay.recent(SPEED_HIST).permute(1, 2, 0) / 5
        b, rel = self.car, lambda v, base: (v / base - 1) / 0.3        # noqa: E731 -- RANDOMIZATION's +/-30%
        params = torch.stack([(self.p.steer_gain - 1) / 0.1, rel(self.p.C_Sf, b.C_Sf), rel(self.p.C_Sr, b.C_Sr),
                              rel(self.p.servo_gain, b.servo_gain), rel(self.p.I, b.I)], dim=-1)
        return torch.cat([track, ego, *others[:3], racef, steer_hist, speed_hist,
                          params.view(E, 1, PRIV_PARAMS).expand(E, C, PRIV_PARAMS),
                          self.a_lat.view(E, 1, 1).expand(E, C, 1) / 5,
                          torch.ones(E, C, 64, device=self.device)], dim=-1)

    def track_visibility(self):
        """(E, C, 64) bool: which of privileged()'s track points each car's
        LiDAR has in line of sight now -- inside its 270 degrees and with no
        wall in between (plan item 2.3)."""
        E = self.E
        s = self.race.s.unsqueeze(-1) + 0.25 * torch.arange(1, 65, device=self.device).view(1, 1, -1)
        px, py, _, _, _, _ = self._centre_at(s)
        yaw = self.x[..., 4]
        fwd = (self.p.lidar_offset_x - self.p.lr).view(-1, 1)
        ox, oy = self.x[..., 0] + fwd * torch.cos(yaw), self.x[..., 1] + fwd * torch.sin(yaw)
        bearing = actuation.wrap_angle(torch.atan2(py - oy.unsqueeze(-1), px - ox.unsqueeze(-1)) - yaw.unsqueeze(-1))
        seen = lidar.line_of_sight(self.pool, self.slot.view(E, 1, 1), ox.unsqueeze(-1), oy.unsqueeze(-1), px, py)
        return seen & (bearing.abs() <= math.radians(135.0))

    def opponent_targets(self):
        """(E, C, 4, 13) ground truth for the opponent forecast head: the other
        cars sorted by distance, presence, then positions now in the car frame.
        (The future positions are filled in by the trainer from later ticks.)"""
        E, C = self.E, self.C
        bx, by, yaw = self._rear_axle()
        out = torch.zeros(E, C, obs.OPP_SLOTS, 3, device=self.device)
        d = torch.cdist(torch.stack([bx, by], -1), torch.stack([bx, by], -1))
        d = d + torch.eye(C, device=self.device) * 1e6
        d = torch.where(self.active.unsqueeze(1), d, torch.full_like(d, 1e6))
        order = d.argsort(-1)[..., :obs.OPP_SLOTS]
        for slot in range(min(obs.OPP_SLOTS, C - 1)):
            k = order[..., slot]
            ox = torch.gather(bx, 1, k) - bx
            oy = torch.gather(by, 1, k) - by
            present = torch.gather(d, 2, k.unsqueeze(-1)).squeeze(-1) < 1e5
            out[..., slot, 0] = present.float()
            out[..., slot, 1] = torch.cos(yaw) * ox + torch.sin(yaw) * oy
            out[..., slot, 2] = -torch.sin(yaw) * ox + torch.cos(yaw) * oy
        return out

    # --- one control tick -------------------------------------------------------------------

    def step(self, actions):
        """actions (E, C, 2): the networks' (steering, speed). Returns a dict of (E, C) / (E,) tensors."""
        E, C = self.E, self.C
        pc = self._pc()
        alive_before = self.alive.clone()
        v_cap = self.car_v_cap()
        a_lat = self.a_lat.view(-1, 1).expand(E, C)
        if self.safety_enabled and not self.sensors:
            raise RuntimeError('the safety wrapper reads the LiDAR: build the sim with sensors=True')
        if self.safety_enabled:
            steer, speed, stopped, intervened, safety_stop, cut = self.safety.apply_detailed(
                actions[..., 0], actions[..., 1], self.scan_now, RANGE_MIN, RANGE_MAX, self.prev_steer, CONTROL_DT, v_cap,
                max_lateral_accel=a_lat)
        else:
            steer, speed, stopped, intervened, safety_stop, cut = limits_only(
                self.safety.cfg, actions[..., 0], actions[..., 1], self.prev_steer, CONTROL_DT, v_cap)
        # m/s the curvature cap took off the (clipped) request
        curv_cut = torch.zeros_like(speed)
        if self.safety_enabled or self.curvature_cap:
            asked = torch.minimum(actions[..., 1].clamp(min=0.0), v_cap)
            curve = curvature_speed(steer, self.safety.cfg.wheelbase, a_lat)
            curv_cut = (asked - torch.minimum(asked, curve)).clamp(min=0.0)
            if not self.safety_enabled:
                speed = torch.minimum(speed, curve)
        steer = torch.where(self.alive, steer, torch.zeros_like(steer))
        speed = torch.where(self.alive, speed, torch.zeros_like(speed))
        d_steer = steer - self.prev_steer
        self.prev_steer = steer

        d_s = self.steer_delay.push(steer)
        d_v = self.speed_delay.push(speed)
        u = actuation.actuate(d_s, d_v, self.x[..., 3], self.x[..., 2], pc)
        dist = (self.p.imu_offset_x - self.p.lr).view(-1, 1)
        for _ in range(SUBSTEPS):
            self.x = rk4(self.x, u, pc, INTEGRATOR_DT)
            self.substep += 1
            if not self.sensors:
                continue
            # Computed every substep and masked, rather than `if take.any():` --
            # that test is a GPU->host sync, five times a tick.
            take = (self.substep % IMU_EVERY == 0).view(-1, 1)
            f = st_derivative(self.x, u, pc)
            gz, ax, ay = imu_mod.specific_force(self.x, f, dist)
            noise = torch.randn(E, C, 3, generator=self.gen, device=self.device)
            std = imu_mod.noise_std(self.p.imu_gyro_noise.view(-1, 1), self.p.imu_accel_noise.view(-1, 1),
                                    self.p.imu_vibration.view(-1, 1), self.x[..., 3],
                                    self.p.imu_gyro_vibration.view(-1, 1))
            sample = torch.stack([gz + self.gyro_bias, ax, ay], dim=-1) + noise * std
            self.imu_held = torch.where(take.unsqueeze(-1), sample, self.imu_held)
        self.x[..., 4] = actuation.wrap_angle(self.x[..., 4])
        bad = ~torch.isfinite(self.x).all(-1)
        self.x = torch.where(bad.unsqueeze(-1), torch.zeros_like(self.x), self.x)

        yaw = self.x[..., 4]
        hx = torch.full_like(yaw, self.car.body_length / 2)
        hy = torch.full_like(yaw, self.car.body_width / 2)
        wall = contact.wall_contact(self.pool, self.slot, self.x[..., 0], self.x[..., 1], yaw, hx, hy) & self.active
        vx = self.x[..., 3] * torch.cos(yaw + self.x[..., 6])
        vy = self.x[..., 3] * torch.sin(yaw + self.x[..., 6])
        touch, fault = contact.car_contacts(self.x[..., 0], self.x[..., 1], yaw, vx, vy, hx, hy, self.active)
        # The no-fault contact cost is per contact, against cars still racing:
        # charged every tick, a car resting against a parked car or a wreck
        # lost 2 a tick (a stopped car in traffic averaged -73 a race,
        # 2026-10-30 audit). Running into a wreck is still a fault, above.
        racing, _ = contact.car_contacts(self.x[..., 0], self.x[..., 1], yaw, vx, vy, hx, hy,
                                         self.active & alive_before)
        new_touch = racing & ~self.touching
        self.touching = racing

        bx, by, _ = self._rear_axle()
        idx, s, _ = race_mod.project(self.pool, self.slot, self.x[..., 0], self.x[..., 1], self.race.idx)
        ds, gained = self.race.update(idx, s, self.pool.length[self.slot], self.alive, self.active)
        L = self.pool.length[self.slot].view(-1, 1)
        done_laps = (self.race.total >= self.laps * L) & self.alive
        n_cars = self.active.sum(-1, keepdim=True).float()
        place = self.race.place(self.active).float()
        # a solo car that finishes gets the winner's bonus: the laps are the race
        bonus = torch.where(n_cars > 1, self.reward_cfg.finish * (n_cars - place) / (n_cars - 1).clamp(min=1),
                            torch.full_like(place, self.reward_cfg.finish))
        cut_frac = torch.where(stopped, torch.ones_like(cut), (cut / self.reward_cfg.v_ref).clamp(max=1.0))
        cut_frac = cut_frac * alive_before.float()
        curv_frac = (curv_cut / self.reward_cfg.v_ref).clamp(max=1.0) * alive_before.float()
        r = race_mod.reward(self.reward_cfg, ds, gained, fault & alive_before, new_touch & alive_before,
                            wall & alive_before, cut_frac, d_steer, CONTROL_DT, wall_speed=self.x[..., 3].abs(),
                            curv_frac=curv_frac)
        r = r + torch.where(done_laps, bonus, torch.zeros_like(bonus))
        # Stalled: held still by the wrapper for 1 s (without the wrapper, any
        # car still for 1 s), or still for 3 s for any reason -- with the
        # wrapper, stopping by choice was a free move: never ended, never charged
        slow = alive_before & (self.x[..., 3].abs() < STALL_SPEED)
        held = (safety_stop if self.safety_enabled else alive_before) & slow
        self.stall_ticks = torch.where(held, self.stall_ticks + 1, torch.zeros_like(self.stall_ticks))
        self.idle_ticks = torch.where(slow, self.idle_ticks + 1, torch.zeros_like(self.idle_ticks))
        stalled = (self.stall_ticks >= STALL_TICKS) | (self.idle_ticks >= IDLE_TICKS)
        r = r - self.reward_cfg.stall * stalled.float()
        crashed = alive_before & (wall | fault | bad | stalled)
        r = torch.where(alive_before, r, torch.zeros_like(r))
        self.alive = self.alive & ~crashed & ~done_laps
        self.crashed |= crashed

        self.t += 1
        timeout = self.t >= self.max_ticks
        env_done = timeout | ~self.alive.any(-1)
        if self.end_with_seat0:
            env_done = env_done | ~self.alive[:, 0]
        if self.sensors:
            # sensors for the next tick: the scan is one tick stale, like the car's
            self.scan_now = self.scan_next
            self.scan_next = self._scan()
            self.hist.push_command(steer, speed)
            self._push_sensors()
        return {'reward': r, 'terminal': crashed | done_laps, 'alive': alive_before,
                'env_done': env_done, 'timeout': timeout, 'wall': wall, 'fault': fault, 'touch': touch,
                'intervened': intervened & alive_before, 'safety_stop': safety_stop & alive_before,
                'stalled': stalled, 'brake_cut': cut_frac, 'curv_cut': curv_cut * alive_before.float(),
                'finished': done_laps, 'progress': ds, 'crashed': self.crashed.clone(),
                'place': place, 'speed': self.x[..., 3]}
