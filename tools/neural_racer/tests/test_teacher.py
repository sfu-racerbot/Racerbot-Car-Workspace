"""The teacher's training regime: sensor-free simulation, the privileged
delay lines, the cornering cap (teacher v4), DAgger warm start, PLR scoring
and spread, the normalisers, the track library and the held-out race.

Runner: PYTHONPATH=<torch> python3 -m pytest tools/neural_racer/tests/test_teacher.py -v   (CPU)
"""
import math

import conftest  # noqa: F401
import numpy as np
import pytest
import torch

from neural_racer.safety_wrapper import WrapperConfig
from neural_racer_train import learn, plr, ppo, raceline_ref, scripted, tracklib, tracks
from neural_racer_train.gpu_sim import race
from neural_racer_train.gpu_sim.params import CarParams
from neural_racer_train.gpu_sim.safety_torch import SafetyTorch, curvature_speed, limits_only, nominal_angles
from neural_racer_train.gpu_sim.sim import PRIV_A_LAT, PRIV_BASE, PRIV_DIM, PRIV_VIS, STALL_TICKS, BatchedRaceSim
from neural_racer_train.learn import Runner
from neural_racer_train.model import (Critic, RunningNorm, TeacherNet, capped_squash, load_expanded, mask_track,
                                      squash)

torch.set_num_threads(4)


def _runner(n=4, **kw):
    kw.setdefault('pool_slots', 4)
    kw.setdefault('initial_tracks', 2)
    return Runner(n, 1, 'cpu', 0, solo=True, **kw)


# --- the simulator -------------------------------------------------------------------------------

def test_sensors_off_leaves_the_dynamics_alone():
    """Same seed, same commands: a sim without LiDAR and IMU drives the same
    cars to the same places with the same rewards (until a reset, where the
    two noise streams have been drawn differently)."""
    a, b = _runner(), _runner(sensors=False)
    a.sim.safety_enabled = b.sim.safety_enabled = False
    act = torch.tensor([[[0.05, 1.5]], [[-0.03, 2.0]], [[0.0, 1.0]], [[0.1, 2.5]]])
    for _ in range(30):
        oa, ob = a.sim.step(act), b.sim.step(act)
        assert torch.equal(oa['reward'], ob['reward']) and torch.equal(oa['terminal'], ob['terminal'])
    assert torch.equal(a.sim.x, b.sim.x)


def test_safety_wrapper_needs_the_lidar():
    r = _runner(sensors=False)
    with pytest.raises(RuntimeError, match='LiDAR'):
        r.sim.step(torch.zeros(4, 1, 2))           # the wrapper is on by default


def test_a_reset_scans_only_its_envs_and_gets_them_right():
    r = _runner()
    quiet = CarParams(lidar_noise_std=0.0, lidar_dropout_rate=0.0, lidar_sectors=0.0)
    sim = BatchedRaceSim(4, 1, r.pool, car=quiet, randomize=False, seed=1)
    s0, s1 = sorted(r.slot_digest)
    lv = [r.maker.make(s, 7 + k, k) for k, s in enumerate([s0, s1, s0, s1])]
    sim.reset_envs([0, 1, 2, 3], lv)
    sim.reset_envs([1, 2], [r.maker.make(s0, 99, 9), r.maker.make(s1, 98, 8)])
    assert torch.allclose(sim.scan_now[[1, 2]], sim._scan()[[1, 2]], atol=1e-5)


def test_privileged_carries_the_delay_lines():
    r = _runner(sensors=False)
    r.sim.safety_enabled = False
    cmds = [0.01, 0.02, 0.03, 0.04]                # small enough that the slew limit never binds
    for c in cmds:
        r.sim.step(torch.tensor([[[c, 1.0]]] * 4))
    priv = r.sim.privileged()[:, 0]
    assert priv.shape == (4, PRIV_DIM)
    assert (priv[0, PRIV_BASE:PRIV_BASE + 4] * 0.33).tolist() == pytest.approx(cmds[::-1], abs=1e-6)
    assert (priv[0, PRIV_BASE + 4:PRIV_BASE + 7] * 5).tolist() == pytest.approx([1.0] * 3, abs=1e-6)


def test_sim_without_safety_stalls_parked_cars():
    r = _runner(sensors=False)
    r.sim.safety_enabled = False
    stalled = False
    for _ in range(STALL_TICKS + 2):
        stalled |= bool(r.sim.step(torch.zeros(4, 1, 2))['stalled'].any())
    assert stalled


def test_limits_only_slews_but_never_brakes():
    steer, speed, stopped, interv, stop, cut = limits_only(
        WrapperConfig(), torch.tensor([[0.3]]), torch.tensor([[9.0]]), torch.zeros(1, 1), 0.025,
        torch.full((1, 1), 5.0))
    assert float(steer) == pytest.approx(0.05) and float(speed) == pytest.approx(5.0)
    assert not (stopped | interv | stop).any() and float(cut) == 0.0


def test_brake_cut_grows_with_the_request():
    st = SafetyTorch(WrapperConfig(), nominal_angles())
    near = torch.full((1, 2, len(nominal_angles())), 0.8)
    speed = torch.tensor([[0.8, 3.0]])
    *_, stop, cut = st.apply_detailed(torch.zeros(1, 2), speed, near, 0.05, 30.0, torch.zeros(1, 2), 0.025,
                                      torch.full((1, 2), 6.0))
    assert float(cut[0, 1]) > float(cut[0, 0]) >= 0.0


def test_wall_cost_scales_with_speed():
    cfg = race.RewardConfig()
    z, t = torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool)
    f, g = ~t, torch.zeros(1, 1, dtype=torch.long)
    slow = race.reward(cfg, z, g, f, f, t, f, z, 0.025, wall_speed=torch.zeros(1, 1))
    fast = race.reward(cfg, z, g, f, f, t, f, z, 0.025, wall_speed=torch.full((1, 1), cfg.v_ref))
    assert float(slow) == pytest.approx(-cfg.wall_min) and float(fast) == pytest.approx(-cfg.wall)


def test_training_tracks_have_no_obstacles():
    assert len(learn._generate(5).spec.obstacles) == 0


# --- teacher v4: the cornering cap ---------------------------------------------------------------

def _steer_until_settled(sim, cmd, ticks=12):
    out = None
    for _ in range(ticks):           # the slew limit reaches the command in 7 ticks
        out = sim.step(cmd)
    return out


def test_sim_holds_the_curvature_cap_without_the_wrapper():
    r = _runner(sensors=False)
    r.sim.safety_enabled, r.sim.curvature_cap = False, True
    r.sim.a_lat[:] = torch.tensor([2.0, 2.0, 5.0, 5.0])
    r.sim.v_cap[:] = 6.0
    out = _steer_until_settled(r.sim, torch.tensor([[[0.3, 6.0]]] * 4))
    applied = r.sim.speed_delay.recent(1)[0, :, 0]
    want = torch.sqrt(r.sim.a_lat * 0.36 / math.tan(0.3))
    assert torch.allclose(applied, want, rtol=1e-5)
    assert (out['curv_cut'][:, 0] > 1.0).all()
    r.sim.curvature_cap = False                 # teacher-v3's regime: the request goes through
    _steer_until_settled(r.sim, torch.tensor([[[0.3, 6.0]]] * 4), 2)
    assert torch.allclose(r.sim.speed_delay.recent(1)[0, :, 0], torch.full((4,), 6.0))


def test_capped_squash_never_asks_more_than_the_cap():
    torch.manual_seed(0)
    n = 4096
    a = torch.randn(n, 2) * 3
    priv = torch.zeros(n, PRIV_DIM)
    prev = (torch.rand(n) * 2 - 1) * 0.33
    priv[:, PRIV_BASE] = prev / 0.33
    priv[:, PRIV_A_LAT] = (2.0 + 3.0 * torch.rand(n)) / 5
    v_cap = 1.0 + 5.37 * torch.rand(n)
    cmd = capped_squash(a, v_cap, priv)
    steer, _, _, _, _, _ = limits_only(WrapperConfig(), cmd[:, 0:1], cmd[:, 1:2], prev.view(-1, 1), 0.025,
                                       v_cap.view(-1, 1))
    top = torch.minimum(v_cap, curvature_speed(steer[:, 0], 0.36, priv[:, PRIV_A_LAT] * 5))
    assert (cmd[:, 1] <= top * (1 + 1e-6)).all() and (cmd[:, 1] >= 0).all()
    assert (cmd[:, 1] < 0.99 * v_cap).any()                       # and it does bind
    straight = capped_squash(torch.tensor([[0.0, 0.7]]), torch.tensor([3.0]), torch.zeros(1, PRIV_DIM))
    assert torch.allclose(straight, squash(torch.tensor([[0.0, 0.7]]), torch.tensor([3.0])))


def test_capped_commands_are_never_cut():
    r = _runner(sensors=False)
    r.sim.safety_enabled, r.sim.curvature_cap = False, True
    a = torch.tensor([[2.0, 4.0], [-2.0, 4.0], [0.5, 4.0], [1.0, -1.0]])
    for _ in range(12):
        priv = r.sim.privileged()[:, 0]
        out = r.sim.step(capped_squash(a, r.sim.v_cap, priv).unsqueeze(1))
        assert float(out['curv_cut'].max()) < 1e-4


def test_curvature_penalty_charges_the_cut():
    act = torch.tensor([[[0.3, 6.0]]] * 4)
    rewards = []
    for w in (0.0, 0.5):
        r = _runner(sensors=False, reward_cfg=race.RewardConfig(curvature=w))
        r.sim.safety_enabled, r.sim.curvature_cap = False, True
        r.sim.v_cap[:] = 6.0
        out = _steer_until_settled(r.sim, act)
        rewards.append((out['reward'][:, 0], out['curv_cut'][:, 0]))
    (r0, cut), (r1, _) = rewards
    assert torch.allclose(r0 - r1, 0.5 * (cut / 4.0).clamp(max=1.0), atol=1e-5)
    assert (cut > 0).all()


def test_privileged_carries_the_cornering_cap():
    r = _runner(sensors=False)
    caps = torch.tensor([r.maker.make(int(r.sim.slot[e]), 3 + e, e).a_lat_cap for e in range(4)])
    r.sim.reset_envs(list(range(4)), [r.maker.make(int(r.sim.slot[e]), 3 + e, e) for e in range(4)])
    assert torch.allclose(r.sim.privileged()[:, 0, PRIV_A_LAT] * 5, caps)
    assert torch.allclose(r.sim.a_lat, caps)


def test_levels_draw_cornering_caps_without_moving_their_other_draws():
    r = _runner(sensors=False)
    slot = next(iter(r.slot_digest))
    m, fixed = r.maker, learn.LevelMaker(1, solo=True, a_lat_range=(2.0, 2.0))
    fixed.specs = m.specs
    lv = [m.make(slot, s, s) for s in range(400)]
    caps = np.array([v.a_lat_cap for v in lv])
    assert caps.min() == 2.0 and caps.max() <= 5.0
    assert np.mean(caps == 2.0) == pytest.approx(learn.A_LAT_CAR_SHARE, abs=0.08)
    assert all(fixed.make(slot, s, s).a_lat_cap == 2.0 for s in range(20))
    assert [v.v_cap for v in lv[:20]] == [fixed.make(slot, s, s).v_cap for s in range(20)]


def test_a_narrower_checkpoint_widens_to_the_same_network():
    """teacher-v3's shapes: no cornering-cap column, no track-flag channel."""
    torch.manual_seed(1)
    net = TeacherNet()
    with torch.no_grad():
        net.enc.state[0].weight[:, -1] = 0.0          # what the widening gives the new inputs
        net.enc.track[0].weight[:, 4] = 0.0
        net.enc.norm.mean.normal_()
        net.enc.norm.var.uniform_(0.5, 2.0)
    full = net.state_dict()
    old = {k: (v[..., :-1] if k in ('enc.state.0.weight', 'enc.norm.mean', 'enc.norm.var', 'enc.keep') else v)
           for k, v in full.items()}
    old['enc.track.0.weight'] = full['enc.track.0.weight'][:, :4]
    wide = TeacherNet()
    assert sorted(load_expanded(wide, old)) == ['enc.keep', 'enc.norm.mean', 'enc.norm.var', 'enc.state.0.weight',
                                                'enc.track.0.weight']
    x = torch.randn(5, PRIV_DIM)
    assert torch.allclose(wide(x), net(x), atol=1e-6)
    y = x.clone()
    y[:, PRIV_A_LAT] += 3.0
    y[:, PRIV_VIS:] = torch.rand(5, 64)
    assert torch.equal(wide(x), wide(y))               # the new inputs change nothing until trained
    with pytest.raises(ValueError, match='does not widen'):
        load_expanded(TeacherNet(), {**old, 'head.0.weight': torch.zeros(3, 3)})


def test_mask_track_hides_only_unseen_points_of_masked_rows():
    priv = torch.rand(2, PRIV_DIM) + 0.5
    vis = torch.ones(2, 64, dtype=torch.bool)
    vis[:, 40:] = False
    out = mask_track(priv, vis, torch.tensor([True, False]))
    for ch in range(4):
        assert (out[0, ch * 64 + 40:ch * 64 + 64] == 0).all() and torch.equal(out[0, ch * 64:ch * 64 + 40],
                                                                               priv[0, ch * 64:ch * 64 + 40])
    assert out[0, PRIV_VIS:PRIV_VIS + 40].eq(1).all() and out[0, PRIV_VIS + 40:].eq(0).all()
    assert torch.equal(out[1, :PRIV_VIS], priv[1, :PRIV_VIS]) and out[1, PRIV_VIS:].eq(1).all()
    assert torch.equal(out[:, 256:PRIV_VIS], priv[:, 256:PRIV_VIS])       # the car's own state untouched


def test_track_visibility_matches_a_dense_march():
    r = _runner(8, sensors=False)
    sim = r.sim
    vis = sim.track_visibility()[:, 0]
    assert vis[:, 0].all()                              # 25 cm ahead of a car on the centreline
    s = sim.race.s[:, :1] + 0.25 * torch.arange(1, 65).view(1, -1)
    px, py, *_ = sim._centre_at(s.unsqueeze(1))
    yaw = sim.x[:, 0, 4]
    fwd = sim.p.lidar_offset_x - sim.p.lr
    ox, oy = sim.x[:, 0, 0] + fwd * torch.cos(yaw), sim.x[:, 0, 1] + fwd * torch.sin(yaw)
    f = torch.linspace(0, 1, 400).view(1, 1, -1)
    qx = ox.view(-1, 1, 1) + (px[:, 0].unsqueeze(-1) - ox.view(-1, 1, 1)) * f
    qy = oy.view(-1, 1, 1) + (py[:, 0].unsqueeze(-1) - oy.view(-1, 1, 1)) * f
    clear = (r.pool.sample(sim.slot.view(-1, 1, 1), qx, qy) >= 0.005).all(-1)
    bearing = torch.atan2(py[:, 0] - oy.view(-1, 1), px[:, 0] - ox.view(-1, 1)) - yaw.view(-1, 1)
    in_fov = torch.cos(bearing) >= math.cos(math.radians(135.0))
    ref = clear & in_fov
    false_seen, agree = float((vis & ~ref).float().mean()), float((vis == ref).float().mean())
    assert false_seen < 0.01 and agree > 0.93, (false_seen, agree)     # errs toward hiding, never showing


def test_set_columns_takes_the_sample_statistics():
    n = RunningNorm(3)
    n.update(torch.randn(100, 3))
    x = torch.randn(500, 3) * 2 + 5
    n.set_columns([2], x)
    assert float(n.mean[2]) == pytest.approx(float(x[:, 2].mean()), abs=1e-5)
    assert float(n.var[2]) == pytest.approx(float(x[:, 2].var(unbiased=False)), rel=1e-4)


def test_raceline_pace_from_rest_on_a_circle():
    R, n, laps = 5.0, 600, 3
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    line = np.stack([R * np.cos(th), R * np.sin(th)], axis=1)
    ref = {'line': line, 'ds': np.hypot(*(np.roll(line, -1, axis=0) - line).T), 'kappa': raceline_ref.curvature(line)}
    assert np.allclose(ref['kappa'], 1 / R, rtol=1e-3)
    v = math.sqrt(2.0 * R)                              # the 2.0 m/s^2 cap, grip to spare at mu 1
    L = laps * ref['ds'].sum()
    d_acc = v * v / (2 * raceline_ref.ACCEL)
    want = v / raceline_ref.ACCEL + (L - d_acc) / v
    assert raceline_ref.race_time(ref, line[0], 6.0, 2.0, 1.0, laps) == pytest.approx(want, rel=0.01)
    slower = raceline_ref.race_time(ref, line[0], 2.0, 2.0, 1.0, laps)      # v_cap below the bend speed
    assert slower > raceline_ref.race_time(ref, line[0], 6.0, 2.0, 1.0, laps)


def test_raceline_hugs_the_outside_of_a_circle():
    pytest.importorskip('scipy')
    R, w = 5.0, 1.0
    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    centre = np.stack([R * np.cos(th), R * np.sin(th)], axis=1)
    spec = tracks.TrackSpec('circle', 0, centre, np.full(400, w), np.full(400, w))
    ref = raceline_ref.optimize(spec)
    r = np.hypot(*ref['line'].T)
    assert float(np.median(r)) == pytest.approx(R + w - raceline_ref.HALF_WIDTH - raceline_ref.MARGIN, abs=0.05)


# --- the scripted driver the teacher clones --------------------------------------------------------

def _priv_bend_at(d: float, radius: float = 1.5):
    """Straight ahead for d metres, then a left bend of `radius`."""
    p = torch.zeros(1, PRIV_DIM)
    s = 0.25 * torch.arange(1, 65, dtype=torch.float32)
    th = (s - d).clamp(min=0) / radius
    x = torch.where(s <= d, s, d + radius * torch.sin(th))
    y = torch.where(s <= d, torch.zeros_like(s), radius * (1 - torch.cos(th)))
    p[0, 0:64], p[0, 64:128], p[0, 128:256] = x / 16, y / 16, 0.5
    p[0, 256], p[0, scripted.MU] = 1.0, 0.8
    return p


def test_pursuit_brakes_for_a_bend_beyond_the_old_4m_window():
    v_cap = torch.tensor([6.37])
    far = scripted.pursue(_priv_bend_at(6.0), v_cap, torch.zeros(1), torch.ones(1))[0, 1]
    near = scripted.pursue(_priv_bend_at(2.0), v_cap, torch.zeros(1), torch.ones(1))[0, 1]
    v_bend = math.sqrt(3.0 * 1.5)                      # a_lat = 3 m/s^2 on the 1.5 m bend
    assert float(near) == pytest.approx(v_bend, rel=0.1)        # inside 4 m: the original limit
    assert float(far) == pytest.approx(math.sqrt(v_bend ** 2 + 2 * scripted.A_BRAKE * 6.25), rel=0.1)
    assert float(far) < 6.37 - 0.1                     # the original 4 m window would not have braked


# --- warm start --------------------------------------------------------------------------------------

def test_dagger_warm_start_fits_pursue():
    from neural_racer_train.teacher import clone_loss, warm_start
    r = _runner(8, sensors=False)
    r.sim.safety_enabled = False
    actor, critic = TeacherNet(), Critic()
    priv = r.sim.privileged()[:, 0]
    target = scripted.pursue(priv, r.sim.v_cap, torch.zeros(8), torch.full((8,), 0.8))
    before = float(clone_loss(actor(priv), target, r.sim.v_cap))
    warm_start(r, actor, critic, 0, 8)
    priv = r.sim.privileged()[:, 0]
    target = scripted.pursue(priv, r.sim.v_cap, torch.zeros(8), torch.full((8,), 0.8))
    assert float(clone_loss(actor(priv), target, r.sim.v_cap)) < 0.5 * before
    assert float(actor.enc.norm.count) > 100          # the normaliser saw the visited states


# --- PLR -------------------------------------------------------------------------------------------

def test_plr_batch_sample_keeps_replay_ratio():
    b = plr.LevelBuffer(capacity=100, seed=0)
    for i in range(50):
        b.add(plr.LevelRecord(i, f'd{i}', i, score=float(i)))
    draws = b.sample(20000)
    assert sum(r for _, r in draws) / len(draws) == pytest.approx(0.7, abs=0.02)
    picked = [i for i, r in draws if r]
    assert sum(i >= 25 for i in picked) > sum(i < 25 for i in picked)     # high scores replayed more


def test_plr_top_level_no_longer_takes_most_draws():
    b = plr.LevelBuffer(capacity=30000, seed=0)
    for i in range(20000):
        b.add(plr.LevelRecord(i, 'd', i, score=float(i)))
    p = b.probabilities(list(b.levels))
    assert p.max() < 0.1                              # temperature 0.3 gave the top level 0.61


def test_plr_scores_go_to_the_level_that_was_played():
    r = _runner(2, warmup_levels=0)
    a_id, b_id = r.env_level
    r.ended = {a_id}                                  # a's episode ended this rollout; b's runs on
    adv = torch.tensor([[1.0, 5.0], [1.0, 5.0], [3.0, 5.0]])
    lvl = torch.tensor([[a_id, b_id], [a_id, b_id], [-1, b_id]])
    r.add_scores(adv, lvl)
    assert r.levels.levels[a_id].score == pytest.approx(1.0)
    assert r.levels.levels[b_id].score == 0.0 and r.score_acc[b_id] == [15.0, 3]


def test_level_cap_spreads_a_dominant_level():
    r = _runner(16, warmup_levels=0, level_cap=2)
    r.levels.replay_prob, r.levels.temperature, r.levels.staleness = 1.0, 0.01, 0.0
    top = r.env_level[0]
    r.levels.update_score(top, 1e6)
    r.reset(list(range(16)))
    assert sum(lv == top for lv in r.env_level) <= 2
    assert max(r.level_envs.values()) <= 2


# --- normalisers and the adaptive step ------------------------------------------------------------

def test_running_norm_matches_full_batch_statistics():
    x = torch.randn(1000, 3) * torch.tensor([1.0, 5.0, 0.2]) + torch.tensor([0.0, 3.0, -1.0])
    n = RunningNorm(3)
    n.update(x[:300])
    n.update(x[300:])
    assert torch.allclose(n.mean, x.mean(0), atol=1e-3)
    assert torch.allclose(n.var, x.var(0, unbiased=False), rtol=1e-3)


def test_micro_batches_give_the_whole_minibatch_gradient():
    torch.manual_seed(0)
    net = torch.nn.Linear(5, 1)
    x, y = torch.randn(10, 5), torch.randn(10)
    mb = torch.randperm(10)
    loss = ppo.value_loss(net(x[mb]).squeeze(-1), y[mb])
    loss.backward()
    whole = [p.grad.clone() for p in net.parameters()]
    net.zero_grad()
    total = ppo.accumulate(lambda p, _: ppo.value_loss(net(x[p]).squeeze(-1), y[p]), mb, micro=3)   # 3+3+3+1
    assert float(total) == pytest.approx(float(loss), rel=1e-6)
    for g, p in zip(whole, net.parameters()):
        assert torch.allclose(g, p.grad, atol=1e-6)


def test_value_norm_round_trips():
    v = ppo.ValueNorm()
    x = torch.randn(5000) * 7 + 3
    v.update(x)
    assert float(v.mean) == pytest.approx(float(x.mean()), abs=1e-2)
    assert float(v.std()) == pytest.approx(float(x.std()), rel=1e-2)
    assert torch.allclose(v.denorm(v.norm(x)), x, atol=1e-4)


def test_adaptive_lr_follows_the_kl():
    cfg = ppo.PPOConfig()
    assert ppo.adapt_lr(3e-4, 1.0, cfg) == pytest.approx(2e-4)
    assert ppo.adapt_lr(3e-4, 1e-4, cfg) == pytest.approx(4.5e-4)
    assert ppo.adapt_lr(3e-4, cfg.desired_kl, cfg) == 3e-4
    assert ppo.adapt_lr(cfg.lr_max, 1e-4, cfg) == cfg.lr_max


def test_entropy_coef_decays_to_zero():
    cfg = ppo.PPOConfig()
    assert ppo.entropy_coef(cfg, 0, 1e9) == pytest.approx(cfg.entropy_coef)
    assert ppo.entropy_coef(cfg, 0.125e9, 1e9) == pytest.approx(cfg.entropy_coef / 2)
    assert ppo.entropy_coef(cfg, 0.5e9, 1e9) == 0.0


def test_teacher_ignores_the_time_left():
    actor = TeacherNet()
    p = torch.randn(3, PRIV_DIM)
    q = p.clone()
    q[:, PRIV_BASE - 1] += 0.7
    assert torch.equal(actor(p), actor(q))


# --- the track library and the held-out race ------------------------------------------------------

def test_track_library_round_trips_and_fills_the_pool(tmp_path, clean_sys_path):
    tracklib.build(tmp_path, count=2, workers=1, seed=3)
    lib = tracklib.Library(tmp_path, learn.TRAIN_TRACKS)
    assert len(lib) == 2
    p = lib.files[0]
    loaded, fresh = tracklib.load(p), learn._generate(int(p.stem))
    assert loaded.digest == fresh.digest
    assert np.allclose(loaded.sdf, fresh.sdf.astype(np.float16).astype(np.float32))
    assert np.array_equal(loaded.occupancy, fresh.occupancy)
    r = Runner(4, 1, 'cpu', 0, pool_slots=4, solo=True, library=tmp_path, sensors=False)
    assert {int(f.stem) for f in lib.files} <= set(r.slot_seed.values())
    with pytest.raises(ValueError, match='built with'):
        tracklib.Library(tmp_path, tracks.TrackGenConfig())       # obstacles on: another config


def test_a_library_runs_on_its_own_config(tmp_path, clean_sys_path):
    """A v2 library gives its runs v2 (and v2 for any track generated after
    it runs out); a v1 library recorded before turn_room existed still loads."""
    import json
    tracklib.build(tmp_path / 'v2', count=2, workers=1, seed=3, config='v2')
    assert tracklib.Library(tmp_path / 'v2').cfg == learn.TRAIN_CONFIGS['v2']
    r = Runner(4, 1, 'cpu', 0, pool_slots=4, solo=True, library=tmp_path / 'v2', sensors=False)
    assert r.feeder.cfg == learn.TRAIN_CONFIGS['v2']
    tracklib.build(tmp_path / 'v1', count=1, workers=1, seed=4)
    meta_path = tmp_path / 'v1' / tracklib.META
    meta = json.loads(meta_path.read_text())
    del meta['config']['turn_room']
    meta_path.write_text(json.dumps(meta))
    assert tracklib.Library(tmp_path / 'v1').cfg == learn.TRAIN_TRACKS
    assert tracklib.Library(tmp_path / 'v1', learn.TRAIN_TRACKS).cfg == learn.TRAIN_TRACKS
    with pytest.raises(ValueError, match='built with'):
        tracklib.Library(tmp_path / 'v2', learn.TRAIN_TRACKS)


def test_a_library_can_hold_a_real_map(tmp_path, clean_sys_path):
    """tracklib --real: every variant of car 1's course, under seeds below the held-out base."""
    tracklib.build(tmp_path, count=2, workers=1, seed=3, config='v3')
    # a generated track that drew a seed in the real-map range keeps it; the real ones step over it
    clash = tracklib.load(tracklib.Library(tmp_path).files[0])
    tracklib.save(clash, tmp_path / f'{tracklib.REAL_SEED_BASE + 1}.npz')
    assert tracklib.add_real(tmp_path, ['car1_course_2026_10_26'], copies=2) == 24
    assert tracklib.add_real(tmp_path, ['car1_course_2026_10_26'], copies=2) == 24      # a rerun writes nothing new
    assert tracklib.add_real(tmp_path, ['car1_course_2026_10_26'], copies=2, refresh=True) == 24   # or rewrites them
    lib = tracklib.Library(tmp_path)
    assert lib.cfg == learn.TRAIN_CONFIGS['v3'] and len(lib) == 27
    assert not tracklib.load(tmp_path / f'{tracklib.REAL_SEED_BASE + 1}.npz').spec.kind.startswith('real:')
    real = [r for r in map(tracklib.load, lib.files) if r.spec.kind.startswith('real:')]
    assert len(real) == 24 and all(r.spec.kind.startswith('real:car1_course') for r in real)
    assert max(int(p.stem) for p in lib.files) < learn.HELDOUT_SEED_BASE
    r = Runner(4, 1, 'cpu', 0, pool_slots=27, solo=True, library=tmp_path, sensors=False)
    assert any(r.maker.specs[s].kind.startswith('real:') for s in r.slot_digest)


def test_heldout_race_reports_the_numbers():
    from neural_racer_train import teacher_eval
    ev = teacher_eval.Evaluator(starts=1, limit=2, max_seconds=4.0, raceline=False)
    res = ev.run(teacher_eval.pursue_policy(0.8), fresh=1)
    assert res['episodes'] == 4 and res['finish'] == 0.0       # 3 laps never fit in 4 s
    assert 0.0 <= res['crash'] <= 1.0 and res['laps'] > 0.0 and res['feasible'] == 0.0
    assert res['finish_lo'] == 0.0 and 0.0 < res['finish_hi'] < 1.0
    assert sum(n for n, _, _ in res['by_vcap'].values()) == 4 and res['a_lat'] == pytest.approx(2.0)


def test_heldout_race_counts_curvature_cuts():
    from neural_racer_train import teacher_eval
    ev = teacher_eval.Evaluator(starts=2, limit=2, max_seconds=2.0, raceline=False)
    hard = lambda n: torch.tensor([[2.0, 4.0]]).expand(n, 2)       # noqa: E731 -- near full lock, fast
    plain = ev.run(lambda priv, v: squash(hard(len(v)), v))
    capped = ev.run(lambda priv, v: capped_squash(hard(len(v)), v, priv))
    assert plain['curv_cut'] > 0.05 and capped['curv_cut'] == 0.0


def test_heldout_race_can_hide_the_unseen_track():
    from neural_racer_train import teacher_eval
    ev = teacher_eval.Evaluator(starts=1, limit=2, max_seconds=1.0, raceline=False)
    seen = []
    res = ev.run(lambda priv, v: (seen.append(priv[:, PRIV_VIS:].min()), squash(torch.zeros(len(v), 2), v))[1],
                 mask=True, episode_type=0)
    assert res['episodes'] == 2 and min(float(m) for m in seen) == 0.0      # some point was hidden


def test_wilson_interval():
    from neural_racer_train.teacher_eval import wilson
    lo, hi = wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert wilson(0, 0) == (0.0, 1.0)


def test_averaged_actor_and_kl_anneal():
    from neural_racer_train.teacher import ema_update, kl_config
    a, b = TeacherNet(), TeacherNet()
    with torch.no_grad():
        b.enc.norm.mean.fill_(3.0)
    want = [(pa + pb) / 2 for pa, pb in zip(a.parameters(), b.parameters())]
    ema_update(a, b, 0.5)
    assert all(torch.allclose(p, w) for p, w in zip(a.parameters(), want))
    assert torch.equal(a.enc.norm.mean, b.enc.norm.mean)          # statistics copied, not averaged
    cfg = ppo.PPOConfig()
    assert kl_config(cfg, 0.5e9, 1e9, 0.2, 0.2).desired_kl == cfg.desired_kl
    assert kl_config(cfg, 0.9e9, 1e9, 0.2, 0.2).desired_kl == pytest.approx(0.6 * cfg.desired_kl)
    assert kl_config(cfg, 1e9, 1e9, 0.2, 0.2).desired_kl == pytest.approx(0.2 * cfg.desired_kl)


def test_run_record_names_the_code(tmp_path):
    import argparse
    import json
    from neural_racer_train import common
    rec = common.write_run_record(argparse.Namespace(run_dir=tmp_path, seed=3))
    line = json.loads((tmp_path / 'run.jsonl').read_text().splitlines()[-1])
    assert line['args']['seed'] == 3 and line['changes_sha256'] == rec['changes_sha256']
    assert len(list(tmp_path.glob('code-*.tar.gz'))) == 1
