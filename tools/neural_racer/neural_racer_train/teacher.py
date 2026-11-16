"""Stage 1: the privileged solo teacher (plan A8, Phase 5).

    python -m neural_racer_train.teacher --run-dir DIR --resume [--steps 2e9] [--library DIR]

One car, no opponents, no obstacles (learn.TRAIN_TRACKS). The actor sees
privileged state, so it learns raw car control at the limit fast; the sim
runs without LiDAR or IMU, which nothing here reads.

A fresh run first clones the scripted pursuit driver by DAgger (the actor
takes over the wheel while the driver keeps labelling), then trains only the
critic for --critic-warmup updates, then PPO over the PLR curriculum. Fresh
levels are scored but not trained on. Separate optimisers and gradient clips
for actor and critic, normalised returns, and a KL-adaptive actor step.

Every --eval-every updates (once the last race is in) a CPU worker races a
snapshot on the held-out tracks (teacher_eval), held to the car's 2.0 m/s^2
cornering cap; the best by the lower Wilson bound of its 3-lap finish rate is
checkpoints/best.pt. With --safety on (teacher v5: the car's whole wrapper,
so its LiDAR braking cap and side margin, are in the teacher's world) every
race -- held-out, final and rerace -- has the wrapper on too, and since they
cast the LiDAR they run on the GPU between updates instead. When training ends, a race still on the CPU is dropped,
the last snapshot races the whole plan on the GPU, and the top --rerace-top
snapshots race again on 1000 fresh races there to settle best.pt. best.pt is
what distillation should start from.

From teacher v4 (--cap-mode) the speed is held to the wrapper's curvature cap
at each race's cornering cap, drawn per level (learn.A_LAT_RANGE) and given to
the networks as an input: `squash` builds the cap into the speed output
(model.capped_squash), `penalty` lets the sim cut the command and charges the
cut, `off` is teacher-v3's regime. --init-from fine-tunes from an earlier
run's best actor and last critic. The actor's weights are also averaged
(--ema); the final eval races both copies. --mask-share: that share of episodes hide,
from the actor only, the track points its LiDAR cannot see while the car is
in reactive mode (model.mask_track, plan item 2.3), so it can label a
student's first lap with commands the student could work out.
"""
import concurrent.futures as cf
import math
import multiprocessing as mp
import sys
import time
from collections import deque
from pathlib import Path

import torch

from neural_racer import observation as obs
from neural_racer_train import common, scripted, teacher_eval
from neural_racer_train.gpu_sim.race import RewardConfig
from neural_racer_train.learn import Runner
from neural_racer_train.gpu_sim.sim import EPISODE_REACTIVE
from neural_racer_train.model import (Critic, TeacherNet, TRACK_POINTS, gaussian_logp, load_expanded, mask_track,
                                      squash, teacher_squash)
from neural_racer_train.ppo import (PPOConfig, Update, ValueNorm, entropy_coef, gae, gaussian_entropy, gaussian_kl,
                                    kl_config, normalize, policy_loss, value_loss)

EXIT_NOT_LEARNING = 86
WARM_PACE = 0.8                 # x the bend speed: the cloned driver is quick but not on the limit
WARM_UPDATES = 800              # at least this many cloning gradient steps, however many envs
WARM_ITERS = 12                 # at least this many DAgger iterations
WARM_EXPERT_ITERS = 6           # the expert's share of the wheel falls from 1 to 0 over these
WARM_KEEP = 131072              # samples kept per iteration; the last 8 iterations are trained on
WARM_MINIBATCH = 4096
MICRO = 131072                  # samples per forward/backward: a PPO minibatch is a quarter of the
                                # replayed samples (up to 524k at 32768 envs), and at that size the
                                # H100's allocator ran out and retried (2026-10-27)
CAR_A_LAT = 2.0                 # m/s^2, the car's cornering cap today: what best.pt is chosen at
RERACE_SEED = teacher_eval.FRESH_SEED + 10_000 * 1_000_000      # a draw no training eval reaches


def parser():
    ap = common.base_parser(__doc__.split('\n')[0])
    ap.add_argument('--steps', type=float, default=2e9)
    ap.add_argument('--rollout', type=int, default=32)
    ap.add_argument('--pool-slots', type=int, default=4096)
    ap.add_argument('--library', default=None, help='a tracklib directory to fill the track pool from')
    ap.add_argument('--warm-start-steps', type=float, default=1e7, help='0 = start PPO from scratch')
    ap.add_argument('--critic-warmup', type=int, default=20, help='updates with the actor frozen')
    ap.add_argument('--safety', choices=('on', 'off'), default='off',
                    help='off: actuator limits only (and no LiDAR in the sim); walls and stalls still end the episode')
    ap.add_argument('--gamma', type=float, default=0.998, help='horizon ~ 1/(1-gamma) ticks: 0.998 is 12.5 s')
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--lr', type=float, default=3e-4, help="the critic's step")
    ap.add_argument('--actor-lr', type=float, default=2e-5,
                    help="the actor's first step; the KL rule moves it. A fresh Adam's first steps are about lr on "
                         "every weight at once: 3e-4 gave a KL of 18 and quadrupled the clone's crash rate")
    ap.add_argument('--wall-min', type=float, default=25.0, help='wall cost at walking pace')
    ap.add_argument('--wall', type=float, default=40.0, help='wall cost at 4 m/s')
    ap.add_argument('--max-seconds', type=float, default=180.0, help='training episode limit (bootstrapped)')
    ap.add_argument('--eval-every', type=int, default=50, help='updates between held-out races; 0 = off')
    ap.add_argument('--eval-starts', type=int, default=8, help='fixed starts per held-out track')
    ap.add_argument('--eval-fresh', type=int, default=8, help='fresh starts per held-out track, for the averaged actor')
    ap.add_argument('--eval-threads', type=int, default=7)
    ap.add_argument('--eval-real', action='store_true',
                    help='held-out races and reraces also on the obstacle-free real maps (car 1\'s course among them)')
    ap.add_argument('--rerace-top', type=int, default=5, help='snapshots raced again at the end; 0 = off')
    ap.add_argument('--rerace-starts', type=int, default=50, help='per held-out track: 50 x 20 = 1000 races')
    ap.add_argument('--cap-mode', choices=('squash', 'penalty', 'off'), default='squash',
                    help="the wrapper's curvature cap: built into the speed output, cut and charged, or absent")
    ap.add_argument('--cap-penalty', type=float, default=0.5, help='penalty mode: cost of a cut of v_ref m/s')
    ap.add_argument('--a-lat-range', default='2.0,5.0', help='cornering caps drawn per level, m/s^2')
    ap.add_argument('--init-from', default=None, help="a teacher run directory: its best actor and last critic")
    ap.add_argument('--init-actor', choices=('best', 'latest'), default='best',
                    help="with --init-from: the run's best.pt actor, or its latest.pt (its last update)")
    ap.add_argument('--init-std', type=float, default=0.0, help='with --init-from: reset the policy std; 0 = keep')
    ap.add_argument('--ema', type=float, default=0.99, help='per-update decay of the averaged actor; 0 = off')
    ap.add_argument('--anneal-frac', type=float, default=0.2, help='final share of the run with a shrinking KL target')
    ap.add_argument('--anneal-kl', type=float, default=0.2, help="the KL target's final multiple")
    ap.add_argument('--mask-share', type=float, default=0.0,
                    help="share of episodes whose reactive-mode ticks hide the track the LiDAR can't see")
    ap.add_argument('--guard-after', type=float, default=5e7,
                    help='steps before the not-learning guard can stop the run; 0 = off')
    return ap


def clone_loss(a, target, v_cap, s=None):
    """MSE on the squashed commands (s, default the plain squash of a),
    steering over its range and speed over the cap: what the car would do (a
    pre-squash MSE is dominated by the saturated targets of full lock and the cap)."""
    s = squash(a, v_cap) if s is None else s
    return (((s[:, 0] - target[:, 0]) / obs.STEER_LIMIT) ** 2 + ((s[:, 1] - target[:, 1]) / v_cap) ** 2).mean()


def warm_start(runner, actor, critic, steps: float, T: int, sq=None) -> int:
    """DAgger on scripted.pursue (Ross et al. 2011): the driver labels every
    state; who steers moves from the driver to the actor over the first
    WARM_EXPERT_ITERS iterations, so the clone also learns to recover from
    its own mistakes. sq(a, v_cap, priv): the actor's squash. Returns the steps used."""
    sq = sq or (lambda a, v_cap, priv: squash(a, v_cap))
    sim, E, dev = runner.sim, runner.E, runner.device
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    offset, pace = torch.zeros(E, device=dev), torch.full((E,), WARM_PACE, device=dev)
    data = deque(maxlen=8)
    done_steps, it, updates, t0, loss = 0, 0, 0, time.time(), torch.zeros(())
    while done_steps < steps or updates < WARM_UPDATES or it < WARM_ITERS:
        runner.refresh_tracks()
        beta = max(0.0, 1.0 - it / WARM_EXPERT_ITERS)
        expert = torch.rand(E, device=dev) < beta
        X, Y, V = [], [], []
        with torch.no_grad():
            for _ in range(T):
                priv = sim.privileged()[:, 0]
                target = scripted.pursue(priv, sim.v_cap, offset, pace)
                own = sq(actor(priv), sim.v_cap, priv)
                X.append(priv), Y.append(target), V.append(sim.v_cap.clone())
                out = sim.step(torch.where(expert.unsqueeze(1), target, own).unsqueeze(1))
                runner.end_episodes(out['env_done'])
        X, Y, V = torch.cat(X), torch.cat(Y), torch.cat(V)
        actor.update_norm(X)
        critic.update_norm(X)
        keep = torch.randperm(X.shape[0], device=dev)[:WARM_KEEP]
        data.append((X[keep], Y[keep], V[keep]))
        X, Y, V = (torch.cat([d[i] for d in data]) for i in range(3))
        for _ in range(2):
            for mb in torch.randperm(X.shape[0], device=dev).chunk(max(4, X.shape[0] // WARM_MINIBATCH)):
                loss = clone_loss(None, Y[mb], V[mb], sq(actor(X[mb]), V[mb], X[mb]))
                opt.zero_grad()
                loss.backward()
                opt.step()
                updates += 1
        done_steps += T * E
        it += 1
        print(f'[teacher] DAgger iteration {it}: expert share {beta:.2f}, {done_steps:.3g} steps, '
              f'{updates} updates, clone loss {float(loss):.4f}, {time.time() - t0:.0f}s', flush=True)
    return done_steps


def _score(res: dict) -> float:
    """Held-out ranking: the lower Wilson bound of the 3-lap finish rate (a
    lucky small sample does not win), laps covered breaks ties."""
    return res['finish_lo'] + 0.001 * res['laps'] / teacher_eval.LAPS


def ema_update(ema, actor, decay: float) -> None:
    """Polyak average of the actor's weights; the normaliser's statistics are copied."""
    with torch.no_grad():
        torch._foreach_lerp_(list(ema.parameters()), list(actor.parameters()), 1.0 - decay)
        for b, a in zip(ema.buffers(), actor.buffers()):
            b.copy_(a)


def stop_pool(pool) -> None:
    """Drop a held-out race still running on the CPU: Python's exit, like
    shutdown(wait=True), waits for a running work item, and one race of the
    whole plan takes the worker over 20 minutes -- longer than the time
    limit's warning, and GPU time spent idle at the end of a run."""
    for p in list(pool._processes.values()):
        p.terminate()
    pool.shutdown(wait=False, cancel_futures=True)


def init_from(run_dir, actor, critic, vnorm, priv0, actor_from: str = 'best') -> None:
    """Fine-tune start: the actor from run_dir's best.pt (or latest.pt), the critic and return
    statistics from its latest.pt, widened to inputs they predate; the new
    columns' normaliser statistics come from priv0, a sample of states."""
    ck = Path(run_dir) / 'checkpoints'
    best = torch.load(ck / f'{actor_from}.pt', map_location=priv0.device, weights_only=False)
    last = torch.load(ck / 'latest.pt', map_location=priv0.device, weights_only=False)
    for net, state in ((actor, best['actor']), (critic, last['critic'])):
        old = state['enc.norm.mean'].shape[0]
        widened = load_expanded(net, state)
        new = net.enc.norm.mean.shape[0]
        if new > old:
            net.enc.norm.set_columns(list(range(old, new)), priv0[:, 4 * TRACK_POINTS:4 * TRACK_POINTS + new])
        print(f'[teacher] {type(net).__name__} from {run_dir} (update {best.get("update")}); '
              f'widened {widened or "nothing"}', flush=True)
    vnorm.load(last['vnorm'])


def train(args, stop) -> bool:
    common.seed_everything(args.seed)
    dev, t_setup = args.device, time.time()
    cfg = PPOConfig(gamma=args.gamma, lam=args.lam, lr=args.lr)
    reward = RewardConfig(wall=args.wall, wall_min=args.wall_min, stall=args.wall,
                          curvature=args.cap_penalty if args.cap_mode == 'penalty' else 0.0)
    a_lat_range = tuple(float(v) for v in args.a_lat_range.split(','))
    runner = Runner(args.num_envs, 1, dev, args.seed, pool_slots=args.pool_slots, cpu_workers=args.cpu_workers,
                    solo=True, library=args.library, sensors=args.safety == 'on', reward_cfg=reward,
                    max_seconds=args.max_seconds, a_lat_range=a_lat_range)
    runner.sim.safety_enabled = args.safety == 'on'
    runner.sim.curvature_cap = args.cap_mode != 'off'
    mode = 'capped' if args.cap_mode == 'squash' else 'plain'
    sq = teacher_squash(mode)
    print(f'[teacher] safety wrapper {args.safety}, curvature cap {args.cap_mode} (cornering caps {a_lat_range} '
          f'm/s^2), {args.num_envs} envs on {len(runner.slot_digest)} tracks; '
          f'setup took {time.time() - t_setup:.0f}s', flush=True)
    actor, critic = TeacherNet().to(dev), Critic().to(dev)
    upd = Update(cfg, actor.parameters(), critic.parameters(), args.actor_lr, cfg.lr, dev, MICRO)
    vnorm = ValueNorm(dev)
    log = common.MetricLog(args.run_dir)
    update, steps, frozen, lr, best, cands = 0, 0, 0, upd.lr, None, []
    ema = TeacherNet().to(dev)
    ck = common.load_checkpoint(args.run_dir) if args.resume else None
    if ck:
        actor.load_state_dict(ck['actor']), critic.load_state_dict(ck['critic'])
        upd.load(ck)
        vnorm.load(ck['vnorm'])
        runner.load(ck['plr'], ck.get('slot_seed'))
        update, steps, frozen, lr, best = ck['update'], ck['steps'], ck['frozen'], upd.lr, ck.get('best')
        cands = ck.get('cands', [])
    elif args.init_from:
        init_from(args.init_from, actor, critic, vnorm, runner.sim.privileged()[:, 0], args.init_actor)
        if args.init_std > 0:
            with torch.no_grad():
                actor.log_std.fill_(math.log(args.init_std))
    elif args.warm_start_steps > 0:
        warm_start(runner, actor, critic, args.warm_start_steps, args.rollout, sq)
    else:
        frozen = args.critic_warmup         # nothing to protect: no frozen phase
    ema.load_state_dict(ck['ema'] if ck and 'ema' in ck else actor.state_dict())
    sim, E, T = runner.sim, args.num_envs, args.rollout
    masked_env = torch.rand(E, device=dev) < args.mask_share
    recent = deque(maxlen=20)
    eval_dir = Path(args.run_dir) / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)
    wrapper = args.safety == 'on'
    real = teacher_eval.OBSTACLE_FREE_REAL if args.eval_real else ()     # its held-out races
    pool = cf.ProcessPoolExecutor(1, mp_context=mp.get_context('spawn')) \
        if args.eval_every > 0 and not wrapper else None
    pending = None

    cpu = lambda net: {k: v.detach().cpu() for k, v in net.state_dict().items()}     # noqa: E731
    cap = args.cap_mode != 'off'
    top_a_lat = a_lat_range[-1]
    main_key = 'ema' if args.ema > 0 else 'actor'
    plan = [{'key': main_key, 'a_lat': CAR_A_LAT, 'starts': args.eval_starts, 'fresh': args.eval_fresh},
            *([{'key': 'actor', 'a_lat': CAR_A_LAT, 'starts': args.eval_starts}] if args.ema > 0 else []),
            *([{'key': main_key, 'a_lat': top_a_lat, 'starts': max(args.eval_starts // 2, 1)}]
              if top_a_lat > CAR_A_LAT else []),
            *([{'key': main_key, 'a_lat': CAR_A_LAT, 'starts': max(args.eval_starts // 2, 1), 'mask': True,
                'episode_type': EPISODE_REACTIVE}] if args.mask_share > 0 else [])]
    if not cap:
        plan = [dict(p, a_lat=None, cap=False) for p in plan[:2]]     # teacher-v3's regime: no cap
    plan = [dict(p, wrapper=wrapper) for p in plan]
    # While training runs, the CPU worker races the main entry (and the masked one) only; the whole
    # plan is raced once, on the GPU, when training is over.
    live_plan = [p for p in plan if p is plan[0] or p.get('mask')]

    def save():
        common.save_checkpoint(args.run_dir, {
            'actor': actor.state_dict(), 'critic': critic.state_dict(), **upd.state(),
            'vnorm': vnorm.state(), **runner.state(), 'frozen': frozen, 'best': best, 'cands': cands,
            'ema': ema.state_dict(), 'squash': mode, 'update': update, 'steps': steps, 'run_name': args.run_dir.name})

    def write_best(path, variant, res):
        snap = torch.load(path, map_location='cpu', weights_only=False)
        # save_checkpoint makes checkpoints/: a short --eval-every can bring a result in before the first save()
        common.save_checkpoint(args.run_dir, {
            'actor': snap[variant], 'squash': mode, 'variant': variant, 'steps': snap['steps'],
            'update': snap['update'], 'eval': {k: v for k, v in res.items() if k != 'path'}}, name='best')

    def snapshot():
        path = eval_dir / f'u{update:06d}.pt'
        torch.save({'actor': cpu(actor), **({'ema': cpu(ema)} if args.ema > 0 else {}), 'squash': mode,
                    'steps': steps, 'update': update}, path)
        return path

    def submit_eval():
        return pool.submit(teacher_eval.evaluate_file, snapshot(), live_plan, args.eval_threads, 'cpu', real)

    def collect_eval(fut, gpu_plan=plan):
        try:
            results = fut.result() if fut is not None else teacher_eval.evaluate_file(snapshot(), gpu_plan, device=dev,
                                                                                      real=real)
        except Exception as e:      # noqa: BLE001 -- a failed race is logged, never fatal to training
            print(f'[teacher] held-out race failed: {e}', flush=True)
            return
        collect(results)

    def collect(results):
        nonlocal best
        for res in results:
            log.write(kind='eval', **{k: v for k, v in res.items() if k != 'path'})
            print(teacher_eval.line(f'{Path(res["path"]).stem} {res["variant"]}', res), flush=True)
            if (cap and res['plan_a_lat'] != CAR_A_LAT) or res.get('mask'):
                continue
            cands.append([_score(res), res['path'], res['variant']])
            if best is None or _score(res) > best['score']:
                best = {'score': _score(res), **res}
                write_best(res['path'], res['variant'], res)

    def rerace() -> bool:
        """The top snapshots on 1000 fresh races, on the GPU now that training is done.
        False if the time-limit signal came first: the next segment races them again."""
        nonlocal best
        top = sorted(cands, key=lambda c: -c[0])[:args.rerace_top]
        if len(top) < 2:
            return True
        ev = teacher_eval.Evaluator(starts=args.rerace_starts, device=dev, real=real)
        scored = []
        for score, path, variant in top:
            if stop.requested:
                return False
            snap = torch.load(path, map_location=dev, weights_only=False)
            rec = ev.races(teacher_eval.teacher_policy(teacher_eval.actor_from(snap[variant], dev), mode),
                           starts=args.rerace_starts, level_seed=RERACE_SEED, sim_seed=RERACE_SEED,
                           a_lat=CAR_A_LAT if cap else None, cap=cap, wrapper=wrapper)
            res = dict(teacher_eval.summarize([rec], ev.N, ev.names), path=path, variant=variant, update=snap['update'],
                       steps=snap['steps'], eval_score=score, wrapper=wrapper)
            log.write(kind='rerace', **{k: v for k, v in res.items() if k != 'path'})
            print(teacher_eval.line(f'rerace {Path(path).stem} {variant}', res), flush=True)
            scored.append((_score(res), path, variant, res))
        s, path, variant, res = max(scored, key=lambda x: x[0])
        best = {'score': s, **res}
        write_best(path, variant, res)
        print(f'[teacher] best.pt: {Path(path).stem} {variant}, finish {res["finish"]:.1%} '
              f'[{res["finish_lo"]:.1%}, {res["finish_hi"]:.1%}] over {res["episodes"]} races', flush=True)
        return True

    while steps < args.steps:
        if stop.requested:
            save()
            if pool:
                stop_pool(pool)     # else the job can end in TIMEOUT, which cancels its chained segment
            return False
        runner.refresh_tracks()
        keys = ('priv', 'pin', 'a', 'mean', 'logp', 'v', 'v_next', 'r', 'term', 'done', 'train', 'lvl')
        buf = {k: [] for k in keys}
        diag = {k: [] for k in ('stalled', 'speed', 'progress', 'finished', 'alive', 'curv_cut', 'safety_stop')}
        t0 = time.time()
        for _ in range(T):
            priv = sim.privileged()[:, 0]
            pin = priv       # what the actor sees; the critic always sees everything
            if args.mask_share > 0:
                pin = mask_track(priv, sim.track_visibility()[:, 0], masked_env & (sim.mode()[:, 0] == 0))
            with torch.no_grad():
                mean = actor(pin)
                a = mean + torch.exp(actor.log_std) * torch.randn_like(mean)    # on-device RNG
                logp, v = gaussian_logp(a, mean, actor.log_std), vnorm.denorm(critic(priv))
            lvl = runner.level_t.clone()
            out = sim.step(sq(a, sim.v_cap, priv).unsqueeze(1))
            with torch.no_grad():       # the state each step reached, before any reset
                v_next = vnorm.denorm(critic(sim.privileged()[:, 0]))
            for k, val in (('priv', priv), ('pin', pin), ('a', a), ('mean', mean), ('logp', logp), ('v', v),
                           ('v_next', v_next),
                           ('r', out['reward'][:, 0]), ('term', out['terminal'][:, 0]), ('done', out['env_done']),
                           ('train', runner.env_replay.clone()), ('lvl', lvl)):
                buf[k].append(val)
            for k in diag:
                diag[k].append(out[k][:, 0].float())
            ended = runner.end_episodes(out['env_done'])
            if ended and args.mask_share > 0:
                d = torch.as_tensor(ended, device=dev)
                masked_env[d] = torch.rand(len(ended), device=dev) < args.mask_share
        t_roll = time.time() - t0
        b = {k: torch.stack(v) for k, v in buf.items()}
        adv, ret = gae(b['r'], b['v'], b['v_next'], b['term'], b['done'], cfg.gamma, cfg.lam)
        runner.add_scores(adv, b['lvl'])
        mask = b['train'].reshape(-1)
        flat = {k: v.reshape(T * E, *v.shape[2:])[mask] for k, v in b.items()
                if k in ('priv', 'pin', 'a', 'mean', 'logp')}
        flat_ret = ret.reshape(-1)[mask]
        n = flat_ret.shape[0]
        train_actor = frozen >= args.critic_warmup
        ent_c = entropy_coef(cfg, steps, args.steps)
        kl_cfg = kl_config(cfg, steps, args.steps, args.anneal_frac, args.anneal_kl)
        log_std_old = actor.log_std.detach().clone()
        if n > 1:
            flat_adv = normalize(adv.reshape(-1)[mask])
            vnorm.update(flat_ret)
            target = vnorm.norm(flat_ret)

        def actor_loss(p, _):
            mean = actor(flat['pin'][p])
            return (policy_loss(gaussian_logp(flat['a'][p], mean, actor.log_std), flat['logp'][p], flat_adv[p],
                                cfg.clip),
                    gaussian_kl(flat['mean'][p], mean.detach(), log_std_old, actor.log_std.detach()))
        res = upd.run(n, lambda p, _: value_loss(critic(flat['priv'][p]), target[p]), actor_loss, train_actor,
                      kl_cfg, actor_extra=lambda: -ent_c * gaussian_entropy(actor.log_std))
        kls, v_loss, lr = res['kls'], res['value_loss'], upd.lr
        all_priv = b['priv'].reshape(T * E, -1)
        actor.update_norm(b['pin'].reshape(T * E, -1))
        critic.update_norm(all_priv)
        if args.ema > 0:
            ema_update(ema, actor, args.ema)
        frozen = min(frozen + 1, args.critic_warmup)
        update += 1
        steps += T * E
        d = {k: torch.stack(v) for k, v in diag.items()}
        alive = d['alive'].sum()
        crashes = (b['term'] & ~d['finished'].bool()).float().sum()
        speed = float((d['speed'] * d['alive']).sum() / alive.clamp(min=1))
        progress = float((d['progress'] * d['alive']).sum() / alive.clamp(min=1))
        recent.append((speed, progress))
        if pending is not None and pending.done():
            collect_eval(pending)
            pending = None
        if pool and update % args.eval_every == 0 and pending is None:
            pending = submit_eval()
        if wrapper and args.eval_every > 0 and update % args.eval_every == 0 and steps < args.steps:
            collect_eval(None, live_plan)
        if update % args.log_every == 0:
            # how much of the return variance the rollout's value estimates explained
            ev = 1 - torch.var(flat_ret - b['v'].reshape(-1)[mask]) / torch.var(flat_ret).clamp(min=1e-8) \
                if n > 1 else torch.zeros(())
            log.write(update=update, steps=steps, reward=float(b['r'].mean()), hazard=float(crashes / alive.clamp(min=1)),
                      crashes=float(crashes), finishes=float(d['finished'].sum()), stalled=float(d['stalled'].mean()),
                      speed=speed, progress=progress, actor_trained=train_actor, lr=lr,
                      kl=sum(kls) / len(kls) if kls else 0.0, value_loss=v_loss, explained_var=float(ev),
                      entropy_coef=ent_c, std=float(torch.exp(actor.log_std.detach()).mean()),
                      curv_cut=float((d['curv_cut'] > teacher_eval.CUT_EPS).float().mul(d['alive']).sum()
                                     / alive.clamp(min=1)), desired_kl=kl_cfg.desired_kl,
                      safety_stop=float((d['safety_stop'] * d['alive']).sum() / alive.clamp(min=1)),
                      levels=len(runner.levels.levels), tracks=len(runner.slot_digest),
                      steps_per_s=T * E / t_roll, steps_per_s_total=T * E / (time.time() - t0))
        if update % args.checkpoint_every == 0:
            save()
        if args.guard_after and steps >= args.guard_after and len(recent) == recent.maxlen:
            speed = sum(s for s, _ in recent) / len(recent)
            prog = sum(p for _, p in recent) / len(recent)
            if speed < 0.2 or prog < 0.002:
                save()
                if pool:
                    stop_pool(pool)
                print(f'[teacher] not learning after {steps:.3g} steps: mean speed {speed:.2f} m/s, '
                      f'progress {prog * 1000:.1f} mm/tick over the last {len(recent)} updates. Stopping the chain.',
                      flush=True)
                sys.exit(EXIT_NOT_LEARNING)
    if pool:
        if pending is not None and pending.done():
            collect_eval(pending)
        stop_pool(pool)
    # The final races can outlast a segment (2026-11-01: teacher v6 was killed in its reraces at the
    # time limit). They start only before the signal and stop between snapshots; a segment that
    # resumes past the step budget runs them again
    if stop.requested:
        save()
        return False
    if args.eval_every > 0 and update >= args.eval_every:
        collect_eval(None)          # the last snapshot, the whole plan, on the GPU
    save()
    if args.rerace_top > 0:
        if not rerace():
            save()
            return False
        save()
    return True


def main(argv=None) -> int:
    return common.run_stage(train, parser().parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
