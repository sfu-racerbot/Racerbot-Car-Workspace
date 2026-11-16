"""Stage 3: one league role (plan A8, Phase 8).

    python -m neural_racer_train.league_learner --run-dir DIR --archive DIR --role main0 \
        --init-from distill/checkpoints/best.pt [--teacher T.pt] [--critic-from TEACHER_RUN] --resume

Roles: main0, main1_mlp (the MLP ablation), main_exploiter, league_exploiter.
Recurrent PPO on seat 0 with a privileged critic. Opponents come from
matchmaking (scripted pool, archive by PFSP with a uniform floor, or the
current self) through an OpponentTable: each race keeps its drivers from
start to finish, and their policies run as one batched pass. Snapshots go
to the shared archive every --snapshot-every updates and are never deleted.

The update is the teacher's (ppo.Update, plan item 1.1): separate optimisers
and clips, normalised returns, the KL-adaptive actor step, and the actor
frozen for the first --critic-warmup updates while a critic that starts
behind a trained student catches up (the teacher's first CPU check measured
a KL of 18 on a first update without it). The student's encoders run over a
whole rollout at once (PolicyNet.unroll, item 1.4). With --teacher, a
kickstarting term (Schmitt et al. 2018, item 1.7) pulls the student's command
toward the teacher's on the same state, its weight fading to zero over the
first --kickstart-frac of the role's steps; a teacher trained alone sees the
state as it did in training (model.solo_view).

The main roles race the held-out tracks every --eval-every updates, as
distillation does (student_eval: solo, the wrapper on, every episode type);
best.pt is the snapshot with the best lower bound of its finish rate.
"""
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

from neural_racer_train import common, matchmaking, student_eval, teacher_eval
from neural_racer_train.archive import Archive
from neural_racer_train.distill import action_error, aux_errors
from neural_racer_train.gpu_sim.race import RewardConfig
from neural_racer_train.gpu_sim.sim import PRIV_DIM
from neural_racer_train.learn import Runner, SeatAssignment, aux_targets, opponent_actions
from neural_racer_train.model import (FULL, GRU, Critic, PolicyNet, fresh_opponent_inputs, gaussian_logp,
                                      load_expanded, solo_view, squash, teacher_squash)
from neural_racer_train.opponents import StackedOpponents
from neural_racer_train.ppo import PPOConfig, Update, ValueNorm, gae

OBS_KEYS = ('scan', 'ego', 'preview', 'race', 'camera')
LOG_STD0 = -1.5         # the exploration noise every role starts (and an exploiter restarts) from
CACHE = 64              # archived snapshots kept in memory


def parser():
    ap = common.base_parser(__doc__.split('\n')[0])
    ap.add_argument('--init-from', type=str, required=True, help='a student checkpoint (distill best.pt)')
    ap.add_argument('--steps', type=float, default=5e9)
    ap.add_argument('--rollout', type=int, default=32)
    ap.add_argument('--opponents', type=int, default=16, help='drivers in each half of the opponent table')
    ap.add_argument('--snapshot-every', type=int, default=200)
    ap.add_argument('--lr', type=float, default=3e-4, help="the critic's step")
    ap.add_argument('--actor-lr', type=float, default=2e-5, help="the actor's first step; the KL rule moves it")
    ap.add_argument('--critic-warmup', type=int, default=20, help='updates with the actor frozen')
    ap.add_argument('--critic-from', default=None, help="a teacher run directory: start from its critic")
    ap.add_argument('--teacher', default=None, help='a teacher checkpoint for the kickstarting term')
    ap.add_argument('--kickstart', type=float, default=1.0, help="the kickstarting term's first weight")
    ap.add_argument('--kickstart-frac', type=float, default=0.1, help='share of the steps it fades over')
    ap.add_argument('--cap-penalty', type=float, default=0.5,
                    help="cost of a curvature-cap cut of v_ref m/s: the student's squash does not hold the cap, "
                         "and gate S counts the cuts as interventions")
    ap.add_argument('--micro', type=int, default=1024, help='environments per forward/backward in the update')
    ap.add_argument('--max-cars', type=int, default=4)
    ap.add_argument('--pool-slots', type=int, default=4096)
    ap.add_argument('--library', default=None, help='a tracklib directory to fill the track pool from')
    ap.add_argument('--eval-every', type=int, default=200, help='updates between held-out races (main roles); 0 = off')
    ap.add_argument('--eval-starts', type=int, default=4, help='per held-out track and episode type')
    ap.add_argument('--eval-tracks', type=int, default=None, help='the first N held-out tracks (smoke runs)')
    ap.add_argument('--eval-seconds', type=float, default=teacher_eval.MAX_SECONDS, help='race limit (smoke runs)')
    return ap


class OpponentTable:
    """The opponents of every race in two halves of n drivers. A new race
    draws from the live half; a refresh fills the other half from matchmaking
    once no car still racing uses it, and makes it live. So a car never
    changes driver mid-race -- it did every rollout before (a pure-pursuit
    car could turn into a parked one mid-lap, and a win counted for whoever
    had driven its last 0.8 s)."""

    def __init__(self, refs: list):
        self.n = len(refs)
        self.refs = list(refs) + list(refs)
        self.live = 0

    @property
    def offset(self) -> int:
        return self.live * self.n

    def in_use(self, seats: SeatAssignment, alive, half: int) -> bool:
        src = seats.source[:, 1:]
        lo = half * self.n
        return bool((((src >= lo) & (src < lo + self.n)) & alive[:, 1:]).any())

    def refresh(self, refs: list, seats: SeatAssignment, alive) -> bool:
        other = 1 - self.live
        if self.in_use(seats, alive, other):
            return False
        self.refs[other * self.n:(other + 1) * self.n] = refs
        self.live = other
        return True


def build_sources(refs, archive, policy, cache: OrderedDict):
    """Opponent refs -> (sources, StackedOpponents or None, names per source).
    Archived snapshots are read once and kept (they never change)."""
    sources, state_dicts, recurrent, names = [], [], [], []
    for ref in refs:
        if ref.kind == 'scripted':
            sources.append(('scripted', ref.name))
            names.append(f'scripted:{ref.name}')
            continue
        if ref.kind == 'self':
            sd, rec = policy.state_dict(), policy.recurrent
        else:
            if ref.name not in cache:
                snap = archive.load(ref.name)
                cache[ref.name] = (snap['policy'], snap.get('meta', {}).get('recurrent', True))
                while len(cache) > CACHE:
                    cache.popitem(last=False)
            cache.move_to_end(ref.name)
            sd, rec = cache[ref.name]
        sources.append(('policy', len(state_dicts)))
        state_dicts.append(sd)
        recurrent.append(rec)
        names.append(ref.name)
    stacked = StackedOpponents(state_dicts, device=next(policy.parameters()).device, width=policy.width,
                               recurrent=recurrent) if state_dicts else None
    return sources, stacked, names


def won(place, crashed, k) -> bool:
    """Seat 0 against seat k at the end of a race (rows of place and crashed):
    a car that crashed loses to one that did not; otherwise the place decides."""
    if bool(crashed[0]) != bool(crashed[k]):
        return not bool(crashed[0])
    return bool(place[0] < place[k])


def kickstart_weight(w0: float, frac: float, steps: float, total: float) -> float:
    return w0 * max(0.0, 1.0 - steps / max(frac * total, 1.0)) if frac > 0 else 0.0


def gaussian_kl_each(mean_old, mean_new, log_std_old, log_std_new):
    """Per-sample KL(old || new) of diagonal Gaussians, summed over action dims."""
    var_old, var_new = torch.exp(2 * log_std_old), torch.exp(2 * log_std_new)
    return (log_std_new - log_std_old + (var_old + (mean_old - mean_new) ** 2) / (2 * var_new) - 0.5).sum(-1)


def train(args, stop) -> bool:
    common.seed_everything(args.seed)
    dev = args.device
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = PPOConfig(lr=args.lr)
    init = torch.load(args.init_from, map_location=dev, weights_only=False)
    recurrent = 'mlp' not in args.role
    if init.get('policy_config', {}).get('recurrent', True) != recurrent:
        print(f'[league] WARNING: {args.role} starts from {args.init_from}, trained with recurrent='
              f'{not recurrent}', flush=True)
    width = init.get('policy_config', {}).get('width', FULL)
    policy = PolicyNet((tuple(width[0]), width[1]), recurrent=recurrent).to(dev)
    policy.load_state_dict(init['policy'])
    with torch.no_grad():
        policy.log_std.fill_(LOG_STD0)
    critic, vnorm = Critic().to(dev), ValueNorm(dev)
    if args.critic_from:
        last = torch.load(f'{args.critic_from}/checkpoints/latest.pt', map_location=dev, weights_only=False)
        load_expanded(critic, last['critic'])
        vnorm.load(last['vnorm'])
    teacher = t_squash = None
    view = lambda p: p          # noqa: E731
    if args.teacher:
        tck = torch.load(args.teacher, map_location=dev, weights_only=False)
        teacher, t_squash = teacher_eval.actor_from(tck['actor'], dev), teacher_squash(tck.get('squash', 'plain'))
        if not tck.get('traffic', False):
            view = solo_view
    upd = Update(cfg, policy.parameters(), critic.parameters(), args.actor_lr, cfg.lr, dev, args.micro)
    archive = Archive(args.archive or (args.run_dir.parent / 'archive'))
    rates = matchmaking.WinRates()
    runner = Runner(args.num_envs, args.max_cars, dev, args.seed, pool_slots=args.pool_slots,
                    cpu_workers=args.cpu_workers, library=args.library,
                    reward_cfg=RewardConfig(curvature=args.cap_penalty))
    sim, E, T = runner.sim, args.num_envs, args.rollout
    seats = SeatAssignment(E, args.max_cars, dev)
    log = common.MetricLog(args.run_dir)
    update, steps, role_steps, frozen, best = 0, 0, 0, 0, None
    ck = common.load_checkpoint(args.run_dir) if args.resume else None
    if ck:
        policy.load_state_dict(ck['policy']), critic.load_state_dict(ck['critic'])
        upd.load(ck)
        vnorm.load(ck['vnorm'])
        rates.load(ck['rates'])
        update, steps, role_steps, frozen = ck['update'], ck['steps'], ck.get('role_steps', 0), ck.get('frozen', 0)
        best = ck.get('best')
    elif args.critic_from:
        # the teacher's critic raced alone: give it the other cars as this league sees them
        fresh_opponent_inputs(critic.enc, sim.privileged()[:, 0])
    ev = None
    if args.eval_every > 0 and args.role.startswith('main') and not args.role.startswith('main_exploiter'):
        ev = teacher_eval.Evaluator(starts=args.eval_starts, device=dev, limit=args.eval_tracks,
                                    max_seconds=args.eval_seconds, real=not args.eval_tracks)
    h = torch.zeros(E, GRU, device=dev)
    start = torch.ones(E, dtype=torch.bool, device=dev)
    cache = OrderedDict()

    def save(name='latest', **extra):
        common.save_checkpoint(args.run_dir, {
            'policy': policy.state_dict(), 'critic': critic.state_dict(), **upd.state(), 'vnorm': vnorm.state(),
            'rates': rates.state(), 'update': update, 'steps': steps, 'role_steps': role_steps, 'frozen': frozen,
            'best': best, 'policy_config': {'width': policy.width, 'recurrent': recurrent},
            'run_name': args.run_dir.name, **extra}, name)

    def evaluate():
        nonlocal best
        t0 = time.time()
        with torch.no_grad():
            res = student_eval.evaluate(ev, student_eval.StudentPolicy(policy), args.eval_starts)
        policy.train()
        for t, r in res.items():
            log.write(kind='eval', type=t, update=update, steps=steps, **r)
            print(student_eval.line(f'u{update:06d} {t}', r), flush=True)
        score = res['all']['finish_lo']
        if best is None or score > best['score']:
            best = {'score': score, 'update': update, 'steps': steps, **res['all']}
            save('best', eval=res)
        print(f'[league] eval took {time.time() - t0:.0f}s', flush=True)

    def opponents():
        entries = [e.entry_id for e in archive.entries()]
        mains = [e for e in entries if e.startswith('main0/') or e.startswith('main1')]
        return matchmaking.sample_opponents(args.role, args.opponents, entries, rates, rng, main_ids=mains), mains

    refs, mains = opponents()
    table = OpponentTable(refs)
    seats.assign(list(range(E)), table.n, gen, table.offset)      # the Runner has just started every race
    while steps < args.steps:
        if stop.requested:
            save()
            return False
        runner.refresh_tracks()
        refs, mains = opponents()
        table.refresh(refs, seats, sim.alive)
        sources, stacked, names = build_sources(table.refs, archive, policy, cache)
        h0 = h.clone()
        keys = (*OBS_KEYS, 'priv', 'a', 'mean', 'logp', 'v', 'v_next', 'r', 'term', 'done', 'lvl', 'start', 'alive',
                'train', 'ego_t', 'opp_t', 'v_cap') + (('kick',) if teacher is not None else ())
        buf = {k: [] for k in keys}
        t0 = time.time()
        for _ in range(T):
            o, priv = sim.observe(), sim.privileged()
            ego_t, opp_t = aux_targets(sim)
            o0, p0 = {k: v[:, 0] for k, v in o.items()}, priv[:, 0]
            with torch.no_grad():
                h = h * (~start).float().unsqueeze(1)
                mean, h, _, _ = policy(o0, h)
                a = mean + torch.exp(policy.log_std) * torch.randn_like(mean)
                logp, v = gaussian_logp(a, mean, policy.log_std), vnorm.denorm(critic(p0))
                if teacher is not None:
                    pt = view(p0)
                    buf['kick'].append(t_squash(teacher(pt), sim.v_cap, pt))
            act = opponent_actions(sim, o, priv, seats, sources, stacked)
            act[:, 0] = squash(a, sim.v_cap)
            lvl = runner.level_t.clone()
            out = sim.step(act)
            with torch.no_grad():       # the state each step reached, before any reset
                v_next = vnorm.denorm(critic(sim.privileged()[:, 0]))
            for k in OBS_KEYS:
                buf[k].append(o0[k])
            for k, val in (('priv', p0), ('a', a), ('mean', mean), ('logp', logp), ('v', v), ('v_next', v_next),
                           ('r', out['reward'][:, 0]), ('term', out['terminal'][:, 0]),
                           ('done', out['terminal'][:, 0] | out['env_done']), ('lvl', lvl),
                           ('start', start.clone()), ('alive', out['alive'][:, 0]), ('train', runner.env_replay.clone()),
                           ('ego_t', ego_t[:, 0]), ('opp_t', opp_t[:, 0]), ('v_cap', sim.v_cap.clone())):
                buf[k].append(val)
            ended = out['env_done'].nonzero().flatten()
            if len(ended):
                place = out['place'][ended].cpu().numpy()
                crashed = out['crashed'][ended].cpu().numpy()
                src = seats.source[ended].cpu().numpy()
                act_mask = sim.active[ended].cpu().numpy()
                for row in range(len(ended)):
                    for k in range(1, sim.C):
                        if act_mask[row, k]:
                            rates.record(names[src[row, k]], won(place[row], crashed[row], k))
            done = runner.end_episodes(out['env_done'])
            start = torch.zeros(E, dtype=torch.bool, device=dev)
            if done:
                start[torch.as_tensor(done, device=dev)] = True
                seats.assign(done, table.n, gen, table.offset)
        t_roll = time.time() - t0
        b = {k: torch.stack(v) for k, v in buf.items()}
        adv, ret = gae(b['r'], b['v'], b['v_next'], b['term'], b['done'], cfg.gamma, cfg.lam)
        runner.add_scores(adv, b['lvl'], b['alive'])
        w_all = (b['alive'] & b['train']).float()
        n_train = w_all.sum()
        adv_n = torch.zeros_like(adv)
        target = torch.zeros_like(ret)
        trained = w_all > 0
        if bool(trained.any()):
            adv_n = torch.where(trained, (adv - adv[trained].mean()) / (adv[trained].std() + 1e-8), adv_n)
            vnorm.update(ret[trained])
            target = vnorm.norm(ret)
        train_actor = frozen >= args.critic_warmup
        log_std_old = policy.log_std.detach().clone()
        k_w = kickstart_weight(args.kickstart, args.kickstart_frac, role_steps, args.steps) if teacher is not None else 0.0

        def critic_loss(p, mb):
            vals = critic(b['priv'][:, p].reshape(-1, PRIV_DIM)).view(T, -1)
            return (0.5 * (vals - target[:, p]) ** 2 * w_all[:, p]).sum() / w_all[:, mb].sum().clamp(min=1.0)

        def actor_loss(p, mb):
            n_mb = w_all[:, mb].sum().clamp(min=1.0)
            alive = b['alive'][:, p].float()
            n_alive = b['alive'][:, mb].float().sum().clamp(min=1.0)
            mean, path, opp, _ = policy.unroll({k: b[k][:, p] for k in OBS_KEYS}, h0[p], b['start'][:, p])
            ratio = torch.exp(gaussian_logp(b['a'][:, p], mean, policy.log_std) - b['logp'][:, p])
            a_n = adv_n[:, p]
            pg = -torch.min(ratio * a_n, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a_n)
            lp, bce, pos, cnt = aux_errors(path, opp, b['ego_t'][:, p], b['opp_t'][:, p])
            n_opp = (b['opp_t'][:, mb, :, 0] * b['alive'][:, mb].float().unsqueeze(-1)).sum().clamp(min=1.0)
            loss = (pg * w_all[:, p]).sum() / n_mb + (cfg.aux_path * (lp * alive).sum()
                                                       + cfg.aux_opp * (bce * alive).sum()) / n_alive \
                + cfg.aux_opp * (pos * alive).sum() / n_opp
            if k_w > 0:
                loss = loss + k_w * (action_error(mean, b['kick'][:, p], b['v_cap'][:, p]) * alive).sum() / n_alive
            kl = (gaussian_kl_each(b['mean'][:, p], mean.detach(), log_std_old, policy.log_std.detach())
                  * w_all[:, p]).sum() / n_mb
            return loss, kl
        res = upd.run(E, critic_loss, actor_loss, train_actor, scale=False)
        critic.update_norm(b['priv'].reshape(-1, PRIV_DIM))
        frozen = min(frozen + 1, args.critic_warmup)
        update += 1
        steps += T * E
        role_steps += T * E
        if update % args.snapshot_every == 0:
            archive.add(args.role, update, {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                        {'steps': steps, 'recurrent': recurrent, 'width': policy.width})
            reset = False
            if args.role.startswith('main_exploiter'):
                vs_main = [rates.p(m) for m in mains[-1:]] or [0.0]
                reset = matchmaking.exploiter_should_reset(float(np.mean(vs_main)), role_steps)
            elif args.role.startswith('league_exploiter'):
                reset = rng.random() < 0.25
            if reset:
                # back to the student, as the role started: its noise and a fresh actor step too
                policy.load_state_dict(init['policy'])
                with torch.no_grad():
                    policy.log_std.fill_(LOG_STD0)
                upd.actor_opt.state.clear()
                upd.lr = args.actor_lr
                for g in upd.actor_opt.param_groups:
                    g['lr'] = upd.lr
                role_steps = 0
        if update % args.log_every == 0:
            alive = b['alive'].float().sum().clamp(min=1.0)
            crashes = (b['term'] & b['alive']).float().sum()
            ev_var = 1 - torch.var(ret[trained] - b['v'][trained]) / torch.var(ret[trained]).clamp(min=1e-8) \
                if bool(trained.any()) else torch.zeros(())
            log.write(update=update, steps=steps, reward=float(b['r'].mean()), hazard=float(crashes / alive),
                      actor_trained=train_actor, kl=res['kl'], lr=upd.lr, value_loss=res['value_loss'],
                      explained_var=float(ev_var), kickstart=k_w, trained=float(n_train),
                      opponent_half=table.live, steps_per_s=T * E / t_roll,
                      steps_per_s_total=T * E / (time.time() - t0))
        if update % args.checkpoint_every == 0:
            save()
        if ev is not None and update % args.eval_every == 0 and not stop.requested:
            evaluate()
    if ev is not None:
        evaluate()
    save()
    return True


def main(argv=None) -> int:
    return common.run_stage(train, parser().parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
