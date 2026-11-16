"""Stage 2: distil the privileged teacher into the deployable student by DAgger
(plan A8, Phase 6).

    python -m neural_racer_train.distill --run-dir DIR --teacher T.pt --resume

The student drives, and the teacher labels what it would have done in every
state the student reaches; a new episode is the teacher's with probability
0.5^i at iteration i (--iteration-steps). The labels are the teacher's own
commands: its checkpoint says which squash it drives with (teacher v4's holds
the wrapper's curvature cap). Opponents are scripted only; --max-cars 1 races
alone. The car's whole safety wrapper is on, as on the car.

The update runs the student's encoders over a stored rollout at once and
steps only its GRU (PolicyNet.unroll, plan item 1.4), in pieces of --micro
environments. --latent adds a training-only head that regresses the teacher's
own encoding of the state (item 2.5); it is never exported. --teacher-mask
labels the reactive-mode states with the teacher's input hidden where the
LiDAR cannot see (item 2.3); use it with a teacher trained that way
(teacher.py --mask-share).

Every --eval-every updates the student races the held-out tracks with the
wrapper on, once per episode type (student_eval); best.pt is the snapshot
with the best lower bound of its finish rate over all of them. A new run
first races the teacher there; if the wrapper brakes it most of the time
(--max-teacher-stop) or it finishes too few races (--min-teacher-finish),
the run stops: the student's world is broken, and no student can learn it.

Run after Phase 0's gate, so the simulated IMU uses measured noise and delay;
until then a run is provisional (decision D8).
"""
import sys
import time
from collections import deque

import torch
from torch import nn

from neural_racer import observation as obs
from neural_racer_train import common, matchmaking, student_eval, teacher_eval
from neural_racer_train.learn import Runner, SeatAssignment, aux_targets, opponent_actions
from neural_racer_train.model import GRU, WIDTHS, PolicyNet, mask_track, solo_view, squash, teacher_squash
from neural_racer_train.ppo import PPOConfig

OBS_KEYS = ('scan', 'ego', 'preview', 'race', 'camera')


def parser():
    ap = common.base_parser(__doc__.split('\n')[0])
    ap.add_argument('--teacher', type=str, required=True)
    ap.add_argument('--steps', type=float, default=3e8)
    ap.add_argument('--seq', type=int, default=64)
    ap.add_argument('--iteration-steps', type=float, default=2e7)
    ap.add_argument('--buffer-rollouts', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max-cars', type=int, default=4)
    ap.add_argument('--pool-slots', type=int, default=4096)
    ap.add_argument('--library', default=None, help='a tracklib directory to fill the track pool from')
    ap.add_argument('--width', choices=sorted(WIDTHS), default='full')
    ap.add_argument('--memory', choices=('gru', 'none'), default='gru', help='none: the MLP ablation')
    ap.add_argument('--latent', type=float, default=0.0, help="weight of the teacher-encoding head; 0 = off")
    ap.add_argument('--teacher-mask', action='store_true', help="reactive-mode labels from the masked teacher")
    ap.add_argument('--a-lat-range', default='2.0,5.0', help='cornering caps drawn per level, m/s^2')
    ap.add_argument('--micro', type=int, default=1024, help='environments per forward/backward in the update')
    ap.add_argument('--eval-every', type=int, default=25, help='updates between held-out races; 0 = off')
    ap.add_argument('--eval-starts', type=int, default=4, help='per held-out track and episode type')
    ap.add_argument('--eval-tracks', type=int, default=None, help='the first N held-out tracks (smoke runs)')
    ap.add_argument('--eval-seconds', type=float, default=teacher_eval.MAX_SECONDS, help='race limit (smoke runs)')
    ap.add_argument('--min-teacher-finish', type=float, default=0.8,
                    help="the teacher's finish rate in the student's races below which the run stops; 0 = off")
    ap.add_argument('--max-teacher-stop', type=float, default=0.2,
                    help="the share of the teacher's ticks the LiDAR braking cap may bind before the run stops")
    return ap


def check_reference(r: dict, min_finish: float, max_stop: float) -> None:
    """Stop before training when the teacher cannot drive the student's races
    (the wrapper's braking cap once held every car to 0.3 m/s there)."""
    if r['safety_stop'] > max_stop or r['finish'] < min_finish:
        raise SystemExit(f"[distill] the teacher finishes {r['finish']:.1%} of the student's races "
                         f"(--min-teacher-finish {min_finish:.0%}) and the braking cap binds on "
                         f"{r['safety_stop']:.1%} of its ticks (--max-teacher-stop {max_stop:.0%}): "
                         'the student cannot learn there; not training')


def action_error(a_student, target, v_cap):
    """Per sample: squared error of the squashed command against the teacher's,
    steering over its range and speed over V_SCALE -- what the car would do."""
    s = squash(a_student, v_cap)
    return ((s[..., 0] - target[..., 0]) / obs.STEER_LIMIT) ** 2 + ((s[..., 1] - target[..., 1]) / obs.V_SCALE) ** 2


def action_loss(a_student, target, v_cap):
    return action_error(a_student, target, v_cap).mean()


def aux_errors(path, opp, ego_t, opp_t):
    """Per sample: the ego path's MSE, the opponents' presence BCE, and their
    position error summed over the slots that are present (with the count)."""
    l_path = ((path - ego_t) ** 2).mean((-1, -2))
    present = opp_t[..., 0]
    bce = nn.functional.binary_cross_entropy_with_logits(opp[..., 0], present, reduction='none').mean(-1)
    pos = (((opp[..., 1:] * obs.AUX_POS_SCALE - opp_t[..., 1:]) ** 2).mean(-1) * present).sum(-1)
    return l_path, bce, pos, present.sum(-1)


def train(args, stop) -> bool:
    common.seed_everything(args.seed)
    dev = args.device
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    student = PolicyNet(WIDTHS[args.width], recurrent=args.memory == 'gru').to(dev)
    tck = torch.load(args.teacher, map_location=dev, weights_only=False)
    teacher = teacher_eval.actor_from(tck['actor'], dev)
    t_squash = teacher_squash(tck.get('squash', 'plain'))
    # a teacher trained alone labels from the state as it saw it in training: in a
    # sim with seats for more cars, the empty seats' parked cars clip its inputs
    view = solo_view if args.max_cars > 1 and not tck.get('traffic', False) else (lambda p: p)
    latent = nn.Linear(GRU, teacher.head[0].in_features).to(dev) if args.latent > 0 else None
    params = list(student.parameters()) + (list(latent.parameters()) if latent else [])
    opt = torch.optim.Adam(params, lr=args.lr, **({'fused': True} if str(dev).startswith('cuda') else {}))
    total_iters = max(int(args.steps / max(args.seq * args.num_envs, 1)), 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_iters * args.buffer_rollouts)
    a_lat_range = tuple(float(v) for v in args.a_lat_range.split(','))
    runner = Runner(args.num_envs, args.max_cars, dev, args.seed, pool_slots=args.pool_slots,
                    cpu_workers=args.cpu_workers, solo=args.max_cars == 1, library=args.library,
                    a_lat_range=a_lat_range)
    sim, E, T = runner.sim, args.num_envs, args.seq
    sources = [('scripted', name) for name in matchmaking.SCRIPTED]
    seats = SeatAssignment(E, args.max_cars, dev)
    seats.assign(list(range(E)), len(sources), gen)
    teacher_drives = torch.zeros(E, dtype=torch.bool, device=dev)
    h = torch.zeros(E, GRU, device=dev)
    start = torch.ones(E, dtype=torch.bool, device=dev)     # a new episode began: the memory restarts
    log = common.MetricLog(args.run_dir)
    steps, update, best = 0, 0, None
    ck = common.load_checkpoint(args.run_dir) if args.resume else None
    if ck:
        student.load_state_dict(ck['policy']), opt.load_state_dict(ck['opt']), sched.load_state_dict(ck['sched'])
        if latent:
            latent.load_state_dict(ck['latent'])
        steps, update, best = ck['steps'], ck['update'], ck.get('best')
    buffer = deque(maxlen=args.buffer_rollouts)
    ev = teacher_eval.Evaluator(starts=args.eval_starts, device=dev, limit=args.eval_tracks,
                                max_seconds=args.eval_seconds, real=not args.eval_tracks) if args.eval_every > 0 else None
    print(f'[distill] teacher {args.teacher} ({tck.get("squash", "plain")} squash), student {args.width} '
          f'{args.memory}, latent {args.latent}, {E} envs x {args.max_cars} cars', flush=True)

    def save(name='latest', **extra):
        common.save_checkpoint(args.run_dir, {
            'policy': student.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
            **({'latent': latent.state_dict()} if latent else {}), 'steps': steps, 'update': update, 'best': best,
            'policy_config': {'width': student.width, 'recurrent': student.recurrent},
            'teacher': args.teacher, 'run_name': args.run_dir.name, **extra}, name)

    def evaluate():
        nonlocal best
        t0 = time.time()
        with torch.no_grad():
            res = student_eval.evaluate(ev, student_eval.StudentPolicy(student), args.eval_starts)
        student.train()
        for t, r in res.items():
            log.write(kind='eval', type=t, update=update, steps=steps, **r)
            print(student_eval.line(f'u{update:06d} {t}', r), flush=True)
        score = res['all']['finish_lo']
        if best is None or score > best['score']:
            best = {'score': score, 'update': update, 'steps': steps, **res['all']}
            save('best', eval=res)
        print(f'[distill] eval took {time.time() - t0:.0f}s', flush=True)

    if ev is not None and not ck:
        with torch.no_grad():      # what the student is measured against, on the same races
            for name, pol in (('teacher', teacher_eval.teacher_policy(teacher, tck.get('squash', 'plain'))),
                              ('gap_follow', student_eval.GapFollowPolicy())):
                res = student_eval.evaluate(ev, pol, args.eval_starts)
                for t, r in res.items():
                    log.write(kind='reference', policy=name, type=t, **r)
                    print(student_eval.line(f'{name} {t}', r), flush=True)
                if name == 'teacher':
                    check_reference(res['all'], args.min_teacher_finish, args.max_teacher_stop)

    while steps < args.steps:
        if stop.requested:
            save()
            return False
        runner.refresh_tracks()
        beta = 0.5 ** int(steps // args.iteration_steps)
        h0 = h.clone()
        keys = (*OBS_KEYS, 'target', 'ego_t', 'opp_t', 'start', 'alive', 'v_cap') + (('latent',) if latent else ())
        seq = {k: [] for k in keys}
        diag = {k: torch.zeros((), device=dev) for k in ('alive', 'crash', 'finish', 'intervened', 'stop', 'curv')}
        t0 = time.time()
        for _ in range(T):
            o, priv = sim.observe(), sim.privileged()
            ego_t, opp_t = aux_targets(sim)
            o0, p0 = {k: v[:, 0] for k, v in o.items()}, view(priv[:, 0])
            if args.teacher_mask:
                p0 = mask_track(p0, sim.track_visibility()[:, 0], o0['ego'][:, obs.EGO['mode']] == 0)
            with torch.no_grad():
                a_s, h, _, _ = student(o0, h * (~start).float().unsqueeze(1))
                z_t = teacher.enc(p0)
                cmd_t = t_squash(teacher.head(z_t), sim.v_cap, p0)
            for k in OBS_KEYS:
                seq[k].append(o0[k])
            for k, val in (('target', cmd_t), ('ego_t', ego_t[:, 0]), ('opp_t', opp_t[:, 0]), ('start', start.clone()),
                           ('alive', sim.alive[:, 0].clone()), ('v_cap', sim.v_cap.clone())):
                seq[k].append(val)
            if latent:
                seq['latent'].append(z_t)
            act = opponent_actions(sim, o, priv, seats, sources) if args.max_cars > 1 else \
                torch.zeros(E, 1, 2, device=dev)
            act[:, 0] = torch.where(teacher_drives.unsqueeze(1), cmd_t, squash(a_s, sim.v_cap))
            out = sim.step(act)
            live = out['alive'][:, 0]
            diag['alive'] += live.float().sum()
            diag['crash'] += (out['terminal'][:, 0] & ~out['finished'][:, 0] & live).float().sum()
            diag['finish'] += out['finished'][:, 0].float().sum()
            diag['intervened'] += (out['intervened'][:, 0] & live).float().sum()
            diag['stop'] += (out['safety_stop'][:, 0] & live).float().sum()
            diag['curv'] += ((out['curv_cut'][:, 0] > teacher_eval.CUT_EPS) & live).float().sum()
            done = runner.end_episodes(out['env_done'])
            start = torch.zeros(E, dtype=torch.bool, device=dev)
            if done:
                d = torch.as_tensor(done, device=dev)
                start[d] = True
                seats.assign(done, len(sources), gen)
                teacher_drives[d] = torch.rand(len(done), generator=gen, device=gen.device).to(dev) < beta
        buffer.append(({k: torch.stack(v) for k, v in seq.items()}, h0))
        steps += T * E
        t_roll = time.time() - t0

        # one pass over the buffered rollouts, minibatched by env
        parts = {k: torch.zeros((), device=dev) for k in ('action', 'path', 'opp', 'latent')}
        for b, hb in buffer:
            for mb in torch.randperm(E, device=dev).chunk(4):
                opt.zero_grad()
                w_all = b['alive'][:, mb].float()
                n = w_all.sum().clamp(min=1.0)
                n_opp = (b['opp_t'][:, mb, :, 0] * w_all.unsqueeze(-1)).sum().clamp(min=1.0)
                for c in mb.split(args.micro):
                    w = b['alive'][:, c].float()
                    a, path, opp, H = student.unroll({k: b[k][:, c] for k in OBS_KEYS}, hb[c], b['start'][:, c])
                    la = (action_error(a, b['target'][:, c], b['v_cap'][:, c]) * w).sum() / n
                    lp, bce, pos, _ = aux_errors(path, opp, b['ego_t'][:, c], b['opp_t'][:, c])
                    lp, lo = (lp * w).sum() / n, (bce * w).sum() / n + (pos * w).sum() / n_opp
                    loss = la + PPOConfig.aux_path * lp + PPOConfig.aux_opp * lo
                    ll = torch.zeros((), device=dev)
                    if latent:
                        ll = (((latent(H) - b['latent'][:, c]) ** 2).mean(-1) * w).sum() / n
                        loss = loss + args.latent * ll
                    loss.backward()
                    for k, v in (('action', la), ('path', lp), ('opp', lo), ('latent', ll)):
                        parts[k] += v.detach()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            sched.step()
        update += 1
        if update % args.log_every == 0:
            nb = 4 * len(buffer)
            alive = diag['alive'].clamp(min=1.0)
            log.write(update=update, steps=steps, beta=beta, lr=sched.get_last_lr()[0],
                      **{f'loss_{k}': float(v) / nb for k, v in parts.items()},
                      hazard=float(diag['crash'] / alive), finishes=float(diag['finish']),
                      intervened=float(diag['intervened'] / alive), safety_stop=float(diag['stop'] / alive),
                      curv_cut=float(diag['curv'] / alive), steps_per_s=T * E / t_roll,
                      steps_per_s_total=T * E / (time.time() - t0))
        if update % args.checkpoint_every == 0:
            save()
        if ev is not None and update % args.eval_every == 0 and not stop.requested:
            evaluate()
    if ev is not None:
        evaluate()
    save()
    save('student_final')
    return True


def main(argv=None) -> int:
    return common.run_stage(train, parser().parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
