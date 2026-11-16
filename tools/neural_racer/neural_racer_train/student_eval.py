"""Held-out evaluation of a student (plan item 1.2, training plan step 3d):
the races of teacher_eval plus the four real course maps (realmaps), run
as the car runs them -- the student's own
LiDAR, IMU and observation, the whole safety wrapper on, the cornering cap
at the car's 2.0 m/s^2 -- once per episode type, since gate S judges map
mode and reactive mode separately. On the GPU: the scan is too slow on a CPU.

    python -m neural_racer_train.student_eval CKPT [--teacher T.pt] [--gap-follow] [--starts 4]

Each result is teacher_eval.summarize's (`real_maps` per map), plus `intervened` (the share of
live ticks the wrapper changed the command: slew, curvature or braking cap)
and `safety_stop` (the LiDAR braking cap bound, or the wrapper stopped the car).
"""
import argparse
import sys

import torch

from neural_racer_train import scripted, teacher_eval
from neural_racer_train.gpu_sim.sim import EPISODE_HANDOVER, EPISODE_MAP, EPISODE_REACTIVE
from neural_racer_train.model import WIDTHS, PolicyNet, squash

TYPES = {'reactive': EPISODE_REACTIVE, 'map': EPISODE_MAP, 'handover': EPISODE_HANDOVER}
CAR_A_LAT = 2.0


class StudentPolicy:
    """The student as the node runs it: its observation, its memory, the plain squash."""
    uses_sim = True

    def __init__(self, net: PolicyNet):
        self.net = net.eval()
        self.h = None

    def reset(self, E: int) -> None:
        self.h = torch.zeros(E, 128, device=next(self.net.parameters()).device)

    def __call__(self, sim):
        o = {k: v[:, 0] for k, v in sim.observe().items()}
        a, self.h, _, _ = self.net(o, self.h)
        return squash(a, sim.v_cap)


class GapFollowPolicy:
    """The scripted gap follower (scripted.gap_follow) on the newest scan: gate S's reactive baseline."""
    uses_sim = True

    def __init__(self, pace: float = 1.0):
        self.pace = pace

    def __call__(self, sim):
        scan = sim.observe()['scan'][:, 0]
        return scripted.gap_follow(scan, sim.v_cap, torch.full_like(sim.v_cap, self.pace))


def evaluate(ev: teacher_eval.Evaluator, policy, starts: int, types=tuple(TYPES), a_lat: float = CAR_A_LAT,
             draw: int | None = None) -> dict:
    """{type: summary} and 'all' (every type's races together), all types
    raced in one batch. draw: fresh starts from that draw instead of the fixed ones."""
    seed = teacher_eval.FIXED_LEVEL_SEED if draw is None else teacher_eval.FRESH_SEED + 10_000 * draw
    sim_seed = ev.seed if draw is None else seed
    rec = ev.races(policy, starts=starts, level_seed=seed, sim_seed=sim_seed, a_lat=a_lat, wrapper=True,
                   episode_type=[TYPES[name] for name in types])
    out = {name: teacher_eval.summarize([teacher_eval.split(rec, rec['episode_type'] == TYPES[name])], ev.N, ev.names)
           for name in types}
    out['all'] = teacher_eval.summarize([rec], ev.N, ev.names)
    return out


def load_student(path, device='cpu') -> PolicyNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get('policy_config', {})
    width = cfg.get('width', WIDTHS['full'])
    net = PolicyNet(tuple((tuple(width[0]), width[1])), recurrent=cfg.get('recurrent', True)).to(device)
    net.load_state_dict(ck['policy'])
    return net.eval()


def line(name: str, r: dict) -> str:
    return (teacher_eval.line(name, r) + f' intervened={r["intervened"]:.4f} safety_stop={r["safety_stop"]:.4f}')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('ckpts', nargs='*')
    ap.add_argument('--teacher', default=None, help='also race this teacher, with the same wrapper')
    ap.add_argument('--gap-follow', action='store_true', help='also race the scripted gap follower')
    ap.add_argument('--starts', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args(argv)
    ev = teacher_eval.Evaluator(starts=a.starts, device=a.device, real=True)
    named = []
    if a.teacher:
        actor, mode = teacher_eval.load_actor(a.teacher, a.device)
        named.append((f'teacher ({mode})', teacher_eval.teacher_policy(actor, mode)))
    if a.gap_follow:
        named.append(('gap_follow', GapFollowPolicy()))
    named += [(p, StudentPolicy(load_student(p, a.device))) for p in a.ckpts]
    for name, pol in named:
        with torch.no_grad():
            res = evaluate(ev, pol, a.starts)
        for t, r in res.items():
            print(line(f'{name} {t}', r), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
