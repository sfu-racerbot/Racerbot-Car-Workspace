"""Where the exported policy loses time or fails on the club's SLAM courses.

    RACERBOT_CAR=rb1 python3 -m neural_racer_train.diagnose \\
        --model ../../src/neural_racer/models/a0eef2af0a3757b6 --out /tmp/diag

Races the exported policy (History -> PolicyNumpy -> SafetyWrapper, the car's
code path) and gap_follow solo on each course, on the referee's calibrated
CPU simulator with each course's own car profile, and records every tick:
where the car is along the line, its speed, the network's speed request and
what the safety wrapper let through (`binding`). Writes runs.json and
report.md: finishes, where each failure happened and what the wrapper was
doing in the second before it, and the 1 m segments where the policy spends
the most time over gap_follow. Offline and in memory: no ROS, no training.
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

DT = 0.025
SEGMENT_M = 1.0
# The club's own SLAM courses, each raced on the profile of the car that mapped it.
CLUB_COURSES = {
    'car1_course_2026_10_26': 'rb1',
    'room_2026_07_27': 'rb1',
    'course_2026_10_30': 'rb2',
    'car2_course_2026_11_06a': 'rb2',
    'car2_course_2026_11_06b': 'rb2',
    'car2_course_2026_11_11': 'rb2',
}


def load_model(model_dir: Path):
    w = dict(np.load(model_dir / 'weights.npz'))
    card = yaml.safe_load((model_dir / 'model_card.yaml').read_text())
    return w, card


def _run(job):
    course, car, driver, seed, laps, v_cap, model_dir = job
    os.environ['RACERBOT_CAR'] = car
    from neural_racer_train import referee
    ticks = []
    extra = {}

    if driver == 'neural':
        w, card = load_model(Path(model_dir))

        def factory(plant):
            d = referee.NeuralDriver(w, card, v_cap=v_cap)
            step, apply = d.policy.step, d.wrapper.apply

            def rec_step(o):
                out = step(o)
                extra['net_speed'], extra['net_steer'] = float(out.speed), float(out.steering)
                return out

            def rec_apply(*a, **k):
                res = apply(*a, **k)
                extra['binding'] = 'stopped:' + str(res.reason) if res.stopped else res.binding
                extra['cmd_speed'] = float(res.speed)
                extra['clearance'] = float(res.clearance)
                return res
            d.policy.step, d.wrapper.apply = rec_step, rec_apply
            return d
    else:
        factory = referee.gap_factory()

    def on_tick(step, i, truth, progress, lap):
        ticks.append({'t': round((step + 1) * DT, 3), 'x': round(float(truth[0]), 3),
                      'y': round(float(truth[1]), 3), 'v': round(float(truth[3]), 3),
                      's': round(float(progress), 3), 'lap': lap,
                      'binding': extra.get('binding', 'n/a'),
                      'net_speed': extra.get('net_speed'), 'cmd_speed': extra.get('cmd_speed'),
                      'clearance': extra.get('clearance')})
        extra.clear()

    r = referee.race(course, {driver: factory}, seed, laps=laps, timeout_s=40.0 * laps, on_tick=on_tick)
    return {'course': course, 'car': car, 'driver': driver, 'seed': seed, 'laps': laps,
            'finish_time': r.finish_time[0], 'contact': r.contact[0], 'progress': r.progress[0],
            'ticks': ticks}


def lap_length(runs):
    """Line length from the first lap any run completed on this course."""
    for r in runs:
        for t in r['ticks']:
            if t['lap'] >= 1:
                return t['s']
    return None


def dwell(run, L):
    """Seconds spent per 1 m segment of the lap, averaged over the laps driven through it."""
    if not L:
        return {}
    acc, laps = defaultdict(float), defaultdict(set)
    for t in run['ticks']:
        s = t['s'] % L
        b = int(s // SEGMENT_M)
        acc[b] += DT
        laps[b].add(int(t['s'] // L))
    return {b: acc[b] / len(laps[b]) for b in acc}


def failure(run):
    if run['finish_time'] is not None:
        return None
    tk = run['ticks']
    last = tk[-40:]
    stuck = sum(1 for t in tk[-80:] if t['v'] < 0.2) > 60
    return {'kind': 'contact' if run['contact'] else ('stuck' if stuck else 'timeout'),
            't': tk[-1]['t'] if tk else 0.0, 's': run['progress'],
            'x': tk[-1]['x'] if tk else None, 'y': tk[-1]['y'] if tk else None,
            'v_last': tk[-1]['v'] if tk else None,
            'bindings_last_1s': dict(Counter(t['binding'] for t in last)),
            'net_speed_last_1s': round(float(np.mean([t['net_speed'] for t in last if t['net_speed'] is not None] or [0])), 2),
            'cmd_speed_last_1s': round(float(np.mean([t['cmd_speed'] for t in last if t['cmd_speed'] is not None] or [0])), 2)}


def summarize(runs, top=5):
    by_course = defaultdict(list)
    for r in runs:
        by_course[r['course']].append(r)
    out = {}
    for course, rs in by_course.items():
        L = lap_length(rs)
        c = {'car': rs[0]['car'], 'lap_length_m': round(L, 2) if L else None, 'drivers': {}}
        for drv in ('neural', 'gap_follow'):
            dr = [r for r in rs if r['driver'] == drv]
            fin = [r['finish_time'] / r['laps'] for r in dr if r['finish_time'] is not None]
            entry = {'finished': f'{len(fin)}/{len(dr)}',
                     'mean_lap_s': round(float(np.mean(fin)), 2) if fin else None,
                     'mean_speed': round(float(np.mean([t['v'] for r in dr for t in r['ticks']] or [0])), 2),
                     'failures': [f for f in (failure(r) for r in dr) if f]}
            if drv == 'neural':
                b = Counter(t['binding'] for r in dr for t in r['ticks'])
                n = sum(b.values()) or 1
                entry['binding_share'] = {k: round(v / n, 3) for k, v in b.most_common()}
                gaps = [t['net_speed'] - t['cmd_speed'] for r in dr for t in r['ticks']
                        if t['net_speed'] is not None and t['cmd_speed'] is not None]
                entry['mean_speed_cut_by_wrapper'] = round(float(np.mean(gaps)), 3) if gaps else None
            c['drivers'][drv] = entry
        nd = [dwell(r, L) for r in rs if r['driver'] == 'neural']
        gd = [dwell(r, L) for r in rs if r['driver'] == 'gap_follow']
        if L and nd and gd:
            segs = []
            for b in range(int(math.ceil(L / SEGMENT_M))):
                n_t = [d[b] for d in nd if b in d]
                g_t = [d[b] for d in gd if b in d]
                if not n_t or not g_t:
                    continue
                pts = [t for r in rs if r['driver'] == 'neural' for t in r['ticks']
                       if int((t['s'] % L) // SEGMENT_M) == b]
                segs.append({'segment_m': [b * SEGMENT_M, (b + 1) * SEGMENT_M],
                             'loss_s': round(float(np.mean(n_t) - np.mean(g_t)), 3),
                             'x': round(float(np.mean([p['x'] for p in pts])), 2),
                             'y': round(float(np.mean([p['y'] for p in pts])), 2),
                             'neural_v': round(float(np.mean([p['v'] for p in pts])), 2),
                             'binding': Counter(p['binding'] for p in pts).most_common(1)[0][0]})
            c['total_segment_loss_s'] = round(sum(s['loss_s'] for s in segs), 2)
            c['worst_segments'] = sorted(segs, key=lambda s: -s['loss_s'])[:top]
        out[course] = c
    return out


def report_md(summary) -> str:
    lines = ['| Course | Car | Neural | gap_follow | Neural mean v | Wrapper binding (non-network share) |',
             '|---|---|---|---|---|---|']
    for course, c in summary.items():
        n, g = c['drivers']['neural'], c['drivers']['gap_follow']
        share = {k: v for k, v in n.get('binding_share', {}).items() if k != 'network'}
        lines.append(f"| {course} | {c['car']} | {n['finished']}, {n['mean_lap_s']} s | "
                     f"{g['finished']}, {g['mean_lap_s']} s | {n['mean_speed']} m/s | "
                     + ', '.join(f'{k} {v:.0%}' for k, v in share.items()) + ' |')
    for course, c in summary.items():
        lines.append(f'\n## {course}\n')
        for f in c['drivers']['neural']['failures']:
            lines.append(f"- Neural {f['kind']} at {f['s']} m ({f['x']}, {f['y']}), t {f['t']} s, "
                         f"v {f['v_last']} m/s; last 1 s: net {f['net_speed_last_1s']} m/s, "
                         f"wrapper {f['cmd_speed_last_1s']} m/s, bindings {f['bindings_last_1s']}")
        for s in c.get('worst_segments', []):
            lines.append(f"- Loses {s['loss_s']} s/lap at {s['segment_m'][0]:.0f}-{s['segment_m'][1]:.0f} m "
                         f"({s['x']}, {s['y']}), neural {s['neural_v']} m/s, mostly {s['binding']}")
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--courses', default=','.join(CLUB_COURSES))
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--laps', type=int, default=3)
    ap.add_argument('--v-cap', type=float, default=5.05)
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    args = ap.parse_args(argv)
    jobs = [(c, CLUB_COURSES[c], d, int(s), args.laps, args.v_cap, str(args.model.resolve()))
            for c in args.courses.split(',') for d in ('neural', 'gap_follow') for s in args.seeds.split(',')]
    with ProcessPoolExecutor(args.workers) as ex:
        runs = list(ex.map(_run, jobs))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'runs.json').write_text(json.dumps(runs))
    summary = summarize(runs)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    (args.out / 'report.md').write_text(report_md(summary))
    print(report_md(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
