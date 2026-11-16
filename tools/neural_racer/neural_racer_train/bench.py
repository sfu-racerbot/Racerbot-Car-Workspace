"""bench: a student against car 1's current algorithms, in the referee (plan Phase 9's panel).

    python -m neural_racer_train.bench CKPT [CKPT ...] [--laps 3] [--v-cap 5.0] [--a-lat 2.0] [--workers 16]

The panel is what the car runs today, from this checkout: gap_follow
(src/gap_follow and its gap_follow.yaml, through tools/f1tenth_sim's
gap_mirror) and pure pursuit on the map's line (run_validation's
PathFollower). The league's cars start from here; this is the floor they
have to beat, not the target.

Each driver races alone on every track (finish, lap time, wall contacts),
then each student races each panel member head to head, grid order
alternating (wins, at-fault contacts). Tracks: car 1's course and the other
obstacle-free real maps, and held-out procedural tracks.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

from neural_racer_train import common  # noqa: F401  (paths)
from neural_racer_train import heldout, teacher_eval

PANEL = ('gap_follow', 'pure_pursuit')


def _factory(driver: str, ckpt: str | None, v_cap: float, a_lat: float, vibration: float = 1.0, seed: int = 0):
    from neural_racer_train import referee
    if driver == 'gap_follow':
        return referee.gap_factory()
    if driver == 'pure_pursuit':
        return referee.pursuit_factory()
    from neural_racer_train.export import load_policy, weights_of
    policy, _ = load_policy(ckpt)
    return referee.neural_factory(weights_of(policy), {'recurrent': policy.recurrent}, v_cap=v_cap, a_lat=a_lat,
                                  imu_vibration=vibration, seed=seed)


def _race(job):
    """(track, [(name, driver, ckpt)], seed, laps, timeout, v_cap, a_lat, vibration) -> result dict or {'error'}."""
    track, entrants, seed, laps, timeout, v_cap, a_lat, vibration = job
    try:
        from neural_racer_train import referee
        drivers = {name: _factory(driver, ckpt, v_cap, a_lat, vibration, seed) for name, driver, ckpt in entrants}
        t0 = time.time()
        r = referee.race(track, drivers, seed=seed, laps=laps, timeout_s=timeout)
        return {'track': track, 'seed': seed, 'names': r.names, 'order': r.order, 'finish': r.finish_time,
                'laps': r.laps, 'contact': r.contact, 'fault': r.at_fault, 'progress': r.progress,
                'wall_s': round(time.time() - t0, 1)}
    except BaseException as exc:  # noqa: BLE001 -- one failed race is reported, never fatal
        return {'track': track, 'seed': seed, 'error': f'{type(exc).__name__}: {exc}'}


def jobs(students: list, tracks: list, laps: int, timeout: float, v_cap: float, a_lat: float,
         solo_seeds: int, h2h_seeds: int, vibration: float = 1.0) -> list:
    out = []
    drivers = [(f'student{k}', 'neural', c) for k, c in enumerate(students)] + [(p, p, None) for p in PANEL]
    for t in tracks:
        for name, driver, ckpt in drivers:
            for s in range(solo_seeds):
                out.append(('solo', name, (t, [(name, driver, ckpt)], s, laps, timeout, v_cap, a_lat, vibration)))
        for k, c in enumerate(students):
            for p in PANEL:
                for s in range(h2h_seeds):
                    pair = [(f'student{k}', 'neural', c), (p, p, None)]
                    out.append((f'student{k}|{p}', None, (t, pair if s % 2 == 0 else pair[::-1], s, laps, timeout,
                                                          v_cap, a_lat, vibration)))
    return out


def summarize(results: list, students: list) -> dict:
    """{'solo': {track: {driver: {...}}}, 'h2h': {pair: {track: {...}}}}."""
    from neural_racer_train import referee
    solo, h2h, errors = {}, {}, []
    for kind, name, res in results:
        if 'error' in res:
            errors.append((kind, res['track'], res['error']))
            continue
        t = res['track']
        if kind == 'solo':
            fin = res['finish'][0]
            row = solo.setdefault(t, {}).setdefault(name, {'races': 0, 'finished': 0, 'lap_s': [], 'contact': 0})
            row['races'] += 1
            row['contact'] += int(res['contact'][0])
            if fin is not None:
                row['finished'] += 1
                row['lap_s'].append(fin / max(res['laps'][0], 1.0))
        else:
            s_name, p = kind.split('|')
            i_s, i_p = res['names'].index(s_name), res['names'].index(p)
            row = h2h.setdefault(kind, {}).setdefault(t, {'races': 0, 'wins': 0, 'losses': 0, 'draws': 0,
                                                          'from_second': 0, 'wins_from_second': 0,
                                                          'fault_student': 0, 'fault_panel': 0})
            # scored on finishing order (referee.finish_outcome); the grid is the drivers' order, front first
            out = referee.finish_outcome(res['names'], res['finish'], s_name, p)
            row['races'] += 1
            row[{'win': 'wins', 'loss': 'losses', 'draw': 'draws'}[out]] += 1
            if i_s > i_p:                         # the student started behind: can it get past?
                row['from_second'] += 1
                row['wins_from_second'] += int(out == 'win')
            row['fault_student'] += int(res['fault'][i_s])
            row['fault_panel'] += int(res['fault'][i_p])
    for t in solo.values():
        for row in t.values():
            laps = row.pop('lap_s')
            row['lap_s'] = round(sum(laps) / len(laps), 2) if laps else None
    return {'students': students, 'solo': solo, 'h2h': h2h, 'errors': errors}


def print_report(rep: dict) -> None:
    drivers = sorted({d for t in rep['solo'].values() for d in t})
    print('BENCH solo (finished/races, mean lap s, wall contacts):', flush=True)
    for t, rows in rep['solo'].items():
        cells = [f"{d} {rows[d]['finished']}/{rows[d]['races']} {rows[d]['lap_s']}s c{rows[d]['contact']}"
                 for d in drivers if d in rows]
        print(f'BENCH   {t:<26} ' + ' | '.join(cells), flush=True)
    print('BENCH head to head, on finishing order (student wins-losses-draws of races; wins when starting '
          'second; at-fault contacts per 100 races, student/panel):', flush=True)
    for pair, by_track in rep['h2h'].items():
        tot = {k: sum(r[k] for r in by_track.values()) for k in next(iter(by_track.values()))}
        n = max(tot['races'], 1)
        print(f"BENCH   {pair:<24} all {tot['wins']}-{tot['losses']}-{tot['draws']} of {tot['races']}, "
              f"from second {tot['wins_from_second']}/{tot['from_second']}, at-fault per 100 "
              f"{100 * tot['fault_student'] / n:.0f}/{100 * tot['fault_panel'] / n:.0f} | " +
              ' '.join(f"{t}:{r['wins']}-{r['losses']}-{r['draws']}" for t, r in by_track.items()), flush=True)
    for e in rep['errors'][:5]:
        print('BENCH error', e, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('students', nargs='+')
    ap.add_argument('--laps', type=int, default=3)
    ap.add_argument('--timeout', type=float, default=150.0, help='s of simulated time per race')
    ap.add_argument('--v-cap', type=float, default=5.0, help="the student's speed cap: gap_follow's race max_speed")
    ap.add_argument('--a-lat', type=float, default=2.0, help="the student's cornering cap (neural_racer.yaml's 2.0)")
    ap.add_argument('--imu-vibration', type=float, default=1.0,
                    help="the student's IMU noise, as a share of what car 1's reads while driving; 0 = exact")
    ap.add_argument('--procedural', type=int, default=6, help='held-out procedural tracks')
    ap.add_argument('--solo-seeds', type=int, default=2)
    ap.add_argument('--h2h-seeds', type=int, default=4)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args(argv)
    heldout.ensure_maps()
    procedural = [n for n in heldout.names() if n.startswith('heldout_')][:a.procedural]
    tracks = list(teacher_eval.OBSTACLE_FREE_REAL) + procedural
    todo = jobs(a.students, tracks, a.laps, a.timeout, a.v_cap, a.a_lat, a.solo_seeds, a.h2h_seeds, a.imu_vibration)
    print(f'[bench] {len(todo)} races on {len(tracks)} tracks, {a.workers} workers', flush=True)
    t0 = time.time()
    with mp.get_context('spawn').Pool(a.workers) as pool:
        results = pool.map(_race, [j for _, _, j in todo], chunksize=1)
    rep = summarize([(k, n, r) for (k, n, _), r in zip(todo, results)], a.students)
    rep.update(laps=a.laps, v_cap=a.v_cap, a_lat=a.a_lat, imu_vibration=a.imu_vibration, tracks=tracks,
               seconds=round(time.time() - t0))
    print_report(rep)
    if a.out:
        a.out.write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
