"""Replay car 2's forward stops on its 2026-10-30 simulator floor.

Run directly to score the compact measurements. To read them again from the
original MCAP files, pass --runs /path/to/Racerbot-Car-2-Workspace/data/racerbot_runs
(revision 9c2c2ed7bbf675674f30ca5937432a62b0eab3a3; needs rosbags==0.10.11).
"""
import argparse
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TOOLS.parent / '.sim' / 'python'), str(TOOLS)]
import numpy as np

from run_analysis.fit_sim import CONTROL_DT, SIM, simulate_commands
from sim_fidelity.calibration import FLOORS, STOP_FLOOR, calibration_for_car, with_floor

FLOOR = 'sfu_course_2026_10_30'
FIXTURE = SIM / 'sim_fidelity' / 'sfu_course_2026_10_30_stops.json'
# All six other driving bags on this date, outside the original floor split.
EXTRA = tuple('20261030-' + s for s in ('181222', '182716', '182852', '184010', '184149', '185653'))


def crossing(t, v, threshold):
    """Interpolated time of the first descending crossing (starts above the threshold)."""
    hits = np.flatnonzero(v <= threshold)
    if not hits.size or hits[0] == 0:
        raise ValueError(f'no downward crossing of {threshold} m/s')
    i = hits[0]
    return t[i - 1] + (t[i] - t[i - 1]) * (v[i - 1] - threshold) / (v[i - 1] - v[i])


def measures(t, wheel):
    end = crossing(t, wheel, 0.3)
    tail = end - crossing(t, wheel, 1.0)
    distance = np.trapz(np.r_[wheel[t < end], 0.3], np.r_[t[t < end], end])
    return dict(stop_s=end, tail_s=tail, wheel_m=distance)


def extract(run, delay):
    from rosbags.highlevel import AnyReader
    command, odom = [], []
    with AnyReader([run / 'bag']) as reader:
        topics = {'/ackermann_cmd', '/odom'}
        for conn, ns, raw in reader.messages(connections=[c for c in reader.connections if c.topic in topics]):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == '/odom':
                odom.append((ns / 1e9, msg.twist.twist.linear.x))
            else:
                command.append((ns / 1e9, msg.drive.speed))
    if not command or not odom:
        return []
    ct, cv = np.asarray(command).T
    ot, ov = np.asarray(odom).T
    stops = []
    for i in np.flatnonzero((cv[:-1] > 0) & (cv[1:] == 0)) + 1:
        start = ct[i]
        history_t = start + CONTROL_DT * np.arange(-delay, 0)
        if history_t[0] < ct[0] or not ot[0] <= start < ot[-1]:
            continue
        # Use the mux output, not the controller's requested speed. Discard a
        # blip, a renewed command, or missing telemetry before the stop ends.
        nonzero = np.flatnonzero(cv[i:] != 0)
        until = min(start + 3, ct[i + nonzero[0]] if nonzero.size else ct[-1])
        v0 = np.interp(start, ot, ov)
        if v0 <= 2 or until - start < 0.6:
            continue
        window = (ot > start) & (ot < until)
        t = np.r_[0.0, ot[window] - start]
        v = np.r_[v0, ov[window]]
        if not np.all(np.isfinite(v)) or v.min() > 0.3:
            continue
        obs = measures(t, v)
        end = obs['stop_s']
        if end < 0.6:
            continue
        # Include the bracketing samples at either end of the interval.
        cmd_stamps = ct[np.searchsorted(ct, history_t[0], 'right')-1:np.searchsorted(ct, start+end)+1]
        odom_stamps = ot[np.searchsorted(ot, start, 'right')-1:np.searchsorted(ot, start+end)+1]
        if np.max(np.diff(cmd_stamps)) > 0.1 or np.max(np.diff(odom_stamps)) > 0.1:
            continue
        prior = cv[np.searchsorted(ct, history_t, 'right') - 1]
        stops.append(dict(start_s=start, wheel_speed=v0, prior_cmd=prior.tolist(), **obs))
    return stops


def predict(stop, cal):
    # A straight open-loop replay: neither future wheel speeds nor the
    # controller enter the simulation. Compare wheel units on both sides.
    v0 = stop['wheel_speed']
    speed, _ = simulate_commands(np.zeros((120, 2)), cal, initial_speed=v0 / cal.odom_speed_scale,
                                 prior_speed_commands=stop['prior_cmd'])
    return measures(CONTROL_DT * np.arange(len(speed) + 1), np.r_[v0, speed * cal.odom_speed_scale])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, help='read the original bags instead of the fixture')
    parser.add_argument('--save', type=Path, help='write the extracted measurements to this JSON file')
    args = parser.parse_args()
    floor = FLOORS[FLOOR]
    base = calibration_for_car('rb2')
    cal = with_floor(STOP_FLOOR, base)
    groups = {'fit': floor.fit_runs, 'held-out': floor.held_out_runs, 'additional': EXTRA}
    if args.runs:
        data = {run: extract(args.runs / run, cal.speed_delay_ticks) for runs in groups.values() for run in runs}
        if args.save:
            args.save.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    else:
        if args.save:
            parser.error('--save requires --runs')
        data = json.loads(FIXTURE.read_text())
    for name, runs in groups.items():
        stops = [s for r in runs for s in data[r]]
        print(f'{name}: {len(stops)} stops in {sum(bool(data[r]) for r in runs)}/{len(runs)} bags')
        if not stops:
            raise ValueError(f'no qualifying stops in the {name} recordings')
        if name == 'fit':
            print(f"  median stop gain: {np.median([np.log(1 / 0.3) / s['tail_s'] for s in stops]):.3f}/s")
        for label, model in [('before', with_floor(FLOOR, base)), ('after', cal)]:
            predictions = [predict(s, model) for s in stops]
            errors = {k: np.sqrt(np.mean([(p[k] - s[k])**2 for p, s in zip(predictions, stops)]))
                      for k in ('stop_s', 'tail_s', 'wheel_m')}
            print('  {} RMSE: stop {:.4f} s, tail {:.4f} s, wheel distance {:.4f} m'.format(label, *errors.values()))


if __name__ == '__main__':
    main()
