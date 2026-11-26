"""
classifier.py -- grade one recorded run, or all of them.

Two steps, kept apart so a grade can be checked without the bag:

  measure(run_dir, rules)  reads the run (bag, auto_record.yaml, metadata/)
                           and returns plain numbers, rounded as stored.
  grade(measured, rules)   turns those numbers into usable / conditional /
                           unusable per use, with the reasons. No I/O.

classify_run() does both, writes <run>/qualification.yaml and updates
data/dataset_inventory.csv. It never raises for a bad recording and never
changes anything else in the run: a run it cannot read is graded unusable
with the error as the reason.
"""

from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path

import numpy as np

from race_diagnostics.run_activity import summarize

from . import metrics as mx
from .bag_stats import JOY, ODOM, SCAN, bag_metadata, read_streams, stream_stats, transform_key
from .rules import CHECKS, RANK, level, load_rules, worst
from .run_info import read_run_info, UNKNOWN

VERSION = 1
QUALIFICATION = 'qualification.yaml'
DIGITS = 4


def _round(value):
    """Plain Python numbers, rounded, so the file is the same on every machine."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return round(value, DIGITS) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return value


def run_location(run_dir):
    """(data root, car folder) for data/<car>/racerbot_runs/<run>, else (None, None)."""
    run_dir = Path(run_dir).resolve()
    if run_dir.parent.name != 'racerbot_runs':
        return None, None
    return run_dir.parent.parent.parent, run_dir.parent.parent.name


# --------------------------------------------------------------------------- measure

def _recording_facts(run_dir, bag_info):
    import yaml
    facts = {'bag': (run_dir / 'bag').is_dir(), 'bag_closed': bag_info is not None}
    if bag_info:
        facts['bag_files'] = len(bag_info.get('relative_file_paths') or [])
        facts['bag_messages'] = int(bag_info.get('message_count') or 0)
        facts['duration_s'] = bag_info.get('duration', {}).get('nanoseconds', 0) / 1e9
    facts['bag_bytes'] = sum(p.stat().st_size for p in (run_dir / 'bag').glob('*') if p.is_file()) \
        if facts['bag'] else 0
    try:
        auto = yaml.safe_load((run_dir / 'auto_record.yaml').read_text()) or {}
    except (OSError, yaml.YAMLError):
        auto = {}
    facts['auto_record'] = bool(auto)
    if auto:
        facts['superseded'] = bool(auto.get('superseded', False))
        facts['recorder_killed'] = auto.get('recorder_killed')
        facts['bag_exit_code'] = auto.get('bag_exit_code')
        facts['recorder_failures'] = [
            f"{f.get('process')} exited {f.get('exit_code')}"
            for f in auto.get('recorder_failures') or [] if isinstance(f, dict)]
        facts['stop_reason'] = auto.get('stop_reason')
    return facts


def _choose(signal, streams, rules):
    """The stream that fills a signal: the first candidate with two valid messages."""
    keys = ([transform_key(t, *signal['transform']) for t in signal['topics']]
            if signal['kind'] == 'transform' else list(signal['topics']))
    present = [k for k in keys if k in streams and len(streams[k].t)]
    for key in present:
        if int(np.sum(streams[key].mask())) >= (1 if signal.get('latched') else 2):
            return key, [k for k in present if k != key]
    return None, present


def _imu_arrays(stream):
    mask = stream.mask()
    t = stream.times()
    gyro = np.column_stack([stream.column(k) for k in ('wx', 'wy', 'wz')]) if mask.any() else np.zeros((0, 3))
    accel = np.column_stack([stream.column(k) for k in ('ax', 'ay', 'az')]) if mask.any() else np.zeros((0, 3))
    return t, gyro, accel


def _metrics(streams, sources, carry, facts, info, rules):
    m = rules['measurement']
    out = dict.fromkeys(('distance_m', 'moving_s', 'max_speed_mps', 'speed_swings', 'lb_held_s',
                         'lb_holds', 'carried_between_holds', 'steering_std_rad',
                         'steering_reversals', 'measured_steering_finite_fraction',
                         'lateral_accel_p95_mps2', 'cornering_s',
                         'measured_lateral_accel_p95_mps2', 'autonomous_s',
                         'autonomous_distance_m'))
    out['duration_s'] = facts.get('duration_s')
    names = info.get('algorithm_nodes')
    out['algorithm_known'] = int(isinstance(names, list) and bool(mx.driving_algorithms(names)))

    odom = streams.get(sources.get('wheel_odom') or '')
    odom_t = odom.times() if odom else np.zeros(0)
    odom_v = odom.column('v') if odom else np.zeros(0)
    if len(odom_t):
        out['distance_m'], out['moving_s'], out['max_speed_mps'] = mx.motion(
            odom_t, odom_v, m['moving_speed_mps'], m['max_integration_gap_s'])
        out['speed_swings'] = mx.reversals(odom_v, m['speed_swing_mps'])

    joy = streams.get(JOY)
    windows = None
    if joy is not None and joy.kind == 'joy' and joy.mask().any():
        windows = mx.lb_windows(joy.times(), joy.column('lb'), m['lb_min_hold_s'])
        out['lb_held_s'] = sum(end - start for start, end in windows)
        out['lb_holds'] = len(windows)
        scan_t, scans = carry.finish()
        scan = streams.get(SCAN)
        if scan is not None and len(scan.t):
            raw_odom = streams.get(ODOM)
            rows = (np.column_stack([raw_odom.times(), raw_odom.column('v')])
                    if raw_odom is not None and raw_odom.kind == 'odometry' else np.zeros((0, 2)))
            out['carried_between_holds'] = sum(mx.moved_between(
                scan_t, scans, rows, windows, m['carried_still_mps'], m['carried_moved_m']))

    steering = streams.get(sources.get('steering_command') or '')
    if steering is not None:
        out['steering_std_rad'], out['steering_reversals'] = mx.steering_excitation(
            steering.times(), steering.column('steering'), odom_t, odom_v,
            m['moving_speed_mps'], m['steering_swing_rad'])

    measured = streams.get(sources.get('measured_steering') or '')
    if measured is not None:
        out['measured_steering_finite_fraction'] = mx.finite_fraction_while_fast(
            measured.times(), measured.column('position'), odom_t, odom_v,
            m['measured_steering_min_speed_mps'])

    imu = streams.get(sources.get('imu') or '')
    if imu is not None:
        t, gyro, accel = _imu_arrays(imu)
        yaw = mx.yaw_rate(gyro, accel, rules.topic_format(imu.topic)['axes'])
        out['lateral_accel_p95_mps2'], out['cornering_s'] = mx.lateral_acceleration(
            t, yaw, odom_t, odom_v, m['moving_speed_mps'], m['cornering_accel_mps2'],
            m['max_integration_gap_s'])

    car_imu = streams.get(sources.get('car_imu') or '')
    if car_imu is not None:
        out['measured_lateral_accel_p95_mps2'] = mx.measured_lateral_p95(
            car_imu.times(), car_imu.column('ay'), odom_t, odom_v, m['moving_speed_mps'])

    command = streams.get(sources.get('drive_command') or '')
    if windows is not None and len(odom_t):
        out['autonomous_s'], out['autonomous_distance_m'] = mx.autonomous(
            odom_t, odom_v, windows, command.times() if command is not None else [],
            m['command_match_s'], m['moving_speed_mps'], m['max_integration_gap_s'])
    return out


def _activity(streams):
    """run_activity's own summary of the run, from the same /odom and /joy."""
    odom, joy = streams.get(ODOM), streams.get(JOY)
    speeds = odom.column('v', valid_only=False) if odom is not None and odom.kind == 'odometry' else []
    states = joy.column('lb', valid_only=False) if joy is not None and joy.kind == 'joy' else []
    return summarize([float(v) for v in speeds], [int(s) for s in states])


def measure(run_dir, rules, backend='auto'):
    """Everything grade() needs, read from the run. Raises only if the bag
    cannot be opened at all."""
    run_dir = Path(run_dir)
    bag_dir = run_dir / 'bag'
    bag_info = bag_metadata(bag_dir) if bag_dir.is_dir() else None
    facts = _recording_facts(run_dir, bag_info)
    info = read_run_info(run_dir, bag_info)
    measured = {'run_info': info, 'recording': facts, 'streams': {}, 'sources': {},
                'alternatives': {}, 'metrics': {}, 'activity': None}
    if not facts['bag'] or not facts['bag_closed']:
        measured['metrics'] = _round(_metrics({}, {}, None, facts, info, rules))
        return measured
    streams, carry, report = read_streams(bag_dir, rules, backend)
    facts['messages_read'] = report['messages_read']
    facts['read_error'] = report['read_error']
    start = bag_info['starting_time']['nanoseconds_since_epoch'] / 1e9
    end = start + facts['duration_s']
    for name, signal in rules['signals'].items():
        key, others = _choose(signal, streams, rules)
        measured['sources'][name] = key
        measured['alternatives'][name] = others
    latched = {k for name, s in rules['signals'].items() if s.get('latched')
               for k in [measured['sources'][name]] + measured['alternatives'][name]}
    for key, stream in sorted(streams.items()):
        measured['streams'][key] = _round(stream_stats(stream, start, end, rules, key in latched))
    measured['metrics'] = _round(_metrics(streams, measured['sources'], carry, facts, info, rules))
    measured['activity'] = _activity(streams)
    return measured


# --------------------------------------------------------------------------- grade

def _fmt(value):
    if isinstance(value, float):
        return f'{value:g}'
    return str(value)


def _check(name, value, levels, direction, unit=''):
    """(status, reason or None) for one value against its levels."""
    status = level(value, levels, direction)
    if status == 'usable':
        return status, None
    need = levels['usable']
    word = 'at least' if direction == 'at_least' else 'at most'
    return status, f'{name} {_fmt(value)}{unit}, usable needs {word} {_fmt(need)}{unit}'


UNITS = {'rate_hz': ' Hz', 'max_gap_s': ' s', 'valid_fraction': '', 'coverage': '',
         'stamp_problem_fraction': '', 'count': ''}


def grade_signal(name, signal, key, stats):
    """(status, reasons) for the stream filling one signal; 'missing' if none."""
    if key is None or stats is None:
        return 'missing', [f'{name}: none of {", ".join(signal["topics"])} in the bag'
                           + (f' with {signal["transform"][0]} -> {signal["transform"][1]}'
                              if signal['kind'] == 'transform' else '')]
    statuses, reasons = [], []
    for check, levels in (signal.get('checks') or {}).items():
        stat, direction = CHECKS[check]
        value = stats.get(stat)
        if value is None:
            if stat == 'stamp_problem_fraction':
                continue  # nothing stamped to check
            statuses.append('unusable')
            reasons.append(f'{name} ({key}): no {stat.replace("_", " ")} '
                           f'from {stats.get("count", 0)} message(s)')
            continue
        status, reason = _check(f'{name} ({key}) {stat.replace("_", " ")}', value, levels,
                                direction, UNITS[stat])
        statuses.append(status)
        if reason:
            reasons.append(reason)
    return worst(statuses), reasons


def _recording_flags(facts, activity, read_min_fraction):
    flags = []
    if not facts.get('bag'):
        return ['no_bag']
    if not facts.get('bag_closed'):
        return ['bag_not_closed']
    if facts.get('read_error'):
        total = facts.get('bag_messages') or 0
        fraction = facts.get('messages_read', 0) / total if total else 0.0
        flags.append('read_incomplete' if fraction >= read_min_fraction else 'read_failed')
    if facts.get('superseded'):
        flags.append('superseded')
    if facts.get('recorder_killed'):
        flags.append('recorder_killed')
    if facts.get('bag_exit_code') not in (None, 0):
        flags.append('bag_recorder_failed')
    failures = facts.get('recorder_failures') or []
    if any(f.startswith('ros2 bag record') for f in failures):
        if 'bag_recorder_failed' not in flags:
            flags.append('bag_recorder_failed')
    if any(not f.startswith('ros2 bag record') for f in failures):
        flags.append('other_recorder_failed')
    if (activity or {}).get('started_late'):
        flags.append('started_late')
    return flags


FLAG_TEXT = {
    'no_bag': 'the run folder holds no bag/',
    'bag_not_closed': 'bag/metadata.yaml is missing, so the bag was never closed: '
                      'run `ros2 bag reindex <run>/bag`, then regrade',
    'read_failed': 'the bag stopped reading early',
    'read_incomplete': 'the bag stopped reading near its end',
    'superseded': 'a stub that another recorder replaced at once (auto_record.yaml)',
    'recorder_killed': 'the recorder was killed, not stopped (auto_record.yaml)',
    'bag_recorder_failed': '`ros2 bag record` failed (auto_record.yaml)',
    'other_recorder_failed': 'a recorder process other than the bag died (the bag is unaffected)',
    'started_late': 'the car was already moving when the bag started',
}


def grade(measured, rules):
    """Applications, signals and recording status from measure()'s output."""
    facts, metrics_ = measured['recording'], measured['metrics']
    flags = _recording_flags(facts, measured.get('activity'), rules['read_min_fraction'])
    flag_caps = rules['recording_flags']
    signals = {}
    for name, signal in rules['signals'].items():
        key = measured['sources'].get(name)
        status, reasons = grade_signal(name, signal, key, measured['streams'].get(key))
        signals[name] = {'topic': key, 'status': status, 'reasons': reasons,
                         'alternatives': measured['alternatives'].get(name, [])}
    recording_status = worst(flag_caps[f].get('all') for f in flags
                             if (flag_caps[f] or {}).get('all'))
    applications = {}
    for app, spec in rules['applications'].items():
        found = []  # (status, reason)
        for flag in flags:
            cap = (flag_caps[flag] or {}).get(app) or (flag_caps[flag] or {}).get('all')
            if cap:
                detail = FLAG_TEXT[flag]
                if flag in ('read_failed', 'read_incomplete'):
                    detail += f" ({facts.get('messages_read')} of {facts.get('bag_messages')} " \
                              f"messages: {facts.get('read_error')})"
                found.append((cap, f'recording: {detail}'))
        for part, default in (('required', 'unusable'), ('references', 'conditional')):
            for name, options in (spec.get(part) or {}).items():
                graded = signals[name]
                if_missing = (options or {}).get('if_missing', default)
                if graded['status'] in ('missing', 'unusable'):
                    what = 'missing' if graded['status'] == 'missing' else 'unusable'
                    role = 'required' if part == 'required' else 'reference'
                    found.append((if_missing, f'{role} {name} {what}: ' + '; '.join(graded['reasons'])))
                elif graded['status'] == 'conditional':
                    found.append(('conditional', '; '.join(graded['reasons'])))
        for metric, criterion in (spec.get('criteria') or {}).items():
            direction = 'at_least' if 'at_least' in criterion else 'at_most'
            value = metrics_.get(metric)
            if value is None:
                if_unknown = criterion.get('if_unknown', 'conditional')
                if if_unknown != 'skip':
                    found.append((if_unknown, f'{metric} could not be measured'))
                continue
            status, reason = _check(metric, value, criterion[direction], direction)
            if reason:
                found.append((status, reason))
        status = worst(s for s, _ in found)
        if status == 'usable':
            shown = ', '.join(f'{k} {_fmt(metrics_.get(k))}' for k in (spec.get('criteria') or {})
                              if metrics_.get(k) is not None)
            explanation = f'usable: every required signal, reference and threshold met ({shown})'
        else:
            explanation = f'{status}: ' + '; '.join(r for s, r in found if s == status)
        applications[app] = {
            'status': status, 'explanation': explanation,
            'reasons': [f'{s}: {r}' for s, r in sorted(found, key=lambda f: -RANK[f[0]])]}
    return {
        'recording': {'status': recording_status, 'flags': flags},
        'signals': signals, 'applications': applications,
    }


# --------------------------------------------------------------------------- one run

def failed(rules, error):
    """The qualification of a run the classifier could not measure."""
    reason = f'classification failed: {error}'
    return {
        'recording': {'status': 'unusable', 'flags': []},
        'signals': {},
        'applications': {app: {'status': 'unusable', 'explanation': f'unusable: {reason}',
                               'reasons': [f'unusable: {reason}']}
                         for app in rules['applications']},
    }


def qualification(run_dir, rules, backend='auto'):
    """The full qualification.yaml content for one run. Never raises for a bad run."""
    run_dir = Path(run_dir)
    _, car = run_location(run_dir)
    error = None
    try:
        measured = measure(run_dir, rules, backend)
        graded = grade(measured, rules)
    except Exception as e:  # noqa: BLE001 -- recorded in the file, never lost
        error = f'{type(e).__name__}: {e}'
        measured = None
        graded = failed(rules, error)
    doc = {
        'schema': 1,
        'run_id': run_dir.name,
        'car': car or UNKNOWN,
        'classifier': {'version': VERSION, 'rules': Path(rules.path).name,
                       'rules_sha256': rules.sha256},
        'summary': {app: a['status'] for app, a in graded['applications'].items()},
        'applications': graded['applications'],
        'recording': dict(graded['recording'], **(measured['recording'] if measured else {})),
        'run_info': measured['run_info'] if measured else None,
        'signals': graded['signals'],
        'metrics': measured['metrics'] if measured else None,
        'activity': measured['activity'] if measured else None,
        'streams': measured['streams'] if measured else None,
        'error': error,
    }
    return _round(doc)


def regrade(doc, rules):
    """grade() again from a stored qualification.yaml, without the bag."""
    recording = {k: v for k, v in doc['recording'].items() if k not in ('status', 'flags')}
    measured = {
        'recording': recording, 'streams': doc['streams'] or {}, 'metrics': doc['metrics'] or {},
        'activity': doc.get('activity'), 'run_info': doc.get('run_info'),
        'sources': {name: s.get('topic') for name, s in (doc['signals'] or {}).items()},
        'alternatives': {name: s.get('alternatives', []) for name, s in (doc['signals'] or {}).items()},
    }
    return _round(grade(measured, rules))


def classify_run(run_dir, rules=None, *, inventory=True, dry_run=False, backend='auto'):
    """Grade one run: write <run>/qualification.yaml (only if it changed) and
    refresh this car's rows of data/dataset_inventory.csv. Returns the content."""
    from .outputs import update_inventory, write_qualification
    rules = rules or load_rules()
    doc = qualification(run_dir, rules, backend)
    if not dry_run:
        write_qualification(run_dir, doc)
        data_root, car = run_location(run_dir)
        if inventory and data_root is not None:
            update_inventory(data_root, rules, cars={car})
    return doc


def find_runs(data_root):
    """Every run folder under data/<car>/racerbot_runs/, sorted by car then time."""
    return sorted(p for p in Path(data_root).glob('*/racerbot_runs/*') if p.is_dir())


def _classify_one(args):
    run_dir, rules_path, dry_run, backend = args
    rules = load_rules(rules_path)
    doc = classify_run(run_dir, rules, inventory=False, dry_run=dry_run, backend=backend)
    return str(run_dir), doc


def classify_all(data_root, rules=None, *, runs=None, jobs=1, dry_run=False, backend='auto'):
    """Grade every run (or the given ones), then rebuild the inventory once.
    Yields (run_dir, qualification) as each finishes."""
    from .outputs import rebuild_inventory
    rules = rules or load_rules()
    runs = list(runs) if runs is not None else find_runs(data_root)
    work = [(str(r), rules.path, dry_run, backend) for r in runs]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            yield from pool.map(_classify_one, work)
    else:
        for item in work:
            yield _classify_one(item)
    if not dry_run and data_root is not None:
        rebuild_inventory(data_root, rules)
