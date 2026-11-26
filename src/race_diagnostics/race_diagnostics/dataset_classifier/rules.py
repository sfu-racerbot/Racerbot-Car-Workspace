"""
rules.py -- load config/classification_rules.yaml, check it, and apply its thresholds.

A rules file is refused, with the key that is wrong, rather than half-used:
an unknown key or a misspelt signal would otherwise grade every recording
against a default nobody chose.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import yaml

from .algorithms import ROLES

STATUSES = ('usable', 'conditional', 'unusable')
RANK = {status: rank for rank, status in enumerate(STATUSES)}

KINDS = ('scan', 'odometry', 'transform', 'imu', 'vesc_state', 'ackermann',
         'joint_state', 'joy', 'drive_intent', 'pose')
# Signal checks: the stream statistic each compares, and whether more is better.
CHECKS = {
    'min_rate_hz': ('rate_hz', 'at_least'),
    'max_gap_s': ('max_gap_s', 'at_most'),
    'min_valid_fraction': ('valid_fraction', 'at_least'),
    'min_coverage': ('coverage', 'at_least'),
    'max_stamp_problem_fraction': ('stamp_problem_fraction', 'at_most'),
    'min_count': ('count', 'at_least'),
}
MEASUREMENT_KEYS = {
    'moving_speed_mps', 'measured_steering_min_speed_mps', 'steering_swing_rad',
    'speed_swing_mps', 'cornering_accel_mps2', 'stamp_tolerance_s', 'coverage_gap_s',
    'max_integration_gap_s', 'command_match_s', 'carried_still_mps', 'carried_moved_m',
    'lb_min_hold_s', 'valid'}
VALID_KEYS = {'scan_min_beam_fraction', 'max_speed_mps', 'max_steering_rad',
              'max_accel_mps2', 'max_gyro_radps'}
TOPIC_KEYS = {'gyro_unit': ('rad/s', 'deg/s'), 'accel_unit': ('m/s^2', 'g'),
              'axes': ('car', 'board', 'sensor'), 'format': ('json',), 'version': None}
FLAGS = ('no_bag', 'bag_not_closed', 'read_failed', 'read_incomplete', 'superseded',
         'recorder_killed', 'bag_recorder_failed', 'other_recorder_failed', 'started_late')
# Everything classifier.measure() puts under `metrics`.
METRICS = ('duration_s', 'distance_m', 'moving_s', 'max_speed_mps', 'speed_swings',
           'lb_held_s', 'lb_holds', 'carried_between_holds', 'steering_std_rad',
           'steering_reversals', 'measured_steering_finite_fraction', 'lateral_accel_p95_mps2',
           'cornering_s', 'measured_lateral_accel_p95_mps2', 'autonomous_s',
           'autonomous_distance_m', 'algorithm_known')
TOP_KEYS = {'schema', 'measurement', 'topics', 'signals', 'recording_flags',
            'read_min_fraction', 'algorithms', 'applications'}
ALGORITHM_KEYS = ('repo', 'node', 'parameter', 'role')
NAME = re.compile(r'^[A-Za-z0-9_]+$')


class RulesError(ValueError):
    """The rules file is not one the classifier can apply."""


@dataclass(frozen=True)
class Rules:
    path: str
    sha256: str
    data: dict

    def __getitem__(self, key):
        return self.data[key]

    def topic_format(self, topic):
        """Units and axes of a topic: SI in car axes unless the rules say otherwise."""
        out = {'gyro_unit': 'rad/s', 'accel_unit': 'm/s^2', 'axes': 'car'}
        out.update(self.data.get('topics', {}).get(topic, {}))
        return out


def default_rules_path():
    """config/classification_rules.yaml in the source tree, else in the installed share."""
    source = Path(__file__).resolve().parents[2] / 'config' / 'classification_rules.yaml'
    if source.is_file():
        return source
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory('race_diagnostics')) / 'config' / 'classification_rules.yaml'


def load_rules(path=None):
    """Read and check a rules file. Raises RulesError naming what is wrong."""
    path = Path(path) if path is not None else default_rules_path()
    raw = path.read_bytes()
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise RulesError(f'{path}: not YAML: {e}') from e
    check_rules(data)
    return Rules(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), data=data)


def _where(*parts):
    return '.'.join(str(p) for p in parts)


def _mapping(value, where):
    if not isinstance(value, dict):
        raise RulesError(f'{where}: expected a mapping, got {type(value).__name__}')
    return value


def _unknown(keys, allowed, where):
    extra = sorted(set(keys) - set(allowed))
    if extra:
        raise RulesError(f'{where}: unknown key(s) {", ".join(map(str, extra))}')


def _missing(keys, required, where):
    absent = sorted(set(required) - set(keys))
    if absent:
        raise RulesError(f'{where}: missing {", ".join(absent)}')


def _number(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RulesError(f'{where}: expected a number, got {value!r}')
    return value


def _levels(value, direction, where):
    """A {usable, conditional} threshold pair; usable must be the stricter."""
    _mapping(value, where)
    _unknown(value, ('usable', 'conditional'), where)
    if 'usable' not in value:
        raise RulesError(f'{where}: needs a usable level')
    usable = _number(value['usable'], _where(where, 'usable'))
    if 'conditional' in value:
        conditional = _number(value['conditional'], _where(where, 'conditional'))
        weaker = conditional > usable if direction == 'at_least' else conditional < usable
        if weaker:
            raise RulesError(f'{where}: the conditional level {conditional} is stricter than '
                             f'the usable level {usable}')


def _status(value, where, allowed=('conditional', 'unusable')):
    if value not in allowed:
        raise RulesError(f'{where}: {value!r} is not one of {", ".join(allowed)}')


def check_rules(data):
    """Raise RulesError if data is not a complete, consistent rules file."""
    _mapping(data, 'rules')
    _unknown(data, TOP_KEYS, 'rules')
    _missing(data, TOP_KEYS - {'topics'}, 'rules')
    if data['schema'] != 1:
        raise RulesError(f'schema: this classifier reads schema 1, not {data["schema"]!r}')

    measurement = _mapping(data['measurement'], 'measurement')
    _unknown(measurement, MEASUREMENT_KEYS, 'measurement')
    _missing(measurement, MEASUREMENT_KEYS, 'measurement')
    valid = _mapping(measurement['valid'], 'measurement.valid')
    _unknown(valid, VALID_KEYS, 'measurement.valid')
    _missing(valid, VALID_KEYS, 'measurement.valid')
    for key, value in measurement.items():
        if key != 'valid' and _number(value, _where('measurement', key)) < 0:
            raise RulesError(f'measurement.{key}: must not be negative')
    for key, value in valid.items():
        if _number(value, _where('measurement.valid', key)) <= 0:
            raise RulesError(f'measurement.valid.{key}: must be positive')

    for topic, fmt in _mapping(data.get('topics') or {}, 'topics').items():
        _mapping(fmt, _where('topics', topic))
        _unknown(fmt, TOPIC_KEYS, _where('topics', topic))
        for key, value in fmt.items():
            if TOPIC_KEYS[key] is not None and value not in TOPIC_KEYS[key]:
                raise RulesError(f'topics.{topic}.{key}: {value!r} is not one of '
                                 f'{", ".join(TOPIC_KEYS[key])}')

    signals = _mapping(data['signals'], 'signals')
    for name, signal in signals.items():
        where = _where('signals', name)
        _mapping(signal, where)
        _unknown(signal, ('topics', 'kind', 'checks', 'transform', 'latched'), where)
        topics = signal.get('topics')
        if (not isinstance(topics, list) or not topics
                or not all(isinstance(t, str) and t.startswith('/') for t in topics)):
            raise RulesError(f'{where}.topics: expected a list of topic names')
        if signal.get('kind') not in KINDS:
            raise RulesError(f'{where}.kind: {signal.get("kind")!r} is not one of {", ".join(KINDS)}')
        transform = signal.get('transform')
        if (signal['kind'] == 'transform') != (transform is not None):
            raise RulesError(f'{where}: a transform signal, and only one, names a transform')
        if transform is not None and (not isinstance(transform, list) or len(transform) != 2
                                      or not all(isinstance(f, str) and f for f in transform)):
            raise RulesError(f'{where}.transform: expected [parent, child]')
        for check, levels in _mapping(signal.get('checks') or {}, _where(where, 'checks')).items():
            if check not in CHECKS:
                raise RulesError(f'{where}.checks: unknown check {check}')
            _levels(levels, CHECKS[check][1], _where(where, 'checks', check))

    for flag, caps in _mapping(data['recording_flags'], 'recording_flags').items():
        if flag not in FLAGS:
            raise RulesError(f'recording_flags: unknown flag {flag}')
        for app, status in _mapping(caps or {}, _where('recording_flags', flag)).items():
            if app != 'all' and app not in (data.get('applications') or {}):
                raise RulesError(f'recording_flags.{flag}: no application {app}')
            _status(status, _where('recording_flags', flag, app))
    _missing(data['recording_flags'], FLAGS, 'recording_flags')
    fraction = _number(data['read_min_fraction'], 'read_min_fraction')
    if not 0 <= fraction <= 1:
        raise RulesError('read_min_fraction: must be between 0 and 1')

    _check_algorithms(_mapping(data['algorithms'], 'algorithms'))

    applications = _mapping(data['applications'], 'applications')
    if not applications:
        raise RulesError('applications: none defined')
    used = set()
    for app, spec in applications.items():
        where = _where('applications', app)
        _mapping(spec, where)
        _unknown(spec, ('description', 'required', 'references', 'criteria'), where)
        for part in ('required', 'references'):
            for signal, options in _mapping(spec.get(part) or {}, _where(where, part)).items():
                if signal not in signals:
                    raise RulesError(f'{where}.{part}: no signal {signal}')
                used.add(signal)
                options = _mapping(options or {}, _where(where, part, signal))
                _unknown(options, ('if_missing',), _where(where, part, signal))
                if 'if_missing' in options:
                    _status(options['if_missing'], _where(where, part, signal, 'if_missing'))
        for metric, criterion in _mapping(spec.get('criteria') or {}, _where(where, 'criteria')).items():
            cwhere = _where(where, 'criteria', metric)
            if metric not in METRICS:
                raise RulesError(f'{where}.criteria: unknown metric {metric}')
            _mapping(criterion, cwhere)
            _unknown(criterion, ('at_least', 'at_most', 'if_unknown'), cwhere)
            directions = [d for d in ('at_least', 'at_most') if d in criterion]
            if len(directions) != 1:
                raise RulesError(f'{cwhere}: needs exactly one of at_least, at_most')
            _levels(criterion[directions[0]], directions[0], _where(cwhere, directions[0]))
            if 'if_unknown' in criterion:
                _status(criterion['if_unknown'], _where(cwhere, 'if_unknown'),
                        ('skip', 'conditional', 'unusable'))
    unused = sorted(set(signals) - used)
    if unused:
        raise RulesError(f'signals: {", ".join(unused)} not used by any application')


def _check_algorithms(algorithms):
    """Each entry is <package>/<executable> with a repo and the node it starts.
    Entries that start the same node each need a parameter of their own, or a
    run that names only the node could never be told apart."""
    if not algorithms:
        raise RulesError('algorithms: none listed')
    by_node = {}
    for code, entry in algorithms.items():
        parts = code.split('/') if isinstance(code, str) else []
        if len(parts) != 2 or not all(NAME.match(part) for part in parts):
            raise RulesError(f'algorithms: {code!r} is not <package>/<executable>')
        where = _where('algorithms', code)
        _mapping(entry, where)
        _unknown(entry, ALGORITHM_KEYS, where)
        _missing(entry, ('repo', 'node'), where)
        for key in ('repo', 'node', 'parameter'):
            value = entry.get(key, '')
            # A repository name may hold a hyphen (racerbot-ws); a ROS name may not.
            if key in entry and not (isinstance(value, str)
                                     and NAME.match(value.replace('-', '_') if key == 'repo' else value)):
                raise RulesError(f'{where}.{key}: expected a name, got {value!r}')
        if 'role' in entry:
            _status(entry['role'], _where(where, 'role'), ROLES)
        by_node.setdefault(entry['node'], []).append(code)
    for node, codes in sorted(by_node.items()):
        parameters = [algorithms[code].get('parameter') for code in codes]
        if len(codes) > 1 and (None in parameters or len(set(parameters)) < len(codes)):
            raise RulesError(f'algorithms: {", ".join(sorted(codes))} all start {node}: '
                             f'each needs a parameter of its own to tell them apart')


def worst(statuses):
    """The worst of some statuses; usable if there are none."""
    return max(statuses, key=RANK.__getitem__, default='usable')


def level(value, levels, direction):
    """usable / conditional / unusable for a value against {usable, conditional} levels.

    A missing conditional level means the value is never worse than conditional."""
    def meets(threshold):
        return value >= threshold if direction == 'at_least' else value <= threshold
    if meets(levels['usable']):
        return 'usable'
    if 'conditional' not in levels or meets(levels['conditional']):
        return 'conditional'
    return 'unusable'
