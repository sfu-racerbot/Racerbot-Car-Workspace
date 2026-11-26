"""
algorithms.py -- which code drove a run: its package, executable and repository.

A node name alone does not say whose code it is: this workspace's gap_follow
package and racerbot_a's reactive package both start a node called
gap_follow_node. Each node a run names is matched to one entry of the
`algorithms:` list in config/classification_rules.yaml, from the run's own
files and never by guessing, in this order:

  1. metadata/processes.txt   the process's install path (.../lib/<package>/<executable>)
                              and its __node:= name: exact
  2. the node name            when only one entry starts a node of that name
  3. metadata/params/<node>.yaml
                              the parameters the node reported (snapshot_params),
                              when exactly one entry with that node name lists a
                              `parameter` among them

Anything else is named unknown/<node>, with the reason. No ROS and no bag here.
"""

from pathlib import Path
import re

import yaml

UNKNOWN = 'unknown'
DRIVING = 'driving'
# driving: an algorithm to judge. mapping (SLAM) and manual (the gamepad layer)
# are named, but never drive the car by themselves.
ROLES = (DRIVING, 'mapping', 'manual')
# '.../lib/<package>/<executable>' and the rest of that process's command line.
_PROCESS = re.compile(r'/lib/(?P<package>[A-Za-z0-9_]+)/(?P<executable>[A-Za-z0-9_]+)'
                      r'(?![A-Za-z0-9_./])(?P<rest>[^\n]*)')
_NODE = re.compile(r'__node:=(?P<node>[A-Za-z0-9_]+)')


def role(entry):
    return entry.get('role', DRIVING)


def from_processes(text, known):
    """[(node name, code)] for every process in processes.txt that runs a listed
    executable, sorted, each once. A process started without __node:= runs
    under its entry's own node name."""
    found = set()
    for match in _PROCESS.finditer(text or ''):
        code = f"{match['package']}/{match['executable']}"
        if code in known:
            remap = _NODE.search(match['rest'])
            found.add((remap['node'] if remap else known[code]['node'], code))
    return sorted(found)


def listed_nodes(known):
    """Every node name an entry starts."""
    return {entry['node'] for entry in known.values()}


def reported_parameters(path):
    """The parameter names in a snapshot_params file (/node: ros__parameters: {...}),
    or None if there is no such file or it is not one."""
    try:
        doc = yaml.safe_load(Path(path).read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    if not isinstance(doc, dict):
        return None
    names = set()
    for value in doc.values():
        if isinstance(value, dict) and isinstance(value.get('ros__parameters'), dict):
            names.update(value['ros__parameters'])
    return names


def _read(path):
    try:
        return Path(path).read_text(errors='replace')
    except OSError:
        return None


def _without_processes(node, known, run_dir):
    """([codes], how they were found) for a node processes.txt does not name."""
    same = sorted(code for code, entry in known.items() if entry['node'] == node)
    if len(same) == 1:
        return same, 'node name'
    if not same:
        return [f'{UNKNOWN}/{node}'], f"{UNKNOWN}: no entry in the rules' algorithms list starts {node}"
    params = Path(run_dir) / 'metadata' / 'params' / f'{node}.yaml' if run_dir is not None else None
    reported = reported_parameters(params) if params is not None else None
    if reported:
        hits = [code for code in same if known[code].get('parameter') in reported]
        if len(hits) == 1:
            return hits, f"metadata/params/{node}.yaml ({known[hits[0]]['parameter']})"
    return [f'{UNKNOWN}/{node}'], f"{UNKNOWN}: {' or '.join(same)}, and nothing in the run tells which"


def identify(names, known, run_dir=None):
    """[{node, code, repo, role, from}] for each node name, sorted by node then code.

    names: the run's algorithm node names (run_info.algorithm_names).
    known: the rules' algorithms list. run_dir: where metadata/processes.txt and
    metadata/params/ are read from; without it only the node name can match.
    Two listed processes running under one node name are both listed."""
    processes = _read(Path(run_dir) / 'metadata' / 'processes.txt') if run_dir is not None else None
    by_process = {}
    for node, code in from_processes(processes, known):
        by_process.setdefault(node, []).append(code)
    out = []
    for name in sorted(set(names)):
        node = name.strip('/').rsplit('/', 1)[-1]
        if node in by_process:
            codes, source = by_process[node], 'metadata/processes.txt'
        else:
            codes, source = _without_processes(node, known, run_dir)
        for code in codes:
            entry = known.get(code, {})
            out.append({'node': name, 'code': code, 'repo': entry.get('repo', UNKNOWN),
                        'role': role(entry), 'from': source})
    return out


def drove(identified):
    """Whether any of it is a driving algorithm. Code not on the list counts:
    the run only names it because it published a drive topic."""
    return any(item['role'] == DRIVING for item in identified)


def driving(identified):
    """(codes, repos) of the driving code, sorted, each once: the inventory's columns."""
    items = [item for item in identified if item['role'] == DRIVING]
    return sorted({item['code'] for item in items}), sorted({item['repo'] for item in items})
