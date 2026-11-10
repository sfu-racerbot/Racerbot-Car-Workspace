"""Identify abandoned auto recorders from a process-table snapshot. No ROS."""

import shlex

from race_diagnostics.auto_record import AUTO_PREFIX, recorder_node_names
from race_diagnostics.recording import bag_record_command


def find_orphans(processes, own_pids):
    """Return matching PIDs, excluding our roots and their entire descendants."""
    owned = set(own_pids)
    while True:
        descendants = {pid for pid, ppid, _args in processes if ppid in owned}
        if descendants <= owned:
            break
        owned.update(descendants)
    diag, snapshot, bag = recorder_node_names(AUTO_PREFIX)
    bag_cmd = bag_record_command('', bag, ())
    flag = bag_cmd[bag_cmd.index(bag) - 1]
    markers = {f'name_prefix:={AUTO_PREFIX}', f'__node:={diag}', f'__node:={snapshot}'}
    found = []
    for pid, _ppid, args in processes:
        if pid in owned:
            continue
        try:
            tokens = shlex.split(args)
        except ValueError:
            tokens = args.split()
        if (markers.intersection(tokens)
                or any(a == flag and b == bag for a, b in zip(tokens, tokens[1:]))):
            found.append(pid)
    return sorted(found)
