"""Which layers are running on the car, from the process list.

Bringup is "the hardware layer" and at most one control layer should sit on
top of it (docs/architecture.md, two-tier launch pattern). The menu reads
this from /proc, the same way racerbot_doctor does, so it works even while
ROS discovery is slow or the graph is on another domain.

A process is matched by its package and executable, both taken from the
installed path `.../lib/<package>/<executable>`, because two packages ship
an executable with the same name (gap_follow's gap_follow_node and Team A's
reactive gap_follow_node).
"""
from __future__ import annotations

from dataclasses import dataclass

from racerbot_doctor import car as doctor_car

# (package, executable) -> what to call it. Every node that can publish a
# drive command and is started by a shipped launch file, ours and Team A's
# (sfu-racerbot/racerbot_a src/*/launch, checked 2026-11-28).
DRIVING = {
    ('joy_teleop', 'joy_teleop'): 'manual',
    ('gap_follow', 'gap_follow_node'): 'gap follow',
    ('pure_pursuit', 'pure_pursuit_node'): 'race (pure pursuit)',
    ('pure_pursuit', 'auto_map_race_node'): 'map and race',
    ('neural_racer', 'neural_racer_node'): 'neural racer',
    ('ftg', 'ftg_node'): 'Team A follow the gap',
    ('disparity_extender', 'disparity_extender_node'): 'Team A disparity extender',
    ('reactive', 'gap_follow_node'): 'Team A reactive',
    ('pure_pursuit_a', 'pure_pursuit_a_node'): 'Team A pure pursuit',
}

SIMULATOR = {('racerbot_sim', 'gym_bridge_node'), ('racerbot_sim', 'sim_joy_node')}


def package_and_exe(argv: list[str]) -> tuple[str | None, str | None]:
    """(package, executable) of a ROS process, or (None, basename) if its
    path has no `lib/<package>/` part. Looks at argv[0] and, for a Python
    entry point run as `python3 <script>`, argv[1]."""
    for arg in argv[:2]:
        parts = arg.split('/')
        exe = parts[-1]
        if len(parts) >= 3 and parts[-3] == 'lib':
            return parts[-2], exe
    if not argv:
        return None, None
    return None, argv[0].rsplit('/', 1)[-1]


@dataclass(frozen=True)
class Layers:
    hardware: str               # 'up', 'partial' or 'off'
    missing: tuple[str, ...]    # bringup executables not running
    driving: tuple[str, ...]    # labels of running driving layers
    simulator: bool
    dashboard: bool = False


def classify(processes) -> Layers:
    """`processes` is any iterable of objects with `.argv` (racerbot_doctor
    snapshot.Process)."""
    running_exes = set()
    driving = []
    simulator = False
    for proc in processes:
        pkg, exe = package_and_exe(list(proc.argv))
        if exe is None:
            continue
        running_exes.add(exe)
        label = DRIVING.get((pkg, exe))
        if label is not None and label not in driving:
            driving.append(label)
        if (pkg, exe) in SIMULATOR:
            simulator = True
    missing = tuple(e for e in doctor_car.BRINGUP_EXES if e not in running_exes)
    if not missing:
        hardware = 'up'
    elif len(missing) == len(doctor_car.BRINGUP_EXES):
        hardware = 'off'
    else:
        hardware = 'partial'
    return Layers(hardware, missing, tuple(sorted(driving)), simulator,
                  dashboard='web_dashboard_node' in running_exes)
