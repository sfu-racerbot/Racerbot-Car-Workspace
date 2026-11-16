"""Access to the existing CPU simulator (tools/f1tenth_sim) -- the referee.

It is imported, never copied or edited: its sim_fidelity layer is calibrated
to the car and carries its own collision geometry, which is why it, not the
GPU training simulator, judges every candidate (plan A7, Phase 9).
"""
import ctypes
import os
import sys
from pathlib import Path

from neural_racer_train.common import REPO

# NR_SIM_ROOT: another workspace whose tools/f1tenth_sim, src/gap_follow and src/pure_pursuit the referee
# uses -- e.g. an upstream main checkout, to race the team's current algorithms (2026-11-03)
TOOLS_SIM = Path(os.environ.get('NR_SIM_ROOT', str(REPO))) / 'tools' / 'f1tenth_sim'
GLIB = ('libglib-2.0.so.0', 'libgobject-2.0.so.0', 'libgio-2.0.so.0')


def prefer_compat_glib() -> None:
    """On the Alliance clusters, load the software stack's own GLib before
    anything asks for one. In a Fir job the gym's cv2 otherwise met the
    host's GLib 2.68 and the stack's libgio, which needs 2.70's
    g_source_set_static_name: 'libgio-2.0.so.0: undefined symbol' (training
    plan item 3a; preloading these three fixed it, job 1524538). A no-op
    anywhere EBROOTGENTOO is unset."""
    root = os.environ.get('EBROOTGENTOO')
    for name in GLIB if root else ():
        path = Path(root) / 'lib64' / name
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def ensure_paths() -> None:
    prefer_compat_glib()
    if str(TOOLS_SIM) not in sys.path:
        sys.path.insert(0, str(TOOLS_SIM))
    from sim_fidelity import bootstrap
    bootstrap.bootstrap()


def calibration_module():
    ensure_paths()
    from sim_fidelity import calibration
    return calibration


def gym_available() -> bool:
    try:
        ensure_paths()
        import f1tenth_gym  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False
