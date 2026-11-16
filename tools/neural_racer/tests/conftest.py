"""Puts the training package and the car-side packages on sys.path, so
`python3 -m pytest tools/neural_racer/tests/` works from the workspace root
with PyTorch on PYTHONPATH (no colcon build, no ROS)."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
for rel in ('tools/neural_racer', 'src/neural_racer', 'src/gap_follow', 'src/drive_intent', 'src/pure_pursuit'):
    p = os.path.join(ROOT, rel)
    if p not in sys.path:
        sys.path.insert(0, p)

# Before any test module is collected: the gym-based modules put the gym
# checkout first on sys.path when their skip marks are evaluated.
CLEAN_SYS_PATH = list(sys.path)


@pytest.fixture
def clean_sys_path(monkeypatch):
    """sys.path without the gym checkout, for a test that starts spawned
    workers: they import from their parent's sys.path, and .sim/python
    carries its own scipy and numpy. With it, the library test's pool never
    returned once the gym modules had been collected (2026-10-27)."""
    monkeypatch.setattr(sys, 'path', list(CLEAN_SYS_PATH))
