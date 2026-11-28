"""Static checks: step 1 of the console cannot command the car.

The package must create no ROS publisher, service client or action client,
must never mention the deadman parameters, and its job list must hold no
job that can move the car or that runs `ros2 launch` (docs/console.md,
safety; design doc, build plan step 1). A failing check here means a
driving path was added before the Get ready checks exist.
"""
import re
from pathlib import Path

from racerbot_console import jobs

PKG = Path(__file__).resolve().parents[1] / 'racerbot_console'
SOURCES = sorted(PKG.glob('*.py'))
FORBIDDEN = [r'create_publisher', r'create_client', r'ActionClient', r'enable_deadman',
             r'deadman_button', r'enable_safety', r'enable_ttc', r'\.publish\(',
             r'topic\s+pub']


def test_sources_were_found():
    names = {p.name for p in SOURCES}
    assert {'app.py', 'ros_reader.py', 'jobs.py'} <= names


def test_no_source_can_send_anything_to_the_graph():
    hits = [(p.name, pat) for p in SOURCES for pat in FORBIDDEN
            if re.search(pat, p.read_text())]
    assert hits == []


def test_no_job_can_move_the_car_or_launch_anything():
    for spec in jobs.JOBS.values():
        assert not spec.moves_car, spec.key
        assert 'launch' not in spec.argv, spec.key
