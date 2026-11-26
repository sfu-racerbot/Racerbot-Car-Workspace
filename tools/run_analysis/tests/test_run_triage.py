"""run_triage.py: what a recording really was.

Runner: python3 -m pytest tools/run_analysis/tests/ (no ROS needed for these).
Oracles: closed forms on built arrays; processes.txt lines copied from run 20261030-180355's metadata.
Reading a real bag is exercised by running it; results in docs/superpowers/specs/2026-10-30-slam-reliability.md.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_analysis import run_triage as rt  # noqa: E402

PROCESSES = """    PID    PPID                  STARTED COMMAND
   3136    2354 Fri Oct 30 17:28:14 2026 /usr/bin/python3 /home/x/install/web_dashboard/lib/web_dashboard/dashboard_node
   9001    9000 Fri Oct 30 18:03:58 2026 /home/x/team_a/install/disparity_extender/lib/disparity_extender/disparity_extender_node
   9002    9000 Fri Oct 30 18:03:58 2026 /home/x/team_a/install/disparity_extender/lib/disparity_extender/disparity_extender_node --log
   9100    9099 Fri Oct 30 18:04:01 2026 /opt/ros/jazzy/lib/slam_toolbox/async_slam_toolbox_node --ros-args
"""


def test_algorithms_names_the_driving_code_once_each_and_not_the_dashboard():
    # package/executable from each install path; disparity_extender runs twice and is listed once.
    assert rt.algorithms(PROCESSES) == ['disparity_extender/disparity_extender_node',
                                        'slam_toolbox/async_slam_toolbox_node']
    assert rt.algorithms('') == []


def test_hold_stats_integrate_the_wheel_speed_inside_each_hold():
    t = np.arange(0.0, 30.0, 0.02)
    v = np.where((t >= 10.0) & (t < 20.0), 2.0, 0.0)
    holds = rt.hold_stats(np.c_[t, v], [(10.0, 20.0), (25.0, 26.0)])
    assert holds[0][1] == pytest.approx(10.0)  # s
    assert holds[0][2] == pytest.approx(20.0, abs=0.05)  # m: 2 m/s for 10 s
    assert holds[0][3] == pytest.approx(2.0)
    assert holds[1][2] == pytest.approx(0.0)


def test_reverse_counts_as_distance():
    t = np.arange(0.0, 5.0, 0.02)
    assert rt.hold_stats(np.c_[t, np.full(len(t), -1.0)], [(0.0, 4.0)])[0][2] == pytest.approx(4.0, abs=0.05)


def test_a_hold_with_no_odometry_is_zero_not_a_crash():
    assert rt.hold_stats(np.zeros((0, 2)), [(1.0, 2.0)]) == [(1.0, 1.0, 0.0, 0.0)]


def test_scan_change_ignores_beams_either_scan_missed():
    # Zero is a beam that returned nothing; two of them would move the median to 4 m if counted.
    a = np.array([1.0, 2.0, np.inf, 3.0, 0.0, 0.0])
    b = np.array([1.2, 2.2, 5.0, np.nan, 4.0, 4.0])
    assert rt.scan_change(a, b) == pytest.approx(0.2)
    assert np.isnan(rt.scan_change(np.full(3, np.inf), np.ones(3)))


def _run(carry=True, wheels=False):
    """Two holds; between them the scan jumps 0.5 m (carried) or stays put, and the wheels turn or don't."""
    scan_t = np.arange(0.0, 20.0, 0.025)
    scans = np.tile(np.linspace(1.0, 4.0, 50), (len(scan_t), 1))
    if carry:
        scans[scan_t >= 11.0] += 0.5
    t = np.arange(0.0, 20.0, 0.02)
    v = np.zeros(len(t))
    if wheels:
        v[(t > 10.5) & (t < 11.5)] = 0.5
    return scan_t, scans, np.c_[t, v], [(2.0, 10.0), (12.0, 19.0)]


def test_a_car_moved_by_hand_between_holds_is_flagged():
    assert rt.moved_between(*_run(carry=True)) == [True]


def test_a_car_left_where_it_stopped_is_not():
    assert rt.moved_between(*_run(carry=False)) == [False]


def test_a_car_that_drove_between_holds_was_not_carried():
    # The wheels turned: odometry saw it, so SLAM can follow it.
    assert rt.moved_between(*_run(carry=True, wheels=True)) == [False]


def test_a_release_too_short_to_see_the_wheels_stop_is_not_carried():
    # recorded: rb1 20261111-153721 released LB for 0.4 s at 2.7 m/s; the scan moved because the car was
    # driving. Nothing in the first 0.5 s after a release counts (the car is coasting), so there is no evidence.
    scan_t, scans, odom, _ = _run(carry=True)
    odom[:, 1] = 2.7
    assert rt.moved_between(scan_t, scans, odom, [(2.0, 10.8), (11.2, 19.0)]) == [False]


def test_one_hold_has_no_gaps():
    scan_t, scans, odom, windows = _run()
    assert rt.moved_between(scan_t, scans, odom, windows[:1]) == []


def test_verdicts():
    long_hold = [(0.0, 60.0, 150.0, 3.0)]
    assert rt.verdict([], []) == 'not a mapping run: LB held under 5 s'
    assert rt.verdict([(0.0, 4.9, 10.0, 3.0)], []) == 'not a mapping run: LB held under 5 s'
    assert rt.verdict(long_hold, []) == 'one continuous car: usable'
    holds = [(100.0, 10.0, 20.0, 3.0), (130.0, 30.0, 70.0, 3.0), (170.0, 40.0, 90.0, 3.0)]
    text = rt.verdict(holds, [True, False])
    assert text.startswith('carried between holds 1 and 2')
    assert '--from-seconds 30 ' in text  # the second hold starts 30 s after the first


def test_with_two_carries_the_replay_starts_after_the_last():
    holds = [(0.0, 10.0, 20.0, 3.0), (20.0, 10.0, 20.0, 3.0), (40.0, 10.0, 20.0, 3.0), (60.0, 30.0, 70.0, 3.0)]
    text = rt.verdict(holds, [True, False, True])
    assert text.startswith('carried between holds 3 and 4')
    assert '--from-seconds 60 ' in text
