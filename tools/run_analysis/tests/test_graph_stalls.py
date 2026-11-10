"""graph_stalls.py: intervals in which every watched topic went quiet at once.

Oracle: constructed receive-time series whose silent intervals are known by
construction (closed form), including the shape measured on car 2's run
20261109-190837 (0.36 s with /scan, /odom, the PF pose and /drive all silent).
Runs unsourced: python3 -m pytest tools/run_analysis/tests/test_graph_stalls.py
"""

import pytest

from run_analysis.graph_stalls import graph_stalls, topic_gaps


def ticks(start, end, hz):
    n = int(round((end - start) * hz))
    return [start + i / hz for i in range(n + 1)]


def test_a_stall_in_every_topic_is_found_with_its_bounds():
    # Every topic silent from 4.61 to 4.97 s, as in run 20261109-190837.
    times = {
        "/scan": ticks(0.0, 4.6, 40) + ticks(4.975, 8.0, 40),
        "/odom": ticks(0.0, 4.6, 50) + ticks(4.98, 8.0, 50),
        "/drive": ticks(0.005, 4.605, 40) + ticks(4.985, 8.0, 40),
    }
    stalls = graph_stalls(times, 0.15)
    assert len(stalls) == 1
    start, end = stalls[0]
    assert start == pytest.approx(4.605, abs=1e-9)   # last message of any topic before
    assert end == pytest.approx(4.975, abs=1e-9)     # first message of any topic after


def test_one_quiet_topic_is_not_a_graph_stall():
    # /drive silent for 1 s while /scan and /odom keep coming: a node, not the graph.
    times = {
        "/scan": ticks(0.0, 6.0, 40),
        "/odom": ticks(0.0, 6.0, 50),
        "/drive": ticks(0.0, 2.0, 40) + ticks(3.0, 6.0, 40),
    }
    assert graph_stalls(times, 0.15) == []
    assert topic_gaps(times["/drive"], 0.15) == [(2.0, 3.0)]


def test_a_gap_exactly_at_the_threshold_is_not_a_stall_and_one_step_past_is():
    # Binary-exact times: 0.25 apart is exactly the threshold, 0.25 + 2**-10 is past it.
    at = [0.0, 0.125, 0.375, 0.5]
    assert graph_stalls({"/a": at, "/b": at}, 0.25) == []
    past = [0.0, 0.125, 0.125 + 0.25 + 2 ** -10, 0.5]
    assert graph_stalls({"/a": past, "/b": past}, 0.25) == [(0.125, 0.125 + 0.25 + 2 ** -10)]


def test_only_the_overlap_of_the_topics_is_judged():
    # /pose starts 3 s late (the filter starting): the time before it is not a stall.
    times = {"/scan": ticks(0.0, 6.0, 40), "/pose": ticks(3.0, 6.0, 50)}
    assert graph_stalls(times, 0.15) == []


def test_empty_single_and_unordered_inputs():
    assert graph_stalls({}, 0.15) == []
    assert graph_stalls({"/scan": ticks(0.0, 2.0, 40)}, 0.15) == []
    assert graph_stalls({"/scan": ticks(0.0, 2.0, 40), "/odom": [1.0]}, 0.15) == []
    shuffled = {"/a": [2.0, 0.0, 1.0], "/b": [1.0, 2.0, 0.0]}
    assert graph_stalls(shuffled, 0.5) == [(0.0, 1.0), (1.0, 2.0)]


@pytest.mark.parametrize("bad", [0.0, -0.1])
def test_a_non_positive_threshold_is_refused(bad):
    with pytest.raises(ValueError, match="min_gap must be positive"):
        graph_stalls({"/a": [0.0, 1.0], "/b": [0.0, 1.0]}, bad)


def test_a_topic_with_one_message_does_not_narrow_the_window():
    # /pose sent once at 1.0 s; /scan and /odom stall together from 0.5 to 0.8 s.
    scan = ticks(0.0, 0.5, 40) + ticks(0.8, 2.0, 40)
    odom = ticks(0.0, 0.5, 50) + ticks(0.8, 2.0, 50)
    assert graph_stalls({"/scan": scan, "/odom": odom, "/pose": [1.0]}, 0.15) == [
        (pytest.approx(0.5, abs=1e-9), pytest.approx(0.8, abs=1e-9))]


def test_a_single_live_topic_is_never_a_graph_stall_even_with_a_gap():
    gappy = ticks(0.0, 1.0, 40) + ticks(2.0, 3.0, 40)
    assert graph_stalls({"/scan": gappy}, 0.15) == []
    assert graph_stalls({"/scan": gappy, "/odom": []}, 0.15) == []


def test_a_gap_before_another_topic_starts_is_outside_the_window():
    # /scan pauses 1.0-1.5 s while the filter has not started yet (it starts at 3 s).
    scan = ticks(0.0, 1.0, 40) + ticks(1.5, 6.0, 40)
    assert graph_stalls({"/scan": scan, "/pose": ticks(3.0, 6.0, 50)}, 0.15) == []


def test_a_gap_after_another_topic_stops_is_outside_the_window():
    # The controller exits at 3 s; /scan pauses 4.0-4.5 s afterwards.
    scan = ticks(0.0, 4.0, 40) + ticks(4.5, 6.0, 40)
    assert graph_stalls({"/scan": scan, "/drive": ticks(0.0, 3.0, 40)}, 0.15) == []
