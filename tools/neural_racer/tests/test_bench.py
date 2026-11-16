"""The bench's race list and its summary, without the gym (the races
themselves run on the referee, tests/test_referee.py).

Runner: PYTHONPATH=<torch> python3 -m pytest tools/neural_racer/tests/test_bench.py -v   (CPU)
"""
import conftest  # noqa: F401

from neural_racer_train import bench


def test_every_driver_races_alone_and_each_student_meets_each_panel_member_both_ways():
    todo = bench.jobs(['a.pt'], ['t1', 't2'], laps=3, timeout=100.0, v_cap=5.0, a_lat=2.0, solo_seeds=2, h2h_seeds=4,
                      vibration=1.0)
    solo = [j for j in todo if j[0] == 'solo']
    h2h = [j for j in todo if j[0] != 'solo']
    assert len(solo) == 2 * 3 * 2 and len(h2h) == 2 * 2 * 4         # tracks x drivers x seeds; x panel x seeds
    firsts = [j[2][1][0][0] for j in h2h if j[0] == 'student0|gap_follow' and j[2][0] == 't1']
    assert firsts.count('student0') == firsts.count('gap_follow') == 2    # grid order alternates


def test_summary_counts_finishes_laps_wins_and_faults():
    res = [
        ('solo', 'student0', {'track': 't', 'seed': 0, 'names': ['student0'], 'order': ['student0'],
                              'finish': [30.0], 'laps': [3.0], 'contact': [False], 'fault': [False]}),
        ('solo', 'student0', {'track': 't', 'seed': 1, 'names': ['student0'], 'order': ['student0'],
                              'finish': [None], 'laps': [1.2], 'contact': [True], 'fault': [True]}),
        ('student0|gap_follow', None, {'track': 't', 'seed': 0, 'names': ['gap_follow', 'student0'],
                                       'order': ['student0', 'gap_follow'], 'finish': [None, 25.0],
                                       'laps': [2.0, 3.0], 'contact': [True, False], 'fault': [True, False]}),
        ('solo', 'gap_follow', {'track': 't', 'seed': 0, 'error': 'RuntimeError: grid overlap'}),
    ]
    rep = bench.summarize(res, ['a.pt'])
    assert rep['solo']['t']['student0'] == {'races': 2, 'finished': 1, 'contact': 1, 'lap_s': 10.0}
    assert rep['h2h']['student0|gap_follow']['t'] == {'races': 1, 'wins': 1, 'losses': 0, 'draws': 0, 'from_second': 1,
                                                      'wins_from_second': 1, 'fault_student': 0, 'fault_panel': 1}
    assert len(rep['errors']) == 1


def test_head_to_heads_are_scored_on_finishing_order():
    """Crashed cars stay on the track (the team, 2026-11-08), so distance covered decided most races
    of the Nov 7 round robin: now a crash never wins, and two non-finishers draw."""
    from neural_racer_train import referee
    names = ['student0', 'gap_follow']
    assert referee.finish_outcome(names, [30.0, None], 'student0', 'gap_follow') == 'win'
    assert referee.finish_outcome(names, [None, 31.0], 'student0', 'gap_follow') == 'loss'
    assert referee.finish_outcome(names, [None, None], 'student0', 'gap_follow') == 'draw'
    assert referee.finish_outcome(names, [29.0, 28.0], 'student0', 'gap_follow') == 'loss'
    res = [('student0|gap_follow', None, {'track': 't', 'seed': s, 'names': n, 'order': n, 'finish': f, 'laps': [1, 1],
                                          'contact': [False, False], 'fault': [False, False]})
           for s, n, f in ((0, ['student0', 'gap_follow'], [None, None]),       # ahead on distance, nobody finished
                           (1, ['gap_follow', 'student0'], [31.0, 30.0]))]      # started second, finished first
    row = bench.summarize(res, ['a.pt'])['h2h']['student0|gap_follow']['t']
    assert (row['wins'], row['losses'], row['draws'], row['from_second'], row['wins_from_second']) == (1, 0, 1, 1, 1)
