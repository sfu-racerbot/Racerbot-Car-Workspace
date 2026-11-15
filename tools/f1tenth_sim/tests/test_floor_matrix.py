"""floor_matrix.py's arm parsing, cell summary and verdict (ROS-free), plus eval_gap's grip_scale reaching the
plant. Run: rb python3 -m pytest tools/f1tenth_sim/tests/test_floor_matrix.py"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import floor_matrix as fm  # noqa: E402


def run(laps=2, collision=False, clear=0.15, times=(11.0, 10.0)):
    return {'laps': laps, 'collision': collision, 'min_chassis_map_clearance_m': clear, 'lap_times_s': list(times)}


def test_parse_arm_types_and_empty():
    assert fm.parse_arm('p71:steering_gain=0.8,max_steering_rate=4,enable_x=false,mode=fast') == (
        'p71', {'steering_gain': 0.8, 'max_steering_rate': 4, 'enable_x': False, 'mode': 'fast'})
    assert fm.parse_arm('shared:') == ('shared', {})


@pytest.mark.parametrize('bad', [':a=1', 'x:a', 'x:=1'])
def test_parse_arm_refuses_malformed(bad):
    with pytest.raises(ValueError):
        fm.parse_arm(bad)


def test_cell_summary_counts_stalls_and_contacts_separately():
    # oracle: definitions in the module docstring; a contact run is not also a stall
    runs = [run(), run(laps=1), run(laps=0.6, collision=True, clear=-0.01), run(clear=0.08, times=(12.0, 9.5))]
    s = fm.cell_summary(runs, laps=2)
    assert s == {'runs': 4, 'stalls': 1, 'contacts': 1, 'min_clearance_m': -0.01, 'median_flying_s': 10.0}


def test_cell_summary_one_lap_short_of_the_target_is_a_stall_exactly_at_it_is_not():
    assert fm.cell_summary([run(laps=1.999)], 2)['stalls'] == 1
    assert fm.cell_summary([run(laps=2.0)], 2)['stalls'] == 0


def summary(stalls=0, contacts=0, clear=0.15):
    return {'runs': 3, 'stalls': stalls, 'contacts': contacts, 'min_clearance_m': clear, 'median_flying_s': 10.0}


def test_verdict_accepts_equal_and_within_slack():
    ref = {'a|car|1.0': summary(clear=0.15)}
    assert fm.verdict(ref, {'a|car|1.0': summary(clear=0.15)}, 0.03) == []
    assert fm.verdict(ref, {'a|car|1.0': summary(clear=0.12 + 1e-9)}, 0.03) == []


def test_verdict_flags_one_step_past_the_clearance_slack():
    ref = {'a|car|1.0': summary(clear=0.15)}
    assert len(fm.verdict(ref, {'a|car|1.0': summary(clear=0.12 - 1e-6)}, 0.03)) == 1


def test_verdict_flags_new_stalls_and_contacts_but_not_shared_ones():
    ref = {'a|f|1.0': summary(stalls=1), 'b|f|1.0': summary(contacts=1)}
    worse = {'a|f|1.0': summary(stalls=2), 'b|f|1.0': summary(contacts=2)}
    same = {'a|f|1.0': summary(stalls=1), 'b|f|1.0': summary(contacts=1)}
    assert len(fm.verdict(ref, worse, 0.03)) == 2
    assert fm.verdict(ref, same, 0.03) == []


def test_verdict_flags_missing_clearance_when_the_reference_has_one():
    assert len(fm.verdict({'c': summary(clear=0.1)}, {'c': summary(clear=None)}, 0.03)) == 1


def test_grip_scale_reaches_the_plant(monkeypatch):
    # eval_gap.run_track(grip_scale=...) must call the plant's set_grip_scale with that value once, after reset.
    import eval_gap
    calls = []
    real = eval_gap.make_plant

    def spy(*a, **k):
        plant = real(*a, **k)
        orig = plant.set_grip_scale
        plant.set_grip_scale = lambda scale, index=None: (calls.append(scale), orig(scale, index))[1]
        return plant
    monkeypatch.setattr(eval_gap, 'make_plant', spy)
    import gap_mirror
    eval_gap.run_track('indoor_oval', gap_mirror.node_params(car_id='rb2'), 12345, 0.5, car_id='rb2',
                       grip_scale=0.85)
    assert calls == [0.85]
    eval_gap.run_track('indoor_oval', gap_mirror.node_params(car_id='rb2'), 12345, 0.5, car_id='rb2')
    assert calls == [0.85]
