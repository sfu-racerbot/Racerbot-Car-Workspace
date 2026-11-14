"""eval_gap --trace: the per-tick trace file, and that asking for it changes no result. Run by the tool-suite command in CLAUDE.md."""
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src', 'gap_follow'))
import eval_gap  # noqa: E402

ARRAYS = ('t', 'x', 'y', 'yaw', 'v_true', 'v_meas', 'cmd_speed', 'steering', 'binding_cap', 'chassis_clearance',
          'controller_min_clear')


def comparable(result):
    return json.dumps({k: v for k, v in result.items() if k != 'wall_time_s'}, sort_keys=True)     # wall time is the only clock


def test_the_trace_has_every_array_at_the_run_length_and_the_results_are_identical_with_and_without_it(tmp_path):
    out = tmp_path / 't.npz'
    PARAMS = eval_gap.parse_set(['enable_opponent_awareness=false'])      # at call time: the car comes from the fixture
    plain = eval_gap.run_track('indoor_oval', PARAMS, 12345, 2.0, False)
    traced = eval_gap.run_track('indoor_oval', PARAMS, 12345, 2.0, False, trace=str(out))
    assert comparable(plain) == comparable(traced)
    z = np.load(out)
    assert set(z.files) == set(ARRAYS)
    n = traced['steps']
    assert all(len(z[k]) == n for k in ARRAYS), {k: len(z[k]) for k in ARRAYS}
    # oracle: the control tick is 25 ms, and the car is started at rest and accelerates
    assert np.allclose(np.diff(z['t']), 0.025, atol=1e-9)
    assert z['v_true'][0] == pytest.approx(0.0, abs=1e-6) and z['v_true'].max() > 0.5
    assert np.all(np.isfinite(z['chassis_clearance'])) and z['chassis_clearance'].min() > 0.0   # a safe start: free space all round
    assert z['cmd_speed'].max() > 0.5 and z['binding_cap'].dtype.kind == 'U'
