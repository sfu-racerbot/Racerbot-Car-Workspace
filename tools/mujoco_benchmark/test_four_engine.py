import csv
from io import StringIO
import os

import numpy as np
import pytest


def test_native_reset_turn_brake_and_complete_rollouts():
    model = os.environ.get('MUSHR_MODEL')
    if not model:
        pytest.skip('set MUSHR_MODEL to the pinned upstream one_car.xml')
    from four_engine import FlatMuSHR, rollout

    plant = FlatMuSHR(model)
    rows = [dict(step=i, speed_cmd=2. if i < 240 else 0., steer_cmd=.15 if i < 240 else 0.,
                 initial_speed=1.7) for i in range(480)]
    plant.reset(1.7)
    np.testing.assert_allclose(plant.observe(), (1.7, 0.), atol=1e-12)
    np.testing.assert_allclose(plant.data.qpos[:7], (0., 0., 0., 1., 0., 0., 0.))
    assert plant.data.joint('buddy_steering_wheel').qpos[0] == 0.
    trace = rollout(plant, rows)
    assert trace.shape == (480, 2)
    assert trace[160:200, 0].mean() > 1.
    assert trace[160:200, 1].mean() > .5
    assert abs(trace[-20:, 0].mean()) < .02
    assert plant.data.time == pytest.approx(12.)
    np.testing.assert_array_equal(trace, rollout(plant, rows))
    fresh = FlatMuSHR(model)
    np.testing.assert_array_equal(trace, rollout(fresh, rows))
    with pytest.raises(ValueError, match='step'):
        rollout(plant, rows[:-1])


def test_csv_keeps_integer_keys_and_refuses_partial_data(tmp_path):
    model = os.environ.get('MUSHR_MODEL')
    if not model:
        pytest.skip('set MUSHR_MODEL to the pinned upstream one_car.xml')
    from four_engine import FlatMuSHR, write_predictions

    plant = FlatMuSHR(model)
    b = 1_794_439_298_762_000_001
    header = 'car,run,b_ns,step,speed_cmd,steer_cmd,initial_speed\n'
    rows = [f'rb1,demo,{b},{i},1.5,0.0,{1.2 if i == 0 else ""}\n' for i in range(480)]
    out = tmp_path / 'predictions.csv'
    with pytest.raises(ValueError, match='incomplete block'):
        write_predictions(plant, StringIO(header + ''.join(rows + [rows[0]])), out)
    assert not out.exists()
    assert write_predictions(plant, StringIO(header + ''.join(rows)), out) == 1
    with out.open(newline='') as f:
        actual = list(csv.DictReader(f))
    assert len(actual) == 480
    for i, row in enumerate(actual):
        assert (row['car'], row['run'], int(row['b_ns']), int(row['step'])) == ('rb1', 'demo', b, i)
    with pytest.raises(FileExistsError):
        write_predictions(plant, StringIO(header), out)
