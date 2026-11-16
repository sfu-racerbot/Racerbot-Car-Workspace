"""The referee's neural driver and the archive worker, without the gym: the
driver runs the car's own observation code on one tick, and a race that
fails in a worker comes back as an error instead of hanging the pool.

Runner: PYTHONPATH=<torch> python3 -m pytest tools/neural_racer/tests/test_referee_driver.py -v   (CPU)
"""
import math

import conftest  # noqa: F401
import numpy as np
import torch

from neural_racer import observation as obs
from neural_racer_train import eval_worker, referee
from neural_racer_train.export import weights_of
from neural_racer_train.model import HALF, PolicyNet


def test_neural_driver_runs_the_car_s_observation():
    """It left out the cornering cap observation spec v2 added: a TypeError on every race's first tick."""
    drv = referee.NeuralDriver(weights_of(PolicyNet()), v_cap=2.5)
    steer, speed = drv.command({'scan': np.full(obs.N_BEAMS, 3.0), 'std_state': np.zeros(7)}, np.zeros(7), 0)
    assert math.isfinite(steer) and math.isfinite(speed)


def test_a_failed_race_is_an_error_not_a_hang(tmp_path, monkeypatch):
    """The gym's bootstrap exits a worker that cannot import it: Pool.map then waited forever."""
    path = tmp_path / 'snap.pt'
    torch.save({'policy': PolicyNet(HALF).state_dict()}, path)

    def no_gym(*_a, **_k):
        raise SystemExit('F1TENTH Gym is not set up')
    monkeypatch.setattr(referee, 'race', no_gym)
    res = eval_worker._race_job((str(path), True, HALF, 'gap_follow', 'heldout_00', 0, True, 2.5))
    assert res == {'error': 'SystemExit: F1TENTH Gym is not set up'}


def test_neural_driver_reads_the_imu_as_noisy_as_training_does():
    """The referee's student gets the sim's IMU noise (gpu_sim.imu.noise_std): car 1's vibration."""
    from neural_racer_train.gpu_sim import imu
    from neural_racer_train.gpu_sim.params import CarParams
    drv = referee.NeuralDriver(weights_of(PolicyNet()), v_cap=2.5)
    p = CarParams()
    for speed in (0.0, 4.5):
        want = imu.noise_std(torch.tensor(p.imu_gyro_noise), torch.tensor(p.imu_accel_noise),
                             torch.tensor(1.0), torch.tensor([speed]))[0].numpy()
        assert np.allclose(drv._imu_std(speed), want, atol=1e-6)
    quiet = referee.NeuralDriver(weights_of(PolicyNet()), v_cap=2.5, imu_vibration=0.0)
    assert np.allclose(quiet._imu_std(4.5), [p.imu_gyro_noise, p.imu_accel_noise, p.imu_accel_noise])
