"""Diagnostic: change only the steering source upstream of the unmodified wrapper.

Use the same environment / CLI as probe.py (scorecard model out track seed).
The neural speed, recurrent state, history, limits, plant, spawn and referee
remain in the loop. This is a counterfactual, not a tuned candidate.
"""
from dataclasses import replace
from pathlib import Path
import json
import os
import sys

from probe import model_store, referee

scorecard, model, out, track, seed = sys.argv[1:]
score = json.loads(Path(scorecard).read_text())['date_split']
assert os.environ['RACERBOT_CAR'] == 'rb2'
assert model_store.weights_hash(Path(model)/'weights.npz') == score['model_hash']
weights, card = model_store.load_dir(Path(model))


def factory(plant):
    driver = referee.NeuralDriver(weights, card, v_cap=score['cap_mps'], imu=False)
    reference = referee.GapDriver()
    src_step, command = driver.policy.step, driver.command
    steer = None

    def tick(obs, truth, step):
        nonlocal steer
        steer, _ = reference.command(obs, truth, step)
        return command(obs, truth, step)

    def policy(obs):
        return replace(src_step(obs), steering=steer)

    driver.command, driver.policy.step = tick, policy
    return driver


r = referee.race(track, {'gap-steer-only': factory}, int(seed), timeout_s=60.)
print(r.as_dict(), flush=True)
(Path(out)/f'{track}-solo-{seed}-gap-steer-only.json').write_text(
    json.dumps(dict(**r.as_dict(), grid='solo', timeout_s=60.))+'\n')
