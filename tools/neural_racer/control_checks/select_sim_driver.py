"""Select a driver for one headless rb2 simulation. No ROS, actuator, manifest or training."""

import argparse
from collections import Counter
import csv
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import yaml

from mapped_replay import ROOT, ASSET, model_store, effective_config, SafetyWrapper, WrapperConfig

TRACK = "car2_course_2026_11_11"
HASH = "a0eef2af0a3757b65e12b6d7dada16ace5c48e61a2ce460992d836499395477b"
sys.path.insert(0, str(ROOT / "tools/f1tenth_sim"))
sys.path.insert(0, str(ROOT / "tools/neural_racer"))
from neural_racer_train import referee
import run_validation as sim


def select(driver, model):
    if driver == "gap-follow":
        return referee.GapDriver(), None
    if driver != "saved-policy":
        raise ValueError(f"unknown driver: {driver}")
    if model is None:
        raise ValueError("missing frozen checkpoint; no fallback")
    pins = Path(__file__).with_name("model-sha256.txt").read_text().splitlines()
    expected = {file: digest for digest, file in (row.split() for row in pins)}
    if expected.get("weights.npz") != HASH or set(expected) != {
        "weights.npz",
        "golden.npz",
        "model_card.yaml",
    }:
        raise ValueError("unexpected checkpoint inventory")
    for file, digest in expected.items():
        if hashlib.sha256((model / file).read_bytes()).hexdigest() != digest:
            raise ValueError(f"incorrect frozen {file}; no fallback")
    w, card = model_store.load_dir(model)
    model_store.check_modalities(card, camera_enabled=False)
    cfg = effective_config("neural_racer.yaml", "neural_racer_node", "rb2")
    if not (
        cfg["enable_deadman"] and cfg["require_imu"] and cfg["shadow"] and cfg["max_speed"] == 1.0
    ):
        raise ValueError("unexpected rb2 safety defaults")
    d = referee.NeuralDriver(w, card, v_cap=cfg["max_speed"], imu=True)
    if not model_store.golden_error(d.policy, model / "golden.npz") <= model_store.GOLDEN_ATOL:
        raise ValueError("checkpoint reference predictions differ")
    wrapper = SafetyWrapper(WrapperConfig(**{f.name: cfg[f.name] for f in fields(WrapperConfig)}))
    probe = Probe(wrapper)
    d.wrapper = probe
    return d, probe


class Probe:
    """Observe the inputs/result of the real safety wrapper; return its exact result."""

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.last = None
        self.calls = 0

    def apply(self, *args):
        result = self.wrapper.apply(*args)
        self.last = ((float(args[0]), float(args[1])), result)
        self.calls += 1
        return result


def run(name, model, trace, release_check=False):
    maps = ROOT / "tools/f1tenth_sim/maps"
    catalog = yaml.safe_load((maps / "catalog.yaml").read_text())
    if ASSET.name not in catalog[TRACK]["source"]:
        raise ValueError("simulator course is not the November rebuilt SLAM map")
    sim.select_profile("car", "rb2")
    driver, probe = select(name, model)
    plant = sim.make_plant(str(maps / TRACK), 1, 7, car_id="rb2", max_laps=2)
    sink = Mock(wraps=plant.step)
    dt = sim.CONTROL_DT
    binding, counts = Counter(), Counter()
    peak = displacement = distance = 0.0
    contact = finished = False
    held_speed = []
    raw_speed = []
    try:
        line = plant.env.unwrapped.track.raceline
        pose = sim.initial_pose(line, 0).reshape(1, 3)
        observation, _ = plant.reset(pose)
        if observation["agent_0"]["collision"]:
            raise ValueError("invalid spawn: not scored")
        previous_xy = plant.truth(0)[:2].copy()
        with trace.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "phase",
                    "tick",
                    "raw_steer",
                    "raw_speed",
                    "limit",
                    "wrapped_steer",
                    "wrapped_speed",
                    "plant_input_steer",
                    "plant_input_speed",
                    "truth_x",
                    "truth_y",
                    "truth_speed",
                    "lap_count",
                    "contact",
                )
            )
            for phase, ticks in (
                ("not_held", 4),
                ("held", 40 if release_check else 2400),
                ("released", 60),
            ):
                for step in range(ticks):
                    branch = "simulated_deadman"
                    raw = wrapped = ("", "")
                    cmd = (0.0, 0.0)
                    if phase == "held":
                        before = probe.calls if probe else None
                        cmd = driver.command(observation["agent_0"], plant.truth(0).copy(), step)
                        if probe:
                            assert (
                                probe.calls == before + 1
                            ), "no safety result for this policy command"
                            raw, result = probe.last
                            wrapped = (result.steering, result.speed)
                            branch = result.reason or result.binding
                            assert cmd == wrapped, "selector did not forward the wrapped command"
                            assert (
                                abs(cmd[0]) <= probe.wrapper.cfg.max_steering_angle
                                and 0 <= cmd[1] <= probe.wrapper.cfg.max_speed
                            )
                            raw_speed.append(raw[1])
                            binding[branch] += 1
                        else:
                            branch = "gap_follow"
                        held_speed.append(cmd[1])
                    assert np.isfinite(cmd).all()
                    # This is the only sink; FidelityPlant.step actually advances the simulated car.
                    observation, _, done, truncated, info = sink([cmd])
                    (delivered,) = sink.call_args.args[0]
                    if phase == "held" and probe:
                        assert delivered == wrapped, "simulated plant received a different command"
                        counts["exact_policy_wrapper_to_plant"] += 1
                    elif phase != "held":
                        assert delivered == (0.0, 0.0), "nonzero input with virtual bumper released"
                    state = plant.truth(0)
                    counts[phase] += 1
                    peak = max(peak, abs(float(state[3])))
                    distance += float(np.linalg.norm(state[:2] - previous_xy))
                    previous_xy = state[:2].copy()
                    displacement = max(displacement, float(np.linalg.norm(state[:2] - pose[0, :2])))
                    hit = bool(observation["agent_0"]["collision"])
                    lap = float(info["lap_counts"][0])
                    writer.writerow(
                        (
                            phase,
                            step,
                            *raw,
                            branch,
                            *wrapped,
                            *delivered,
                            *state[:2],
                            state[3],
                            lap,
                            hit,
                        )
                    )
                    if phase == "held":
                        finished |= lap >= 1
                        contact |= hit
                        if finished:
                            break
                    if hit or done or truncated:
                        raise RuntimeError(
                            f"episode terminated early: tick={step}, phase={phase}, contact={hit}"
                        )
        assert counts["released"] == 60
        return dict(
            driver=name,
            scenario="moving_release" if release_check else "one_lap_or_timeout",
            checkpoint_sha256=HASH if probe else None,
            track=TRACK,
            car="rb2",
            observation_mode="reactive on SLAM geometry" if probe else "gap-follow",
            seed=7,
            control_period_s=dt,
            spawn_xy_yaw=pose[0].tolist(),
            phase_counts=dict(counts),
            wrapper_bindings=dict(binding),
            positive_held_speed_commands=int(np.count_nonzero(np.array(held_speed) > 0)),
            peak_raw_policy_request_mps=max(raw_speed, default=None),
            peak_command_mps=max(held_speed),
            peak_truth_speed_mps=peak,
            traveled_m=distance,
            max_displacement_m=displacement,
            speed_after_release_mps=float(plant.truth(0)[3]),
            contact=contact,
            finished_lap=finished,
            held_horizon_s=counts["held"] * dt,
            release_s=counts["released"] * dt,
            trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
            scope="CPU simulation; simulated held/released input, no ROS graph or hardware",
        )
    finally:
        plant.close()


if __name__ == "__main__":
    if (
        not __debug__
        or Path("/proc/device-tree/serial-number").exists()
        or Path("/dev/sensors").exists()
    ):
        raise SystemExit("use normal Python on a laptop, never a car")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", choices=("saved-policy", "gap-follow"), required=True)
    p.add_argument("--model", type=Path)
    p.add_argument(
        "--moving-release",
        action="store_true",
        help="release after one second rather than wait for a lap",
    )
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    a = p.parse_args()
    a.result.write_text(
        json.dumps(run(a.driver, a.model, a.trace, a.moving_release), indent=2, allow_nan=False)
        + "\n"
    )
